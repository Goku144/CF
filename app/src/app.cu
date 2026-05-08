#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

#include "MATH/cf_math.h"
#include "AI/cf_tokenizer.h"

static int app_check_cf_status(const char *step, cf_status state)
{
  if (state == CF_OK) return 0;

  printf("%s failed: %s\n", step, cf_status_as_char(state));
  return 1;
}

static int app_check_cuda_status(const char *step, cudaError_t state)
{
  if (state == cudaSuccess) return 0;

  printf("%s failed: %s\n", step, cudaGetErrorString(state));
  return 1;
}

static cf_u16 *app_image_device_ptr(const cf_math_handle *handle, const cf_math *image)
{
  return (cf_u16 *)((cf_uptr)handle->storage.backend + (cf_uptr)image->byte_offset);
}

static __half *app_f16_device_ptr(const cf_math_handle *handle, const cf_math *math)
{
  return (__half *)((cf_uptr)handle->storage.backend + (cf_uptr)math->byte_offset);
}

int main(int argc, char **argv)
{
  const char *image_path = argc > 1 ? argv[1] : "public/img/test_image.jpg";
  const cf_usize workspace_capacity = 64 * 1024 * 1024;
  const cf_usize storage_capacity = 64 * 1024 * 1024;

  cf_math_context ctx = {0};
  cf_math_workspace workspace = {0};
  cf_math_handle handle = {0};
  cf_math raw_image = {0};
  cf_math_desc norm_desc = {0};
  cf_math norm_input = {0};
  cf_math norm_output = {0};
  cf_u16 first_pixels[10] = {0};
  __half norm_input_host[10] = {0};
  __half norm_output_host[10] = {0};
  cf_u16 *device_ptr = CF_NULL;
  __half *norm_input_ptr = CF_NULL;
  __half *norm_output_ptr = CF_NULL;
  cf_usize pixel_count = 0;
  cf_usize copy_size = 0;
  int norm_dim[4] = {1, 1, 1, 10};
  int exit_code = 1;

  printf("Starting image transfer test for %s\n", image_path);

  if (app_check_cf_status("cf_math_context_create",
                          cf_math_context_create(&ctx, 0, CF_MATH_DEVICE_CUDA)))
    goto done;

  if (app_check_cf_status("cf_math_workspace_create",
                          cf_math_workspace_create(&workspace, workspace_capacity, CF_MATH_DEVICE_CUDA)))
    goto done;

  if (app_check_cf_status("cf_math_handle_create",
                          cf_math_handle_create(&handle, &ctx, &workspace, storage_capacity, CF_MATH_DEVICE_CUDA)))
    goto done;

  if (app_check_cf_status("cf_tokenizer_load_and_transfer_image_u16",
                          cf_tokenizer_load_and_transfer_image_u16(&handle, &raw_image, image_path)))
    goto done;

  if (app_check_cuda_status("cudaStreamSynchronize",
                            cudaStreamSynchronize(handle.workspace->stream)))
    goto done;

  if (raw_image.elem_len == 0) {
    printf("Image load failed: elem_len is 0.\n");
    goto done;
  }

  device_ptr = app_image_device_ptr(&handle, &raw_image);
  if (device_ptr == CF_NULL) {
    printf("Image load failed: device pointer is NULL.\n");
    goto done;
  }

  pixel_count = raw_image.elem_len < 10 ? raw_image.elem_len : 10;
  copy_size = pixel_count * sizeof(first_pixels[0]);

  if (app_check_cuda_status("cudaMemcpyAsync",
                            cudaMemcpyAsync(first_pixels,
                                            device_ptr,
                                            copy_size,
                                            cudaMemcpyDeviceToHost,
                                            handle.workspace->stream)))
    goto done;

  if (app_check_cuda_status("cudaStreamSynchronize",
                            cudaStreamSynchronize(handle.workspace->stream)))
    goto done;

  printf("Image transfer complete: %zu pixels\n", raw_image.elem_len);
  printf("--- First %zu pixels (0-65535) ---\n", pixel_count);
  for (cf_usize i = 0; i < pixel_count; ++i) {
    printf("Pixel %zu: %u\n", i, (unsigned)first_pixels[i]);
  }

  if (app_check_cf_status("cf_math_desc_create(norm_desc)",
                          cf_math_desc_create(&norm_desc, 4, norm_dim, CF_MATH_DTYPE_F16)))
    goto done;

  if (app_check_cf_status("cf_math_bind(norm_input)",
                          cf_math_bind(&handle, &norm_input, &norm_desc)))
    goto done;

  if (app_check_cf_status("cf_math_bind(norm_output)",
                          cf_math_bind(&handle, &norm_output, &norm_desc)))
    goto done;

  for (cf_usize i = 0; i < pixel_count; ++i) {
    norm_input_host[i] = __float2half((float)first_pixels[i]);
  }

  norm_input_ptr = app_f16_device_ptr(&handle, &norm_input);
  norm_output_ptr = app_f16_device_ptr(&handle, &norm_output);

  if (app_check_cuda_status("cudaMemcpyAsync(norm_input)",
                            cudaMemcpyAsync(norm_input_ptr,
                                            norm_input_host,
                                            copy_size,
                                            cudaMemcpyHostToDevice,
                                            handle.workspace->stream)))
    goto done;

  cf_math_norm_f16(&handle, &norm_output, &norm_input, 65535.0f);

  if (app_check_cuda_status("cudaMemcpyAsync(norm_output)",
                            cudaMemcpyAsync(norm_output_host,
                                            norm_output_ptr,
                                            copy_size,
                                            cudaMemcpyDeviceToHost,
                                            handle.workspace->stream)))
    goto done;

  if (app_check_cuda_status("cudaStreamSynchronize(norm_f16)",
                            cudaStreamSynchronize(handle.workspace->stream)))
    goto done;

  printf("--- First %zu pixels normalized with cf_math_norm_f16 / 65535 ---\n", pixel_count);
  for (cf_usize i = 0; i < pixel_count; ++i) {
    printf("Norm %zu: %.6f\n", i, __half2float(norm_output_host[i]));
  }

  exit_code = 0;

done:
  cf_math_desc_destroy(&norm_desc);
  cf_math_handle_destroy(&handle);
  cf_math_workspace_destroy(&workspace);
  cf_math_context_destroy(&ctx);

  printf("Test finished.\n");
  return exit_code;
}
