#include <cuda_runtime.h>
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

int main(int argc, char **argv)
{
  const char *image_path = argc > 1 ? argv[1] : "public/doc/test_image.jpg";
  const cf_usize workspace_capacity = 16 * 1024 * 1024;
  const cf_usize storage_capacity = 16 * 1024 * 1024;

  cf_math_context ctx = {0};
  cf_math_workspace workspace = {0};
  cf_math_handle handle = {0};
  cf_math raw_image = {0};
  cf_u16 first_pixels[10] = {0};
  cf_u16 *device_ptr = CF_NULL;
  cf_usize pixel_count = 0;
  cf_usize copy_size = 0;
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

  cf_load_and_transfer_image_u16(&handle, &raw_image, image_path);

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

  exit_code = 0;

done:
  if (device_ptr != CF_NULL && handle.workspace != CF_NULL) {
    cudaFreeAsync(device_ptr, handle.workspace->stream);
    cudaStreamSynchronize(handle.workspace->stream);
  }

  cf_math_handle_destroy(&handle);
  cf_math_workspace_destroy(&workspace);
  cf_math_context_destroy(&ctx);

  printf("Test finished.\n");
  return exit_code;
}
