#include "MATH/cf_math.h"
#include "RUNTIME/cf_status.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#define CF_APP_DEFAULT_ELEMENTS (1u << 24)
#define CF_APP_DEFAULT_ITERS 200
#define CF_APP_DEFAULT_WARMUP 20

static int cf_app_fail_status(const char *step, cf_status status)
{
  fprintf(stderr, "%s failed: %s\n", step, cf_status_as_char(status));
  return 1;
}

static int cf_app_fail_cuda(const char *step, cudaError_t status)
{
  fprintf(stderr, "%s failed: %s\n", step, cudaGetErrorString(status));
  return 1;
}

static cf_usize cf_app_parse_usize(const char *text, cf_usize fallback)
{
  char *end = NULL;
  errno = 0;
  unsigned long long value = strtoull(text, &end, 10);

  if(errno != 0 || end == text || *end != '\0' || value == 0)
    return fallback;

  return (cf_usize)value;
}

static void *cf_app_add_f16_device_ptr(cf_math_handle *handle, cf_math *math)
{
  return (void *)(math->byte_offset + (cf_u8 *)handle->storage.backend);
}

static __global__ void cf_app_init_half_inputs(__half *a, __half *b, cf_usize n)
{
  cf_usize index = (cf_usize)threadIdx.x + (cf_usize)blockDim.x * (cf_usize)blockIdx.x;

  if(index >= n)
    return;

  float af = (float)(index % 251) * 0.25f;
  float bf = (float)(index % 127) * 0.50f;

  a[index] = __float2half(af);
  b[index] = __float2half(bf);
}

typedef void (*cf_app_f16_binary_op)(cf_math_handle *handle, cf_math *C, cf_math *A, cf_math *B);

static int cf_app_benchmark_f16_op(const char *name,
                                   cf_app_f16_binary_op op,
                                   cf_math_handle *handle,
                                   cf_math_workspace *workspace,
                                   cf_math *C,
                                   cf_math *A,
                                   cf_math *B,
                                   __half *C_D,
                                   cf_usize element_count,
                                   int iterations,
                                   int warmup)
{
  cudaEvent_t start = NULL;
  cudaEvent_t stop = NULL;
  cudaError_t cuda_state = cudaSuccess;
  float total_ms = 0.0f;
  __half sample[8];

  for(int i = 0; i < warmup; ++i)
    op(handle, C, A, B);

  cuda_state = cudaGetLastError();
  if(cuda_state != cudaSuccess)
    return cf_app_fail_cuda(name, cuda_state);

  cuda_state = cudaStreamSynchronize(workspace->stream);
  if(cuda_state != cudaSuccess)
    return cf_app_fail_cuda("cudaStreamSynchronize(warmup)", cuda_state);

  cuda_state = cudaEventCreate(&start);
  if(cuda_state != cudaSuccess)
    return cf_app_fail_cuda("cudaEventCreate(start)", cuda_state);

  cuda_state = cudaEventCreate(&stop);
  if(cuda_state != cudaSuccess)
  {
    cudaEventDestroy(start);
    return cf_app_fail_cuda("cudaEventCreate(stop)", cuda_state);
  }

  cuda_state = cudaEventRecord(start, workspace->stream);
  if(cuda_state != cudaSuccess)
  {
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return cf_app_fail_cuda("cudaEventRecord(start)", cuda_state);
  }

  for(int i = 0; i < iterations; ++i)
    op(handle, C, A, B);

  cuda_state = cudaEventRecord(stop, workspace->stream);
  if(cuda_state != cudaSuccess)
  {
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return cf_app_fail_cuda("cudaEventRecord(stop)", cuda_state);
  }

  cuda_state = cudaEventSynchronize(stop);
  if(cuda_state != cudaSuccess)
  {
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return cf_app_fail_cuda("cudaEventSynchronize(stop)", cuda_state);
  }

  cuda_state = cudaGetLastError();
  if(cuda_state != cudaSuccess)
  {
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return cf_app_fail_cuda(name, cuda_state);
  }

  cuda_state = cudaEventElapsedTime(&total_ms, start, stop);
  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  if(cuda_state != cudaSuccess)
    return cf_app_fail_cuda("cudaEventElapsedTime", cuda_state);

  cuda_state = cudaMemcpyAsync(sample, C_D, sizeof(sample), cudaMemcpyDeviceToHost, workspace->stream);
  if(cuda_state != cudaSuccess)
    return cf_app_fail_cuda("cudaMemcpyAsync(sample)", cuda_state);

  cuda_state = cudaStreamSynchronize(workspace->stream);
  if(cuda_state != cudaSuccess)
    return cf_app_fail_cuda("cudaStreamSynchronize(sample)", cuda_state);

  double avg_ms = (double)total_ms / (double)iterations;
  double logical_bytes = (double)element_count * (double)sizeof(__half) * 3.0;
  double logical_gib = logical_bytes / (1024.0 * 1024.0 * 1024.0);
  double logical_gib_s = logical_gib / (avg_ms / 1000.0);

  printf("%s performance check\n", name);
  printf("total time: %.3f ms\n", total_ms);
  printf("average time: %.6f ms\n", avg_ms);
  printf("logical bandwidth: %.3f GiB/s\n", logical_gib_s);
  printf("sample C[0..7]:");

  for(int i = 0; i < 8; ++i)
    printf(" %.3f", (double)__half2float(sample[i]));

  printf("\n\n");

  return 0;
}

int main(int argc, char **argv)
{
  cf_usize element_count = CF_APP_DEFAULT_ELEMENTS;
  int iterations = CF_APP_DEFAULT_ITERS;
  int warmup = CF_APP_DEFAULT_WARMUP;

  if(argc > 1)
    element_count = cf_app_parse_usize(argv[1], element_count);
  if(argc > 2)
    iterations = (int)cf_app_parse_usize(argv[2], (cf_usize)iterations);
  if(argc > 3)
    warmup = (int)cf_app_parse_usize(argv[3], (cf_usize)warmup);

  int device_count = 0;
  cudaError_t cuda_state = cudaGetDeviceCount(&device_count);
  if(cuda_state != cudaSuccess)
    return cf_app_fail_cuda("cudaGetDeviceCount", cuda_state);
  if(device_count <= 0)
  {
    fprintf(stderr, "No CUDA device found.\n");
    return 1;
  }

  if(element_count > (cf_usize)INT_MAX)
  {
    fprintf(stderr, "Element count is too large for cf_math_desc_create dims.\n");
    return 1;
  }

  cf_math_context ctx;
  cf_math_workspace workspace;
  cf_math_handle handle;
  cf_math_desc desc;
  cf_math A;
  cf_math B;
  cf_math C;

  cf_status status = CF_OK;
  int rc = 1;
  cf_usize element_bytes = 0;
  cf_usize handle_bytes = 0;
  int dims[1] = {0};
  __half *A_D = NULL;
  __half *B_D = NULL;
  __half *C_D = NULL;
  cf_usize launched_items = 0;
  int threads = 256;
  int blocks = 0;

  status = cf_math_context_create(&ctx, 0, CF_MATH_DEVICE_CUDA);
  if(status != CF_OK)
    return cf_app_fail_status("cf_math_context_create", status);

  status = cf_math_workspace_create(&workspace, 4096, CF_MATH_DEVICE_CUDA);
  if(status != CF_OK)
  {
    rc = cf_app_fail_status("cf_math_workspace_create", status);
    goto destroy_ctx;
  }

  element_bytes = element_count * (cf_usize)sizeof(__half);
  if(element_bytes > (((cf_usize)-1) - 4096u) / 24u)
  {
    fprintf(stderr, "Requested tensor size is too large.\n");
    goto destroy_workspace;
  }

  handle_bytes = element_bytes * 24u + 4096u;

  status = cf_math_handle_create(&handle, &ctx, &workspace, handle_bytes, CF_MATH_DEVICE_CUDA);
  if(status != CF_OK)
  {
    rc = cf_app_fail_status("cf_math_handle_create", status);
    goto destroy_workspace;
  }

  dims[0] = (int)element_count;
  status = cf_math_desc_create(&desc, 1, dims, CF_MATH_DTYPE_F16, CF_MATH_DESC_NONE);
  if(status != CF_OK)
  {
    rc = cf_app_fail_status("cf_math_desc_create", status);
    goto destroy_handle;
  }

  status = cf_math_bind(&handle, &A, &desc);
  if(status != CF_OK)
  {
    rc = cf_app_fail_status("cf_math_bind(A)", status);
    goto destroy_desc;
  }

  status = cf_math_bind(&handle, &B, &desc);
  if(status != CF_OK)
  {
    rc = cf_app_fail_status("cf_math_bind(B)", status);
    goto destroy_desc;
  }

  status = cf_math_bind(&handle, &C, &desc);
  if(status != CF_OK)
  {
    rc = cf_app_fail_status("cf_math_bind(C)", status);
    goto destroy_desc;
  }

  A_D = (__half *)cf_app_add_f16_device_ptr(&handle, &A);
  B_D = (__half *)cf_app_add_f16_device_ptr(&handle, &B);
  C_D = (__half *)cf_app_add_f16_device_ptr(&handle, &C);

  launched_items = C.elem_len;
  blocks = (int)((launched_items + (cf_usize)threads - 1u) / (cf_usize)threads);

  cf_app_init_half_inputs<<<blocks, threads, 0, workspace.stream>>>(A_D, B_D, launched_items);
  cuda_state = cudaGetLastError();
  if(cuda_state != cudaSuccess)
  {
    rc = cf_app_fail_cuda("cf_app_init_half_inputs launch", cuda_state);
    goto destroy_desc;
  }

  cuda_state = cudaMemsetAsync(C_D, 0, launched_items * sizeof(__half), workspace.stream);
  if(cuda_state != cudaSuccess)
  {
    rc = cf_app_fail_cuda("cudaMemsetAsync(C)", cuda_state);
    goto destroy_desc;
  }

  printf("elements: %zu\n", (size_t)element_count);
  printf("iterations: %d\n", iterations);
  printf("warmup: %d\n", warmup);
  printf("launched items from C.elem_len: %zu\n", (size_t)launched_items);
  printf("\n");

  rc = cf_app_benchmark_f16_op("cf_math_add_f16", cf_math_add_f16, &handle, &workspace, &C, &A, &B, C_D, element_count, iterations, warmup);
  if(rc != 0)
    goto destroy_desc;

  rc = cf_app_benchmark_f16_op("cf_math_sub_f16", cf_math_sub_f16, &handle, &workspace, &C, &A, &B, C_D, element_count, iterations, warmup);
  if(rc != 0)
    goto destroy_desc;

  rc = cf_app_benchmark_f16_op("cf_math_mul_f16", cf_math_mul_f16, &handle, &workspace, &C, &A, &B, C_D, element_count, iterations, warmup);
  if(rc != 0)
    goto destroy_desc;

  rc = 0;

destroy_desc:
  cf_math_desc_destroy(&desc);
destroy_handle:
  cudaStreamSynchronize(workspace.stream);
  cf_math_handle_destroy(&handle);
destroy_workspace:
  cudaStreamSynchronize(workspace.stream);
  cf_math_workspace_destroy(&workspace);
destroy_ctx:
  cf_math_context_destroy(&ctx);

  return rc;
}
