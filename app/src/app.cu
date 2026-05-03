#include "MATH/cf_math.h"
#include "RUNTIME/cf_time.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

static cf_usize app_round_up(cf_usize n, cf_usize d)
{
  return ((n + d - 1) / d) * d;
}

static float *app_math_f32_data(cf_math_handle *handle, const cf_math *math)
{
  return (float *)((cf_u8 *)handle->storage.backend + math->byte_offset);
}

static long long app_bench_binary(cf_math_handle *handle, cf_math *out, const cf_math *a, const cf_math *b, cf_math_op_kind op, int iter)
{
  cf_time_point start;
  cf_time_point end;

  cudaStreamSynchronize(handle->workspace->stream);
  cf_time_now_mono(&start);
  for (int i = 0; i < iter; i++)
    cf_math_wise_op(handle, out, a, b, op);
  cudaStreamSynchronize(handle->workspace->stream);
  cf_time_now_mono(&end);

  return (long long)cf_time_as_ns(cf_time_elapsed(start, end));
}

static long long app_bench_unary(cf_math_handle *handle, cf_math *out, const cf_math *a, cf_math_op_kind op, int iter)
{
  cf_time_point start;
  cf_time_point end;

  cudaStreamSynchronize(handle->workspace->stream);
  cf_time_now_mono(&start);
  for (int i = 0; i < iter; i++)
    cf_math_wise_op(handle, out, a, CF_NULL, op);
  cudaStreamSynchronize(handle->workspace->stream);
  cf_time_now_mono(&end);

  return (long long)cf_time_as_ns(cf_time_elapsed(start, end));
}

static void app_print_result(const char *name, long long ns, int iter, cf_usize n)
{
  double ns_per_op = (double)ns / (double)iter;
  double elems_per_sec = ((double)n * (double)iter) / ((double)ns / 1000000000.0);

  printf("%-4s total: %lld ns | per op: %.2f ns | %.2f Melem/s\n",
         name,
         ns,
         ns_per_op,
         elems_per_sec / 1000000.0);
}

int main(void)
{
  enum { ITER = 1000 };
  const cf_usize n = (cf_usize)(1024 * 1024 + 3);
  const cf_usize padded_n = app_round_up(n, 4);
  const int dim[1] = { (int)n };
  const cf_usize arena_size = padded_n * sizeof(float) * 3;
  const cf_usize bytes = padded_n * sizeof(float);

  cf_math_context context;
  cf_math_workspace workspace;
  cf_math_handle handle;
  cf_math_desc desc;
  cf_math a;
  cf_math b;
  cf_math c;

  cf_math_context_create(&context, 0, CF_MATH_DEVICE_CUDA);
  cf_math_workspace_create(&workspace, 1024 * 1024, CF_MATH_DEVICE_CUDA);
  cf_math_handle_create(&handle, &context, &workspace, arena_size, CF_MATH_DEVICE_CUDA);
  cf_math_desc_create(&desc, 1, dim, CF_MATH_DTYPE_F32, CF_MATH_DESC_NONE);

  cf_math_bind(&handle, &a, &desc);
  cf_math_bind(&handle, &b, &desc);
  cf_math_bind(&handle, &c, &desc);

  float *a_host = (float *)malloc(bytes);
  float *b_host = (float *)malloc(bytes);
  float *c_host = (float *)malloc(bytes);
  float *a_dev = app_math_f32_data(&handle, &a);
  float *b_dev = app_math_f32_data(&handle, &b);
  float *c_dev = app_math_f32_data(&handle, &c);

  for (cf_usize i = 0; i < padded_n; i++)
  {
    a_host[i] = (float)i * 0.5f;
    b_host[i] = (float)i + 1.0f;
    c_host[i] = 0.0f;
  }

  cudaMemcpyAsync(a_dev, a_host, bytes, cudaMemcpyHostToDevice, workspace.stream);
  cudaMemcpyAsync(b_dev, b_host, bytes, cudaMemcpyHostToDevice, workspace.stream);
  cudaMemcpyAsync(c_dev, c_host, bytes, cudaMemcpyHostToDevice, workspace.stream);
  cudaStreamSynchronize(workspace.stream);

  printf("gpu wise op benchmark: F32 elements=%llu padded=%llu iterations=%d\n",
         (unsigned long long)n,
         (unsigned long long)padded_n,
         ITER);

  long long add_ns = app_bench_binary(&handle, &c, &a, &b, CF_MATH_OP_ADD, ITER);
  long long sub_ns = app_bench_binary(&handle, &c, &a, &b, CF_MATH_OP_SUB, ITER);
  long long mul_ns = app_bench_binary(&handle, &c, &a, &b, CF_MATH_OP_MUL, ITER);
  long long div_ns = app_bench_binary(&handle, &c, &a, &b, CF_MATH_OP_DIV, ITER);
  long long neg_ns = app_bench_unary(&handle, &c, &a, CF_MATH_OP_NEG, ITER);

  app_print_result("ADD", add_ns, ITER, padded_n);
  app_print_result("SUB", sub_ns, ITER, padded_n);
  app_print_result("MUL", mul_ns, ITER, padded_n);
  app_print_result("DIV", div_ns, ITER, padded_n);
  app_print_result("NEG", neg_ns, ITER, padded_n);

  cf_math_wise_op(&handle, &c, &a, &b, CF_MATH_OP_ADD);
  cudaMemcpyAsync(c_host, c_dev, bytes, cudaMemcpyDeviceToHost, workspace.stream);
  cudaStreamSynchronize(workspace.stream);

  printf("examples:\n");
  printf("A[0] + B[0] = C[0] -> %.2f + %.2f = %.2f\n", a_host[0], b_host[0], c_host[0]);
  printf("A[7] + B[7] = C[7] -> %.2f + %.2f = %.2f\n", a_host[7], b_host[7], c_host[7]);

  free(a_host);
  free(b_host);
  free(c_host);
  cf_math_desc_destroy(&desc);
  cf_math_handle_destroy(&handle);
  cf_math_workspace_destroy(&workspace);
  cf_math_context_destroy(&context);
  return 0;
}
