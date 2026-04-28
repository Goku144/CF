#include "MEMORY/cf_memory.h"
#include "MEMORY/cf_array.h"

#include "RUNTIME/cf_io.h"
#include "RUNTIME/cf_log.h"
#include "RUNTIME/cf_status.h"
#include "RUNTIME/cf_time.h"

#include "SECURITY/cf_aes.h"
#include "SECURITY/cf_base64.h"
#include "SECURITY/cf_hex.h"

#include "MATH/cf_math.h"
#include "MATH/cf_tensor.h"

#include "TEXT/cf_ascii.h"
#include "TEXT/cf_string.h"

#include <stdio.h>
#include <string.h>

#ifndef CF_APP_ELEMENTWISE_LEN
#define CF_APP_ELEMENTWISE_LEN ((cf_usize)16777216)
#endif

#ifndef CF_APP_MATRIX_N
#define CF_APP_MATRIX_N ((cf_usize)1024)
#endif

#ifndef CF_APP_BATCH_COUNT
#define CF_APP_BATCH_COUNT ((cf_usize)16)
#endif

#ifndef CF_APP_BATCH_N
#define CF_APP_BATCH_N ((cf_usize)256)
#endif

typedef enum cf_app_elementwise_op
{
  CF_APP_OP_ADD = 0,
  CF_APP_OP_MUL,
  CF_APP_OP_SCALAR_MUL,
} cf_app_elementwise_op;

static double cf_app_seconds_between(cf_time_point start, cf_time_point end)
{
  cf_time elapsed = cf_time_elapsed(start, end);
  return (double)cf_time_as_ns(elapsed) / 1000000000.0;
}

#ifdef CF_CUDA_AVAILABLE
static double cf_app_abs_double(double value)
{
  return value < 0.0 ? -value : value;
}

static double cf_app_max_abs_diff(const cf_tensor *lhs, const cf_tensor *rhs)
{
  const double *a = (const double *)lhs->data;
  const double *b = (const double *)rhs->data;
  double max_diff = 0.0;

  if(lhs->data == CF_NULL || rhs->data == CF_NULL || lhs->metadata.len != rhs->metadata.len)
    return -1.0;

  for(cf_usize i = 0; i < lhs->metadata.len; i++)
  {
    double diff = cf_app_abs_double(a[i] - b[i]);
    if(diff > max_diff) max_diff = diff;
  }

  return max_diff;
}
#endif

static double cf_app_gib_per_sec(double bytes, double seconds)
{
  if(seconds <= 0.0) return 0.0;
  return bytes / seconds / 1073741824.0;
}

static double cf_app_gflops(double flops, double seconds)
{
  if(seconds <= 0.0) return 0.0;
  return flops / seconds / 1000000000.0;
}

static void cf_app_fill_elementwise_inputs(cf_tensor *a, cf_tensor *b)
{
  double *a_values = (double *)a->data;
  double *b_values = (double *)b->data;

  for(cf_usize i = 0; i < a->metadata.len; i++)
  {
    a_values[i] = (double)(i % 1024) * 0.25 + 1.0;
    b_values[i] = (double)((i * 3) % 1024) * 0.125 + 0.5;
  }
}

static void cf_app_fill_matrix_inputs(cf_tensor *a, cf_tensor *b)
{
  double *a_values = (double *)a->data;
  double *b_values = (double *)b->data;

  for(cf_usize i = 0; i < a->metadata.len; i++)
    a_values[i] = (double)((i % 17) + 1) * 0.01;

  for(cf_usize i = 0; i < b->metadata.len; i++)
    b_values[i] = (double)((i % 19) + 1) * 0.02;
}

#ifdef CF_CUDA_AVAILABLE
static void cf_app_warmup_gpu(void)
{
  cf_tensor a = {0};
  cf_tensor b = {0};
  cf_usize vec_dim[CF_TENSOR_HIGHEST_RANK] = {1024, 0, 0, 0, 0, 0, 0, 0};
  cf_usize mat_dim[CF_TENSOR_HIGHEST_RANK] = {16, 16, 0, 0, 0, 0, 0, 0};
  double scalar = 2.0;
  cf_status status;

  status = cf_tensor_init_cpu(&a, vec_dim, 1, CF_TENSOR_DOUBLE);
  if(status == CF_OK) status = cf_tensor_init_cpu(&b, vec_dim, 1, CF_TENSOR_DOUBLE);
  if(status == CF_OK)
  {
    cf_app_fill_elementwise_inputs(&a, &b);
    status = cf_tensor_to_gpu(&a);
    if(status == CF_OK) status = cf_tensor_to_gpu(&b);
    if(status == CF_OK) status = cf_tensor_add_gpu(&a, &b);
    if(status == CF_OK) status = cf_tensor_mul_gpu(&a, &b);
    if(status == CF_OK) status = cf_tensor_scalar_mul_gpu(&a, &scalar);
    if(status == CF_OK) (void)cf_tensor_sync_gpu();
  }
  cf_tensor_destroy_gpu(&a);
  cf_tensor_destroy_gpu(&b);

  a = (cf_tensor){0};
  b = (cf_tensor){0};
  status = cf_tensor_init_cpu(&a, mat_dim, 2, CF_TENSOR_DOUBLE);
  if(status == CF_OK) status = cf_tensor_init_cpu(&b, mat_dim, 2, CF_TENSOR_DOUBLE);
  if(status == CF_OK)
  {
    cf_app_fill_matrix_inputs(&a, &b);
    status = cf_tensor_to_gpu(&a);
    if(status == CF_OK) status = cf_tensor_to_gpu(&b);
    if(status == CF_OK) status = cf_tensor_matrix_mul_gpu(&a, &b);
    if(status == CF_OK) (void)cf_tensor_sync_gpu();
  }
  cf_tensor_destroy_gpu(&a);
  cf_tensor_destroy_gpu(&b);
}
#endif

static void cf_app_print_elementwise_result(
  const char *name,
  cf_status cpu_status,
  double cpu_seconds,
  cf_status gpu_status,
  double gpu_upload_seconds,
  double gpu_seconds,
  double gpu_download_seconds,
  const cf_tensor *cpu,
  const cf_tensor *gpu,
  double bytes)
{
  printf("\n%s huge tensor len=%zu\n", name, (size_t)cpu->metadata.len);
  printf("cpu: %s | %.6f s | %.3f GiB/s\n",
         cf_status_as_char(cpu_status),
         cpu_seconds,
         cf_app_gib_per_sec(bytes, cpu_seconds));

#ifdef CF_CUDA_AVAILABLE
  double diff = cpu_status == CF_OK && gpu_status == CF_OK ? cf_app_max_abs_diff(cpu, gpu) : -1.0;
  const double *cpu_values = (const double *)cpu->data;
  const double *gpu_values = (const double *)gpu->data;
  cf_usize last = cpu->metadata.len == 0 ? 0 : cpu->metadata.len - 1;
  cf_usize mid = cpu->metadata.len / 2;

  printf("gpu: %s | upload %.6f s | op %.6f s | download %.6f s | %.3f GiB/s\n",
         cf_status_as_char(gpu_status),
         gpu_upload_seconds,
         gpu_seconds,
         gpu_download_seconds,
         cf_app_gib_per_sec(bytes, gpu_seconds));
  printf("max abs diff: %.12f | %s\n", diff, diff >= 0.0 && diff <= 0.000000001 ? "MATCH" : "DIFF");
  if(cpu->data != CF_NULL && gpu->data != CF_NULL && cpu->metadata.len != 0 && gpu->metadata.len != 0)
  {
    printf("samples cpu/gpu: [0] %.6f / %.6f | [mid] %.6f / %.6f | [last] %.6f / %.6f\n",
           cpu_values[0], gpu_values[0],
           cpu_values[mid], gpu_values[mid],
           cpu_values[last], gpu_values[last]);
  }
#else
  (void)gpu_status;
  (void)gpu_upload_seconds;
  (void)gpu_seconds;
  (void)gpu_download_seconds;
  (void)gpu;
  printf("gpu: skipped; build without CF_CUDA_AVAILABLE\n");
#endif
}

static void cf_app_print_matmul_result(
  const char *name,
  cf_status cpu_status,
  double cpu_seconds,
  cf_status gpu_status,
  double gpu_upload_seconds,
  double gpu_seconds,
  double gpu_download_seconds,
  const cf_tensor *cpu,
  const cf_tensor *gpu,
  double flops)
{
  printf("\n%s huge tensor\n", name);
  printf("cpu: %s | %.6f s | %.3f GFLOP/s\n",
         cf_status_as_char(cpu_status),
         cpu_seconds,
         cf_app_gflops(flops, cpu_seconds));

#ifdef CF_CUDA_AVAILABLE
  double diff = cpu_status == CF_OK && gpu_status == CF_OK ? cf_app_max_abs_diff(cpu, gpu) : -1.0;
  const double *cpu_values = (const double *)cpu->data;
  const double *gpu_values = (const double *)gpu->data;
  cf_usize last = cpu->metadata.len == 0 ? 0 : cpu->metadata.len - 1;
  cf_usize mid = cpu->metadata.len / 2;

  printf("gpu: %s | upload %.6f s | op %.6f s | download %.6f s | %.3f GFLOP/s\n",
         cf_status_as_char(gpu_status),
         gpu_upload_seconds,
         gpu_seconds,
         gpu_download_seconds,
         cf_app_gflops(flops, gpu_seconds));
  printf("max abs diff: %.12f | %s\n", diff, diff >= 0.0 && diff <= 0.000000001 ? "MATCH" : "DIFF");
  if(cpu->data != CF_NULL && gpu->data != CF_NULL && cpu->metadata.len != 0 && gpu->metadata.len != 0)
  {
    printf("samples cpu/gpu: [0] %.6f / %.6f | [mid] %.6f / %.6f | [last] %.6f / %.6f\n",
           cpu_values[0], gpu_values[0],
           cpu_values[mid], gpu_values[mid],
           cpu_values[last], gpu_values[last]);
  }
#else
  (void)gpu_status;
  (void)gpu_upload_seconds;
  (void)gpu_seconds;
  (void)gpu_download_seconds;
  (void)cpu;
  (void)gpu;
  printf("gpu: skipped; build without CF_CUDA_AVAILABLE\n");
#endif
}

static cf_status cf_app_run_elementwise_cpu(cf_app_elementwise_op op, cf_tensor *a, const cf_tensor *b)
{
  double scalar = 2.0;

  switch(op)
  {
    case CF_APP_OP_ADD: return cf_tensor_add_cpu(a, b);
    case CF_APP_OP_MUL: return cf_tensor_mul_cpu(a, b);
    case CF_APP_OP_SCALAR_MUL: return cf_tensor_scalar_mul_cpu(a, &scalar);
    default: return CF_ERR_INVALID;
  }
}

#ifdef CF_CUDA_AVAILABLE
static cf_status cf_app_run_elementwise_gpu(cf_app_elementwise_op op, cf_tensor *a, const cf_tensor *b)
{
  double scalar = 2.0;

  switch(op)
  {
    case CF_APP_OP_ADD: return cf_tensor_add_gpu(a, b);
    case CF_APP_OP_MUL: return cf_tensor_mul_gpu(a, b);
    case CF_APP_OP_SCALAR_MUL: return cf_tensor_scalar_mul_gpu(a, &scalar);
    default: return CF_ERR_INVALID;
  }
}
#endif

static void cf_app_benchmark_elementwise(const char *name, cf_app_elementwise_op op)
{
  cf_tensor cpu_a = {0};
  cf_tensor cpu_b = {0};
  cf_tensor gpu_a = {0};
#ifdef CF_CUDA_AVAILABLE
  cf_tensor gpu_b = {0};
#endif
  cf_usize dim[CF_TENSOR_HIGHEST_RANK] = {CF_APP_ELEMENTWISE_LEN, 0, 0, 0, 0, 0, 0, 0};
  cf_time_point start = {0};
  cf_time_point end = {0};
  cf_status cpu_status;
  cf_status gpu_status = CF_ERR_UNSUPPORTED;
  double cpu_seconds = 0.0;
  double gpu_upload_seconds = 0.0;
  double gpu_seconds = 0.0;
  double gpu_download_seconds = 0.0;
  double bytes = (double)CF_APP_ELEMENTWISE_LEN * sizeof(double) * (op == CF_APP_OP_SCALAR_MUL ? 2.0 : 3.0);

  cpu_status = cf_tensor_init_cpu(&cpu_a, dim, 1, CF_TENSOR_DOUBLE);
  if(cpu_status == CF_OK) cpu_status = cf_tensor_init_cpu(&cpu_b, dim, 1, CF_TENSOR_DOUBLE);
  if(cpu_status == CF_OK)
  {
    cf_app_fill_elementwise_inputs(&cpu_a, &cpu_b);
    (void)cf_time_now_mono(&start);
    cpu_status = cf_app_run_elementwise_cpu(op, &cpu_a, &cpu_b);
    (void)cf_time_now_mono(&end);
    cpu_seconds = cf_app_seconds_between(start, end);
  }

#ifdef CF_CUDA_AVAILABLE
  gpu_status = cf_tensor_init_cpu(&gpu_a, dim, 1, CF_TENSOR_DOUBLE);
  if(gpu_status == CF_OK) gpu_status = cf_tensor_init_cpu(&gpu_b, dim, 1, CF_TENSOR_DOUBLE);
  if(gpu_status == CF_OK)
  {
    cf_app_fill_elementwise_inputs(&gpu_a, &gpu_b);
    (void)cf_time_now_mono(&start);
    gpu_status = cf_tensor_to_gpu(&gpu_a);
    if(gpu_status == CF_OK && op != CF_APP_OP_SCALAR_MUL) gpu_status = cf_tensor_to_gpu(&gpu_b);
    (void)cf_time_now_mono(&end);
    gpu_upload_seconds = cf_app_seconds_between(start, end);
  }
  if(gpu_status == CF_OK)
  {
    (void)cf_time_now_mono(&start);
    gpu_status = cf_app_run_elementwise_gpu(op, &gpu_a, &gpu_b);
    if(gpu_status == CF_OK) gpu_status = cf_tensor_sync_gpu();
    (void)cf_time_now_mono(&end);
    gpu_seconds = cf_app_seconds_between(start, end);
  }
  if(gpu_status == CF_OK)
  {
    (void)cf_time_now_mono(&start);
    gpu_status = cf_tensor_to_cpu(&gpu_a);
    (void)cf_time_now_mono(&end);
    gpu_download_seconds = cf_app_seconds_between(start, end);
  }
#endif

  cf_app_print_elementwise_result(name, cpu_status, cpu_seconds, gpu_status, gpu_upload_seconds, gpu_seconds, gpu_download_seconds, &cpu_a, &gpu_a, bytes);

  cf_tensor_destroy_cpu(&cpu_a);
  cf_tensor_destroy_cpu(&cpu_b);
#ifdef CF_CUDA_AVAILABLE
  cf_tensor_destroy_gpu(&gpu_a);
  cf_tensor_destroy_gpu(&gpu_b);
#endif
}

static void cf_app_benchmark_matrix_mul(void)
{
  cf_tensor cpu_a = {0};
  cf_tensor cpu_b = {0};
  cf_tensor gpu_a = {0};
#ifdef CF_CUDA_AVAILABLE
  cf_tensor gpu_b = {0};
#endif
  cf_usize a_dim[CF_TENSOR_HIGHEST_RANK] = {CF_APP_MATRIX_N, CF_APP_MATRIX_N, 0, 0, 0, 0, 0, 0};
  cf_usize b_dim[CF_TENSOR_HIGHEST_RANK] = {CF_APP_MATRIX_N, CF_APP_MATRIX_N, 0, 0, 0, 0, 0, 0};
  cf_time_point start = {0};
  cf_time_point end = {0};
  cf_status cpu_status;
  cf_status gpu_status = CF_ERR_UNSUPPORTED;
  double cpu_seconds = 0.0;
  double gpu_upload_seconds = 0.0;
  double gpu_seconds = 0.0;
  double gpu_download_seconds = 0.0;
  double flops = 2.0 * (double)CF_APP_MATRIX_N * (double)CF_APP_MATRIX_N * (double)CF_APP_MATRIX_N;

  cpu_status = cf_tensor_init_cpu(&cpu_a, a_dim, 2, CF_TENSOR_DOUBLE);
  if(cpu_status == CF_OK) cpu_status = cf_tensor_init_cpu(&cpu_b, b_dim, 2, CF_TENSOR_DOUBLE);
  if(cpu_status == CF_OK)
  {
    cf_app_fill_matrix_inputs(&cpu_a, &cpu_b);
    (void)cf_time_now_mono(&start);
    cpu_status = cf_tensor_matrix_mul_cpu(&cpu_a, &cpu_b);
    (void)cf_time_now_mono(&end);
    cpu_seconds = cf_app_seconds_between(start, end);
  }

#ifdef CF_CUDA_AVAILABLE
  gpu_status = cf_tensor_init_cpu(&gpu_a, a_dim, 2, CF_TENSOR_DOUBLE);
  if(gpu_status == CF_OK) gpu_status = cf_tensor_init_cpu(&gpu_b, b_dim, 2, CF_TENSOR_DOUBLE);
  if(gpu_status == CF_OK)
  {
    cf_app_fill_matrix_inputs(&gpu_a, &gpu_b);
    (void)cf_time_now_mono(&start);
    gpu_status = cf_tensor_to_gpu(&gpu_a);
    if(gpu_status == CF_OK) gpu_status = cf_tensor_to_gpu(&gpu_b);
    (void)cf_time_now_mono(&end);
    gpu_upload_seconds = cf_app_seconds_between(start, end);
  }
  if(gpu_status == CF_OK)
  {
    (void)cf_time_now_mono(&start);
    gpu_status = cf_tensor_matrix_mul_gpu(&gpu_a, &gpu_b);
    if(gpu_status == CF_OK) gpu_status = cf_tensor_sync_gpu();
    (void)cf_time_now_mono(&end);
    gpu_seconds = cf_app_seconds_between(start, end);
  }
  if(gpu_status == CF_OK)
  {
    (void)cf_time_now_mono(&start);
    gpu_status = cf_tensor_to_cpu(&gpu_a);
    (void)cf_time_now_mono(&end);
    gpu_download_seconds = cf_app_seconds_between(start, end);
  }
#endif

  cf_app_print_matmul_result("matrix mul", cpu_status, cpu_seconds, gpu_status, gpu_upload_seconds, gpu_seconds, gpu_download_seconds, &cpu_a, &gpu_a, flops);

  cf_tensor_destroy_cpu(&cpu_a);
  cf_tensor_destroy_cpu(&cpu_b);
#ifdef CF_CUDA_AVAILABLE
  cf_tensor_destroy_gpu(&gpu_a);
  cf_tensor_destroy_gpu(&gpu_b);
#endif
}

static void cf_app_benchmark_batch_mul(void)
{
  cf_tensor cpu_a = {0};
  cf_tensor cpu_b = {0};
  cf_tensor gpu_a = {0};
#ifdef CF_CUDA_AVAILABLE
  cf_tensor gpu_b = {0};
#endif
  cf_usize a_dim[CF_TENSOR_HIGHEST_RANK] = {CF_APP_BATCH_COUNT, CF_APP_BATCH_N, CF_APP_BATCH_N, 0, 0, 0, 0, 0};
  cf_usize b_dim[CF_TENSOR_HIGHEST_RANK] = {CF_APP_BATCH_COUNT, CF_APP_BATCH_N, CF_APP_BATCH_N, 0, 0, 0, 0, 0};
  cf_time_point start = {0};
  cf_time_point end = {0};
  cf_status cpu_status;
  cf_status gpu_status = CF_ERR_UNSUPPORTED;
  double cpu_seconds = 0.0;
  double gpu_upload_seconds = 0.0;
  double gpu_seconds = 0.0;
  double gpu_download_seconds = 0.0;
  double flops = 2.0 * (double)CF_APP_BATCH_COUNT * (double)CF_APP_BATCH_N * (double)CF_APP_BATCH_N * (double)CF_APP_BATCH_N;

  cpu_status = cf_tensor_init_cpu(&cpu_a, a_dim, 3, CF_TENSOR_DOUBLE);
  if(cpu_status == CF_OK) cpu_status = cf_tensor_init_cpu(&cpu_b, b_dim, 3, CF_TENSOR_DOUBLE);
  if(cpu_status == CF_OK)
  {
    cf_app_fill_matrix_inputs(&cpu_a, &cpu_b);
    (void)cf_time_now_mono(&start);
    cpu_status = cf_tensor_batch_mul_cpu(&cpu_a, &cpu_b);
    (void)cf_time_now_mono(&end);
    cpu_seconds = cf_app_seconds_between(start, end);
  }

#ifdef CF_CUDA_AVAILABLE
  gpu_status = cf_tensor_init_cpu(&gpu_a, a_dim, 3, CF_TENSOR_DOUBLE);
  if(gpu_status == CF_OK) gpu_status = cf_tensor_init_cpu(&gpu_b, b_dim, 3, CF_TENSOR_DOUBLE);
  if(gpu_status == CF_OK)
  {
    cf_app_fill_matrix_inputs(&gpu_a, &gpu_b);
    (void)cf_time_now_mono(&start);
    gpu_status = cf_tensor_to_gpu(&gpu_a);
    if(gpu_status == CF_OK) gpu_status = cf_tensor_to_gpu(&gpu_b);
    (void)cf_time_now_mono(&end);
    gpu_upload_seconds = cf_app_seconds_between(start, end);
  }
  if(gpu_status == CF_OK)
  {
    (void)cf_time_now_mono(&start);
    gpu_status = cf_tensor_batch_mul_gpu(&gpu_a, &gpu_b);
    if(gpu_status == CF_OK) gpu_status = cf_tensor_sync_gpu();
    (void)cf_time_now_mono(&end);
    gpu_seconds = cf_app_seconds_between(start, end);
  }
  if(gpu_status == CF_OK)
  {
    (void)cf_time_now_mono(&start);
    gpu_status = cf_tensor_to_cpu(&gpu_a);
    (void)cf_time_now_mono(&end);
    gpu_download_seconds = cf_app_seconds_between(start, end);
  }
#endif

  cf_app_print_matmul_result("batched matrix mul", cpu_status, cpu_seconds, gpu_status, gpu_upload_seconds, gpu_seconds, gpu_download_seconds, &cpu_a, &gpu_a, flops);

  cf_tensor_destroy_cpu(&cpu_a);
  cf_tensor_destroy_cpu(&cpu_b);
#ifdef CF_CUDA_AVAILABLE
  cf_tensor_destroy_gpu(&gpu_a);
  cf_tensor_destroy_gpu(&gpu_b);
#endif
}

int main(void)
{
  printf("CPU vs GPU huge tensor performance\n");
  printf("elementwise len=%zu | matrix=%zux%zu | batch=%zu x %zux%zu\n",
         (size_t)CF_APP_ELEMENTWISE_LEN,
         (size_t)CF_APP_MATRIX_N,
         (size_t)CF_APP_MATRIX_N,
         (size_t)CF_APP_BATCH_COUNT,
         (size_t)CF_APP_BATCH_N,
         (size_t)CF_APP_BATCH_N);

#ifdef CF_CUDA_AVAILABLE
  cf_app_warmup_gpu();
#endif

  cf_app_benchmark_elementwise("tensor add", CF_APP_OP_ADD);
  cf_app_benchmark_elementwise("tensor elementwise mul", CF_APP_OP_MUL);
  cf_app_benchmark_elementwise("tensor scalar mul", CF_APP_OP_SCALAR_MUL);
  cf_app_benchmark_matrix_mul();
  cf_app_benchmark_batch_mul();

  return 0;
}
