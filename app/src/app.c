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

#define CF_APP_VEC_LEN ((cf_usize)4)
#define CF_APP_BATCH_A_LEN ((cf_usize)12)
#define CF_APP_BATCH_B_LEN ((cf_usize)24)

#ifdef CF_CUDA_AVAILABLE
static double cf_app_abs_double(double value)
{
  return value < 0.0 ? -value : value;
}

static double cf_app_max_abs_diff(const double *lhs, const double *rhs, cf_usize len)
{
  double max_diff = 0.0;

  for(cf_usize i = 0; i < len; i++)
  {
    double diff = cf_app_abs_double(lhs[i] - rhs[i]);
    if(diff > max_diff) max_diff = diff;
  }

  return max_diff;
}

static void cf_app_print_tensor_sample(const char *label, const cf_tensor *tensor)
{
  const double *values = (const double *)tensor->data;

  printf("%s shape [", label);
  for(cf_usize i = 0; i < tensor->rank; i++)
  {
    printf("%s%zu", i == 0 ? "" : " x ", (size_t)tensor->dim[i]);
  }
  printf("] values:");

  for(cf_usize i = 0; i < tensor->metadata.len; i++)
  {
    printf(" %.2f", values[i]);
  }
  printf("\n");
}

static void cf_app_print_compare(const char *name, cf_status cpu_status, cf_status gpu_status, const cf_tensor *cpu, const cf_tensor *gpu)
{
  printf("\n%s\n", name);

  if(cpu_status != CF_OK || gpu_status != CF_OK || cpu->data == CF_NULL || gpu->data == CF_NULL || cpu->metadata.len != gpu->metadata.len)
  {
    printf("cpu: %s | gpu: %s | compare skipped\n",
           cf_status_as_char(cpu_status),
           cf_status_as_char(gpu_status));
    return;
  }

  double max_diff = cf_app_max_abs_diff((const double *)cpu->data, (const double *)gpu->data, cpu->metadata.len);

  printf("cpu: %s | gpu: %s | max abs diff: %.12f | %s\n",
         cf_status_as_char(cpu_status),
         cf_status_as_char(gpu_status),
         max_diff,
         max_diff <= 0.000000001 ? "MATCH" : "DIFF");
  cf_app_print_tensor_sample("cpu", cpu);
  cf_app_print_tensor_sample("gpu", gpu);
}

static cf_status cf_app_copy_vector_pair(cf_tensor *a, cf_tensor *b, const double *a_values, const double *b_values)
{
  cf_status status;

  status = cf_tensor_copy_from_array_cpu(a, a_values, CF_APP_VEC_LEN);
  if(status != CF_OK) return status;

  return cf_tensor_copy_from_array_cpu(b, b_values, CF_APP_VEC_LEN);
}

static cf_status cf_app_copy_vector_pair_gpu(cf_tensor *a, cf_tensor *b, const double *a_values, const double *b_values)
{
  cf_status status;

  status = cf_tensor_copy_from_array_gpu(a, a_values, CF_APP_VEC_LEN);
  if(status != CF_OK) return status;

  return cf_tensor_copy_from_array_gpu(b, b_values, CF_APP_VEC_LEN);
}

static void cf_app_test_elementwise(void)
{
  cf_tensor cpu_a = {0};
  cf_tensor cpu_b = {0};
  cf_tensor gpu_a = {0};
  cf_tensor gpu_b = {0};
  cf_usize dim[CF_TENSOR_HIGHEST_RANK] = {CF_APP_VEC_LEN, 0, 0, 0, 0, 0, 0, 0};
  double a_values[CF_APP_VEC_LEN] = {1.0, 2.0, 3.0, 4.0};
  double b_values[CF_APP_VEC_LEN] = {10.0, 20.0, 30.0, 40.0};
  double scalar = 2.0;
  cf_status cpu_status;
  cf_status gpu_status;

  (void)cf_tensor_init_cpu(&cpu_a, dim, 1, CF_TENSOR_DOUBLE);
  (void)cf_tensor_init_cpu(&cpu_b, dim, 1, CF_TENSOR_DOUBLE);
  (void)cf_tensor_init_gpu(&gpu_a, dim, 1, CF_TENSOR_DOUBLE);
  (void)cf_tensor_init_gpu(&gpu_b, dim, 1, CF_TENSOR_DOUBLE);

  cpu_status = cf_app_copy_vector_pair(&cpu_a, &cpu_b, a_values, b_values);
  if(cpu_status == CF_OK) cpu_status = cf_tensor_add_cpu(&cpu_a, &cpu_b);
  gpu_status = cf_app_copy_vector_pair_gpu(&gpu_a, &gpu_b, a_values, b_values);
  if(gpu_status == CF_OK) gpu_status = cf_tensor_add_gpu(&gpu_a, &gpu_b);
  if(gpu_status == CF_OK) gpu_status = cf_tensor_to_cpu(&gpu_a);
  cf_app_print_compare("tensor add", cpu_status, gpu_status, &cpu_a, &gpu_a);

  cpu_status = cf_app_copy_vector_pair(&cpu_a, &cpu_b, a_values, b_values);
  if(cpu_status == CF_OK) cpu_status = cf_tensor_mul_cpu(&cpu_a, &cpu_b);
  gpu_status = cf_app_copy_vector_pair_gpu(&gpu_a, &gpu_b, a_values, b_values);
  if(gpu_status == CF_OK) gpu_status = cf_tensor_mul_gpu(&gpu_a, &gpu_b);
  if(gpu_status == CF_OK) gpu_status = cf_tensor_to_cpu(&gpu_a);
  cf_app_print_compare("tensor elementwise mul", cpu_status, gpu_status, &cpu_a, &gpu_a);

  cpu_status = cf_tensor_copy_from_array_cpu(&cpu_a, a_values, CF_APP_VEC_LEN);
  if(cpu_status == CF_OK) cpu_status = cf_tensor_scalar_mul_cpu(&cpu_a, &scalar);
  gpu_status = cf_tensor_copy_from_array_gpu(&gpu_a, a_values, CF_APP_VEC_LEN);
  if(gpu_status == CF_OK) gpu_status = cf_tensor_scalar_mul_gpu(&gpu_a, &scalar);
  if(gpu_status == CF_OK) gpu_status = cf_tensor_to_cpu(&gpu_a);
  cf_app_print_compare("tensor scalar mul", cpu_status, gpu_status, &cpu_a, &gpu_a);

  cf_tensor_destroy_cpu(&cpu_a);
  cf_tensor_destroy_cpu(&cpu_b);
  cf_tensor_destroy_gpu(&gpu_a);
  cf_tensor_destroy_gpu(&gpu_b);
}

static void cf_app_test_matrix_mul(void)
{
  cf_tensor cpu_a = {0};
  cf_tensor cpu_b = {0};
  cf_tensor gpu_a = {0};
  cf_tensor gpu_b = {0};
  cf_usize a_dim[CF_TENSOR_HIGHEST_RANK] = {2, 3, 0, 0, 0, 0, 0, 0};
  cf_usize b_dim[CF_TENSOR_HIGHEST_RANK] = {3, 2, 0, 0, 0, 0, 0, 0};
  double a_values[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double b_values[6] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  cf_status cpu_status;
  cf_status gpu_status;

  cpu_status = cf_tensor_init_cpu(&cpu_a, a_dim, 2, CF_TENSOR_DOUBLE);
  if(cpu_status == CF_OK) cpu_status = cf_tensor_init_cpu(&cpu_b, b_dim, 2, CF_TENSOR_DOUBLE);
  if(cpu_status == CF_OK)
  {
    memcpy(cpu_a.data, a_values, sizeof(a_values));
    memcpy(cpu_b.data, b_values, sizeof(b_values));
    cpu_status = cf_tensor_matrix_mul_cpu(&cpu_a, &cpu_b);
  }

  gpu_status = cf_tensor_init_gpu(&gpu_a, a_dim, 2, CF_TENSOR_DOUBLE);
  if(gpu_status == CF_OK) gpu_status = cf_tensor_init_gpu(&gpu_b, b_dim, 2, CF_TENSOR_DOUBLE);
  if(gpu_status == CF_OK) gpu_status = cf_tensor_copy_from_array_gpu(&gpu_a, a_values, 6);
  if(gpu_status == CF_OK) gpu_status = cf_tensor_reshape_gpu(&gpu_a, a_dim, 2);
  if(gpu_status == CF_OK) gpu_status = cf_tensor_copy_from_array_gpu(&gpu_b, b_values, 6);
  if(gpu_status == CF_OK) gpu_status = cf_tensor_reshape_gpu(&gpu_b, b_dim, 2);
  if(gpu_status == CF_OK) gpu_status = cf_tensor_matrix_mul_gpu(&gpu_a, &gpu_b);
  if(gpu_status == CF_OK) gpu_status = cf_tensor_to_cpu(&gpu_a);

  cf_app_print_compare("tensor matrix mul", cpu_status, gpu_status, &cpu_a, &gpu_a);

  cf_tensor_destroy_cpu(&cpu_a);
  cf_tensor_destroy_cpu(&cpu_b);
  cf_tensor_destroy_gpu(&gpu_a);
  cf_tensor_destroy_gpu(&gpu_b);
}

static void cf_app_test_batch_mul(void)
{
  cf_tensor cpu_a = {0};
  cf_tensor cpu_b = {0};
  cf_tensor gpu_a = {0};
  cf_tensor gpu_b = {0};
  cf_usize a_dim[CF_TENSOR_HIGHEST_RANK] = {2, 1, 2, 3, 0, 0, 0, 0};
  cf_usize b_dim[CF_TENSOR_HIGHEST_RANK] = {1, 4, 3, 2, 0, 0, 0, 0};
  double a_values[CF_APP_BATCH_A_LEN] = {
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    10.0, 11.0, 12.0
  };
  double b_values[CF_APP_BATCH_B_LEN] = {
    1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    2.0, 0.0, 0.0, 2.0, 1.0, 0.0,
    0.0, 1.0, 1.0, 0.0, 0.0, 2.0,
    1.0, 1.0, 2.0, 0.0, 0.0, 2.0
  };
  cf_status cpu_status;
  cf_status gpu_status;

  cpu_status = cf_tensor_init_cpu(&cpu_a, a_dim, 4, CF_TENSOR_DOUBLE);
  if(cpu_status == CF_OK) cpu_status = cf_tensor_init_cpu(&cpu_b, b_dim, 4, CF_TENSOR_DOUBLE);
  if(cpu_status == CF_OK)
  {
    memcpy(cpu_a.data, a_values, sizeof(a_values));
    memcpy(cpu_b.data, b_values, sizeof(b_values));
    cpu_status = cf_tensor_batch_mul_cpu(&cpu_a, &cpu_b);
  }

  gpu_status = cf_tensor_init_gpu(&gpu_a, a_dim, 4, CF_TENSOR_DOUBLE);
  if(gpu_status == CF_OK) gpu_status = cf_tensor_init_gpu(&gpu_b, b_dim, 4, CF_TENSOR_DOUBLE);
  if(gpu_status == CF_OK) gpu_status = cf_tensor_copy_from_array_gpu(&gpu_a, a_values, CF_APP_BATCH_A_LEN);
  if(gpu_status == CF_OK) gpu_status = cf_tensor_reshape_gpu(&gpu_a, a_dim, 4);
  if(gpu_status == CF_OK) gpu_status = cf_tensor_copy_from_array_gpu(&gpu_b, b_values, CF_APP_BATCH_B_LEN);
  if(gpu_status == CF_OK) gpu_status = cf_tensor_reshape_gpu(&gpu_b, b_dim, 4);
  if(gpu_status == CF_OK) gpu_status = cf_tensor_batch_mul_gpu(&gpu_a, &gpu_b);
  if(gpu_status == CF_OK) gpu_status = cf_tensor_to_cpu(&gpu_a);

  cf_app_print_compare("tensor batched matrix mul", cpu_status, gpu_status, &cpu_a, &gpu_a);

  cf_tensor_destroy_cpu(&cpu_a);
  cf_tensor_destroy_cpu(&cpu_b);
  cf_tensor_destroy_gpu(&gpu_a);
  cf_tensor_destroy_gpu(&gpu_b);
}
#endif

int main(void)
{
  printf("CPU vs GPU tensor operation tests\n");

#ifdef CF_CUDA_AVAILABLE
  cf_app_test_elementwise();
  cf_app_test_matrix_mul();
  cf_app_test_batch_mul();
#else
  printf("gpu tests skipped; build without CF_CUDA_AVAILABLE\n");
#endif

  return 0;
}
