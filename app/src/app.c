/*
 * CF Framework
 * Copyright (C) 2026 Orion
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "MATH/cf_math.h"
#include "RUNTIME/cf_status.h"

#include <stdio.h>

static double *f64(cf_math *x)
{
  return (double *)x->data;
}

static const double *cf64(const cf_math *x)
{
  return (const double *)x->data;
}

static void print_status(const char *label, cf_status status)
{
  printf("%s: %s\n", label, cf_status_as_char(status));
}

static void print_tensor_1d(const char *label, const cf_math *x)
{
  printf("%s [", label);
  for(cf_usize i = 0; i < x->metadata.len; i++)
  {
    printf("%s%.6f", i == 0 ? "" : ", ", cf64(x)[i]);
  }
  printf("]\n");
}

static cf_status cpu_examples(void)
{
  cf_math a = {0};
  cf_math b = {0};
  cf_math out = {0};
  cf_usize vec_dim[CF_MATH_HIGHEST_RANK] = {4};
  cf_usize mat_a_dim[CF_MATH_HIGHEST_RANK] = {2, 3};
  cf_usize mat_b_dim[CF_MATH_HIGHEST_RANK] = {3, 2};
  double avec[] = {1.0, 2.0, 3.0, 4.0};
  double bvec[] = {10.0, 20.0, 30.0, 40.0};
  double amat[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double bmat[] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  cf_status status;

  status = cf_math_alloc(&a, vec_dim, 1, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL);
  if(status != CF_OK) return status;
  status = cf_math_alloc(&b, vec_dim, 1, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < 4U; i++)
  {
    f64(&a)[i] = avec[i];
    f64(&b)[i] = bvec[i];
  }

  status = cf_math_add(&out, &a, &b, CF_NULL);
  if(status != CF_OK) return status;
  print_tensor_1d("cpu add", &out);

  status = cf_math_softmax_fwd(&out, &a, 0U, CF_SOFTMAX_CHANNEL, CF_NULL);
  if(status != CF_OK) return status;
  print_tensor_1d("cpu softmax", &out);

  cf_math_free(&out, CF_NULL);
  cf_math_free(&b, CF_NULL);
  cf_math_free(&a, CF_NULL);

  status = cf_math_alloc(&a, mat_a_dim, 2, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL);
  if(status != CF_OK) return status;
  status = cf_math_alloc(&b, mat_b_dim, 2, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < 6U; i++)
  {
    f64(&a)[i] = amat[i];
    f64(&b)[i] = bmat[i];
  }

  status = cf_math_matmul(&out, &a, &b, CF_NULL);
  if(status == CF_OK) print_tensor_1d("cpu matmul flattened", &out);

  cf_math_free(&out, CF_NULL);
  cf_math_free(&b, CF_NULL);
  cf_math_free(&a, CF_NULL);
  return status;
}

static void cuda_example_if_available(void)
{
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  int device_count = 0;
  cudaError_t cuda_status = cudaGetDeviceCount(&device_count);

  if(cuda_status != cudaSuccess || device_count <= 0)
  {
    printf("cuda example: skipped, no usable CUDA device was reported\n");
    return;
  }

  cf_math_cuda_context ctx = {0};
  cf_math host = {0};
  cf_math device = {0};
  cf_math back = {0};
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {4};
  cf_status status = cf_math_context_init(&ctx, 0);
  if(status != CF_OK)
  {
    print_status("cuda context", status);
    return;
  }

  status = cf_math_alloc(&host, dim, 1, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL);
  if(status == CF_OK)
  {
    for(cf_usize i = 0; i < 4U; i++) f64(&host)[i] = (double)(i + 1U);
    status = cf_math_to_device(&device, &host, 0, &ctx);
  }
  if(status == CF_OK) status = cf_math_to_host(&back, &device, &ctx);
  if(status == CF_OK) cudaDeviceSynchronize();

  print_status("cuda roundtrip", status);
  if(status == CF_OK) print_tensor_1d("cuda roundtrip host copy", &back);

  cf_math_free(&back, &ctx);
  cf_math_free(&device, &ctx);
  cf_math_free(&host, CF_NULL);
  cf_math_context_destroy(&ctx);
#else
  printf("cuda example: skipped, CUDA runtime headers are not available in this build\n");
#endif
}

int main(void)
{
  cf_status status = cpu_examples();
  print_status("cpu examples", status);
  cuda_example_if_available();
  return status == CF_OK ? 0 : 1;
}
