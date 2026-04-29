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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;

static void test_fail(const char *file, int line, const char *expr)
{
  printf("FAIL %s:%d: %s\n", file, line, expr);
  g_failures++;
}

#define CHECK_TRUE(expr) do { if(!(expr)) test_fail(__FILE__, __LINE__, #expr); } while(0)

#define CHECK_STATUS(expr) do { \
  cf_status cf_test_status__ = (expr); \
  if(cf_test_status__ != CF_OK) { \
    printf("FAIL %s:%d: %s -> %s\n", __FILE__, __LINE__, #expr, cf_status_as_char(cf_test_status__)); \
    g_failures++; \
  } \
} while(0)

#define CHECK_NEAR(actual, expected, eps) do { \
  double cf_test_actual__ = (double)(actual); \
  double cf_test_expected__ = (double)(expected); \
  if(fabs(cf_test_actual__ - cf_test_expected__) > (eps)) { \
    printf("FAIL %s:%d: %s expected %.12g got %.12g\n", __FILE__, __LINE__, #actual, cf_test_expected__, cf_test_actual__); \
    g_failures++; \
  } \
} while(0)

static double *f64(cf_math *x)
{
  return (double *)x->data;
}

static const double *cf64(const cf_math *x)
{
  return (const double *)x->data;
}

static cf_i32 *i32(cf_math *x)
{
  return (cf_i32 *)x->data;
}

static void fill_f64(cf_math *x, const double *values)
{
  for(cf_usize i = 0; i < x->metadata.len; i++) f64(x)[i] = values[i];
}

static void check_f64_array(const cf_math *x, const double *expected, cf_usize count, double eps)
{
  CHECK_TRUE(x->metadata.len == count);
  for(cf_usize i = 0; i < count; i++) CHECK_NEAR(cf64(x)[i], expected[i], eps);
}

static void free_sparse(cf_math_sparse *x)
{
  if(x == CF_NULL) return;
  free(x->values);
  free(x->row_offsets);
  free(x->col_indices);
  memset(x, 0, sizeof(*x));
}

static void test_primitives(void)
{
  CHECK_TRUE(cf_math_g8_mul_mod(0x57U, 0x83U) == 0xc1U);
  CHECK_TRUE(cf_math_rotl8(0x12U, 4U) == 0x21U);
  CHECK_TRUE(cf_math_rotr8(0x12U, 4U) == 0x21U);
  CHECK_TRUE(cf_math_rotl32(0x12345678U, 8U) == 0x34567812U);
  CHECK_TRUE(cf_math_rotr32(0x12345678U, 8U) == 0x78123456U);
  CHECK_TRUE(cf_math_min_usize(7U, 3U) == 3U);
  CHECK_TRUE(cf_math_max_usize(7U, 3U) == 7U);
  CHECK_TRUE(cf_math_dtype_size(CF_DTYPE_F64) == sizeof(double));
  CHECK_TRUE(cf_math_dtype_size(CF_DTYPE_F32) == sizeof(float));
  CHECK_TRUE(cf_math_dtype_size(CF_DTYPE_I32) == sizeof(cf_i32));
}

static void test_lifecycle_views_and_shape(void)
{
  cf_math x = {0};
  cf_math clone = {0};
  cf_math view = {0};
  cf_math reshaped = {0};
  cf_math flat = {0};
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {2, 3};
  cf_usize view_dim[CF_MATH_HIGHEST_RANK] = {2};
  cf_usize reshape_dim[CF_MATH_HIGHEST_RANK] = {3, 2};
  double values[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

  CHECK_STATUS(cf_math_alloc(&x, dim, 2, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL));
  fill_f64(&x, values);
  CHECK_TRUE(x.metadata.len == 6U);
  CHECK_TRUE(x.metadata.strides[0] == 3U);
  CHECK_TRUE(x.metadata.strides[1] == 1U);

  CHECK_STATUS(cf_math_clone(&clone, &x, CF_NULL));
  check_f64_array(&clone, values, 6U, 1e-12);

  CHECK_STATUS(cf_math_view(&view, &x, 2U, view_dim, 1U));
  CHECK_NEAR(cf64(&view)[0], 3.0, 1e-12);
  CHECK_NEAR(cf64(&view)[1], 4.0, 1e-12);

  CHECK_STATUS(cf_math_reshape(&reshaped, &x, reshape_dim, 2U));
  CHECK_TRUE(reshaped.dim[0] == 3U);
  CHECK_TRUE(reshaped.dim[1] == 2U);
  CHECK_NEAR(cf64(&reshaped)[5], 6.0, 1e-12);

  CHECK_STATUS(cf_math_flatten(&flat, &x, 0U, 1U));
  CHECK_TRUE(flat.rank == 1U);
  CHECK_TRUE(flat.dim[0] == 6U);

  CHECK_STATUS(cf_math_free(&flat, CF_NULL));
  CHECK_STATUS(cf_math_free(&reshaped, CF_NULL));
  CHECK_STATUS(cf_math_free(&view, CF_NULL));
  CHECK_STATUS(cf_math_free(&clone, CF_NULL));
  CHECK_STATUS(cf_math_free(&x, CF_NULL));
}

static void test_elementwise_and_reductions(void)
{
  cf_math x = {0};
  cf_math y = {0};
  cf_math out = {0};
  cf_math scalar = {0};
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {4};
  double xv[] = {1.0, 4.0, 9.0, 16.0};
  double yv[] = {2.0, 2.0, 3.0, 4.0};
  double expect_add[] = {3.0, 6.0, 12.0, 20.0};
  double expect_mul[] = {2.0, 8.0, 27.0, 64.0};
  double expect_sqrt[] = {1.0, 2.0, 3.0, 4.0};
  double expect_cumsum[] = {1.0, 5.0, 14.0, 30.0};

  CHECK_STATUS(cf_math_alloc(&x, dim, 1, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL));
  CHECK_STATUS(cf_math_alloc(&y, dim, 1, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL));
  fill_f64(&x, xv);
  fill_f64(&y, yv);

  CHECK_STATUS(cf_math_add(&out, &x, &y, CF_NULL));
  check_f64_array(&out, expect_add, 4U, 1e-12);
  CHECK_STATUS(cf_math_mul(&out, &x, &y, CF_NULL));
  check_f64_array(&out, expect_mul, 4U, 1e-12);
  CHECK_STATUS(cf_math_sqrt(&out, &x, CF_NULL));
  check_f64_array(&out, expect_sqrt, 4U, 1e-12);
  CHECK_STATUS(cf_math_cumsum(&out, &x, CF_NULL));
  check_f64_array(&out, expect_cumsum, 4U, 1e-12);
  CHECK_STATUS(cf_math_clamp(&out, &x, 3.0, 10.0, CF_NULL));
  CHECK_NEAR(cf64(&out)[0], 3.0, 1e-12);
  CHECK_NEAR(cf64(&out)[3], 10.0, 1e-12);

  CHECK_STATUS(cf_math_sum(&scalar, &x, CF_NULL));
  CHECK_NEAR(cf64(&scalar)[0], 30.0, 1e-12);
  CHECK_STATUS(cf_math_mean(&scalar, &x, CF_NULL));
  CHECK_NEAR(cf64(&scalar)[0], 7.5, 1e-12);
  CHECK_STATUS(cf_math_norm1(&scalar, &x, CF_NULL));
  CHECK_NEAR(cf64(&scalar)[0], 30.0, 1e-12);
  CHECK_STATUS(cf_math_dot(&scalar, &x, &y, CF_NULL));
  CHECK_NEAR(cf64(&scalar)[0], 101.0, 1e-12);
  CHECK_STATUS(cf_math_argmax(&scalar, &x, CF_NULL));
  CHECK_TRUE(i32(&scalar)[0] == 3);

  CHECK_STATUS(cf_math_free(&scalar, CF_NULL));
  CHECK_STATUS(cf_math_free(&out, CF_NULL));
  CHECK_STATUS(cf_math_free(&y, CF_NULL));
  CHECK_STATUS(cf_math_free(&x, CF_NULL));
}

static void test_linalg_activation_and_loss(void)
{
  cf_math a = {0};
  cf_math b = {0};
  cf_math out = {0};
  cf_math bias = {0};
  cf_math target = {0};
  cf_math loss = {0};
  cf_usize a_dim[CF_MATH_HIGHEST_RANK] = {2, 3};
  cf_usize b_dim[CF_MATH_HIGHEST_RANK] = {3, 2};
  cf_usize w_dim[CF_MATH_HIGHEST_RANK] = {2, 3};
  cf_usize bias_dim[CF_MATH_HIGHEST_RANK] = {2};
  double av[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double bv[] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  double matmul_expected[] = {58.0, 64.0, 139.0, 154.0};
  double wv[] = {1.0, 0.0, 1.0, 0.0, 1.0, 1.0};
  double biasv[] = {1.0, -1.0};
  double linear_expected[] = {5.0, 4.0, 11.0, 10.0};
  double targetv[] = {1.0, 1.0, 1.0, 1.0};

  CHECK_STATUS(cf_math_alloc(&a, a_dim, 2, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL));
  CHECK_STATUS(cf_math_alloc(&b, b_dim, 2, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL));
  fill_f64(&a, av);
  fill_f64(&b, bv);

  CHECK_STATUS(cf_math_matmul(&out, &a, &b, CF_NULL));
  check_f64_array(&out, matmul_expected, 4U, 1e-12);
  CHECK_STATUS(cf_math_free(&out, CF_NULL));
  CHECK_STATUS(cf_math_free(&b, CF_NULL));

  CHECK_STATUS(cf_math_alloc(&b, w_dim, 2, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL));
  fill_f64(&b, wv);
  CHECK_STATUS(cf_math_alloc(&bias, bias_dim, 1, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL));
  fill_f64(&bias, biasv);
  CHECK_STATUS(cf_math_linear(&out, &a, &b, &bias, CF_NULL));
  check_f64_array(&out, linear_expected, 4U, 1e-12);

  CHECK_STATUS(cf_math_relu(&out, &out, CF_NULL));
  CHECK_NEAR(cf64(&out)[0], 5.0, 1e-12);
  CHECK_STATUS(cf_math_sigmoid(&out, &bias, CF_NULL));
  CHECK_NEAR(cf64(&out)[0], 1.0 / (1.0 + exp(-1.0)), 1e-12);

  CHECK_STATUS(cf_math_alloc(&target, bias_dim, 1, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL));
  fill_f64(&target, targetv);
  CHECK_STATUS(cf_math_mse_loss(&loss, &out, &target, CF_NULL));
  CHECK_TRUE(isfinite(cf64(&loss)[0]));

  CHECK_STATUS(cf_math_free(&loss, CF_NULL));
  CHECK_STATUS(cf_math_free(&target, CF_NULL));
  CHECK_STATUS(cf_math_free(&bias, CF_NULL));
  CHECK_STATUS(cf_math_free(&out, CF_NULL));
  CHECK_STATUS(cf_math_free(&b, CF_NULL));
  CHECK_STATUS(cf_math_free(&a, CF_NULL));
}

static void test_dropout_embedding_sparse_and_unsupported(void)
{
  cf_math w = {0};
  cf_math idx = {0};
  cf_math out = {0};
  cf_math dense = {0};
  cf_math restored = {0};
  cf_math_dropout_state dropout = {0};
  cf_math_sparse csr = {0};
  cf_usize w_dim[CF_MATH_HIGHEST_RANK] = {3, 2};
  cf_usize idx_dim[CF_MATH_HIGHEST_RANK] = {2};
  cf_usize dense_dim[CF_MATH_HIGHEST_RANK] = {2, 3};
  double wv[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double densev[] = {10.0, 0.0, 3.0, 0.0, 2.0, 0.0};

  CHECK_STATUS(cf_math_alloc(&w, w_dim, 2, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL));
  CHECK_STATUS(cf_math_alloc(&idx, idx_dim, 1, CF_DTYPE_I32, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL));
  fill_f64(&w, wv);
  i32(&idx)[0] = 2;
  i32(&idx)[1] = 0;
  CHECK_STATUS(cf_math_embed_fwd(&out, &w, &idx, CF_NULL));
  CHECK_NEAR(cf64(&out)[0], 5.0, 1e-12);
  CHECK_NEAR(cf64(&out)[3], 2.0, 1e-12);
  CHECK_STATUS(cf_math_free(&out, CF_NULL));

  CHECK_STATUS(cf_math_dropout_fwd(&out, &dropout, &w, 0.0, 123U, CF_NULL));
  check_f64_array(&out, wv, 6U, 1e-12);
  CHECK_STATUS(cf_math_dropout_train_set(&dropout, 0.5, CF_FALSE, CF_NULL));
  CHECK_NEAR(dropout.probability, 0.0, 1e-12);

  CHECK_STATUS(cf_math_alloc(&dense, dense_dim, 2, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL));
  fill_f64(&dense, densev);
  CHECK_STATUS(cf_math_dense_to_csr(&csr, &dense, 0.0, CF_NULL));
  CHECK_TRUE(csr.nnz == 3U);
  CHECK_STATUS(cf_math_csr_to_dense(&restored, &csr, CF_NULL));
  check_f64_array(&restored, densev, 6U, 1e-12);

  CHECK_TRUE(cf_math_grad_allreduce(&w, 1U, CF_NULL) == CF_OK);
  CHECK_TRUE(cf_math_grad_allreduce(&w, 2U, CF_NULL) == CF_ERR_UNSUPPORTED);
  CHECK_TRUE(cf_math_rnn_fwd_train(&out, CF_NULL, &w, CF_NULL, CF_NULL) == CF_ERR_UNSUPPORTED);

  free(dropout.reserve);
  free_sparse(&csr);
  CHECK_STATUS(cf_math_free(&restored, CF_NULL));
  CHECK_STATUS(cf_math_free(&dense, CF_NULL));
  CHECK_STATUS(cf_math_free(&out, CF_NULL));
  CHECK_STATUS(cf_math_free(&idx, CF_NULL));
  CHECK_STATUS(cf_math_free(&w, CF_NULL));
}

static void test_cuda_guard(void)
{
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  int device_count = 0;
  cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
  if(cuda_status == cudaSuccess && device_count > 0)
  {
    cf_math_cuda_context ctx = {0};
    cf_status status = cf_math_context_init(&ctx, 0);
    CHECK_TRUE(status == CF_OK || status == CF_ERR_CUDA_DEVICE || status == CF_ERR_CUDA_RUNTIME || status == CF_ERR_CUDA);
    if(status == CF_OK) CHECK_STATUS(cf_math_context_destroy(&ctx));
  }
  else
  {
    printf("CUDA runtime present, but no usable CUDA device was reported; GPU math tests skipped.\n");
  }
#else
  printf("CUDA runtime headers are not available in this build; GPU math tests skipped.\n");
#endif
}

int main(void)
{
  test_primitives();
  test_lifecycle_views_and_shape();
  test_elementwise_and_reductions();
  test_linalg_activation_and_loss();
  test_dropout_embedding_sparse_and_unsupported();
  test_cuda_guard();

  if(g_failures != 0)
  {
    printf("math tests failed: %d\n", g_failures);
    return 1;
  }

  printf("math tests passed\n");
  return 0;
}
