/*
 * CF Framework
 * Copyright (C) 2026 Orion
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#define _POSIX_C_SOURCE 200809L

#include "MATH/cf_math.h"
#include "RUNTIME/cf_status.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BENCH_ITERS_FAST 50000U
#define BENCH_ITERS_TENSOR 2000U
#define BENCH_ITERS_HEAVY 40U
#define BENCH_ITERS_STRESS 8U

static double bench_now_ms(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static double *f64(cf_math *x)
{
  return (double *)x->data;
}

static cf_i32 *i32(cf_math *x)
{
  return (cf_i32 *)x->data;
}

static void bench_fill_pattern(cf_math *x, double start)
{
  for(cf_usize i = 0; i < x->metadata.len; i++)
    f64(x)[i] = start + (double)(i % 17U) * 0.125;
}

static void bench_print_status(const char *name, cf_status status, cf_usize iters, double elapsed_ms)
{
  double per_op_us = iters == 0 ? 0.0 : (elapsed_ms * 1000.0) / (double)iters;
  printf("%-34s status=%-18s total=%10.3f ms  avg=%10.3f us/op\n",
         name,
         cf_status_as_char(status),
         elapsed_ms,
         per_op_us);
}

static void bench_print_u64(const char *name, unsigned long long value, cf_usize iters, double elapsed_ms)
{
  double per_op_us = iters == 0 ? 0.0 : (elapsed_ms * 1000.0) / (double)iters;
  printf("%-34s value=%llu total=%10.3f ms  avg=%10.3f us/op\n",
         name,
         value,
         elapsed_ms,
         per_op_us);
}

static void bench_print_tensor(const char *name, const cf_math *x)
{
  if(x == CF_NULL || x->data == CF_NULL)
  {
    printf("  %s result: <empty>\n", name);
    return;
  }

  if(x->metadata.device == CF_DEVICE_CUDA)
  {
    printf("  %s result: <cuda tensor>\n", name);
    return;
  }

  if(x->rank == 2 && x->metadata.len <= 16U)
  {
    printf("  %s matrix result:\n", name);
    cf_math_print(x);
    return;
  }

  if(x->metadata.len <= 8U)
  {
    printf("  %s vector result:\n", name);
    cf_math_print(x);
  }
}

#define BENCH_STATUS(name, iters, expr, result) do { \
  cf_status bench_status__ = CF_OK; \
  double bench_start__ = bench_now_ms(); \
  for(cf_usize bench_i__ = 0; bench_i__ < (iters); bench_i__++) bench_status__ = (expr); \
  double bench_elapsed__ = bench_now_ms() - bench_start__; \
  bench_print_status((name), bench_status__, (iters), bench_elapsed__); \
  bench_print_tensor((name), (result)); \
} while(0)

#define BENCH_VALUE(name, iters, type, expr) do { \
  type bench_value__ = 0; \
  double bench_start__ = bench_now_ms(); \
  for(cf_usize bench_i__ = 0; bench_i__ < (iters); bench_i__++) bench_value__ = (expr); \
  double bench_elapsed__ = bench_now_ms() - bench_start__; \
  bench_print_u64((name), (unsigned long long)bench_value__, (iters), bench_elapsed__); \
} while(0)

static void bench_free_tensor(cf_math *x)
{
  cf_status status = cf_math_free(x, CF_NULL);
  if(status != CF_OK) printf("  free status=%s\n", cf_status_as_char(status));
}

static void bench_free_sparse(cf_math_sparse *x)
{
  if(x == CF_NULL) return;
  free(x->values);
  free(x->row_offsets);
  free(x->col_indices);
  memset(x, 0, sizeof(*x));
}

static void bench_cpu_surface(void)
{
  cf_usize dim4[CF_MATH_HIGHEST_RANK] = {4};
  cf_usize dim2[CF_MATH_HIGHEST_RANK] = {2};
  cf_usize dim23[CF_MATH_HIGHEST_RANK] = {2, 3};
  cf_usize dim32[CF_MATH_HIGHEST_RANK] = {3, 2};
  cf_usize dim22[CF_MATH_HIGHEST_RANK] = {2, 2};
  cf_usize dim122[CF_MATH_HIGHEST_RANK] = {1, 2, 2};
  cf_usize dim133[CF_MATH_HIGHEST_RANK] = {1, 1, 3, 3};
  cf_usize dim1122[CF_MATH_HIGHEST_RANK] = {1, 1, 2, 2};
  cf_usize dim11133[CF_MATH_HIGHEST_RANK] = {1, 1, 1, 3, 3};
  cf_usize dim11122[CF_MATH_HIGHEST_RANK] = {1, 1, 1, 2, 2};
  cf_usize reshape_dim[CF_MATH_HIGHEST_RANK] = {3, 2};
  cf_usize expand_dim[CF_MATH_HIGHEST_RANK] = {2, 4};
  cf_usize axes[CF_MATH_HIGHEST_RANK] = {1, 0};
  cf_usize slice_start[CF_MATH_HIGHEST_RANK] = {0, 1};
  cf_usize slice_len[CF_MATH_HIGHEST_RANK] = {2, 2};
  cf_usize pad_before[CF_MATH_HIGHEST_RANK] = {1, 1};
  cf_usize pad_after[CF_MATH_HIGHEST_RANK] = {1, 1};
  cf_math_conv2d_params conv = {0, 0, 1, 1, 1, 1, 1};
  cf_math a = {0};
  cf_math b = {0};
  cf_math c = {0};
  cf_math out = {0};
  cf_math aux = {0};
  cf_math aux2 = {0};
  cf_math aux3 = {0};
  cf_math loss = {0};
  cf_math idx = {0};
  cf_math labels = {0};
  cf_math conv_x = {0};
  cf_math conv_w = {0};
  cf_math conv_b = {0};
  cf_math conv_out = {0};
  cf_math conv3_x = {0};
  cf_math conv3_w = {0};
  cf_math gamma = {0};
  cf_math beta = {0};
  cf_math sparse_vec = {0};
  cf_math_sparse csr = {0};
  cf_math_dropout_state dropout = {0};
  cf_math_rnn_state rnn = {0};
  const cf_math *concat_inputs[2];
  cf_math split_outs[2] = {{0}, {0}};

  printf("\n== CPU cf_math benchmark surface ==\n");

  BENCH_VALUE("cf_math_g8_mul_mod", BENCH_ITERS_FAST, cf_u8, cf_math_g8_mul_mod(0x57U, 0x83U));
  BENCH_VALUE("cf_math_rotl8", BENCH_ITERS_FAST, cf_u8, cf_math_rotl8(0x12U, 4U));
  BENCH_VALUE("cf_math_rotr8", BENCH_ITERS_FAST, cf_u8, cf_math_rotr8(0x12U, 4U));
  BENCH_VALUE("cf_math_rotl32", BENCH_ITERS_FAST, cf_u32, cf_math_rotl32(0x12345678U, 8U));
  BENCH_VALUE("cf_math_rotr32", BENCH_ITERS_FAST, cf_u32, cf_math_rotr32(0x12345678U, 8U));
  BENCH_VALUE("cf_math_min_usize", BENCH_ITERS_FAST, cf_usize, cf_math_min_usize(7U, 3U));
  BENCH_VALUE("cf_math_max_usize", BENCH_ITERS_FAST, cf_usize, cf_math_max_usize(7U, 3U));
  BENCH_VALUE("cf_math_dtype_size", BENCH_ITERS_FAST, cf_usize, cf_math_dtype_size(CF_DTYPE_F64));

  BENCH_STATUS("cf_math_context_init(NULL)", 1U, cf_math_context_init(CF_NULL, 0), CF_NULL);
  BENCH_STATUS("cf_math_context_destroy(NULL)", 1U, cf_math_context_destroy(CF_NULL), CF_NULL);
  BENCH_STATUS("cf_math_workspace_reserve(NULL)", 1U, cf_math_workspace_reserve(CF_NULL, 4096U), CF_NULL);

  BENCH_STATUS("cf_math_alloc", 1U, cf_math_alloc(&a, dim23, 2, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL), &a);
  bench_fill_pattern(&a, 1.0);
  BENCH_STATUS("cf_math_alloc_pinned", 1U, cf_math_alloc_pinned(&aux, dim4, 1, CF_DTYPE_F64, CF_NULL), &aux);
  BENCH_STATUS("cf_math_alloc_managed", 1U, cf_math_alloc_managed(&aux2, dim4, 1, CF_DTYPE_F64, CF_NULL), &aux2);
  bench_free_tensor(&aux);
  bench_free_tensor(&aux2);

  BENCH_STATUS("cf_math_clone", BENCH_ITERS_TENSOR, cf_math_clone(&out, &a, CF_NULL), &out);
  bench_free_tensor(&out);
  BENCH_STATUS("cf_math_view", BENCH_ITERS_TENSOR, cf_math_view(&out, &a, 1U, dim4, 1), &out);
  bench_free_tensor(&out);
  BENCH_STATUS("cf_math_contiguous", BENCH_ITERS_TENSOR, cf_math_contiguous(&out, &a, CF_NULL), &out);
  bench_free_tensor(&out);
  BENCH_STATUS("cf_math_to_host", BENCH_ITERS_TENSOR, cf_math_to_host(&out, &a, CF_NULL), &out);
  bench_free_tensor(&out);
  BENCH_STATUS("cf_math_to_device", 1U, cf_math_to_device(&out, &a, 0, CF_NULL), &out);
  bench_free_tensor(&out);

  BENCH_STATUS("cf_math_fill", BENCH_ITERS_TENSOR, cf_math_fill(&a, 2.0, CF_NULL), &a);
  BENCH_STATUS("cf_math_zeros", BENCH_ITERS_TENSOR, cf_math_zeros(&a, CF_NULL), &a);
  BENCH_STATUS("cf_math_ones", BENCH_ITERS_TENSOR, cf_math_ones(&a, CF_NULL), &a);
  BENCH_STATUS("cf_math_rand_uniform", BENCH_ITERS_TENSOR, cf_math_rand_uniform(&a, -1.0, 1.0, 11U, CF_NULL), &a);
  BENCH_STATUS("cf_math_rand_normal", BENCH_ITERS_TENSOR, cf_math_rand_normal(&a, 0.0, 1.0, 12U, CF_NULL), &a);
  BENCH_STATUS("cf_math_rand_bernoulli", BENCH_ITERS_TENSOR, cf_math_rand_bernoulli(&a, 0.5, 13U, CF_NULL), &a);
  BENCH_STATUS("cf_math_init_xavier_uniform", BENCH_ITERS_TENSOR, cf_math_init_xavier_uniform(&a, 3U, 2U, 14U, CF_NULL), &a);
  BENCH_STATUS("cf_math_init_xavier_normal", BENCH_ITERS_TENSOR, cf_math_init_xavier_normal(&a, 3U, 2U, 15U, CF_NULL), &a);
  BENCH_STATUS("cf_math_init_kaiming_normal", BENCH_ITERS_TENSOR, cf_math_init_kaiming_normal(&a, 3U, 16U, CF_NULL), &a);
  BENCH_STATUS("cf_math_init_kaiming_uniform", BENCH_ITERS_TENSOR, cf_math_init_kaiming_uniform(&a, 3U, 17U, CF_NULL), &a);
  BENCH_STATUS("cf_math_init_orthogonal", BENCH_ITERS_TENSOR, cf_math_init_orthogonal(&a, 18U, CF_NULL), &a);
  BENCH_STATUS("cf_math_init_eye", BENCH_ITERS_TENSOR, cf_math_init_eye(&a, CF_NULL), &a);

  BENCH_STATUS("cf_math_alloc b", 1U, cf_math_alloc(&b, dim23, 2, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL), &b);
  bench_fill_pattern(&a, 1.0);
  bench_fill_pattern(&b, 0.5);
  BENCH_STATUS("cf_math_add", BENCH_ITERS_TENSOR, cf_math_add(&out, &a, &b, CF_NULL), &out);
  BENCH_STATUS("cf_math_add_scalar", BENCH_ITERS_TENSOR, cf_math_add_scalar(&out, &a, 3.0, CF_NULL), &out);
  BENCH_STATUS("cf_math_sub", BENCH_ITERS_TENSOR, cf_math_sub(&out, &a, &b, CF_NULL), &out);
  BENCH_STATUS("cf_math_mul", BENCH_ITERS_TENSOR, cf_math_mul(&out, &a, &b, CF_NULL), &out);
  BENCH_STATUS("cf_math_mul_scalar", BENCH_ITERS_TENSOR, cf_math_mul_scalar(&out, &a, 2.0, CF_NULL), &out);
  BENCH_STATUS("cf_math_div", BENCH_ITERS_TENSOR, cf_math_div(&out, &a, &b, CF_NULL), &out);
  BENCH_STATUS("cf_math_div_scalar", BENCH_ITERS_TENSOR, cf_math_div_scalar(&out, &a, 2.0, CF_NULL), &out);
  BENCH_STATUS("cf_math_pow", BENCH_ITERS_TENSOR, cf_math_pow(&out, &a, 2.0, CF_NULL), &out);
  BENCH_STATUS("cf_math_sqrt", BENCH_ITERS_TENSOR, cf_math_sqrt(&out, &a, CF_NULL), &out);
  BENCH_STATUS("cf_math_rsqrt", BENCH_ITERS_TENSOR, cf_math_rsqrt(&out, &a, CF_NULL), &out);
  BENCH_STATUS("cf_math_exp", BENCH_ITERS_TENSOR, cf_math_exp(&out, &a, CF_NULL), &out);
  BENCH_STATUS("cf_math_log", BENCH_ITERS_TENSOR, cf_math_log(&out, &a, CF_NULL), &out);
  BENCH_STATUS("cf_math_abs", BENCH_ITERS_TENSOR, cf_math_abs(&out, &a, CF_NULL), &out);
  BENCH_STATUS("cf_math_neg", BENCH_ITERS_TENSOR, cf_math_neg(&out, &a, CF_NULL), &out);
  BENCH_STATUS("cf_math_clamp", BENCH_ITERS_TENSOR, cf_math_clamp(&out, &a, 0.5, 1.5, CF_NULL), &out);
  BENCH_STATUS("cf_math_sign", BENCH_ITERS_TENSOR, cf_math_sign(&out, &a, CF_NULL), &out);

  BENCH_STATUS("cf_math_sum", BENCH_ITERS_TENSOR, cf_math_sum(&c, &a, CF_NULL), &c);
  BENCH_STATUS("cf_math_sum_axis", BENCH_ITERS_TENSOR, cf_math_sum_axis(&c, &a, 0U, CF_NULL), &c);
  BENCH_STATUS("cf_math_mean", BENCH_ITERS_TENSOR, cf_math_mean(&c, &a, CF_NULL), &c);
  BENCH_STATUS("cf_math_mean_axis", BENCH_ITERS_TENSOR, cf_math_mean_axis(&c, &a, 1U, CF_NULL), &c);
  BENCH_STATUS("cf_math_var", BENCH_ITERS_TENSOR, cf_math_var(&c, &a, CF_NULL), &c);
  BENCH_STATUS("cf_math_std", BENCH_ITERS_TENSOR, cf_math_std(&c, &a, CF_NULL), &c);
  BENCH_STATUS("cf_math_norm2", BENCH_ITERS_TENSOR, cf_math_norm2(&c, &a, CF_NULL), &c);
  BENCH_STATUS("cf_math_norm1", BENCH_ITERS_TENSOR, cf_math_norm1(&c, &a, CF_NULL), &c);
  BENCH_STATUS("cf_math_max", BENCH_ITERS_TENSOR, cf_math_max(&c, &a, CF_NULL), &c);
  BENCH_STATUS("cf_math_min", BENCH_ITERS_TENSOR, cf_math_min(&c, &a, CF_NULL), &c);
  BENCH_STATUS("cf_math_argmax", BENCH_ITERS_TENSOR, cf_math_argmax(&c, &a, CF_NULL), &c);
  BENCH_STATUS("cf_math_argmin", BENCH_ITERS_TENSOR, cf_math_argmin(&c, &a, CF_NULL), &c);
  BENCH_STATUS("cf_math_dot", BENCH_ITERS_TENSOR, cf_math_dot(&c, &a, &b, CF_NULL), &c);
  BENCH_STATUS("cf_math_cumsum", BENCH_ITERS_TENSOR, cf_math_cumsum(&out, &a, CF_NULL), &out);

  bench_free_tensor(&out);
  bench_free_tensor(&c);
  bench_free_tensor(&b);
  bench_free_tensor(&a);

  BENCH_STATUS("cf_math_alloc a23", 1U, cf_math_alloc(&a, dim23, 2, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL), &a);
  BENCH_STATUS("cf_math_alloc b32", 1U, cf_math_alloc(&b, dim32, 2, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL), &b);
  BENCH_STATUS("cf_math_alloc c22", 1U, cf_math_alloc(&c, dim22, 2, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL), &c);
  bench_fill_pattern(&a, 1.0);
  bench_fill_pattern(&b, 0.5);
  bench_fill_pattern(&c, 0.25);
  BENCH_STATUS("cf_math_matmul", BENCH_ITERS_HEAVY, cf_math_matmul(&out, &a, &b, CF_NULL), &out);
  BENCH_STATUS("cf_math_matmul_t", BENCH_ITERS_HEAVY, cf_math_matmul_t(&out, &a, &b, CF_FALSE, CF_FALSE, CF_NULL), &out);
  bench_free_tensor(&out);
  BENCH_STATUS("cf_math_alloc batched a", 1U, cf_math_alloc(&out, dim122, 3, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL), &out);
  bench_fill_pattern(&out, 1.0);
  BENCH_STATUS("cf_math_matmul_batched", BENCH_ITERS_HEAVY, cf_math_matmul_batched(&aux, &out, &out, CF_NULL), &aux);
  BENCH_STATUS("cf_math_linear", BENCH_ITERS_HEAVY, cf_math_linear(&aux2, &a, &b, &c, CF_NULL), &aux2);
  BENCH_STATUS("cf_math_linear_fused_relu", BENCH_ITERS_HEAVY, cf_math_linear_fused_relu(&aux2, &a, &b, &c, CF_NULL), &aux2);
  BENCH_STATUS("cf_math_linear_fused_gelu", BENCH_ITERS_HEAVY, cf_math_linear_fused_gelu(&aux2, &a, &b, &c, CF_NULL), &aux2);
  BENCH_STATUS("cf_math_linear_backward_W", BENCH_ITERS_HEAVY, cf_math_linear_backward_W(&aux, &aux2, &a, CF_NULL), &aux);
  BENCH_STATUS("cf_math_linear_backward_x", BENCH_ITERS_HEAVY, cf_math_linear_backward_x(&aux, &aux2, &b, CF_NULL), &aux);
  BENCH_STATUS("cf_math_linear_backward_b", BENCH_ITERS_HEAVY, cf_math_linear_backward_b(&aux, &aux2, CF_NULL), &aux);
  BENCH_STATUS("cf_math_outer", BENCH_ITERS_HEAVY, cf_math_outer(&aux, &c, &c, CF_NULL), &aux);
  BENCH_STATUS("cf_math_matvec", BENCH_ITERS_HEAVY, cf_math_matvec(&aux3, &a, &b, CF_NULL), &aux3);
  BENCH_STATUS("cf_math_transpose", BENCH_ITERS_HEAVY, cf_math_transpose(&aux, &a, CF_NULL), &aux);
  BENCH_STATUS("cf_math_scale", BENCH_ITERS_TENSOR, cf_math_scale(&aux, &a, 0.5, CF_NULL), &aux);

  BENCH_STATUS("cf_math_relu", BENCH_ITERS_TENSOR, cf_math_relu(&aux, &a, CF_NULL), &aux);
  BENCH_STATUS("cf_math_relu_bwd", BENCH_ITERS_TENSOR, cf_math_relu_bwd(&aux, &a, &a, CF_NULL), &aux);
  BENCH_STATUS("cf_math_leaky_relu", BENCH_ITERS_TENSOR, cf_math_leaky_relu(&aux, &a, 0.01, CF_NULL), &aux);
  BENCH_STATUS("cf_math_elu", BENCH_ITERS_TENSOR, cf_math_elu(&aux, &a, 1.0, CF_NULL), &aux);
  BENCH_STATUS("cf_math_sigmoid", BENCH_ITERS_TENSOR, cf_math_sigmoid(&aux, &a, CF_NULL), &aux);
  BENCH_STATUS("cf_math_sigmoid_bwd", BENCH_ITERS_TENSOR, cf_math_sigmoid_bwd(&aux, &a, &a, CF_NULL), &aux);
  BENCH_STATUS("cf_math_tanh", BENCH_ITERS_TENSOR, cf_math_tanh(&aux, &a, CF_NULL), &aux);
  BENCH_STATUS("cf_math_tanh_bwd", BENCH_ITERS_TENSOR, cf_math_tanh_bwd(&aux, &a, &a, CF_NULL), &aux);
  BENCH_STATUS("cf_math_gelu", BENCH_ITERS_TENSOR, cf_math_gelu(&aux, &a, CF_NULL), &aux);
  BENCH_STATUS("cf_math_gelu_approx", BENCH_ITERS_TENSOR, cf_math_gelu_approx(&aux, &a, CF_NULL), &aux);
  BENCH_STATUS("cf_math_gelu_bwd", BENCH_ITERS_TENSOR, cf_math_gelu_bwd(&aux, &a, &a, CF_NULL), &aux);
  BENCH_STATUS("cf_math_swish", BENCH_ITERS_TENSOR, cf_math_swish(&aux, &a, 1.0, CF_NULL), &aux);
  BENCH_STATUS("cf_math_silu", BENCH_ITERS_TENSOR, cf_math_silu(&aux, &a, CF_NULL), &aux);
  BENCH_STATUS("cf_math_softplus", BENCH_ITERS_TENSOR, cf_math_softplus(&aux, &a, CF_NULL), &aux);
  BENCH_STATUS("cf_math_mish", BENCH_ITERS_TENSOR, cf_math_mish(&aux, &a, CF_NULL), &aux);

  BENCH_STATUS("cf_math_softmax_fwd", BENCH_ITERS_TENSOR, cf_math_softmax_fwd(&aux, &a, 1U, CF_SOFTMAX_CHANNEL, CF_NULL), &aux);
  BENCH_STATUS("cf_math_softmax_bwd", BENCH_ITERS_TENSOR, cf_math_softmax_bwd(&aux3, &a, &aux, 1U, CF_NULL), &aux3);
  BENCH_STATUS("cf_math_log_softmax_fwd", BENCH_ITERS_TENSOR, cf_math_log_softmax_fwd(&aux, &a, 1U, CF_NULL), &aux);
  BENCH_STATUS("cf_math_log_softmax_bwd", BENCH_ITERS_TENSOR, cf_math_log_softmax_bwd(&aux3, &a, &aux, 1U, CF_NULL), &aux3);
  BENCH_STATUS("cf_math_cross_entropy", BENCH_ITERS_TENSOR, cf_math_cross_entropy(&loss, &aux3, &a, &aux, 1U, CF_NULL), &loss);
  BENCH_STATUS("cf_math_cross_entropy_bwd", BENCH_ITERS_TENSOR, cf_math_cross_entropy_bwd(&aux3, &aux, &a, CF_NULL), &aux3);

  BENCH_STATUS("cf_math_alloc labels", 1U, cf_math_alloc(&labels, dim2, 1, CF_DTYPE_I32, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL), &labels);
  i32(&labels)[0] = 0;
  i32(&labels)[1] = 1;
  BENCH_STATUS("cf_math_nll_loss", BENCH_ITERS_TENSOR, cf_math_nll_loss(&loss, &aux, &labels, CF_NULL), &loss);
  BENCH_STATUS("cf_math_mse_loss", BENCH_ITERS_TENSOR, cf_math_mse_loss(&loss, &a, &aux, CF_NULL), &loss);
  BENCH_STATUS("cf_math_mse_loss_bwd", BENCH_ITERS_TENSOR, cf_math_mse_loss_bwd(&aux3, &a, &aux, CF_NULL), &aux3);
  BENCH_STATUS("cf_math_bce_loss", BENCH_ITERS_TENSOR, cf_math_bce_loss(&loss, &aux, &a, CF_NULL), &loss);
  BENCH_STATUS("cf_math_huber_loss", BENCH_ITERS_TENSOR, cf_math_huber_loss(&loss, &a, &aux, 1.0, CF_NULL), &loss);
  BENCH_STATUS("cf_math_focal_loss", BENCH_ITERS_TENSOR, cf_math_focal_loss(&loss, &aux, &a, 0.25, 2.0, CF_NULL), &loss);

  BENCH_STATUS("cf_math_attn_scores", BENCH_ITERS_HEAVY, cf_math_attn_scores(&aux, &c, &c, 0.70710678, CF_NULL), &aux);
  BENCH_STATUS("cf_math_attn_mask_add", BENCH_ITERS_HEAVY, cf_math_attn_mask_add(&aux3, &aux, &c, CF_NULL), &aux3);
  BENCH_STATUS("cf_math_attn_softmax", BENCH_ITERS_HEAVY, cf_math_attn_softmax(&aux3, &aux, CF_NULL), &aux3);
  BENCH_STATUS("cf_math_attn_context", BENCH_ITERS_HEAVY, cf_math_attn_context(&aux3, &aux, &c, CF_NULL), &aux3);
  BENCH_STATUS("cf_math_attn_proj", BENCH_ITERS_HEAVY, cf_math_attn_proj(&aux3, &c, &c, CF_NULL), &aux3);
  BENCH_STATUS("cf_math_mha_fwd", BENCH_ITERS_HEAVY, cf_math_mha_fwd(&aux3, &c, &c, &c, &c, 1U, CF_NULL), &aux3);
  BENCH_STATUS("cf_math_mha_bwd", 1U, cf_math_mha_bwd(&aux, &aux2, &aux3, &out, &c, CF_NULL), &aux);
  BENCH_STATUS("cf_math_attn_dropout_fwd", BENCH_ITERS_TENSOR, cf_math_attn_dropout_fwd(&aux, &dropout, &a, 0.0, 20U, CF_NULL), &aux);
  BENCH_STATUS("cf_math_rope_fwd", BENCH_ITERS_HEAVY, cf_math_rope_fwd(&aux, &a, &a, &a, CF_NULL), &aux);
  BENCH_STATUS("cf_math_rope_bwd", BENCH_ITERS_HEAVY, cf_math_rope_bwd(&aux, &a, &a, &a, CF_NULL), &aux);
  BENCH_STATUS("cf_math_causal_mask", BENCH_ITERS_HEAVY, cf_math_causal_mask(&c, CF_NULL), &c);

  BENCH_STATUS("cf_math_dropout_fwd", BENCH_ITERS_TENSOR, cf_math_dropout_fwd(&aux, &dropout, &a, 0.0, 21U, CF_NULL), &aux);
  BENCH_STATUS("cf_math_dropout_bwd", BENCH_ITERS_TENSOR, cf_math_dropout_bwd(&aux3, &dropout, &a, CF_NULL), &aux3);
  BENCH_STATUS("cf_math_dropout_train_set", BENCH_ITERS_TENSOR, cf_math_dropout_train_set(&dropout, 0.2, CF_TRUE, CF_NULL), CF_NULL);

  BENCH_STATUS("cf_math_alloc idx", 1U, cf_math_alloc(&idx, dim2, 1, CF_DTYPE_I32, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL), &idx);
  i32(&idx)[0] = 0;
  i32(&idx)[1] = 1;
  BENCH_STATUS("cf_math_embed_fwd", BENCH_ITERS_TENSOR, cf_math_embed_fwd(&aux, &b, &idx, CF_NULL), &aux);
  BENCH_STATUS("cf_math_embed_bwd", BENCH_ITERS_TENSOR, cf_math_embed_bwd(&b, &idx, &aux, CF_NULL), &b);
  BENCH_STATUS("cf_math_embed_bwd_atomic", BENCH_ITERS_TENSOR, cf_math_embed_bwd_atomic(&b, &idx, &aux, CF_NULL), &b);

  BENCH_STATUS("cf_math_rnn_fwd_train", 1U, cf_math_rnn_fwd_train(&aux, &rnn, &a, &c, CF_NULL), &aux);
  BENCH_STATUS("cf_math_rnn_fwd_infer", 1U, cf_math_rnn_fwd_infer(&aux, &rnn, &a, &c, CF_NULL), &aux);
  BENCH_STATUS("cf_math_rnn_bwd_data", 1U, cf_math_rnn_bwd_data(&aux, &aux2, &rnn, &a, CF_NULL), &aux);
  BENCH_STATUS("cf_math_rnn_bwd_weights", 1U, cf_math_rnn_bwd_weights(&aux, &rnn, &a, &c, CF_NULL), &aux);
  BENCH_STATUS("cf_math_lstm_fwd_train", 1U, cf_math_lstm_fwd_train(&aux, &rnn, &a, &c, &c, CF_NULL), &aux);
  BENCH_STATUS("cf_math_lstm_bwd_data", 1U, cf_math_lstm_bwd_data(&aux, &aux2, &aux3, &rnn, &a, CF_NULL), &aux);
  BENCH_STATUS("cf_math_gru_fwd_train", 1U, cf_math_gru_fwd_train(&aux, &rnn, &a, &c, CF_NULL), &aux);

  BENCH_STATUS("cf_math_alloc sparse_vec", 1U, cf_math_alloc(&sparse_vec, dim2, 1, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL), &sparse_vec);
  bench_fill_pattern(&sparse_vec, 1.0);
  BENCH_STATUS("cf_math_dense_to_csr", BENCH_ITERS_HEAVY, cf_math_dense_to_csr(&csr, &c, 0.0, CF_NULL), CF_NULL);
  BENCH_STATUS("cf_math_spmv", BENCH_ITERS_HEAVY, cf_math_spmv(&aux, &csr, &sparse_vec, CF_NULL), &aux);
  BENCH_STATUS("cf_math_spmm", BENCH_ITERS_HEAVY, cf_math_spmm(&aux, &csr, &c, CF_NULL), &aux);
  BENCH_STATUS("cf_math_spgemm", 1U, cf_math_spgemm(&csr, &csr, &csr, CF_NULL), CF_NULL);
  BENCH_STATUS("cf_math_csr_to_dense", BENCH_ITERS_HEAVY, cf_math_csr_to_dense(&aux, &csr, CF_NULL), &aux);
  BENCH_STATUS("cf_math_sparse_attn", BENCH_ITERS_HEAVY, cf_math_sparse_attn(&aux, &csr, &c, CF_NULL), &aux);

  BENCH_STATUS("cf_math_sgd_step", BENCH_ITERS_TENSOR, cf_math_sgd_step(&a, &b, 0.001, CF_NULL), &a);
  BENCH_STATUS("cf_math_sgd_momentum", BENCH_ITERS_TENSOR, cf_math_sgd_momentum(&a, &b, &b, 0.001, 0.9, CF_NULL), &a);
  BENCH_STATUS("cf_math_adam_step", BENCH_ITERS_TENSOR, cf_math_adam_step(&a, &b, &aux, &b, 0.001, 0.9, 0.999, 1e-8, 1U, CF_NULL), &a);
  BENCH_STATUS("cf_math_adamw_step", BENCH_ITERS_TENSOR, cf_math_adamw_step(&a, &b, &aux, &b, 0.001, 0.9, 0.999, 1e-8, 0.01, 1U, CF_NULL), &a);
  BENCH_STATUS("cf_math_rmsprop_step", BENCH_ITERS_TENSOR, cf_math_rmsprop_step(&a, &b, &b, 0.001, 0.99, 1e-8, CF_NULL), &a);
  BENCH_STATUS("cf_math_grad_clip_norm", BENCH_ITERS_TENSOR, cf_math_grad_clip_norm(&a, 1.0, CF_NULL), &a);
  BENCH_STATUS("cf_math_grad_clip_value", BENCH_ITERS_TENSOR, cf_math_grad_clip_value(&a, 0.5, CF_NULL), &a);
  BENCH_STATUS("cf_math_weight_decay", BENCH_ITERS_TENSOR, cf_math_weight_decay(&a, &b, 0.01, CF_NULL), &a);
  BENCH_STATUS("cf_math_lr_scale", BENCH_ITERS_TENSOR, cf_math_lr_scale(&a, 0.1, CF_NULL), &a);
  BENCH_STATUS("cf_math_grad_allreduce", BENCH_ITERS_TENSOR, cf_math_grad_allreduce(&a, 1U, CF_NULL), &a);
  BENCH_STATUS("cf_math_grad_zero", BENCH_ITERS_TENSOR, cf_math_grad_zero(&a, CF_NULL), &a);

  BENCH_STATUS("cf_math_reshape", BENCH_ITERS_TENSOR, cf_math_reshape(&aux, &a, reshape_dim, 2U), &aux);
  BENCH_STATUS("cf_math_permute", BENCH_ITERS_TENSOR, cf_math_permute(&aux2, &a, axes, CF_NULL), &aux2);
  BENCH_STATUS("cf_math_squeeze", BENCH_ITERS_TENSOR, cf_math_squeeze(&aux3, &out), &aux3);
  BENCH_STATUS("cf_math_unsqueeze", BENCH_ITERS_TENSOR, cf_math_unsqueeze(&aux3, &a, 0U), &aux3);
  BENCH_STATUS("cf_math_expand", BENCH_ITERS_TENSOR, cf_math_expand(&aux3, &c, expand_dim, 2U), &aux3);
  concat_inputs[0] = &c;
  concat_inputs[1] = &c;
  BENCH_STATUS("cf_math_concat", BENCH_ITERS_TENSOR, cf_math_concat(&aux3, concat_inputs, 2U, 0U, CF_NULL), &aux3);
  BENCH_STATUS("cf_math_split", BENCH_ITERS_TENSOR, cf_math_split(split_outs, 2U, &aux3, 0U), &split_outs[0]);
  BENCH_STATUS("cf_math_slice", BENCH_ITERS_TENSOR, cf_math_slice(&aux3, &a, slice_start, slice_len), &aux3);
  BENCH_STATUS("cf_math_pad", BENCH_ITERS_TENSOR, cf_math_pad(&aux3, &a, pad_before, pad_after, CF_NULL), &aux3);
  BENCH_STATUS("cf_math_flatten", BENCH_ITERS_TENSOR, cf_math_flatten(&aux3, &a, 0U, 1U), &aux3);
  printf("%-34s\n", "cf_math_print");
  cf_math_print(&a);

  BENCH_STATUS("cf_math_alloc conv_x", 1U, cf_math_alloc(&conv_x, dim133, 4, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL), &conv_x);
  BENCH_STATUS("cf_math_alloc conv_w", 1U, cf_math_alloc(&conv_w, dim1122, 4, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL), &conv_w);
  BENCH_STATUS("cf_math_alloc conv_b", 1U, cf_math_alloc(&conv_b, dim2, 1, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL), &conv_b);
  bench_fill_pattern(&conv_x, 1.0);
  bench_fill_pattern(&conv_w, 0.25);
  bench_fill_pattern(&conv_b, 0.0);
  BENCH_STATUS("cf_math_conv2d_fwd", BENCH_ITERS_HEAVY, cf_math_conv2d_fwd(&conv_out, &conv_x, &conv_w, &conv_b, conv, CF_NULL), &conv_out);
  BENCH_STATUS("cf_math_conv2d_bwd_data", BENCH_ITERS_HEAVY, cf_math_conv2d_bwd_data(&conv_x, &conv_out, &conv_w, conv, CF_NULL), &conv_x);
  BENCH_STATUS("cf_math_conv2d_bwd_filter", BENCH_ITERS_HEAVY, cf_math_conv2d_bwd_filter(&conv_w, &conv_out, &conv_x, conv, CF_NULL), &conv_w);
  BENCH_STATUS("cf_math_conv2d_bwd_bias", BENCH_ITERS_HEAVY, cf_math_conv2d_bwd_bias(&conv_b, &conv_out, CF_NULL), &conv_b);
  BENCH_STATUS("cf_math_conv2d_depthwise_fwd", BENCH_ITERS_HEAVY, cf_math_conv2d_depthwise_fwd(&conv_out, &conv_x, &conv_w, conv, CF_NULL), &conv_out);
  BENCH_STATUS("cf_math_conv2d_dilated_fwd", BENCH_ITERS_HEAVY, cf_math_conv2d_dilated_fwd(&conv_out, &conv_x, &conv_w, &conv_b, conv, CF_NULL), &conv_out);
  BENCH_STATUS("cf_math_conv2d_transpose_fwd", BENCH_ITERS_HEAVY, cf_math_conv2d_transpose_fwd(&conv_x, &conv_out, &conv_w, conv, CF_NULL), &conv_x);
  BENCH_STATUS("cf_math_conv1d_fwd", BENCH_ITERS_HEAVY, cf_math_conv1d_fwd(&conv_out, &conv_x, &conv_w, &conv_b, conv, CF_NULL), &conv_out);
  BENCH_STATUS("cf_math_alloc conv3_x", 1U, cf_math_alloc(&conv3_x, dim11133, 5, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL), &conv3_x);
  BENCH_STATUS("cf_math_alloc conv3_w", 1U, cf_math_alloc(&conv3_w, dim11122, 5, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL), &conv3_w);
  bench_fill_pattern(&conv3_x, 1.0);
  bench_fill_pattern(&conv3_w, 0.25);
  BENCH_STATUS("cf_math_conv3d_fwd", BENCH_ITERS_HEAVY, cf_math_conv3d_fwd(&conv_out, &conv3_x, &conv3_w, &conv_b, conv, CF_NULL), &conv_out);

  BENCH_STATUS("cf_math_alloc gamma", 1U, cf_math_alloc(&gamma, dim2, 1, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL), &gamma);
  BENCH_STATUS("cf_math_alloc beta", 1U, cf_math_alloc(&beta, dim2, 1, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL), &beta);
  cf_math_ones(&gamma, CF_NULL);
  cf_math_zeros(&beta, CF_NULL);
  BENCH_STATUS("cf_math_bn_fwd_train", BENCH_ITERS_HEAVY, cf_math_bn_fwd_train(&aux, &aux2, &aux3, &a, &gamma, &beta, 1e-5, CF_NULL), &aux);
  BENCH_STATUS("cf_math_bn_fwd_infer", BENCH_ITERS_HEAVY, cf_math_bn_fwd_infer(&aux, &a, &gamma, &beta, &aux2, &aux3, 1e-5, CF_NULL), &aux);
  BENCH_STATUS("cf_math_bn_bwd", BENCH_ITERS_HEAVY, cf_math_bn_bwd(&aux, &aux2, &aux3, &a, &a, &gamma, &aux2, &aux3, CF_NULL), &aux);
  BENCH_STATUS("cf_math_ln_fwd", BENCH_ITERS_HEAVY, cf_math_ln_fwd(&aux, &a, &gamma, &beta, 1e-5, CF_NULL), &aux);
  BENCH_STATUS("cf_math_ln_bwd", BENCH_ITERS_HEAVY, cf_math_ln_bwd(&aux, &aux2, &aux3, &a, &a, &gamma, 1e-5, CF_NULL), &aux);
  BENCH_STATUS("cf_math_in_fwd", BENCH_ITERS_HEAVY, cf_math_in_fwd(&aux, &a, &gamma, &beta, 1e-5, CF_NULL), &aux);
  BENCH_STATUS("cf_math_gn_fwd", BENCH_ITERS_HEAVY, cf_math_gn_fwd(&aux, &a, &gamma, &beta, 1U, 1e-5, CF_NULL), &aux);
  BENCH_STATUS("cf_math_rms_norm_fwd", BENCH_ITERS_HEAVY, cf_math_rms_norm_fwd(&aux, &a, &gamma, 1e-5, CF_NULL), &aux);
  BENCH_STATUS("cf_math_rms_norm_bwd", BENCH_ITERS_HEAVY, cf_math_rms_norm_bwd(&aux, &aux2, &a, &a, &gamma, 1e-5, CF_NULL), &aux);

  bench_free_sparse(&csr);
  free(dropout.reserve);
  bench_free_tensor(&split_outs[0]);
  bench_free_tensor(&split_outs[1]);
  bench_free_tensor(&sparse_vec);
  bench_free_tensor(&labels);
  bench_free_tensor(&idx);
  bench_free_tensor(&loss);
  bench_free_tensor(&gamma);
  bench_free_tensor(&beta);
  bench_free_tensor(&conv3_w);
  bench_free_tensor(&conv3_x);
  bench_free_tensor(&conv_out);
  bench_free_tensor(&conv_b);
  bench_free_tensor(&conv_w);
  bench_free_tensor(&conv_x);
  bench_free_tensor(&aux3);
  bench_free_tensor(&aux2);
  bench_free_tensor(&aux);
  bench_free_tensor(&out);
  bench_free_tensor(&c);
  bench_free_tensor(&b);
  bench_free_tensor(&a);
}

static void bench_cpu_stress(void)
{
  cf_usize vec_dim[CF_MATH_HIGHEST_RANK] = {262144U};
  cf_usize mat_dim[CF_MATH_HIGHEST_RANK] = {128U, 128U};
  cf_math x = {0};
  cf_math y = {0};
  cf_math a = {0};
  cf_math b = {0};
  cf_math out = {0};

  printf("\n== CPU stress benchmark ==\n");
  BENCH_STATUS("cpu stress alloc x", 1U, cf_math_alloc(&x, vec_dim, 1, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL), &x);
  BENCH_STATUS("cpu stress alloc y", 1U, cf_math_alloc(&y, vec_dim, 1, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL), &y);
  BENCH_STATUS("cpu stress alloc a", 1U, cf_math_alloc(&a, mat_dim, 2, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL), &a);
  BENCH_STATUS("cpu stress alloc b", 1U, cf_math_alloc(&b, mat_dim, 2, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL), &b);
  bench_fill_pattern(&x, 1.0);
  bench_fill_pattern(&y, 2.0);
  bench_fill_pattern(&a, 0.01);
  bench_fill_pattern(&b, 0.02);

  BENCH_STATUS("cpu stress add 262k", BENCH_ITERS_STRESS, cf_math_add(&out, &x, &y, CF_NULL), &out);
  bench_free_tensor(&out);
  BENCH_STATUS("cpu stress dot 262k", BENCH_ITERS_STRESS, cf_math_dot(&out, &x, &y, CF_NULL), &out);
  bench_free_tensor(&out);
  BENCH_STATUS("cpu stress matmul 128x128", BENCH_ITERS_STRESS, cf_math_matmul(&out, &a, &b, CF_NULL), &out);

  bench_free_tensor(&out);
  bench_free_tensor(&b);
  bench_free_tensor(&a);
  bench_free_tensor(&y);
  bench_free_tensor(&x);
}

static void bench_gpu_stress(void)
{
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  cf_math_cuda_context ctx = {0};
  cf_status status = cf_math_context_init(&ctx, 0);
  cf_usize vec_dim[CF_MATH_HIGHEST_RANK] = {262144U};
  cf_math host = {0};
  cf_math device = {0};
  cf_math back = {0};

  printf("\n== GPU stress benchmark ==\n");
  if(status != CF_OK)
  {
    printf("GPU context unavailable: %s\n", cf_status_as_char(status));
    return;
  }

  BENCH_STATUS("gpu host alloc", 1U, cf_math_alloc(&host, vec_dim, 1, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL), &host);
  bench_fill_pattern(&host, 1.0);
  BENCH_STATUS("gpu to_device 262k", BENCH_ITERS_STRESS, cf_math_to_device(&device, &host, 0, &ctx), &device);
  BENCH_STATUS("gpu to_host 262k", BENCH_ITERS_STRESS, cf_math_to_host(&back, &device, &ctx), &back);
  BENCH_STATUS("gpu add attempt", BENCH_ITERS_STRESS, cf_math_add(&device, &device, &device, &ctx), &device);

  bench_free_tensor(&back);
  cf_math_free(&device, &ctx);
  bench_free_tensor(&host);
  cf_math_context_destroy(&ctx);
#else
  printf("\n== GPU stress benchmark ==\n");
  printf("CUDA runtime headers are not available in this build; GPU stress skipped.\n");
#endif
}

int main(void)
{
  printf("cf_math benchmark test - run manually when timing numbers are needed.\n");
  bench_cpu_surface();
  bench_cpu_stress();
  bench_gpu_stress();
  return 0;
}
