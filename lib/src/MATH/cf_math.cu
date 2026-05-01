/*
 * CF Framework
 * Copyright (C) 2026 Orion
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include "MATH/cf_math.h"
#include "MATH/cf_math_storage.h"
#include "RUNTIME/cf_random.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(CF_MATH_USE_OPENBLAS)
#include <cblas.h>
#endif

#if defined(CF_CUDA_AVAILABLE)
#include <cub/cub.cuh>
#endif

static cf_status cf_math_resolve_ptr(const cf_math *x, void **ptr);
static cf_status cf_math_copy_bytes(const cf_math *x, void *dst, const void *src, cf_usize bytes);

#if defined(__cplusplus)
#define CF_MATH_RESTRICT __restrict__
#else
#define CF_MATH_RESTRICT restrict
#endif

#define CF_MATH_CPU_PARALLEL_THRESHOLD ((cf_usize)16384U)

cf_u8 cf_math_g8_mul_mod(cf_u8 p, cf_u8 q)
{
  cf_u8 res = 0;
  do
  {
    if(q & 0x01) res ^= p;
    if(p & 0x80) p = (p << 1) ^ 0x1B;
    else p <<= 1;
  } while(q >>= 1);
  return res;
}

cf_u8 cf_math_rotl8(cf_u8 x, cf_u8 n)
{
  n &= 7;
  return n == 0 ? x : (cf_u8)((x << n) | (x >> (8 - n)));
}

cf_u8 cf_math_rotr8(cf_u8 x, cf_u8 n)
{
  n &= 7;
  return n == 0 ? x : (cf_u8)((x >> n) | (x << (8 - n)));
}

cf_u32 cf_math_rotl32(cf_u32 x, cf_u8 n)
{
  n &= 31;
  return n == 0 ? x : (x << n) | (x >> (32 - n));
}

cf_u32 cf_math_rotr32(cf_u32 x, cf_u8 n)
{
  n &= 31;
  return n == 0 ? x : (x >> n) | (x << (32 - n));
}

cf_usize cf_math_min_usize(cf_usize a, cf_usize b)
{
  return a <= b ? a : b;
}

cf_usize cf_math_max_usize(cf_usize a, cf_usize b)
{
  return a >= b ? a : b;
}

static cf_usize cf_math_dtype_size(cf_math_dtype dtype)
{
  switch (dtype)
  {
    case CF_MATH_DTYPE_BOOL: return sizeof(cf_bool);
    case CF_MATH_DTYPE_I8: return sizeof(cf_i8);
    case CF_MATH_DTYPE_U8: return sizeof(cf_u8);
    case CF_MATH_DTYPE_I32: return sizeof(cf_i32);
    case CF_MATH_DTYPE_F64: return sizeof(double);
    case CF_MATH_DTYPE_F32: return sizeof(float);
    case CF_MATH_DTYPE_F16: return sizeof(cf_u16);
    case CF_MATH_DTYPE_BF16: return sizeof(cf_u16);
    case CF_MATH_DTYPE_FP8E4M3: return sizeof(cf_u8);
    case CF_MATH_DTYPE_FP8E5M2: return sizeof(cf_u8);
  }
  return 0;
}

#if defined(CF_CUDA_AVAILABLE)
#define CF_MATH_CUDA_BINARY_KERNEL(name, type) \
static __global__ void name(cf_math_op_kind op, type *op1, const type *op2, cf_usize len) \
{ \
  cf_usize i = (cf_usize)blockIdx.x * (cf_usize)blockDim.x + (cf_usize)threadIdx.x; \
  if(i >= len) return; \
  switch(op) \
  { \
    case CF_MATH_OP_ADD: op1[i] = op1[i] + op2[i]; break; \
    case CF_MATH_OP_SUB: op1[i] = op1[i] - op2[i]; break; \
    case CF_MATH_OP_MUL: op1[i] = op1[i] * op2[i]; break; \
    case CF_MATH_OP_DIV: op1[i] = op1[i] / op2[i]; break; \
    default: break; \
  } \
}

#define CF_MATH_CUDA_UNARY_KERNEL(name, type, exp_fn, log_fn, sqrt_fn, tanh_fn) \
static __global__ void name(cf_math_op_kind op, type *x, cf_usize len) \
{ \
  cf_usize i = (cf_usize)blockIdx.x * (cf_usize)blockDim.x + (cf_usize)threadIdx.x; \
  type v = (type)0; \
  if(i >= len) return; \
  v = x[i]; \
  switch(op) \
  { \
    case CF_MATH_OP_NEG: x[i] = -v; break; \
    case CF_MATH_OP_RELU: x[i] = v > (type)0 ? v : (type)0; break; \
    case CF_MATH_OP_GELU: x[i] = (type)0.5 * v * ((type)1 + tanh_fn((type)0.7978845608028654 * (v + (type)0.044715 * v * v * v))); break; \
    case CF_MATH_OP_EXP: x[i] = exp_fn(v); break; \
    case CF_MATH_OP_LOG: x[i] = log_fn(v); break; \
    case CF_MATH_OP_SQRT: x[i] = sqrt_fn(v); break; \
    case CF_MATH_OP_SIGMOID: x[i] = (type)1 / ((type)1 + exp_fn(-v)); break; \
    case CF_MATH_OP_TANH: x[i] = tanh_fn(v); break; \
    default: break; \
  } \
}

#define CF_MATH_CUDA_SCALAR_KERNEL(name, type) \
static __global__ void name(cf_math_op_kind op, type *x, type scalar, cf_usize len) \
{ \
  cf_usize i = (cf_usize)blockIdx.x * (cf_usize)blockDim.x + (cf_usize)threadIdx.x; \
  if(i >= len) return; \
  switch(op) \
  { \
    case CF_MATH_OP_ADD: x[i] = x[i] + scalar; break; \
    case CF_MATH_OP_SUB: x[i] = x[i] - scalar; break; \
    case CF_MATH_OP_MUL: x[i] = x[i] * scalar; break; \
    case CF_MATH_OP_DIV: x[i] = x[i] / scalar; break; \
    default: break; \
  } \
}

CF_MATH_CUDA_BINARY_KERNEL(cf_math_op_kernel_f32, float)
CF_MATH_CUDA_BINARY_KERNEL(cf_math_op_kernel_f64, double)
CF_MATH_CUDA_BINARY_KERNEL(cf_math_op_kernel_i32, cf_i32)
CF_MATH_CUDA_UNARY_KERNEL(cf_math_unary_kernel_f32, float, expf, logf, sqrtf, tanhf)
CF_MATH_CUDA_UNARY_KERNEL(cf_math_unary_kernel_f64, double, exp, log, sqrt, tanh)
CF_MATH_CUDA_SCALAR_KERNEL(cf_math_scalar_kernel_f32, float)
CF_MATH_CUDA_SCALAR_KERNEL(cf_math_scalar_kernel_f64, double)
CF_MATH_CUDA_SCALAR_KERNEL(cf_math_scalar_kernel_i32, cf_i32)

template <typename T>
static __global__ void cf_math_mean_finalize_kernel(T *out, cf_usize len)
{
  if(len != 0) out[0] = (T)(out[0] / (T)len);
}

#endif

cf_status cf_math_metadata_init(cf_math_metadata *metadata, cf_usize dim[CF_MATH_MAX_RANK], cf_usize rank, cf_math_shape shape, cf_math_layout layout)
{
  cf_usize len = 1;

  if(metadata == CF_NULL) return CF_ERR_NULL;
  if(rank > CF_MATH_MAX_RANK) return CF_ERR_INVALID;
  if(dim == CF_NULL && rank != 0) return CF_ERR_INVALID;

  memset(metadata, 0, sizeof(*metadata));
  if(rank != 0) memcpy(metadata->dim, dim, rank * sizeof(cf_usize));
  metadata->rank = rank;
  metadata->shape = shape;
  metadata->layout = layout;

  if(rank == 0)
  {
    metadata->len = 1;
    return CF_OK;
  }

  for(cf_usize i = 0; i < rank; ++i)
  {
    if(dim[i] != 0 && len > (cf_usize)-1 / dim[i]) return CF_ERR_OVERFLOW;
    len *= dim[i];
  }

  if(layout == CF_MATH_LAYOUT_COL_MAJOR)
  {
    metadata->strides[0] = 1;
    for(cf_usize i = 1; i < rank; ++i)
    {
      if(metadata->strides[i - 1] != 0 && dim[i - 1] > (cf_usize)-1 / metadata->strides[i - 1])
        return CF_ERR_OVERFLOW;
      metadata->strides[i] = metadata->strides[i - 1] * dim[i - 1];
    }
  }
  else
  {
    metadata->strides[rank - 1] = 1;
    for(cf_usize i = rank - 1; i > 0; --i)
    {
      if(metadata->strides[i] != 0 && dim[i] > (cf_usize)-1 / metadata->strides[i])
        return CF_ERR_OVERFLOW;
      metadata->strides[i - 1] = metadata->strides[i] * dim[i];
    }
  }

  metadata->len = len;
  return CF_OK;
}

static cf_bool cf_math_is_elementwise_binary_op(cf_math_op_kind op)
{
  return op == CF_MATH_OP_ADD || op == CF_MATH_OP_SUB || op == CF_MATH_OP_MUL || op == CF_MATH_OP_DIV;
}

static cf_bool cf_math_is_unary_op(cf_math_op_kind op)
{
  return op == CF_MATH_OP_NEG || op == CF_MATH_OP_RELU || op == CF_MATH_OP_GELU || op == CF_MATH_OP_EXP || op == CF_MATH_OP_LOG || op == CF_MATH_OP_SQRT || op == CF_MATH_OP_SIGMOID || op == CF_MATH_OP_TANH;
}

static cf_bool cf_math_is_supported_binary_dtype(cf_math_dtype dtype)
{
  return dtype == CF_MATH_DTYPE_F32 || dtype == CF_MATH_DTYPE_F64 || dtype == CF_MATH_DTYPE_I32;
}

static cf_bool cf_math_is_supported_float_dtype(cf_math_dtype dtype)
{
  return dtype == CF_MATH_DTYPE_F32 || dtype == CF_MATH_DTYPE_F64;
}

static cf_bool cf_math_is_compact_row_major(const cf_math_metadata *metadata)
{
  cf_usize expected = 1;

  if(metadata == CF_NULL) return CF_FALSE;
  if(metadata->layout != CF_MATH_LAYOUT_ROW_MAJOR) return CF_FALSE;
  if(metadata->rank == 0) return CF_TRUE;

  for(cf_usize i = metadata->rank; i > 0; --i)
  {
    cf_usize index = i - 1U;
    if(metadata->strides[index] != expected) return CF_FALSE;
    if(metadata->dim[index] != 0 && expected > (cf_usize)-1 / metadata->dim[index])
      return CF_FALSE;
    expected *= metadata->dim[index];
  }
  return expected == metadata->len;
}

static cf_bool cf_math_is_bound(const cf_math *x)
{
  return x != CF_NULL && x->handler != CF_NULL && x->metadata != CF_NULL;
}

static cf_status cf_math_check_same_view_shape(const cf_math *a, const cf_math *b)
{
  if(a == CF_NULL || b == CF_NULL) return CF_ERR_NULL;
  if(cf_math_is_bound(a) == CF_FALSE || cf_math_is_bound(b) == CF_FALSE) return CF_ERR_STATE;
  if(a->handler->storage.device != b->handler->storage.device) return CF_ERR_INVALID;
  if(a->handler->storage.dtype != b->handler->storage.dtype) return CF_ERR_INVALID;
  if(a->metadata->len != b->metadata->len) return CF_ERR_INVALID;
  return CF_OK;
}

#if defined(CF_MATH_USE_OPENMP)
#define CF_MATH_OMP_FOR _Pragma("omp parallel for simd schedule(static) if(len >= CF_MATH_CPU_PARALLEL_THRESHOLD)")
#define CF_MATH_OMP_REDUCTION_SUM _Pragma("omp parallel for simd reduction(+:sum) schedule(static) if(len >= CF_MATH_CPU_PARALLEL_THRESHOLD)")
#define CF_MATH_OMP_REDUCTION_ISUM _Pragma("omp parallel for simd reduction(+:isum) schedule(static) if(len >= CF_MATH_CPU_PARALLEL_THRESHOLD)")
#else
#define CF_MATH_OMP_FOR
#define CF_MATH_OMP_REDUCTION_SUM
#define CF_MATH_OMP_REDUCTION_ISUM
#endif

static cf_status cf_math_check_blas_dim(cf_usize value)
{
  return value <= (cf_usize)INT_MAX ? CF_OK : CF_ERR_BOUNDS;
}

static cf_status cf_math_check_blas3(cf_usize m, cf_usize k, cf_usize n)
{
  if(cf_math_check_blas_dim(m) != CF_OK) return CF_ERR_BOUNDS;
  if(cf_math_check_blas_dim(k) != CF_OK) return CF_ERR_BOUNDS;
  if(cf_math_check_blas_dim(n) != CF_OK) return CF_ERR_BOUNDS;
  return CF_OK;
}

static cf_status cf_math_op_cpu(cf_math_op_kind op, cf_math_dtype dtype, void *op1_ptr, const void *op2_ptr, cf_usize len)
{
  switch(dtype)
  {
    case CF_MATH_DTYPE_F32:
      {
        float *CF_MATH_RESTRICT a = (float *)op1_ptr;
        const float *CF_MATH_RESTRICT b = (const float *)op2_ptr;
        CF_MATH_OMP_FOR
        for(cf_usize i = 0; i < len; ++i)
        {
          switch(op)
          {
            case CF_MATH_OP_ADD: a[i] = a[i] + b[i]; break;
            case CF_MATH_OP_SUB: a[i] = a[i] - b[i]; break;
            case CF_MATH_OP_MUL: a[i] = a[i] * b[i]; break;
            case CF_MATH_OP_DIV: a[i] = a[i] / b[i]; break;
            default: break;
          }
        }
        return CF_OK;
      }
    case CF_MATH_DTYPE_F64:
      {
        double *CF_MATH_RESTRICT a = (double *)op1_ptr;
        const double *CF_MATH_RESTRICT b = (const double *)op2_ptr;
        CF_MATH_OMP_FOR
        for(cf_usize i = 0; i < len; ++i)
        {
          switch(op)
          {
            case CF_MATH_OP_ADD: a[i] = a[i] + b[i]; break;
            case CF_MATH_OP_SUB: a[i] = a[i] - b[i]; break;
            case CF_MATH_OP_MUL: a[i] = a[i] * b[i]; break;
            case CF_MATH_OP_DIV: a[i] = a[i] / b[i]; break;
            default: break;
          }
        }
        return CF_OK;
      }
    case CF_MATH_DTYPE_I32:
      {
        cf_i32 *CF_MATH_RESTRICT a = (cf_i32 *)op1_ptr;
        const cf_i32 *CF_MATH_RESTRICT b = (const cf_i32 *)op2_ptr;
        CF_MATH_OMP_FOR
        for(cf_usize i = 0; i < len; ++i)
        {
          switch(op)
          {
            case CF_MATH_OP_ADD: a[i] = a[i] + b[i]; break;
            case CF_MATH_OP_SUB: a[i] = a[i] - b[i]; break;
            case CF_MATH_OP_MUL: a[i] = a[i] * b[i]; break;
            case CF_MATH_OP_DIV: a[i] = a[i] / b[i]; break;
            default: break;
          }
        }
        return CF_OK;
      }
    default: return CF_ERR_UNSUPPORTED;
  }
}

static cf_status cf_math_unary_cpu(cf_math_op_kind op, cf_math_dtype dtype, void *x_ptr, cf_usize len)
{
  switch(dtype)
  {
    case CF_MATH_DTYPE_F32:
    {
      float *x = (float *)x_ptr;
      CF_MATH_OMP_FOR
      for(cf_usize i = 0; i < len; ++i)
      {
        float v = x[i];
        switch(op)
        {
          case CF_MATH_OP_NEG: x[i] = -v; break;
          case CF_MATH_OP_RELU: x[i] = v > 0.0f ? v : 0.0f; break;
          case CF_MATH_OP_GELU: x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v))); break;
          case CF_MATH_OP_EXP: x[i] = expf(v); break;
          case CF_MATH_OP_LOG: x[i] = logf(v); break;
          case CF_MATH_OP_SQRT: x[i] = sqrtf(v); break;
          case CF_MATH_OP_SIGMOID: x[i] = 1.0f / (1.0f + expf(-v)); break;
          case CF_MATH_OP_TANH: x[i] = tanhf(v); break;
          default: break;
        }
      }
      return CF_OK;
    }
    case CF_MATH_DTYPE_F64:
    {
      double *x = (double *)x_ptr;
      CF_MATH_OMP_FOR
      for(cf_usize i = 0; i < len; ++i)
      {
        double v = x[i];
        switch(op)
        {
          case CF_MATH_OP_NEG: x[i] = -v; break;
          case CF_MATH_OP_RELU: x[i] = v > 0.0 ? v : 0.0; break;
          case CF_MATH_OP_GELU: x[i] = 0.5 * v * (1.0 + tanh(0.7978845608028654 * (v + 0.044715 * v * v * v))); break;
          case CF_MATH_OP_EXP: x[i] = exp(v); break;
          case CF_MATH_OP_LOG: x[i] = log(v); break;
          case CF_MATH_OP_SQRT: x[i] = sqrt(v); break;
          case CF_MATH_OP_SIGMOID: x[i] = 1.0 / (1.0 + exp(-v)); break;
          case CF_MATH_OP_TANH: x[i] = tanh(v); break;
          default: break;
        }
      }
      return CF_OK;
    }
    default: return CF_ERR_UNSUPPORTED;
  }
}

static cf_status cf_math_scalar_cpu(cf_math_op_kind op, cf_math_dtype dtype, void *x_ptr, cf_usize len, double scalar)
{
  switch(dtype)
  {
    case CF_MATH_DTYPE_F32:
    {
      float *x = (float *)x_ptr;
      float s = (float)scalar;
      CF_MATH_OMP_FOR
      for(cf_usize i = 0; i < len; ++i)
      {
        switch(op)
        {
          case CF_MATH_OP_ADD: x[i] = x[i] + s; break;
          case CF_MATH_OP_SUB: x[i] = x[i] - s; break;
          case CF_MATH_OP_MUL: x[i] = x[i] * s; break;
          case CF_MATH_OP_DIV: x[i] = x[i] / s; break;
          default: break;
        }
      }
      return CF_OK;
    }
    case CF_MATH_DTYPE_F64:
    {
      double *x = (double *)x_ptr;
      CF_MATH_OMP_FOR
      for(cf_usize i = 0; i < len; ++i)
      {
        switch(op)
        {
          case CF_MATH_OP_ADD: x[i] = x[i] + scalar; break;
          case CF_MATH_OP_SUB: x[i] = x[i] - scalar; break;
          case CF_MATH_OP_MUL: x[i] = x[i] * scalar; break;
          case CF_MATH_OP_DIV: x[i] = x[i] / scalar; break;
          default: break;
        }
      }
      return CF_OK;
    }
    case CF_MATH_DTYPE_I32:
    {
      cf_i32 *x = (cf_i32 *)x_ptr;
      cf_i32 s = (cf_i32)scalar;
      CF_MATH_OMP_FOR
      for(cf_usize i = 0; i < len; ++i)
      {
        switch(op)
        {
          case CF_MATH_OP_ADD: x[i] = x[i] + s; break;
          case CF_MATH_OP_SUB: x[i] = x[i] - s; break;
          case CF_MATH_OP_MUL: x[i] = x[i] * s; break;
          case CF_MATH_OP_DIV: x[i] = x[i] / s; break;
          default: break;
        }
      }
      return CF_OK;
    }
    default: return CF_ERR_UNSUPPORTED;
  }
}

static cf_status cf_math_reduce_cpu(cf_math_op_kind op, cf_math_dtype dtype, void *out_ptr, const void *x_ptr, cf_usize len)
{
  if(op != CF_MATH_OP_SUM && op != CF_MATH_OP_MEAN) return CF_ERR_UNSUPPORTED;

  switch(dtype)
  {
    case CF_MATH_DTYPE_F32:
    {
      const float *x = (const float *)x_ptr;
      double sum = 0.0;
      CF_MATH_OMP_REDUCTION_SUM
      for(cf_usize i = 0; i < len; ++i) sum += (double)x[i];
      if(op == CF_MATH_OP_MEAN && len != 0) sum /= (double)len;
      ((float *)out_ptr)[0] = (float)sum;
      return CF_OK;
    }
    case CF_MATH_DTYPE_F64:
    {
      const double *x = (const double *)x_ptr;
      double sum = 0.0;
      CF_MATH_OMP_REDUCTION_SUM
      for(cf_usize i = 0; i < len; ++i) sum += x[i];
      if(op == CF_MATH_OP_MEAN && len != 0) sum /= (double)len;
      ((double *)out_ptr)[0] = sum;
      return CF_OK;
    }
    case CF_MATH_DTYPE_I32:
    {
      const cf_i32 *x = (const cf_i32 *)x_ptr;
      cf_i64 isum = 0;
      CF_MATH_OMP_REDUCTION_ISUM
      for(cf_usize i = 0; i < len; ++i) isum += (cf_i64)x[i];
      if(op == CF_MATH_OP_MEAN && len != 0) isum /= (cf_i64)len;
      ((cf_i32 *)out_ptr)[0] = (cf_i32)isum;
      return CF_OK;
    }
    default:
      return CF_ERR_UNSUPPORTED;
  }
}

static cf_status cf_math_matmul_cpu(cf_math_dtype dtype, void *out_ptr, const void *a_ptr, const void *b_ptr, cf_usize m, cf_usize k, cf_usize n)
{
  cf_status status = cf_math_check_blas3(m, k, n);
  if(status != CF_OK) return status;

  switch(dtype)
  {
    case CF_MATH_DTYPE_F32:
#if defined(CF_MATH_USE_OPENBLAS)
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)m, (int)n, (int)k, 1.0f, (const float *)a_ptr, (int)k, (const float *)b_ptr, (int)n, 0.0f, (float *)out_ptr, (int)n);
#else
      {
        float *out = (float *)out_ptr;
        const float *a = (const float *)a_ptr;
        const float *b = (const float *)b_ptr;
        cf_usize len = m;
        CF_MATH_OMP_FOR
        for(cf_usize row = 0; row < m; ++row)
        {
          for(cf_usize col = 0; col < n; ++col)
          {
            float sum = 0.0f;
            for(cf_usize inner = 0; inner < k; ++inner)
              sum += a[row * k + inner] * b[inner * n + col];
            out[row * n + col] = sum;
          }
        }
      }
#endif
      return CF_OK;
    case CF_MATH_DTYPE_F64:
#if defined(CF_MATH_USE_OPENBLAS)
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)m, (int)n, (int)k, 1.0, (const double *)a_ptr, (int)k, (const double *)b_ptr, (int)n, 0.0, (double *)out_ptr, (int)n);
#else
      {
        double *out = (double *)out_ptr;
        const double *a = (const double *)a_ptr;
        const double *b = (const double *)b_ptr;
        cf_usize len = m;
        CF_MATH_OMP_FOR
        for(cf_usize row = 0; row < m; ++row)
        {
          for(cf_usize col = 0; col < n; ++col)
          {
            double sum = 0.0;
            for(cf_usize inner = 0; inner < k; ++inner)
              sum += a[row * k + inner] * b[inner * n + col];
            out[row * n + col] = sum;
          }
        }
      }
#endif
      return CF_OK;
    default:
      return CF_ERR_UNSUPPORTED;
  }
}

static cf_status cf_math_matvec_cpu(cf_math_dtype dtype, void *out_ptr, const void *a_ptr, const void *x_ptr, cf_usize m, cf_usize n)
{
  cf_status status = cf_math_check_blas3(m, n, 1U);
  if(status != CF_OK) return status;

  switch(dtype)
  {
    case CF_MATH_DTYPE_F32:
#if defined(CF_MATH_USE_OPENBLAS)
      cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)m, (int)n, 1.0f, (const float *)a_ptr, (int)n, (const float *)x_ptr, 1, 0.0f, (float *)out_ptr, 1);
#else
      {
        float *out = (float *)out_ptr;
        const float *a = (const float *)a_ptr;
        const float *x = (const float *)x_ptr;
        cf_usize len = m;
        CF_MATH_OMP_FOR
        for(cf_usize row = 0; row < m; ++row)
        {
          float sum = 0.0f;
          for(cf_usize col = 0; col < n; ++col) sum += a[row * n + col] * x[col];
          out[row] = sum;
        }
      }
#endif
      return CF_OK;
    case CF_MATH_DTYPE_F64:
#if defined(CF_MATH_USE_OPENBLAS)
      cblas_dgemv(CblasRowMajor, CblasNoTrans, (int)m, (int)n, 1.0, (const double *)a_ptr, (int)n, (const double *)x_ptr, 1, 0.0, (double *)out_ptr, 1);
#else
      {
        double *out = (double *)out_ptr;
        const double *a = (const double *)a_ptr;
        const double *x = (const double *)x_ptr;
        cf_usize len = m;
        CF_MATH_OMP_FOR
        for(cf_usize row = 0; row < m; ++row)
        {
          double sum = 0.0;
          for(cf_usize col = 0; col < n; ++col) sum += a[row * n + col] * x[col];
          out[row] = sum;
        }
      }
#endif
      return CF_OK;
    default:
      return CF_ERR_UNSUPPORTED;
  }
}

static cf_status cf_math_dot_cpu(cf_math_dtype dtype, void *out_ptr, const void *a_ptr, const void *b_ptr, cf_usize len)
{
  cf_status status = cf_math_check_blas_dim(len);
  if(status != CF_OK) return status;

  switch(dtype)
  {
    case CF_MATH_DTYPE_F32:
#if defined(CF_MATH_USE_OPENBLAS)
      ((float *)out_ptr)[0] = cblas_sdot((int)len, (const float *)a_ptr, 1, (const float *)b_ptr, 1);
#else
      {
        const float *a = (const float *)a_ptr;
        const float *b = (const float *)b_ptr;
        double sum = 0.0;
        CF_MATH_OMP_REDUCTION_SUM
        for(cf_usize i = 0; i < len; ++i) sum += (double)a[i] * (double)b[i];
        ((float *)out_ptr)[0] = (float)sum;
      }
#endif
      return CF_OK;
    case CF_MATH_DTYPE_F64:
#if defined(CF_MATH_USE_OPENBLAS)
      ((double *)out_ptr)[0] = cblas_ddot((int)len, (const double *)a_ptr, 1, (const double *)b_ptr, 1);
#else
      {
        const double *a = (const double *)a_ptr;
        const double *b = (const double *)b_ptr;
        double sum = 0.0;
        CF_MATH_OMP_REDUCTION_SUM
        for(cf_usize i = 0; i < len; ++i) sum += a[i] * b[i];
        ((double *)out_ptr)[0] = sum;
      }
#endif
      return CF_OK;
    default:
      return CF_ERR_UNSUPPORTED;
  }
}

static cf_status cf_math_batched_matmul_cpu(cf_math_dtype dtype, void *out_ptr, const void *a_ptr, const void *b_ptr, cf_usize batch, cf_usize m, cf_usize k, cf_usize n)
{
  cf_usize a_stride = 0;
  cf_usize b_stride = 0;
  cf_usize out_stride = 0;
  cf_status status = cf_math_check_blas3(m, k, n);
  if(status != CF_OK) return status;
  if(cf_math_check_blas_dim(batch) != CF_OK) return CF_ERR_BOUNDS;
  if(m != 0 && k > (cf_usize)-1 / m) return CF_ERR_OVERFLOW;
  if(k != 0 && n > (cf_usize)-1 / k) return CF_ERR_OVERFLOW;
  if(m != 0 && n > (cf_usize)-1 / m) return CF_ERR_OVERFLOW;

  a_stride = m * k;
  b_stride = k * n;
  out_stride = m * n;

  switch(dtype)
  {
    case CF_MATH_DTYPE_F32:
      {
        float *out = (float *)out_ptr;
        const float *a = (const float *)a_ptr;
        const float *b = (const float *)b_ptr;
        for(cf_usize i = 0; i < batch; ++i)
        {
          status = cf_math_matmul_cpu(dtype, out + i * out_stride, a + i * a_stride, b + i * b_stride, m, k, n);
          if(status != CF_OK) return status;
        }
      }
      return CF_OK;
    case CF_MATH_DTYPE_F64:
      {
        double *out = (double *)out_ptr;
        const double *a = (const double *)a_ptr;
        const double *b = (const double *)b_ptr;
        for(cf_usize i = 0; i < batch; ++i)
        {
          status = cf_math_matmul_cpu(dtype, out + i * out_stride, a + i * a_stride, b + i * b_stride, m, k, n);
          if(status != CF_OK) return status;
        }
      }
      return CF_OK;
    default:
      return CF_ERR_UNSUPPORTED;
  }
}

#if defined(CF_CUDA_AVAILABLE)
static cf_status cf_math_cuda_launch_dims(cf_usize len, int *blocks, int *threads)
{
  if(blocks == CF_NULL || threads == CF_NULL) return CF_ERR_NULL;
  *threads = 256;
  if(len == 0)
  {
    *blocks = 0;
    return CF_OK;
  }
  if(len > (cf_usize)INT_MAX * (cf_usize)*threads) return CF_ERR_BOUNDS;
  *blocks = (int)((len + (cf_usize)*threads - 1U) / (cf_usize)*threads);
  return CF_OK;
}

static cf_status cf_math_op_cuda(cf_math_op_kind op, const cf_math_handle_t *handler, cf_math_dtype dtype, void *op1_ptr, const void *op2_ptr, cf_usize len)
{
  int threads = 0;
  int blocks = 0;
  cudaStream_t stream = CF_NULL;
  cf_status status = CF_OK;

  if(len == 0) return CF_OK;

  status = cf_math_cuda_launch_dims(len, &blocks, &threads);
  if(status != CF_OK) return status;
  stream = handler != CF_NULL && handler->cuda_ctx != CF_NULL ? handler->cuda_ctx->stream : CF_NULL;

  switch(dtype)
  {
    case CF_MATH_DTYPE_F32:
      cf_math_op_kernel_f32<<<blocks, threads, 0, stream>>>(op, (float *)op1_ptr, (const float *)op2_ptr, len);
      break;
    case CF_MATH_DTYPE_F64:
      cf_math_op_kernel_f64<<<blocks, threads, 0, stream>>>(op, (double *)op1_ptr, (const double *)op2_ptr, len);
      break;
    case CF_MATH_DTYPE_I32:
      cf_math_op_kernel_i32<<<blocks, threads, 0, stream>>>(op, (cf_i32 *)op1_ptr, (const cf_i32 *)op2_ptr, len);
      break;
    default:
      return CF_ERR_UNSUPPORTED;
  }

  if(cudaGetLastError() != cudaSuccess) return CF_ERR_CUDA_LAUNCH;
  return CF_OK;
}

static cf_status cf_math_unary_cuda(cf_math_op_kind op, const cf_math_handle_t *handler, cf_math_dtype dtype, void *x_ptr, cf_usize len)
{
  int threads = 0;
  int blocks = 0;
  cudaStream_t stream = CF_NULL;
  cf_status status = CF_OK;

  if(len == 0) return CF_OK;
  status = cf_math_cuda_launch_dims(len, &blocks, &threads);
  if(status != CF_OK) return status;
  stream = handler != CF_NULL && handler->cuda_ctx != CF_NULL ? handler->cuda_ctx->stream : CF_NULL;

  switch(dtype)
  {
    case CF_MATH_DTYPE_F32:
      cf_math_unary_kernel_f32<<<blocks, threads, 0, stream>>>(op, (float *)x_ptr, len);
      break;
    case CF_MATH_DTYPE_F64:
      cf_math_unary_kernel_f64<<<blocks, threads, 0, stream>>>(op, (double *)x_ptr, len);
      break;
    default:
      return CF_ERR_UNSUPPORTED;
  }

  if(cudaGetLastError() != cudaSuccess) return CF_ERR_CUDA_LAUNCH;
  return CF_OK;
}

static cf_status cf_math_scalar_cuda(cf_math_op_kind op, const cf_math_handle_t *handler, cf_math_dtype dtype, void *x_ptr, cf_usize len, double scalar)
{
  int threads = 0;
  int blocks = 0;
  cudaStream_t stream = CF_NULL;
  cf_status status = CF_OK;

  if(len == 0) return CF_OK;
  status = cf_math_cuda_launch_dims(len, &blocks, &threads);
  if(status != CF_OK) return status;
  stream = handler != CF_NULL && handler->cuda_ctx != CF_NULL ? handler->cuda_ctx->stream : CF_NULL;

  switch(dtype)
  {
    case CF_MATH_DTYPE_F32:
      cf_math_scalar_kernel_f32<<<blocks, threads, 0, stream>>>(op, (float *)x_ptr, (float)scalar, len);
      break;
    case CF_MATH_DTYPE_F64:
      cf_math_scalar_kernel_f64<<<blocks, threads, 0, stream>>>(op, (double *)x_ptr, scalar, len);
      break;
    case CF_MATH_DTYPE_I32:
      cf_math_scalar_kernel_i32<<<blocks, threads, 0, stream>>>(op, (cf_i32 *)x_ptr, (cf_i32)scalar, len);
      break;
    default:
      return CF_ERR_UNSUPPORTED;
  }

  if(cudaGetLastError() != cudaSuccess) return CF_ERR_CUDA_LAUNCH;
  return CF_OK;
}

static cf_status cf_math_reduce_cuda(cf_math_op_kind op, const cf_math_handle_t *handler, cf_math_dtype dtype, void *out_ptr, const void *x_ptr, cf_usize len)
{
  cudaStream_t stream = handler != CF_NULL && handler->cuda_ctx != CF_NULL ? handler->cuda_ctx->stream : CF_NULL;
  size_t temp_bytes = 0;
  cudaError_t cuda_status = cudaSuccess;
  cf_usize elem_size = cf_math_dtype_size(dtype);

  if(handler == CF_NULL || handler->cuda_ctx == CF_NULL) return CF_ERR_STATE;
  if(len > (cf_usize)INT_MAX) return CF_ERR_BOUNDS;
  if(op != CF_MATH_OP_SUM && op != CF_MATH_OP_MEAN) return CF_ERR_UNSUPPORTED;
  if(elem_size == 0) return CF_ERR_UNSUPPORTED;
  if(len == 0)
  {
    if(cudaMemsetAsync(out_ptr, 0, (size_t)elem_size, stream) != cudaSuccess) return CF_ERR_CUDA_MEMORY;
    return CF_OK;
  }

  switch(dtype)
  {
    case CF_MATH_DTYPE_F32:
      if(handler->cuda_ctx->cuda_workspace.ptr != CF_NULL && handler->cuda_ctx->cuda_workspace.size >= len * elem_size)
      {
        temp_bytes = (size_t)handler->cuda_ctx->cuda_workspace.size;
        cuda_status = cudaSuccess;
      }
      else
      {
        cuda_status = cub::DeviceReduce::Sum(CF_NULL, temp_bytes, (const float *)x_ptr, (float *)out_ptr, (int)len, stream);
      }
      break;
    case CF_MATH_DTYPE_F64:
      if(handler->cuda_ctx->cuda_workspace.ptr != CF_NULL && handler->cuda_ctx->cuda_workspace.size >= len * elem_size)
      {
        temp_bytes = (size_t)handler->cuda_ctx->cuda_workspace.size;
        cuda_status = cudaSuccess;
      }
      else
      {
        cuda_status = cub::DeviceReduce::Sum(CF_NULL, temp_bytes, (const double *)x_ptr, (double *)out_ptr, (int)len, stream);
      }
      break;
    case CF_MATH_DTYPE_I32:
      if(handler->cuda_ctx->cuda_workspace.ptr != CF_NULL && handler->cuda_ctx->cuda_workspace.size >= len * elem_size)
      {
        temp_bytes = (size_t)handler->cuda_ctx->cuda_workspace.size;
        cuda_status = cudaSuccess;
      }
      else
      {
        cuda_status = cub::DeviceReduce::Sum(CF_NULL, temp_bytes, (const cf_i32 *)x_ptr, (cf_i32 *)out_ptr, (int)len, stream);
      }
      break;
    default:
      return CF_ERR_UNSUPPORTED;
  }
  if(cuda_status != cudaSuccess) return CF_ERR_CUDA;

  if(temp_bytes != 0)
  {
    cf_usize reserve_bytes = (cf_usize)temp_bytes < len * elem_size ? len * elem_size : (cf_usize)temp_bytes;
    cf_status reserve_status = cf_math_cuda_context_reserve(handler->cuda_ctx, reserve_bytes);
    if(reserve_status != CF_OK) return reserve_status;
  }

  switch(dtype)
  {
    case CF_MATH_DTYPE_F32:
      cuda_status = cub::DeviceReduce::Sum(handler->cuda_ctx->cuda_workspace.ptr, temp_bytes, (const float *)x_ptr, (float *)out_ptr, (int)len, stream);
      if(cuda_status != cudaSuccess) return CF_ERR_CUDA;
      if(op == CF_MATH_OP_MEAN) cf_math_mean_finalize_kernel<float><<<1, 1, 0, stream>>>((float *)out_ptr, len);
      break;
    case CF_MATH_DTYPE_F64:
      cuda_status = cub::DeviceReduce::Sum(handler->cuda_ctx->cuda_workspace.ptr, temp_bytes, (const double *)x_ptr, (double *)out_ptr, (int)len, stream);
      if(cuda_status != cudaSuccess) return CF_ERR_CUDA;
      if(op == CF_MATH_OP_MEAN) cf_math_mean_finalize_kernel<double><<<1, 1, 0, stream>>>((double *)out_ptr, len);
      break;
    case CF_MATH_DTYPE_I32:
      cuda_status = cub::DeviceReduce::Sum(handler->cuda_ctx->cuda_workspace.ptr, temp_bytes, (const cf_i32 *)x_ptr, (cf_i32 *)out_ptr, (int)len, stream);
      if(cuda_status != cudaSuccess) return CF_ERR_CUDA;
      if(op == CF_MATH_OP_MEAN) cf_math_mean_finalize_kernel<cf_i32><<<1, 1, 0, stream>>>((cf_i32 *)out_ptr, len);
      break;
    default:
      return CF_ERR_UNSUPPORTED;
  }

  if(cudaGetLastError() != cudaSuccess) return CF_ERR_CUDA_LAUNCH;
  return CF_OK;
}

static cf_status cf_math_matmul_cuda(cf_math_dtype dtype, const cf_math_handle_t *handler, void *out_ptr, const void *a_ptr, const void *b_ptr, cf_usize m, cf_usize k, cf_usize n)
{
  if(handler == CF_NULL || handler->cuda_ctx == CF_NULL || handler->cuda_ctx->cublas == CF_NULL) return CF_ERR_STATE;
  if(m > (cf_usize)INT_MAX || k > (cf_usize)INT_MAX || n > (cf_usize)INT_MAX) return CF_ERR_BOUNDS;

  switch(dtype)
  {
    case CF_MATH_DTYPE_F32:
    {
      const float alpha = 1.0f;
      const float beta = 0.0f;
      cublasStatus_t st = cublasSgemm(handler->cuda_ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N, (int)n, (int)m, (int)k, &alpha, (const float *)b_ptr, (int)n, (const float *)a_ptr, (int)k, &beta, (float *)out_ptr, (int)n);
      if(st != CUBLAS_STATUS_SUCCESS) return CF_ERR_CUDA;
      return CF_OK;
    }
    case CF_MATH_DTYPE_F64:
    {
      const double alpha = 1.0;
      const double beta = 0.0;
      cublasStatus_t st = cublasDgemm(handler->cuda_ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N, (int)n, (int)m, (int)k, &alpha, (const double *)b_ptr, (int)n, (const double *)a_ptr, (int)k, &beta, (double *)out_ptr, (int)n);
      if(st != CUBLAS_STATUS_SUCCESS) return CF_ERR_CUDA;
      return CF_OK;
    }
    default:
      return CF_ERR_UNSUPPORTED;
  }
}

static cf_status cf_math_matvec_cuda(cf_math_dtype dtype, const cf_math_handle_t *handler, void *out_ptr, const void *a_ptr, const void *x_ptr, cf_usize m, cf_usize n)
{
  if(handler == CF_NULL || handler->cuda_ctx == CF_NULL || handler->cuda_ctx->cublas == CF_NULL) return CF_ERR_STATE;
  if(m > (cf_usize)INT_MAX || n > (cf_usize)INT_MAX) return CF_ERR_BOUNDS;

  switch(dtype)
  {
    case CF_MATH_DTYPE_F32:
    {
      const float alpha = 1.0f;
      const float beta = 0.0f;
      cublasStatus_t st = cublasSgemv(handler->cuda_ctx->cublas, CUBLAS_OP_T, (int)n, (int)m, &alpha, (const float *)a_ptr, (int)n, (const float *)x_ptr, 1, &beta, (float *)out_ptr, 1);
      return st == CUBLAS_STATUS_SUCCESS ? CF_OK : CF_ERR_CUDA;
    }
    case CF_MATH_DTYPE_F64:
    {
      const double alpha = 1.0;
      const double beta = 0.0;
      cublasStatus_t st = cublasDgemv(handler->cuda_ctx->cublas, CUBLAS_OP_T, (int)n, (int)m, &alpha, (const double *)a_ptr, (int)n, (const double *)x_ptr, 1, &beta, (double *)out_ptr, 1);
      return st == CUBLAS_STATUS_SUCCESS ? CF_OK : CF_ERR_CUDA;
    }
    default:
      return CF_ERR_UNSUPPORTED;
  }
}

static cf_status cf_math_dot_cuda(cf_math_dtype dtype, const cf_math_handle_t *handler, void *out_ptr, const void *a_ptr, const void *b_ptr, cf_usize len)
{
  cublasPointerMode_t old_mode = CUBLAS_POINTER_MODE_HOST;
  cublasStatus_t st = CUBLAS_STATUS_SUCCESS;

  if(handler == CF_NULL || handler->cuda_ctx == CF_NULL || handler->cuda_ctx->cublas == CF_NULL) return CF_ERR_STATE;
  if(len > (cf_usize)INT_MAX) return CF_ERR_BOUNDS;
  if(cublasGetPointerMode(handler->cuda_ctx->cublas, &old_mode) != CUBLAS_STATUS_SUCCESS) return CF_ERR_CUDA;
  if(cublasSetPointerMode(handler->cuda_ctx->cublas, CUBLAS_POINTER_MODE_DEVICE) != CUBLAS_STATUS_SUCCESS) return CF_ERR_CUDA;

  switch(dtype)
  {
    case CF_MATH_DTYPE_F32:
      st = cublasSdot(handler->cuda_ctx->cublas, (int)len, (const float *)a_ptr, 1, (const float *)b_ptr, 1, (float *)out_ptr);
      break;
    case CF_MATH_DTYPE_F64:
      st = cublasDdot(handler->cuda_ctx->cublas, (int)len, (const double *)a_ptr, 1, (const double *)b_ptr, 1, (double *)out_ptr);
      break;
    default:
      CF_UNUSED(cublasSetPointerMode(handler->cuda_ctx->cublas, old_mode));
      return CF_ERR_UNSUPPORTED;
  }

  if(cublasSetPointerMode(handler->cuda_ctx->cublas, old_mode) != CUBLAS_STATUS_SUCCESS) return CF_ERR_CUDA;
  return st == CUBLAS_STATUS_SUCCESS ? CF_OK : CF_ERR_CUDA;
}

static cf_status cf_math_batched_matmul_cuda(cf_math_dtype dtype, const cf_math_handle_t *handler, void *out_ptr, const void *a_ptr, const void *b_ptr, cf_usize batch, cf_usize m, cf_usize k, cf_usize n)
{
  long long int a_stride = 0;
  long long int b_stride = 0;
  long long int out_stride = 0;

  if(handler == CF_NULL || handler->cuda_ctx == CF_NULL || handler->cuda_ctx->cublas == CF_NULL) return CF_ERR_STATE;
  if(m > (cf_usize)INT_MAX || k > (cf_usize)INT_MAX || n > (cf_usize)INT_MAX || batch > (cf_usize)INT_MAX) return CF_ERR_BOUNDS;
  if(m != 0 && k > (cf_usize)LLONG_MAX / m) return CF_ERR_OVERFLOW;
  if(k != 0 && n > (cf_usize)LLONG_MAX / k) return CF_ERR_OVERFLOW;
  if(m != 0 && n > (cf_usize)LLONG_MAX / m) return CF_ERR_OVERFLOW;

  a_stride = (long long int)(m * k);
  b_stride = (long long int)(k * n);
  out_stride = (long long int)(m * n);

  switch(dtype)
  {
    case CF_MATH_DTYPE_F32:
    {
      const float alpha = 1.0f;
      const float beta = 0.0f;
      cublasStatus_t st = cublasSgemmStridedBatched(handler->cuda_ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N, (int)n, (int)m, (int)k, &alpha, (const float *)b_ptr, (int)n, b_stride, (const float *)a_ptr, (int)k, a_stride, &beta, (float *)out_ptr, (int)n, out_stride, (int)batch);
      return st == CUBLAS_STATUS_SUCCESS ? CF_OK : CF_ERR_CUDA;
    }
    case CF_MATH_DTYPE_F64:
    {
      const double alpha = 1.0;
      const double beta = 0.0;
      cublasStatus_t st = cublasDgemmStridedBatched(handler->cuda_ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N, (int)n, (int)m, (int)k, &alpha, (const double *)b_ptr, (int)n, b_stride, (const double *)a_ptr, (int)k, a_stride, &beta, (double *)out_ptr, (int)n, out_stride, (int)batch);
      return st == CUBLAS_STATUS_SUCCESS ? CF_OK : CF_ERR_CUDA;
    }
    default:
      return CF_ERR_UNSUPPORTED;
  }
}
#endif

cf_status cf_math_op_check(cf_math_op_kind op, const cf_math *op1, const cf_math *op2)
{
  cf_status status = CF_OK;

  if(cf_math_is_elementwise_binary_op(op) == CF_FALSE) return CF_ERR_UNSUPPORTED;
  status = cf_math_check_same_view_shape(op1, op2);
  if(status != CF_OK) return status;
  return cf_math_is_supported_binary_dtype(op1->handler->storage.dtype) == CF_TRUE ? CF_OK : CF_ERR_UNSUPPORTED;
}

cf_status cf_math_op(cf_math_op_kind op, cf_math *op1, const cf_math *op2)
{
  void *op1_ptr = CF_NULL;
  void *op2_ptr = CF_NULL;
  cf_status status = CF_OK;
  cf_math_dtype dtype = CF_MATH_DTYPE_BOOL;
  cf_usize len = 0;

  if(op1 == CF_NULL || op2 == CF_NULL) return CF_ERR_NULL;
  if(cf_math_is_elementwise_binary_op(op) == CF_FALSE) return CF_ERR_UNSUPPORTED;

  status = cf_math_resolve_ptr(op1, &op1_ptr);
  if(status != CF_OK) return status;
  status = cf_math_resolve_ptr(op2, &op2_ptr);
  if(status != CF_OK) return status;
  if(op1->handler == CF_NULL || op1->metadata == CF_NULL) return CF_ERR_STATE;

  dtype = op1->handler->storage.dtype;
  len = op1->metadata->len;

  if(op1->handler->storage.device == CF_MATH_DEVICE_CPU || (op1->handler->storage.allocator.mem_flag & CF_MATH_MEM_PINNED) != 0)
    return cf_math_op_cpu(op, dtype, op1_ptr, op2_ptr, len);

#if defined(CF_CUDA_AVAILABLE)
  return cf_math_op_cuda(op, op1->handler, dtype, op1_ptr, op2_ptr, len);
#else
  return CF_ERR_UNSUPPORTED;
#endif
}

cf_status cf_math_op_out(cf_math_op_kind op, cf_math *out, const cf_math *a, const cf_math *b)
{
  void *out_ptr = CF_NULL;
  void *a_ptr = CF_NULL;
  cf_status status = CF_OK;

  status = cf_math_op_check(op, out, a);
  if(status != CF_OK) return status;
  status = cf_math_op_check(op, out, b);
  if(status != CF_OK) return status;

  status = cf_math_resolve_ptr(out, &out_ptr);
  if(status != CF_OK) return status;
  status = cf_math_resolve_ptr(a, &a_ptr);
  if(status != CF_OK) return status;

  status = cf_math_copy_bytes(out, out_ptr, a_ptr, out->byte_size);
  if(status != CF_OK) return status;

  return cf_math_op(op, out, b);
}

cf_status cf_math_unary(cf_math_op_kind op, cf_math *x)
{
  void *x_ptr = CF_NULL;
  cf_status status = CF_OK;
  cf_math_dtype dtype = CF_MATH_DTYPE_BOOL;
  cf_usize len = 0;

  if(x == CF_NULL) return CF_ERR_NULL;
  if(cf_math_is_unary_op(op) == CF_FALSE) return CF_ERR_UNSUPPORTED;

  status = cf_math_resolve_ptr(x, &x_ptr);
  if(status != CF_OK) return status;
  if(x->handler == CF_NULL || x->metadata == CF_NULL) return CF_ERR_STATE;

  dtype = x->handler->storage.dtype;
  len = x->metadata->len;

  if(x->handler->storage.device == CF_MATH_DEVICE_CPU || (x->handler->storage.allocator.mem_flag & CF_MATH_MEM_PINNED) != 0)
    return cf_math_unary_cpu(op, dtype, x_ptr, len);

#if defined(CF_CUDA_AVAILABLE)
  return cf_math_unary_cuda(op, x->handler, dtype, x_ptr, len);
#else
  return CF_ERR_UNSUPPORTED;
#endif
}

cf_status cf_math_unary_out(cf_math_op_kind op, cf_math *out, const cf_math *x)
{
  void *out_ptr = CF_NULL;
  void *x_ptr = CF_NULL;
  cf_status status = CF_OK;

  if(cf_math_is_unary_op(op) == CF_FALSE) return CF_ERR_UNSUPPORTED;
  status = cf_math_check_same_view_shape(out, x);
  if(status != CF_OK) return status;
  if(cf_math_is_supported_float_dtype(out->handler->storage.dtype) == CF_FALSE) return CF_ERR_UNSUPPORTED;

  status = cf_math_resolve_ptr(out, &out_ptr);
  if(status != CF_OK) return status;
  status = cf_math_resolve_ptr(x, &x_ptr);
  if(status != CF_OK) return status;
  status = cf_math_copy_bytes(out, out_ptr, x_ptr, out->byte_size);
  if(status != CF_OK) return status;
  return cf_math_unary(op, out);
}

cf_status cf_math_scalar(cf_math_op_kind op, cf_math *x, double scalar)
{
  void *x_ptr = CF_NULL;
  cf_status status = CF_OK;
  cf_math_dtype dtype = CF_MATH_DTYPE_BOOL;
  cf_usize len = 0;

  if(x == CF_NULL) return CF_ERR_NULL;
  if(cf_math_is_elementwise_binary_op(op) == CF_FALSE) return CF_ERR_UNSUPPORTED;
  status = cf_math_resolve_ptr(x, &x_ptr);
  if(status != CF_OK) return status;
  if(x->handler == CF_NULL || x->metadata == CF_NULL) return CF_ERR_STATE;

  dtype = x->handler->storage.dtype;
  len = x->metadata->len;

  if(x->handler->storage.device == CF_MATH_DEVICE_CPU || (x->handler->storage.allocator.mem_flag & CF_MATH_MEM_PINNED) != 0)
    return cf_math_scalar_cpu(op, dtype, x_ptr, len, scalar);

#if defined(CF_CUDA_AVAILABLE)
  return cf_math_scalar_cuda(op, x->handler, dtype, x_ptr, len, scalar);
#else
  return CF_ERR_UNSUPPORTED;
#endif
}

cf_status cf_math_scalar_out(cf_math_op_kind op, cf_math *out, const cf_math *x, double scalar)
{
  void *out_ptr = CF_NULL;
  void *x_ptr = CF_NULL;
  cf_status status = CF_OK;

  if(cf_math_is_elementwise_binary_op(op) == CF_FALSE) return CF_ERR_UNSUPPORTED;
  status = cf_math_check_same_view_shape(out, x);
  if(status != CF_OK) return status;
  if(cf_math_is_supported_binary_dtype(out->handler->storage.dtype) == CF_FALSE) return CF_ERR_UNSUPPORTED;

  status = cf_math_resolve_ptr(out, &out_ptr);
  if(status != CF_OK) return status;
  status = cf_math_resolve_ptr(x, &x_ptr);
  if(status != CF_OK) return status;
  status = cf_math_copy_bytes(out, out_ptr, x_ptr, out->byte_size);
  if(status != CF_OK) return status;
  return cf_math_scalar(op, out, scalar);
}

static cf_status cf_math_reduce(cf_math_op_kind op, cf_math *out, const cf_math *x)
{
  void *out_ptr = CF_NULL;
  void *x_ptr = CF_NULL;
  cf_status status = CF_OK;
  cf_math_dtype dtype = CF_MATH_DTYPE_BOOL;

  if(out == CF_NULL || x == CF_NULL) return CF_ERR_NULL;
  if(op != CF_MATH_OP_SUM && op != CF_MATH_OP_MEAN) return CF_ERR_UNSUPPORTED;
  if(cf_math_is_bound(out) == CF_FALSE || cf_math_is_bound(x) == CF_FALSE) return CF_ERR_STATE;
  if(out->handler->storage.device != x->handler->storage.device) return CF_ERR_INVALID;
  if(out->handler->storage.dtype != x->handler->storage.dtype) return CF_ERR_INVALID;
  if(out->metadata->len != 1) return CF_ERR_INVALID;

  dtype = out->handler->storage.dtype;
  if(cf_math_is_supported_binary_dtype(dtype) == CF_FALSE) return CF_ERR_UNSUPPORTED;

  status = cf_math_resolve_ptr(out, &out_ptr);
  if(status != CF_OK) return status;
  status = cf_math_resolve_ptr(x, &x_ptr);
  if(status != CF_OK) return status;

  if(out->handler->storage.device == CF_MATH_DEVICE_CPU || (out->handler->storage.allocator.mem_flag & CF_MATH_MEM_PINNED) != 0)
    return cf_math_reduce_cpu(op, dtype, out_ptr, x_ptr, x->metadata->len);

#if defined(CF_CUDA_AVAILABLE)
  return cf_math_reduce_cuda(op, out->handler, dtype, out_ptr, x_ptr, x->metadata->len);
#else
  return CF_ERR_UNSUPPORTED;
#endif
}

cf_status cf_math_reduce_sum(cf_math *out, const cf_math *x)
{
  return cf_math_reduce(CF_MATH_OP_SUM, out, x);
}

cf_status cf_math_reduce_mean(cf_math *out, const cf_math *x)
{
  return cf_math_reduce(CF_MATH_OP_MEAN, out, x);
}

cf_status cf_math_dot(cf_math *out, const cf_math *a, const cf_math *b)
{
  void *out_ptr = CF_NULL;
  void *a_ptr = CF_NULL;
  void *b_ptr = CF_NULL;
  cf_status status = CF_OK;
  cf_math_dtype dtype = CF_MATH_DTYPE_BOOL;

  if(out == CF_NULL || a == CF_NULL || b == CF_NULL) return CF_ERR_NULL;
  if(cf_math_is_bound(out) == CF_FALSE || cf_math_is_bound(a) == CF_FALSE || cf_math_is_bound(b) == CF_FALSE) return CF_ERR_STATE;
  if(out->handler->storage.device != a->handler->storage.device || out->handler->storage.device != b->handler->storage.device) return CF_ERR_INVALID;
  if(out->handler->storage.dtype != a->handler->storage.dtype || out->handler->storage.dtype != b->handler->storage.dtype) return CF_ERR_INVALID;
  if(out->metadata->len != 1 || a->metadata->len != b->metadata->len) return CF_ERR_INVALID;
  if(cf_math_is_compact_row_major(a->metadata) == CF_FALSE || cf_math_is_compact_row_major(b->metadata) == CF_FALSE || cf_math_is_compact_row_major(out->metadata) == CF_FALSE) return CF_ERR_UNSUPPORTED;

  dtype = out->handler->storage.dtype;
  if(cf_math_is_supported_float_dtype(dtype) == CF_FALSE) return CF_ERR_UNSUPPORTED;

  status = cf_math_resolve_ptr(out, &out_ptr);
  if(status != CF_OK) return status;
  status = cf_math_resolve_ptr(a, &a_ptr);
  if(status != CF_OK) return status;
  status = cf_math_resolve_ptr(b, &b_ptr);
  if(status != CF_OK) return status;

  if(out->handler->storage.device == CF_MATH_DEVICE_CPU || (out->handler->storage.allocator.mem_flag & CF_MATH_MEM_PINNED) != 0)
    return cf_math_dot_cpu(dtype, out_ptr, a_ptr, b_ptr, a->metadata->len);

#if defined(CF_CUDA_AVAILABLE)
  return cf_math_dot_cuda(dtype, out->handler, out_ptr, a_ptr, b_ptr, a->metadata->len);
#else
  return CF_ERR_UNSUPPORTED;
#endif
}

cf_status cf_math_matmul(cf_math *out, const cf_math *a, const cf_math *b)
{
  void *out_ptr = CF_NULL;
  void *a_ptr = CF_NULL;
  void *b_ptr = CF_NULL;
  cf_status status = CF_OK;
  cf_math_dtype dtype = CF_MATH_DTYPE_BOOL;
  cf_usize m = 0;
  cf_usize k = 0;
  cf_usize n = 0;

  if(out == CF_NULL || a == CF_NULL || b == CF_NULL) return CF_ERR_NULL;
  if(cf_math_is_bound(out) == CF_FALSE || cf_math_is_bound(a) == CF_FALSE || cf_math_is_bound(b) == CF_FALSE) return CF_ERR_STATE;
  if(out->handler->storage.device != a->handler->storage.device || out->handler->storage.device != b->handler->storage.device) return CF_ERR_INVALID;
  if(out->handler->storage.dtype != a->handler->storage.dtype || out->handler->storage.dtype != b->handler->storage.dtype) return CF_ERR_INVALID;
  if(out->metadata->rank != 2 || a->metadata->rank != 2 || b->metadata->rank != 2) return CF_ERR_INVALID;
  if(cf_math_is_compact_row_major(out->metadata) == CF_FALSE || cf_math_is_compact_row_major(a->metadata) == CF_FALSE || cf_math_is_compact_row_major(b->metadata) == CF_FALSE) return CF_ERR_UNSUPPORTED;

  m = a->metadata->dim[0];
  k = a->metadata->dim[1];
  n = b->metadata->dim[1];
  if(b->metadata->dim[0] != k || out->metadata->dim[0] != m || out->metadata->dim[1] != n) return CF_ERR_INVALID;

  dtype = out->handler->storage.dtype;
  if(cf_math_is_supported_float_dtype(dtype) == CF_FALSE) return CF_ERR_UNSUPPORTED;

  status = cf_math_resolve_ptr(out, &out_ptr);
  if(status != CF_OK) return status;
  status = cf_math_resolve_ptr(a, &a_ptr);
  if(status != CF_OK) return status;
  status = cf_math_resolve_ptr(b, &b_ptr);
  if(status != CF_OK) return status;

  if(out->handler->storage.device == CF_MATH_DEVICE_CPU || (out->handler->storage.allocator.mem_flag & CF_MATH_MEM_PINNED) != 0)
    return cf_math_matmul_cpu(dtype, out_ptr, a_ptr, b_ptr, m, k, n);

#if defined(CF_CUDA_AVAILABLE)
  return cf_math_matmul_cuda(dtype, out->handler, out_ptr, a_ptr, b_ptr, m, k, n);
#else
  return CF_ERR_UNSUPPORTED;
#endif
}

cf_status cf_math_matvec(cf_math *out, const cf_math *a, const cf_math *x)
{
  void *out_ptr = CF_NULL;
  void *a_ptr = CF_NULL;
  void *x_ptr = CF_NULL;
  cf_status status = CF_OK;
  cf_math_dtype dtype = CF_MATH_DTYPE_BOOL;
  cf_usize m = 0;
  cf_usize n = 0;

  if(out == CF_NULL || a == CF_NULL || x == CF_NULL) return CF_ERR_NULL;
  if(cf_math_is_bound(out) == CF_FALSE || cf_math_is_bound(a) == CF_FALSE || cf_math_is_bound(x) == CF_FALSE) return CF_ERR_STATE;
  if(out->handler->storage.device != a->handler->storage.device || out->handler->storage.device != x->handler->storage.device) return CF_ERR_INVALID;
  if(out->handler->storage.dtype != a->handler->storage.dtype || out->handler->storage.dtype != x->handler->storage.dtype) return CF_ERR_INVALID;
  if(out->metadata->rank != 1 || a->metadata->rank != 2 || x->metadata->rank != 1) return CF_ERR_INVALID;
  if(cf_math_is_compact_row_major(out->metadata) == CF_FALSE || cf_math_is_compact_row_major(a->metadata) == CF_FALSE || cf_math_is_compact_row_major(x->metadata) == CF_FALSE) return CF_ERR_UNSUPPORTED;

  m = a->metadata->dim[0];
  n = a->metadata->dim[1];
  if(x->metadata->dim[0] != n || out->metadata->dim[0] != m) return CF_ERR_INVALID;

  dtype = out->handler->storage.dtype;
  if(cf_math_is_supported_float_dtype(dtype) == CF_FALSE) return CF_ERR_UNSUPPORTED;

  status = cf_math_resolve_ptr(out, &out_ptr);
  if(status != CF_OK) return status;
  status = cf_math_resolve_ptr(a, &a_ptr);
  if(status != CF_OK) return status;
  status = cf_math_resolve_ptr(x, &x_ptr);
  if(status != CF_OK) return status;

  if(out->handler->storage.device == CF_MATH_DEVICE_CPU || (out->handler->storage.allocator.mem_flag & CF_MATH_MEM_PINNED) != 0)
    return cf_math_matvec_cpu(dtype, out_ptr, a_ptr, x_ptr, m, n);

#if defined(CF_CUDA_AVAILABLE)
  return cf_math_matvec_cuda(dtype, out->handler, out_ptr, a_ptr, x_ptr, m, n);
#else
  return CF_ERR_UNSUPPORTED;
#endif
}

cf_status cf_math_batched_matmul(cf_math *out, const cf_math *a, const cf_math *b)
{
  void *out_ptr = CF_NULL;
  void *a_ptr = CF_NULL;
  void *b_ptr = CF_NULL;
  cf_status status = CF_OK;
  cf_math_dtype dtype = CF_MATH_DTYPE_BOOL;
  cf_usize batch = 0;
  cf_usize m = 0;
  cf_usize k = 0;
  cf_usize n = 0;

  if(out == CF_NULL || a == CF_NULL || b == CF_NULL) return CF_ERR_NULL;
  if(cf_math_is_bound(out) == CF_FALSE || cf_math_is_bound(a) == CF_FALSE || cf_math_is_bound(b) == CF_FALSE) return CF_ERR_STATE;
  if(out->handler->storage.device != a->handler->storage.device || out->handler->storage.device != b->handler->storage.device) return CF_ERR_INVALID;
  if(out->handler->storage.dtype != a->handler->storage.dtype || out->handler->storage.dtype != b->handler->storage.dtype) return CF_ERR_INVALID;
  if(out->metadata->rank != 3 || a->metadata->rank != 3 || b->metadata->rank != 3) return CF_ERR_INVALID;
  if(cf_math_is_compact_row_major(out->metadata) == CF_FALSE || cf_math_is_compact_row_major(a->metadata) == CF_FALSE || cf_math_is_compact_row_major(b->metadata) == CF_FALSE) return CF_ERR_UNSUPPORTED;

  batch = a->metadata->dim[0];
  m = a->metadata->dim[1];
  k = a->metadata->dim[2];
  n = b->metadata->dim[2];
  if(b->metadata->dim[0] != batch || out->metadata->dim[0] != batch) return CF_ERR_INVALID;
  if(b->metadata->dim[1] != k || out->metadata->dim[1] != m || out->metadata->dim[2] != n) return CF_ERR_INVALID;

  dtype = out->handler->storage.dtype;
  if(cf_math_is_supported_float_dtype(dtype) == CF_FALSE) return CF_ERR_UNSUPPORTED;

  status = cf_math_resolve_ptr(out, &out_ptr);
  if(status != CF_OK) return status;
  status = cf_math_resolve_ptr(a, &a_ptr);
  if(status != CF_OK) return status;
  status = cf_math_resolve_ptr(b, &b_ptr);
  if(status != CF_OK) return status;

  if(out->handler->storage.device == CF_MATH_DEVICE_CPU || (out->handler->storage.allocator.mem_flag & CF_MATH_MEM_PINNED) != 0)
    return cf_math_batched_matmul_cpu(dtype, out_ptr, a_ptr, b_ptr, batch, m, k, n);

#if defined(CF_CUDA_AVAILABLE)
  return cf_math_batched_matmul_cuda(dtype, out->handler, out_ptr, a_ptr, b_ptr, batch, m, k, n);
#else
  return CF_ERR_UNSUPPORTED;
#endif
}

cf_status cf_math_bind(cf_math *x, cf_math_handle_t *handler, cf_math_metadata *metadata)
{
  void *ptr = CF_NULL;
  cf_usize elem_size = 0;
  cf_usize bytes = 0;
  cf_usize offset = 0;
  cf_usize required = 0;
  cf_status status = CF_OK;

  if(x == CF_NULL || handler == CF_NULL || metadata == CF_NULL) return CF_ERR_NULL;

  elem_size = cf_math_dtype_size(handler->storage.dtype);
  if(elem_size == 0) return CF_ERR_INVALID;
  if(metadata->len > (cf_usize)-1 / elem_size) return CF_ERR_OVERFLOW;

  bytes = metadata->len * elem_size;
  offset = handler->storage.arena.offset;
  if((handler->storage.allocator.mem_flag & CF_MATH_MEM_ALIGNED128) != 0)
  {
    if(offset > (cf_usize)-1 - 127U) return CF_ERR_OVERFLOW;
    offset = (offset + 127U) & ~((cf_usize)127U);
  }
  if(offset > (cf_usize)-1 - bytes) return CF_ERR_OVERFLOW;

  required = offset + bytes;
  if(required > handler->storage.arena.capacity)
  {
    status = cf_math_handle_reserve(handler, required);
    if(status != CF_OK) return status;
  }

  status = cf_math_handle_alloc(handler, bytes, &ptr);
  if(status != CF_OK) return status;

  x->byte_offset = ptr != CF_NULL ? (cf_usize)((cf_u8 *)ptr - (cf_u8 *)handler->storage.allocator.backend) : 0;
  x->byte_size = bytes;
  x->metadata = metadata;
  x->handler = handler;
  x->grad = CF_NULL;
  x->grad_fn = CF_NULL;
  x->grad_state = CF_MATH_GRAD_NONE;

  return CF_OK;
}

static cf_status cf_math_resolve_ptr(const cf_math *x, void **ptr)
{
  if(x == CF_NULL || ptr == CF_NULL) return CF_ERR_NULL;
  *ptr = CF_NULL;
  if(x->handler == CF_NULL || x->metadata == CF_NULL) return CF_ERR_STATE;
  if(x->byte_size == 0) return CF_OK;
  if(x->handler->storage.allocator.backend == CF_NULL) return CF_ERR_STATE;
  if(x->byte_offset > x->handler->storage.arena.capacity) return CF_ERR_BOUNDS;
  if(x->byte_size > x->handler->storage.arena.capacity - x->byte_offset) return CF_ERR_BOUNDS;

  *ptr = (void *)((cf_u8 *)x->handler->storage.allocator.backend + x->byte_offset);
  return CF_OK;
}

static cf_status cf_math_copy_bytes(const cf_math *x, void *dst, const void *src, cf_usize bytes)
{
  if(bytes == 0) return CF_OK;
  if(x == CF_NULL || dst == CF_NULL || src == CF_NULL) return CF_ERR_NULL;
  if(x->handler == CF_NULL) return CF_ERR_STATE;

  if(x->handler->storage.device == CF_MATH_DEVICE_CPU || (x->handler->storage.allocator.mem_flag & CF_MATH_MEM_PINNED) != 0)
  {
    memcpy(dst, src, bytes);
    return CF_OK;
  }

#if defined(CF_CUDA_AVAILABLE)
  if(cudaMemcpyAsync(dst, src, (size_t)bytes, cudaMemcpyDefault, x->handler->cuda_ctx != CF_NULL ? x->handler->cuda_ctx->stream : CF_NULL) != cudaSuccess)
    return CF_ERR_CUDA_COPY;
  return CF_OK;
#else
  return CF_ERR_UNSUPPORTED;
#endif
}

cf_status cf_math_cpy_h2d(cf_math *dst, const void *host_data, cf_usize count)
{
  void *ptr = CF_NULL;
  cf_usize elem_size = 0;
  cf_usize bytes = 0;
  cf_status status = CF_OK;

  if(dst == CF_NULL || host_data == CF_NULL) return CF_ERR_NULL;
  if(dst->handler == CF_NULL || dst->metadata == CF_NULL) return CF_ERR_STATE;
  if(count > dst->metadata->len) return CF_ERR_BOUNDS;

  status = cf_math_resolve_ptr(dst, &ptr);
  if(status != CF_OK) return status;
  if(ptr == CF_NULL && count != 0) return CF_ERR_STATE;

  elem_size = cf_math_dtype_size(dst->handler->storage.dtype);
  if(elem_size == 0) return CF_ERR_INVALID;
  if(count > (cf_usize)-1 / elem_size) return CF_ERR_OVERFLOW;

  bytes = count * elem_size;
  if(bytes > dst->byte_size) return CF_ERR_BOUNDS;

  return cf_math_copy_bytes(dst, ptr, host_data, bytes);
}

cf_status cf_math_cpy_d2h(const cf_math *src, void *host_data, cf_usize count)
{
  void *ptr = CF_NULL;
  cf_usize elem_size = 0;
  cf_usize bytes = 0;
  cf_status status = CF_OK;

  if(src == CF_NULL || host_data == CF_NULL) return CF_ERR_NULL;
  if(src->handler == CF_NULL || src->metadata == CF_NULL) return CF_ERR_STATE;
  if(count > src->metadata->len) return CF_ERR_BOUNDS;

  status = cf_math_resolve_ptr(src, &ptr);
  if(status != CF_OK) return status;
  if(ptr == CF_NULL && count != 0) return CF_ERR_STATE;

  elem_size = cf_math_dtype_size(src->handler->storage.dtype);
  if(elem_size == 0) return CF_ERR_INVALID;
  if(count > (cf_usize)-1 / elem_size) return CF_ERR_OVERFLOW;

  bytes = count * elem_size;
  if(bytes > src->byte_size) return CF_ERR_BOUNDS;

  status = cf_math_copy_bytes(src, host_data, ptr, bytes);
  if(status != CF_OK) return status;
  return cf_math_handle_sync(src->handler);
}

cf_status cf_math_rebind(cf_math *x, cf_math_handle_t *handler, cf_math_metadata *metadata)
{
  cf_status status = CF_OK;

  if(x == CF_NULL || handler == CF_NULL || metadata == CF_NULL) return CF_ERR_NULL;

  status = cf_math_unbind(x);
  if(status != CF_OK) return status;

  return cf_math_bind(x, handler, metadata);
}

cf_status cf_math_unbind(cf_math *x)
{
  cf_status status = CF_OK;
  cf_bool released = CF_FALSE;

  if(x == CF_NULL) return CF_ERR_NULL;

  if(x->handler != CF_NULL && x->byte_size != 0)
  {
    status = cf_math_storage_release_slice(&x->handler->storage, x->byte_offset, x->byte_size, &released);
    if(status != CF_OK) return status;
  }

  x->byte_offset = 0;
  x->byte_size = 0;
  x->metadata = CF_NULL;
  x->handler = CF_NULL;
  x->grad = CF_NULL;
  x->grad_fn = CF_NULL;
  x->grad_state = CF_MATH_GRAD_NONE;

  return CF_OK;
}
