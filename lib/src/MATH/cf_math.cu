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

#if defined(CF_CUDA_AVAILABLE)
#include <cub/cub.cuh>
#endif

static cf_status cf_math_resolve_ptr(const cf_math *x, void **ptr);
static cf_status cf_math_copy_bytes(const cf_math *x, void *dst, const void *src, cf_usize bytes);

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

#define CF_MATH_CPU_OP_LOOP(type) \
  do \
  { \
    type *dst = (type *)op1_ptr; \
    const type *src = (const type *)op2_ptr; \
    switch(op) \
    { \
      case CF_MATH_OP_ADD: for(cf_usize i = 0; i < len; ++i) dst[i] = dst[i] + src[i]; return CF_OK; \
      case CF_MATH_OP_SUB: for(cf_usize i = 0; i < len; ++i) dst[i] = dst[i] - src[i]; return CF_OK; \
      case CF_MATH_OP_MUL: for(cf_usize i = 0; i < len; ++i) dst[i] = dst[i] * src[i]; return CF_OK; \
      case CF_MATH_OP_DIV: for(cf_usize i = 0; i < len; ++i) dst[i] = dst[i] / src[i]; return CF_OK; \
      default: return CF_ERR_UNSUPPORTED; \
    } \
  } while(0)

static cf_status cf_math_op_cpu(cf_math_op_kind op, cf_math_dtype dtype, void *op1_ptr, const void *op2_ptr, cf_usize len)
{
  switch(dtype)
  {
    case CF_MATH_DTYPE_F32: CF_MATH_CPU_OP_LOOP(float);
    case CF_MATH_DTYPE_F64: CF_MATH_CPU_OP_LOOP(double);
    case CF_MATH_DTYPE_I32: CF_MATH_CPU_OP_LOOP(cf_i32);
    default: return CF_ERR_UNSUPPORTED;
  }
}

#define CF_MATH_CPU_UNARY_LOOP(type, exp_fn, log_fn, sqrt_fn, tanh_fn) \
  do \
  { \
    type *dst = (type *)x_ptr; \
    for(cf_usize i = 0; i < len; ++i) \
    { \
      type v = dst[i]; \
      switch(op) \
      { \
        case CF_MATH_OP_NEG: dst[i] = -v; break; \
        case CF_MATH_OP_RELU: dst[i] = v > (type)0 ? v : (type)0; break; \
        case CF_MATH_OP_GELU: dst[i] = (type)0.5 * v * ((type)1 + tanh_fn((type)0.7978845608028654 * (v + (type)0.044715 * v * v * v))); break; \
        case CF_MATH_OP_EXP: dst[i] = exp_fn(v); break; \
        case CF_MATH_OP_LOG: dst[i] = log_fn(v); break; \
        case CF_MATH_OP_SQRT: dst[i] = sqrt_fn(v); break; \
        case CF_MATH_OP_SIGMOID: dst[i] = (type)1 / ((type)1 + exp_fn(-v)); break; \
        case CF_MATH_OP_TANH: dst[i] = tanh_fn(v); break; \
        default: return CF_ERR_UNSUPPORTED; \
      } \
    } \
    return CF_OK; \
  } while(0)

static cf_status cf_math_unary_cpu(cf_math_op_kind op, cf_math_dtype dtype, void *x_ptr, cf_usize len)
{
  switch(dtype)
  {
    case CF_MATH_DTYPE_F32: CF_MATH_CPU_UNARY_LOOP(float, expf, logf, sqrtf, tanhf);
    case CF_MATH_DTYPE_F64: CF_MATH_CPU_UNARY_LOOP(double, exp, log, sqrt, tanh);
    default: return CF_ERR_UNSUPPORTED;
  }
}

#define CF_MATH_CPU_SCALAR_LOOP(type, scalar_expr) \
  do \
  { \
    type *dst = (type *)x_ptr; \
    type s = (type)(scalar_expr); \
    switch(op) \
    { \
      case CF_MATH_OP_ADD: for(cf_usize i = 0; i < len; ++i) dst[i] = dst[i] + s; return CF_OK; \
      case CF_MATH_OP_SUB: for(cf_usize i = 0; i < len; ++i) dst[i] = dst[i] - s; return CF_OK; \
      case CF_MATH_OP_MUL: for(cf_usize i = 0; i < len; ++i) dst[i] = dst[i] * s; return CF_OK; \
      case CF_MATH_OP_DIV: for(cf_usize i = 0; i < len; ++i) dst[i] = dst[i] / s; return CF_OK; \
      default: return CF_ERR_UNSUPPORTED; \
    } \
  } while(0)

static cf_status cf_math_scalar_cpu(cf_math_op_kind op, cf_math_dtype dtype, void *x_ptr, cf_usize len, double scalar)
{
  switch(dtype)
  {
    case CF_MATH_DTYPE_F32: CF_MATH_CPU_SCALAR_LOOP(float, scalar);
    case CF_MATH_DTYPE_F64: CF_MATH_CPU_SCALAR_LOOP(double, scalar);
    case CF_MATH_DTYPE_I32: CF_MATH_CPU_SCALAR_LOOP(cf_i32, (cf_i32)scalar);
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
      float sum = 0.0f;
      const float *x = (const float *)x_ptr;
      for(cf_usize i = 0; i < len; ++i) sum += x[i];
      ((float *)out_ptr)[0] = op == CF_MATH_OP_MEAN && len != 0 ? sum / (float)len : sum;
      return CF_OK;
    }
    case CF_MATH_DTYPE_F64:
    {
      double sum = 0.0;
      const double *x = (const double *)x_ptr;
      for(cf_usize i = 0; i < len; ++i) sum += x[i];
      ((double *)out_ptr)[0] = op == CF_MATH_OP_MEAN && len != 0 ? sum / (double)len : sum;
      return CF_OK;
    }
    case CF_MATH_DTYPE_I32:
    {
      cf_i32 sum = 0;
      const cf_i32 *x = (const cf_i32 *)x_ptr;
      for(cf_usize i = 0; i < len; ++i) sum += x[i];
      ((cf_i32 *)out_ptr)[0] = op == CF_MATH_OP_MEAN && len != 0 ? sum / (cf_i32)len : sum;
      return CF_OK;
    }
    default:
      return CF_ERR_UNSUPPORTED;
  }
}

static cf_status cf_math_matmul_cpu(cf_math_dtype dtype, void *out_ptr, const void *a_ptr, const void *b_ptr, cf_usize m, cf_usize k, cf_usize n)
{
  switch(dtype)
  {
    case CF_MATH_DTYPE_F32:
    {
      float *out = (float *)out_ptr;
      const float *a = (const float *)a_ptr;
      const float *b = (const float *)b_ptr;
      for(cf_usize i = 0; i < m; ++i)
      {
        for(cf_usize j = 0; j < n; ++j)
        {
          float sum = 0.0f;
          for(cf_usize p = 0; p < k; ++p) sum += a[i * k + p] * b[p * n + j];
          out[i * n + j] = sum;
        }
      }
      return CF_OK;
    }
    case CF_MATH_DTYPE_F64:
    {
      double *out = (double *)out_ptr;
      const double *a = (const double *)a_ptr;
      const double *b = (const double *)b_ptr;
      for(cf_usize i = 0; i < m; ++i)
      {
        for(cf_usize j = 0; j < n; ++j)
        {
          double sum = 0.0;
          for(cf_usize p = 0; p < k; ++p) sum += a[i * k + p] * b[p * n + j];
          out[i * n + j] = sum;
        }
      }
      return CF_OK;
    }
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
  if(stream != CF_NULL)
    return cudaStreamSynchronize(stream) == cudaSuccess ? CF_OK : CF_ERR_CUDA_SYNC;
  return cudaDeviceSynchronize() == cudaSuccess ? CF_OK : CF_ERR_CUDA_SYNC;
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
  if(stream != CF_NULL) return cudaStreamSynchronize(stream) == cudaSuccess ? CF_OK : CF_ERR_CUDA_SYNC;
  return cudaDeviceSynchronize() == cudaSuccess ? CF_OK : CF_ERR_CUDA_SYNC;
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
  if(stream != CF_NULL) return cudaStreamSynchronize(stream) == cudaSuccess ? CF_OK : CF_ERR_CUDA_SYNC;
  return cudaDeviceSynchronize() == cudaSuccess ? CF_OK : CF_ERR_CUDA_SYNC;
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
    if(stream != CF_NULL) return cudaStreamSynchronize(stream) == cudaSuccess ? CF_OK : CF_ERR_CUDA_SYNC;
    return cudaDeviceSynchronize() == cudaSuccess ? CF_OK : CF_ERR_CUDA_SYNC;
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
  if(stream != CF_NULL) return cudaStreamSynchronize(stream) == cudaSuccess ? CF_OK : CF_ERR_CUDA_SYNC;
  return cudaDeviceSynchronize() == cudaSuccess ? CF_OK : CF_ERR_CUDA_SYNC;
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
      return handler->cuda_ctx->stream != CF_NULL && cudaStreamSynchronize(handler->cuda_ctx->stream) != cudaSuccess ? CF_ERR_CUDA_SYNC : CF_OK;
    }
    case CF_MATH_DTYPE_F64:
    {
      const double alpha = 1.0;
      const double beta = 0.0;
      cublasStatus_t st = cublasDgemm(handler->cuda_ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N, (int)n, (int)m, (int)k, &alpha, (const double *)b_ptr, (int)n, (const double *)a_ptr, (int)k, &beta, (double *)out_ptr, (int)n);
      if(st != CUBLAS_STATUS_SUCCESS) return CF_ERR_CUDA;
      return handler->cuda_ctx->stream != CF_NULL && cudaStreamSynchronize(handler->cuda_ctx->stream) != cudaSuccess ? CF_ERR_CUDA_SYNC : CF_OK;
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
  if(out->metadata->layout != CF_MATH_LAYOUT_ROW_MAJOR || a->metadata->layout != CF_MATH_LAYOUT_ROW_MAJOR || b->metadata->layout != CF_MATH_LAYOUT_ROW_MAJOR) return CF_ERR_UNSUPPORTED;

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

#if defined(CF_CUDA_AVAILABLE)
static cf_status cf_math_copy_sync(const cf_math_handle_t *handler)
{
  if(handler != CF_NULL && handler->cuda_ctx != CF_NULL && handler->cuda_ctx->stream != CF_NULL)
    return cudaStreamSynchronize(handler->cuda_ctx->stream) == cudaSuccess ? CF_OK : CF_ERR_CUDA_SYNC;
  return cudaDeviceSynchronize() == cudaSuccess ? CF_OK : CF_ERR_CUDA_SYNC;
}
#endif

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
  if(cudaMemcpy(dst, src, (size_t)bytes, cudaMemcpyDefault) != cudaSuccess)
    return CF_ERR_CUDA_COPY;
  return cf_math_copy_sync(x->handler);
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

  return cf_math_copy_bytes(src, host_data, ptr, bytes);
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
