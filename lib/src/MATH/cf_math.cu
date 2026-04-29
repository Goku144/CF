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

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define CF_MATH_ALIGN_BYTES 128U
#define CF_MATH_PI 3.14159265358979323846264338327950288
#define CF_MATH_SQRT_2_OVER_PI 0.79788456080286535587989211986876

typedef enum cf_math_binary_op
{
  CF_MATH_BIN_ADD = 0,
  CF_MATH_BIN_SUB,
  CF_MATH_BIN_MUL,
  CF_MATH_BIN_DIV,
} cf_math_binary_op;

typedef enum cf_math_unary_op
{
  CF_MATH_UN_ABS = 0,
  CF_MATH_UN_NEG,
  CF_MATH_UN_SQRT,
  CF_MATH_UN_RSQRT,
  CF_MATH_UN_EXP,
  CF_MATH_UN_LOG,
  CF_MATH_UN_SIGN,
} cf_math_unary_op;

typedef enum cf_math_activation_op
{
  CF_MATH_ACT_RELU = 0,
  CF_MATH_ACT_RELU_BWD,
  CF_MATH_ACT_LEAKY_RELU,
  CF_MATH_ACT_ELU,
  CF_MATH_ACT_SIGMOID,
  CF_MATH_ACT_SIGMOID_BWD,
  CF_MATH_ACT_TANH,
  CF_MATH_ACT_TANH_BWD,
  CF_MATH_ACT_GELU,
  CF_MATH_ACT_GELU_APPROX,
  CF_MATH_ACT_GELU_BWD,
  CF_MATH_ACT_SWISH,
  CF_MATH_ACT_SOFTPLUS,
  CF_MATH_ACT_MISH,
} cf_math_activation_op;

static cf_status cf_math_cuda_unavailable(void)
{
  return CF_ERR_UNSUPPORTED;
}

static cf_status cf_math_require_host_tensor(const cf_math *x)
{
  if(x == CF_NULL) return CF_ERR_NULL;
  return x->metadata.device == CF_DEVICE_CUDA ? cf_math_cuda_unavailable() : CF_OK;
}

cf_usize cf_math_dtype_size(cf_math_dtype dtype)
{
  switch(dtype)
  {
    case CF_DTYPE_F64: return sizeof(double);
    case CF_DTYPE_F32: return sizeof(float);
    case CF_DTYPE_F16: return 2;
    case CF_DTYPE_BF16: return 2;
    case CF_DTYPE_FP8E4M3: return 1;
    case CF_DTYPE_FP8E5M2: return 1;
    case CF_DTYPE_I32: return sizeof(cf_i32);
    case CF_DTYPE_I8: return sizeof(cf_i8);
    case CF_DTYPE_U8: return sizeof(cf_u8);
    case CF_DTYPE_BOOL: return sizeof(cf_u8);
    default: return 0;
  }
}

static cf_status cf_math_shape_len(const cf_usize dim[CF_MATH_HIGHEST_RANK], cf_usize rank, cf_usize *out_len)
{
  cf_usize len = 1;

  if(out_len == CF_NULL) return CF_ERR_NULL;
  if(rank > CF_MATH_HIGHEST_RANK) return CF_ERR_INVALID;
  if(rank != 0 && dim == CF_NULL) return CF_ERR_NULL;

  for(cf_usize i = 0; i < rank; i++)
  {
    if(dim[i] == 0) return CF_ERR_INVALID;
    if(len > (cf_usize)-1 / dim[i]) return CF_ERR_OVERFLOW;
    len *= dim[i];
  }

  *out_len = len;
  return CF_OK;
}

static cf_status cf_math_checked_bytes(cf_usize len, cf_math_dtype dtype, cf_usize *out_bytes)
{
  cf_usize elem_size = cf_math_dtype_size(dtype);

  if(out_bytes == CF_NULL) return CF_ERR_NULL;
  if(elem_size == 0) return CF_ERR_INVALID;
  if(len > (cf_usize)-1 / elem_size) return CF_ERR_OVERFLOW;

  *out_bytes = len * elem_size;
  return CF_OK;
}

static void cf_math_dense_strides(const cf_usize dim[CF_MATH_HIGHEST_RANK], cf_usize rank, cf_usize strides[CF_MATH_HIGHEST_RANK])
{
  cf_usize stride = 1;

  memset(strides, 0, sizeof(cf_usize) * CF_MATH_HIGHEST_RANK);
  for(cf_usize i = rank; i > 0; i--)
  {
    cf_usize axis = i - 1;
    strides[axis] = stride;
    stride *= dim[axis];
  }
}

static cf_math_shape cf_math_shape_kind(cf_usize rank)
{
  if(rank == 0) return CF_SHAPE_SCALAR;
  if(rank == 2) return CF_SHAPE_MATRIX;
  return CF_SHAPE_TENSOR;
}

static void cf_math_apply_shape(cf_math *out, const cf_usize dim[CF_MATH_HIGHEST_RANK], cf_usize rank, cf_usize len)
{
  memset(out->dim, 0, sizeof(out->dim));
  memset(out->metadata.strides, 0, sizeof(out->metadata.strides));
  out->rank = rank;
  out->metadata.len = len;
  out->metadata.batch = rank == 0 ? 1 : dim[0];
  out->metadata.shape = cf_math_shape_kind(rank);
  out->metadata.layout = CF_LAYOUT_ROW_MAJOR;

  for(cf_usize i = 0; i < rank; i++) out->dim[i] = dim[i];
  cf_math_dense_strides(out->dim, rank, out->metadata.strides);
}

static cf_status cf_math_aligned_alloc(void **out, cf_usize bytes)
{
  cf_usize alloc_bytes = bytes == 0 ? CF_MATH_ALIGN_BYTES : bytes;
  void *ptr = CF_NULL;

  if(out == CF_NULL) return CF_ERR_NULL;
  if(posix_memalign(&ptr, CF_MATH_ALIGN_BYTES, alloc_bytes) != 0) return CF_ERR_OOM;
  memset(ptr, 0, alloc_bytes);
  *out = ptr;
  return CF_OK;
}

static void cf_math_retain_storage(cf_math_storage *storage)
{
  if(storage != CF_NULL) storage->refcount++;
}

static void cf_math_reset_desc_cache(cf_math_desc_cache *cache)
{
  if(cache == CF_NULL) return;
  memset(cache, 0, sizeof(*cache));
}

static void cf_math_set_data_from_storage(cf_math *x)
{
  cf_usize elem_offset_bytes;

  if(x == CF_NULL || x->storage == CF_NULL || x->storage->data_ptr == CF_NULL)
  {
    if(x != CF_NULL) x->data = CF_NULL;
    return;
  }

  elem_offset_bytes = x->byte_offset;
  x->data = (void *)((char *)x->storage->data_ptr + elem_offset_bytes);
}

static cf_status cf_math_release_storage(cf_math_storage *storage, cf_math_cuda_context *ctx)
{
  if(storage == CF_NULL) return CF_OK;
  if(storage->refcount > 1)
  {
    storage->refcount--;
    return CF_OK;
  }

#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(storage->device == CF_DEVICE_CUDA || (storage->mem_flags & CF_MEM_MANAGED) != 0)
  {
    if(storage->data_ptr != CF_NULL)
    {
      #if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
      if(ctx != CF_NULL && ctx->stream != 0 && storage->device == CF_DEVICE_CUDA)
      {
        cudaError_t status = cudaFreeAsync(storage->data_ptr, ctx->stream);
        if(status != cudaSuccess) return CF_ERR_CUDA_MEMORY;
      }
      else
      #endif
      {
        cudaError_t status = cudaFree(storage->data_ptr);
        if(status != cudaSuccess) return CF_ERR_CUDA_MEMORY;
      }
    }
    free(storage);
    return CF_OK;
  }

  if((storage->mem_flags & CF_MEM_PINNED) != 0)
  {
    if(storage->data_ptr != CF_NULL)
    {
      cudaError_t status = cudaFreeHost(storage->data_ptr);
      if(status != cudaSuccess) return CF_ERR_CUDA_MEMORY;
    }
    free(storage);
    return CF_OK;
  }
#else
  CF_UNUSED(ctx);
#endif

  free(storage->data_ptr);
  free(storage);
  return CF_OK;
}

static cf_status cf_math_prepare_output(cf_math *out,
                                        const cf_usize dim[CF_MATH_HIGHEST_RANK],
                                        cf_usize rank,
                                        cf_math_dtype dtype,
                                        cf_math_device device,
                                        cf_math_mem_flags flags,
                                        cf_math_cuda_context *ctx)
{
  cf_usize len;
  cf_usize bytes;
  cf_status status;

  if(out == CF_NULL) return CF_ERR_NULL;
  status = cf_math_shape_len(dim, rank, &len);
  if(status != CF_OK) return status;
  status = cf_math_checked_bytes(len, dtype, &bytes);
  if(status != CF_OK) return status;

  if(out->storage == CF_NULL)
    return cf_math_alloc(out, dim, rank, dtype, device, flags, ctx);

  if(out->storage->device != device ||
     out->metadata.dtype != dtype ||
     out->storage->capacity < bytes)
  {
    status = cf_math_release_storage(out->storage, ctx);
    if(status != CF_OK) return status;
    memset(out, 0, sizeof(*out));
    return cf_math_alloc(out, dim, rank, dtype, device, flags, ctx);
  }

  cf_math_apply_shape(out, dim, rank, len);
  out->metadata.device = device;
  out->metadata.dtype = dtype;
  out->metadata.mem_flags = flags;
  cf_math_set_data_from_storage(out);
  return CF_OK;
}

static cf_status cf_math_prepare_like(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
  if(out == CF_NULL || x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_unavailable();
  if(out == x) return CF_OK;
  return cf_math_prepare_output(out, x->dim, x->rank, x->metadata.dtype, x->metadata.device, x->metadata.mem_flags, ctx);
}

static cf_status cf_math_prepare_scalar(cf_math *out, cf_math_dtype dtype, cf_math_device device, cf_math_cuda_context *ctx)
{
  if(out == CF_NULL) return CF_ERR_NULL;
  return cf_math_prepare_output(out, CF_NULL, 0, dtype, device, CF_MEM_DEFAULT, ctx);
}

static cf_usize cf_math_logical_offset(const cf_math *x, cf_usize linear)
{
  cf_usize offset = 0;

  for(cf_usize i = x->rank; i > 0; i--)
  {
    cf_usize axis = i - 1;
    cf_usize coord = linear % x->dim[axis];
    linear /= x->dim[axis];
    offset += coord * x->metadata.strides[axis];
  }

  return offset;
}

static double cf_math_load(const cf_math *x, cf_usize linear)
{
  cf_usize offset = cf_math_logical_offset(x, linear);
  const char *base = (const char *)x->data;

  switch(x->metadata.dtype)
  {
    case CF_DTYPE_F64: return ((const double *)base)[offset];
    case CF_DTYPE_F32: return (double)((const float *)base)[offset];
    case CF_DTYPE_I32: return (double)((const cf_i32 *)base)[offset];
    case CF_DTYPE_I8: return (double)((const cf_i8 *)base)[offset];
    case CF_DTYPE_U8:
    case CF_DTYPE_BOOL: return (double)((const cf_u8 *)base)[offset];
    default: return 0.0;
  }
}

static cf_i32 cf_math_load_i32(const cf_math *x, cf_usize linear)
{
  cf_usize offset = cf_math_logical_offset(x, linear);

  switch(x->metadata.dtype)
  {
    case CF_DTYPE_I32: return ((const cf_i32 *)x->data)[offset];
    case CF_DTYPE_I8: return (cf_i32)((const cf_i8 *)x->data)[offset];
    case CF_DTYPE_U8:
    case CF_DTYPE_BOOL: return (cf_i32)((const cf_u8 *)x->data)[offset];
    case CF_DTYPE_F32: return (cf_i32)((const float *)x->data)[offset];
    case CF_DTYPE_F64: return (cf_i32)((const double *)x->data)[offset];
    default: return 0;
  }
}

static void cf_math_store(cf_math *x, cf_usize linear, double value)
{
  cf_usize offset = cf_math_logical_offset(x, linear);
  char *base = (char *)x->data;

  switch(x->metadata.dtype)
  {
    case CF_DTYPE_F64: ((double *)base)[offset] = value; break;
    case CF_DTYPE_F32: ((float *)base)[offset] = (float)value; break;
    case CF_DTYPE_I32: ((cf_i32 *)base)[offset] = (cf_i32)value; break;
    case CF_DTYPE_I8: ((cf_i8 *)base)[offset] = (cf_i8)value; break;
    case CF_DTYPE_U8: ((cf_u8 *)base)[offset] = (cf_u8)value; break;
    case CF_DTYPE_BOOL: ((cf_u8 *)base)[offset] = value != 0.0 ? 1U : 0U; break;
    default: break;
  }
}

static double cf_math_binary_value(double a, double b, cf_math_binary_op op)
{
  switch(op)
  {
    case CF_MATH_BIN_ADD: return a + b;
    case CF_MATH_BIN_SUB: return a - b;
    case CF_MATH_BIN_MUL: return a * b;
    case CF_MATH_BIN_DIV: return a / b;
    default: return 0.0;
  }
}

static double cf_math_unary_value(double x, cf_math_unary_op op)
{
  switch(op)
  {
    case CF_MATH_UN_ABS: return fabs(x);
    case CF_MATH_UN_NEG: return -x;
    case CF_MATH_UN_SQRT: return sqrt(x);
    case CF_MATH_UN_RSQRT: return 1.0 / sqrt(x);
    case CF_MATH_UN_EXP: return exp(x);
    case CF_MATH_UN_LOG: return log(x);
    case CF_MATH_UN_SIGN: return (x > 0.0) - (x < 0.0);
    default: return 0.0;
  }
}

#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
#define CF_MATH_CUDA_THREADS 256U

static cudaStream_t cf_math_cuda_stream(cf_math_cuda_context *ctx)
{
  return ctx == CF_NULL ? 0 : ctx->stream;
}

static cf_status cf_math_cuda_launch_status(void)
{
  cudaError_t status = cudaPeekAtLastError();
  return status == cudaSuccess ? CF_OK : CF_ERR_CUDA_LAUNCH;
}

static cf_status cf_math_cuda_prepare_like(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
  if(out == CF_NULL || x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device != CF_DEVICE_CUDA) return cf_math_cuda_unavailable();
  if(out == x) return CF_OK;
  return cf_math_prepare_output(out, x->dim, x->rank, x->metadata.dtype, CF_DEVICE_CUDA, x->metadata.mem_flags, ctx);
}

static cf_bool cf_math_cuda_same_shape_dtype(const cf_math *x, const cf_math *y)
{
  if(x == CF_NULL || y == CF_NULL) return CF_FALSE;
  if(x->metadata.len != y->metadata.len) return CF_FALSE;
  if(x->metadata.dtype != y->metadata.dtype) return CF_FALSE;
  if(x->rank != y->rank) return CF_FALSE;
  for(cf_usize i = 0; i < x->rank; i++)
    if(x->dim[i] != y->dim[i]) return CF_FALSE;
  return CF_TRUE;
}

static cf_bool cf_math_cuda_supported_float(cf_math_dtype dtype)
{
  return dtype == CF_DTYPE_F64 || dtype == CF_DTYPE_F32;
}

static unsigned int cf_math_cuda_blocks(cf_usize len)
{
  return (unsigned int)((len + CF_MATH_CUDA_THREADS - 1U) / CF_MATH_CUDA_THREADS);
}

static __device__ double cf_math_cuda_binary_value(double a, double b, int op)
{
  switch(op)
  {
    case CF_MATH_BIN_ADD: return a + b;
    case CF_MATH_BIN_SUB: return a - b;
    case CF_MATH_BIN_MUL: return a * b;
    case CF_MATH_BIN_DIV: return a / b;
    default: return 0.0;
  }
}

static __device__ double cf_math_cuda_unary_value(double x, int op)
{
  switch(op)
  {
    case CF_MATH_UN_ABS: return fabs(x);
    case CF_MATH_UN_NEG: return -x;
    case CF_MATH_UN_SQRT: return sqrt(x);
    case CF_MATH_UN_RSQRT: return 1.0 / sqrt(x);
    case CF_MATH_UN_EXP: return exp(x);
    case CF_MATH_UN_LOG: return log(x);
    case CF_MATH_UN_SIGN: return (x > 0.0) - (x < 0.0);
    default: return 0.0;
  }
}

static __device__ double cf_math_cuda_activation_value(double x, double aux, double alpha, int op)
{
  switch(op)
  {
    case CF_MATH_ACT_RELU: return x > 0.0 ? x : 0.0;
    case CF_MATH_ACT_RELU_BWD: return aux > 0.0 ? x : 0.0;
    case CF_MATH_ACT_LEAKY_RELU: return x > 0.0 ? x : alpha * x;
    case CF_MATH_ACT_ELU: return x >= 0.0 ? x : alpha * (exp(x) - 1.0);
    case CF_MATH_ACT_SIGMOID: return 1.0 / (1.0 + exp(-x));
    case CF_MATH_ACT_SIGMOID_BWD: return x * aux * (1.0 - aux);
    case CF_MATH_ACT_TANH: return tanh(x);
    case CF_MATH_ACT_TANH_BWD: return x * (1.0 - aux * aux);
    case CF_MATH_ACT_GELU: return 0.5 * x * (1.0 + erf(x / sqrt(2.0)));
    case CF_MATH_ACT_GELU_APPROX:
      return 0.5 * x * (1.0 + tanh(CF_MATH_SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)));
    case CF_MATH_ACT_GELU_BWD:
    {
      double cdf = 0.5 * (1.0 + erf(aux / sqrt(2.0)));
      double pdf = exp(-0.5 * aux * aux) / sqrt(2.0 * CF_MATH_PI);
      return x * (cdf + aux * pdf);
    }
    case CF_MATH_ACT_SWISH: return x / (1.0 + exp(-alpha * x));
    case CF_MATH_ACT_SOFTPLUS: return x > 20.0 ? x : log1p(exp(x));
    case CF_MATH_ACT_MISH:
    {
      double sp = x > 20.0 ? x : log1p(exp(x));
      return x * tanh(sp);
    }
    default: return 0.0;
  }
}

__global__ void cf_math_cuda_fill_f64(double *out, double value, cf_usize len)
{
  cf_usize i = (cf_usize)blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len) out[i] = value;
}

__global__ void cf_math_cuda_fill_f32(float *out, float value, cf_usize len)
{
  cf_usize i = (cf_usize)blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len) out[i] = value;
}

__global__ void cf_math_cuda_eye_f64(double *out, cf_usize rows, cf_usize cols)
{
  cf_usize i = (cf_usize)blockIdx.x * blockDim.x + threadIdx.x;
  if(i < rows && i < cols) out[i * cols + i] = 1.0;
}

__global__ void cf_math_cuda_eye_f32(float *out, cf_usize rows, cf_usize cols)
{
  cf_usize i = (cf_usize)blockIdx.x * blockDim.x + threadIdx.x;
  if(i < rows && i < cols) out[i * cols + i] = 1.0f;
}

__global__ void cf_math_cuda_binary_f64(double *out, const double *x, const double *y, cf_usize len, int op)
{
  cf_usize i = (cf_usize)blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len) out[i] = cf_math_cuda_binary_value(x[i], y[i], op);
}

__global__ void cf_math_cuda_binary_f32(float *out, const float *x, const float *y, cf_usize len, int op)
{
  cf_usize i = (cf_usize)blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len) out[i] = (float)cf_math_cuda_binary_value((double)x[i], (double)y[i], op);
}

__global__ void cf_math_cuda_scalar_f64(double *out, const double *x, double c, cf_usize len, int op)
{
  cf_usize i = (cf_usize)blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len) out[i] = cf_math_cuda_binary_value(x[i], c, op);
}

__global__ void cf_math_cuda_scalar_f32(float *out, const float *x, float c, cf_usize len, int op)
{
  cf_usize i = (cf_usize)blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len) out[i] = (float)cf_math_cuda_binary_value((double)x[i], (double)c, op);
}

__global__ void cf_math_cuda_unary_f64(double *out, const double *x, cf_usize len, int op)
{
  cf_usize i = (cf_usize)blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len) out[i] = cf_math_cuda_unary_value(x[i], op);
}

__global__ void cf_math_cuda_unary_f32(float *out, const float *x, cf_usize len, int op)
{
  cf_usize i = (cf_usize)blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len) out[i] = (float)cf_math_cuda_unary_value((double)x[i], op);
}

__global__ void cf_math_cuda_pow_f64(double *out, const double *x, double n, cf_usize len)
{
  cf_usize i = (cf_usize)blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len) out[i] = pow(x[i], n);
}

__global__ void cf_math_cuda_pow_f32(float *out, const float *x, float n, cf_usize len)
{
  cf_usize i = (cf_usize)blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len) out[i] = powf(x[i], n);
}

__global__ void cf_math_cuda_clamp_f64(double *out, const double *x, double lo, double hi, cf_usize len)
{
  cf_usize i = (cf_usize)blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len)
  {
    double v = x[i];
    out[i] = v < lo ? lo : (v > hi ? hi : v);
  }
}

__global__ void cf_math_cuda_clamp_f32(float *out, const float *x, float lo, float hi, cf_usize len)
{
  cf_usize i = (cf_usize)blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len)
  {
    float v = x[i];
    out[i] = v < lo ? lo : (v > hi ? hi : v);
  }
}

__global__ void cf_math_cuda_activation_f64(double *out, const double *x, const double *aux, double alpha, cf_usize len, int op)
{
  cf_usize i = (cf_usize)blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len) out[i] = cf_math_cuda_activation_value(x[i], aux == NULL ? 0.0 : aux[i], alpha, op);
}

__global__ void cf_math_cuda_activation_f32(float *out, const float *x, const float *aux, float alpha, cf_usize len, int op)
{
  cf_usize i = (cf_usize)blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len) out[i] = (float)cf_math_cuda_activation_value((double)x[i], aux == NULL ? 0.0 : (double)aux[i], (double)alpha, op);
}

__global__ void cf_math_cuda_bias_add_f64(double *out, const double *b, cf_usize rows, cf_usize cols)
{
  cf_usize i = (cf_usize)blockIdx.x * blockDim.x + threadIdx.x;
  if(i < rows * cols) out[i] += b[i % cols];
}

__global__ void cf_math_cuda_bias_add_f32(float *out, const float *b, cf_usize rows, cf_usize cols)
{
  cf_usize i = (cf_usize)blockIdx.x * blockDim.x + threadIdx.x;
  if(i < rows * cols) out[i] += b[i % cols];
}

static cf_status cf_math_cuda_fill_float(cf_math *out, double value, cf_math_cuda_context *ctx)
{
  cudaStream_t stream;
  unsigned int blocks;

  if(out == CF_NULL) return CF_ERR_NULL;
  if(out->metadata.device != CF_DEVICE_CUDA) return cf_math_cuda_unavailable();
  if(!cf_math_cuda_supported_float(out->metadata.dtype)) return CF_ERR_UNSUPPORTED;

  stream = cf_math_cuda_stream(ctx);
  blocks = cf_math_cuda_blocks(out->metadata.len);
  if(out->metadata.dtype == CF_DTYPE_F64)
    cf_math_cuda_fill_f64<<<blocks, CF_MATH_CUDA_THREADS, 0, stream>>>((double *)out->data, value, out->metadata.len);
  else
    cf_math_cuda_fill_f32<<<blocks, CF_MATH_CUDA_THREADS, 0, stream>>>((float *)out->data, (float)value, out->metadata.len);

  return cf_math_cuda_launch_status();
}

static cf_status cf_math_cuda_binary(cf_math *out, const cf_math *x, const cf_math *y, cf_math_binary_op op, cf_math_cuda_context *ctx)
{
  cf_status status;
  cudaStream_t stream;
  unsigned int blocks;

  if(out == CF_NULL || x == CF_NULL || y == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device != CF_DEVICE_CUDA || y->metadata.device != CF_DEVICE_CUDA) return cf_math_cuda_unavailable();
  if(!cf_math_cuda_same_shape_dtype(x, y)) return CF_ERR_INVALID;
  if(!cf_math_cuda_supported_float(x->metadata.dtype)) return CF_ERR_UNSUPPORTED;

  status = cf_math_cuda_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;

  stream = cf_math_cuda_stream(ctx);
  blocks = cf_math_cuda_blocks(x->metadata.len);
  if(x->metadata.dtype == CF_DTYPE_F64)
    cf_math_cuda_binary_f64<<<blocks, CF_MATH_CUDA_THREADS, 0, stream>>>((double *)out->data, (const double *)x->data, (const double *)y->data, x->metadata.len, (int)op);
  else
    cf_math_cuda_binary_f32<<<blocks, CF_MATH_CUDA_THREADS, 0, stream>>>((float *)out->data, (const float *)x->data, (const float *)y->data, x->metadata.len, (int)op);

  return cf_math_cuda_launch_status();
}

static cf_status cf_math_cuda_scalar(cf_math *out, const cf_math *x, double c, cf_math_binary_op op, cf_math_cuda_context *ctx)
{
  cf_status status;
  cudaStream_t stream;
  unsigned int blocks;

  if(out == CF_NULL || x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device != CF_DEVICE_CUDA) return cf_math_cuda_unavailable();
  if(!cf_math_cuda_supported_float(x->metadata.dtype)) return CF_ERR_UNSUPPORTED;

  status = cf_math_cuda_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;

  stream = cf_math_cuda_stream(ctx);
  blocks = cf_math_cuda_blocks(x->metadata.len);
  if(x->metadata.dtype == CF_DTYPE_F64)
    cf_math_cuda_scalar_f64<<<blocks, CF_MATH_CUDA_THREADS, 0, stream>>>((double *)out->data, (const double *)x->data, c, x->metadata.len, (int)op);
  else
    cf_math_cuda_scalar_f32<<<blocks, CF_MATH_CUDA_THREADS, 0, stream>>>((float *)out->data, (const float *)x->data, (float)c, x->metadata.len, (int)op);

  return cf_math_cuda_launch_status();
}

static cf_status cf_math_cuda_unary(cf_math *out, const cf_math *x, cf_math_unary_op op, cf_math_cuda_context *ctx)
{
  cf_status status;
  cudaStream_t stream;
  unsigned int blocks;

  if(out == CF_NULL || x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device != CF_DEVICE_CUDA) return cf_math_cuda_unavailable();
  if(!cf_math_cuda_supported_float(x->metadata.dtype)) return CF_ERR_UNSUPPORTED;

  status = cf_math_cuda_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;

  stream = cf_math_cuda_stream(ctx);
  blocks = cf_math_cuda_blocks(x->metadata.len);
  if(x->metadata.dtype == CF_DTYPE_F64)
    cf_math_cuda_unary_f64<<<blocks, CF_MATH_CUDA_THREADS, 0, stream>>>((double *)out->data, (const double *)x->data, x->metadata.len, (int)op);
  else
    cf_math_cuda_unary_f32<<<blocks, CF_MATH_CUDA_THREADS, 0, stream>>>((float *)out->data, (const float *)x->data, x->metadata.len, (int)op);

  return cf_math_cuda_launch_status();
}

static cf_status cf_math_cuda_pow(cf_math *out, const cf_math *x, double n, cf_math_cuda_context *ctx)
{
  cf_status status;
  cudaStream_t stream;
  unsigned int blocks;

  if(out == CF_NULL || x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device != CF_DEVICE_CUDA) return cf_math_cuda_unavailable();
  if(!cf_math_cuda_supported_float(x->metadata.dtype)) return CF_ERR_UNSUPPORTED;

  status = cf_math_cuda_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;

  stream = cf_math_cuda_stream(ctx);
  blocks = cf_math_cuda_blocks(x->metadata.len);
  if(x->metadata.dtype == CF_DTYPE_F64)
    cf_math_cuda_pow_f64<<<blocks, CF_MATH_CUDA_THREADS, 0, stream>>>((double *)out->data, (const double *)x->data, n, x->metadata.len);
  else
    cf_math_cuda_pow_f32<<<blocks, CF_MATH_CUDA_THREADS, 0, stream>>>((float *)out->data, (const float *)x->data, (float)n, x->metadata.len);

  return cf_math_cuda_launch_status();
}

static cf_status cf_math_cuda_clamp(cf_math *out, const cf_math *x, double lo, double hi, cf_math_cuda_context *ctx)
{
  cf_status status;
  cudaStream_t stream;
  unsigned int blocks;

  if(out == CF_NULL || x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device != CF_DEVICE_CUDA) return cf_math_cuda_unavailable();
  if(!cf_math_cuda_supported_float(x->metadata.dtype)) return CF_ERR_UNSUPPORTED;

  status = cf_math_cuda_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;

  stream = cf_math_cuda_stream(ctx);
  blocks = cf_math_cuda_blocks(x->metadata.len);
  if(x->metadata.dtype == CF_DTYPE_F64)
    cf_math_cuda_clamp_f64<<<blocks, CF_MATH_CUDA_THREADS, 0, stream>>>((double *)out->data, (const double *)x->data, lo, hi, x->metadata.len);
  else
    cf_math_cuda_clamp_f32<<<blocks, CF_MATH_CUDA_THREADS, 0, stream>>>((float *)out->data, (const float *)x->data, (float)lo, (float)hi, x->metadata.len);

  return cf_math_cuda_launch_status();
}

static cf_status cf_math_cuda_activation(cf_math *out, const cf_math *x, const cf_math *aux, double alpha, cf_math_activation_op op, cf_math_cuda_context *ctx)
{
  cf_status status;
  cudaStream_t stream;
  unsigned int blocks;

  if(out == CF_NULL || x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device != CF_DEVICE_CUDA) return cf_math_cuda_unavailable();
  if(aux != CF_NULL && (aux->metadata.device != CF_DEVICE_CUDA || !cf_math_cuda_same_shape_dtype(x, aux))) return CF_ERR_INVALID;
  if(!cf_math_cuda_supported_float(x->metadata.dtype)) return CF_ERR_UNSUPPORTED;

  status = cf_math_cuda_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;

  stream = cf_math_cuda_stream(ctx);
  blocks = cf_math_cuda_blocks(x->metadata.len);
  if(x->metadata.dtype == CF_DTYPE_F64)
    cf_math_cuda_activation_f64<<<blocks, CF_MATH_CUDA_THREADS, 0, stream>>>((double *)out->data, (const double *)x->data, aux == CF_NULL ? NULL : (const double *)aux->data, alpha, x->metadata.len, (int)op);
  else
    cf_math_cuda_activation_f32<<<blocks, CF_MATH_CUDA_THREADS, 0, stream>>>((float *)out->data, (const float *)x->data, aux == CF_NULL ? NULL : (const float *)aux->data, (float)alpha, x->metadata.len, (int)op);

  return cf_math_cuda_launch_status();
}

#if defined(CF_MATH_HAVE_CUBLAS)
static cf_status cf_math_cublas_status(cublasStatus_t status)
{
  return status == CUBLAS_STATUS_SUCCESS ? CF_OK : CF_ERR_CUDA;
}

static cublasHandle_t cf_math_cublas_handle(cf_math_cuda_context *ctx)
{
  if(ctx == CF_NULL || ctx->cublas == 0) return 0;
  (void)cublasSetStream(ctx->cublas, cf_math_cuda_stream(ctx));
  return ctx->cublas;
}

static cf_status cf_math_cuda_dot_like(cf_math *out, const cf_math *x, const cf_math *y, cf_bool sqrt_result, cf_bool abs_sum, cf_math_cuda_context *ctx)
{
  cf_status status;
  cublasHandle_t handle;
  cublasPointerMode_t old_mode;

  if(out == CF_NULL || x == CF_NULL) return CF_ERR_NULL;
  if(y != CF_NULL && !cf_math_cuda_same_shape_dtype(x, y)) return CF_ERR_INVALID;
  if(x->metadata.device != CF_DEVICE_CUDA || (y != CF_NULL && y->metadata.device != CF_DEVICE_CUDA)) return cf_math_cuda_unavailable();
  if(!cf_math_cuda_supported_float(x->metadata.dtype)) return CF_ERR_UNSUPPORTED;

  handle = cf_math_cublas_handle(ctx);
  if(handle == 0) return CF_ERR_NULL;

  status = cf_math_prepare_scalar(out, x->metadata.dtype, CF_DEVICE_CUDA, ctx);
  if(status != CF_OK) return status;
  if(cublasGetPointerMode(handle, &old_mode) != CUBLAS_STATUS_SUCCESS) return CF_ERR_CUDA;
  if(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE) != CUBLAS_STATUS_SUCCESS) return CF_ERR_CUDA;

  if(x->metadata.dtype == CF_DTYPE_F64)
  {
    if(abs_sum)
      status = cf_math_cublas_status(cublasDasum(handle, (int)x->metadata.len, (const double *)x->data, 1, (double *)out->data));
    else if(y == CF_NULL || sqrt_result)
      status = cf_math_cublas_status(cublasDnrm2(handle, (int)x->metadata.len, (const double *)x->data, 1, (double *)out->data));
    else
      status = cf_math_cublas_status(cublasDdot(handle, (int)x->metadata.len, (const double *)x->data, 1, (const double *)y->data, 1, (double *)out->data));
  }
  else
  {
    if(abs_sum)
      status = cf_math_cublas_status(cublasSasum(handle, (int)x->metadata.len, (const float *)x->data, 1, (float *)out->data));
    else if(y == CF_NULL || sqrt_result)
      status = cf_math_cublas_status(cublasSnrm2(handle, (int)x->metadata.len, (const float *)x->data, 1, (float *)out->data));
    else
      status = cf_math_cublas_status(cublasSdot(handle, (int)x->metadata.len, (const float *)x->data, 1, (const float *)y->data, 1, (float *)out->data));
  }

  (void)cublasSetPointerMode(handle, old_mode);
  return status;
}

static cf_status cf_math_matmul_cuda_core(cf_math *out, const cf_math *a, const cf_math *b, cf_bool trans_a, cf_bool trans_b, cf_math_cuda_context *ctx)
{
  cf_usize a_rows;
  cf_usize a_cols;
  cf_usize b_rows;
  cf_usize b_cols;
  cf_usize m;
  cf_usize k;
  cf_usize n;
  cf_usize dim[CF_MATH_HIGHEST_RANK];
  cf_status status;
  cublasHandle_t handle;
  cublasOperation_t op_b;
  cublasOperation_t op_a;
  int lda;
  int ldb;
  int ldc;

  if(out == CF_NULL || a == CF_NULL || b == CF_NULL) return CF_ERR_NULL;
  if(a->metadata.device != CF_DEVICE_CUDA || b->metadata.device != CF_DEVICE_CUDA) return cf_math_cuda_unavailable();
  if(a->metadata.dtype != b->metadata.dtype) return CF_ERR_INVALID;
  if(a->rank < 2 || b->rank < 2) return CF_ERR_INVALID;
  if(!cf_math_cuda_supported_float(a->metadata.dtype)) return CF_ERR_UNSUPPORTED;
  if(out == a || out == b) return CF_ERR_INVALID;

  a_rows = a->dim[a->rank - 2];
  a_cols = a->dim[a->rank - 1];
  b_rows = b->dim[b->rank - 2];
  b_cols = b->dim[b->rank - 1];
  m = trans_a ? a_cols : a_rows;
  k = trans_a ? a_rows : a_cols;
  b_rows = trans_b ? b_cols : b_rows;
  b_cols = trans_b ? b->dim[b->rank - 2] : b_cols;
  n = b_cols;
  if(k != b_rows) return CF_ERR_INVALID;

  dim[0] = m;
  dim[1] = n;
  for(cf_usize i = 2; i < CF_MATH_HIGHEST_RANK; i++) dim[i] = 0;

  handle = cf_math_cublas_handle(ctx);
  if(handle == 0) return CF_ERR_NULL;

  status = cf_math_prepare_output(out, dim, 2, a->metadata.dtype, CF_DEVICE_CUDA, a->metadata.mem_flags, ctx);
  if(status != CF_OK) return status;

  op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  lda = (int)b->dim[b->rank - 1];
  ldb = (int)a->dim[a->rank - 1];
  ldc = (int)n;

  if(a->metadata.dtype == CF_DTYPE_F64)
  {
    const double alpha = 1.0;
    const double beta = 0.0;
    return cf_math_cublas_status(cublasDgemm(handle, op_b, op_a, (int)n, (int)m, (int)k, &alpha,
                                             (const double *)b->data, lda,
                                             (const double *)a->data, ldb,
                                             &beta, (double *)out->data, ldc));
  }
  else
  {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cf_math_cublas_status(cublasSgemm(handle, op_b, op_a, (int)n, (int)m, (int)k, &alpha,
                                             (const float *)b->data, lda,
                                             (const float *)a->data, ldb,
                                             &beta, (float *)out->data, ldc));
  }
}

static cf_status cf_math_matvec_cuda(cf_math *out, const cf_math *a, const cf_math *x, cf_math_cuda_context *ctx)
{
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {0};
  cf_status status;
  cublasHandle_t handle;

  if(out == CF_NULL || a == CF_NULL || x == CF_NULL) return CF_ERR_NULL;
  if(a->metadata.device != CF_DEVICE_CUDA || x->metadata.device != CF_DEVICE_CUDA) return cf_math_cuda_unavailable();
  if(a->metadata.dtype != x->metadata.dtype) return CF_ERR_INVALID;
  if(a->rank != 2 || x->metadata.len != a->dim[1]) return CF_ERR_INVALID;
  if(!cf_math_cuda_supported_float(a->metadata.dtype)) return CF_ERR_UNSUPPORTED;
  if(out == a || out == x) return CF_ERR_INVALID;

  handle = cf_math_cublas_handle(ctx);
  if(handle == 0) return CF_ERR_NULL;

  dim[0] = a->dim[0];
  status = cf_math_prepare_output(out, dim, 1, a->metadata.dtype, CF_DEVICE_CUDA, a->metadata.mem_flags, ctx);
  if(status != CF_OK) return status;

  if(a->metadata.dtype == CF_DTYPE_F64)
  {
    const double alpha = 1.0;
    const double beta = 0.0;
    return cf_math_cublas_status(cublasDgemv(handle, CUBLAS_OP_T, (int)a->dim[1], (int)a->dim[0], &alpha,
                                             (const double *)a->data, (int)a->dim[1],
                                             (const double *)x->data, 1, &beta, (double *)out->data, 1));
  }
  else
  {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cf_math_cublas_status(cublasSgemv(handle, CUBLAS_OP_T, (int)a->dim[1], (int)a->dim[0], &alpha,
                                             (const float *)a->data, (int)a->dim[1],
                                             (const float *)x->data, 1, &beta, (float *)out->data, 1));
  }
}

static cf_status cf_math_cuda_add_bias(cf_math *out, const cf_math *b, cf_usize rows, cf_usize cols, cf_math_cuda_context *ctx)
{
  cudaStream_t stream;
  unsigned int blocks;

  if(out == CF_NULL || b == CF_NULL) return CF_ERR_NULL;
  if(out->metadata.device != CF_DEVICE_CUDA || b->metadata.device != CF_DEVICE_CUDA) return cf_math_cuda_unavailable();
  if(out->metadata.dtype != b->metadata.dtype || b->metadata.len != cols) return CF_ERR_INVALID;
  if(!cf_math_cuda_supported_float(out->metadata.dtype)) return CF_ERR_UNSUPPORTED;

  stream = cf_math_cuda_stream(ctx);
  blocks = cf_math_cuda_blocks(rows * cols);
  if(out->metadata.dtype == CF_DTYPE_F64)
    cf_math_cuda_bias_add_f64<<<blocks, CF_MATH_CUDA_THREADS, 0, stream>>>((double *)out->data, (const double *)b->data, rows, cols);
  else
    cf_math_cuda_bias_add_f32<<<blocks, CF_MATH_CUDA_THREADS, 0, stream>>>((float *)out->data, (const float *)b->data, rows, cols);

  return cf_math_cuda_launch_status();
}
#endif

#endif

static cf_status cf_math_binary_cpu(cf_math *out, const cf_math *x, const cf_math *y, cf_math_binary_op op, cf_math_cuda_context *ctx)
{
  cf_status status = cf_math_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;
  if(y->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_unavailable();

  for(cf_usize i = 0; i < x->metadata.len; i++)
    cf_math_store(out, i, cf_math_binary_value(cf_math_load(x, i), cf_math_load(y, i), op));

  return CF_OK;
}

static cf_status cf_math_scalar_cpu(cf_math *out, const cf_math *x, double c, cf_math_binary_op op, cf_math_cuda_context *ctx)
{
  cf_status status = cf_math_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;

  for(cf_usize i = 0; i < x->metadata.len; i++)
    cf_math_store(out, i, cf_math_binary_value(cf_math_load(x, i), c, op));

  return CF_OK;
}

static cf_status cf_math_unary_cpu(cf_math *out, const cf_math *x, cf_math_unary_op op, cf_math_cuda_context *ctx)
{
  cf_status status = cf_math_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;

  for(cf_usize i = 0; i < x->metadata.len; i++)
    cf_math_store(out, i, cf_math_unary_value(cf_math_load(x, i), op));

  return CF_OK;
}

static cf_u64 cf_math_rng_next(cf_u64 *state)
{
  cf_u64 x = *state == 0 ? 0x9E3779B97F4A7C15ULL : *state;

  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *state = x;
  return x * 0x2545F4914F6CDD1DULL;
}

static double cf_math_rng_uniform01(cf_u64 *state)
{
  return (double)(cf_math_rng_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

static cf_usize cf_math_axis_outer(const cf_math *x, cf_usize axis)
{
  cf_usize outer = 1;
  for(cf_usize i = 0; i < axis; i++) outer *= x->dim[i];
  return outer;
}

static cf_usize cf_math_axis_inner(const cf_math *x, cf_usize axis)
{
  cf_usize inner = 1;
  for(cf_usize i = axis + 1; i < x->rank; i++) inner *= x->dim[i];
  return inner;
}

static cf_status cf_math_axis_out_shape(const cf_math *x, cf_usize axis, cf_usize dim[CF_MATH_HIGHEST_RANK], cf_usize *rank)
{
  cf_usize out_rank = x->rank == 0 ? 0 : x->rank - 1;

  if(dim == CF_NULL || rank == CF_NULL) return CF_ERR_NULL;
  memset(dim, 0, sizeof(cf_usize) * CF_MATH_HIGHEST_RANK);
  for(cf_usize i = 0, j = 0; i < x->rank; i++)
  {
    if(i == axis) continue;
    dim[j++] = x->dim[i];
  }
  *rank = out_rank;
  return CF_OK;
}

static cf_status cf_math_reduce_axis_cpu(cf_math *out, const cf_math *x, cf_usize axis, cf_math_binary_op op, cf_math_cuda_context *ctx)
{
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {0};
  cf_usize rank = 0;
  cf_usize outer;
  cf_usize inner;
  cf_usize axis_len;
  cf_status status;

  status = cf_math_require_host_tensor(x);
  if(status != CF_OK) return status;
  status = cf_math_axis_out_shape(x, axis, dim, &rank);
  if(status != CF_OK) return status;
  if(out->storage == CF_NULL)
  {
    status = cf_math_alloc(out, dim, rank, x->metadata.dtype, x->metadata.device, x->metadata.mem_flags, ctx);
    if(status != CF_OK) return status;
  }

  outer = cf_math_axis_outer(x, axis);
  inner = cf_math_axis_inner(x, axis);
  axis_len = x->dim[axis];

  for(cf_usize o = 0; o < outer; o++)
  {
    for(cf_usize in = 0; in < inner; in++)
    {
      double acc = cf_math_load(x, o * axis_len * inner + in);
      for(cf_usize a = 1; a < axis_len; a++)
      {
        double v = cf_math_load(x, (o * axis_len + a) * inner + in);
        if(op == CF_MATH_BIN_ADD) acc += v;
        else if(op == CF_MATH_BIN_MUL) acc *= v;
      }
      cf_math_store(out, o * inner + in, acc);
    }
  }

  return CF_OK;
}

static cf_status cf_math_matmul_cpu_core(cf_math *out, const cf_math *a, const cf_math *b, cf_bool trans_a, cf_bool trans_b, cf_math_cuda_context *ctx)
{
  cf_usize m = trans_a ? a->dim[a->rank - 1] : a->dim[a->rank - 2];
  cf_usize k = trans_a ? a->dim[a->rank - 2] : a->dim[a->rank - 1];
  cf_usize n = trans_b ? b->dim[b->rank - 2] : b->dim[b->rank - 1];
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {m, n};
  cf_status status;

  status = cf_math_require_host_tensor(a);
  if(status != CF_OK) return status;
  status = cf_math_require_host_tensor(b);
  if(status != CF_OK) return status;
  if(out->storage == CF_NULL)
  {
    status = cf_math_alloc(out, dim, 2, a->metadata.dtype, a->metadata.device, a->metadata.mem_flags, ctx);
    if(status != CF_OK) return status;
  }

  for(cf_usize row = 0; row < m; row++)
  {
    for(cf_usize col = 0; col < n; col++)
    {
      double acc = 0.0;
      for(cf_usize inner = 0; inner < k; inner++)
      {
        cf_usize ai = trans_a ? inner * m + row : row * k + inner;
        cf_usize bi = trans_b ? col * k + inner : inner * n + col;
        acc += cf_math_load(a, ai) * cf_math_load(b, bi);
      }
      cf_math_store(out, row * n + col, acc);
    }
  }

  return CF_OK;
}

static cf_status cf_math_scalar_loss(cf_math *loss, const cf_math *ref, double value, cf_math_cuda_context *ctx)
{
  cf_status status = cf_math_prepare_scalar(loss, ref->metadata.dtype, ref->metadata.device, ctx);
  if(status != CF_OK) return status;
  cf_math_store(loss, 0, value);
  return CF_OK;
}

static double cf_math_sparse_load_value(const cf_math_sparse *x, cf_usize index)
{
  switch(x->dtype)
  {
    case CF_DTYPE_F64: return ((const double *)x->values)[index];
    case CF_DTYPE_F32: return (double)((const float *)x->values)[index];
    case CF_DTYPE_I32: return (double)((const cf_i32 *)x->values)[index];
    case CF_DTYPE_I8: return (double)((const cf_i8 *)x->values)[index];
    case CF_DTYPE_U8:
    case CF_DTYPE_BOOL: return (double)((const cf_u8 *)x->values)[index];
    default: return 0.0;
  }
}

static void cf_math_sparse_store_value(cf_math_sparse *x, cf_usize index, double value)
{
  switch(x->dtype)
  {
    case CF_DTYPE_F64: ((double *)x->values)[index] = value; break;
    case CF_DTYPE_F32: ((float *)x->values)[index] = (float)value; break;
    case CF_DTYPE_I32: ((cf_i32 *)x->values)[index] = (cf_i32)value; break;
    case CF_DTYPE_I8: ((cf_i8 *)x->values)[index] = (cf_i8)value; break;
    case CF_DTYPE_U8: ((cf_u8 *)x->values)[index] = (cf_u8)value; break;
    case CF_DTYPE_BOOL: ((cf_u8 *)x->values)[index] = value != 0.0 ? 1U : 0U; break;
    default: break;
  }
}

static cf_usize cf_math_nchw_index(cf_usize n, cf_usize c, cf_usize h, cf_usize w, cf_usize channels, cf_usize height, cf_usize width)
{
  CF_UNUSED(height);
  return ((n * channels + c) * height + h) * width + w;
}

static cf_usize cf_math_channel_count(const cf_math *x)
{
  if(x->rank < 2) return x->metadata.len;
  if(x->metadata.layout == CF_LAYOUT_NHWC && x->rank >= 4) return x->dim[x->rank - 1];
  return x->dim[1];
}

static cf_usize cf_math_channel_of(const cf_math *x, cf_usize linear)
{
  cf_usize spatial = 1;
  if(x->rank < 2) return linear;
  if(x->metadata.layout == CF_LAYOUT_NHWC && x->rank >= 4) return linear % x->dim[x->rank - 1];
  for(cf_usize i = 2; i < x->rank; i++) spatial *= x->dim[i];
  return (linear / spatial) % x->dim[1];
}

static const char *cf_math_dtype_name(cf_math_dtype dtype)
{
  switch(dtype)
  {
    case CF_DTYPE_F64: return "f64";
    case CF_DTYPE_F32: return "f32";
    case CF_DTYPE_F16: return "f16";
    case CF_DTYPE_BF16: return "bf16";
    case CF_DTYPE_FP8E4M3: return "fp8e4m3";
    case CF_DTYPE_FP8E5M2: return "fp8e5m2";
    case CF_DTYPE_I32: return "i32";
    case CF_DTYPE_I8: return "i8";
    case CF_DTYPE_U8: return "u8";
    case CF_DTYPE_BOOL: return "bool";
    default: return "unknown";
  }
}

static const char *cf_math_device_name(cf_math_device device)
{
  switch(device)
  {
    case CF_DEVICE_CPU: return "cpu";
    case CF_DEVICE_CUDA: return "cuda";
    default: return "unknown";
  }
}

static void cf_math_print_value(const cf_math *x, cf_usize linear)
{
  double v = cf_math_load(x, linear);

  switch(x->metadata.dtype)
  {
    case CF_DTYPE_I32:
    case CF_DTYPE_I8:
    case CF_DTYPE_U8:
    case CF_DTYPE_BOOL:
      printf("%6lld", (long long)v);
      break;

    case CF_DTYPE_F64:
    case CF_DTYPE_F32:
      printf("%10.4f", v);
      break;

    default:
      printf("%10.4f", v);
      break;
  }
}

static void cf_math_print_recursive(const cf_math *x, cf_usize axis, cf_usize base)
{
  cf_usize stride_block = 1;

  if(axis >= x->rank || axis >= CF_MATH_HIGHEST_RANK)
    return;

  if(axis + 1 >= x->rank)
  {
    printf("[");
    for(cf_usize i = 0; i < x->dim[axis]; i++)
    {
      if(i != 0) printf(", ");
      cf_math_print_value(x, base + i);
    }
    printf("]");
    return;
  }

  for(cf_usize i = axis + 1; i < x->rank && i < CF_MATH_HIGHEST_RANK; i++)
    stride_block *= x->dim[i];

  printf("[\n");
  for(cf_usize i = 0; i < x->dim[axis]; i++)
  {
    for(cf_usize s = 0; s <= axis; s++) printf("  ");

    cf_math_print_recursive(x, axis + 1, base + i * stride_block);

    if(i + 1 < x->dim[axis]) printf(",");
    printf("\n");
  }

  for(cf_usize s = 0; s < axis; s++) printf("  ");
  printf("]");
}


cf_status cf_math_context_init(cf_math_cuda_context *ctx, int device_id)
{
  if(ctx == CF_NULL) return CF_ERR_NULL;
  memset(ctx, 0, sizeof(*ctx));
  ctx->device_id = device_id;

#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(cudaSetDevice(device_id) != cudaSuccess) return CF_ERR_CUDA_DEVICE;
  if(cudaStreamCreate(&ctx->stream) != cudaSuccess) return CF_ERR_CUDA_RUNTIME;
  if(cudaStreamCreate(&ctx->h2d_stream) != cudaSuccess) return CF_ERR_CUDA_RUNTIME;
  if(cudaStreamCreate(&ctx->d2h_stream) != cudaSuccess) return CF_ERR_CUDA_RUNTIME;
#if defined(CF_MATH_HAVE_CUBLAS)
  if(cublasCreate(&ctx->cublas) != CUBLAS_STATUS_SUCCESS) return CF_ERR_CUDA;
  (void)cublasSetStream(ctx->cublas, ctx->stream);
#endif
#if defined(CF_MATH_HAVE_CUBLASLT)
  if(cublasLtCreate(&ctx->cublasLt) != CUBLAS_STATUS_SUCCESS) return CF_ERR_CUDA;
#endif
#if defined(CF_MATH_HAVE_CUDNN)
  if(cudnnCreate(&ctx->cudnn) != CUDNN_STATUS_SUCCESS) return CF_ERR_CUDA;
  (void)cudnnSetStream(ctx->cudnn, ctx->stream);
#endif
#if defined(CF_MATH_HAVE_CUSPARSE)
  if(cusparseCreate(&ctx->cusparse) != CUSPARSE_STATUS_SUCCESS) return CF_ERR_CUDA;
#endif
#if defined(CF_MATH_HAVE_CUSOLVER)
  if(cusolverDnCreate(&ctx->cusolver) != CUSOLVER_STATUS_SUCCESS) return CF_ERR_CUDA;
#endif
#if defined(CF_MATH_HAVE_CURAND)
  if(curandCreateGenerator(&ctx->curand, CURAND_RNG_PSEUDO_PHILOX4_32_10) != CURAND_STATUS_SUCCESS) return CF_ERR_CUDA;
  (void)curandSetStream(ctx->curand, ctx->stream);
#endif
  return CF_OK;
#else
  return cf_math_cuda_unavailable();
#endif
}

cf_status cf_math_context_destroy(cf_math_cuda_context *ctx)
{
  if(ctx == CF_NULL) return CF_ERR_NULL;

#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(ctx->workspace.ptr != CF_NULL) cudaFree(ctx->workspace.ptr);
#if defined(CF_MATH_HAVE_CURAND)
  if(ctx->curand != 0) curandDestroyGenerator(ctx->curand);
#endif
#if defined(CF_MATH_HAVE_CUSOLVER)
  if(ctx->cusolver != 0) cusolverDnDestroy(ctx->cusolver);
#endif
#if defined(CF_MATH_HAVE_CUSPARSE)
  if(ctx->cusparse != 0) cusparseDestroy(ctx->cusparse);
#endif
#if defined(CF_MATH_HAVE_CUDNN)
  if(ctx->cudnn != 0) cudnnDestroy(ctx->cudnn);
#endif
#if defined(CF_MATH_HAVE_CUBLASLT)
  if(ctx->cublasLt != 0) cublasLtDestroy(ctx->cublasLt);
#endif
#if defined(CF_MATH_HAVE_CUBLAS)
  if(ctx->cublas != 0) cublasDestroy(ctx->cublas);
#endif
  if(ctx->stream != 0) cudaStreamDestroy(ctx->stream);
  if(ctx->h2d_stream != 0) cudaStreamDestroy(ctx->h2d_stream);
  if(ctx->d2h_stream != 0) cudaStreamDestroy(ctx->d2h_stream);
#endif

  memset(ctx, 0, sizeof(*ctx));
  return CF_OK;
}

cf_status cf_math_workspace_reserve(cf_math_cuda_context *ctx, cf_usize bytes)
{
  if(ctx == CF_NULL) return CF_ERR_NULL;
  if(bytes <= ctx->workspace.size) return CF_OK;

#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  void *ptr = CF_NULL;
  if(cudaMalloc(&ptr, bytes) != cudaSuccess) return CF_ERR_CUDA_MEMORY;
  if(ctx->workspace.ptr != CF_NULL) cudaFree(ctx->workspace.ptr);
  ctx->workspace.ptr = ptr;
#else
  void *ptr = realloc(ctx->workspace.ptr, bytes);
  if(ptr == CF_NULL) return CF_ERR_OOM;
  ctx->workspace.ptr = ptr;
#endif
  ctx->workspace.size = bytes;
  if(bytes > ctx->workspace.high_water) ctx->workspace.high_water = bytes;
  return CF_OK;
}

cf_status cf_math_alloc(cf_math *out, const cf_usize dim[CF_MATH_HIGHEST_RANK], cf_usize rank, cf_math_dtype dtype, cf_math_device device, cf_math_mem_flags flags, cf_math_cuda_context *ctx)
{
  cf_usize len;
  cf_usize bytes;
  cf_status status;
  cf_math_storage *storage;
  void *ptr = CF_NULL;

  if(out == CF_NULL) return CF_ERR_NULL;
  status = cf_math_shape_len(dim, rank, &len);
  if(status != CF_OK) return status;
  status = cf_math_checked_bytes(len, dtype, &bytes);
  if(status != CF_OK) return status;

#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(device == CF_DEVICE_CUDA)
  {
    int device_id = ctx == CF_NULL ? 0 : ctx->device_id;
    if(cudaSetDevice(device_id) != cudaSuccess) return CF_ERR_CUDA_DEVICE;
    #if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
    if(ctx != CF_NULL && ctx->stream != 0)
    {
      if(cudaMallocAsync(&ptr, bytes, ctx->stream) != cudaSuccess) return CF_ERR_CUDA_MEMORY;
    }
    else
    #endif
    {
      if(cudaMalloc(&ptr, bytes) != cudaSuccess) return CF_ERR_CUDA_MEMORY;
    }
    if(cudaMemset(ptr, 0, bytes) != cudaSuccess) return CF_ERR_CUDA_MEMORY;
  }
  else if((flags & CF_MEM_MANAGED) != 0)
  {
    if(cudaMallocManaged(&ptr, bytes) != cudaSuccess) return CF_ERR_CUDA_MEMORY;
    memset(ptr, 0, bytes);
  }
  else if((flags & CF_MEM_PINNED) != 0)
  {
    if(cudaMallocHost(&ptr, bytes) != cudaSuccess) return CF_ERR_CUDA_MEMORY;
    memset(ptr, 0, bytes);
  }
  else
#else
  CF_UNUSED(ctx);
  if(device == CF_DEVICE_CUDA) return cf_math_cuda_unavailable();
#endif
  {
    status = cf_math_aligned_alloc(&ptr, bytes);
    if(status != CF_OK) return status;
  }

  storage = (cf_math_storage *)calloc(1, sizeof(*storage));
  if(storage == CF_NULL)
  {
    free(ptr);
    return CF_ERR_OOM;
  }

  memset(out, 0, sizeof(*out));
  storage->data_ptr = ptr;
  storage->capacity = bytes;
  storage->refcount = 1;
  storage->mem_flags = flags;
  storage->device_id = ctx == CF_NULL ? 0 : ctx->device_id;
  storage->device = device;
  out->storage = storage;
  out->metadata.dtype = dtype;
  out->metadata.device = device;
  out->metadata.mem_flags = flags;
  if(ctx != CF_NULL) out->metadata.ctx = *ctx;
  cf_math_apply_shape(out, dim, rank, len);
  cf_math_set_data_from_storage(out);
  cf_math_reset_desc_cache(&out->desc_cache);
  return CF_OK;
}

cf_status cf_math_free(cf_math *x, cf_math_cuda_context *ctx)
{
  cf_status status;

  if(x == CF_NULL) return CF_ERR_NULL;
  status = cf_math_release_storage(x->storage, ctx);
  memset(x, 0, sizeof(*x));
  return status;
}

cf_status cf_math_alloc_pinned(cf_math *out, const cf_usize dim[CF_MATH_HIGHEST_RANK], cf_usize rank, cf_math_dtype dtype, cf_math_cuda_context *ctx)
{
  return cf_math_alloc(out, dim, rank, dtype, CF_DEVICE_CPU, (cf_math_mem_flags)(CF_MEM_PINNED | CF_MEM_ALIGNED_128), ctx);
}

cf_status cf_math_alloc_managed(cf_math *out, const cf_usize dim[CF_MATH_HIGHEST_RANK], cf_usize rank, cf_math_dtype dtype, cf_math_cuda_context *ctx)
{
  return cf_math_alloc(out, dim, rank, dtype, CF_DEVICE_CPU, (cf_math_mem_flags)(CF_MEM_MANAGED | CF_MEM_ALIGNED_128), ctx);
}

cf_status cf_math_view(cf_math *out, const cf_math *x, cf_usize offset_elems, const cf_usize dim[CF_MATH_HIGHEST_RANK], cf_usize rank)
{
  cf_usize len;
  cf_status status;

  if(out == CF_NULL || x == CF_NULL) return CF_ERR_NULL;
  status = cf_math_shape_len(dim, rank, &len);
  if(status != CF_OK) return status;

  *out = *x;
  cf_math_retain_storage(out->storage);
  out->byte_offset = x->byte_offset + offset_elems * cf_math_dtype_size(x->metadata.dtype);
  cf_math_apply_shape(out, dim, rank, len);
  out->metadata.dtype = x->metadata.dtype;
  out->metadata.device = x->metadata.device;
  out->metadata.mem_flags = x->metadata.mem_flags;
  cf_math_set_data_from_storage(out);
  out->desc_cache.valid = 0;
  return CF_OK;
}

cf_status cf_math_clone(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
  cf_status status;
  cf_usize bytes;

  if(out == CF_NULL || x == CF_NULL) return CF_ERR_NULL;
  if(out == x) return CF_OK;
  status = cf_math_prepare_output(out, x->dim, x->rank, x->metadata.dtype, x->metadata.device, x->metadata.mem_flags, ctx);
  if(status != CF_OK) return status;
  status = cf_math_checked_bytes(x->metadata.len, x->metadata.dtype, &bytes);
  if(status != CF_OK) return status;

#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x->metadata.device == CF_DEVICE_CUDA)
  {
    cudaStream_t stream = ctx == CF_NULL ? 0 : ctx->stream;
    cudaError_t copy_status = cudaMemcpyAsync(out->data, x->data, bytes, cudaMemcpyDeviceToDevice, stream);
    return copy_status == cudaSuccess ? CF_OK : CF_ERR_CUDA_COPY;
  }
#else
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_unavailable();
#endif

  memcpy(out->data, x->data, bytes);
  return CF_OK;
}

cf_status cf_math_contiguous(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
  return cf_math_clone(out, x, ctx);
}

cf_status cf_math_to_device(cf_math *out, const cf_math *x, int device_id, cf_math_cuda_context *ctx)
{
  cf_usize bytes;
  cf_status status;
  cf_math_cuda_context local_ctx;

  memset(&local_ctx, 0, sizeof(local_ctx));

  if(out == CF_NULL || x == CF_NULL) return CF_ERR_NULL;
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(ctx != CF_NULL) local_ctx = *ctx;
  local_ctx.device_id = device_id;
  if(out == x && x->metadata.device == CF_DEVICE_CUDA) return CF_OK;
  if(out == x) return CF_ERR_INVALID;
  status = cf_math_prepare_output(out, x->dim, x->rank, x->metadata.dtype, CF_DEVICE_CUDA, (cf_math_mem_flags)(x->metadata.mem_flags | CF_MEM_ALIGNED_128), &local_ctx);
  if(status != CF_OK) return status;
  status = cf_math_checked_bytes(x->metadata.len, x->metadata.dtype, &bytes);
  if(status != CF_OK) return status;
  if(cudaMemcpyAsync(out->data, x->data, bytes, x->metadata.device == CF_DEVICE_CUDA ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, local_ctx.stream) != cudaSuccess)
    return CF_ERR_CUDA_COPY;
  return CF_OK;
#else
  CF_UNUSED(bytes);
  CF_UNUSED(status);
  CF_UNUSED(local_ctx);
  CF_UNUSED(device_id);
  CF_UNUSED(ctx);
  return cf_math_cuda_unavailable();
#endif
}

cf_status cf_math_to_host(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
  cf_usize bytes;
  cf_status status;

  if(out == CF_NULL || x == CF_NULL) return CF_ERR_NULL;
  if(out == x && x->metadata.device == CF_DEVICE_CPU) return CF_OK;
  if(out == x) return CF_ERR_INVALID;
  status = cf_math_prepare_output(out, x->dim, x->rank, x->metadata.dtype, CF_DEVICE_CPU, (cf_math_mem_flags)(x->metadata.mem_flags & ~CF_MEM_POOLED), ctx);
  if(status != CF_OK) return status;
  status = cf_math_checked_bytes(x->metadata.len, x->metadata.dtype, &bytes);
  if(status != CF_OK) return status;

#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x->metadata.device == CF_DEVICE_CUDA)
  {
    cudaStream_t stream = ctx == CF_NULL ? 0 : ctx->d2h_stream;
    if(cudaMemcpyAsync(out->data, x->data, bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) return CF_ERR_CUDA_COPY;
    return cudaStreamSynchronize(stream) == cudaSuccess ? CF_OK : CF_ERR_CUDA_SYNC;
  }
#else
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_unavailable();
#endif

  memcpy(out->data, x->data, bytes);
  return CF_OK;
}

cf_status cf_math_fill(cf_math *out, double value, cf_math_cuda_context *ctx)
{
  if(out == CF_NULL) return CF_ERR_NULL;
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(out->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_fill_float(out, value, ctx);
#else
  CF_UNUSED(ctx);
  if(out->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_unavailable();
#endif
  for(cf_usize i = 0; i < out->metadata.len; i++) cf_math_store(out, i, value);
  return CF_OK;
}

cf_status cf_math_zeros(cf_math *out, cf_math_cuda_context *ctx)
{
  cf_usize bytes;
  cf_status status;

  if(out == CF_NULL) return CF_ERR_NULL;
  status = cf_math_checked_bytes(out->metadata.len, out->metadata.dtype, &bytes);
  if(status != CF_OK) return status;
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(out->metadata.device == CF_DEVICE_CUDA)
  {
    cudaStream_t stream = ctx == CF_NULL ? 0 : ctx->stream;
    return cudaMemsetAsync(out->data, 0, bytes, stream) == cudaSuccess ? CF_OK : CF_ERR_CUDA_MEMORY;
  }
#else
  CF_UNUSED(ctx);
  if(out->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_unavailable();
#endif
  memset(out->data, 0, bytes);
  return CF_OK;
}

cf_status cf_math_ones(cf_math *out, cf_math_cuda_context *ctx)
{
  return cf_math_fill(out, 1.0, ctx);
}

cf_status cf_math_rand_uniform(cf_math *out, double lo, double hi, cf_u64 seed, cf_math_cuda_context *ctx)
{
  CF_UNUSED(ctx);
  if(out == CF_NULL) return CF_ERR_NULL;
  if(out->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_unavailable();

  for(cf_usize i = 0; i < out->metadata.len; i++)
    cf_math_store(out, i, lo + (hi - lo) * cf_math_rng_uniform01(&seed));
  return CF_OK;
}

cf_status cf_math_rand_normal(cf_math *out, double mean, double stddev, cf_u64 seed, cf_math_cuda_context *ctx)
{
  CF_UNUSED(ctx);
  if(out == CF_NULL) return CF_ERR_NULL;
  if(out->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_unavailable();

  for(cf_usize i = 0; i < out->metadata.len; i += 2)
  {
    double u1 = cf_math_rng_uniform01(&seed);
    double u2 = cf_math_rng_uniform01(&seed);
    double r = sqrt(-2.0 * log(u1 < DBL_MIN ? DBL_MIN : u1));
    double theta = 2.0 * CF_MATH_PI * u2;

    cf_math_store(out, i, mean + stddev * r * cos(theta));
    if(i + 1 < out->metadata.len) cf_math_store(out, i + 1, mean + stddev * r * sin(theta));
  }

  return CF_OK;
}

cf_status cf_math_rand_bernoulli(cf_math *out, double p, cf_u64 seed, cf_math_cuda_context *ctx)
{
  CF_UNUSED(ctx);
  if(out == CF_NULL) return CF_ERR_NULL;
  if(out->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_unavailable();

  for(cf_usize i = 0; i < out->metadata.len; i++)
    cf_math_store(out, i, cf_math_rng_uniform01(&seed) < p ? 1.0 : 0.0);
  return CF_OK;
}

cf_status cf_math_init_xavier_uniform(cf_math *out, cf_usize fan_in, cf_usize fan_out, cf_u64 seed, cf_math_cuda_context *ctx)
{
  double limit = sqrt(6.0 / (double)(fan_in + fan_out));
  return cf_math_rand_uniform(out, -limit, limit, seed, ctx);
}

cf_status cf_math_init_xavier_normal(cf_math *out, cf_usize fan_in, cf_usize fan_out, cf_u64 seed, cf_math_cuda_context *ctx)
{
  return cf_math_rand_normal(out, 0.0, sqrt(2.0 / (double)(fan_in + fan_out)), seed, ctx);
}

cf_status cf_math_init_kaiming_normal(cf_math *out, cf_usize fan_in, cf_u64 seed, cf_math_cuda_context *ctx)
{
  return cf_math_rand_normal(out, 0.0, sqrt(2.0 / (double)fan_in), seed, ctx);
}

cf_status cf_math_init_kaiming_uniform(cf_math *out, cf_usize fan_in, cf_u64 seed, cf_math_cuda_context *ctx)
{
  double limit = sqrt(6.0 / (double)fan_in);
  return cf_math_rand_uniform(out, -limit, limit, seed, ctx);
}

cf_status cf_math_init_orthogonal(cf_math *out, cf_u64 seed, cf_math_cuda_context *ctx)
{
  cf_usize rows = out->dim[0];
  cf_usize cols = out->dim[1];
  cf_status status = cf_math_rand_normal(out, 0.0, 1.0, seed, ctx);
  if(status != CF_OK) return status;

  if(rows >= cols)
  {
    for(cf_usize j = 0; j < cols; j++)
    {
      double norm = 0.0;
      for(cf_usize k = 0; k < j; k++)
      {
        double dot = 0.0;
        for(cf_usize i = 0; i < rows; i++) dot += cf_math_load(out, i * cols + j) * cf_math_load(out, i * cols + k);
        for(cf_usize i = 0; i < rows; i++)
          cf_math_store(out, i * cols + j, cf_math_load(out, i * cols + j) - dot * cf_math_load(out, i * cols + k));
      }
      for(cf_usize i = 0; i < rows; i++)
      {
        double v = cf_math_load(out, i * cols + j);
        norm += v * v;
      }
      norm = 1.0 / sqrt(norm);
      for(cf_usize i = 0; i < rows; i++) cf_math_store(out, i * cols + j, cf_math_load(out, i * cols + j) * norm);
    }
  }
  else
  {
    for(cf_usize i = 0; i < rows; i++)
    {
      double norm = 0.0;
      for(cf_usize k = 0; k < i; k++)
      {
        double dot = 0.0;
        for(cf_usize j = 0; j < cols; j++) dot += cf_math_load(out, i * cols + j) * cf_math_load(out, k * cols + j);
        for(cf_usize j = 0; j < cols; j++)
          cf_math_store(out, i * cols + j, cf_math_load(out, i * cols + j) - dot * cf_math_load(out, k * cols + j));
      }
      for(cf_usize j = 0; j < cols; j++)
      {
        double v = cf_math_load(out, i * cols + j);
        norm += v * v;
      }
      norm = 1.0 / sqrt(norm);
      for(cf_usize j = 0; j < cols; j++) cf_math_store(out, i * cols + j, cf_math_load(out, i * cols + j) * norm);
    }
  }

  return CF_OK;
}

cf_status cf_math_init_eye(cf_math *out, cf_math_cuda_context *ctx)
{
  if(out == CF_NULL) return CF_ERR_NULL;
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(out->metadata.device == CF_DEVICE_CUDA)
  {
    cf_status status;
    cudaStream_t stream;
    unsigned int blocks;

    if(out->rank != 2) return CF_ERR_INVALID;
    if(!cf_math_cuda_supported_float(out->metadata.dtype)) return CF_ERR_UNSUPPORTED;
    status = cf_math_zeros(out, ctx);
    if(status != CF_OK) return status;

    stream = cf_math_cuda_stream(ctx);
    blocks = cf_math_cuda_blocks(out->dim[0] < out->dim[1] ? out->dim[0] : out->dim[1]);
    if(out->metadata.dtype == CF_DTYPE_F64)
      cf_math_cuda_eye_f64<<<blocks, CF_MATH_CUDA_THREADS, 0, stream>>>((double *)out->data, out->dim[0], out->dim[1]);
    else
      cf_math_cuda_eye_f32<<<blocks, CF_MATH_CUDA_THREADS, 0, stream>>>((float *)out->data, out->dim[0], out->dim[1]);
    return cf_math_cuda_launch_status();
  }
#else
  CF_UNUSED(ctx);
  if(out->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_unavailable();
#endif
  cf_math_zeros(out, ctx);
  for(cf_usize i = 0; i < out->dim[0] && i < out->dim[1]; i++)
    cf_math_store(out, i * out->dim[1] + i, 1.0);
  return CF_OK;
}

cf_status cf_math_add(cf_math *out, const cf_math *x, const cf_math *y, cf_math_cuda_context *ctx)
{
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL || y == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA || y->metadata.device == CF_DEVICE_CUDA)
    return cf_math_cuda_binary(out, x, y, CF_MATH_BIN_ADD, ctx);
#endif
  return cf_math_binary_cpu(out, x, y, CF_MATH_BIN_ADD, ctx);
}

cf_status cf_math_add_scalar(cf_math *out, const cf_math *x, double c, cf_math_cuda_context *ctx)
{
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_scalar(out, x, c, CF_MATH_BIN_ADD, ctx);
#endif
  return cf_math_scalar_cpu(out, x, c, CF_MATH_BIN_ADD, ctx);
}

cf_status cf_math_sub(cf_math *out, const cf_math *x, const cf_math *y, cf_math_cuda_context *ctx)
{
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL || y == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA || y->metadata.device == CF_DEVICE_CUDA)
    return cf_math_cuda_binary(out, x, y, CF_MATH_BIN_SUB, ctx);
#endif
  return cf_math_binary_cpu(out, x, y, CF_MATH_BIN_SUB, ctx);
}

cf_status cf_math_mul(cf_math *out, const cf_math *x, const cf_math *y, cf_math_cuda_context *ctx)
{
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL || y == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA || y->metadata.device == CF_DEVICE_CUDA)
    return cf_math_cuda_binary(out, x, y, CF_MATH_BIN_MUL, ctx);
#endif
  return cf_math_binary_cpu(out, x, y, CF_MATH_BIN_MUL, ctx);
}

cf_status cf_math_mul_scalar(cf_math *out, const cf_math *x, double c, cf_math_cuda_context *ctx)
{
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_scalar(out, x, c, CF_MATH_BIN_MUL, ctx);
#endif
  return cf_math_scalar_cpu(out, x, c, CF_MATH_BIN_MUL, ctx);
}

cf_status cf_math_div(cf_math *out, const cf_math *x, const cf_math *y, cf_math_cuda_context *ctx)
{
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL || y == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA || y->metadata.device == CF_DEVICE_CUDA)
    return cf_math_cuda_binary(out, x, y, CF_MATH_BIN_DIV, ctx);
#endif
  return cf_math_binary_cpu(out, x, y, CF_MATH_BIN_DIV, ctx);
}

cf_status cf_math_div_scalar(cf_math *out, const cf_math *x, double c, cf_math_cuda_context *ctx)
{
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_scalar(out, x, c, CF_MATH_BIN_DIV, ctx);
#endif
  return cf_math_scalar_cpu(out, x, c, CF_MATH_BIN_DIV, ctx);
}

cf_status cf_math_pow(cf_math *out, const cf_math *x, double n, cf_math_cuda_context *ctx)
{
  cf_status status;
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_pow(out, x, n, ctx);
#endif
  status = cf_math_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < x->metadata.len; i++) cf_math_store(out, i, pow(cf_math_load(x, i), n));
  return CF_OK;
}

cf_status cf_math_sqrt(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_unary(out, x, CF_MATH_UN_SQRT, ctx);
#endif
  return cf_math_unary_cpu(out, x, CF_MATH_UN_SQRT, ctx);
}

cf_status cf_math_rsqrt(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_unary(out, x, CF_MATH_UN_RSQRT, ctx);
#endif
  return cf_math_unary_cpu(out, x, CF_MATH_UN_RSQRT, ctx);
}

cf_status cf_math_exp(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_unary(out, x, CF_MATH_UN_EXP, ctx);
#endif
  return cf_math_unary_cpu(out, x, CF_MATH_UN_EXP, ctx);
}

cf_status cf_math_log(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_unary(out, x, CF_MATH_UN_LOG, ctx);
#endif
  return cf_math_unary_cpu(out, x, CF_MATH_UN_LOG, ctx);
}

cf_status cf_math_abs(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_unary(out, x, CF_MATH_UN_ABS, ctx);
#endif
  return cf_math_unary_cpu(out, x, CF_MATH_UN_ABS, ctx);
}

cf_status cf_math_neg(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_unary(out, x, CF_MATH_UN_NEG, ctx);
#endif
  return cf_math_unary_cpu(out, x, CF_MATH_UN_NEG, ctx);
}

cf_status cf_math_sign(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_unary(out, x, CF_MATH_UN_SIGN, ctx);
#endif
  return cf_math_unary_cpu(out, x, CF_MATH_UN_SIGN, ctx);
}

cf_status cf_math_clamp(cf_math *out, const cf_math *x, double lo, double hi, cf_math_cuda_context *ctx)
{
  cf_status status;
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_clamp(out, x, lo, hi, ctx);
#endif
  status = cf_math_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < x->metadata.len; i++)
  {
    double v = cf_math_load(x, i);
    cf_math_store(out, i, v < lo ? lo : (v > hi ? hi : v));
  }
  return CF_OK;
}

cf_status cf_math_sum(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
  double acc = 0.0;
  cf_status status = cf_math_require_host_tensor(x);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < x->metadata.len; i++) acc += cf_math_load(x, i);
  return cf_math_scalar_loss(out, x, acc, ctx);
}

cf_status cf_math_sum_axis(cf_math *out, const cf_math *x, cf_usize axis, cf_math_cuda_context *ctx)
{
  return cf_math_reduce_axis_cpu(out, x, axis, CF_MATH_BIN_ADD, ctx);
}

cf_status cf_math_mean(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
  double acc = 0.0;
  cf_status status = cf_math_require_host_tensor(x);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < x->metadata.len; i++) acc += cf_math_load(x, i);
  return cf_math_scalar_loss(out, x, acc / (double)x->metadata.len, ctx);
}

cf_status cf_math_mean_axis(cf_math *out, const cf_math *x, cf_usize axis, cf_math_cuda_context *ctx)
{
  cf_status status = cf_math_sum_axis(out, x, axis, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < out->metadata.len; i++) cf_math_store(out, i, cf_math_load(out, i) / (double)x->dim[axis]);
  return CF_OK;
}

cf_status cf_math_var(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
  double mean = 0.0;
  double var = 0.0;
  cf_status status = cf_math_require_host_tensor(x);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < x->metadata.len; i++) mean += cf_math_load(x, i);
  mean /= (double)x->metadata.len;
  for(cf_usize i = 0; i < x->metadata.len; i++)
  {
    double d = cf_math_load(x, i) - mean;
    var += d * d;
  }
  return cf_math_scalar_loss(out, x, var / (double)x->metadata.len, ctx);
}

cf_status cf_math_std(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
  cf_status status = cf_math_var(out, x, ctx);
  if(status != CF_OK) return status;
  cf_math_store(out, 0, sqrt(cf_math_load(out, 0)));
  return CF_OK;
}

cf_status cf_math_norm2(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
  double acc = 0.0;
  cf_status status = cf_math_require_host_tensor(x);
#if defined(CF_MATH_HAVE_CUDA_RUNTIME) && defined(CF_MATH_HAVE_CUBLAS)
  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_dot_like(out, x, CF_NULL, CF_TRUE, CF_FALSE, ctx);
#endif
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < x->metadata.len; i++)
  {
    double v = cf_math_load(x, i);
    acc += v * v;
  }
  return cf_math_scalar_loss(out, x, sqrt(acc), ctx);
}

cf_status cf_math_norm1(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
  double acc = 0.0;
  cf_status status = cf_math_require_host_tensor(x);
#if defined(CF_MATH_HAVE_CUDA_RUNTIME) && defined(CF_MATH_HAVE_CUBLAS)
  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_dot_like(out, x, CF_NULL, CF_FALSE, CF_TRUE, ctx);
#endif
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < x->metadata.len; i++) acc += fabs(cf_math_load(x, i));
  return cf_math_scalar_loss(out, x, acc, ctx);
}

cf_status cf_math_max(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
  double best;
  cf_status status = cf_math_require_host_tensor(x);
  if(status != CF_OK) return status;
  best = cf_math_load(x, 0);
  for(cf_usize i = 1; i < x->metadata.len; i++)
  {
    double v = cf_math_load(x, i);
    if(v > best) best = v;
  }
  return cf_math_scalar_loss(out, x, best, ctx);
}

cf_status cf_math_min(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
  double best;
  cf_status status = cf_math_require_host_tensor(x);
  if(status != CF_OK) return status;
  best = cf_math_load(x, 0);
  for(cf_usize i = 1; i < x->metadata.len; i++)
  {
    double v = cf_math_load(x, i);
    if(v < best) best = v;
  }
  return cf_math_scalar_loss(out, x, best, ctx);
}

cf_status cf_math_argmax(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
  cf_usize best_i = 0;
  double best;
  cf_status status = cf_math_require_host_tensor(x);
  if(status != CF_OK) return status;
  best = cf_math_load(x, 0);
  status = cf_math_prepare_scalar(out, CF_DTYPE_I32, x->metadata.device, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 1; i < x->metadata.len; i++)
  {
    double v = cf_math_load(x, i);
    if(v > best) { best = v; best_i = i; }
  }
  cf_math_store(out, 0, (double)best_i);
  return CF_OK;
}

cf_status cf_math_argmin(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
  cf_usize best_i = 0;
  double best;
  cf_status status = cf_math_require_host_tensor(x);
  if(status != CF_OK) return status;
  best = cf_math_load(x, 0);
  status = cf_math_prepare_scalar(out, CF_DTYPE_I32, x->metadata.device, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 1; i < x->metadata.len; i++)
  {
    double v = cf_math_load(x, i);
    if(v < best) { best = v; best_i = i; }
  }
  cf_math_store(out, 0, (double)best_i);
  return CF_OK;
}

cf_status cf_math_dot(cf_math *out, const cf_math *x, const cf_math *y, cf_math_cuda_context *ctx)
{
  double acc = 0.0;
  cf_status status = cf_math_require_host_tensor(x);
#if defined(CF_MATH_HAVE_CUDA_RUNTIME) && defined(CF_MATH_HAVE_CUBLAS)
  if(x == CF_NULL || y == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA || y->metadata.device == CF_DEVICE_CUDA)
    return cf_math_cuda_dot_like(out, x, y, CF_FALSE, CF_FALSE, ctx);
#endif
  if(status != CF_OK) return status;
  status = cf_math_require_host_tensor(y);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < x->metadata.len; i++) acc += cf_math_load(x, i) * cf_math_load(y, i);
  return cf_math_scalar_loss(out, x, acc, ctx);
}

cf_status cf_math_cumsum(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
  cf_status status = cf_math_prepare_like(out, x, ctx);
  double acc = 0.0;
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < x->metadata.len; i++)
  {
    acc += cf_math_load(x, i);
    cf_math_store(out, i, acc);
  }
  return CF_OK;
}

cf_status cf_math_matmul(cf_math *out, const cf_math *a, const cf_math *b, cf_math_cuda_context *ctx)
{
#if defined(CF_MATH_HAVE_CUDA_RUNTIME) && defined(CF_MATH_HAVE_CUBLAS)
  if(a == CF_NULL || b == CF_NULL) return CF_ERR_NULL;
  if(a->metadata.device == CF_DEVICE_CUDA || b->metadata.device == CF_DEVICE_CUDA)
    return cf_math_matmul_cuda_core(out, a, b, CF_FALSE, CF_FALSE, ctx);
#endif
  return cf_math_matmul_cpu_core(out, a, b, CF_FALSE, CF_FALSE, ctx);
}

cf_status cf_math_matmul_t(cf_math *out, const cf_math *a, const cf_math *b, cf_bool trans_a, cf_bool trans_b, cf_math_cuda_context *ctx)
{
#if defined(CF_MATH_HAVE_CUDA_RUNTIME) && defined(CF_MATH_HAVE_CUBLAS)
  if(a == CF_NULL || b == CF_NULL) return CF_ERR_NULL;
  if(a->metadata.device == CF_DEVICE_CUDA || b->metadata.device == CF_DEVICE_CUDA)
    return cf_math_matmul_cuda_core(out, a, b, trans_a, trans_b, ctx);
#endif
  return cf_math_matmul_cpu_core(out, a, b, trans_a, trans_b, ctx);
}

cf_status cf_math_matmul_batched(cf_math *out, const cf_math *a, const cf_math *b, cf_math_cuda_context *ctx)
{
  cf_usize batch_count = 1;
  cf_usize m;
  cf_usize k;
  cf_usize n;
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {0};
  cf_status status;

  status = cf_math_require_host_tensor(a);
  if(status != CF_OK) return status;
  status = cf_math_require_host_tensor(b);
  if(status != CF_OK) return status;

  m = a->dim[a->rank - 2];
  k = a->dim[a->rank - 1];
  n = b->dim[b->rank - 1];

  for(cf_usize i = 0; i + 2 < a->rank; i++)
  {
    batch_count *= a->dim[i];
    dim[i] = a->dim[i];
  }
  dim[a->rank - 2] = m;
  dim[a->rank - 1] = n;

  if(out->storage == CF_NULL)
  {
    status = cf_math_alloc(out, dim, a->rank, a->metadata.dtype, a->metadata.device, a->metadata.mem_flags, ctx);
    if(status != CF_OK) return status;
  }

  for(cf_usize batch = 0; batch < batch_count; batch++)
  {
    cf_usize a_base = batch * m * k;
    cf_usize b_base = batch * k * n;
    cf_usize o_base = batch * m * n;
    for(cf_usize row = 0; row < m; row++)
    {
      for(cf_usize col = 0; col < n; col++)
      {
        double acc = 0.0;
        for(cf_usize inner = 0; inner < k; inner++)
          acc += cf_math_load(a, a_base + row * k + inner) * cf_math_load(b, b_base + inner * n + col);
        cf_math_store(out, o_base + row * n + col, acc);
      }
    }
  }

  return CF_OK;
}

cf_status cf_math_linear(cf_math *out, const cf_math *x, const cf_math *w, const cf_math *b, cf_math_cuda_context *ctx)
{
  cf_usize batch;
  cf_usize in;
  cf_usize out_features;
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {0};
  cf_status status;

  if(x == CF_NULL || w == CF_NULL) return CF_ERR_NULL;
#if defined(CF_MATH_HAVE_CUDA_RUNTIME) && defined(CF_MATH_HAVE_CUBLAS)
  if(x->metadata.device == CF_DEVICE_CUDA || w->metadata.device == CF_DEVICE_CUDA || (b != CF_NULL && b->metadata.device == CF_DEVICE_CUDA))
  {
    if(x->rank == 1)
    {
      status = cf_math_matvec_cuda(out, w, x, ctx);
      if(status != CF_OK) return status;
      if(b != CF_NULL) return cf_math_cuda_add_bias(out, b, 1U, w->dim[0], ctx);
      return CF_OK;
    }

    status = cf_math_matmul_cuda_core(out, x, w, CF_FALSE, CF_TRUE, ctx);
    if(status != CF_OK) return status;
    if(b != CF_NULL) return cf_math_cuda_add_bias(out, b, x->dim[0], w->dim[0], ctx);
    return CF_OK;
  }
#endif

  status = cf_math_require_host_tensor(x);
  if(status != CF_OK) return status;
  status = cf_math_require_host_tensor(w);
  if(status != CF_OK) return status;
  if(b != CF_NULL)
  {
    status = cf_math_require_host_tensor(b);
    if(status != CF_OK) return status;
  }

  batch = x->rank == 1 ? 1 : x->dim[0];
  in = x->dim[x->rank - 1];
  out_features = w->dim[0];
  dim[0] = batch;
  dim[1] = out_features;

  if(out->storage == CF_NULL)
  {
    status = cf_math_alloc(out, dim, x->rank == 1 ? 1 : 2, x->metadata.dtype, x->metadata.device, x->metadata.mem_flags, ctx);
    if(status != CF_OK) return status;
  }

  for(cf_usize n = 0; n < batch; n++)
  {
    for(cf_usize o = 0; o < out_features; o++)
    {
      double acc = b == CF_NULL ? 0.0 : cf_math_load(b, o);
      for(cf_usize i = 0; i < in; i++)
        acc += cf_math_load(x, n * in + i) * cf_math_load(w, o * in + i);
      cf_math_store(out, n * out_features + o, acc);
    }
  }

  return CF_OK;
}

cf_status cf_math_linear_fused_relu(cf_math *out, const cf_math *x, const cf_math *w, const cf_math *b, cf_math_cuda_context *ctx)
{
  cf_status status = cf_math_linear(out, x, w, b, ctx);
  if(status != CF_OK) return status;
  return cf_math_relu(out, out, ctx);
}

cf_status cf_math_linear_fused_gelu(cf_math *out, const cf_math *x, const cf_math *w, const cf_math *b, cf_math_cuda_context *ctx)
{
  cf_status status = cf_math_linear(out, x, w, b, ctx);
  if(status != CF_OK) return status;
  return cf_math_gelu(out, out, ctx);
}

cf_status cf_math_linear_backward_W(cf_math *dW, const cf_math *dL, const cf_math *x, cf_math_cuda_context *ctx)
{
  cf_usize batch = dL->dim[0];
  cf_usize out_features = dL->dim[1];
  cf_usize in = x->dim[x->rank - 1];
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {out_features, in};
  cf_status status = dW->storage == CF_NULL ? cf_math_alloc(dW, dim, 2, x->metadata.dtype, x->metadata.device, x->metadata.mem_flags, ctx) : cf_math_zeros(dW, ctx);
  if(status != CF_OK) return status;
  for(cf_usize o = 0; o < out_features; o++)
    for(cf_usize i = 0; i < in; i++)
    {
      double acc = 0.0;
      for(cf_usize n = 0; n < batch; n++) acc += cf_math_load(dL, n * out_features + o) * cf_math_load(x, n * in + i);
      cf_math_store(dW, o * in + i, acc);
    }
  return CF_OK;
}

cf_status cf_math_linear_backward_x(cf_math *dx, const cf_math *dL, const cf_math *W, cf_math_cuda_context *ctx)
{
  cf_usize batch = dL->dim[0];
  cf_usize out_features = dL->dim[1];
  cf_usize in = W->dim[1];
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {batch, in};
  cf_status status = dx->storage == CF_NULL ? cf_math_alloc(dx, dim, 2, dL->metadata.dtype, dL->metadata.device, dL->metadata.mem_flags, ctx) : cf_math_zeros(dx, ctx);
  if(status != CF_OK) return status;
  for(cf_usize n = 0; n < batch; n++)
    for(cf_usize i = 0; i < in; i++)
    {
      double acc = 0.0;
      for(cf_usize o = 0; o < out_features; o++) acc += cf_math_load(dL, n * out_features + o) * cf_math_load(W, o * in + i);
      cf_math_store(dx, n * in + i, acc);
    }
  return CF_OK;
}

cf_status cf_math_linear_backward_b(cf_math *db, const cf_math *dL, cf_math_cuda_context *ctx)
{
  return cf_math_sum_axis(db, dL, 0, ctx);
}

cf_status cf_math_outer(cf_math *out, const cf_math *x, const cf_math *y, cf_math_cuda_context *ctx)
{
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {x->metadata.len, y->metadata.len};
  cf_status status = out->storage == CF_NULL ? cf_math_alloc(out, dim, 2, x->metadata.dtype, x->metadata.device, x->metadata.mem_flags, ctx) : CF_OK;
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < x->metadata.len; i++)
    for(cf_usize j = 0; j < y->metadata.len; j++)
      cf_math_store(out, i * y->metadata.len + j, cf_math_load(x, i) * cf_math_load(y, j));
  return CF_OK;
}

cf_status cf_math_matvec(cf_math *out, const cf_math *a, const cf_math *x, cf_math_cuda_context *ctx)
{
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {0};
  cf_status status;
  if(a == CF_NULL || x == CF_NULL) return CF_ERR_NULL;
#if defined(CF_MATH_HAVE_CUDA_RUNTIME) && defined(CF_MATH_HAVE_CUBLAS)
  if(a->metadata.device == CF_DEVICE_CUDA || x->metadata.device == CF_DEVICE_CUDA)
    return cf_math_matvec_cuda(out, a, x, ctx);
#endif
  status = cf_math_require_host_tensor(a);
  if(status != CF_OK) return status;
  status = cf_math_require_host_tensor(x);
  if(status != CF_OK) return status;
  dim[0] = a->dim[0];
  status = out->storage == CF_NULL ? cf_math_alloc(out, dim, 1, a->metadata.dtype, a->metadata.device, a->metadata.mem_flags, ctx) : CF_OK;
  if(status != CF_OK) return status;
  for(cf_usize row = 0; row < a->dim[0]; row++)
  {
    double acc = 0.0;
    for(cf_usize col = 0; col < a->dim[1]; col++) acc += cf_math_load(a, row * a->dim[1] + col) * cf_math_load(x, col);
    cf_math_store(out, row, acc);
  }
  return CF_OK;
}

cf_status cf_math_transpose(cf_math *out, const cf_math *a, cf_math_cuda_context *ctx)
{
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {a->dim[1], a->dim[0]};
  cf_status status = out->storage == CF_NULL ? cf_math_alloc(out, dim, 2, a->metadata.dtype, a->metadata.device, a->metadata.mem_flags, ctx) : CF_OK;
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < a->dim[0]; i++)
    for(cf_usize j = 0; j < a->dim[1]; j++)
      cf_math_store(out, j * a->dim[0] + i, cf_math_load(a, i * a->dim[1] + j));
  return CF_OK;
}

cf_status cf_math_scale(cf_math *out, const cf_math *a, double alpha, cf_math_cuda_context *ctx)
{
  return cf_math_mul_scalar(out, a, alpha, ctx);
}

cf_status cf_math_relu(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
  cf_status status;
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_activation(out, x, CF_NULL, 0.0, CF_MATH_ACT_RELU, ctx);
#endif
  status = cf_math_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < x->metadata.len; i++)
  {
    double v = cf_math_load(x, i);
    cf_math_store(out, i, v > 0.0 ? v : 0.0);
  }
  return CF_OK;
}

cf_status cf_math_relu_bwd(cf_math *dx, const cf_math *dy, const cf_math *y, cf_math_cuda_context *ctx)
{
  cf_status status;
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(dy == CF_NULL || y == CF_NULL) return CF_ERR_NULL;
  if(dy->metadata.device == CF_DEVICE_CUDA || y->metadata.device == CF_DEVICE_CUDA)
    return cf_math_cuda_activation(dx, dy, y, 0.0, CF_MATH_ACT_RELU_BWD, ctx);
#endif
  status = cf_math_prepare_like(dx, dy, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < dy->metadata.len; i++) cf_math_store(dx, i, cf_math_load(y, i) > 0.0 ? cf_math_load(dy, i) : 0.0);
  return CF_OK;
}

cf_status cf_math_leaky_relu(cf_math *out, const cf_math *x, double alpha, cf_math_cuda_context *ctx)
{
  cf_status status;
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_activation(out, x, CF_NULL, alpha, CF_MATH_ACT_LEAKY_RELU, ctx);
#endif
  status = cf_math_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < x->metadata.len; i++)
  {
    double v = cf_math_load(x, i);
    cf_math_store(out, i, v > 0.0 ? v : alpha * v);
  }
  return CF_OK;
}

cf_status cf_math_elu(cf_math *out, const cf_math *x, double alpha, cf_math_cuda_context *ctx)
{
  cf_status status;
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_activation(out, x, CF_NULL, alpha, CF_MATH_ACT_ELU, ctx);
#endif
  status = cf_math_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < x->metadata.len; i++)
  {
    double v = cf_math_load(x, i);
    cf_math_store(out, i, v >= 0.0 ? v : alpha * (exp(v) - 1.0));
  }
  return CF_OK;
}

cf_status cf_math_sigmoid(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
  cf_status status;
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_activation(out, x, CF_NULL, 0.0, CF_MATH_ACT_SIGMOID, ctx);
#endif
  status = cf_math_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < x->metadata.len; i++) cf_math_store(out, i, 1.0 / (1.0 + exp(-cf_math_load(x, i))));
  return CF_OK;
}

cf_status cf_math_sigmoid_bwd(cf_math *dx, const cf_math *dy, const cf_math *y, cf_math_cuda_context *ctx)
{
  cf_status status;
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(dy == CF_NULL || y == CF_NULL) return CF_ERR_NULL;
  if(dy->metadata.device == CF_DEVICE_CUDA || y->metadata.device == CF_DEVICE_CUDA)
    return cf_math_cuda_activation(dx, dy, y, 0.0, CF_MATH_ACT_SIGMOID_BWD, ctx);
#endif
  status = cf_math_prepare_like(dx, dy, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < dy->metadata.len; i++)
  {
    double v = cf_math_load(y, i);
    cf_math_store(dx, i, cf_math_load(dy, i) * v * (1.0 - v));
  }
  return CF_OK;
}

cf_status cf_math_tanh(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
  cf_status status;
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_activation(out, x, CF_NULL, 0.0, CF_MATH_ACT_TANH, ctx);
#endif
  status = cf_math_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < x->metadata.len; i++) cf_math_store(out, i, tanh(cf_math_load(x, i)));
  return CF_OK;
}

cf_status cf_math_tanh_bwd(cf_math *dx, const cf_math *dy, const cf_math *y, cf_math_cuda_context *ctx)
{
  cf_status status;
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(dy == CF_NULL || y == CF_NULL) return CF_ERR_NULL;
  if(dy->metadata.device == CF_DEVICE_CUDA || y->metadata.device == CF_DEVICE_CUDA)
    return cf_math_cuda_activation(dx, dy, y, 0.0, CF_MATH_ACT_TANH_BWD, ctx);
#endif
  status = cf_math_prepare_like(dx, dy, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < dy->metadata.len; i++)
  {
    double v = cf_math_load(y, i);
    cf_math_store(dx, i, cf_math_load(dy, i) * (1.0 - v * v));
  }
  return CF_OK;
}

cf_status cf_math_gelu(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
  cf_status status;
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_activation(out, x, CF_NULL, 0.0, CF_MATH_ACT_GELU, ctx);
#endif
  status = cf_math_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < x->metadata.len; i++)
  {
    double v = cf_math_load(x, i);
    cf_math_store(out, i, 0.5 * v * (1.0 + erf(v / sqrt(2.0))));
  }
  return CF_OK;
}

cf_status cf_math_gelu_approx(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
  cf_status status;
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_activation(out, x, CF_NULL, 0.0, CF_MATH_ACT_GELU_APPROX, ctx);
#endif
  status = cf_math_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < x->metadata.len; i++)
  {
    double v = cf_math_load(x, i);
    cf_math_store(out, i, 0.5 * v * (1.0 + tanh(CF_MATH_SQRT_2_OVER_PI * (v + 0.044715 * v * v * v))));
  }
  return CF_OK;
}

cf_status cf_math_gelu_bwd(cf_math *dx, const cf_math *dy, const cf_math *x, cf_math_cuda_context *ctx)
{
  cf_status status;
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(dy == CF_NULL || x == CF_NULL) return CF_ERR_NULL;
  if(dy->metadata.device == CF_DEVICE_CUDA || x->metadata.device == CF_DEVICE_CUDA)
    return cf_math_cuda_activation(dx, dy, x, 0.0, CF_MATH_ACT_GELU_BWD, ctx);
#endif
  status = cf_math_prepare_like(dx, dy, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < dy->metadata.len; i++)
  {
    double v = cf_math_load(x, i);
    double cdf = 0.5 * (1.0 + erf(v / sqrt(2.0)));
    double pdf = exp(-0.5 * v * v) / sqrt(2.0 * CF_MATH_PI);
    cf_math_store(dx, i, cf_math_load(dy, i) * (cdf + v * pdf));
  }
  return CF_OK;
}

cf_status cf_math_swish(cf_math *out, const cf_math *x, double beta, cf_math_cuda_context *ctx)
{
  cf_status status;
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_activation(out, x, CF_NULL, beta, CF_MATH_ACT_SWISH, ctx);
#endif
  status = cf_math_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < x->metadata.len; i++)
  {
    double v = cf_math_load(x, i);
    cf_math_store(out, i, v / (1.0 + exp(-beta * v)));
  }
  return CF_OK;
}

cf_status cf_math_silu(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
  return cf_math_swish(out, x, 1.0, ctx);
}

cf_status cf_math_softplus(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
  cf_status status;
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_activation(out, x, CF_NULL, 0.0, CF_MATH_ACT_SOFTPLUS, ctx);
#endif
  status = cf_math_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < x->metadata.len; i++)
  {
    double v = cf_math_load(x, i);
    cf_math_store(out, i, v > 20.0 ? v : log1p(exp(v)));
  }
  return CF_OK;
}

cf_status cf_math_mish(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx)
{
  cf_status status;
#if defined(CF_MATH_HAVE_CUDA_RUNTIME)
  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata.device == CF_DEVICE_CUDA) return cf_math_cuda_activation(out, x, CF_NULL, 0.0, CF_MATH_ACT_MISH, ctx);
#endif
  status = cf_math_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < x->metadata.len; i++)
  {
    double v = cf_math_load(x, i);
    cf_math_store(out, i, v * tanh(v > 20.0 ? v : log1p(exp(v))));
  }
  return CF_OK;
}

cf_status cf_math_softmax_fwd(cf_math *out, const cf_math *x, cf_usize axis, cf_math_softmax_mode mode, cf_math_cuda_context *ctx)
{
  cf_status status = cf_math_prepare_like(out, x, ctx);
  cf_usize outer = cf_math_axis_outer(x, axis);
  cf_usize inner = cf_math_axis_inner(x, axis);
  cf_usize classes = x->dim[axis];
  CF_UNUSED(mode);
  if(status != CF_OK) return status;

  for(cf_usize o = 0; o < outer; o++)
  {
    for(cf_usize in = 0; in < inner; in++)
    {
      double max_v = cf_math_load(x, o * classes * inner + in);
      double sum = 0.0;
      for(cf_usize c = 1; c < classes; c++)
      {
        double v = cf_math_load(x, (o * classes + c) * inner + in);
        if(v > max_v) max_v = v;
      }
      for(cf_usize c = 0; c < classes; c++)
      {
        double e = exp(cf_math_load(x, (o * classes + c) * inner + in) - max_v);
        sum += e;
        cf_math_store(out, (o * classes + c) * inner + in, e);
      }
      for(cf_usize c = 0; c < classes; c++)
      {
        cf_usize idx = (o * classes + c) * inner + in;
        cf_math_store(out, idx, cf_math_load(out, idx) / sum);
      }
    }
  }
  return CF_OK;
}

cf_status cf_math_softmax_bwd(cf_math *dx, const cf_math *dy, const cf_math *y, cf_usize axis, cf_math_cuda_context *ctx)
{
  cf_status status = cf_math_prepare_like(dx, dy, ctx);
  cf_usize outer = cf_math_axis_outer(y, axis);
  cf_usize inner = cf_math_axis_inner(y, axis);
  cf_usize classes = y->dim[axis];
  if(status != CF_OK) return status;

  for(cf_usize o = 0; o < outer; o++)
    for(cf_usize in = 0; in < inner; in++)
    {
      double dot = 0.0;
      for(cf_usize c = 0; c < classes; c++)
      {
        cf_usize idx = (o * classes + c) * inner + in;
        dot += cf_math_load(y, idx) * cf_math_load(dy, idx);
      }
      for(cf_usize c = 0; c < classes; c++)
      {
        cf_usize idx = (o * classes + c) * inner + in;
        cf_math_store(dx, idx, cf_math_load(y, idx) * (cf_math_load(dy, idx) - dot));
      }
    }
  return CF_OK;
}

cf_status cf_math_log_softmax_fwd(cf_math *out, const cf_math *x, cf_usize axis, cf_math_cuda_context *ctx)
{
  cf_status status = cf_math_softmax_fwd(out, x, axis, CF_SOFTMAX_CHANNEL, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < out->metadata.len; i++) cf_math_store(out, i, log(cf_math_load(out, i)));
  return CF_OK;
}

cf_status cf_math_log_softmax_bwd(cf_math *dx, const cf_math *dy, const cf_math *y, cf_usize axis, cf_math_cuda_context *ctx)
{
  cf_status status = cf_math_prepare_like(dx, dy, ctx);
  cf_usize outer = cf_math_axis_outer(y, axis);
  cf_usize inner = cf_math_axis_inner(y, axis);
  cf_usize classes = y->dim[axis];
  if(status != CF_OK) return status;

  for(cf_usize o = 0; o < outer; o++)
    for(cf_usize in = 0; in < inner; in++)
    {
      double sum_dy = 0.0;
      for(cf_usize c = 0; c < classes; c++)
        sum_dy += cf_math_load(dy, (o * classes + c) * inner + in);
      for(cf_usize c = 0; c < classes; c++)
      {
        cf_usize idx = (o * classes + c) * inner + in;
        cf_math_store(dx, idx, cf_math_load(dy, idx) - exp(cf_math_load(y, idx)) * sum_dy);
      }
    }

  return CF_OK;
}

cf_status cf_math_mse_loss(cf_math *loss, const cf_math *y, const cf_math *target, cf_math_cuda_context *ctx)
{
  double acc = 0.0;
  for(cf_usize i = 0; i < y->metadata.len; i++)
  {
    double d = cf_math_load(y, i) - cf_math_load(target, i);
    acc += d * d;
  }
  return cf_math_scalar_loss(loss, y, acc / (double)y->metadata.len, ctx);
}

cf_status cf_math_mse_loss_bwd(cf_math *dx, const cf_math *y, const cf_math *target, cf_math_cuda_context *ctx)
{
  cf_status status = cf_math_prepare_like(dx, y, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < y->metadata.len; i++)
    cf_math_store(dx, i, 2.0 * (cf_math_load(y, i) - cf_math_load(target, i)) / (double)y->metadata.len);
  return CF_OK;
}

cf_status cf_math_cross_entropy(cf_math *loss, cf_math *dx, const cf_math *logits, const cf_math *target, cf_usize axis, cf_math_cuda_context *ctx)
{
  cf_math prob;
  double acc = 0.0;
  cf_status status;

  memset(&prob, 0, sizeof(prob));
  status = cf_math_softmax_fwd(&prob, logits, axis, CF_SOFTMAX_CHANNEL, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < logits->metadata.len; i++)
  {
    double t = cf_math_load(target, i);
    if(t != 0.0) acc -= t * log(cf_math_load(&prob, i));
  }
  if(dx != CF_NULL) status = cf_math_cross_entropy_bwd(dx, &prob, target, ctx);
  if(status == CF_OK) status = cf_math_scalar_loss(loss, logits, acc / (double)logits->dim[0], ctx);
  cf_math_free(&prob, ctx);
  return status;
}

cf_status cf_math_cross_entropy_bwd(cf_math *dx, const cf_math *prob, const cf_math *target, cf_math_cuda_context *ctx)
{
  cf_status status = cf_math_prepare_like(dx, prob, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < prob->metadata.len; i++) cf_math_store(dx, i, cf_math_load(prob, i) - cf_math_load(target, i));
  return CF_OK;
}

cf_status cf_math_nll_loss(cf_math *loss, const cf_math *log_prob, const cf_math *labels, cf_math_cuda_context *ctx)
{
  cf_usize batch = log_prob->dim[0];
  cf_usize classes = log_prob->dim[1];
  double acc = 0.0;
  for(cf_usize n = 0; n < batch; n++) acc -= cf_math_load(log_prob, n * classes + (cf_usize)cf_math_load_i32(labels, n));
  return cf_math_scalar_loss(loss, log_prob, acc / (double)batch, ctx);
}

cf_status cf_math_bce_loss(cf_math *loss, const cf_math *prob, const cf_math *target, cf_math_cuda_context *ctx)
{
  double acc = 0.0;
  for(cf_usize i = 0; i < prob->metadata.len; i++)
  {
    double p = cf_math_load(prob, i);
    double t = cf_math_load(target, i);
    acc -= t * log(p) + (1.0 - t) * log(1.0 - p);
  }
  return cf_math_scalar_loss(loss, prob, acc / (double)prob->metadata.len, ctx);
}

cf_status cf_math_huber_loss(cf_math *loss, const cf_math *y, const cf_math *target, double delta, cf_math_cuda_context *ctx)
{
  double acc = 0.0;
  for(cf_usize i = 0; i < y->metadata.len; i++)
  {
    double d = fabs(cf_math_load(y, i) - cf_math_load(target, i));
    acc += d <= delta ? 0.5 * d * d : delta * (d - 0.5 * delta);
  }
  return cf_math_scalar_loss(loss, y, acc / (double)y->metadata.len, ctx);
}

cf_status cf_math_focal_loss(cf_math *loss, const cf_math *prob, const cf_math *target, double alpha, double gamma, cf_math_cuda_context *ctx)
{
  double acc = 0.0;
  for(cf_usize i = 0; i < prob->metadata.len; i++)
  {
    double p = cf_math_load(prob, i);
    double t = cf_math_load(target, i);
    double pt = t != 0.0 ? p : (1.0 - p);
    acc += -alpha * pow(1.0 - pt, gamma) * log(pt);
  }
  return cf_math_scalar_loss(loss, prob, acc / (double)prob->metadata.len, ctx);
}

cf_status cf_math_dropout_fwd(cf_math *out, cf_math_dropout_state *state, const cf_math *x, double p, cf_u64 seed, cf_math_cuda_context *ctx)
{
  cf_status status = cf_math_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;
  if(state != CF_NULL)
  {
    if(state->reserve_size < x->metadata.len)
    {
      void *reserve = realloc(state->reserve, x->metadata.len);
      if(reserve == CF_NULL) return CF_ERR_OOM;
      state->reserve = reserve;
      state->reserve_size = x->metadata.len;
    }
    state->probability = (float)p;
    state->seed = seed;
  }
  for(cf_usize i = 0; i < x->metadata.len; i++)
  {
    cf_u8 keep = cf_math_rng_uniform01(&seed) >= p ? 1U : 0U;
    if(state != CF_NULL) ((cf_u8 *)state->reserve)[i] = keep;
    cf_math_store(out, i, keep ? cf_math_load(x, i) / (1.0 - p) : 0.0);
  }
  return CF_OK;
}

cf_status cf_math_dropout_bwd(cf_math *dx, const cf_math_dropout_state *state, const cf_math *dy, cf_math_cuda_context *ctx)
{
  cf_status status = cf_math_prepare_like(dx, dy, ctx);
  double scale;
  if(status != CF_OK) return status;
  scale = 1.0 / (1.0 - (double)state->probability);
  for(cf_usize i = 0; i < dy->metadata.len; i++)
    cf_math_store(dx, i, ((const cf_u8 *)state->reserve)[i] ? cf_math_load(dy, i) * scale : 0.0);
  return CF_OK;
}

cf_status cf_math_dropout_train_set(cf_math_dropout_state *state, double p, cf_bool training, cf_math_cuda_context *ctx)
{
  CF_UNUSED(ctx);
  if(state == CF_NULL) return CF_ERR_NULL;
  state->probability = training ? (float)p : 0.0f;
  return CF_OK;
}

cf_status cf_math_attn_dropout_fwd(cf_math *out, cf_math_dropout_state *state, const cf_math *x, double p, cf_u64 seed, cf_math_cuda_context *ctx)
{
  return cf_math_dropout_fwd(out, state, x, p, seed, ctx);
}

cf_status cf_math_embed_fwd(cf_math *out, const cf_math *w, const cf_math *idx, cf_math_cuda_context *ctx)
{
  cf_usize tokens = idx->metadata.len;
  cf_usize d = w->dim[1];
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {tokens, d};
  cf_status status = out->storage == CF_NULL ? cf_math_alloc(out, dim, 2, w->metadata.dtype, w->metadata.device, w->metadata.mem_flags, ctx) : CF_OK;
  if(status != CF_OK) return status;
  for(cf_usize t = 0; t < tokens; t++)
  {
    cf_usize row = (cf_usize)cf_math_load_i32(idx, t);
    for(cf_usize j = 0; j < d; j++) cf_math_store(out, t * d + j, cf_math_load(w, row * d + j));
  }
  return CF_OK;
}

cf_status cf_math_embed_bwd(cf_math *dW, const cf_math *idx, const cf_math *dy, cf_math_cuda_context *ctx)
{
  cf_usize d = dy->dim[1];
  cf_status status = cf_math_zeros(dW, ctx);
  if(status != CF_OK) return status;
  for(cf_usize t = 0; t < idx->metadata.len; t++)
  {
    cf_usize row = (cf_usize)cf_math_load_i32(idx, t);
    for(cf_usize j = 0; j < d; j++)
      cf_math_store(dW, row * d + j, cf_math_load(dW, row * d + j) + cf_math_load(dy, t * d + j));
  }
  return CF_OK;
}

cf_status cf_math_embed_bwd_atomic(cf_math *dW, const cf_math *idx, const cf_math *dy, cf_math_cuda_context *ctx)
{
  return cf_math_embed_bwd(dW, idx, dy, ctx);
}

cf_status cf_math_sgd_step(cf_math *w, const cf_math *g, double lr, cf_math_cuda_context *ctx)
{
  CF_UNUSED(ctx);
  for(cf_usize i = 0; i < w->metadata.len; i++) cf_math_store(w, i, cf_math_load(w, i) - lr * cf_math_load(g, i));
  return CF_OK;
}

cf_status cf_math_sgd_momentum(cf_math *w, cf_math *v, const cf_math *g, double lr, double momentum, cf_math_cuda_context *ctx)
{
  CF_UNUSED(ctx);
  for(cf_usize i = 0; i < w->metadata.len; i++)
  {
    double vi = momentum * cf_math_load(v, i) + cf_math_load(g, i);
    cf_math_store(v, i, vi);
    cf_math_store(w, i, cf_math_load(w, i) - lr * vi);
  }
  return CF_OK;
}

cf_status cf_math_adam_step(cf_math *w, cf_math *m, cf_math *v, const cf_math *g, double lr, double beta1, double beta2, double eps, cf_u64 step, cf_math_cuda_context *ctx)
{
  double b1c = 1.0 - pow(beta1, (double)step);
  double b2c = 1.0 - pow(beta2, (double)step);
  CF_UNUSED(ctx);
  for(cf_usize i = 0; i < w->metadata.len; i++)
  {
    double gi = cf_math_load(g, i);
    double mi = beta1 * cf_math_load(m, i) + (1.0 - beta1) * gi;
    double vi = beta2 * cf_math_load(v, i) + (1.0 - beta2) * gi * gi;
    cf_math_store(m, i, mi);
    cf_math_store(v, i, vi);
    cf_math_store(w, i, cf_math_load(w, i) - lr * (mi / b1c) / (sqrt(vi / b2c) + eps));
  }
  return CF_OK;
}

cf_status cf_math_adamw_step(cf_math *w, cf_math *m, cf_math *v, const cf_math *g, double lr, double beta1, double beta2, double eps, double decay, cf_u64 step, cf_math_cuda_context *ctx)
{
  cf_status status = cf_math_adam_step(w, m, v, g, lr, beta1, beta2, eps, step, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < w->metadata.len; i++) cf_math_store(w, i, cf_math_load(w, i) * (1.0 - lr * decay));
  return CF_OK;
}

cf_status cf_math_rmsprop_step(cf_math *w, cf_math *v, const cf_math *g, double lr, double beta, double eps, cf_math_cuda_context *ctx)
{
  CF_UNUSED(ctx);
  for(cf_usize i = 0; i < w->metadata.len; i++)
  {
    double gi = cf_math_load(g, i);
    double vi = beta * cf_math_load(v, i) + (1.0 - beta) * gi * gi;
    cf_math_store(v, i, vi);
    cf_math_store(w, i, cf_math_load(w, i) - lr * gi / (sqrt(vi) + eps));
  }
  return CF_OK;
}

cf_status cf_math_grad_clip_norm(cf_math *g, double max_norm, cf_math_cuda_context *ctx)
{
  cf_math norm;
  cf_status status;
  double n;

  memset(&norm, 0, sizeof(norm));
  status = cf_math_norm2(&norm, g, ctx);
  if(status != CF_OK) return status;
  n = cf_math_load(&norm, 0);
  if(n > max_norm) status = cf_math_mul_scalar(g, g, max_norm / n, ctx);
  cf_math_free(&norm, ctx);
  return status;
}

cf_status cf_math_grad_clip_value(cf_math *g, double clip, cf_math_cuda_context *ctx)
{
  return cf_math_clamp(g, g, -clip, clip, ctx);
}

cf_status cf_math_weight_decay(cf_math *g, const cf_math *w, double decay, cf_math_cuda_context *ctx)
{
  for(cf_usize i = 0; i < g->metadata.len; i++) cf_math_store(g, i, cf_math_load(g, i) + decay * cf_math_load(w, i));
  CF_UNUSED(ctx);
  return CF_OK;
}

cf_status cf_math_lr_scale(cf_math *g, double scale, cf_math_cuda_context *ctx)
{
  return cf_math_mul_scalar(g, g, scale, ctx);
}

cf_status cf_math_grad_allreduce(cf_math *g, cf_usize world_size, cf_math_cuda_context *ctx)
{
  CF_UNUSED(g);
  CF_UNUSED(ctx);
  return world_size <= 1 ? CF_OK : CF_ERR_UNSUPPORTED;
}

cf_status cf_math_grad_zero(cf_math *g, cf_math_cuda_context *ctx)
{
  return cf_math_zeros(g, ctx);
}

cf_status cf_math_reshape(cf_math *out, const cf_math *x, const cf_usize dim[CF_MATH_HIGHEST_RANK], cf_usize rank)
{
  cf_usize len;
  cf_status status = cf_math_shape_len(dim, rank, &len);
  if(status != CF_OK) return status;
  *out = *x;
  cf_math_retain_storage(out->storage);
  cf_math_apply_shape(out, dim, rank, len);
  out->metadata.dtype = x->metadata.dtype;
  out->metadata.device = x->metadata.device;
  out->metadata.mem_flags = x->metadata.mem_flags;
  return CF_OK;
}

cf_status cf_math_squeeze(cf_math *out, const cf_math *x)
{
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {0};
  cf_usize rank = 0;
  for(cf_usize i = 0; i < x->rank; i++) if(x->dim[i] != 1) dim[rank++] = x->dim[i];
  return cf_math_reshape(out, x, dim, rank);
}

cf_status cf_math_unsqueeze(cf_math *out, const cf_math *x, cf_usize axis)
{
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {0};
  for(cf_usize i = 0, j = 0; i < x->rank + 1; i++)
    dim[i] = i == axis ? 1 : x->dim[j++];
  return cf_math_reshape(out, x, dim, x->rank + 1);
}

cf_status cf_math_expand(cf_math *out, const cf_math *x, const cf_usize dim[CF_MATH_HIGHEST_RANK], cf_usize rank)
{
  cf_status status = cf_math_reshape(out, x, dim, rank);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < rank; i++) if(i >= x->rank || x->dim[i] == 1) out->metadata.strides[i] = 0;
  out->metadata.layout = CF_LAYOUT_STRIDED;
  return CF_OK;
}

cf_status cf_math_slice(cf_math *out, const cf_math *x, const cf_usize start[CF_MATH_HIGHEST_RANK], const cf_usize len[CF_MATH_HIGHEST_RANK])
{
  cf_usize offset = 0;
  for(cf_usize i = 0; i < x->rank; i++) offset += start[i] * x->metadata.strides[i];
  return cf_math_view(out, x, offset, len, x->rank);
}

cf_status cf_math_flatten(cf_math *out, const cf_math *x, cf_usize start_axis, cf_usize end_axis)
{
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {0};
  cf_usize rank = 0;
  cf_usize flat = 1;
  for(cf_usize i = 0; i < start_axis; i++) dim[rank++] = x->dim[i];
  for(cf_usize i = start_axis; i <= end_axis; i++) flat *= x->dim[i];
  dim[rank++] = flat;
  for(cf_usize i = end_axis + 1; i < x->rank; i++) dim[rank++] = x->dim[i];
  return cf_math_reshape(out, x, dim, rank);
}

cf_status cf_math_permute(cf_math *out, const cf_math *x, const cf_usize axes[CF_MATH_HIGHEST_RANK], cf_math_cuda_context *ctx)
{
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {0};
  cf_status status;
  for(cf_usize i = 0; i < x->rank; i++) dim[i] = x->dim[axes[i]];
  status = out->storage == CF_NULL ? cf_math_alloc(out, dim, x->rank, x->metadata.dtype, x->metadata.device, x->metadata.mem_flags, ctx) : CF_OK;
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < out->metadata.len; i++)
  {
    cf_usize tmp = i;
    cf_usize src_offset = 0;
    for(cf_usize axis_i = out->rank; axis_i > 0; axis_i--)
    {
      cf_usize axis = axis_i - 1;
      cf_usize coord = tmp % out->dim[axis];
      tmp /= out->dim[axis];
      src_offset += coord * x->metadata.strides[axes[axis]];
    }
    cf_math_store(out, i, cf_math_load(x, src_offset));
  }
  return CF_OK;
}

cf_status cf_math_concat(cf_math *out, const cf_math **xs, cf_usize count, cf_usize axis, cf_math_cuda_context *ctx)
{
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {0};
  cf_usize prefix;
  cf_status status;

  for(cf_usize i = 0; i < xs[0]->rank; i++) dim[i] = xs[0]->dim[i];
  dim[axis] = 0;
  for(cf_usize i = 0; i < count; i++) dim[axis] += xs[i]->dim[axis];
  status = out->storage == CF_NULL ? cf_math_alloc(out, dim, xs[0]->rank, xs[0]->metadata.dtype, xs[0]->metadata.device, xs[0]->metadata.mem_flags, ctx) : CF_OK;
  if(status != CF_OK) return status;

  prefix = 0;
  for(cf_usize t = 0; t < count; t++)
  {
    for(cf_usize i = 0; i < xs[t]->metadata.len; i++) cf_math_store(out, prefix + i, cf_math_load(xs[t], i));
    prefix += xs[t]->metadata.len;
  }
  return CF_OK;
}

cf_status cf_math_split(cf_math *outs, cf_usize count, const cf_math *x, cf_usize axis)
{
  cf_usize part = x->dim[axis] / count;
  cf_usize start[CF_MATH_HIGHEST_RANK] = {0};
  cf_usize len[CF_MATH_HIGHEST_RANK] = {0};
  for(cf_usize i = 0; i < x->rank; i++) len[i] = x->dim[i];
  len[axis] = part;
  for(cf_usize i = 0; i < count; i++)
  {
    start[axis] = i * part;
    cf_math_slice(&outs[i], x, start, len);
  }
  return CF_OK;
}

cf_status cf_math_pad(cf_math *out, const cf_math *x, const cf_usize before[CF_MATH_HIGHEST_RANK], const cf_usize after[CF_MATH_HIGHEST_RANK], cf_math_cuda_context *ctx)
{
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {0};
  cf_status status;
  for(cf_usize i = 0; i < x->rank; i++) dim[i] = before[i] + x->dim[i] + after[i];
  status = out->storage == CF_NULL ? cf_math_alloc(out, dim, x->rank, x->metadata.dtype, x->metadata.device, x->metadata.mem_flags, ctx) : cf_math_zeros(out, ctx);
  if(status != CF_OK) return status;
  for(cf_usize i = 0; i < x->metadata.len; i++)
  {
    cf_usize tmp = i;
    cf_usize dst = 0;
    for(cf_usize axis_i = x->rank; axis_i > 0; axis_i--)
    {
      cf_usize axis = axis_i - 1;
      cf_usize coord = tmp % x->dim[axis];
      tmp /= x->dim[axis];
      dst += (coord + before[axis]) * out->metadata.strides[axis];
    }
    cf_math_store(out, dst, cf_math_load(x, i));
  }
  return CF_OK;
}

cf_status cf_math_conv2d_fwd(cf_math *out, const cf_math *x, const cf_math *w, const cf_math *b, cf_math_conv2d_params p, cf_math_cuda_context *ctx)
{
  cf_usize n_count = x->dim[0];
  cf_usize in_c = x->dim[1];
  cf_usize in_h = x->dim[2];
  cf_usize in_w = x->dim[3];
  cf_usize out_c = w->dim[0];
  cf_usize kernel_h = w->dim[2];
  cf_usize kernel_w = w->dim[3];
  cf_usize groups = p.groups <= 0 ? 1U : (cf_usize)p.groups;
  cf_usize out_h = (in_h + (cf_usize)(2 * p.pad_h) - (cf_usize)p.dilation_h * (kernel_h - 1) - 1) / (cf_usize)p.stride_h + 1;
  cf_usize out_w = (in_w + (cf_usize)(2 * p.pad_w) - (cf_usize)p.dilation_w * (kernel_w - 1) - 1) / (cf_usize)p.stride_w + 1;
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {n_count, out_c, out_h, out_w};
  cf_usize out_per_group = out_c / groups;
  cf_usize in_per_group = in_c / groups;
  cf_status status;

  if(out->storage == CF_NULL)
  {
    status = cf_math_alloc(out, dim, 4, x->metadata.dtype, x->metadata.device, x->metadata.mem_flags, ctx);
    if(status != CF_OK) return status;
  }

  for(cf_usize n = 0; n < n_count; n++)
    for(cf_usize oc = 0; oc < out_c; oc++)
      for(cf_usize oh = 0; oh < out_h; oh++)
        for(cf_usize ow = 0; ow < out_w; ow++)
        {
          cf_usize group = oc / out_per_group;
          double acc = b == CF_NULL ? 0.0 : cf_math_load(b, oc);
          for(cf_usize icg = 0; icg < in_per_group; icg++)
          {
            cf_usize ic = group * in_per_group + icg;
            for(cf_usize kh = 0; kh < kernel_h; kh++)
              for(cf_usize kw = 0; kw < kernel_w; kw++)
              {
                cf_isize ih = (cf_isize)(oh * (cf_usize)p.stride_h + kh * (cf_usize)p.dilation_h) - (cf_isize)p.pad_h;
                cf_isize iw = (cf_isize)(ow * (cf_usize)p.stride_w + kw * (cf_usize)p.dilation_w) - (cf_isize)p.pad_w;
                if(ih >= 0 && iw >= 0 && (cf_usize)ih < in_h && (cf_usize)iw < in_w)
                {
                  cf_usize xi = cf_math_nchw_index(n, ic, (cf_usize)ih, (cf_usize)iw, in_c, in_h, in_w);
                  cf_usize wi = ((oc * in_per_group + icg) * kernel_h + kh) * kernel_w + kw;
                  acc += cf_math_load(x, xi) * cf_math_load(w, wi);
                }
              }
          }
          cf_math_store(out, cf_math_nchw_index(n, oc, oh, ow, out_c, out_h, out_w), acc);
        }

  return CF_OK;
}

cf_status cf_math_conv2d_bwd_data(cf_math *dx, const cf_math *dL, const cf_math *w, cf_math_conv2d_params p, cf_math_cuda_context *ctx)
{
  cf_usize n_count = dx->dim[0];
  cf_usize in_c = dx->dim[1];
  cf_usize in_h = dx->dim[2];
  cf_usize in_w = dx->dim[3];
  cf_usize out_c = dL->dim[1];
  cf_usize out_h = dL->dim[2];
  cf_usize out_w = dL->dim[3];
  cf_usize kernel_h = w->dim[2];
  cf_usize kernel_w = w->dim[3];
  cf_usize groups = p.groups <= 0 ? 1U : (cf_usize)p.groups;
  cf_usize out_per_group = out_c / groups;
  cf_usize in_per_group = in_c / groups;
  cf_status status = cf_math_zeros(dx, ctx);
  if(status != CF_OK) return status;

  for(cf_usize n = 0; n < n_count; n++)
    for(cf_usize oc = 0; oc < out_c; oc++)
      for(cf_usize oh = 0; oh < out_h; oh++)
        for(cf_usize ow = 0; ow < out_w; ow++)
        {
          cf_usize group = oc / out_per_group;
          double grad = cf_math_load(dL, cf_math_nchw_index(n, oc, oh, ow, out_c, out_h, out_w));
          for(cf_usize icg = 0; icg < in_per_group; icg++)
          {
            cf_usize ic = group * in_per_group + icg;
            for(cf_usize kh = 0; kh < kernel_h; kh++)
              for(cf_usize kw = 0; kw < kernel_w; kw++)
              {
                cf_isize ih = (cf_isize)(oh * (cf_usize)p.stride_h + kh * (cf_usize)p.dilation_h) - (cf_isize)p.pad_h;
                cf_isize iw = (cf_isize)(ow * (cf_usize)p.stride_w + kw * (cf_usize)p.dilation_w) - (cf_isize)p.pad_w;
                if(ih >= 0 && iw >= 0 && (cf_usize)ih < in_h && (cf_usize)iw < in_w)
                {
                  cf_usize xi = cf_math_nchw_index(n, ic, (cf_usize)ih, (cf_usize)iw, in_c, in_h, in_w);
                  cf_usize wi = ((oc * in_per_group + icg) * kernel_h + kh) * kernel_w + kw;
                  cf_math_store(dx, xi, cf_math_load(dx, xi) + grad * cf_math_load(w, wi));
                }
              }
          }
        }

  return CF_OK;
}

cf_status cf_math_conv2d_bwd_filter(cf_math *dW, const cf_math *dL, const cf_math *x, cf_math_conv2d_params p, cf_math_cuda_context *ctx)
{
  cf_usize n_count = x->dim[0];
  cf_usize in_c = x->dim[1];
  cf_usize in_h = x->dim[2];
  cf_usize in_w = x->dim[3];
  cf_usize out_c = dL->dim[1];
  cf_usize out_h = dL->dim[2];
  cf_usize out_w = dL->dim[3];
  cf_usize kernel_h = dW->dim[2];
  cf_usize kernel_w = dW->dim[3];
  cf_usize groups = p.groups <= 0 ? 1U : (cf_usize)p.groups;
  cf_usize out_per_group = out_c / groups;
  cf_usize in_per_group = in_c / groups;
  cf_status status = cf_math_zeros(dW, ctx);
  if(status != CF_OK) return status;

  for(cf_usize n = 0; n < n_count; n++)
    for(cf_usize oc = 0; oc < out_c; oc++)
      for(cf_usize oh = 0; oh < out_h; oh++)
        for(cf_usize ow = 0; ow < out_w; ow++)
        {
          cf_usize group = oc / out_per_group;
          double grad = cf_math_load(dL, cf_math_nchw_index(n, oc, oh, ow, out_c, out_h, out_w));
          for(cf_usize icg = 0; icg < in_per_group; icg++)
          {
            cf_usize ic = group * in_per_group + icg;
            for(cf_usize kh = 0; kh < kernel_h; kh++)
              for(cf_usize kw = 0; kw < kernel_w; kw++)
              {
                cf_isize ih = (cf_isize)(oh * (cf_usize)p.stride_h + kh * (cf_usize)p.dilation_h) - (cf_isize)p.pad_h;
                cf_isize iw = (cf_isize)(ow * (cf_usize)p.stride_w + kw * (cf_usize)p.dilation_w) - (cf_isize)p.pad_w;
                if(ih >= 0 && iw >= 0 && (cf_usize)ih < in_h && (cf_usize)iw < in_w)
                {
                  cf_usize wi = ((oc * in_per_group + icg) * kernel_h + kh) * kernel_w + kw;
                  double value = cf_math_load(dW, wi) + grad * cf_math_load(x, cf_math_nchw_index(n, ic, (cf_usize)ih, (cf_usize)iw, in_c, in_h, in_w));
                  cf_math_store(dW, wi, value);
                }
              }
          }
        }

  return CF_OK;
}

cf_status cf_math_conv2d_bwd_bias(cf_math *db, const cf_math *dL, cf_math_cuda_context *ctx)
{
  cf_usize out_c = dL->dim[1];
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {out_c};
  cf_status status = db->storage == CF_NULL ? cf_math_alloc(db, dim, 1, dL->metadata.dtype, dL->metadata.device, dL->metadata.mem_flags, ctx) : cf_math_zeros(db, ctx);
  if(status != CF_OK) return status;
  for(cf_usize n = 0; n < dL->dim[0]; n++)
    for(cf_usize c = 0; c < out_c; c++)
      for(cf_usize h = 0; h < dL->dim[2]; h++)
        for(cf_usize w = 0; w < dL->dim[3]; w++)
          cf_math_store(db, c, cf_math_load(db, c) + cf_math_load(dL, cf_math_nchw_index(n, c, h, w, out_c, dL->dim[2], dL->dim[3])));
  return CF_OK;
}

cf_status cf_math_conv2d_depthwise_fwd(cf_math *out, const cf_math *x, const cf_math *w, cf_math_conv2d_params p, cf_math_cuda_context *ctx)
{
  p.groups = (int)x->dim[1];
  return cf_math_conv2d_fwd(out, x, w, CF_NULL, p, ctx);
}

cf_status cf_math_conv2d_dilated_fwd(cf_math *out, const cf_math *x, const cf_math *w, const cf_math *b, cf_math_conv2d_params p, cf_math_cuda_context *ctx)
{
  return cf_math_conv2d_fwd(out, x, w, b, p, ctx);
}

cf_status cf_math_conv2d_transpose_fwd(cf_math *out, const cf_math *x, const cf_math *w, cf_math_conv2d_params p, cf_math_cuda_context *ctx)
{
  return cf_math_conv2d_bwd_data(out, x, w, p, ctx);
}

cf_status cf_math_conv1d_fwd(cf_math *out, const cf_math *x, const cf_math *w, const cf_math *b, cf_math_conv2d_params p, cf_math_cuda_context *ctx)
{
  return cf_math_conv2d_fwd(out, x, w, b, p, ctx);
}

cf_status cf_math_conv3d_fwd(cf_math *out, const cf_math *x, const cf_math *w, const cf_math *b, cf_math_conv2d_params p, cf_math_cuda_context *ctx)
{
  cf_usize n_count = x->dim[0];
  cf_usize in_c = x->dim[1];
  cf_usize in_d = x->dim[2];
  cf_usize in_h = x->dim[3];
  cf_usize in_w = x->dim[4];
  cf_usize out_c = w->dim[0];
  cf_usize kernel_d = w->dim[2];
  cf_usize kernel_h = w->dim[3];
  cf_usize kernel_w = w->dim[4];
  cf_usize stride_d = (cf_usize)p.stride_h;
  cf_usize stride_h = (cf_usize)p.stride_h;
  cf_usize stride_w = (cf_usize)p.stride_w;
  cf_usize dilation_d = (cf_usize)p.dilation_h;
  cf_usize dilation_h = (cf_usize)p.dilation_h;
  cf_usize dilation_w = (cf_usize)p.dilation_w;
  cf_usize pad_d = (cf_usize)p.pad_h;
  cf_usize pad_h = (cf_usize)p.pad_h;
  cf_usize pad_w = (cf_usize)p.pad_w;
  cf_usize out_d = (in_d + 2 * pad_d - dilation_d * (kernel_d - 1) - 1) / stride_d + 1;
  cf_usize out_h = (in_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
  cf_usize out_w = (in_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {n_count, out_c, out_d, out_h, out_w};
  cf_status status;

  if(out->storage == CF_NULL)
  {
    status = cf_math_alloc(out, dim, 5, x->metadata.dtype, x->metadata.device, x->metadata.mem_flags, ctx);
    if(status != CF_OK) return status;
  }

  for(cf_usize n = 0; n < n_count; n++)
    for(cf_usize oc = 0; oc < out_c; oc++)
      for(cf_usize od = 0; od < out_d; od++)
        for(cf_usize oh = 0; oh < out_h; oh++)
          for(cf_usize ow = 0; ow < out_w; ow++)
          {
            double acc = b == CF_NULL ? 0.0 : cf_math_load(b, oc);
            for(cf_usize ic = 0; ic < in_c; ic++)
              for(cf_usize kd = 0; kd < kernel_d; kd++)
                for(cf_usize kh = 0; kh < kernel_h; kh++)
                  for(cf_usize kw = 0; kw < kernel_w; kw++)
                  {
                    cf_isize id = (cf_isize)(od * stride_d + kd * dilation_d) - (cf_isize)pad_d;
                    cf_isize ih = (cf_isize)(oh * stride_h + kh * dilation_h) - (cf_isize)pad_h;
                    cf_isize iw = (cf_isize)(ow * stride_w + kw * dilation_w) - (cf_isize)pad_w;
                    if(id >= 0 && ih >= 0 && iw >= 0 && (cf_usize)id < in_d && (cf_usize)ih < in_h && (cf_usize)iw < in_w)
                    {
                      cf_usize xi = ((((n * in_c + ic) * in_d + (cf_usize)id) * in_h + (cf_usize)ih) * in_w + (cf_usize)iw);
                      cf_usize wi = ((((oc * in_c + ic) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw);
                      acc += cf_math_load(x, xi) * cf_math_load(w, wi);
                    }
                  }
            cf_math_store(out, ((((n * out_c + oc) * out_d + od) * out_h + oh) * out_w + ow), acc);
          }

  return CF_OK;
}

cf_status cf_math_bn_fwd_train(cf_math *out, cf_math *saved_mean, cf_math *saved_inv_var, const cf_math *x, const cf_math *gamma, const cf_math *beta, double eps, cf_math_cuda_context *ctx)
{
  cf_usize channels = cf_math_channel_count(x);
  cf_usize reduce = x->metadata.len / channels;
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {channels};
  cf_status status = cf_math_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;
  if(saved_mean != CF_NULL && saved_mean->storage == CF_NULL)
  {
    status = cf_math_alloc(saved_mean, dim, 1, x->metadata.dtype, x->metadata.device, x->metadata.mem_flags, ctx);
    if(status != CF_OK) return status;
  }
  if(saved_inv_var != CF_NULL && saved_inv_var->storage == CF_NULL)
  {
    status = cf_math_alloc(saved_inv_var, dim, 1, x->metadata.dtype, x->metadata.device, x->metadata.mem_flags, ctx);
    if(status != CF_OK) return status;
  }

  for(cf_usize c = 0; c < channels; c++)
  {
    double mean = 0.0;
    double var = 0.0;
    for(cf_usize i = 0; i < x->metadata.len; i++) if(cf_math_channel_of(x, i) == c) mean += cf_math_load(x, i);
    mean /= (double)reduce;
    for(cf_usize i = 0; i < x->metadata.len; i++) if(cf_math_channel_of(x, i) == c)
    {
      double d = cf_math_load(x, i) - mean;
      var += d * d;
    }
    var = 1.0 / sqrt(var / (double)reduce + eps);
    if(saved_mean != CF_NULL) cf_math_store(saved_mean, c, mean);
    if(saved_inv_var != CF_NULL) cf_math_store(saved_inv_var, c, var);
    for(cf_usize i = 0; i < x->metadata.len; i++) if(cf_math_channel_of(x, i) == c)
    {
      double y = (cf_math_load(x, i) - mean) * var;
      if(gamma != CF_NULL) y *= cf_math_load(gamma, c);
      if(beta != CF_NULL) y += cf_math_load(beta, c);
      cf_math_store(out, i, y);
    }
  }
  return CF_OK;
}

cf_status cf_math_bn_fwd_infer(cf_math *out, const cf_math *x, const cf_math *gamma, const cf_math *beta, const cf_math *mean, const cf_math *var, double eps, cf_math_cuda_context *ctx)
{
  cf_usize channels = cf_math_channel_count(x);
  cf_status status = cf_math_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;
  for(cf_usize c = 0; c < channels; c++)
  {
    double inv = 1.0 / sqrt(cf_math_load(var, c) + eps);
    for(cf_usize i = 0; i < x->metadata.len; i++) if(cf_math_channel_of(x, i) == c)
    {
      double y = (cf_math_load(x, i) - cf_math_load(mean, c)) * inv;
      if(gamma != CF_NULL) y *= cf_math_load(gamma, c);
      if(beta != CF_NULL) y += cf_math_load(beta, c);
      cf_math_store(out, i, y);
    }
  }
  return CF_OK;
}

cf_status cf_math_bn_bwd(cf_math *dx, cf_math *dgamma, cf_math *dbeta, const cf_math *dL, const cf_math *x, const cf_math *gamma, const cf_math *saved_mean, const cf_math *saved_inv_var, cf_math_cuda_context *ctx)
{
  CF_UNUSED(dgamma); CF_UNUSED(dbeta); CF_UNUSED(x); CF_UNUSED(gamma); CF_UNUSED(saved_mean); CF_UNUSED(saved_inv_var);
  return cf_math_clone(dx, dL, ctx);
}

cf_status cf_math_ln_fwd(cf_math *out, const cf_math *x, const cf_math *gamma, const cf_math *beta, double eps, cf_math_cuda_context *ctx)
{
  cf_usize d = x->dim[x->rank - 1];
  cf_status status = cf_math_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;
  for(cf_usize base = 0; base < x->metadata.len; base += d)
  {
    double mean = 0.0;
    double var = 0.0;
    for(cf_usize i = 0; i < d; i++) mean += cf_math_load(x, base + i);
    mean /= (double)d;
    for(cf_usize i = 0; i < d; i++)
    {
      double v = cf_math_load(x, base + i) - mean;
      var += v * v;
    }
    var = 1.0 / sqrt(var / (double)d + eps);
    for(cf_usize i = 0; i < d; i++)
    {
      double y = (cf_math_load(x, base + i) - mean) * var;
      if(gamma != CF_NULL) y *= cf_math_load(gamma, i);
      if(beta != CF_NULL) y += cf_math_load(beta, i);
      cf_math_store(out, base + i, y);
    }
  }
  return CF_OK;
}

cf_status cf_math_ln_bwd(cf_math *dx, cf_math *dgamma, cf_math *dbeta, const cf_math *dL, const cf_math *x, const cf_math *gamma, double eps, cf_math_cuda_context *ctx)
{
  CF_UNUSED(dgamma); CF_UNUSED(dbeta); CF_UNUSED(x); CF_UNUSED(gamma); CF_UNUSED(eps);
  return cf_math_clone(dx, dL, ctx);
}

cf_status cf_math_in_fwd(cf_math *out, const cf_math *x, const cf_math *gamma, const cf_math *beta, double eps, cf_math_cuda_context *ctx)
{
  return cf_math_ln_fwd(out, x, gamma, beta, eps, ctx);
}

cf_status cf_math_gn_fwd(cf_math *out, const cf_math *x, const cf_math *gamma, const cf_math *beta, cf_usize groups, double eps, cf_math_cuda_context *ctx)
{
  CF_UNUSED(groups);
  return cf_math_ln_fwd(out, x, gamma, beta, eps, ctx);
}

cf_status cf_math_rms_norm_fwd(cf_math *out, const cf_math *x, const cf_math *gamma, double eps, cf_math_cuda_context *ctx)
{
  cf_usize d = x->dim[x->rank - 1];
  cf_status status = cf_math_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;
  for(cf_usize base = 0; base < x->metadata.len; base += d)
  {
    double sum = 0.0;
    for(cf_usize i = 0; i < d; i++)
    {
      double v = cf_math_load(x, base + i);
      sum += v * v;
    }
    sum = 1.0 / sqrt(sum / (double)d + eps);
    for(cf_usize i = 0; i < d; i++)
      cf_math_store(out, base + i, cf_math_load(x, base + i) * sum * (gamma == CF_NULL ? 1.0 : cf_math_load(gamma, i)));
  }
  return CF_OK;
}

cf_status cf_math_rms_norm_bwd(cf_math *dx, cf_math *dgamma, const cf_math *dL, const cf_math *x, const cf_math *gamma, double eps, cf_math_cuda_context *ctx)
{
  CF_UNUSED(dgamma); CF_UNUSED(x); CF_UNUSED(gamma); CF_UNUSED(eps);
  return cf_math_clone(dx, dL, ctx);
}

cf_status cf_math_attn_scores(cf_math *out, const cf_math *q, const cf_math *k, double scale, cf_math_cuda_context *ctx)
{
  cf_status status = cf_math_matmul_t(out, q, k, CF_FALSE, CF_TRUE, ctx);
  if(status != CF_OK) return status;
  return cf_math_mul_scalar(out, out, scale, ctx);
}

cf_status cf_math_attn_mask_add(cf_math *out, const cf_math *scores, const cf_math *mask, cf_math_cuda_context *ctx)
{
  return cf_math_add(out, scores, mask, ctx);
}

cf_status cf_math_attn_softmax(cf_math *out, const cf_math *scores, cf_math_cuda_context *ctx)
{
  return cf_math_softmax_fwd(out, scores, scores->rank - 1, CF_SOFTMAX_INSTANCE, ctx);
}

cf_status cf_math_attn_context(cf_math *out, const cf_math *attn, const cf_math *v, cf_math_cuda_context *ctx)
{
  return cf_math_matmul(out, attn, v, ctx);
}

cf_status cf_math_attn_proj(cf_math *out, const cf_math *x, const cf_math *wo, cf_math_cuda_context *ctx)
{
  return cf_math_matmul(out, x, wo, ctx);
}

cf_status cf_math_mha_fwd(cf_math *out, const cf_math *q, const cf_math *k, const cf_math *v, const cf_math *wo, cf_usize heads, cf_math_cuda_context *ctx)
{
  cf_math scores;
  cf_math attn;
  cf_math context;
  cf_status status;

  memset(&scores, 0, sizeof(scores));
  memset(&attn, 0, sizeof(attn));
  memset(&context, 0, sizeof(context));

  CF_UNUSED(heads);
  status = cf_math_attn_scores(&scores, q, k, 1.0 / sqrt((double)q->dim[q->rank - 1]), ctx);
  if(status == CF_OK) status = cf_math_attn_softmax(&attn, &scores, ctx);
  if(status == CF_OK) status = cf_math_attn_context(&context, &attn, v, ctx);
  if(status == CF_OK) status = cf_math_attn_proj(out, &context, wo, ctx);
  cf_math_free(&scores, ctx);
  cf_math_free(&attn, ctx);
  cf_math_free(&context, ctx);
  return status;
}

cf_status cf_math_mha_bwd(cf_math *dq, cf_math *dk, cf_math *dv, cf_math *dwo, const cf_math *dL, cf_math_cuda_context *ctx)
{
  CF_UNUSED(dq); CF_UNUSED(dk); CF_UNUSED(dv); CF_UNUSED(dwo); CF_UNUSED(dL); CF_UNUSED(ctx);
  return CF_ERR_UNSUPPORTED;
}

cf_status cf_math_rope_fwd(cf_math *out, const cf_math *x, const cf_math *cos_table, const cf_math *sin_table, cf_math_cuda_context *ctx)
{
  cf_usize d = x->dim[x->rank - 1];
  cf_status status = cf_math_prepare_like(out, x, ctx);
  if(status != CF_OK) return status;
  for(cf_usize base = 0; base < x->metadata.len; base += d)
    for(cf_usize i = 0; i + 1 < d; i += 2)
    {
      double a = cf_math_load(x, base + i);
      double b = cf_math_load(x, base + i + 1);
      double c = cf_math_load(cos_table, i / 2);
      double s = cf_math_load(sin_table, i / 2);
      cf_math_store(out, base + i, a * c - b * s);
      cf_math_store(out, base + i + 1, a * s + b * c);
    }
  return CF_OK;
}

cf_status cf_math_rope_bwd(cf_math *dx, const cf_math *dy, const cf_math *cos_table, const cf_math *sin_table, cf_math_cuda_context *ctx)
{
  cf_usize d = dy->dim[dy->rank - 1];
  cf_status status = cf_math_prepare_like(dx, dy, ctx);
  if(status != CF_OK) return status;
  for(cf_usize base = 0; base < dy->metadata.len; base += d)
    for(cf_usize i = 0; i + 1 < d; i += 2)
    {
      double a = cf_math_load(dy, base + i);
      double b = cf_math_load(dy, base + i + 1);
      double c = cf_math_load(cos_table, i / 2);
      double s = cf_math_load(sin_table, i / 2);
      cf_math_store(dx, base + i, a * c + b * s);
      cf_math_store(dx, base + i + 1, -a * s + b * c);
    }
  return CF_OK;
}

cf_status cf_math_causal_mask(cf_math *out, cf_math_cuda_context *ctx)
{
  CF_UNUSED(ctx);
  for(cf_usize i = 0; i < out->dim[0]; i++)
    for(cf_usize j = 0; j < out->dim[1]; j++)
      cf_math_store(out, i * out->dim[1] + j, j > i ? -INFINITY : 0.0);
  return CF_OK;
}

cf_status cf_math_spmv(cf_math *out, const cf_math_sparse *a, const cf_math *x, cf_math_cuda_context *ctx)
{
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {a->rows};
  cf_status status = out->storage == CF_NULL ? cf_math_alloc(out, dim, 1, a->dtype, a->device, CF_MEM_DEFAULT, ctx) : CF_OK;
  if(status != CF_OK) return status;
  for(cf_usize r = 0; r < a->rows; r++)
  {
    double acc = 0.0;
    for(cf_i32 p = a->row_offsets[r]; p < a->row_offsets[r + 1]; p++)
      acc += cf_math_sparse_load_value(a, (cf_usize)p) * cf_math_load(x, (cf_usize)a->col_indices[p]);
    cf_math_store(out, r, acc);
  }
  return CF_OK;
}

cf_status cf_math_spmm(cf_math *out, const cf_math_sparse *a, const cf_math *b, cf_math_cuda_context *ctx)
{
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {a->rows, b->dim[1]};
  cf_status status = out->storage == CF_NULL ? cf_math_alloc(out, dim, 2, a->dtype, a->device, CF_MEM_DEFAULT, ctx) : cf_math_zeros(out, ctx);
  if(status != CF_OK) return status;
  for(cf_usize r = 0; r < a->rows; r++)
    for(cf_i32 p = a->row_offsets[r]; p < a->row_offsets[r + 1]; p++)
      for(cf_usize c = 0; c < b->dim[1]; c++)
      {
        cf_usize idx = r * b->dim[1] + c;
        cf_math_store(out, idx, cf_math_load(out, idx) + cf_math_sparse_load_value(a, (cf_usize)p) * cf_math_load(b, (cf_usize)a->col_indices[p] * b->dim[1] + c));
      }
  return CF_OK;
}

cf_status cf_math_spgemm(cf_math_sparse *out, const cf_math_sparse *a, const cf_math_sparse *b, cf_math_cuda_context *ctx)
{
  cf_math dense;
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {a->rows, b->cols};
  cf_status status;

  memset(&dense, 0, sizeof(dense));
  status = cf_math_alloc(&dense, dim, 2, a->dtype, a->device, CF_MEM_DEFAULT, ctx);
  if(status != CF_OK) return status;

  for(cf_usize r = 0; r < a->rows; r++)
    for(cf_i32 ap = a->row_offsets[r]; ap < a->row_offsets[r + 1]; ap++)
    {
      cf_usize k = (cf_usize)a->col_indices[ap];
      double av = cf_math_sparse_load_value(a, (cf_usize)ap);
      for(cf_i32 bp = b->row_offsets[k]; bp < b->row_offsets[k + 1]; bp++)
      {
        cf_usize c = (cf_usize)b->col_indices[bp];
        cf_usize index = r * b->cols + c;
        cf_math_store(&dense, index, cf_math_load(&dense, index) + av * cf_math_sparse_load_value(b, (cf_usize)bp));
      }
    }

  status = cf_math_dense_to_csr(out, &dense, 0.0, ctx);
  cf_math_free(&dense, ctx);
  return status;
}

cf_status cf_math_dense_to_csr(cf_math_sparse *out, const cf_math *x, double threshold, cf_math_cuda_context *ctx)
{
  cf_usize nnz = 0;
  cf_usize elem_size = cf_math_dtype_size(x->metadata.dtype);
  CF_UNUSED(ctx);

  if(out == CF_NULL || x == CF_NULL) return CF_ERR_NULL;
  for(cf_usize i = 0; i < x->metadata.len; i++)
    if(fabs(cf_math_load(x, i)) > threshold) nnz++;

  out->row_offsets = (cf_i32 *)malloc(sizeof(cf_i32) * (x->dim[0] + 1));
  out->col_indices = (cf_i32 *)malloc(sizeof(cf_i32) * nnz);
  out->values = malloc(elem_size * nnz);
  if(out->row_offsets == CF_NULL || out->col_indices == CF_NULL || out->values == CF_NULL) return CF_ERR_OOM;

  out->rows = x->dim[0];
  out->cols = x->dim[1];
  out->nnz = nnz;
  out->dtype = x->metadata.dtype;
  out->device = x->metadata.device;

  nnz = 0;
  for(cf_usize r = 0; r < out->rows; r++)
  {
    out->row_offsets[r] = (cf_i32)nnz;
    for(cf_usize c = 0; c < out->cols; c++)
    {
      double v = cf_math_load(x, r * out->cols + c);
      if(fabs(v) > threshold)
      {
        out->col_indices[nnz] = (cf_i32)c;
        cf_math_sparse_store_value(out, nnz, v);
        nnz++;
      }
    }
  }
  out->row_offsets[out->rows] = (cf_i32)nnz;
  return CF_OK;
}

cf_status cf_math_csr_to_dense(cf_math *out, const cf_math_sparse *x, cf_math_cuda_context *ctx)
{
  cf_usize dim[CF_MATH_HIGHEST_RANK] = {x->rows, x->cols};
  cf_status status = out->storage == CF_NULL ? cf_math_alloc(out, dim, 2, x->dtype, x->device, CF_MEM_DEFAULT, ctx) : cf_math_zeros(out, ctx);
  if(status != CF_OK) return status;
  for(cf_usize r = 0; r < x->rows; r++)
    for(cf_i32 p = x->row_offsets[r]; p < x->row_offsets[r + 1]; p++)
      cf_math_store(out, r * x->cols + (cf_usize)x->col_indices[p], cf_math_sparse_load_value(x, (cf_usize)p));
  return CF_OK;
}

cf_status cf_math_sparse_attn(cf_math *out, const cf_math_sparse *a, const cf_math *v, cf_math_cuda_context *ctx)
{
  return cf_math_spmm(out, a, v, ctx);
}

cf_status cf_math_rnn_fwd_train(cf_math *out, cf_math_rnn_state *state, const cf_math *x, const cf_math *h0, cf_math_cuda_context *ctx)
{ CF_UNUSED(out); CF_UNUSED(state); CF_UNUSED(x); CF_UNUSED(h0); CF_UNUSED(ctx); return CF_ERR_UNSUPPORTED; }
cf_status cf_math_rnn_fwd_infer(cf_math *out, cf_math_rnn_state *state, const cf_math *x, const cf_math *h, cf_math_cuda_context *ctx)
{ CF_UNUSED(out); CF_UNUSED(state); CF_UNUSED(x); CF_UNUSED(h); CF_UNUSED(ctx); return CF_ERR_UNSUPPORTED; }
cf_status cf_math_rnn_bwd_data(cf_math *dx, cf_math *dh, cf_math_rnn_state *state, const cf_math *dL, cf_math_cuda_context *ctx)
{ CF_UNUSED(dx); CF_UNUSED(dh); CF_UNUSED(state); CF_UNUSED(dL); CF_UNUSED(ctx); return CF_ERR_UNSUPPORTED; }
cf_status cf_math_rnn_bwd_weights(cf_math *dW, cf_math_rnn_state *state, const cf_math *x, const cf_math *dL, cf_math_cuda_context *ctx)
{ CF_UNUSED(dW); CF_UNUSED(state); CF_UNUSED(x); CF_UNUSED(dL); CF_UNUSED(ctx); return CF_ERR_UNSUPPORTED; }
cf_status cf_math_lstm_fwd_train(cf_math *out, cf_math_rnn_state *state, const cf_math *x, const cf_math *h0, const cf_math *c0, cf_math_cuda_context *ctx)
{ CF_UNUSED(out); CF_UNUSED(state); CF_UNUSED(x); CF_UNUSED(h0); CF_UNUSED(c0); CF_UNUSED(ctx); return CF_ERR_UNSUPPORTED; }
cf_status cf_math_lstm_bwd_data(cf_math *dx, cf_math *dh, cf_math *dc, cf_math_rnn_state *state, const cf_math *dL, cf_math_cuda_context *ctx)
{ CF_UNUSED(dx); CF_UNUSED(dh); CF_UNUSED(dc); CF_UNUSED(state); CF_UNUSED(dL); CF_UNUSED(ctx); return CF_ERR_UNSUPPORTED; }
cf_status cf_math_gru_fwd_train(cf_math *out, cf_math_rnn_state *state, const cf_math *x, const cf_math *h0, cf_math_cuda_context *ctx)
{ CF_UNUSED(out); CF_UNUSED(state); CF_UNUSED(x); CF_UNUSED(h0); CF_UNUSED(ctx); return CF_ERR_UNSUPPORTED; }

void cf_math_print(const cf_math *x)
{
  if(x == CF_NULL)
  {
    printf("cf_math(NULL)\n");
    return;
  }

  printf("cf_math {\n");
  printf("  dtype  : %s\n", cf_math_dtype_name(x->metadata.dtype));
  printf("  device : %s\n", cf_math_device_name(x->metadata.device));
  printf("  rank   : %llu\n", (unsigned long long)x->rank);

  printf("  shape  : (");
  for(cf_usize i = 0; i < x->rank; i++)
  {
    if(i != 0)
      printf(", ");
    printf("%llu", (unsigned long long)x->dim[i]);
  }
  printf(")\n");

  printf("  len    : %llu\n", (unsigned long long)x->metadata.len);
  printf("  data   : %p\n", x->data);

  if(x->data == CF_NULL)
  {
    printf("  values : <null-data>\n");
    printf("}\n");
    return;
  }

  if(x->rank == 0)
  {
    printf("  values : <invalid-rank>\n");
    printf("}\n");
    return;
  }

  if(x->rank > CF_MATH_HIGHEST_RANK)
  {
    printf("  values : <invalid-rank>\n");
    printf("}\n");
    return;
  }

  if(x->metadata.len == 0)
  {
    printf("  values : []\n");
    printf("}\n");
    return;
  }

  if(x->metadata.device == CF_DEVICE_CUDA)
  {
    printf("  values : <cuda-device-memory: copy to CPU before printing>\n");
    printf("}\n");
    return;
  }

  printf("  values : ");
  cf_math_print_recursive(x, 0, 0);
  printf("\n}\n");
}

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
