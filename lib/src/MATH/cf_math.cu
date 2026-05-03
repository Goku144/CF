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

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define CF_MATH_PACKED_BINARY_KERNEL(TYPE, NAME, OP)                             \
__global__ void NAME(TYPE *__restrict__ out,                                     \
                     const TYPE *__restrict__ a,                                 \
                     const TYPE *__restrict__ b,                                 \
                     size_t vec_n)                                               \
{                                                                                \
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;                            \
  if (tid < vec_n)                                                               \
    out[tid] = OP(a[tid], b[tid]);                                               \
}

#define CF_MATH_PACKED_UNARY_KERNEL(TYPE, NAME, OP)                              \
__global__ void NAME(TYPE *__restrict__ out,                                     \
                     const TYPE *__restrict__ a,                                 \
                     size_t vec_n)                                               \
{                                                                                \
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;                            \
                                                                                 \
  if (tid < vec_n)                                                               \
    out[tid] = OP(a[tid]);                                                       \
}

#define CF_MATH_F16_VEC2_BINARY_KERNEL(NAME, OP)                                 \
  CF_MATH_PACKED_BINARY_KERNEL(__half2, NAME, OP)

#define CF_MATH_BF16_VEC2_BINARY_KERNEL(NAME, OP)                                \
  CF_MATH_PACKED_BINARY_KERNEL(__nv_bfloat162, NAME, OP)

#define CF_MATH_F16_VEC2_UNARY_KERNEL(NAME, OP)                                  \
  CF_MATH_PACKED_UNARY_KERNEL(__half2, NAME, OP)

#define CF_MATH_BF16_VEC2_UNARY_KERNEL(NAME, OP)                                 \
  CF_MATH_PACKED_UNARY_KERNEL(__nv_bfloat162, NAME, OP)

#define CF_MATH_CPU_BINARY_FUNC(TYPE, NAME, OP)                                  \
static void NAME(TYPE *__restrict__ out,                                         \
                 const TYPE *__restrict__ a,                                     \
                 const TYPE *__restrict__ b,                                     \
                 cf_usize n)                                                     \
{                                                                                \
  for (cf_usize i = 0; i < n; i++)                                               \
    out[i] = a[i] OP b[i];                                                       \
}

#define CF_MATH_CPU_UNARY_FUNC(TYPE, NAME, OP)                                   \
static void NAME(TYPE *__restrict__ out,                                         \
                 const TYPE *__restrict__ a,                                     \
                 cf_usize n)                                                     \
{                                                                                \
  for (cf_usize i = 0; i < n; i++)                                               \
    out[i] = OP a[i];                                                            \
}

#define CF_MATH_CPU_PACKED_BINARY_FUNC(TYPE, NAME, OP)                           \
static void NAME(TYPE *__restrict__ out,                                         \
                 const TYPE *__restrict__ a,                                     \
                 const TYPE *__restrict__ b,                                     \
                 cf_usize n)                                                     \
{                                                                                \
  for (cf_usize i = 0; i < n; i++)                                               \
    out[i] = OP(a[i], b[i]);                                                     \
}

#define CF_MATH_CPU_PACKED_UNARY_FUNC(TYPE, NAME, OP)                            \
static void NAME(TYPE *__restrict__ out,                                         \
                 const TYPE *__restrict__ a,                                     \
                 cf_usize n)                                                     \
{                                                                                \
  for (cf_usize i = 0; i < n; i++)                                               \
    out[i] = OP(a[i]);                                                           \
}

#define CF_MATH_F32_VEC4_BINARY_KERNEL(NAME, OP)                                 \
__global__ void NAME(float4 *__restrict__ out,                                   \
                     const float4 *__restrict__ a,                               \
                     const float4 *__restrict__ b,                               \
                     size_t vec_n)                                               \
{                                                                                \
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;                            \
                                                                                 \
  if (tid < vec_n)                                                               \
  {                                                                              \
    const float4 av = a[tid];                                                    \
    const float4 bv = b[tid];                                                    \
    out[tid] = make_float4(                                                      \
      av.x OP bv.x,                                                              \
      av.y OP bv.y,                                                              \
      av.z OP bv.z,                                                              \
      av.w OP bv.w);                                                             \
  }                                                                              \
}

#define CF_MATH_F32_VEC4_UNARY_KERNEL(NAME, OP)                                  \
__global__ void NAME(float4 *__restrict__ out,                                   \
                     const float4 *__restrict__ a,                               \
                     size_t vec_n)                                               \
{                                                                                \
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;                            \
                                                                                 \
  if (tid < vec_n)                                                               \
  {                                                                              \
    const float4 av = a[tid];                                                    \
    out[tid] = make_float4(OP av.x, OP av.y, OP av.z, OP av.w);                  \
  }                                                                              \
}

#define CF_MATH_F64_VEC2_BINARY_KERNEL(NAME, OP)                                 \
__global__ void NAME(double2 *__restrict__ out,                                  \
                     const double2 *__restrict__ a,                              \
                     const double2 *__restrict__ b,                              \
                     size_t vec_n)                                               \
{                                                                                \
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;                            \
                                                                                 \
  if (tid < vec_n)                                                               \
  {                                                                              \
    const double2 av = a[tid];                                                   \
    const double2 bv = b[tid];                                                   \
    out[tid] = make_double2(                                                     \
      av.x OP bv.x,                                                              \
      av.y OP bv.y);                                                             \
  }                                                                              \
}

#define CF_MATH_F64_VEC2_UNARY_KERNEL(NAME, OP)                                  \
__global__ void NAME(double2 *__restrict__ out,                                  \
                     const double2 *__restrict__ a,                              \
                     size_t vec_n)                                               \
{                                                                                \
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;                            \
                                                                                 \
  if (tid < vec_n)                                                               \
  {                                                                              \
    const double2 av = a[tid];                                                   \
    out[tid] = make_double2(OP av.x, OP av.y);                                   \
  }                                                                              \
}

#define CF_MATH_I32_VEC4_BINARY_KERNEL(NAME, OP)                                 \
__global__ void NAME(int4 *__restrict__ out,                                     \
                     const int4 *__restrict__ a,                                 \
                     const int4 *__restrict__ b,                                 \
                     size_t vec_n)                                               \
{                                                                                \
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;                            \
                                                                                 \
  if (tid < vec_n)                                                               \
  {                                                                              \
    const int4 av = a[tid];                                                      \
    const int4 bv = b[tid];                                                      \
    out[tid] = make_int4(                                                        \
      av.x OP bv.x,                                                              \
      av.y OP bv.y,                                                              \
      av.z OP bv.z,                                                              \
      av.w OP bv.w);                                                             \
  }                                                                              \
}

#define CF_MATH_I32_VEC4_UNARY_KERNEL(NAME, OP)                                  \
__global__ void NAME(int4 *__restrict__ out,                                     \
                     const int4 *__restrict__ a,                                 \
                     size_t vec_n)                                               \
{                                                                                \
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;                            \
                                                                                 \
  if (tid < vec_n)                                                               \
  {                                                                              \
    const int4 av = a[tid];                                                      \
    out[tid] = make_int4(OP av.x, OP av.y, OP av.z, OP av.w);                    \
  }                                                                              \
}

CF_MATH_CPU_BINARY_FUNC(float, cf_math_add_f32_cpu, +)
CF_MATH_CPU_BINARY_FUNC(float, cf_math_sub_f32_cpu, -)
CF_MATH_CPU_BINARY_FUNC(float, cf_math_mul_f32_cpu, *)
CF_MATH_CPU_BINARY_FUNC(float, cf_math_div_f32_cpu, /)
CF_MATH_CPU_UNARY_FUNC(float, cf_math_neg_f32_cpu, -)

CF_MATH_CPU_BINARY_FUNC(double, cf_math_add_f64_cpu, +)
CF_MATH_CPU_BINARY_FUNC(double, cf_math_sub_f64_cpu, -)
CF_MATH_CPU_BINARY_FUNC(double, cf_math_mul_f64_cpu, *)
CF_MATH_CPU_BINARY_FUNC(double, cf_math_div_f64_cpu, /)
CF_MATH_CPU_UNARY_FUNC(double, cf_math_neg_f64_cpu, -)

CF_MATH_CPU_BINARY_FUNC(cf_i32, cf_math_add_i32_cpu, +)
CF_MATH_CPU_BINARY_FUNC(cf_i32, cf_math_sub_i32_cpu, -)
CF_MATH_CPU_BINARY_FUNC(cf_i32, cf_math_mul_i32_cpu, *)
CF_MATH_CPU_BINARY_FUNC(cf_i32, cf_math_div_i32_cpu, /)
CF_MATH_CPU_UNARY_FUNC(cf_i32, cf_math_neg_i32_cpu, -)

CF_MATH_CPU_PACKED_BINARY_FUNC(__half2, cf_math_add_f16_cpu, __hadd2)
CF_MATH_CPU_PACKED_BINARY_FUNC(__half2, cf_math_sub_f16_cpu, __hsub2)
CF_MATH_CPU_PACKED_BINARY_FUNC(__half2, cf_math_mul_f16_cpu, __hmul2)
CF_MATH_CPU_PACKED_BINARY_FUNC(__half2, cf_math_div_f16_cpu, __h2div)
CF_MATH_CPU_PACKED_UNARY_FUNC(__half2, cf_math_neg_f16_cpu, __hneg2)

CF_MATH_CPU_PACKED_BINARY_FUNC(__nv_bfloat162, cf_math_add_bf16_cpu, __hadd2)
CF_MATH_CPU_PACKED_BINARY_FUNC(__nv_bfloat162, cf_math_sub_bf16_cpu, __hsub2)
CF_MATH_CPU_PACKED_BINARY_FUNC(__nv_bfloat162, cf_math_mul_bf16_cpu, __hmul2)
CF_MATH_CPU_PACKED_BINARY_FUNC(__nv_bfloat162, cf_math_div_bf16_cpu, __h2div)
CF_MATH_CPU_PACKED_UNARY_FUNC(__nv_bfloat162, cf_math_neg_bf16_cpu, __hneg2)

CF_MATH_F32_VEC4_BINARY_KERNEL(cf_math_add_f32_vec4_kernel, +)
CF_MATH_F32_VEC4_BINARY_KERNEL(cf_math_sub_f32_vec4_kernel, -)
CF_MATH_F32_VEC4_BINARY_KERNEL(cf_math_mul_f32_vec4_kernel, *)
CF_MATH_F32_VEC4_BINARY_KERNEL(cf_math_div_f32_vec4_kernel, /)
CF_MATH_F32_VEC4_UNARY_KERNEL(cf_math_neg_f32_vec4_kernel, -)

CF_MATH_F64_VEC2_BINARY_KERNEL(cf_math_add_f64_vec2_kernel, +)
CF_MATH_F64_VEC2_BINARY_KERNEL(cf_math_sub_f64_vec2_kernel, -)
CF_MATH_F64_VEC2_BINARY_KERNEL(cf_math_mul_f64_vec2_kernel, *)
CF_MATH_F64_VEC2_BINARY_KERNEL(cf_math_div_f64_vec2_kernel, /)
CF_MATH_F64_VEC2_UNARY_KERNEL(cf_math_neg_f64_vec2_kernel, -)

CF_MATH_I32_VEC4_BINARY_KERNEL(cf_math_add_i32_vec4_kernel, +)
CF_MATH_I32_VEC4_BINARY_KERNEL(cf_math_sub_i32_vec4_kernel, -)
CF_MATH_I32_VEC4_BINARY_KERNEL(cf_math_mul_i32_vec4_kernel, *)
CF_MATH_I32_VEC4_BINARY_KERNEL(cf_math_div_i32_vec4_kernel, /)
CF_MATH_I32_VEC4_UNARY_KERNEL(cf_math_neg_i32_vec4_kernel, -)

CF_MATH_F16_VEC2_BINARY_KERNEL(cf_math_add_f16_vec2_kernel, __hadd2)
CF_MATH_F16_VEC2_BINARY_KERNEL(cf_math_sub_f16_vec2_kernel, __hsub2)
CF_MATH_F16_VEC2_BINARY_KERNEL(cf_math_mul_f16_vec2_kernel, __hmul2)
CF_MATH_F16_VEC2_BINARY_KERNEL(cf_math_div_f16_vec2_kernel, __h2div)
CF_MATH_F16_VEC2_UNARY_KERNEL(cf_math_neg_f16_vec2_kernel, __hneg2)

CF_MATH_BF16_VEC2_BINARY_KERNEL(cf_math_add_bf16_vec2_kernel, __hadd2)
CF_MATH_BF16_VEC2_BINARY_KERNEL(cf_math_sub_bf16_vec2_kernel, __hsub2)
CF_MATH_BF16_VEC2_BINARY_KERNEL(cf_math_mul_bf16_vec2_kernel, __hmul2)
CF_MATH_BF16_VEC2_BINARY_KERNEL(cf_math_div_bf16_vec2_kernel, __h2div)
CF_MATH_BF16_VEC2_UNARY_KERNEL(cf_math_neg_bf16_vec2_kernel, __hneg2)

static cudnnDataType_t cf_math_cudnn_dtype(cf_math_dtype dtype)
{
  switch (dtype)
  {
    case CF_MATH_DTYPE_I8: return CUDNN_DATA_INT8;
    case CF_MATH_DTYPE_U8: return CUDNN_DATA_UINT8;
    case CF_MATH_DTYPE_I32: return CUDNN_DATA_INT32;
    case CF_MATH_DTYPE_BF16: return CUDNN_DATA_BFLOAT16;
    case CF_MATH_DTYPE_F16: return CUDNN_DATA_HALF;
    case CF_MATH_DTYPE_F32: return CUDNN_DATA_FLOAT;
    case CF_MATH_DTYPE_F64: return CUDNN_DATA_DOUBLE;
    default: return CUDNN_DATA_FLOAT;
  }
}

static cudaDataType_t cf_math_cuda_dtype(cf_math_dtype dtype)
{
  switch (dtype)
  {
    case CF_MATH_DTYPE_BOOL: return CUDA_R_8U;
    case CF_MATH_DTYPE_I8: return CUDA_R_8I;
    case CF_MATH_DTYPE_U8: return CUDA_R_8U;
    case CF_MATH_DTYPE_I32: return CUDA_R_32I;
    case CF_MATH_DTYPE_BF16: return CUDA_R_16BF;
    case CF_MATH_DTYPE_F16: return CUDA_R_16F;
    case CF_MATH_DTYPE_F32: return CUDA_R_32F;
    case CF_MATH_DTYPE_F64: return CUDA_R_64F;
    default: return CUDA_R_32F;
  }
}

static dnnl_data_type_t cf_math_dnnl_dtype(cf_math_dtype dtype)
{
  switch (dtype)
  {
    case CF_MATH_DTYPE_I8: return dnnl_s8;
    case CF_MATH_DTYPE_U8: return dnnl_u8;
    case CF_MATH_DTYPE_I32: return dnnl_s32;
    case CF_MATH_DTYPE_BF16: return dnnl_bf16;
    case CF_MATH_DTYPE_F16: return dnnl_f16;
    case CF_MATH_DTYPE_F32: return dnnl_f32;
    case CF_MATH_DTYPE_F64: return dnnl_f64;
    default: return dnnl_f32;
  }
}

cf_status cf_math_desc_create(cf_math_desc *desc, int rank, const int *dim, cf_math_dtype dtype, cf_math_desc_type desc_type)
{
  if(desc == CF_NULL || dim == CF_NULL) return CF_ERR_NULL;
  cf_status state = CF_OK;
  *desc = {0};

  desc->rank = rank;
  desc->dtype = dtype;
  desc->desc_type = desc_type;

  for (int i = 0; i < rank; i++)
  {
    desc->dim[i] = dim[i];
  }
  desc->strides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; i--)
    desc->strides[i] = desc->strides[i + 1] * desc->dim[i + 1];

  switch (desc_type)
  {
    case CF_MATH_DESC_NONE:
    break;

    case CF_MATH_DESC_CUDNN:
      if(cudnnCreateTensorDescriptor(&desc->desc.cudnn_tensor) != CUDNN_STATUS_SUCCESS) { state = CF_ERR_CUDA; goto fail; }
      if(cudnnSetTensorNdDescriptor(desc->desc.cudnn_tensor, cf_math_cudnn_dtype(dtype), rank, desc->dim, desc->strides) != CUDNN_STATUS_SUCCESS) { state = CF_ERR_CUDA; goto fail; }
    break;

    case CF_MATH_DESC_LT:
      if(cublasLtMatrixLayoutCreate(&desc->desc.lt_layout, cf_math_cuda_dtype(dtype), dim[0], dim[1], desc->strides[0]) != CUBLAS_STATUS_SUCCESS) { state = CF_ERR_CUDA; goto fail; }
    break;

    case CF_MATH_DESC_DNNL:
    {
      dnnl_dims_t dnnl_dim = {0};
      dnnl_dims_t dnnl_strides = {0};
      for (int i = 0; i < rank; i++)
      {
        dnnl_dim[i] = (dnnl_dim_t) dim[i];
        dnnl_strides[i] = (dnnl_dim_t) desc->strides[i];
      }
      if(dnnl_memory_desc_create_with_strides(&desc->desc.dnnl_desc, rank, dnnl_dim, cf_math_dnnl_dtype(dtype), dnnl_strides) != dnnl_success) { state = CF_ERR_INTERNAL; goto fail; }
    }
    break;

    default: state = CF_ERR_INVALID;
  }

  return state;

fail:
  cf_math_desc_destroy(desc);
  return state;
}

void cf_math_desc_destroy(cf_math_desc *desc)
{
  if(desc == CF_NULL) return;

  switch (desc->desc_type)
  {
    case CF_MATH_DESC_CUDNN:
      if(desc->desc.cudnn_tensor != CF_NULL) cudnnDestroyTensorDescriptor(desc->desc.cudnn_tensor);
    break;

    case CF_MATH_DESC_LT:
      if(desc->desc.lt_layout != CF_NULL) cublasLtMatrixLayoutDestroy(desc->desc.lt_layout);
    break;

    case CF_MATH_DESC_DNNL:
      if(desc->desc.dnnl_desc != CF_NULL) dnnl_memory_desc_destroy(desc->desc.dnnl_desc);
    break;
  }

  *desc = (cf_math_desc) {0};
}

cf_status cf_math_bind(cf_math_handle *handle, cf_math *math, cf_math_desc *desc)
{
  if(handle == CF_NULL || math == CF_NULL || desc == CF_NULL) return CF_ERR_NULL;

  *math = (cf_math) {0};
  math->desc = desc;

  cf_status state = cf_math_handle_add(handle, math);
  if(state != CF_OK) *math = (cf_math) {0};
  return state;
}

cf_status cf_math_rebind(cf_math_handle *handle, cf_math *math, cf_math_desc *desc)
{
  return cf_math_bind(handle, math, desc);
}

void cf_math_unbind(cf_math *math)
{
  if(math == CF_NULL) return;
  *math = (cf_math) {0};
}

static cf_usize cf_math_len(const cf_math *math)
{
  return (cf_usize)math->desc->dim[0] * (cf_usize)math->desc->strides[0];
}

static void *cf_math_data(cf_math_handle *handle, const cf_math *math)
{
  return (void *)((cf_u8 *)handle->storage.backend + math->byte_offset);
}

static int cf_math_grid(cf_usize vec_n)
{
  enum { CF_MATH_BLOCK = 256 };
  return (int)((vec_n + CF_MATH_BLOCK - 1) / CF_MATH_BLOCK);
}

static cf_usize cf_math_ceil_div(cf_usize n, cf_usize d)
{
  return (n + d - 1) / d;
}

static cf_usize cf_math_round_up(cf_usize n, cf_usize d)
{
  return cf_math_ceil_div(n, d) * d;
}

static void cf_math_add(cf_math_handle *handle, void *out_data, const void *a_data, const void *b_data, cf_math_dtype dtype, cf_usize n)
{
  switch (handle->device)
  {
    case CF_MATH_DEVICE_CPU:
      switch (dtype)
      {
        case CF_MATH_DTYPE_F16: cf_math_add_f16_cpu((__half2 *)out_data, (const __half2 *)a_data, (const __half2 *)b_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_BF16: cf_math_add_bf16_cpu((__nv_bfloat162 *)out_data, (const __nv_bfloat162 *)a_data, (const __nv_bfloat162 *)b_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_F32: cf_math_add_f32_cpu((float *)out_data, (const float *)a_data, (const float *)b_data, cf_math_round_up(n, 4)); return;
        case CF_MATH_DTYPE_F64: cf_math_add_f64_cpu((double *)out_data, (const double *)a_data, (const double *)b_data, cf_math_round_up(n, 2)); return;
        case CF_MATH_DTYPE_I32: cf_math_add_i32_cpu((cf_i32 *)out_data, (const cf_i32 *)a_data, (const cf_i32 *)b_data, cf_math_round_up(n, 4)); return;
        default: return;
      }

    case CF_MATH_DEVICE_CUDA:
      switch (dtype)
      {
        case CF_MATH_DTYPE_F16: cf_math_add_f16_vec2_kernel<<<cf_math_grid(cf_math_ceil_div(n, 2)), 256, 0, handle->workspace->stream>>>((__half2 *)out_data, (const __half2 *)a_data, (const __half2 *)b_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_BF16: cf_math_add_bf16_vec2_kernel<<<cf_math_grid(cf_math_ceil_div(n, 2)), 256, 0, handle->workspace->stream>>>((__nv_bfloat162 *)out_data, (const __nv_bfloat162 *)a_data, (const __nv_bfloat162 *)b_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_F32: cf_math_add_f32_vec4_kernel<<<cf_math_grid(cf_math_ceil_div(n, 4)), 256, 0, handle->workspace->stream>>>((float4 *)out_data, (const float4 *)a_data, (const float4 *)b_data, cf_math_ceil_div(n, 4)); return;
        case CF_MATH_DTYPE_F64: cf_math_add_f64_vec2_kernel<<<cf_math_grid(cf_math_ceil_div(n, 2)), 256, 0, handle->workspace->stream>>>((double2 *)out_data, (const double2 *)a_data, (const double2 *)b_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_I32: cf_math_add_i32_vec4_kernel<<<cf_math_grid(cf_math_ceil_div(n, 4)), 256, 0, handle->workspace->stream>>>((int4 *)out_data, (const int4 *)a_data, (const int4 *)b_data, cf_math_ceil_div(n, 4)); return;
        default: return;
      }

    default: return;
  }
}

static void cf_math_sub(cf_math_handle *handle, void *out_data, const void *a_data, const void *b_data, cf_math_dtype dtype, cf_usize n)
{
  switch (handle->device)
  {
    case CF_MATH_DEVICE_CPU:
      switch (dtype)
      {
        case CF_MATH_DTYPE_F16: cf_math_sub_f16_cpu((__half2 *)out_data, (const __half2 *)a_data, (const __half2 *)b_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_BF16: cf_math_sub_bf16_cpu((__nv_bfloat162 *)out_data, (const __nv_bfloat162 *)a_data, (const __nv_bfloat162 *)b_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_F32: cf_math_sub_f32_cpu((float *)out_data, (const float *)a_data, (const float *)b_data, cf_math_round_up(n, 4)); return;
        case CF_MATH_DTYPE_F64: cf_math_sub_f64_cpu((double *)out_data, (const double *)a_data, (const double *)b_data, cf_math_round_up(n, 2)); return;
        case CF_MATH_DTYPE_I32: cf_math_sub_i32_cpu((cf_i32 *)out_data, (const cf_i32 *)a_data, (const cf_i32 *)b_data, cf_math_round_up(n, 4)); return;
        default: return;
      }

    case CF_MATH_DEVICE_CUDA:
      switch (dtype)
      {
        case CF_MATH_DTYPE_F16: cf_math_sub_f16_vec2_kernel<<<cf_math_grid(cf_math_ceil_div(n, 2)), 256, 0, handle->workspace->stream>>>((__half2 *)out_data, (const __half2 *)a_data, (const __half2 *)b_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_BF16: cf_math_sub_bf16_vec2_kernel<<<cf_math_grid(cf_math_ceil_div(n, 2)), 256, 0, handle->workspace->stream>>>((__nv_bfloat162 *)out_data, (const __nv_bfloat162 *)a_data, (const __nv_bfloat162 *)b_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_F32: cf_math_sub_f32_vec4_kernel<<<cf_math_grid(cf_math_ceil_div(n, 4)), 256, 0, handle->workspace->stream>>>((float4 *)out_data, (const float4 *)a_data, (const float4 *)b_data, cf_math_ceil_div(n, 4)); return;
        case CF_MATH_DTYPE_F64: cf_math_sub_f64_vec2_kernel<<<cf_math_grid(cf_math_ceil_div(n, 2)), 256, 0, handle->workspace->stream>>>((double2 *)out_data, (const double2 *)a_data, (const double2 *)b_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_I32: cf_math_sub_i32_vec4_kernel<<<cf_math_grid(cf_math_ceil_div(n, 4)), 256, 0, handle->workspace->stream>>>((int4 *)out_data, (const int4 *)a_data, (const int4 *)b_data, cf_math_ceil_div(n, 4)); return;
        default: return;
      }

    default: return;
  }
}

static void cf_math_mul(cf_math_handle *handle, void *out_data, const void *a_data, const void *b_data, cf_math_dtype dtype, cf_usize n)
{
  switch (handle->device)
  {
    case CF_MATH_DEVICE_CPU:
      switch (dtype)
      {
        case CF_MATH_DTYPE_F16: cf_math_mul_f16_cpu((__half2 *)out_data, (const __half2 *)a_data, (const __half2 *)b_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_BF16: cf_math_mul_bf16_cpu((__nv_bfloat162 *)out_data, (const __nv_bfloat162 *)a_data, (const __nv_bfloat162 *)b_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_F32: cf_math_mul_f32_cpu((float *)out_data, (const float *)a_data, (const float *)b_data, cf_math_round_up(n, 4)); return;
        case CF_MATH_DTYPE_F64: cf_math_mul_f64_cpu((double *)out_data, (const double *)a_data, (const double *)b_data, cf_math_round_up(n, 2)); return;
        case CF_MATH_DTYPE_I32: cf_math_mul_i32_cpu((cf_i32 *)out_data, (const cf_i32 *)a_data, (const cf_i32 *)b_data, cf_math_round_up(n, 4)); return;
        default: return;
      }

    case CF_MATH_DEVICE_CUDA:
      switch (dtype)
      {
        case CF_MATH_DTYPE_F16: cf_math_mul_f16_vec2_kernel<<<cf_math_grid(cf_math_ceil_div(n, 2)), 256, 0, handle->workspace->stream>>>((__half2 *)out_data, (const __half2 *)a_data, (const __half2 *)b_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_BF16: cf_math_mul_bf16_vec2_kernel<<<cf_math_grid(cf_math_ceil_div(n, 2)), 256, 0, handle->workspace->stream>>>((__nv_bfloat162 *)out_data, (const __nv_bfloat162 *)a_data, (const __nv_bfloat162 *)b_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_F32: cf_math_mul_f32_vec4_kernel<<<cf_math_grid(cf_math_ceil_div(n, 4)), 256, 0, handle->workspace->stream>>>((float4 *)out_data, (const float4 *)a_data, (const float4 *)b_data, cf_math_ceil_div(n, 4)); return;
        case CF_MATH_DTYPE_F64: cf_math_mul_f64_vec2_kernel<<<cf_math_grid(cf_math_ceil_div(n, 2)), 256, 0, handle->workspace->stream>>>((double2 *)out_data, (const double2 *)a_data, (const double2 *)b_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_I32: cf_math_mul_i32_vec4_kernel<<<cf_math_grid(cf_math_ceil_div(n, 4)), 256, 0, handle->workspace->stream>>>((int4 *)out_data, (const int4 *)a_data, (const int4 *)b_data, cf_math_ceil_div(n, 4)); return;
        default: return;
      }

    default: return;
  }
}

static void cf_math_div(cf_math_handle *handle, void *out_data, const void *a_data, const void *b_data, cf_math_dtype dtype, cf_usize n)
{
  switch (handle->device)
  {
    case CF_MATH_DEVICE_CPU:
      switch (dtype)
      {
        case CF_MATH_DTYPE_F16: cf_math_div_f16_cpu((__half2 *)out_data, (const __half2 *)a_data, (const __half2 *)b_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_BF16: cf_math_div_bf16_cpu((__nv_bfloat162 *)out_data, (const __nv_bfloat162 *)a_data, (const __nv_bfloat162 *)b_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_F32: cf_math_div_f32_cpu((float *)out_data, (const float *)a_data, (const float *)b_data, cf_math_round_up(n, 4)); return;
        case CF_MATH_DTYPE_F64: cf_math_div_f64_cpu((double *)out_data, (const double *)a_data, (const double *)b_data, cf_math_round_up(n, 2)); return;
        case CF_MATH_DTYPE_I32: cf_math_div_i32_cpu((cf_i32 *)out_data, (const cf_i32 *)a_data, (const cf_i32 *)b_data, cf_math_round_up(n, 4)); return;
        default: return;
      }

    case CF_MATH_DEVICE_CUDA:
      switch (dtype)
      {
        case CF_MATH_DTYPE_F16: cf_math_div_f16_vec2_kernel<<<cf_math_grid(cf_math_ceil_div(n, 2)), 256, 0, handle->workspace->stream>>>((__half2 *)out_data, (const __half2 *)a_data, (const __half2 *)b_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_BF16: cf_math_div_bf16_vec2_kernel<<<cf_math_grid(cf_math_ceil_div(n, 2)), 256, 0, handle->workspace->stream>>>((__nv_bfloat162 *)out_data, (const __nv_bfloat162 *)a_data, (const __nv_bfloat162 *)b_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_F32: cf_math_div_f32_vec4_kernel<<<cf_math_grid(cf_math_ceil_div(n, 4)), 256, 0, handle->workspace->stream>>>((float4 *)out_data, (const float4 *)a_data, (const float4 *)b_data, cf_math_ceil_div(n, 4)); return;
        case CF_MATH_DTYPE_F64: cf_math_div_f64_vec2_kernel<<<cf_math_grid(cf_math_ceil_div(n, 2)), 256, 0, handle->workspace->stream>>>((double2 *)out_data, (const double2 *)a_data, (const double2 *)b_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_I32: cf_math_div_i32_vec4_kernel<<<cf_math_grid(cf_math_ceil_div(n, 4)), 256, 0, handle->workspace->stream>>>((int4 *)out_data, (const int4 *)a_data, (const int4 *)b_data, cf_math_ceil_div(n, 4)); return;
        default: return;
      }

    default: return;
  }
}

static void cf_math_neg(cf_math_handle *handle, void *out_data, const void *a_data, cf_math_dtype dtype, cf_usize n)
{
  switch (handle->device)
  {
    case CF_MATH_DEVICE_CPU:
      switch (dtype)
      {
        case CF_MATH_DTYPE_F16: cf_math_neg_f16_cpu((__half2 *)out_data, (const __half2 *)a_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_BF16: cf_math_neg_bf16_cpu((__nv_bfloat162 *)out_data, (const __nv_bfloat162 *)a_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_F32: cf_math_neg_f32_cpu((float *)out_data, (const float *)a_data, cf_math_round_up(n, 4)); return;
        case CF_MATH_DTYPE_F64: cf_math_neg_f64_cpu((double *)out_data, (const double *)a_data, cf_math_round_up(n, 2)); return;
        case CF_MATH_DTYPE_I32: cf_math_neg_i32_cpu((cf_i32 *)out_data, (const cf_i32 *)a_data, cf_math_round_up(n, 4)); return;
        default: return;
      }

    case CF_MATH_DEVICE_CUDA:
      switch (dtype)
      {
        case CF_MATH_DTYPE_F16: cf_math_neg_f16_vec2_kernel<<<cf_math_grid(cf_math_ceil_div(n, 2)), 256, 0, handle->workspace->stream>>>((__half2 *)out_data, (const __half2 *)a_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_BF16: cf_math_neg_bf16_vec2_kernel<<<cf_math_grid(cf_math_ceil_div(n, 2)), 256, 0, handle->workspace->stream>>>((__nv_bfloat162 *)out_data, (const __nv_bfloat162 *)a_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_F32: cf_math_neg_f32_vec4_kernel<<<cf_math_grid(cf_math_ceil_div(n, 4)), 256, 0, handle->workspace->stream>>>((float4 *)out_data, (const float4 *)a_data, cf_math_ceil_div(n, 4)); return;
        case CF_MATH_DTYPE_F64: cf_math_neg_f64_vec2_kernel<<<cf_math_grid(cf_math_ceil_div(n, 2)), 256, 0, handle->workspace->stream>>>((double2 *)out_data, (const double2 *)a_data, cf_math_ceil_div(n, 2)); return;
        case CF_MATH_DTYPE_I32: cf_math_neg_i32_vec4_kernel<<<cf_math_grid(cf_math_ceil_div(n, 4)), 256, 0, handle->workspace->stream>>>((int4 *)out_data, (const int4 *)a_data, cf_math_ceil_div(n, 4)); return;
        default: return;
      }

    default: return;
  }
}

void cf_math_wise_op(cf_math_handle *handle, cf_math *out, const cf_math *a, const cf_math *b, cf_math_op_kind op)
{
  if(handle == CF_NULL || out == CF_NULL || a == CF_NULL || out->desc == CF_NULL || a->desc == CF_NULL) return;
  if(handle->device == CF_MATH_DEVICE_CUDA && handle->workspace == CF_NULL) return;

  cf_math_dtype dtype = out->desc->dtype;
  cf_usize n = cf_math_len(out);
  void *out_data = cf_math_data(handle, out);
  const void *a_data = cf_math_data(handle, a);

  if(op == CF_MATH_OP_NEG)
  {
    if(out->desc->dtype != a->desc->dtype) return;
    cf_math_neg(handle, out_data, a_data, dtype, n);
    return;
  }

  if(b == CF_NULL || b->desc == CF_NULL) return;
  if(out->desc->dtype != a->desc->dtype || out->desc->dtype != b->desc->dtype) return;

  const void *b_data = cf_math_data(handle, b);

  switch (op)
  {
    case CF_MATH_OP_ADD: cf_math_add(handle, out_data, a_data, b_data, dtype, n); return;
    case CF_MATH_OP_SUB: cf_math_sub(handle, out_data, a_data, b_data, dtype, n); return;
    case CF_MATH_OP_MUL: cf_math_mul(handle, out_data, a_data, b_data, dtype, n); return;
    case CF_MATH_OP_DIV: cf_math_div(handle, out_data, a_data, b_data, dtype, n); return;
    default: return;
  }
}
