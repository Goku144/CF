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
#include <vector_types.h>
#include <cuda/cmath>

#include <stdint.h>
#include <string.h>


#if(CF_MATH_USE_DNNL == 1)
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
#endif

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

#if(CF_MATH_USE_DNNL == 1)
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
#endif

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
#if(CF_MATH_USE_DNNL == 1)
      if(cudnnCreateTensorDescriptor(&desc->desc.cudnn_tensor) != CUDNN_STATUS_SUCCESS) { state = CF_ERR_CUDA; goto fail; }
      if(cudnnSetTensorNdDescriptor(desc->desc.cudnn_tensor, cf_math_cudnn_dtype(dtype), rank, desc->dim, desc->strides) != CUDNN_STATUS_SUCCESS) { state = CF_ERR_CUDA; goto fail; }
#endif
    break;

    case CF_MATH_DESC_LT:
      if(cublasLtMatrixLayoutCreate(&desc->desc.lt_layout, cf_math_cuda_dtype(dtype), dim[0], dim[1], desc->strides[0]) != CUBLAS_STATUS_SUCCESS) { state = CF_ERR_CUDA; goto fail; }
    break;

    case CF_MATH_DESC_DNNL:
    {
#if(CF_MATH_USE_DNNL == 1)
      dnnl_dims_t dnnl_dim = {0};
      dnnl_dims_t dnnl_strides = {0};
      for (int i = 0; i < rank; i++)
      {
        dnnl_dim[i] = (dnnl_dim_t) dim[i];
        dnnl_strides[i] = (dnnl_dim_t) desc->strides[i];
      }
      if(dnnl_memory_desc_create_with_strides(&desc->desc.dnnl_desc, rank, dnnl_dim, cf_math_dnnl_dtype(dtype), dnnl_strides) != dnnl_success) { state = CF_ERR_INTERNAL; goto fail; }
#endif
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
#if(CF_MATH_USE_DNNL == 1)
      if(desc->desc.dnnl_desc != CF_NULL) dnnl_memory_desc_destroy(desc->desc.dnnl_desc);
#endif
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

static __device__ __forceinline__ unsigned int cf_math_device_half2_to_u32(__half2 x)
{
  union
  {
    unsigned int u;
    __half2 h;
  } v;
  v.h = x;
  return v.u;
}

static __device__ __forceinline__ __half2 cf_math_device_u32_to_half2(unsigned int x)
{
  union
  {
    unsigned int u;
    __half2 h;
  } v;
  v.u = x;
  return v.h;
}

#define CF_MATH_KERNEL_OP_CREATE(name, op)\
__global__ void cf_math_kernel_##name##_f16(uint4 * __restrict__ C, const uint4 * __restrict__ A, const uint4 * __restrict__ B, int N8)\
{\
  int index = threadIdx.x + blockDim.x * blockIdx.x;\
  if(index >= N8) return;\
\
  uint4 a = A[index];\
  uint4 b = B[index];\
  uint4 c;\
  c.x = cf_math_device_half2_to_u32(op(cf_math_device_u32_to_half2(a.x), cf_math_device_u32_to_half2(b.x)));\
  c.y = cf_math_device_half2_to_u32(op(cf_math_device_u32_to_half2(a.y), cf_math_device_u32_to_half2(b.y)));\
  c.z = cf_math_device_half2_to_u32(op(cf_math_device_u32_to_half2(a.z), cf_math_device_u32_to_half2(b.z)));\
  c.w = cf_math_device_half2_to_u32(op(cf_math_device_u32_to_half2(a.w), cf_math_device_u32_to_half2(b.w)));\
\
  C[index] = c;\
}\


__global__ void cf_math_kernel_div_f16(uint4 * __restrict__ C, const uint4 * __restrict__ A, const uint4 * __restrict__ B, int N8)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index >= N8) return;

  uint4 a = A[index];
  uint4 b = B[index];
  uint4 c;
  c.x = cf_math_device_half2_to_u32(__hmul2(cf_math_device_u32_to_half2(a.x), h2rcp(cf_math_device_u32_to_half2(b.x))));
  c.y = cf_math_device_half2_to_u32(__hmul2(cf_math_device_u32_to_half2(a.y), h2rcp(cf_math_device_u32_to_half2(b.y))));
  c.z = cf_math_device_half2_to_u32(__hmul2(cf_math_device_u32_to_half2(a.z), h2rcp(cf_math_device_u32_to_half2(b.z))));
  c.w = cf_math_device_half2_to_u32(__hmul2(cf_math_device_u32_to_half2(a.w), h2rcp(cf_math_device_u32_to_half2(b.w))));
  C[index] = c;
}

__global__ void cf_math_kernel_neg_f16(uint4 * __restrict__ $A, const uint4 * __restrict__ A, int N8)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index >= N8) return;

  uint4 a = A[index];
  uint4 $a;
  $a.x = cf_math_device_half2_to_u32(__hneg2(cf_math_device_u32_to_half2(a.x)));
  $a.y = cf_math_device_half2_to_u32(__hneg2(cf_math_device_u32_to_half2(a.y)));
  $a.z = cf_math_device_half2_to_u32(__hneg2(cf_math_device_u32_to_half2(a.z)));
  $a.w = cf_math_device_half2_to_u32(__hneg2(cf_math_device_u32_to_half2(a.w)));
  $A[index] = $a;
}

#define CF_MATH_KERNEL_TAIL_OP_CREATE(name, op)\
__global__ void cf_math_kernel_tail_##name##_f16(__half * __restrict__ C, const __half * __restrict__ A, const __half * __restrict__ B, int N8)\
{\
  int index = threadIdx.x + blockDim.x * blockIdx.x;\
  if(index >= N8) return;\
\
  C[index] = op(A[index], B[index]);\
}\

__global__ void cf_math_kernel_tail_div_f16(__half * __restrict__ C, const __half * __restrict__ A, const __half * __restrict__ B, int N8)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index >= N8) return;

  C[index] = __hdiv(A[index], B[index]);
}

__global__ void cf_math_kernel_tail_neg_f16(__half * __restrict__ $A, const __half * __restrict__ A, int N8)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index >= N8) return;

  $A[index] = __hneg(A[index]);
}

#define CF_MATH_OP_CREATE(op)\
void cf_math_##op##_f16(cf_math_handle *handle, cf_math *C, cf_math *A, cf_math *B)\
{\
  int N = (int) C->elem_len;\
\
  cf_u8 *ptr = (cf_u8 *) handle->storage.backend;\
  \
  __half * A_D = (__half *) (ptr + A->byte_offset);\
  __half * B_D = (__half *) (ptr + B->byte_offset);\
  __half * C_D = (__half *) (ptr + C->byte_offset);\
\
  int threads = N <= 256 ? 256 : N <= 512 ? 512 : 1024;\
\
  int N8 = N / 8;\
  if(N8 > 0)\
  {\
    int blocks = cuda::ceil_div(N8, threads);\
    cf_math_kernel_##op##_f16<<<blocks, threads, 0, handle->workspace->stream>>>((uint4 *) C_D, (uint4 *) A_D, (uint4 *) B_D, N8);\
  }\
\
  int tail_start = N8 * 8;\
  if(tail_start < N) \
  {\
    int tail_end = N - tail_start;\
    int blocks = cuda::ceil_div(tail_end, threads);\
    cf_math_kernel_tail_##op##_f16<<<blocks, threads, 0, handle->workspace->stream>>>(C_D, A_D, B_D, tail_end);\
  }\
}\

void cf_math_neg_f16(cf_math_handle *handle, cf_math *$A, cf_math *A)
{
  int N = (int) $A->elem_len;

  cf_u8 *ptr = (cf_u8 *) handle->storage.backend;
  
  __half * A_D = (__half *) (ptr + A->byte_offset);
  __half * $A_D = (__half *) (ptr + $A->byte_offset);

  int threads = N <= 256 ? 256 : N <= 512 ? 512 : 1024;

  int N8 = N / 8;
  if(N8 > 0)
  {
    int blocks = cuda::ceil_div(N8, threads);
    cf_math_kernel_neg_f16<<<blocks, threads, 0, handle->workspace->stream>>>((uint4 *) $A_D, (uint4 *) A_D, N8);
  }

  int tail_start = N8 * 8;
  if(tail_start < N) 
  {
    int tail_end = N - tail_start;
    int blocks = cuda::ceil_div(tail_end, threads);
    cf_math_kernel_tail_neg_f16<<<blocks, threads, 0, handle->workspace->stream>>>($A_D, A_D, tail_end);
  }
}

CF_MATH_KERNEL_OP_CREATE(add, __hadd2)
CF_MATH_KERNEL_OP_CREATE(sub, __hsub2)
CF_MATH_KERNEL_OP_CREATE(mul, __hmul2)

CF_MATH_KERNEL_TAIL_OP_CREATE(add, __hadd)
CF_MATH_KERNEL_TAIL_OP_CREATE(sub, __hsub)
CF_MATH_KERNEL_TAIL_OP_CREATE(mul, __hmul)

CF_MATH_OP_CREATE(add)
CF_MATH_OP_CREATE(sub)
CF_MATH_OP_CREATE(mul)
CF_MATH_OP_CREATE(div)
