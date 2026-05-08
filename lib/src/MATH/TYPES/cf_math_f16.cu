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

#include "MATH/TYPES/cf_math_f16.h"
#include "MATH/cf_math.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector_types.h>
#include <cuda/cmath>
#include <cub/cub.cuh>

#include <stdint.h>
#include <string.h>
#include <math.h>

#define CF_MATH_TYPE_REDUCTION_ADD_CREATE(func, type)\
struct cf_math_##type##_add\
{\
  __device__ __forceinline__ type operator()(const type &a, const type &b) const\
  {\
    return func(a, b);\
  }\
};\

CF_MATH_TYPE_REDUCTION_ADD_CREATE(__hadd2, half2)
CF_MATH_TYPE_REDUCTION_ADD_CREATE(__hadd, half)

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

CF_MATH_KERNEL_OP_CREATE(add, __hadd2)
CF_MATH_KERNEL_OP_CREATE(sub, __hsub2)
CF_MATH_KERNEL_OP_CREATE(mul, __hmul2)

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


#define CF_MATH_KERNEL_TAIL_OP_CREATE(name, op)\
__global__ void cf_math_kernel_tail_##name##_f16(__half * __restrict__ C, const __half * __restrict__ A, const __half * __restrict__ B, int N8)\
{\
  int index = threadIdx.x + blockDim.x * blockIdx.x;\
  if(index >= N8) return;\
\
  C[index] = op(A[index], B[index]);\
}\

CF_MATH_KERNEL_TAIL_OP_CREATE(add, __hadd)
CF_MATH_KERNEL_TAIL_OP_CREATE(sub, __hsub)
CF_MATH_KERNEL_TAIL_OP_CREATE(mul, __hmul)

__global__ void cf_math_kernel_tail_div_f16(__half * __restrict__ C, const __half * __restrict__ A, const __half * __restrict__ B, int N8)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index >= N8) return;

  C[index] = __hdiv(A[index], B[index]);
}

__global__ void cf_math_kernel_tail_neg_f16(__half * __restrict__ C, const __half * __restrict__ A, int N8)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index >= N8) return;

  C[index] = __hneg(A[index]);
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
    cf_math_kernel_tail_##op##_f16<<<blocks, threads, 0, handle->workspace->stream>>>(C_D + tail_start, A_D + tail_start, B_D + tail_start, tail_end);\
  }\
}\

CF_MATH_OP_CREATE(add)
CF_MATH_OP_CREATE(sub)
CF_MATH_OP_CREATE(mul)
CF_MATH_OP_CREATE(div)

#define CF_MATH_KERNEL_FUNC_HALF2_CREATE(name, func) \
static __device__ __forceinline__ half2 cf_math_half2_##name(half2 x) \
{ \
  __half lo_h = __low2half(x); \
  __half hi_h = __high2half(x); \
    float lo = __half2float(lo_h); \
    float hi = __half2float(hi_h); \
  __half lo_out = __float2half(func(lo)); \
  __half hi_out = __float2half(func(hi)); \
  return __halves2half2(lo_out, hi_out); \
} \

CF_MATH_KERNEL_FUNC_HALF2_CREATE(sqrt, sqrtf)
CF_MATH_KERNEL_FUNC_HALF2_CREATE(exp, expf)
CF_MATH_KERNEL_FUNC_HALF2_CREATE(log, logf)
CF_MATH_KERNEL_FUNC_HALF2_CREATE(tanh, tanhf)

static __device__ __forceinline__ half2 cf_math_half2_sigmoid(half2 x)
{
  __half lo_h = __low2half(x);
  __half hi_h = __high2half(x);
    float lo = __half2float(lo_h);
    float hi = __half2float(hi_h);
  __half lo_out = hrcp(__float2half(1.0f + expf(-lo)));
  __half hi_out = hrcp(__float2half(1.0f + expf(-hi)));
  return __halves2half2(lo_out, hi_out);
}

static __device__ __forceinline__ half2 cf_math_half2_gelu(half2 x)
{
    half2 x_sq = __hmul2(x, x);
    
    half2 poly = __hfma2(__float2half2_rn(CF_GELU_COEFF_B), x_sq, __float2half2_rn(CF_GELU_COEFF_A));
    poly = __hmul2(poly, x);

    float2 p_f = __half22float2(poly);
    p_f.x = tanhf(p_f.x);
    p_f.y = tanhf(p_f.y);
    
    half2 t = __float22half2_rn(p_f);
    
    half2 one = __float2half2_rn(1.0f);
    half2 half_v = __float2half2_rn(0.5f);
    
    return __hmul2(__hmul2(half_v, x), __hadd2(one, t));
}

static __device__ __forceinline__ half2 cf_math_half2_norm(half2 x, float scalar)
{
  half2 norm_rcp = __float2half2_rn(1.0f / scalar);
  return __hmul2(x, norm_rcp);
}

#define CF_MATH_KERNEL_FUNC_F16_CREATE(name) \
__global__ void cf_math_kernel_##name##_f16(uint4 * __restrict__ C, const uint4 * __restrict__ A, int N8)\
{ \
  int index = threadIdx.x + blockDim.x * blockIdx.x; \
  if(index >= N8) return; \
 \
  uint4 a = A[index]; \
  uint4 c; \
  c.x = cf_math_device_half2_to_u32(cf_math_half2_##name(cf_math_device_u32_to_half2(a.x))); \
  c.y = cf_math_device_half2_to_u32(cf_math_half2_##name(cf_math_device_u32_to_half2(a.y))); \
  c.z = cf_math_device_half2_to_u32(cf_math_half2_##name(cf_math_device_u32_to_half2(a.z))); \
  c.w = cf_math_device_half2_to_u32(cf_math_half2_##name(cf_math_device_u32_to_half2(a.w))); \
  C[index] = c; \
} \

CF_MATH_KERNEL_FUNC_F16_CREATE(sqrt)
CF_MATH_KERNEL_FUNC_F16_CREATE(exp)
CF_MATH_KERNEL_FUNC_F16_CREATE(log)
CF_MATH_KERNEL_FUNC_F16_CREATE(tanh)
CF_MATH_KERNEL_FUNC_F16_CREATE(sigmoid)
CF_MATH_KERNEL_FUNC_F16_CREATE(gelu)

__global__ void cf_math_kernel_neg_f16(uint4 * __restrict__ C, const uint4 * __restrict__ A, int N8)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index >= N8) return;

  uint4 a = A[index];
  uint4 c;
  c.x = cf_math_device_half2_to_u32(__hneg2(cf_math_device_u32_to_half2(a.x)));
  c.y = cf_math_device_half2_to_u32(__hneg2(cf_math_device_u32_to_half2(a.y)));
  c.z = cf_math_device_half2_to_u32(__hneg2(cf_math_device_u32_to_half2(a.z)));
  c.w = cf_math_device_half2_to_u32(__hneg2(cf_math_device_u32_to_half2(a.w)));
  C[index] = c;
}

__global__ void cf_math_kernel_relu_f16(uint4 * __restrict__ C, const uint4 * __restrict__ A, int N8)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index >= N8) return;

  __half2 zero = __float2half2_rn(0.0f);

  uint4 a = A[index];
  uint4 c;
  c.x = cf_math_device_half2_to_u32(__hmax2(cf_math_device_u32_to_half2(a.x), zero));
  c.y = cf_math_device_half2_to_u32(__hmax2(cf_math_device_u32_to_half2(a.y), zero));
  c.z = cf_math_device_half2_to_u32(__hmax2(cf_math_device_u32_to_half2(a.z), zero));
  c.w = cf_math_device_half2_to_u32(__hmax2(cf_math_device_u32_to_half2(a.w), zero));
  C[index] = c;
}

__global__ void cf_math_kernel_reduce_mean_f16(__half *C, int N)
{
  C[0] = __hdiv(C[0], __float2half((float)N));
}

__global__ void cf_math_kernel_norm_f16(uint4 * __restrict__ C,  const uint4 * __restrict__ A, const float scalar, int N8)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index >= N8) return;

  float s = scalar;

  uint4 a = A[index];
  uint4 c;
  c.x = cf_math_device_half2_to_u32(cf_math_half2_norm(cf_math_device_u32_to_half2(a.x), s));
  c.y = cf_math_device_half2_to_u32(cf_math_half2_norm(cf_math_device_u32_to_half2(a.y), s));
  c.z = cf_math_device_half2_to_u32(cf_math_half2_norm(cf_math_device_u32_to_half2(a.z), s));
  c.w = cf_math_device_half2_to_u32(cf_math_half2_norm(cf_math_device_u32_to_half2(a.w), s));
  C[index] = c;
}

#define CF_MATH_KERNEL_TAIL_FUNC_F16_CREATE(name, func) \
__global__ void cf_math_kernel_tail_##name##_f16(__half * __restrict__ C, const __half * __restrict__ A, int N8) \
{ \
  int index = threadIdx.x + blockDim.x * blockIdx.x; \
  if(index >= N8) return; \
 \
  float a_f = __half2float(A[index]); \
  C[index] = __float2half(func(a_f)); \
} \

CF_MATH_KERNEL_TAIL_FUNC_F16_CREATE(sqrt, sqrtf)
CF_MATH_KERNEL_TAIL_FUNC_F16_CREATE(exp, expf)
CF_MATH_KERNEL_TAIL_FUNC_F16_CREATE(log, logf)
CF_MATH_KERNEL_TAIL_FUNC_F16_CREATE(tanh, tanhf)

__global__ void cf_math_kernel_tail_relu_f16(__half * __restrict__ C, const __half * __restrict__ A, int N8)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index >= N8) return;

  __half zero = __float2half_rn(0.0f);
  C[index] = __hmax(A[index], zero);
}

__global__ void cf_math_kernel_tail_sigmoid_f16(__half * __restrict__ C, const __half * __restrict__ A, int N8)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index >= N8) return;

  float a_f = __half2float(A[index]);
  C[index] = hrcp(__float2half(1.0f + expf(-a_f)));
}

__global__ void cf_math_kernel_tail_gelu_f16(__half * __restrict__ C, const __half * __restrict__ A, int N8)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index >= N8) return;

  half a_f = A[index];
  half x_sq = __hmul(a_f , a_f);
    
  half poly = __hfma(__float2half_rn(CF_GELU_COEFF_B), x_sq, __float2half_rn(CF_GELU_COEFF_A));
  poly = __hmul(poly, a_f );

  float p_f = __half2float(poly);
  p_f = tanhf(p_f);
  
  half t = __float2half_rn(p_f);
  
  half one = __float2half_rn(1.0f);
  half half_v = __float2half_rn(0.5f);
    
  C[index] = __hmul(__hmul(half_v, a_f ), __hadd(one, t));
}

__global__ void cf_math_kernel_tail_norm_f16(__half * __restrict__ C, const __half * __restrict__ A, const float scalar, int N8)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index >= N8) return;

  float a_f = __half2float(A[index]);
  C[index] = __float2half(a_f * (1.0f / scalar));
}

#define CF_MATH_UNARY_F16_CREATE(name) \
void cf_math_##name##_f16(cf_math_handle *handle, cf_math *C, cf_math *A) \
{ \
  int N = (int) C->elem_len; \
 \
  cf_u8 *ptr = (cf_u8 *) handle->storage.backend; \
  \
  __half * A_D = (__half *) (ptr + A->byte_offset); \
  __half * C_D = (__half *) (ptr + C->byte_offset); \
 \
  int threads = N <= 256 ? 256 : N <= 512 ? 512 : 1024; \
 \
  int N8 = N / 8; \
  if(N8 > 0) \
  { \
    int blocks = cuda::ceil_div(N8, threads); \
    cf_math_kernel_##name##_f16<<<blocks, threads, 0, handle->workspace->stream>>>((uint4 *) C_D, (uint4 *) A_D, N8); \
  } \
 \
  int tail_start = N8 * 8; \
  if(tail_start < N)  \
  { \
    int tail_end = N - tail_start; \
    int blocks = cuda::ceil_div(tail_end, threads); \
    cf_math_kernel_tail_##name##_f16<<<blocks, threads, 0, handle->workspace->stream>>>(C_D + tail_start, A_D + tail_start, tail_end); \
  } \
} \

CF_MATH_UNARY_F16_CREATE(neg)
CF_MATH_UNARY_F16_CREATE(sqrt)
CF_MATH_UNARY_F16_CREATE(exp)
CF_MATH_UNARY_F16_CREATE(log)
CF_MATH_UNARY_F16_CREATE(tanh)
CF_MATH_UNARY_F16_CREATE(relu)
CF_MATH_UNARY_F16_CREATE(sigmoid)
CF_MATH_UNARY_F16_CREATE(gelu)

void cf_math_norm_f16(cf_math_handle *handle, cf_math *C, cf_math *A, const float scalar)
{
  int N = (int) C->elem_len;

  cf_u8 *ptr = (cf_u8 *) handle->storage.backend;

  __half * A_D = (__half *) (ptr + A->byte_offset);
  __half * C_D = (__half *) (ptr + C->byte_offset);

  int threads = N <= 256 ? 256 : N <= 512 ? 512 : 1024;

  int N8 = N / 8;
  if(N8 > 0)
  {
    int blocks = cuda::ceil_div(N8, threads);
    cf_math_kernel_norm_f16<<<blocks, threads, 0, handle->workspace->stream>>>((uint4 *) C_D, (uint4 *) A_D, scalar, N8);
  }

  int tail_start = N8 * 8;
  if(tail_start < N) 
  {
    int tail_end = N - tail_start;
    int blocks = cuda::ceil_div(tail_end, threads);
    cf_math_kernel_tail_norm_f16<<<blocks, threads, 0, handle->workspace->stream>>>(C_D + tail_start, A_D + tail_start, scalar, tail_end);
  }
}

void cf_math_reduce_sum_f16(cf_math_handle *handle, cf_math *C, cf_math *A)
{
  cf_u8 *ptr = (cf_u8 *)handle->storage.backend;
  __half *A_D = (__half *)(ptr + A->byte_offset);
  __half *C_D = (__half *)(ptr + C->byte_offset);

  void *temp_storage = handle->workspace->scratchpad;
  size_t temp_storage_bytes = handle->workspace->scratchpad_size;

  cudaStream_t stream = handle->workspace->stream;

  cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, A_D, C_D, (int) A->elem_len, stream);
}

void cf_math_reduce_mean_f16(cf_math_handle *handle, cf_math *C, cf_math *A)
{
  CF_UNUSED(cf_math_reduce_sum_f16(handle, C, A));

  cf_u8 *ptr = (cf_u8 *)handle->storage.backend;
  __half *C_D = (__half *)(ptr + C->byte_offset);

  cf_math_kernel_reduce_mean_f16<<<1, 1, 0, handle->workspace->stream>>>(C_D, (int)A->elem_len);
}

void cf_math_zero_f16(cf_math_handle *handle, cf_math *C)
{
  cf_u8 *ptr = (cf_u8 *)handle->storage.backend;
  void *C_D = (void *)(ptr + C->byte_offset);
  cudaMemsetAsync(C_D, 0, C->elem_len * sizeof(__half), handle->workspace->stream);
}

// ----------------------------------------------------------------------------
// Phase 1: Core AI Training Functions (Stubs)
// See implementation_plan.md for the optimal CUDA backend for each function.
// ----------------------------------------------------------------------------

void cf_math_matmul_f16(cf_math_handle *handle, cf_math *C, cf_math *A, cf_math *B)
{
  cf_u8 *ptr = (cf_u8 *)handle->storage.backend;
  __half *A_D = (__half *)(ptr + A->byte_offset);
  __half *B_D = (__half *)(ptr + B->byte_offset);
  __half *C_D = (__half *)(ptr + C->byte_offset);

  float alpha = 1.0f;
  float beta = 0.0f;

  cublasLtMatmul(
    handle->ctx->context.cuda.cublasLt,

    C->desc->cublastlt.op,

    &alpha,
    A_D,
    A->desc->cublastlt.layout,

    B_D,
    B->desc->cublastlt.layout,

    &beta,
    C_D,
    C->desc->cublastlt.layout,

    C_D,
    C->desc->cublastlt.layout,

    CF_NULL,

    handle->workspace->scratchpad,
    handle->workspace->scratchpad_size,

    handle->workspace->stream
  );
}

void cf_math_matmul_trans_b_f16(cf_math_handle *handle, cf_math *C, cf_math *A, cf_math *B)
{
  cf_u8 *ptr = (cf_u8 *)handle->storage.backend;

  __half *A_D = (__half *)(ptr + A->byte_offset);
  __half *B_D = (__half *)(ptr + B->byte_offset);
  __half *C_D = (__half *)(ptr + C->byte_offset);

  float alpha = 1.0f;
  float beta = 0.0f;

  cublasOperation_t transb = CUBLAS_OP_T;

  cublasLtMatmulDescSetAttribute(
      C->desc->cublastlt.op,
      CUBLASLT_MATMUL_DESC_TRANSB,
      &transb,
      sizeof(transb)
  );

  cublasLtMatmul(
      handle->ctx->context.cuda.cublasLt,
      C->desc->cublastlt.op,

      &alpha,
      A_D,
      A->desc->cublastlt.layout,

      B_D,
      B->desc->cublastlt.layout,

      &beta,
      C_D,
      C->desc->cublastlt.layout,

      C_D,
      C->desc->cublastlt.layout,

      CF_NULL,

      handle->workspace->scratchpad,
      handle->workspace->scratchpad_size,

      handle->workspace->stream
  );

  transb = CUBLAS_OP_N;

  cublasLtMatmulDescSetAttribute(
      C->desc->cublastlt.op,
      CUBLASLT_MATMUL_DESC_TRANSB,
      &transb,
      sizeof(transb)
  );
}


void cf_math_linear_bias_f16(cf_math_handle *handle, cf_math *Output, cf_math *Input, cf_math *Weight, cf_math *Bias)
{
  cf_u8 *ptr = (cf_u8 *)handle->storage.backend;

  __half *Input_D = (__half *)(ptr + Input->byte_offset);
  __half *Weight_D = (__half *)(ptr + Weight->byte_offset);
  __half *Bias_D = (__half *)(ptr + Bias->byte_offset);
  __half *Output_D = (__half *)(ptr + Output->byte_offset);

  float alpha = 1.0f;
  float beta = 0.0f;

  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  cublasLtMatmulDescSetAttribute(Output->desc->cublastlt.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));

  cublasLtMatmulDescSetAttribute(Output->desc->cublastlt.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &Bias_D, sizeof(Bias_D));

  cublasLtMatmul(handle->ctx->context.cuda.cublasLt, Output->desc->cublastlt.op,
      &alpha,
      Input_D,
      Input->desc->cublastlt.layout,

      Weight_D,
      Weight->desc->cublastlt.layout,

      &beta,
      Output_D,
      Output->desc->cublastlt.layout,

      Output_D,
      Output->desc->cublastlt.layout,

      CF_NULL,

      handle->workspace->scratchpad,
      handle->workspace->scratchpad_size,
      handle->workspace->stream
  );

  epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  cublasLtMatmulDescSetAttribute(
      Output->desc->cublastlt.op,
      CUBLASLT_MATMUL_DESC_EPILOGUE,
      &epilogue,
      sizeof(epilogue)
  );
}

void cf_math_layer_norm_f16(cf_math_handle *handle, cf_math *Out, cf_math *In, cf_math *Weight, cf_math *Bias, float eps)
{
  // TODO: Custom kernel. cub::BlockReduce for mean/var. Accumulate in FP32.
  (void)handle; (void)Out; (void)In; (void)Weight; (void)Bias; (void)eps;
}

void cf_math_layer_norm_backward_f16(cf_math_handle *handle, cf_math *dIn, cf_math *dWeight, cf_math *dBias, cf_math *dOut, cf_math *In, cf_math *Weight, cf_math *Mean, cf_math *Var, float eps)
{
  // TODO: Custom fused kernel
  (void)handle; (void)dIn; (void)dWeight; (void)dBias; (void)dOut; (void)In; (void)Weight; (void)Mean; (void)Var; (void)eps;
}

void cf_math_softmax_f16(cf_math_handle *handle, cf_math *Out, cf_math *In, int dim)
{
  // TODO: Custom kernel. Max trick, Exp, Sum trick.
  (void)handle; (void)Out; (void)In; (void)dim;
}

void cf_math_cross_entropy_f16(cf_math_handle *handle, cf_math *Loss, cf_math *Logits, cf_math *Targets)
{
  // TODO: Custom fused kernel (LogSoftmax + NLL)
  (void)handle; (void)Loss; (void)Logits; (void)Targets;
}

void cf_math_cross_entropy_backward_f16(cf_math_handle *handle, cf_math *dLogits, cf_math *Logits, cf_math *Targets)
{
  // TODO: Custom kernel (Probs - Targets)
  (void)handle; (void)dLogits; (void)Logits; (void)Targets;
}

void cf_math_adamw_update_f16(cf_math_handle *handle, cf_math *Weight, cf_math *Grad, cf_math *M, cf_math *V, float lr, float beta1, float beta2, float eps, float weight_decay, int step)
{
  // TODO: Custom fused kernel using float4 vectorization if contiguous
  (void)handle; (void)Weight; (void)Grad; (void)M; (void)V; (void)lr; (void)beta1; (void)beta2; (void)eps; (void)weight_decay; (void)step;
}

// layer 0

void cf_math_zero_grad_f16(cf_math_handle *handle, cf_math *Grad)
{
  cf_math_zero_f16(handle, Grad->grad_fn->grad);
}
