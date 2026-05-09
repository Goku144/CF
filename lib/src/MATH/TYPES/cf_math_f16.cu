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

static __device__ __forceinline__ half2 cf_math_half2_relu_backward(half2 dOut, half2 in)
{
  __half zero = __float2half_rn(0.0f);

  __half lo_in = __low2half(in);
  __half hi_in = __high2half(in);
  __half lo_out = __low2half(dOut);
  __half hi_out = __high2half(dOut);

  __half lo = __half2float(lo_in) > 0.0f ? lo_out : zero;
  __half hi = __half2float(hi_in) > 0.0f ? hi_out : zero;

  return __halves2half2(lo, hi);
}

static __device__ __forceinline__ half2 cf_math_half2_sgd_update(half2 weight, half2 grad, float lr)
{
  half2 neg_lr = __float2half2_rn(-lr);
  return __hfma2(grad, neg_lr, weight);
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

__global__ void cf_math_kernel_relu_backward_f16(uint4 * __restrict__ dIn, const uint4 * __restrict__ dOut, const uint4 * __restrict__ In, int N8)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index >= N8) return;

  uint4 dy = dOut[index];
  uint4 x = In[index];
  uint4 dx;
  dx.x = cf_math_device_half2_to_u32(cf_math_half2_relu_backward(cf_math_device_u32_to_half2(dy.x), cf_math_device_u32_to_half2(x.x)));
  dx.y = cf_math_device_half2_to_u32(cf_math_half2_relu_backward(cf_math_device_u32_to_half2(dy.y), cf_math_device_u32_to_half2(x.y)));
  dx.z = cf_math_device_half2_to_u32(cf_math_half2_relu_backward(cf_math_device_u32_to_half2(dy.z), cf_math_device_u32_to_half2(x.z)));
  dx.w = cf_math_device_half2_to_u32(cf_math_half2_relu_backward(cf_math_device_u32_to_half2(dy.w), cf_math_device_u32_to_half2(x.w)));
  dIn[index] = dx;
}

__global__ void cf_math_kernel_sgd_update_f16(uint4 * __restrict__ Weight, const uint4 * __restrict__ Grad, float lr, int N8)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index >= N8) return;

  uint4 w = Weight[index];
  uint4 g = Grad[index];
  uint4 out;
  out.x = cf_math_device_half2_to_u32(cf_math_half2_sgd_update(cf_math_device_u32_to_half2(w.x), cf_math_device_u32_to_half2(g.x), lr));
  out.y = cf_math_device_half2_to_u32(cf_math_half2_sgd_update(cf_math_device_u32_to_half2(w.y), cf_math_device_u32_to_half2(g.y), lr));
  out.z = cf_math_device_half2_to_u32(cf_math_half2_sgd_update(cf_math_device_u32_to_half2(w.z), cf_math_device_u32_to_half2(g.z), lr));
  out.w = cf_math_device_half2_to_u32(cf_math_half2_sgd_update(cf_math_device_u32_to_half2(w.w), cf_math_device_u32_to_half2(g.w), lr));
  Weight[index] = out;
}

__global__ void cf_math_kernel_reduce_sum_rows_f16(__half * __restrict__ C, const __half * __restrict__ A, int rows, int cols)
{
  __shared__ float smem[256 * 8];

  int col8 = blockIdx.x;
  int lane = threadIdx.x;
  int col = col8 * 8;

  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for(int row = lane; row < rows; row += blockDim.x)
  {
    uint4 chunk = *((const uint4 *)(A + row * cols + col));
    __half2 h0 = cf_math_device_u32_to_half2(chunk.x);
    __half2 h1 = cf_math_device_u32_to_half2(chunk.y);
    __half2 h2 = cf_math_device_u32_to_half2(chunk.z);
    __half2 h3 = cf_math_device_u32_to_half2(chunk.w);

    float2 f0 = __half22float2(h0);
    float2 f1 = __half22float2(h1);
    float2 f2 = __half22float2(h2);
    float2 f3 = __half22float2(h3);

    acc[0] += f0.x;
    acc[1] += f0.y;
    acc[2] += f1.x;
    acc[3] += f1.y;
    acc[4] += f2.x;
    acc[5] += f2.y;
    acc[6] += f3.x;
    acc[7] += f3.y;
  }

  #pragma unroll
  for(int i = 0; i < 8; i++)
  {
    smem[lane * 8 + i] = acc[i];
  }

  __syncthreads();

  for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
  {
    if(lane < stride)
    {
      #pragma unroll
      for(int i = 0; i < 8; i++)
      {
        smem[lane * 8 + i] += smem[(lane + stride) * 8 + i];
      }
    }

    __syncthreads();
  }

  if(lane == 0)
  {
    __half out[8];

    #pragma unroll
    for(int i = 0; i < 8; i++)
    {
      out[i] = __float2half_rn(smem[i]);
    }

    *((uint4 *)(C + col)) = *(uint4 *)&out;
  }
}

__global__ void cf_math_kernel_fused_cross_entropy(__half *dY,float *batch_losses,float *loss,const __half *P,const uint8_t *targets,int rows,float inv_batch_size)
{
  __shared__ float loss_smem[1024];

  int row = threadIdx.x;
  float row_loss = 0.0f;

  if(row < rows)
  {
    uint8_t target_class = targets[row];
    int row_offset = row * 16;

    const uint4 *P_row = (const uint4 *)(&P[row_offset]);
    uint4 *dY_row = (uint4 *)(&dY[row_offset]);

    uint4 chunk0 = P_row[0];
    uint4 chunk1 = P_row[1];

    __half *h0 = (__half *)&chunk0;
    __half *h1 = (__half *)&chunk1;

    __half out_grad0[8];
    __half out_grad1[8];

    const float epsilon = 1e-7f;

    #pragma unroll
    for(int i = 0; i < 8; i++)
    {
      float p = __half2float(h0[i]);
      float t = (i == target_class) ? 1.0f : 0.0f;

      out_grad0[i] = __float2half_rn((p - t) * inv_batch_size);

      if(t == 1.0f)
      {
        row_loss = -logf(p + epsilon);
      }
    }

    #pragma unroll
    for(int i = 0; i < 8; i++)
    {
      int class_idx = i + 8;
      float p = __half2float(h1[i]);
      float t = (class_idx == target_class) ? 1.0f : 0.0f;

      out_grad1[i] = __float2half_rn((p - t) * inv_batch_size);

      if(t == 1.0f)
      {
        row_loss = -logf(p + epsilon);
      }
    }

    dY_row[0] = *(uint4 *)&out_grad0;
    dY_row[1] = *(uint4 *)&out_grad1;

    batch_losses[row] = row_loss;
  }
  loss_smem[row] = row_loss;

  __syncthreads();

  for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
  {
    if(row < stride)
    {
      loss_smem[row] += loss_smem[row + stride];
    }

    __syncthreads();
  }

  if(row == 0)
  {
    *loss = loss_smem[0] * inv_batch_size;
  }
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

__global__ void cf_math_kernel_tail_relu_backward_f16(__half * __restrict__ dIn, const __half * __restrict__ dOut, const __half * __restrict__ In, int N)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index >= N) return;

  dIn[index] = __half2float(In[index]) > 0.0f ? dOut[index] : __float2half_rn(0.0f);
}

__global__ void cf_math_kernel_tail_sgd_update_f16(__half * __restrict__ Weight, const __half * __restrict__ Grad, float lr, int N)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index >= N) return;

  Weight[index] = __hfma(Grad[index], __float2half_rn(-lr), Weight[index]);
}

__global__ void cf_math_kernel_tail_reduce_sum_rows_f16(__half * __restrict__ C, const __half * __restrict__ A, int rows, int cols, int tail_start, int tail_cols)
{
  __shared__ float smem[256];

  int tail_col = blockIdx.x;
  int lane = threadIdx.x;
  int col = tail_start + tail_col;

  if(tail_col >= tail_cols) return;

  float acc = 0.0f;
  for(int row = lane; row < rows; row += blockDim.x)
  {
    acc += __half2float(A[row * cols + col]);
  }

  smem[lane] = acc;

  __syncthreads();

  for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
  {
    if(lane < stride)
    {
      smem[lane] += smem[lane + stride];
    }

    __syncthreads();
  }

  if(lane == 0)
  {
    C[col] = __float2half_rn(smem[0]);
  }
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

  cublasLtMatmul(handle->ctx->context.cuda.cublasLt, C->desc->cublastlt.op, &alpha, A_D, A->desc->cublastlt.layout, B_D,
     B->desc->cublastlt.layout, &beta, C_D, C->desc->cublastlt.layout, C_D, C->desc->cublastlt.layout, CF_NULL,
     handle->workspace->scratchpad, handle->workspace->scratchpad_size, handle->workspace->stream);
}

void cf_math_matmul_trans_a_f16(cf_math_handle *handle, cf_math *C, cf_math *A, cf_math *B)
{
  cf_u8 *ptr = (cf_u8 *)handle->storage.backend;

  __half *A_D = (__half *)(ptr + A->byte_offset);
  __half *B_D = (__half *)(ptr + B->byte_offset);
  __half *C_D = (__half *)(ptr + C->byte_offset);

  float alpha = 1.0f;
  float beta = 0.0f;

  cublasOperation_t transa = CUBLAS_OP_T;

  cublasLtMatmulDescSetAttribute(C->desc->cublastlt.op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));

  cublasLtMatmul(handle->ctx->context.cuda.cublasLt, C->desc->cublastlt.op, &alpha, A_D, A->desc->cublastlt.layout, B_D, B->desc->cublastlt.layout,
    &beta, C_D, C->desc->cublastlt.layout, C_D, C->desc->cublastlt.layout, CF_NULL, handle->workspace->scratchpad, handle->workspace->scratchpad_size,
    handle->workspace->stream);

  transa = CUBLAS_OP_N;

  cublasLtMatmulDescSetAttribute(C->desc->cublastlt.op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
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

  cublasLtMatmulDescSetAttribute(C->desc->cublastlt.op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));

  cublasLtMatmul(handle->ctx->context.cuda.cublasLt, C->desc->cublastlt.op, &alpha, A_D, A->desc->cublastlt.layout, B_D, B->desc->cublastlt.layout,
    &beta, C_D, C->desc->cublastlt.layout, C_D, C->desc->cublastlt.layout, CF_NULL, handle->workspace->scratchpad, handle->workspace->scratchpad_size,
    handle->workspace->stream);

  transb = CUBLAS_OP_N;

  cublasLtMatmulDescSetAttribute(C->desc->cublastlt.op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
}

// layer 0

void cf_math_zero_grad_f16(cf_math_handle *handle, cf_math *Grad)
{
  cf_math_zero_f16(handle, Grad->grad_fn->grad);
}

// layer 1

void cf_math_conv2d_f16(
  cf_math_handle *handle,
  cf_math *Out,
  cf_math *In,
  cf_math *Weight,
  int pad_h,
  int pad_w,
  int stride_h,
  int stride_w,
  int dilation_h,
  int dilation_w)
{
  cf_u8 *ptr = (cf_u8 *)handle->storage.backend;

  __half *In_D = (__half *)(ptr + In->byte_offset);
  __half *Weight_D = (__half *)(ptr + Weight->byte_offset);
  __half *Out_D = (__half *)(ptr + Out->byte_offset);

  float alpha = 1.0f;
  float beta = 0.0f;

  cudnnHandle_t cudnn = handle->ctx->context.cuda.cudnn;

  cudnnSetStream(cudnn, handle->workspace->stream);

  cudnnSetConvolution2dDescriptor(Out->desc->cudnn.conv, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

  cudnnSetConvolutionMathType(Out->desc->cudnn.conv, CUDNN_TENSOR_OP_MATH);

  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

  cudnnConvolutionForward(cudnn, &alpha, In->desc->cudnn.tensor, In_D, Weight->desc->cudnn.filter, Weight_D, Out->desc->cudnn.conv, algo, handle->workspace->scratchpad,
     handle->workspace->scratchpad_size, &beta, Out->desc->cudnn.tensor, Out_D);
}

void cf_math_pooling_f16(cf_math_handle *handle, cf_math *Out, cf_math *In, int mode, int window_h, int window_w, int pad_h, int pad_w, int stride_h, int stride_w)
{
  cf_u8 *ptr = (cf_u8 *)handle->storage.backend;

  __half *In_D = (__half *)(ptr + In->byte_offset);
  __half *Out_D = (__half *)(ptr + Out->byte_offset);

  float alpha = 1.0f;
  float beta = 0.0f;

  cudnnHandle_t cudnn = handle->ctx->context.cuda.cudnn;

  cudnnSetStream(cudnn, handle->workspace->stream);

  cudnnPoolingMode_t pooling_mode = mode == CF_MATH_POOLING_MAX ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

  cudnnSetPooling2dDescriptor(Out->desc->cudnn.pooling, pooling_mode, CUDNN_PROPAGATE_NAN, window_h, window_w, pad_h, pad_w, stride_h, stride_w);

  cudnnPoolingForward(cudnn, Out->desc->cudnn.pooling, &alpha, In->desc->cudnn.tensor, In_D, &beta, Out->desc->cudnn.tensor, Out_D );
}

// layer 2

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

  cublasLtMatmul(handle->ctx->context.cuda.cublasLt, Output->desc->cublastlt.op, &alpha, Input_D, Input->desc->cublastlt.layout,
    Weight_D, Weight->desc->cublastlt.layout, &beta, Output_D, Output->desc->cublastlt.layout, Output_D, Output->desc->cublastlt.layout,
     CF_NULL, handle->workspace->scratchpad, handle->workspace->scratchpad_size, handle->workspace->stream);

  epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  cublasLtMatmulDescSetAttribute(Output->desc->cublastlt.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
}

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

// layer 3

int cf_math_argmax_f16(cf_math_handle *handle, cf_math *A)
{
  typedef cub::KeyValuePair<int, __half> argmax_result;

  int N = (int)A->elem_len;
  if(N <= 0) return -1;

  cf_u8 *ptr = (cf_u8 *)handle->storage.backend;
  __half *A_D = (__half *)(ptr + A->byte_offset);

  cf_u8 *workspace = (cf_u8 *)handle->workspace->scratchpad;
  cf_usize result_bytes = (sizeof(argmax_result) + 255) & ~(cf_usize)255;
  if(handle->workspace->scratchpad_size <= result_bytes) return -1;

  argmax_result *Result_D = (argmax_result *)workspace;
  void *temp_storage = workspace + result_bytes;
  size_t temp_storage_bytes = (size_t)(handle->workspace->scratchpad_size - result_bytes);
  cudaStream_t stream = handle->workspace->stream;

  cub::DeviceReduce::ArgMax(temp_storage, temp_storage_bytes, A_D, Result_D, N, stream);

  argmax_result result;
  cudaMemcpyAsync(&result, Result_D, sizeof(result), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  return result.key;
}

void cf_math_softmax_f16(cf_math_handle *handle, cf_math *Out, cf_math *In, int dim)
{
  int rank = In->desc->rank;
  if(dim < 0) dim += rank;
  if(dim < 0 || dim >= rank) return;

  cf_u8 *ptr = (cf_u8 *)handle->storage.backend;
  __half *In_D = (__half *)(ptr + In->byte_offset);
  __half *Out_D = (__half *)(ptr + Out->byte_offset);

  int axis_len = In->desc->dim[dim];
  if(axis_len <= 0) return;

  int inner = In->desc->strides[dim];
  int outer = (int)(In->elem_len / ((cf_usize)axis_len * (cf_usize)inner));
  if(outer <= 0 || inner <= 0) return;

  cudnnHandle_t cudnn = handle->ctx->context.cuda.cudnn;
  cudnnSetStream(cudnn, handle->workspace->stream);

  cudnnTensorDescriptor_t tensor;
  cudnnCreateTensorDescriptor(&tensor);
  cudnnSetTensor4dDescriptor(tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, outer, axis_len, inner, 1);

  float alpha = 1.0f;
  float beta = 0.0f;

  cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, tensor, In_D, &beta, tensor, Out_D);

  cudnnDestroyTensorDescriptor(tensor);
}

// layer 4

void cf_math_fused_cross_entropy(cf_math_handle *handle, cf_math *dY, cf_math *batch_L, cf_math *loss, cf_math *P, cf_math *T)
{
  if(P->desc->rank < 2 || P->desc->dim[P->desc->rank - 1] != 16) return;

  cf_u8 *ptr = (cf_u8 *)handle->storage.backend;

  __half *dY_D = (__half *)(ptr + dY->byte_offset);
  float *batch_L_D = (float *)(ptr + batch_L->byte_offset);
  float *loss_D = (float *)(ptr + loss->byte_offset);
  const __half *P_D = (const __half *)(ptr + P->byte_offset);
  const uint8_t *T_D = (const uint8_t *)(ptr + T->byte_offset);

  int rows = P->desc->dim[0];
  if(rows <= 0 || rows > 1024) return;

  int threads = 1;
  while(threads < rows) threads <<= 1;

  const float inv_batch_size = 1.0f / (float)rows;

  cf_math_kernel_fused_cross_entropy<<<1, threads, 0, handle->workspace->stream>>>(dY_D,batch_L_D,loss_D,P_D,T_D,rows,inv_batch_size);
}

// layer 5: backward helpers

void cf_math_reduce_sum_rows_f16(cf_math_handle *handle, cf_math *C, cf_math *A)
{
  int cols = (int)C->elem_len;
  int rows = (int)(A->elem_len / C->elem_len);

  cf_u8 *ptr = (cf_u8 *)handle->storage.backend;

  __half *A_D = (__half *)(ptr + A->byte_offset);
  __half *C_D = (__half *)(ptr + C->byte_offset);

  int threads = 256;

  int cols8 = cols % 8 == 0 ? cols / 8 : 0;
  if(cols8 > 0)
  {
    cf_math_kernel_reduce_sum_rows_f16<<<cols8, threads, 0, handle->workspace->stream>>>(C_D, A_D, rows, cols);
  }

  int tail_start = cols8 * 8;
  if(tail_start < cols)
  {
    int tail_cols = cols - tail_start;
    cf_math_kernel_tail_reduce_sum_rows_f16<<<tail_cols, threads, 0, handle->workspace->stream>>>(C_D, A_D, rows, cols, tail_start, tail_cols);
  }
}

void cf_math_relu_backward_f16(cf_math_handle *handle, cf_math *dIn, cf_math *dOut, cf_math *In)
{
  int N = (int)dIn->elem_len;

  cf_u8 *ptr = (cf_u8 *)handle->storage.backend;

  __half *dIn_D = (__half *)(ptr + dIn->byte_offset);
  __half *dOut_D = (__half *)(ptr + dOut->byte_offset);
  __half *In_D = (__half *)(ptr + In->byte_offset);

  int threads = N <= 256 ? 256 : N <= 512 ? 512 : 1024;

  int N8 = N / 8;
  if(N8 > 0)
  {
    int blocks = cuda::ceil_div(N8, threads);
    cf_math_kernel_relu_backward_f16<<<blocks, threads, 0, handle->workspace->stream>>>((uint4 *)dIn_D, (uint4 *)dOut_D, (uint4 *)In_D, N8);
  }

  int tail_start = N8 * 8;
  if(tail_start < N)
  {
    int tail_end = N - tail_start;
    int blocks = cuda::ceil_div(tail_end, threads);
    cf_math_kernel_tail_relu_backward_f16<<<blocks, threads, 0, handle->workspace->stream>>>(dIn_D + tail_start, dOut_D + tail_start, In_D + tail_start, tail_end);
  }
}

void cf_math_sgd_update_f16(cf_math_handle *handle, cf_math *Weight, cf_math *Grad, float lr)
{
  int N = (int)Weight->elem_len;

  cf_u8 *ptr = (cf_u8 *)handle->storage.backend;

  __half *Weight_D = (__half *)(ptr + Weight->byte_offset);
  __half *Grad_D = (__half *)(ptr + Grad->byte_offset);

  int threads = N <= 256 ? 256 : N <= 512 ? 512 : 1024;

  int N8 = N / 8;
  if(N8 > 0)
  {
    int blocks = cuda::ceil_div(N8, threads);
    cf_math_kernel_sgd_update_f16<<<blocks, threads, 0, handle->workspace->stream>>>((uint4 *)Weight_D, (uint4 *)Grad_D, lr, N8);
  }

  int tail_start = N8 * 8;
  if(tail_start < N)
  {
    int tail_end = N - tail_start;
    int blocks = cuda::ceil_div(tail_end, threads);
    cf_math_kernel_tail_sgd_update_f16<<<blocks, threads, 0, handle->workspace->stream>>>(Weight_D + tail_start, Grad_D + tail_start, lr, tail_end);
  }
}

void cf_math_pooling_backward_f16(
  cf_math_handle *handle,
  cf_math *dIn,
  cf_math *dOut,
  cf_math *Out,
  cf_math *In,
  int mode,
  int window_h,
  int window_w,
  int pad_h,
  int pad_w,
  int stride_h,
  int stride_w)
{
  cf_u8 *ptr = (cf_u8 *)handle->storage.backend;

  __half *dIn_D = (__half *)(ptr + dIn->byte_offset);
  __half *dOut_D = (__half *)(ptr + dOut->byte_offset);
  __half *Out_D = (__half *)(ptr + Out->byte_offset);
  __half *In_D = (__half *)(ptr + In->byte_offset);

  float alpha = 1.0f;
  float beta = 0.0f;

  cudnnHandle_t cudnn = handle->ctx->context.cuda.cudnn;

  cudnnSetStream(cudnn, handle->workspace->stream);

  cudnnPoolingMode_t pooling_mode = mode == CF_MATH_POOLING_MAX ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

  cudnnSetPooling2dDescriptor(Out->desc->cudnn.pooling, pooling_mode, CUDNN_PROPAGATE_NAN, window_h, window_w, pad_h, pad_w, stride_h, stride_w);

  cudnnPoolingBackward(cudnn, Out->desc->cudnn.pooling, &alpha, Out->desc->cudnn.tensor, Out_D, dOut->desc->cudnn.tensor, dOut_D,
    In->desc->cudnn.tensor, In_D, &beta, dIn->desc->cudnn.tensor, dIn_D);
}

void cf_math_conv2d_backward_data_f16(
  cf_math_handle *handle,
  cf_math *dIn,
  cf_math *dOut,
  cf_math *Weight,
  int pad_h,
  int pad_w,
  int stride_h,
  int stride_w,
  int dilation_h,
  int dilation_w)
{
  cf_u8 *ptr = (cf_u8 *)handle->storage.backend;

  __half *dIn_D = (__half *)(ptr + dIn->byte_offset);
  __half *dOut_D = (__half *)(ptr + dOut->byte_offset);
  __half *Weight_D = (__half *)(ptr + Weight->byte_offset);

  float alpha = 1.0f;
  float beta = 0.0f;

  cudnnHandle_t cudnn = handle->ctx->context.cuda.cudnn;

  cudnnSetStream(cudnn, handle->workspace->stream);

  cudnnSetConvolution2dDescriptor(dOut->desc->cudnn.conv, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

  cudnnSetConvolutionMathType(dOut->desc->cudnn.conv, CUDNN_TENSOR_OP_MATH);

  cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;

  cudnnConvolutionBackwardData(cudnn, &alpha, Weight->desc->cudnn.filter, Weight_D, dOut->desc->cudnn.tensor, dOut_D,
    dOut->desc->cudnn.conv, algo, handle->workspace->scratchpad, handle->workspace->scratchpad_size, &beta, dIn->desc->cudnn.tensor, dIn_D);
}

void cf_math_conv2d_backward_filter_f16(
  cf_math_handle *handle,
  cf_math *dWeight,
  cf_math *dOut,
  cf_math *In,
  int pad_h,
  int pad_w,
  int stride_h,
  int stride_w,
  int dilation_h,
  int dilation_w)
{
  cf_u8 *ptr = (cf_u8 *)handle->storage.backend;

  __half *dWeight_D = (__half *)(ptr + dWeight->byte_offset);
  __half *dOut_D = (__half *)(ptr + dOut->byte_offset);
  __half *In_D = (__half *)(ptr + In->byte_offset);

  float alpha = 1.0f;
  float beta = 0.0f;

  cudnnHandle_t cudnn = handle->ctx->context.cuda.cudnn;

  cudnnSetStream(cudnn, handle->workspace->stream);

  cudnnSetConvolution2dDescriptor(dOut->desc->cudnn.conv, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

  cudnnSetConvolutionMathType(dOut->desc->cudnn.conv, CUDNN_TENSOR_OP_MATH);

  cudnnConvolutionBwdFilterAlgo_t algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

  cudnnConvolutionBackwardFilter(cudnn, &alpha, In->desc->cudnn.tensor, In_D, dOut->desc->cudnn.tensor, dOut_D,
    dOut->desc->cudnn.conv, algo, handle->workspace->scratchpad, handle->workspace->scratchpad_size, &beta, dWeight->desc->cudnn.filter, dWeight_D);
}

void cf_math_conv2d_backward_bias_f16(cf_math_handle *handle, cf_math *dBias, cf_math *dOut)
{
  cf_u8 *ptr = (cf_u8 *)handle->storage.backend;

  __half *dBias_D = (__half *)(ptr + dBias->byte_offset);
  __half *dOut_D = (__half *)(ptr + dOut->byte_offset);

  float alpha = 1.0f;
  float beta = 0.0f;

  cudnnHandle_t cudnn = handle->ctx->context.cuda.cudnn;

  cudnnSetStream(cudnn, handle->workspace->stream);

  cudnnConvolutionBackwardBias(cudnn, &alpha, dOut->desc->cudnn.tensor, dOut_D, &beta, dBias->desc->cudnn.tensor, dBias_D);
}
