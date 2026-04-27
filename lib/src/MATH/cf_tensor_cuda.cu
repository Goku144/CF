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

#include "MATH/cf_tensor.h"

#include <cuda_runtime.h>

#define DEFINE_ADD_KERNEL(name, type) \
__global__ void name(const type *a, const type *b, type *out, cf_usize len) \
{ \
  cf_usize i = blockIdx.x * blockDim.x + threadIdx.x; \
  if(i < len) out[i] = a[i] + b[i]; \
}

DEFINE_ADD_KERNEL(cf_tensor_add_int_kernel, int)
DEFINE_ADD_KERNEL(cf_tensor_add_char_kernel, char)
DEFINE_ADD_KERNEL(cf_tensor_add_short_kernel, short)
DEFINE_ADD_KERNEL(cf_tensor_add_long_kernel, long)
DEFINE_ADD_KERNEL(cf_tensor_add_ll_kernel, long long)
DEFINE_ADD_KERNEL(cf_tensor_add_float_kernel, float)
DEFINE_ADD_KERNEL(cf_tensor_add_double_kernel, double)
DEFINE_ADD_KERNEL(cf_tensor_add_u8_kernel, cf_u8)
DEFINE_ADD_KERNEL(cf_tensor_add_u16_kernel, cf_u16)
DEFINE_ADD_KERNEL(cf_tensor_add_u32_kernel, cf_u32)
DEFINE_ADD_KERNEL(cf_tensor_add_u64_kernel, cf_u64)

#define CF_TENSOR_LAUNCH_ADD_CASE(tensor_type, kernel, type) \
case tensor_type: \
  kernel<<<blocks, threads>>>((const type *) d_a, (const type *) d_b, (type *) d_out, t1->metadata.len); \
  break

static cf_status cf_tensor_require_data(cf_tensor *tensor)
{
  if(tensor == CF_NULL) return CF_ERR_NULL;
  if(!cf_tensor_is_valid(tensor)) return CF_ERR_INVALID;
  if(tensor->data == CF_NULL) return CF_ERR_NULL;
  return CF_OK;
}

static cf_status cf_tensor_require_same_type(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out)
{
  if(t1->metadata.elem_type != t2->metadata.elem_type || t1->metadata.elem_type != t_out->metadata.elem_type)
    return CF_ERR_INVALID;

  if(t1->metadata.elem_size != t2->metadata.elem_size || t1->metadata.elem_size != t_out->metadata.elem_size)
    return CF_ERR_INVALID;

  return CF_OK;
}

extern "C" cf_status cf_tensor_add_gpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out)
{
  cf_status status = cf_tensor_require_data(t1);
  if(status != CF_OK) return status;
  status = cf_tensor_require_data(t2);
  if(status != CF_OK) return status;
  status = cf_tensor_require_data(t_out);
  if(status != CF_OK) return status;
  status = cf_tensor_require_same_type(t1, t2, t_out);
  if(status != CF_OK) return status;

  if(t1->rank != t2->rank || t1->rank != t_out->rank) return CF_ERR_INVALID;

  for (cf_usize i = 0; i < t_out->rank; i++)
    if(t1->dim[i] != t2->dim[i] || t1->dim[i] != t_out->dim[i]) return CF_ERR_INVALID;

  if(t_out->metadata.elem_type == CF_TENSOR_LD || t_out->metadata.elem_type == CF_TENSOR_U128)
    return CF_ERR_UNSUPPORTED;

  int threads = 256;
  int blocks = (t1->metadata.len + threads - 1) / threads;

  void *d_a = NULL;
  void *d_b = NULL;
  void *d_out = NULL;
  cudaError_t cuda_status;

  cf_usize bytes = t1->metadata.len * t1->metadata.elem_size;

  cuda_status = cudaMalloc(&d_a, bytes);
  if(cuda_status != cudaSuccess)
  {
    status = CF_ERR_CUDA_MEMORY;
    goto cleanup;
  }

  cuda_status = cudaMalloc(&d_b, bytes);
  if(cuda_status != cudaSuccess)
  {
    status = CF_ERR_CUDA_MEMORY;
    goto cleanup;
  }

  cuda_status = cudaMalloc(&d_out, bytes);
  if(cuda_status != cudaSuccess)
  {
    status = CF_ERR_CUDA_MEMORY;
    goto cleanup;
  }

  cuda_status = cudaMemcpy(d_a, t1->data, bytes, cudaMemcpyHostToDevice);
  if(cuda_status != cudaSuccess)
  {
    status = CF_ERR_CUDA_COPY;
    goto cleanup;
  }

  cuda_status = cudaMemcpy(d_b, t2->data, bytes, cudaMemcpyHostToDevice);
  if(cuda_status != cudaSuccess)
  {
    status = CF_ERR_CUDA_COPY;
    goto cleanup;
  }

  switch(t_out->metadata.elem_type)
  {
    CF_TENSOR_LAUNCH_ADD_CASE(CF_TENSOR_CHAR, cf_tensor_add_char_kernel, char);
    CF_TENSOR_LAUNCH_ADD_CASE(CF_TENSOR_SHORT, cf_tensor_add_short_kernel, short);
    CF_TENSOR_LAUNCH_ADD_CASE(CF_TENSOR_INT, cf_tensor_add_int_kernel, int);
    CF_TENSOR_LAUNCH_ADD_CASE(CF_TENSOR_LONG, cf_tensor_add_long_kernel, long);
    CF_TENSOR_LAUNCH_ADD_CASE(CF_TENSOR_LL, cf_tensor_add_ll_kernel, long long);
    CF_TENSOR_LAUNCH_ADD_CASE(CF_TENSOR_FLOAT, cf_tensor_add_float_kernel, float);
    CF_TENSOR_LAUNCH_ADD_CASE(CF_TENSOR_DOUBLE, cf_tensor_add_double_kernel, double);
    CF_TENSOR_LAUNCH_ADD_CASE(CF_TENSOR_U8, cf_tensor_add_u8_kernel, cf_u8);
    CF_TENSOR_LAUNCH_ADD_CASE(CF_TENSOR_U16, cf_tensor_add_u16_kernel, cf_u16);
    CF_TENSOR_LAUNCH_ADD_CASE(CF_TENSOR_U32, cf_tensor_add_u32_kernel, cf_u32);
    CF_TENSOR_LAUNCH_ADD_CASE(CF_TENSOR_U64, cf_tensor_add_u64_kernel, cf_u64);

    default:
      status = CF_ERR_UNSUPPORTED;
      goto cleanup;
  }

  cuda_status = cudaGetLastError();
  if(cuda_status != cudaSuccess)
  {
    status = CF_ERR_CUDA_LAUNCH;
    goto cleanup;
  }

  cuda_status = cudaDeviceSynchronize();
  if(cuda_status != cudaSuccess)
  {
    status = CF_ERR_CUDA_SYNC;
    goto cleanup;
  }

  cuda_status = cudaMemcpy(t_out->data, d_out, bytes, cudaMemcpyDeviceToHost);
  if(cuda_status != cudaSuccess)
  {
    status = CF_ERR_CUDA_COPY;
    goto cleanup;
  }

  status = CF_OK;

cleanup:
  if(d_a != NULL) cudaFree(d_a);
  if(d_b != NULL) cudaFree(d_b);
  if(d_out != NULL) cudaFree(d_out);
  return status;
}
extern "C" cf_status cf_tensor_scalar_mul_gpu(cf_tensor *t1, void *scalar, cf_tensor *t_out)
{
  CF_UNUSED(t1);
  CF_UNUSED(scalar);
  CF_UNUSED(t_out);
  return CF_ERR_UNSUPPORTED;
}

extern "C" cf_status cf_tensor_mul_gpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out)
{
  CF_UNUSED(t1);
  CF_UNUSED(t2);
  CF_UNUSED(t_out);
  return CF_ERR_UNSUPPORTED;
}

extern "C" cf_status cf_tensor_matrix_mul_gpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out)
{
  CF_UNUSED(t1);
  CF_UNUSED(t2);
  CF_UNUSED(t_out);
  return CF_ERR_UNSUPPORTED;
}
