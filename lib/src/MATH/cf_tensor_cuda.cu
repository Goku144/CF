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

#include <cuda_runtime_api.h>
#include <stdlib.h>

/*
 * Generate one elementwise addition kernel per supported CUDA tensor type.
 */
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

/*
 * Dispatch a tensor element type to its matching CUDA kernel launch.
 */
#define CF_TENSOR_LAUNCH_ADD_CASE(tensor_type, kernel, type) \
case tensor_type: \
  kernel<<<blocks, threads>>>((const type *) d_a, (const type *) d_b, (type *) d_out, t1->metadata.len); \
  break

/*
 * Map framework tensor types to CUDA-supported storage widths. Types that do
 * not have a CUDA backend path return 0 and are reported as unsupported.
 */
static cf_usize cf_tensor_cuda_type_size(cf_tensor_type elem_type)
{
  switch(elem_type)
  {
    case CF_TENSOR_CHAR: return sizeof(char);
    case CF_TENSOR_SHORT: return sizeof(short);
    case CF_TENSOR_INT: return sizeof(int);
    case CF_TENSOR_LONG: return sizeof(long);
    case CF_TENSOR_LL: return sizeof(long long);
    case CF_TENSOR_FLOAT: return sizeof(float);
    case CF_TENSOR_DOUBLE: return sizeof(double);
    case CF_TENSOR_U8: return sizeof(cf_u8);
    case CF_TENSOR_U16: return sizeof(cf_u16);
    case CF_TENSOR_U32: return sizeof(cf_u32);
    case CF_TENSOR_U64: return sizeof(cf_u64);
    default: return 0;
  }
}

/*
 * Build dense row-major tensor metadata for a GPU-owned tensor before device
 * memory allocation.
 */
static cf_status cf_tensor_cuda_setup_metadata(cf_tensor *tensor, cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank, cf_tensor_type elem_type)
{
  if(tensor == CF_NULL || dim == CF_NULL) return CF_ERR_NULL;
  if(rank > CF_TENSOR_HIGHEST_RANK) return CF_ERR_INVALID;

  cf_usize elem_size = cf_tensor_cuda_type_size(elem_type);
  if(elem_size == 0) return CF_ERR_UNSUPPORTED;

  *tensor = (cf_tensor) {0};
  tensor->metadata.elem_size = elem_size;
  tensor->metadata.elem_type = elem_type;

  for(cf_usize i = 0; i < rank; i++)
  {
    if(dim[i] == 0) return CF_ERR_INVALID;
    tensor->dim[i] = dim[i];
  }
  tensor->rank = rank;

  tensor->metadata.len = 1;
  cf_usize end = rank > 0 ? rank - 1 : 0;
  for(cf_usize i = 0; i < rank; i++)
  {
    tensor->metadata.stride[end - i] = tensor->metadata.len;
    if(tensor->metadata.len > CF_USIZE_MAX / dim[end - i]) return CF_ERR_OVERFLOW;
    tensor->metadata.len *= dim[end - i];
  }

  return CF_OK;
}

/*
 * Compute the byte size required by CUDA allocation/copy operations.
 */
static cf_status cf_tensor_cuda_bytes(cf_tensor *tensor, cf_usize *out_bytes)
{
  if(tensor == CF_NULL || out_bytes == CF_NULL) return CF_ERR_NULL;
  if(tensor->metadata.elem_size != 0 && tensor->metadata.len > CF_USIZE_MAX / tensor->metadata.elem_size)
    return CF_ERR_OVERFLOW;

  *out_bytes = tensor->metadata.len * tensor->metadata.elem_size;
  return CF_OK;
}

/*
 * Convert logical tensor coordinates to a flat element offset for single-value
 * host/device get and set operations.
 */
static cf_status cf_tensor_cuda_index(cf_tensor *tensor, cf_usize indexs[CF_TENSOR_HIGHEST_RANK], cf_usize *out_index)
{
  if(tensor == CF_NULL || indexs == CF_NULL || out_index == CF_NULL) return CF_ERR_NULL;

  cf_usize index = 0;
  for(cf_usize i = 0; i < tensor->rank; i++)
  {
    if(indexs[i] >= tensor->dim[i]) return CF_ERR_BOUNDS;
    index += tensor->metadata.stride[i] * indexs[i];
  }

  *out_index = index;
  return CF_OK;
}

/*
 * Minimal CUDA operation precondition: tensor metadata must be valid, storage
 * must exist somewhere, and the type must be supported by CUDA kernels.
 */
static cf_status cf_tensor_require_storage(cf_tensor *tensor)
{
  if(tensor == CF_NULL) return CF_ERR_NULL;
  if(!cf_tensor_is_valid(tensor)) return CF_ERR_INVALID;
  if(tensor->data == CF_NULL && tensor->device_data == NULL) return CF_ERR_NULL;
  if(cf_tensor_cuda_type_size(tensor->metadata.elem_type) == 0) return CF_ERR_UNSUPPORTED;
  return CF_OK;
}

/*
 * Ensure all operands in a CUDA tensor operation share element type and size.
 */
static cf_status cf_tensor_require_same_type(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out)
{
  if(t1->metadata.elem_type != t2->metadata.elem_type || t1->metadata.elem_type != t_out->metadata.elem_type)
    return CF_ERR_INVALID;

  if(t1->metadata.elem_size != t2->metadata.elem_size || t1->metadata.elem_size != t_out->metadata.elem_size)
    return CF_ERR_INVALID;

  return CF_OK;
}

/*
 * Initialize a GPU-resident tensor with zeroed CUDA device storage.
 */
extern "C" cf_status cf_tensor_init_gpu(cf_tensor *tensor, cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank, cf_tensor_type elem_type)
{
  cf_status status = cf_tensor_cuda_setup_metadata(tensor, dim, rank, elem_type);
  if(status != CF_OK) return status;

  tensor->device = CF_TENSOR_DEVICE_CUDA;

  cf_usize bytes;
  status = cf_tensor_cuda_bytes(tensor, &bytes);
  if(status != CF_OK) return status;

  cudaError_t cuda_status = cudaMalloc(&tensor->device_data, bytes);
  if(cuda_status != cudaSuccess)
  {
    *tensor = (cf_tensor) {0};
    return CF_ERR_CUDA_MEMORY;
  }

  cuda_status = cudaMemset(tensor->device_data, 0, bytes);
  if(cuda_status != cudaSuccess)
  {
    cudaFree(tensor->device_data);
    *tensor = (cf_tensor) {0};
    return CF_ERR_CUDA_MEMORY;
  }

  return CF_OK;
}

/*
 * Release both device and optional host mirrors for a CUDA tensor.
 */
extern "C" void cf_tensor_destroy_gpu(cf_tensor *tensor)
{
  if(tensor == CF_NULL) return;
  if(tensor->device_data != NULL) cudaFree(tensor->device_data);
  if(tensor->data != CF_NULL) free(tensor->data);
  *tensor = (cf_tensor) {0};
}

/*
 * Read one logical element from CUDA device storage into host memory.
 */
extern "C" cf_status cf_tensor_get_gpu(void *out_value, cf_tensor *tensor, cf_usize indexs[CF_TENSOR_HIGHEST_RANK])
{
  if(out_value == CF_NULL) return CF_ERR_NULL;
  if(tensor == CF_NULL || tensor->device_data == NULL) return CF_ERR_NULL;

  cf_usize index;
  cf_status status = cf_tensor_cuda_index(tensor, indexs, &index);
  if(status != CF_OK) return status;

  cudaError_t cuda_status = cudaMemcpy(out_value, (char *) tensor->device_data + index * tensor->metadata.elem_size, tensor->metadata.elem_size, cudaMemcpyDeviceToHost);
  return cuda_status == cudaSuccess ? CF_OK : CF_ERR_CUDA_COPY;
}

/*
 * Write one logical element from host memory into CUDA device storage.
 */
extern "C" cf_status cf_tensor_set_gpu(cf_tensor *tensor, cf_usize indexs[CF_TENSOR_HIGHEST_RANK], void *value)
{
  if(value == CF_NULL) return CF_ERR_NULL;
  if(tensor == CF_NULL || tensor->device_data == NULL) return CF_ERR_NULL;

  cf_usize index;
  cf_status status = cf_tensor_cuda_index(tensor, indexs, &index);
  if(status != CF_OK) return status;

  cudaError_t cuda_status = cudaMemcpy((char *) tensor->device_data + index * tensor->metadata.elem_size, value, tensor->metadata.elem_size, cudaMemcpyHostToDevice);
  return cuda_status == cudaSuccess ? CF_OK : CF_ERR_CUDA_COPY;
}

/*
 * Upload a CPU tensor mirror into CUDA device storage and mark CUDA active.
 */
extern "C" cf_status cf_tensor_to_gpu(cf_tensor *tensor)
{
  if(tensor == CF_NULL) return CF_ERR_NULL;
  if(!cf_tensor_is_valid(tensor)) return CF_ERR_INVALID;
  if(tensor->data == CF_NULL) return CF_ERR_STATE;
  if(cf_tensor_cuda_type_size(tensor->metadata.elem_type) == 0) return CF_ERR_UNSUPPORTED;

  cf_usize bytes;
  cf_status status = cf_tensor_cuda_bytes(tensor, &bytes);
  if(status != CF_OK) return status;

  if(tensor->device_data == NULL)
  {
    cudaError_t cuda_status = cudaMalloc(&tensor->device_data, bytes);
    if(cuda_status != cudaSuccess) return CF_ERR_CUDA_MEMORY;
  }

  cudaError_t cuda_status = cudaMemcpy(tensor->device_data, tensor->data, bytes, cudaMemcpyHostToDevice);
  if(cuda_status != cudaSuccess) return CF_ERR_CUDA_COPY;

  tensor->device = CF_TENSOR_DEVICE_CUDA;
  return CF_OK;
}

/*
 * Download CUDA device storage into a CPU mirror and mark CPU active.
 */
extern "C" cf_status cf_tensor_to_cpu(cf_tensor *tensor)
{
  if(tensor == CF_NULL) return CF_ERR_NULL;
  if(!cf_tensor_is_valid(tensor)) return CF_ERR_INVALID;
  if(tensor->device_data == NULL) return CF_ERR_NULL;

  cf_usize bytes;
  cf_status status = cf_tensor_cuda_bytes(tensor, &bytes);
  if(status != CF_OK) return status;

  if(tensor->data == CF_NULL)
  {
    tensor->data = malloc(bytes);
    if(tensor->data == CF_NULL) return CF_ERR_OOM;
  }

  cudaError_t cuda_status = cudaMemcpy(tensor->data, tensor->device_data, bytes, cudaMemcpyDeviceToHost);
  if(cuda_status != cudaSuccess) return CF_ERR_CUDA_COPY;

  tensor->device = CF_TENSOR_DEVICE_CPU;
  return CF_OK;
}

/*
 * Free only the CUDA storage side of a tensor, preserving host data when it
 * exists and resetting GPU-only tensors.
 */
extern "C" cf_status cf_tensor_free_gpu(cf_tensor *tensor)
{
  if(tensor == CF_NULL) return CF_ERR_NULL;
  if(tensor->device_data == NULL) return CF_OK;

  cudaError_t cuda_status = cudaFree(tensor->device_data);
  tensor->device_data = NULL;
  if(tensor->data == CF_NULL)
    *tensor = (cf_tensor) {0};
  else if(tensor->device == CF_TENSOR_DEVICE_CUDA)
    tensor->device = CF_TENSOR_DEVICE_CPU;

  return cuda_status == cudaSuccess ? CF_OK : CF_ERR_CUDA_MEMORY;
}

/*
 * CUDA elementwise addition. CPU-backed inputs are copied temporarily; CUDA
 * resident tensors avoid transfers and keep the result on device.
 */
extern "C" cf_status cf_tensor_add_gpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out)
{
  cf_status status = cf_tensor_require_storage(t1);
  if(status != CF_OK) return status;
  status = cf_tensor_require_storage(t2);
  if(status != CF_OK) return status;
  status = cf_tensor_require_storage(t_out);
  if(status != CF_OK) return status;
  status = cf_tensor_require_same_type(t1, t2, t_out);
  if(status != CF_OK) return status;

  if(t1->rank != t2->rank || t1->rank != t_out->rank) return CF_ERR_INVALID;
  for(cf_usize i = 0; i < t_out->rank; i++)
    if(t1->dim[i] != t2->dim[i] || t1->dim[i] != t_out->dim[i]) return CF_ERR_INVALID;

  int threads = 256;
  int blocks = (int)((t1->metadata.len + threads - 1) / threads);

  void *d_a = t1->device == CF_TENSOR_DEVICE_CUDA ? t1->device_data : NULL;
  void *d_b = t2->device == CF_TENSOR_DEVICE_CUDA ? t2->device_data : NULL;
  void *d_out = t_out->device == CF_TENSOR_DEVICE_CUDA ? t_out->device_data : NULL;
  cf_bool free_a = CF_FALSE;
  cf_bool free_b = CF_FALSE;
  cf_bool free_out = CF_FALSE;
  cudaError_t cuda_status;

  cf_usize bytes;
  status = cf_tensor_cuda_bytes(t1, &bytes);
  if(status != CF_OK) return status;

  if(d_a == NULL)
  {
    cuda_status = cudaMalloc(&d_a, bytes);
    if(cuda_status != cudaSuccess)
    {
      status = CF_ERR_CUDA_MEMORY;
      goto cleanup;
    }
    free_a = CF_TRUE;

    cuda_status = cudaMemcpy(d_a, t1->data, bytes, cudaMemcpyHostToDevice);
    if(cuda_status != cudaSuccess)
    {
      status = CF_ERR_CUDA_COPY;
      goto cleanup;
    }
  }

  if(d_b == NULL)
  {
    cuda_status = cudaMalloc(&d_b, bytes);
    if(cuda_status != cudaSuccess)
    {
      status = CF_ERR_CUDA_MEMORY;
      goto cleanup;
    }
    free_b = CF_TRUE;

    cuda_status = cudaMemcpy(d_b, t2->data, bytes, cudaMemcpyHostToDevice);
    if(cuda_status != cudaSuccess)
    {
      status = CF_ERR_CUDA_COPY;
      goto cleanup;
    }
  }

  if(d_out == NULL)
  {
    cuda_status = cudaMalloc(&d_out, bytes);
    if(cuda_status != cudaSuccess)
    {
      status = CF_ERR_CUDA_MEMORY;
      goto cleanup;
    }
    free_out = CF_TRUE;
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

  if(t_out->device == CF_TENSOR_DEVICE_CUDA && t_out->device_data == d_out)
  {
    status = CF_OK;
  }
  else
  {
    cuda_status = cudaMemcpy(t_out->data, d_out, bytes, cudaMemcpyDeviceToHost);
    status = cuda_status == cudaSuccess ? CF_OK : CF_ERR_CUDA_COPY;
  }

cleanup:
  if(free_a && d_a != NULL) cudaFree(d_a);
  if(free_b && d_b != NULL) cudaFree(d_b);
  if(free_out && d_out != NULL) cudaFree(d_out);
  return status;
}

/*
 * CUDA scalar multiplication placeholder for the public tensor backend.
 */
extern "C" cf_status cf_tensor_scalar_mul_gpu(cf_tensor *t1, void *scalar, cf_tensor *t_out)
{
  CF_UNUSED(t1);
  CF_UNUSED(scalar);
  CF_UNUSED(t_out);
  return CF_ERR_UNSUPPORTED;
}

/*
 * CUDA elementwise multiplication placeholder for the public tensor backend.
 */
extern "C" cf_status cf_tensor_mul_gpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out)
{
  CF_UNUSED(t1);
  CF_UNUSED(t2);
  CF_UNUSED(t_out);
  return CF_ERR_UNSUPPORTED;
}

/*
 * CUDA matrix multiplication placeholder. A future implementation should route
 * float/double GEMM through cuBLAS and tensor contractions through cuTENSOR.
 */
extern "C" cf_status cf_tensor_matrix_mul_gpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out)
{
  CF_UNUSED(t1);
  CF_UNUSED(t2);
  CF_UNUSED(t_out);
  return CF_ERR_UNSUPPORTED;
}
