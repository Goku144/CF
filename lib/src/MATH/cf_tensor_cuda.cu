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

#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <string.h>

#define CF_TENSOR_CUDA_THREADS 256
#define CF_TENSOR_CUBLASLT_WORKSPACE_BYTES ((size_t)32 * 1024 * 1024)

extern "C" void cf_tensor_destroy_gpu(cf_tensor *tensor);
extern "C" void cf_tensor_destroy_many_gpu(cf_tensor **tensors, cf_usize count);

#define DEFINE_ADD_KERNEL(name, type) \
__global__ void name(type *a, const type *b, cf_usize len) \
{ \
  cf_usize i = blockIdx.x * blockDim.x + threadIdx.x; \
  if(i < len) a[i] = (type)(a[i] + b[i]); \
}

#define DEFINE_MUL_KERNEL(name, type) \
__global__ void name(type *a, const type *b, cf_usize len) \
{ \
  cf_usize i = blockIdx.x * blockDim.x + threadIdx.x; \
  if(i < len) a[i] = (type)(a[i] * b[i]); \
}

#define DEFINE_SCALAR_KERNEL(name, type) \
__global__ void name(type *a, type scalar, cf_usize len) \
{ \
  cf_usize i = blockIdx.x * blockDim.x + threadIdx.x; \
  if(i < len) a[i] = (type)(a[i] * scalar); \
}

DEFINE_ADD_KERNEL(cf_tensor_add_char_kernel, char)
DEFINE_ADD_KERNEL(cf_tensor_add_short_kernel, short)
DEFINE_ADD_KERNEL(cf_tensor_add_int_kernel, int)
DEFINE_ADD_KERNEL(cf_tensor_add_long_kernel, long)
DEFINE_ADD_KERNEL(cf_tensor_add_ll_kernel, long long)
DEFINE_ADD_KERNEL(cf_tensor_add_float_kernel, float)
DEFINE_ADD_KERNEL(cf_tensor_add_double_kernel, double)
DEFINE_ADD_KERNEL(cf_tensor_add_u8_kernel, cf_u8)
DEFINE_ADD_KERNEL(cf_tensor_add_u16_kernel, cf_u16)
DEFINE_ADD_KERNEL(cf_tensor_add_u32_kernel, cf_u32)
DEFINE_ADD_KERNEL(cf_tensor_add_u64_kernel, cf_u64)

DEFINE_MUL_KERNEL(cf_tensor_mul_char_kernel, char)
DEFINE_MUL_KERNEL(cf_tensor_mul_short_kernel, short)
DEFINE_MUL_KERNEL(cf_tensor_mul_int_kernel, int)
DEFINE_MUL_KERNEL(cf_tensor_mul_long_kernel, long)
DEFINE_MUL_KERNEL(cf_tensor_mul_ll_kernel, long long)
DEFINE_MUL_KERNEL(cf_tensor_mul_float_kernel, float)
DEFINE_MUL_KERNEL(cf_tensor_mul_double_kernel, double)
DEFINE_MUL_KERNEL(cf_tensor_mul_u8_kernel, cf_u8)
DEFINE_MUL_KERNEL(cf_tensor_mul_u16_kernel, cf_u16)
DEFINE_MUL_KERNEL(cf_tensor_mul_u32_kernel, cf_u32)
DEFINE_MUL_KERNEL(cf_tensor_mul_u64_kernel, cf_u64)

DEFINE_SCALAR_KERNEL(cf_tensor_scalar_char_kernel, char)
DEFINE_SCALAR_KERNEL(cf_tensor_scalar_short_kernel, short)
DEFINE_SCALAR_KERNEL(cf_tensor_scalar_int_kernel, int)
DEFINE_SCALAR_KERNEL(cf_tensor_scalar_long_kernel, long)
DEFINE_SCALAR_KERNEL(cf_tensor_scalar_ll_kernel, long long)
DEFINE_SCALAR_KERNEL(cf_tensor_scalar_float_kernel, float)
DEFINE_SCALAR_KERNEL(cf_tensor_scalar_double_kernel, double)
DEFINE_SCALAR_KERNEL(cf_tensor_scalar_u8_kernel, cf_u8)
DEFINE_SCALAR_KERNEL(cf_tensor_scalar_u16_kernel, cf_u16)
DEFINE_SCALAR_KERNEL(cf_tensor_scalar_u32_kernel, cf_u32)
DEFINE_SCALAR_KERNEL(cf_tensor_scalar_u64_kernel, cf_u64)

#define CF_TENSOR_LAUNCH_ADD_CASE(tensor_type, kernel, type) \
case tensor_type: \
  kernel<<<blocks, CF_TENSOR_CUDA_THREADS>>>((type *)d_a, (const type *)d_b, len); \
  break

#define CF_TENSOR_LAUNCH_MUL_CASE(tensor_type, kernel, type) \
case tensor_type: \
  kernel<<<blocks, CF_TENSOR_CUDA_THREADS>>>((type *)d_a, (const type *)d_b, len); \
  break

#define CF_TENSOR_LAUNCH_SCALAR_CASE(tensor_type, kernel, type) \
case tensor_type: \
  kernel<<<blocks, CF_TENSOR_CUDA_THREADS>>>((type *)d_a, *((const type *)scalar), len); \
  break

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

static cf_status cf_tensor_cuda_checked_bytes(cf_usize count, cf_usize elem_size, cf_usize *out_bytes)
{
  if(out_bytes == NULL) return CF_ERR_NULL;
  if(elem_size != 0 && count > SIZE_MAX / elem_size) return CF_ERR_OVERFLOW;

  *out_bytes = count * elem_size;
  return CF_OK;
}

static cf_status cf_tensor_cuda_shape_len(const cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank, cf_usize *out_len)
{
  cf_usize len = 1;

  if(out_len == NULL) return CF_ERR_NULL;
  if(rank > CF_TENSOR_HIGHEST_RANK) return CF_ERR_INVALID;
  if(rank != 0 && dim == NULL) return CF_ERR_NULL;

  for(cf_usize i = 0; i < rank; i++)
  {
    if(dim[i] == 0) return CF_ERR_INVALID;
    if(len > SIZE_MAX / dim[i]) return CF_ERR_OVERFLOW;
    len *= dim[i];
  }

  *out_len = len;
  return CF_OK;
}

static void cf_tensor_cuda_apply_shape(cf_tensor *tensor, const cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank, cf_usize len)
{
  memset(tensor->dim, 0, sizeof(tensor->dim));
  memset(tensor->metadata.stride, 0, sizeof(tensor->metadata.stride));

  tensor->rank = rank;
  tensor->metadata.len = len;

  for(cf_usize i = 0; i < rank; i++) tensor->dim[i] = dim[i];

  cf_usize stride = 1;
  for(cf_usize i = rank; i > 0; i--)
  {
    cf_usize index = i - 1;
    tensor->metadata.stride[index] = stride;
    stride *= tensor->dim[index];
  }
}

static void cf_tensor_cuda_dense_stride(const cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank, cf_usize stride[CF_TENSOR_HIGHEST_RANK])
{
  memset(stride, 0, sizeof(cf_usize) * CF_TENSOR_HIGHEST_RANK);

  cf_usize value = 1;
  for(cf_usize i = rank; i > 0; i--)
  {
    cf_usize index = i - 1;
    stride[index] = value;
    value *= dim[index];
  }
}

static cf_status cf_tensor_cuda_batch_mul_shape(const cf_tensor *op1, const cf_tensor *op2, cf_usize out_dim[CF_TENSOR_HIGHEST_RANK], cf_usize *out_rank, cf_usize *out_len, cf_usize *batch_count)
{
  cf_usize rank;
  cf_usize batch_rank;
  cf_usize batches = 1;
  cf_usize rows;
  cf_usize cols;

  if(op1 == NULL || op2 == NULL || out_dim == NULL || out_rank == NULL || out_len == NULL || batch_count == NULL)
    return CF_ERR_NULL;
  if(op1->rank < 2 || op2->rank < 2) return CF_ERR_INVALID;
  if(op1->metadata.elem_type != op2->metadata.elem_type) return CF_ERR_INVALID;
  if(op1->metadata.elem_size != op2->metadata.elem_size) return CF_ERR_INVALID;
  if(op1->device_data == NULL || op2->device_data == NULL) return CF_ERR_STATE;
  if(op1->dim[op1->rank - 1] != op2->dim[op2->rank - 2]) return CF_ERR_INVALID;

  rank = op1->rank > op2->rank ? op1->rank : op2->rank;
  batch_rank = rank - 2;
  memset(out_dim, 0, sizeof(cf_usize) * CF_TENSOR_HIGHEST_RANK);

  for(cf_usize axis = 0; axis < batch_rank; axis++)
  {
    cf_isize a_axis = (cf_isize)axis - (cf_isize)(rank - op1->rank);
    cf_isize b_axis = (cf_isize)axis - (cf_isize)(rank - op2->rank);
    cf_usize a_dim = a_axis < 0 ? 1 : op1->dim[(cf_usize)a_axis];
    cf_usize b_dim = b_axis < 0 ? 1 : op2->dim[(cf_usize)b_axis];
    cf_usize dim;

    if(a_dim != b_dim && a_dim != 1 && b_dim != 1) return CF_ERR_INVALID;

    dim = a_dim > b_dim ? a_dim : b_dim;
    if(batches > SIZE_MAX / dim) return CF_ERR_OVERFLOW;
    batches *= dim;
    out_dim[axis] = dim;
  }

  rows = op1->dim[op1->rank - 2];
  cols = op2->dim[op2->rank - 1];
  out_dim[batch_rank] = rows;
  out_dim[batch_rank + 1] = cols;

  if(rows != 0 && batches > SIZE_MAX / rows) return CF_ERR_OVERFLOW;
  if(cols != 0 && batches * rows > SIZE_MAX / cols) return CF_ERR_OVERFLOW;

  *out_rank = rank;
  *out_len = batches * rows * cols;
  *batch_count = batches;

  return CF_OK;
}

static cf_status cf_tensor_cuda_index(const cf_tensor *tensor, const cf_usize indexs[CF_TENSOR_HIGHEST_RANK], cf_usize *out_index)
{
  cf_usize index = 0;

  if(tensor == NULL || indexs == NULL || out_index == NULL) return CF_ERR_NULL;
  if(!cf_tensor_is_valid(tensor)) return CF_ERR_INVALID;

  for(cf_usize i = 0; i < tensor->rank; i++)
  {
    if(indexs[i] >= tensor->dim[i]) return CF_ERR_BOUNDS;
    index += tensor->metadata.stride[i] * indexs[i];
  }

  *out_index = index;
  return CF_OK;
}

static cf_status cf_tensor_cublas_status(cublasStatus_t status)
{
  switch(status)
  {
    case CUBLAS_STATUS_SUCCESS: return CF_OK;
    case CUBLAS_STATUS_ALLOC_FAILED: return CF_ERR_CUDA_MEMORY;
    case CUBLAS_STATUS_INVALID_VALUE: return CF_ERR_INVALID;
    case CUBLAS_STATUS_NOT_SUPPORTED:
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return CF_ERR_UNSUPPORTED;
    case CUBLAS_STATUS_EXECUTION_FAILED: return CF_ERR_CUDA_LAUNCH;
    default: return CF_ERR_CUDA;
  }
}

static cf_status cf_tensor_cublaslt_handle(cublasLtHandle_t *out_handle)
{
  static cublasLtHandle_t handle = NULL;

  if(out_handle == NULL) return CF_ERR_NULL;
  if(handle == NULL)
  {
    cublasStatus_t status = cublasLtCreate(&handle);
    if(status != CUBLAS_STATUS_SUCCESS) return cf_tensor_cublas_status(status);
  }

  *out_handle = handle;
  return CF_OK;
}

static void cf_tensor_cublaslt_workspace(void **out_workspace, size_t *out_workspace_bytes)
{
  static void *workspace = NULL;
  static size_t workspace_bytes = 0;

  if(workspace == NULL)
  {
    cudaError_t status = cudaMalloc(&workspace, CF_TENSOR_CUBLASLT_WORKSPACE_BYTES);
    if(status == cudaSuccess)
      workspace_bytes = CF_TENSOR_CUBLASLT_WORKSPACE_BYTES;
    else
      workspace = NULL;
  }

  *out_workspace = workspace;
  *out_workspace_bytes = workspace_bytes;
}

static cf_status cf_tensor_cublaslt_type(cf_tensor_type elem_type, cudaDataType_t *data_type, cudaDataType_t *scale_type)
{
  if(data_type == NULL || scale_type == NULL) return CF_ERR_NULL;

  switch(elem_type)
  {
    case CF_TENSOR_FLOAT:
      *data_type = CUDA_R_32F;
      *scale_type = CUDA_R_32F;
      return CF_OK;
    case CF_TENSOR_DOUBLE:
      *data_type = CUDA_R_64F;
      *scale_type = CUDA_R_64F;
      return CF_OK;
    default:
      return CF_ERR_UNSUPPORTED;
  }
}

static cf_status cf_tensor_cublaslt_set_row_major(cublasLtMatrixLayout_t layout)
{
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));

  return cf_tensor_cublas_status(status);
}

extern "C" cf_status cf_tensor_init_gpu(cf_tensor *tensor, const cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank, cf_tensor_type elem_type)
{
  cf_usize elem_size;
  cf_usize len;
  cf_usize bytes;
  cf_status status;
  cudaError_t cuda_status;

  if(tensor == NULL) return CF_ERR_NULL;

  elem_size = cf_tensor_cuda_type_size(elem_type);
  if(elem_size == 0) return CF_ERR_UNSUPPORTED;

  status = cf_tensor_cuda_shape_len(dim, rank, &len);
  if(status != CF_OK) return status;
  status = cf_tensor_cuda_checked_bytes(len, elem_size, &bytes);
  if(status != CF_OK) return status;

  *tensor = (cf_tensor){0};
  tensor->device = CF_TENSOR_DEVICE_CUDA;
  tensor->metadata.capacity = len;
  tensor->metadata.elem_size = elem_size;
  tensor->metadata.elem_type = elem_type;
  cf_tensor_cuda_apply_shape(tensor, dim, rank, len);

  cuda_status = cudaMalloc(&tensor->device_data, bytes);
  if(cuda_status != cudaSuccess)
  {
    *tensor = (cf_tensor){0};
    return CF_ERR_CUDA_MEMORY;
  }

  cuda_status = cudaMemset(tensor->device_data, 0, bytes);
  if(cuda_status != cudaSuccess)
  {
    cudaFree(tensor->device_data);
    *tensor = (cf_tensor){0};
    return CF_ERR_CUDA_MEMORY;
  }

  return CF_OK;
}

extern "C" cf_status cf_tensor_init_many_gpu(cf_tensor **tensors, cf_usize count, const cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank, cf_tensor_type elem_type)
{
  if(tensors == NULL) return CF_ERR_NULL;

  for(cf_usize i = 0; i < count; i++)
  {
    cf_status status;

    if(tensors[i] == NULL)
    {
      cf_tensor_destroy_many_gpu(tensors, i);
      return CF_ERR_NULL;
    }

    status = cf_tensor_init_gpu(tensors[i], dim, rank, elem_type);
    if(status != CF_OK)
    {
      cf_tensor_destroy_many_gpu(tensors, i);
      return status;
    }
  }

  return CF_OK;
}

extern "C" void cf_tensor_destroy_gpu(cf_tensor *tensor)
{
  if(tensor == NULL) return;

  if(tensor->device_data != NULL) cudaFree(tensor->device_data);
  if(tensor->data != NULL) free(tensor->data);
  *tensor = (cf_tensor){0};
}

extern "C" void cf_tensor_destroy_many_gpu(cf_tensor **tensors, cf_usize count)
{
  if(tensors == NULL) return;
  for(cf_usize i = 0; i < count; i++) cf_tensor_destroy_gpu(tensors[i]);
}

extern "C" cf_status cf_tensor_reserve_gpu(cf_tensor *tensor, cf_usize capacity)
{
  cf_usize new_bytes;
  cf_usize active_bytes;
  cf_status status;
  void *device_data = NULL;
  cudaError_t cuda_status;

  if(tensor == NULL) return CF_ERR_NULL;
  if(tensor->metadata.elem_size == 0) return CF_ERR_INVALID;
  if(cf_tensor_cuda_type_size(tensor->metadata.elem_type) == 0) return CF_ERR_UNSUPPORTED;
  if(capacity <= tensor->metadata.capacity) return CF_OK;

  status = cf_tensor_cuda_checked_bytes(capacity, tensor->metadata.elem_size, &new_bytes);
  if(status != CF_OK) return status;
  status = cf_tensor_cuda_checked_bytes(tensor->metadata.len, tensor->metadata.elem_size, &active_bytes);
  if(status != CF_OK) return status;
  if(tensor->device_data == NULL && tensor->data == NULL) active_bytes = 0;

  cuda_status = cudaMalloc(&device_data, new_bytes);
  if(cuda_status != cudaSuccess) return CF_ERR_CUDA_MEMORY;

  if(tensor->device_data != NULL && active_bytes != 0)
  {
    cuda_status = cudaMemcpy(device_data, tensor->device_data, active_bytes, cudaMemcpyDeviceToDevice);
  }
  else if(tensor->data != NULL && active_bytes != 0)
  {
    cuda_status = cudaMemcpy(device_data, tensor->data, active_bytes, cudaMemcpyHostToDevice);
  }
  else
  {
    cuda_status = cudaSuccess;
  }

  if(cuda_status != cudaSuccess)
  {
    cudaFree(device_data);
    return CF_ERR_CUDA_COPY;
  }

  if(new_bytes > active_bytes)
  {
    cuda_status = cudaMemset((char *)device_data + active_bytes, 0, new_bytes - active_bytes);
    if(cuda_status != cudaSuccess)
    {
      cudaFree(device_data);
      return CF_ERR_CUDA_MEMORY;
    }
  }

  if(tensor->device_data != NULL) cudaFree(tensor->device_data);
  if(tensor->data != NULL)
  {
    free(tensor->data);
    tensor->data = NULL;
  }

  tensor->device_data = device_data;
  tensor->metadata.capacity = capacity;
  tensor->device = CF_TENSOR_DEVICE_CUDA;
  return CF_OK;
}

extern "C" cf_status cf_tensor_reshape_gpu(cf_tensor *tensor, const cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank)
{
  cf_usize len;
  cf_status status;

  if(tensor == NULL) return CF_ERR_NULL;
  if(tensor->device_data == NULL) return CF_ERR_STATE;

  status = cf_tensor_cuda_shape_len(dim, rank, &len);
  if(status != CF_OK) return status;
  if(len > tensor->metadata.capacity) return CF_ERR_BOUNDS;

  cf_tensor_cuda_apply_shape(tensor, dim, rank, len);
  tensor->device = CF_TENSOR_DEVICE_CUDA;
  return CF_OK;
}

extern "C" cf_status cf_tensor_resize_gpu(cf_tensor *tensor, const cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank)
{
  cf_usize len;
  cf_usize old_len;
  cf_usize old_bytes;
  cf_usize new_bytes;
  cf_status status;
  cudaError_t cuda_status;

  if(tensor == NULL) return CF_ERR_NULL;

  status = cf_tensor_cuda_shape_len(dim, rank, &len);
  if(status != CF_OK) return status;

  old_len = tensor->metadata.len;
  if(len > tensor->metadata.capacity)
  {
    status = cf_tensor_reserve_gpu(tensor, len);
    if(status != CF_OK) return status;
  }
  else if(len > old_len)
  {
    status = cf_tensor_cuda_checked_bytes(old_len, tensor->metadata.elem_size, &old_bytes);
    if(status != CF_OK) return status;
    status = cf_tensor_cuda_checked_bytes(len, tensor->metadata.elem_size, &new_bytes);
    if(status != CF_OK) return status;
    cuda_status = cudaMemset((char *)tensor->device_data + old_bytes, 0, new_bytes - old_bytes);
    if(cuda_status != cudaSuccess) return CF_ERR_CUDA_MEMORY;
  }

  if(tensor->data != NULL)
  {
    free(tensor->data);
    tensor->data = NULL;
  }

  cf_tensor_cuda_apply_shape(tensor, dim, rank, len);
  tensor->device = CF_TENSOR_DEVICE_CUDA;
  return CF_OK;
}

extern "C" cf_status cf_tensor_copy_gpu(cf_tensor *dst, const cf_tensor *src)
{
  cf_usize bytes;
  cf_status status;
  cudaError_t cuda_status;

  if(dst == NULL || src == NULL) return CF_ERR_NULL;
  if(src->device_data == NULL) return CF_ERR_STATE;
  if(dst == src) return CF_OK;

  if(dst->metadata.elem_size == 0 && dst->data == NULL && dst->device_data == NULL)
  {
    status = cf_tensor_init_gpu(dst, src->dim, src->rank, src->metadata.elem_type);
    if(status != CF_OK) return status;
  }
  else
  {
    if(dst->metadata.elem_type != src->metadata.elem_type) return CF_ERR_INVALID;
    if(dst->metadata.elem_size != src->metadata.elem_size) return CF_ERR_INVALID;
    status = cf_tensor_resize_gpu(dst, src->dim, src->rank);
    if(status != CF_OK) return status;
  }

  status = cf_tensor_cuda_checked_bytes(src->metadata.len, src->metadata.elem_size, &bytes);
  if(status != CF_OK) return status;

  cuda_status = cudaMemcpy(dst->device_data, src->device_data, bytes, cudaMemcpyDeviceToDevice);
  if(cuda_status != cudaSuccess) return CF_ERR_CUDA_COPY;

  if(dst->data != NULL)
  {
    free(dst->data);
    dst->data = NULL;
  }

  dst->device = CF_TENSOR_DEVICE_CUDA;
  return CF_OK;
}

extern "C" cf_status cf_tensor_copy_from_array_gpu(cf_tensor *tensor, const void *array, cf_usize count)
{
  cf_usize dim[CF_TENSOR_HIGHEST_RANK] = {0};
  cf_usize bytes;
  cf_status status;
  cudaError_t cuda_status;

  if(tensor == NULL || array == NULL) return CF_ERR_NULL;
  if(count == 0) return CF_ERR_INVALID;
  if(tensor->metadata.elem_size == 0) return CF_ERR_INVALID;

  if(count > tensor->metadata.capacity)
  {
    status = cf_tensor_reserve_gpu(tensor, count);
    if(status != CF_OK) return status;
  }

  dim[0] = count;
  cf_tensor_cuda_apply_shape(tensor, dim, 1, count);

  status = cf_tensor_cuda_checked_bytes(count, tensor->metadata.elem_size, &bytes);
  if(status != CF_OK) return status;

  cuda_status = cudaMemcpy(tensor->device_data, array, bytes, cudaMemcpyHostToDevice);
  if(cuda_status != cudaSuccess) return CF_ERR_CUDA_COPY;

  if(tensor->data != NULL)
  {
    free(tensor->data);
    tensor->data = NULL;
  }

  tensor->device = CF_TENSOR_DEVICE_CUDA;
  return CF_OK;
}

extern "C" cf_status cf_tensor_copy_to_array_gpu(void *array, const cf_tensor *tensor, cf_usize count)
{
  cf_usize bytes;
  cf_status status;
  cudaError_t cuda_status;

  if(array == NULL || tensor == NULL) return CF_ERR_NULL;
  if(tensor->device_data == NULL) return CF_ERR_STATE;
  if(count < tensor->metadata.len) return CF_ERR_BOUNDS;

  status = cf_tensor_cuda_checked_bytes(tensor->metadata.len, tensor->metadata.elem_size, &bytes);
  if(status != CF_OK) return status;

  cuda_status = cudaMemcpy(array, tensor->device_data, bytes, cudaMemcpyDeviceToHost);
  return cuda_status == cudaSuccess ? CF_OK : CF_ERR_CUDA_COPY;
}

extern "C" cf_status cf_tensor_get_gpu(void *out_value, const cf_tensor *tensor, const cf_usize indexs[CF_TENSOR_HIGHEST_RANK])
{
  cf_usize index;
  cf_status status;
  cudaError_t cuda_status;

  if(out_value == NULL) return CF_ERR_NULL;
  if(tensor == NULL || tensor->device_data == NULL) return CF_ERR_NULL;

  status = cf_tensor_cuda_index(tensor, indexs, &index);
  if(status != CF_OK) return status;

  cuda_status = cudaMemcpy(out_value, (const char *)tensor->device_data + index * tensor->metadata.elem_size, tensor->metadata.elem_size, cudaMemcpyDeviceToHost);

  return cuda_status == cudaSuccess ? CF_OK : CF_ERR_CUDA_COPY;
}

extern "C" cf_status cf_tensor_set_gpu(cf_tensor *tensor, const cf_usize indexs[CF_TENSOR_HIGHEST_RANK], const void *value)
{
  cf_usize index;
  cf_status status;
  cudaError_t cuda_status;

  if(value == NULL) return CF_ERR_NULL;
  if(tensor == NULL || tensor->device_data == NULL) return CF_ERR_NULL;

  status = cf_tensor_cuda_index(tensor, indexs, &index);
  if(status != CF_OK) return status;

  cuda_status = cudaMemcpy((char *)tensor->device_data + index * tensor->metadata.elem_size, value, tensor->metadata.elem_size, cudaMemcpyHostToDevice);
  if(cuda_status != cudaSuccess) return CF_ERR_CUDA_COPY;

  if(tensor->data != NULL)
  {
    free(tensor->data);
    tensor->data = NULL;
  }

  tensor->device = CF_TENSOR_DEVICE_CUDA;
  return CF_OK;
}

extern "C" cf_status cf_tensor_to_gpu(cf_tensor *tensor)
{
  cf_usize capacity_bytes;
  cf_usize active_bytes;
  cf_status status;
  cudaError_t cuda_status;

  if(tensor == NULL) return CF_ERR_NULL;
  if(tensor->device_data != NULL && tensor->data == NULL)
  {
    tensor->device = CF_TENSOR_DEVICE_CUDA;
    return CF_OK;
  }
  if(tensor->data == NULL) return CF_ERR_STATE;
  if(cf_tensor_cuda_type_size(tensor->metadata.elem_type) == 0) return CF_ERR_UNSUPPORTED;

  status = cf_tensor_cuda_checked_bytes(tensor->metadata.capacity, tensor->metadata.elem_size, &capacity_bytes);
  if(status != CF_OK) return status;
  status = cf_tensor_cuda_checked_bytes(tensor->metadata.len, tensor->metadata.elem_size, &active_bytes);
  if(status != CF_OK) return status;

  if(tensor->device_data == NULL)
  {
    cuda_status = cudaMalloc(&tensor->device_data, capacity_bytes);
    if(cuda_status != cudaSuccess) return CF_ERR_CUDA_MEMORY;
  }

  cuda_status = cudaMemcpy(tensor->device_data, tensor->data, active_bytes, cudaMemcpyHostToDevice);
  if(cuda_status != cudaSuccess) return CF_ERR_CUDA_COPY;

  tensor->device = CF_TENSOR_DEVICE_CUDA;
  return CF_OK;
}

extern "C" cf_status cf_tensor_to_cpu(cf_tensor *tensor)
{
  cf_usize capacity_bytes;
  cf_usize active_bytes;
  cf_status status;
  cudaError_t cuda_status;

  if(tensor == NULL) return CF_ERR_NULL;
  if(tensor->device_data == NULL) return CF_ERR_STATE;

  status = cf_tensor_cuda_checked_bytes(tensor->metadata.capacity, tensor->metadata.elem_size, &capacity_bytes);
  if(status != CF_OK) return status;
  status = cf_tensor_cuda_checked_bytes(tensor->metadata.len, tensor->metadata.elem_size, &active_bytes);
  if(status != CF_OK) return status;

  if(tensor->data == NULL)
  {
    tensor->data = malloc(capacity_bytes);
    if(tensor->data == NULL) return CF_ERR_OOM;
  }

  cuda_status = cudaMemcpy(tensor->data, tensor->device_data, active_bytes, cudaMemcpyDeviceToHost);
  if(cuda_status != cudaSuccess) return CF_ERR_CUDA_COPY;

  tensor->device = CF_TENSOR_DEVICE_CPU;
  return CF_OK;
}

extern "C" cf_status cf_tensor_free_gpu(cf_tensor *tensor)
{
  cudaError_t cuda_status;

  if(tensor == NULL) return CF_ERR_NULL;
  if(tensor->device_data == NULL) return CF_OK;

  cuda_status = cudaFree(tensor->device_data);
  tensor->device_data = NULL;

  if(tensor->data == NULL)
    *tensor = (cf_tensor){0};
  else if(tensor->device == CF_TENSOR_DEVICE_CUDA)
    tensor->device = CF_TENSOR_DEVICE_CPU;

  return cuda_status == cudaSuccess ? CF_OK : CF_ERR_CUDA_MEMORY;
}

extern "C" cf_status cf_tensor_add_gpu(cf_tensor *op1, const cf_tensor *op2)
{
  cf_usize len = op1->metadata.len;
  int blocks = (int)((len + CF_TENSOR_CUDA_THREADS - 1) / CF_TENSOR_CUDA_THREADS);
  void *d_a = op1->device_data;
  const void *d_b = op2->device_data;
  cudaError_t cuda_status;

  switch(op1->metadata.elem_type)
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
    default: return CF_ERR_UNSUPPORTED;
  }

  cuda_status = cudaGetLastError();
  if(cuda_status != cudaSuccess) return CF_ERR_CUDA_LAUNCH;

  if(op1->data != NULL)
  {
    free(op1->data);
    op1->data = NULL;
  }

  op1->device = CF_TENSOR_DEVICE_CUDA;
  return CF_OK;
}

extern "C" cf_status cf_tensor_mul_gpu(cf_tensor *op1, const cf_tensor *op2)
{
  cf_usize len = op1->metadata.len;
  int blocks = (int)((len + CF_TENSOR_CUDA_THREADS - 1) / CF_TENSOR_CUDA_THREADS);
  void *d_a = op1->device_data;
  const void *d_b = op2->device_data;
  cudaError_t cuda_status;

  switch(op1->metadata.elem_type)
  {
    CF_TENSOR_LAUNCH_MUL_CASE(CF_TENSOR_CHAR, cf_tensor_mul_char_kernel, char);
    CF_TENSOR_LAUNCH_MUL_CASE(CF_TENSOR_SHORT, cf_tensor_mul_short_kernel, short);
    CF_TENSOR_LAUNCH_MUL_CASE(CF_TENSOR_INT, cf_tensor_mul_int_kernel, int);
    CF_TENSOR_LAUNCH_MUL_CASE(CF_TENSOR_LONG, cf_tensor_mul_long_kernel, long);
    CF_TENSOR_LAUNCH_MUL_CASE(CF_TENSOR_LL, cf_tensor_mul_ll_kernel, long long);
    CF_TENSOR_LAUNCH_MUL_CASE(CF_TENSOR_FLOAT, cf_tensor_mul_float_kernel, float);
    CF_TENSOR_LAUNCH_MUL_CASE(CF_TENSOR_DOUBLE, cf_tensor_mul_double_kernel, double);
    CF_TENSOR_LAUNCH_MUL_CASE(CF_TENSOR_U8, cf_tensor_mul_u8_kernel, cf_u8);
    CF_TENSOR_LAUNCH_MUL_CASE(CF_TENSOR_U16, cf_tensor_mul_u16_kernel, cf_u16);
    CF_TENSOR_LAUNCH_MUL_CASE(CF_TENSOR_U32, cf_tensor_mul_u32_kernel, cf_u32);
    CF_TENSOR_LAUNCH_MUL_CASE(CF_TENSOR_U64, cf_tensor_mul_u64_kernel, cf_u64);
    default: return CF_ERR_UNSUPPORTED;
  }

  cuda_status = cudaGetLastError();
  if(cuda_status != cudaSuccess) return CF_ERR_CUDA_LAUNCH;

  if(op1->data != NULL)
  {
    free(op1->data);
    op1->data = NULL;
  }

  op1->device = CF_TENSOR_DEVICE_CUDA;
  return CF_OK;
}

extern "C" cf_status cf_tensor_scalar_mul_gpu(cf_tensor *op1, const void *scalar)
{
  cf_usize len;
  int blocks;
  void *d_a;
  cudaError_t cuda_status;

  if(scalar == NULL) return CF_ERR_NULL;

  len = op1->metadata.len;
  blocks = (int)((len + CF_TENSOR_CUDA_THREADS - 1) / CF_TENSOR_CUDA_THREADS);
  d_a = op1->device_data;

  switch(op1->metadata.elem_type)
  {
    CF_TENSOR_LAUNCH_SCALAR_CASE(CF_TENSOR_CHAR, cf_tensor_scalar_char_kernel, char);
    CF_TENSOR_LAUNCH_SCALAR_CASE(CF_TENSOR_SHORT, cf_tensor_scalar_short_kernel, short);
    CF_TENSOR_LAUNCH_SCALAR_CASE(CF_TENSOR_INT, cf_tensor_scalar_int_kernel, int);
    CF_TENSOR_LAUNCH_SCALAR_CASE(CF_TENSOR_LONG, cf_tensor_scalar_long_kernel, long);
    CF_TENSOR_LAUNCH_SCALAR_CASE(CF_TENSOR_LL, cf_tensor_scalar_ll_kernel, long long);
    CF_TENSOR_LAUNCH_SCALAR_CASE(CF_TENSOR_FLOAT, cf_tensor_scalar_float_kernel, float);
    CF_TENSOR_LAUNCH_SCALAR_CASE(CF_TENSOR_DOUBLE, cf_tensor_scalar_double_kernel, double);
    CF_TENSOR_LAUNCH_SCALAR_CASE(CF_TENSOR_U8, cf_tensor_scalar_u8_kernel, cf_u8);
    CF_TENSOR_LAUNCH_SCALAR_CASE(CF_TENSOR_U16, cf_tensor_scalar_u16_kernel, cf_u16);
    CF_TENSOR_LAUNCH_SCALAR_CASE(CF_TENSOR_U32, cf_tensor_scalar_u32_kernel, cf_u32);
    CF_TENSOR_LAUNCH_SCALAR_CASE(CF_TENSOR_U64, cf_tensor_scalar_u64_kernel, cf_u64);
    default: return CF_ERR_UNSUPPORTED;
  }

  cuda_status = cudaGetLastError();
  if(cuda_status != cudaSuccess) return CF_ERR_CUDA_LAUNCH;

  if(op1->data != NULL)
  {
    free(op1->data);
    op1->data = NULL;
  }

  op1->device = CF_TENSOR_DEVICE_CUDA;
  return CF_OK;
}

static cf_status cf_tensor_matrix_mul_cublaslt_2d(const void *a_device, const void *b_device, void *out, cf_tensor_type elem_type, cf_usize m, cf_usize k, cf_usize n, cublasComputeType_t compute_type)
{
  cudaDataType_t data_type;
  cudaDataType_t scale_type;
  cublasLtHandle_t handle;
  cublasLtMatmulDesc_t operation = NULL;
  cublasLtMatrixLayout_t a_layout = NULL;
  cublasLtMatrixLayout_t b_layout = NULL;
  cublasLtMatrixLayout_t c_layout = NULL;
  cublasLtMatmulPreference_t preference = NULL;
  cublasLtMatmulHeuristicResult_t heuristic = {0};
  int returned_results = 0;
  void *workspace = NULL;
  size_t workspace_bytes = 0;
  uint64_t max_workspace_bytes = 0;
  cf_status status;
  cublasStatus_t cublas_status;

  status = cf_tensor_cublaslt_type(elem_type, &data_type, &scale_type);
  if(status != CF_OK) return status;

  status = cf_tensor_cublaslt_handle(&handle);
  if(status != CF_OK) return status;

  cublas_status = cublasLtMatmulDescCreate(&operation, compute_type, scale_type);
  if(cublas_status != CUBLAS_STATUS_SUCCESS)
  {
    status = cf_tensor_cublas_status(cublas_status);
    goto cleanup;
  }

  cublas_status = cublasLtMatrixLayoutCreate(&a_layout, data_type, (uint64_t)m, (uint64_t)k, (int64_t)k);
  if(cublas_status != CUBLAS_STATUS_SUCCESS)
  {
    status = cf_tensor_cublas_status(cublas_status);
    goto cleanup;
  }

  cublas_status = cublasLtMatrixLayoutCreate(&b_layout, data_type, (uint64_t)k, (uint64_t)n, (int64_t)n);
  if(cublas_status != CUBLAS_STATUS_SUCCESS)
  {
    status = cf_tensor_cublas_status(cublas_status);
    goto cleanup;
  }

  cublas_status = cublasLtMatrixLayoutCreate(&c_layout, data_type, (uint64_t)m, (uint64_t)n, (int64_t)n);
  if(cublas_status != CUBLAS_STATUS_SUCCESS)
  {
    status = cf_tensor_cublas_status(cublas_status);
    goto cleanup;
  }

  status = cf_tensor_cublaslt_set_row_major(a_layout);
  if(status != CF_OK) goto cleanup;
  status = cf_tensor_cublaslt_set_row_major(b_layout);
  if(status != CF_OK) goto cleanup;
  status = cf_tensor_cublaslt_set_row_major(c_layout);
  if(status != CF_OK) goto cleanup;

  cublas_status = cublasLtMatmulPreferenceCreate(&preference);
  if(cublas_status != CUBLAS_STATUS_SUCCESS)
  {
    status = cf_tensor_cublas_status(cublas_status);
    goto cleanup;
  }

  cf_tensor_cublaslt_workspace(&workspace, &workspace_bytes);
  max_workspace_bytes = (uint64_t)workspace_bytes;
  cublas_status = cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_bytes, sizeof(max_workspace_bytes));
  if(cublas_status != CUBLAS_STATUS_SUCCESS)
  {
    status = cf_tensor_cublas_status(cublas_status);
    goto cleanup;
  }

  cublas_status = cublasLtMatmulAlgoGetHeuristic(handle, operation, a_layout, b_layout, c_layout, c_layout, preference, 1, &heuristic, &returned_results);
  if(cublas_status != CUBLAS_STATUS_SUCCESS || returned_results == 0)
  {
    status = cublas_status == CUBLAS_STATUS_SUCCESS ? CF_ERR_UNSUPPORTED : cf_tensor_cublas_status(cublas_status);
    goto cleanup;
  }

  if(elem_type == CF_TENSOR_FLOAT)
  {
    float alpha = 1.0f;
    float beta = 0.0f;
    cublas_status = cublasLtMatmul(handle, operation, &alpha, a_device, a_layout, b_device, b_layout, &beta, out, c_layout, out, c_layout, &heuristic.algo, workspace, heuristic.workspaceSize, 0);
  }
  else
  {
    double alpha = 1.0;
    double beta = 0.0;
    cublas_status = cublasLtMatmul(handle, operation, &alpha, a_device, a_layout, b_device, b_layout, &beta, out, c_layout, out, c_layout, &heuristic.algo, workspace, heuristic.workspaceSize, 0);
  }

  status = cf_tensor_cublas_status(cublas_status);

cleanup:
  if(preference != NULL) cublasLtMatmulPreferenceDestroy(preference);
  if(c_layout != NULL) cublasLtMatrixLayoutDestroy(c_layout);
  if(b_layout != NULL) cublasLtMatrixLayoutDestroy(b_layout);
  if(a_layout != NULL) cublasLtMatrixLayoutDestroy(a_layout);
  if(operation != NULL) cublasLtMatmulDescDestroy(operation);
  return status;
}

extern "C" cf_status cf_tensor_batch_mul_gpu(cf_tensor *op1, const cf_tensor *op2)
{
  cf_usize rows;
  cf_usize inner;
  cf_usize cols;
  cf_usize out_len;
  cf_usize out_bytes;
  cf_usize out_dim[CF_TENSOR_HIGHEST_RANK] = {0};
  cf_usize out_stride[CF_TENSOR_HIGHEST_RANK] = {0};
  cf_usize out_rank;
  cf_usize batch_rank;
  cf_usize batch_count;
  cf_status status;
  void *result = NULL;
  cudaError_t cuda_status;

  status = cf_tensor_cuda_batch_mul_shape(op1, op2, out_dim, &out_rank, &out_len, &batch_count);
  if(status != CF_OK) return status;

  batch_rank = out_rank - 2;
  rows = op1->dim[op1->rank - 2];
  inner = op1->dim[op1->rank - 1];
  cols = op2->dim[op2->rank - 1];
  cf_tensor_cuda_dense_stride(out_dim, out_rank, out_stride);

  status = cf_tensor_cuda_checked_bytes(out_len, op1->metadata.elem_size, &out_bytes);
  if(status != CF_OK) return status;

  cuda_status = cudaMalloc(&result, out_bytes);
  if(cuda_status != cudaSuccess) return CF_ERR_CUDA_MEMORY;

  for(cf_usize batch = 0; batch < batch_count; batch++)
  {
    cf_usize a_base = 0;
    cf_usize b_base = 0;
    cf_usize out_base = 0;
    cf_usize rest = batch;
    const char *a_device;
    const char *b_device;
    char *out_device;

    for(cf_usize batch_axis_i = batch_rank; batch_axis_i > 0; batch_axis_i--)
    {
      cf_usize axis = batch_axis_i - 1;
      cf_usize coord = rest % out_dim[axis];
      cf_isize a_axis = (cf_isize)axis - (cf_isize)(out_rank - op1->rank);
      cf_isize b_axis = (cf_isize)axis - (cf_isize)(out_rank - op2->rank);

      rest /= out_dim[axis];
      out_base += coord * out_stride[axis];
      if(a_axis >= 0 && op1->dim[(cf_usize)a_axis] != 1)
        a_base += coord * op1->metadata.stride[(cf_usize)a_axis];
      if(b_axis >= 0 && op2->dim[(cf_usize)b_axis] != 1)
        b_base += coord * op2->metadata.stride[(cf_usize)b_axis];
    }

    a_device = (const char *)op1->device_data + a_base * op1->metadata.elem_size;
    b_device = (const char *)op2->device_data + b_base * op2->metadata.elem_size;
    out_device = (char *)result + out_base * op1->metadata.elem_size;

    switch(op1->metadata.elem_type)
    {
      case CF_TENSOR_FLOAT:
        status = cf_tensor_matrix_mul_cublaslt_2d(a_device, b_device, out_device, op1->metadata.elem_type, rows, inner, cols, CUBLAS_COMPUTE_32F_FAST_TF32);
        if(status == CF_ERR_UNSUPPORTED)
          status = cf_tensor_matrix_mul_cublaslt_2d(a_device, b_device, out_device, op1->metadata.elem_type, rows, inner, cols, CUBLAS_COMPUTE_32F);
        break;
      case CF_TENSOR_DOUBLE:
        status = cf_tensor_matrix_mul_cublaslt_2d(a_device, b_device, out_device, op1->metadata.elem_type, rows, inner, cols, CUBLAS_COMPUTE_64F);
        break;
      default:
        cudaFree(result);
        return CF_ERR_UNSUPPORTED;
    }

    if(status != CF_OK)
    {
      cudaFree(result);
      return status;
    }
  }

  if(op1->device_data != NULL && out_len <= op1->metadata.capacity)
  {
    cuda_status = cudaMemcpy(op1->device_data, result, out_bytes, cudaMemcpyDeviceToDevice);
    cudaFree(result);
    if(cuda_status != cudaSuccess) return CF_ERR_CUDA_COPY;
  }
  else
  {
    if(op1->device_data != NULL) cudaFree(op1->device_data);
    op1->device_data = result;
    op1->metadata.capacity = out_len;
  }

  if(op1->data != NULL)
  {
    free(op1->data);
    op1->data = NULL;
  }

  cf_tensor_cuda_apply_shape(op1, out_dim, out_rank, out_len);
  op1->device = CF_TENSOR_DEVICE_CUDA;

  return CF_OK;
}

extern "C" cf_status cf_tensor_matrix_mul_gpu(cf_tensor *op1, const cf_tensor *op2)
{
  return cf_tensor_batch_mul_gpu(op1, op2);
}
