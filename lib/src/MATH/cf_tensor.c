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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CF_TENSOR_TYPE_SIZE_CASE(tensor_type, type) \
case tensor_type: \
  return sizeof(type)

#define CF_TENSOR_GET_CASE(tensor_type, type) \
case tensor_type: \
  *((type *)out_value) = ((const type *)tensor->data)[index]; \
  break

#define CF_TENSOR_SET_CASE(tensor_type, type) \
case tensor_type: \
  ((type *)tensor->data)[index] = *((const type *)value); \
  break

#define CF_TENSOR_ADD_CASE(tensor_type, type) \
case tensor_type: \
{ \
  type *a = (type *)op1->data; \
  const type *b = (const type *)op2->data; \
  for(cf_usize i = 0; i < len; i++) a[i] = (type)(a[i] + b[i]); \
  break; \
}

#define CF_TENSOR_MUL_CASE(tensor_type, type) \
case tensor_type: \
{ \
  type *a = (type *)op1->data; \
  const type *b = (const type *)op2->data; \
  for(cf_usize i = 0; i < len; i++) a[i] = (type)(a[i] * b[i]); \
  break; \
}

#define CF_TENSOR_SCALAR_CASE(tensor_type, type) \
case tensor_type: \
{ \
  type *a = (type *)op1->data; \
  type value = *((const type *)scalar); \
  for(cf_usize i = 0; i < len; i++) a[i] = (type)(a[i] * value); \
  break; \
}

#define CF_TENSOR_BATCH_MUL_CASE(tensor_type, type) \
case tensor_type: \
{ \
  const type *a = (const type *)op1->data; \
  const type *b = (const type *)op2->data; \
  type *out = (type *)result; \
  for(cf_usize batch = 0; batch < batch_count; batch++) \
  { \
    cf_usize a_base = 0; \
    cf_usize b_base = 0; \
    cf_usize out_base = 0; \
    cf_usize rest = batch; \
    for(cf_usize batch_axis_i = batch_rank; batch_axis_i > 0; batch_axis_i--) \
    { \
      cf_usize axis = batch_axis_i - 1; \
      cf_usize coord = rest % out_dim[axis]; \
      cf_isize a_axis = (cf_isize)axis - (cf_isize)(out_rank - op1->rank); \
      cf_isize b_axis = (cf_isize)axis - (cf_isize)(out_rank - op2->rank); \
      rest /= out_dim[axis]; \
      out_base += coord * out_stride[axis]; \
      if(a_axis >= 0 && op1->dim[(cf_usize)a_axis] != 1) \
        a_base += coord * op1->metadata.stride[(cf_usize)a_axis]; \
      if(b_axis >= 0 && op2->dim[(cf_usize)b_axis] != 1) \
        b_base += coord * op2->metadata.stride[(cf_usize)b_axis]; \
    } \
    for(cf_usize row = 0; row < rows; row++) \
    { \
      for(cf_usize col = 0; col < cols; col++) \
      { \
        type sum = (type)0; \
        for(cf_usize inner_index = 0; inner_index < inner; inner_index++) \
        { \
          sum = (type)(sum + \
            a[a_base + row * op1->metadata.stride[op1->rank - 2] + inner_index * op1->metadata.stride[op1->rank - 1]] * \
            b[b_base + inner_index * op2->metadata.stride[op2->rank - 2] + col * op2->metadata.stride[op2->rank - 1]]); \
        } \
        out[out_base + row * out_stride[batch_rank] + col * out_stride[batch_rank + 1]] = sum; \
      } \
    } \
  } \
  break; \
}

cf_usize cf_tensor_element_size(cf_tensor_type elem_type)
{
  switch(elem_type)
  {
    CF_TENSOR_TYPE_SIZE_CASE(CF_TENSOR_CHAR, char);
    CF_TENSOR_TYPE_SIZE_CASE(CF_TENSOR_SHORT, short);
    CF_TENSOR_TYPE_SIZE_CASE(CF_TENSOR_INT, int);
    CF_TENSOR_TYPE_SIZE_CASE(CF_TENSOR_LONG, long);
    CF_TENSOR_TYPE_SIZE_CASE(CF_TENSOR_LL, long long);
    CF_TENSOR_TYPE_SIZE_CASE(CF_TENSOR_FLOAT, float);
    CF_TENSOR_TYPE_SIZE_CASE(CF_TENSOR_DOUBLE, double);
    CF_TENSOR_TYPE_SIZE_CASE(CF_TENSOR_LD, long double);
    CF_TENSOR_TYPE_SIZE_CASE(CF_TENSOR_U8, cf_u8);
    CF_TENSOR_TYPE_SIZE_CASE(CF_TENSOR_U16, cf_u16);
    CF_TENSOR_TYPE_SIZE_CASE(CF_TENSOR_U32, cf_u32);
    CF_TENSOR_TYPE_SIZE_CASE(CF_TENSOR_U64, cf_u64);
    CF_TENSOR_TYPE_SIZE_CASE(CF_TENSOR_U128, cf_u128);
    default: return 0;
  }
}

static cf_status cf_tensor_checked_bytes(cf_usize count, cf_usize elem_size, cf_usize *out_bytes)
{
  if(out_bytes == CF_NULL) return CF_ERR_NULL;
  if(elem_size != 0 && count > SIZE_MAX / elem_size) return CF_ERR_OVERFLOW;

  *out_bytes = count * elem_size;
  return CF_OK;
}

static cf_status cf_tensor_shape_len(const cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank, cf_usize *out_len)
{
  cf_usize len = 1;

  if(out_len == CF_NULL) return CF_ERR_NULL;
  if(rank > CF_TENSOR_HIGHEST_RANK) return CF_ERR_INVALID;
  if(rank != 0 && dim == CF_NULL) return CF_ERR_NULL;

  for(cf_usize i = 0; i < rank; i++)
  {
    if(dim[i] == 0) return CF_ERR_INVALID;
    if(len > SIZE_MAX / dim[i]) return CF_ERR_OVERFLOW;
    len *= dim[i];
  }

  *out_len = len;
  return CF_OK;
}

static void cf_tensor_apply_shape(cf_tensor *tensor, const cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank, cf_usize len)
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

static void cf_tensor_dense_stride(const cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank, cf_usize stride[CF_TENSOR_HIGHEST_RANK])
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

static cf_status cf_tensor_batch_mul_shape(const cf_tensor *op1, const cf_tensor *op2, cf_usize out_dim[CF_TENSOR_HIGHEST_RANK], cf_usize *out_rank, cf_usize *out_len, cf_usize *batch_count)
{
  cf_usize rank;
  cf_usize batch_rank;
  cf_usize batches = 1;
  cf_usize rows;
  cf_usize cols;

  if(op1 == CF_NULL || op2 == CF_NULL || out_dim == CF_NULL || out_rank == CF_NULL || out_len == CF_NULL || batch_count == CF_NULL)
    return CF_ERR_NULL;
  if(op1->rank < 2 || op2->rank < 2) return CF_ERR_INVALID;
  if(op1->metadata.elem_type != op2->metadata.elem_type) return CF_ERR_INVALID;
  if(op1->metadata.elem_size != op2->metadata.elem_size) return CF_ERR_INVALID;
  if(op1->data == CF_NULL || op2->data == CF_NULL) return CF_ERR_STATE;
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

static cf_status cf_tensor_flat_index(const cf_tensor *tensor, const cf_usize indexs[CF_TENSOR_HIGHEST_RANK], cf_usize *out_index)
{
  cf_usize index = 0;

  if(tensor == CF_NULL || indexs == CF_NULL || out_index == CF_NULL) return CF_ERR_NULL;
  if(!cf_tensor_is_valid(tensor)) return CF_ERR_INVALID;

  for(cf_usize i = 0; i < tensor->rank; i++)
  {
    if(indexs[i] >= tensor->dim[i]) return CF_ERR_BOUNDS;
    index += tensor->metadata.stride[i] * indexs[i];
  }

  *out_index = index;
  return CF_OK;
}

cf_bool cf_tensor_is_valid(const cf_tensor *tensor)
{
  cf_usize elem_size;
  cf_usize len;

  if(tensor == CF_NULL) return CF_FALSE;
  if(tensor->rank > CF_TENSOR_HIGHEST_RANK) return CF_FALSE;
  if(tensor->device != CF_TENSOR_DEVICE_CPU && tensor->device != CF_TENSOR_DEVICE_CUDA)
    return CF_FALSE;

  elem_size = cf_tensor_element_size(tensor->metadata.elem_type);
  if(elem_size == 0 || elem_size != tensor->metadata.elem_size) return CF_FALSE;
  if(tensor->metadata.len == 0 || tensor->metadata.capacity < tensor->metadata.len)
    return CF_FALSE;

  if(cf_tensor_shape_len(tensor->dim, tensor->rank, &len) != CF_OK) return CF_FALSE;
  if(len != tensor->metadata.len) return CF_FALSE;

  cf_usize stride = 1;
  for(cf_usize i = tensor->rank; i > 0; i--)
  {
    cf_usize index = i - 1;
    if(tensor->metadata.stride[index] != stride) return CF_FALSE;
    if(stride > SIZE_MAX / tensor->dim[index]) return CF_FALSE;
    stride *= tensor->dim[index];
  }

  if(tensor->device == CF_TENSOR_DEVICE_CPU && tensor->data == CF_NULL) return CF_FALSE;
  if(tensor->device == CF_TENSOR_DEVICE_CUDA && tensor->device_data == CF_NULL) return CF_FALSE;
  if(tensor->data == CF_NULL && tensor->device_data == CF_NULL) return CF_FALSE;

  return CF_TRUE;
}

cf_status cf_tensor_init_cpu(cf_tensor *tensor, const cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank, cf_tensor_type elem_type)
{
  cf_usize elem_size;
  cf_usize len;
  cf_usize bytes;
  cf_status status;
  void *data;

  if(tensor == CF_NULL) return CF_ERR_NULL;

  elem_size = cf_tensor_element_size(elem_type);
  if(elem_size == 0) return CF_ERR_INVALID;

  status = cf_tensor_shape_len(dim, rank, &len);
  if(status != CF_OK) return status;

  status = cf_tensor_checked_bytes(len, elem_size, &bytes);
  if(status != CF_OK) return status;

  data = malloc(bytes);
  if(data == CF_NULL) return CF_ERR_OOM;
  memset(data, 0, bytes);

  *tensor = (cf_tensor){0};
  tensor->data = data;
  tensor->device = CF_TENSOR_DEVICE_CPU;
  tensor->metadata.capacity = len;
  tensor->metadata.elem_size = elem_size;
  tensor->metadata.elem_type = elem_type;
  cf_tensor_apply_shape(tensor, dim, rank, len);

  return CF_OK;
}

cf_status cf_tensor_init_many_cpu(cf_tensor **tensors, cf_usize count, const cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank, cf_tensor_type elem_type)
{
  if(tensors == CF_NULL) return CF_ERR_NULL;

  for(cf_usize i = 0; i < count; i++)
  {
    cf_status status;

    if(tensors[i] == CF_NULL)
    {
      cf_tensor_destroy_many_cpu(tensors, i);
      return CF_ERR_NULL;
    }

    status = cf_tensor_init_cpu(tensors[i], dim, rank, elem_type);
    if(status != CF_OK)
    {
      cf_tensor_destroy_many_cpu(tensors, i);
      return status;
    }
  }

  return CF_OK;
}

void cf_tensor_destroy_cpu(cf_tensor *tensor)
{
  if(tensor == CF_NULL) return;

#ifdef CF_CUDA_AVAILABLE
  if(tensor->device_data != CF_NULL) (void)cf_tensor_free_gpu(tensor);
#endif

  if(tensor->data != CF_NULL) free(tensor->data);
  *tensor = (cf_tensor){0};
}

void cf_tensor_destroy_many_cpu(cf_tensor **tensors, cf_usize count)
{
  if(tensors == CF_NULL) return;
  for(cf_usize i = 0; i < count; i++) cf_tensor_destroy_cpu(tensors[i]);
}

cf_status cf_tensor_reserve_cpu(cf_tensor *tensor, cf_usize capacity)
{
  cf_usize bytes;
  cf_usize old_bytes;
  cf_status status;
  void *data;

  if(tensor == CF_NULL) return CF_ERR_NULL;
  if(tensor->metadata.elem_size == 0) return CF_ERR_INVALID;
  if(capacity <= tensor->metadata.capacity) return CF_OK;

  status = cf_tensor_checked_bytes(capacity, tensor->metadata.elem_size, &bytes);
  if(status != CF_OK) return status;
  status = cf_tensor_checked_bytes(tensor->metadata.capacity, tensor->metadata.elem_size, &old_bytes);
  if(status != CF_OK) return status;

  if(tensor->data == CF_NULL)
  {
    old_bytes = 0;
    data = malloc(bytes);
  }
  else
  {
    data = realloc(tensor->data, bytes);
  }

  if(data == CF_NULL) return CF_ERR_OOM;

  tensor->data = data;
  memset((char *)tensor->data + old_bytes, 0, bytes - old_bytes);
  tensor->metadata.capacity = capacity;
  tensor->device = CF_TENSOR_DEVICE_CPU;

#ifdef CF_CUDA_AVAILABLE
  if(tensor->device_data != CF_NULL) (void)cf_tensor_free_gpu(tensor);
#endif

  return CF_OK;
}

cf_status cf_tensor_reshape_cpu(cf_tensor *tensor, const cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank)
{
  cf_usize len;
  cf_status status;

  if(tensor == CF_NULL) return CF_ERR_NULL;
  status = cf_tensor_shape_len(dim, rank, &len);
  if(status != CF_OK) return status;
  if(len > tensor->metadata.capacity) return CF_ERR_BOUNDS;

  cf_tensor_apply_shape(tensor, dim, rank, len);
  return CF_OK;
}

cf_status cf_tensor_resize_cpu(cf_tensor *tensor, const cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank)
{
  cf_usize len;
  cf_usize old_len;
  cf_usize old_bytes;
  cf_usize new_bytes;
  cf_status status;

  if(tensor == CF_NULL) return CF_ERR_NULL;

  status = cf_tensor_shape_len(dim, rank, &len);
  if(status != CF_OK) return status;

  old_len = tensor->metadata.len;
  if(len > tensor->metadata.capacity)
  {
    status = cf_tensor_reserve_cpu(tensor, len);
    if(status != CF_OK) return status;
  }
  else if(len > old_len && tensor->data != CF_NULL)
  {
    status = cf_tensor_checked_bytes(old_len, tensor->metadata.elem_size, &old_bytes);
    if(status != CF_OK) return status;
    status = cf_tensor_checked_bytes(len, tensor->metadata.elem_size, &new_bytes);
    if(status != CF_OK) return status;
    memset((char *)tensor->data + old_bytes, 0, new_bytes - old_bytes);
  }

  cf_tensor_apply_shape(tensor, dim, rank, len);
  tensor->device = CF_TENSOR_DEVICE_CPU;
  return CF_OK;
}

cf_status cf_tensor_copy_cpu(cf_tensor *dst, const cf_tensor *src)
{
  cf_status status;
  cf_usize bytes;

  if(dst == CF_NULL || src == CF_NULL) return CF_ERR_NULL;
  if(src->data == CF_NULL) return CF_ERR_STATE;
  if(dst == src) return CF_OK;

  if(dst->metadata.elem_size == 0 && dst->data == CF_NULL && dst->device_data == CF_NULL)
  {
    status = cf_tensor_init_cpu(dst, src->dim, src->rank, src->metadata.elem_type);
    if(status != CF_OK) return status;
  }
  else
  {
    if(dst->metadata.elem_type != src->metadata.elem_type) return CF_ERR_INVALID;
    if(dst->metadata.elem_size != src->metadata.elem_size) return CF_ERR_INVALID;
    status = cf_tensor_resize_cpu(dst, src->dim, src->rank);
    if(status != CF_OK) return status;
  }

  status = cf_tensor_checked_bytes(src->metadata.len, src->metadata.elem_size, &bytes);
  if(status != CF_OK) return status;
  memcpy(dst->data, src->data, bytes);
  dst->device = CF_TENSOR_DEVICE_CPU;

  return CF_OK;
}

cf_status cf_tensor_copy_from_array_cpu(cf_tensor *tensor, const void *array, cf_usize count)
{
  cf_usize dim[CF_TENSOR_HIGHEST_RANK] = {0};
  cf_usize bytes;
  cf_status status;

  if(tensor == CF_NULL || array == CF_NULL) return CF_ERR_NULL;
  if(count == 0) return CF_ERR_INVALID;
  if(tensor->metadata.elem_size == 0) return CF_ERR_INVALID;

  if(count > tensor->metadata.capacity)
  {
    status = cf_tensor_reserve_cpu(tensor, count);
    if(status != CF_OK) return status;
  }

  dim[0] = count;
  cf_tensor_apply_shape(tensor, dim, 1, count);

  status = cf_tensor_checked_bytes(count, tensor->metadata.elem_size, &bytes);
  if(status != CF_OK) return status;

  memcpy(tensor->data, array, bytes);
  tensor->device = CF_TENSOR_DEVICE_CPU;
  return CF_OK;
}

cf_status cf_tensor_copy_to_array_cpu(void *array, const cf_tensor *tensor, cf_usize count)
{
  cf_usize bytes;
  cf_status status;

  if(array == CF_NULL || tensor == CF_NULL) return CF_ERR_NULL;
  if(tensor->data == CF_NULL) return CF_ERR_STATE;
  if(count < tensor->metadata.len) return CF_ERR_BOUNDS;

  status = cf_tensor_checked_bytes(tensor->metadata.len, tensor->metadata.elem_size, &bytes);
  if(status != CF_OK) return status;

  memcpy(array, tensor->data, bytes);
  return CF_OK;
}

cf_status cf_tensor_get_cpu(void *out_value, const cf_tensor *tensor, const cf_usize indexs[CF_TENSOR_HIGHEST_RANK])
{
  cf_usize index;
  cf_status status;

  if(out_value == CF_NULL) return CF_ERR_NULL;

  status = cf_tensor_flat_index(tensor, indexs, &index);
  if(status != CF_OK) return status;

  switch(tensor->metadata.elem_type)
  {
    CF_TENSOR_GET_CASE(CF_TENSOR_CHAR, char);
    CF_TENSOR_GET_CASE(CF_TENSOR_SHORT, short);
    CF_TENSOR_GET_CASE(CF_TENSOR_INT, int);
    CF_TENSOR_GET_CASE(CF_TENSOR_LONG, long);
    CF_TENSOR_GET_CASE(CF_TENSOR_LL, long long);
    CF_TENSOR_GET_CASE(CF_TENSOR_FLOAT, float);
    CF_TENSOR_GET_CASE(CF_TENSOR_DOUBLE, double);
    CF_TENSOR_GET_CASE(CF_TENSOR_LD, long double);
    CF_TENSOR_GET_CASE(CF_TENSOR_U8, cf_u8);
    CF_TENSOR_GET_CASE(CF_TENSOR_U16, cf_u16);
    CF_TENSOR_GET_CASE(CF_TENSOR_U32, cf_u32);
    CF_TENSOR_GET_CASE(CF_TENSOR_U64, cf_u64);
    CF_TENSOR_GET_CASE(CF_TENSOR_U128, cf_u128);
    default: return CF_ERR_INVALID;
  }

  return CF_OK;
}

cf_status cf_tensor_set_cpu(cf_tensor *tensor, const cf_usize indexs[CF_TENSOR_HIGHEST_RANK], const void *value)
{
  cf_usize index;
  cf_status status;

  if(value == CF_NULL) return CF_ERR_NULL;

  status = cf_tensor_flat_index(tensor, indexs, &index);
  if(status != CF_OK) return status;

  switch(tensor->metadata.elem_type)
  {
    CF_TENSOR_SET_CASE(CF_TENSOR_CHAR, char);
    CF_TENSOR_SET_CASE(CF_TENSOR_SHORT, short);
    CF_TENSOR_SET_CASE(CF_TENSOR_INT, int);
    CF_TENSOR_SET_CASE(CF_TENSOR_LONG, long);
    CF_TENSOR_SET_CASE(CF_TENSOR_LL, long long);
    CF_TENSOR_SET_CASE(CF_TENSOR_FLOAT, float);
    CF_TENSOR_SET_CASE(CF_TENSOR_DOUBLE, double);
    CF_TENSOR_SET_CASE(CF_TENSOR_LD, long double);
    CF_TENSOR_SET_CASE(CF_TENSOR_U8, cf_u8);
    CF_TENSOR_SET_CASE(CF_TENSOR_U16, cf_u16);
    CF_TENSOR_SET_CASE(CF_TENSOR_U32, cf_u32);
    CF_TENSOR_SET_CASE(CF_TENSOR_U64, cf_u64);
    CF_TENSOR_SET_CASE(CF_TENSOR_U128, cf_u128);
    default: return CF_ERR_INVALID;
  }

  tensor->device = CF_TENSOR_DEVICE_CPU;
  return CF_OK;
}

cf_status cf_tensor_add_cpu(cf_tensor *op1, const cf_tensor *op2)
{
  cf_usize len = op1->metadata.len;

  switch(op1->metadata.elem_type)
  {
    CF_TENSOR_ADD_CASE(CF_TENSOR_CHAR, char);
    CF_TENSOR_ADD_CASE(CF_TENSOR_SHORT, short);
    CF_TENSOR_ADD_CASE(CF_TENSOR_INT, int);
    CF_TENSOR_ADD_CASE(CF_TENSOR_LONG, long);
    CF_TENSOR_ADD_CASE(CF_TENSOR_LL, long long);
    CF_TENSOR_ADD_CASE(CF_TENSOR_FLOAT, float);
    CF_TENSOR_ADD_CASE(CF_TENSOR_DOUBLE, double);
    CF_TENSOR_ADD_CASE(CF_TENSOR_LD, long double);
    CF_TENSOR_ADD_CASE(CF_TENSOR_U8, cf_u8);
    CF_TENSOR_ADD_CASE(CF_TENSOR_U16, cf_u16);
    CF_TENSOR_ADD_CASE(CF_TENSOR_U32, cf_u32);
    CF_TENSOR_ADD_CASE(CF_TENSOR_U64, cf_u64);
    CF_TENSOR_ADD_CASE(CF_TENSOR_U128, cf_u128);
    default: return CF_ERR_UNSUPPORTED;
  }

  op1->device = CF_TENSOR_DEVICE_CPU;
  return CF_OK;
}

cf_status cf_tensor_mul_cpu(cf_tensor *op1, const cf_tensor *op2)
{
  cf_usize len = op1->metadata.len;

  switch(op1->metadata.elem_type)
  {
    CF_TENSOR_MUL_CASE(CF_TENSOR_CHAR, char);
    CF_TENSOR_MUL_CASE(CF_TENSOR_SHORT, short);
    CF_TENSOR_MUL_CASE(CF_TENSOR_INT, int);
    CF_TENSOR_MUL_CASE(CF_TENSOR_LONG, long);
    CF_TENSOR_MUL_CASE(CF_TENSOR_LL, long long);
    CF_TENSOR_MUL_CASE(CF_TENSOR_FLOAT, float);
    CF_TENSOR_MUL_CASE(CF_TENSOR_DOUBLE, double);
    CF_TENSOR_MUL_CASE(CF_TENSOR_LD, long double);
    CF_TENSOR_MUL_CASE(CF_TENSOR_U8, cf_u8);
    CF_TENSOR_MUL_CASE(CF_TENSOR_U16, cf_u16);
    CF_TENSOR_MUL_CASE(CF_TENSOR_U32, cf_u32);
    CF_TENSOR_MUL_CASE(CF_TENSOR_U64, cf_u64);
    CF_TENSOR_MUL_CASE(CF_TENSOR_U128, cf_u128);
    default: return CF_ERR_UNSUPPORTED;
  }

  op1->device = CF_TENSOR_DEVICE_CPU;
  return CF_OK;
}

cf_status cf_tensor_scalar_mul_cpu(cf_tensor *op1, const void *scalar)
{
  cf_usize len;

  if(scalar == CF_NULL) return CF_ERR_NULL;
  len = op1->metadata.len;

  switch(op1->metadata.elem_type)
  {
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_CHAR, char);
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_SHORT, short);
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_INT, int);
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_LONG, long);
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_LL, long long);
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_FLOAT, float);
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_DOUBLE, double);
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_LD, long double);
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_U8, cf_u8);
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_U16, cf_u16);
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_U32, cf_u32);
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_U64, cf_u64);
    CF_TENSOR_SCALAR_CASE(CF_TENSOR_U128, cf_u128);
    default: return CF_ERR_UNSUPPORTED;
  }

  op1->device = CF_TENSOR_DEVICE_CPU;
  return CF_OK;
}

cf_status cf_tensor_batch_mul_cpu(cf_tensor *op1, const cf_tensor *op2)
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
  void *result;

  status = cf_tensor_batch_mul_shape(op1, op2, out_dim, &out_rank, &out_len, &batch_count);
  if(status != CF_OK) return status;

  batch_rank = out_rank - 2;
  rows = op1->dim[op1->rank - 2];
  inner = op1->dim[op1->rank - 1];
  cols = op2->dim[op2->rank - 1];
  cf_tensor_dense_stride(out_dim, out_rank, out_stride);

  status = cf_tensor_checked_bytes(out_len, op1->metadata.elem_size, &out_bytes);
  if(status != CF_OK) return status;

  result = malloc(out_bytes);
  if(result == CF_NULL) return CF_ERR_OOM;

  switch(op1->metadata.elem_type)
  {
    CF_TENSOR_BATCH_MUL_CASE(CF_TENSOR_CHAR, char);
    CF_TENSOR_BATCH_MUL_CASE(CF_TENSOR_SHORT, short);
    CF_TENSOR_BATCH_MUL_CASE(CF_TENSOR_INT, int);
    CF_TENSOR_BATCH_MUL_CASE(CF_TENSOR_LONG, long);
    CF_TENSOR_BATCH_MUL_CASE(CF_TENSOR_LL, long long);
    CF_TENSOR_BATCH_MUL_CASE(CF_TENSOR_FLOAT, float);
    CF_TENSOR_BATCH_MUL_CASE(CF_TENSOR_DOUBLE, double);
    CF_TENSOR_BATCH_MUL_CASE(CF_TENSOR_LD, long double);
    CF_TENSOR_BATCH_MUL_CASE(CF_TENSOR_U8, cf_u8);
    CF_TENSOR_BATCH_MUL_CASE(CF_TENSOR_U16, cf_u16);
    CF_TENSOR_BATCH_MUL_CASE(CF_TENSOR_U32, cf_u32);
    CF_TENSOR_BATCH_MUL_CASE(CF_TENSOR_U64, cf_u64);
    CF_TENSOR_BATCH_MUL_CASE(CF_TENSOR_U128, cf_u128);
    default:
      free(result);
      return CF_ERR_UNSUPPORTED;
  }

#ifdef CF_CUDA_AVAILABLE
  if(op1->device_data != CF_NULL) (void)cf_tensor_free_gpu(op1);
#endif

  if(op1->data != CF_NULL && out_len <= op1->metadata.capacity)
  {
    memcpy(op1->data, result, out_bytes);
    free(result);
  }
  else
  {
    if(op1->data != CF_NULL) free(op1->data);
    op1->data = result;
    op1->metadata.capacity = out_len;
  }

  cf_tensor_apply_shape(op1, out_dim, out_rank, out_len);
  op1->device = CF_TENSOR_DEVICE_CPU;

  return CF_OK;
}

cf_status cf_tensor_matrix_mul_cpu(cf_tensor *op1, const cf_tensor *op2)
{
  return cf_tensor_batch_mul_cpu(op1, op2);
}

static void cf_tensor_print_indent(cf_usize indent)
{
  for(cf_usize i = 0; i < indent; i++) printf(" ");
}

static void cf_tensor_print_u128(cf_u128 value)
{
  char buffer[40];
  cf_usize i = sizeof(buffer);

  buffer[--i] = '\0';
  if(value == 0)
  {
    printf("%10s", "0");
    return;
  }

  while(value != 0)
  {
    buffer[--i] = (char)('0' + (value % 10));
    value /= 10;
  }

  printf("%10s", &buffer[i]);
}

static void cf_tensor_print_value(const cf_tensor *tensor, cf_usize index)
{
  switch(tensor->metadata.elem_type)
  {
    case CF_TENSOR_CHAR:
      printf("%10c", ((const char *)tensor->data)[index]);
      break;
    case CF_TENSOR_SHORT:
      printf("%10hd", ((const short *)tensor->data)[index]);
      break;
    case CF_TENSOR_INT:
      printf("%10d", ((const int *)tensor->data)[index]);
      break;
    case CF_TENSOR_LONG:
      printf("%10ld", ((const long *)tensor->data)[index]);
      break;
    case CF_TENSOR_LL:
      printf("%10lld", ((const long long *)tensor->data)[index]);
      break;
    case CF_TENSOR_FLOAT:
      printf("%10g", (double)((const float *)tensor->data)[index]);
      break;
    case CF_TENSOR_DOUBLE:
      printf("%10g", ((const double *)tensor->data)[index]);
      break;
    case CF_TENSOR_LD:
      printf("%10Lg", ((const long double *)tensor->data)[index]);
      break;
    case CF_TENSOR_U8:
      printf("%10u", (unsigned int)((const cf_u8 *)tensor->data)[index]);
      break;
    case CF_TENSOR_U16:
      printf("%10u", (unsigned int)((const cf_u16 *)tensor->data)[index]);
      break;
    case CF_TENSOR_U32:
      printf("%10u", ((const cf_u32 *)tensor->data)[index]);
      break;
    case CF_TENSOR_U64:
      printf("%10llu", (unsigned long long)((const cf_u64 *)tensor->data)[index]);
      break;
    case CF_TENSOR_U128:
      cf_tensor_print_u128(((const cf_u128 *)tensor->data)[index]);
      break;
    default:
      printf("%10s", "?");
      break;
  }
}

static void cf_tensor_print_axis(const cf_tensor *tensor, cf_usize axis, cf_usize base, cf_usize indent)
{
  if(axis == tensor->rank - 1)
  {
    printf("[");
    for(cf_usize i = 0; i < tensor->dim[axis]; i++)
    {
      if(i != 0) printf(", ");
      cf_tensor_print_value(tensor, base + i * tensor->metadata.stride[axis]);
    }
    printf("]");
    return;
  }

  printf("[\n");
  for(cf_usize i = 0; i < tensor->dim[axis]; i++)
  {
    cf_tensor_print_indent(indent + 2);
    cf_tensor_print_axis(tensor, axis + 1, base + i * tensor->metadata.stride[axis], indent + 2);

    if(i + 1 != tensor->dim[axis]) printf(",");
    printf("\n");
  }
  cf_tensor_print_indent(indent);
  printf("]");
}

void cf_tensor_print(const cf_tensor *tensor)
{
  if(tensor == CF_NULL)
  {
    printf("(null)\n");
    return;
  }

  if(tensor->data == CF_NULL)
  {
    printf("(empty tensor)\n");
    return;
  }

  if(!cf_tensor_is_valid(tensor))
  {
    printf("(invalid tensor)\n");
    return;
  }

  if(tensor->rank == 0)
  {
    cf_tensor_print_value(tensor, 0);
    printf("\n");
    return;
  }

  cf_tensor_print_axis(tensor, 0, 0, 0);
  printf("\n");
}
