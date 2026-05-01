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

#include "MATH/cf_math_print.h"
#include "MATH/cf_math.h"
#include "ALLOCATOR/cf_alloc.h"

#include <stdio.h>

static const char *cf_math_shape_name(cf_math_shape shape)
{
  switch(shape)
  {
    case CF_MATH_SHAPE_SCALAR: return "scalar";
    case CF_MATH_SHAPE_VECTOR: return "vector";
    case CF_MATH_SHAPE_MATRIX: return "matrix";
    case CF_MATH_SHAPE_TENSOR: return "tensor";
  }
  return "unknown";
}

static const char *cf_math_layout_name(cf_math_layout layout)
{
  switch(layout)
  {
    case CF_MATH_LAYOUT_ROW_MAJOR: return "row_major";
    case CF_MATH_LAYOUT_COL_MAJOR: return "col_major";
    case CF_MATH_LAYOUT_NCHW: return "nchw";
    case CF_MATH_LAYOUT_NHWC: return "nhwc";
    case CF_MATH_LAYOUT_STRIDED: return "strided";
  }
  return "unknown";
}

static const char *cf_math_dtype_name(cf_math_dtype dtype)
{
  switch(dtype)
  {
    case CF_MATH_DTYPE_BOOL: return "bool";
    case CF_MATH_DTYPE_I8: return "i8";
    case CF_MATH_DTYPE_U8: return "u8";
    case CF_MATH_DTYPE_I32: return "i32";
    case CF_MATH_DTYPE_F64: return "f64";
    case CF_MATH_DTYPE_F32: return "f32";
    case CF_MATH_DTYPE_F16: return "f16";
    case CF_MATH_DTYPE_BF16: return "bf16";
    case CF_MATH_DTYPE_FP8E4M3: return "fp8e4m3";
    case CF_MATH_DTYPE_FP8E5M2: return "fp8e5m2";
  }
  return "unknown";
}

static const char *cf_math_device_name(cf_math_device device)
{
  switch(device)
  {
    case CF_MATH_DEVICE_CPU: return "cpu";
    case CF_MATH_DEVICE_CUDA: return "cuda";
  }
  return "unknown";
}

static void cf_math_print_data_values(const cf_math *x, const void *host_data)
{
  const cf_math_metadata *metadata = x->metadata;
  cf_math_dtype dtype = x->handler->storage.dtype;
  cf_usize count = metadata->len < 16 ? metadata->len : 16;

  printf("  data: [");
  for(cf_usize i = 0; i < count; ++i)
  {
    printf("%s", i == 0 ? "" : ", ");
    switch(dtype)
    {
      case CF_MATH_DTYPE_BOOL:
        printf("%s", ((const cf_bool *)host_data)[i] == CF_TRUE ? "true" : "false");
        break;
      case CF_MATH_DTYPE_I8:
        printf("%d", (int)((const cf_i8 *)host_data)[i]);
        break;
      case CF_MATH_DTYPE_U8:
      case CF_MATH_DTYPE_FP8E4M3:
      case CF_MATH_DTYPE_FP8E5M2:
        printf("%u", (unsigned)((const cf_u8 *)host_data)[i]);
        break;
      case CF_MATH_DTYPE_I32:
        printf("%d", ((const cf_i32 *)host_data)[i]);
        break;
      case CF_MATH_DTYPE_F64:
        printf("%g", ((const double *)host_data)[i]);
        break;
      case CF_MATH_DTYPE_F32:
        printf("%g", (double)((const float *)host_data)[i]);
        break;
      case CF_MATH_DTYPE_F16:
      case CF_MATH_DTYPE_BF16:
        printf("0x%04x", (unsigned)((const cf_u16 *)host_data)[i]);
        break;
    }
  }
  if(metadata->len > count) printf(", ...");
  printf("]\n");
}

static void cf_math_print_data_value(cf_math_dtype dtype, const void *host_data, cf_usize index)
{
  switch(dtype)
  {
    case CF_MATH_DTYPE_BOOL:
      printf("%s", ((const cf_bool *)host_data)[index] == CF_TRUE ? "true" : "false");
      break;
    case CF_MATH_DTYPE_I8:
      printf("%d", (int)((const cf_i8 *)host_data)[index]);
      break;
    case CF_MATH_DTYPE_U8:
    case CF_MATH_DTYPE_FP8E4M3:
    case CF_MATH_DTYPE_FP8E5M2:
      printf("%u", (unsigned)((const cf_u8 *)host_data)[index]);
      break;
    case CF_MATH_DTYPE_I32:
      printf("%d", ((const cf_i32 *)host_data)[index]);
      break;
    case CF_MATH_DTYPE_F64:
      printf("%g", ((const double *)host_data)[index]);
      break;
    case CF_MATH_DTYPE_F32:
      printf("%g", (double)((const float *)host_data)[index]);
      break;
    case CF_MATH_DTYPE_F16:
    case CF_MATH_DTYPE_BF16:
      printf("0x%04x", (unsigned)((const cf_u16 *)host_data)[index]);
      break;
  }
}

static void cf_math_print_indent(cf_usize depth)
{
  for(cf_usize i = 0; i < depth; ++i)
    printf("  ");
}

static void cf_math_print_tensor_level(const cf_math_metadata *metadata, cf_math_dtype dtype, const void *host_data, cf_usize level, cf_usize offset, cf_usize indent)
{
  printf("[");
  if(level < metadata->rank)
  {
    for(cf_usize i = 0; i < metadata->dim[level]; ++i)
    {
      if(i != 0) printf(",");
      if(level + 1 < metadata->rank)
      {
        printf("\n");
        cf_math_print_indent(indent + 1);
      }
      else if(i != 0)
      {
        printf(" ");
      }

      if(level + 1 == metadata->rank)
      {
        cf_math_print_data_value(dtype, host_data, offset + i * metadata->strides[level]);
      }
      else
      {
        cf_math_print_tensor_level(metadata, dtype, host_data, level + 1, offset + i * metadata->strides[level], indent + 1);
      }
    }

    if(level + 1 < metadata->rank && metadata->dim[level] != 0)
    {
      printf("\n");
      cf_math_print_indent(indent);
    }
  }
  printf("]");
}

cf_status cf_math_print_shape(const cf_math *x)
{
  const cf_math_metadata *metadata = CF_NULL;
  void *host_data = CF_NULL;
  cf_alloc allocator;
  cf_status status = CF_OK;

  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata == CF_NULL) return CF_ERR_STATE;

  metadata = x->metadata;

  printf("cf_math {\n");
  printf("  shape: %s\n", cf_math_shape_name(metadata->shape));
  printf("  rank: %zu\n", (size_t)metadata->rank);
  printf("  dim: [");
  for(cf_usize i = 0; i < metadata->rank; ++i)
    printf("%s%zu", i == 0 ? "" : ", ", (size_t)metadata->dim[i]);
  printf("]\n");
  printf("  strides: [");
  for(cf_usize i = 0; i < metadata->rank; ++i)
    printf("%s%zu", i == 0 ? "" : ", ", (size_t)metadata->strides[i]);
  printf("]\n");
  printf("  len: %zu\n", (size_t)metadata->len);
  printf("  layout: %s\n", cf_math_layout_name(metadata->layout));
  if(x->handler != CF_NULL)
  {
    printf("  dtype: %s\n", cf_math_dtype_name(x->handler->storage.dtype));
    printf("  device: %s\n", cf_math_device_name(x->handler->storage.device));
  }
  else
  {
    printf("  dtype: unknown\n");
    printf("  device: unknown\n");
  }
  printf("  bytes: offset=%zu size=%zu\n", (size_t)x->byte_offset, (size_t)x->byte_size);
  if(x->handler != CF_NULL && x->byte_size != 0)
  {
    cf_alloc_new(&allocator);
    host_data = allocator.alloc(allocator.ctx, x->byte_size);
    if(host_data == CF_NULL) return CF_ERR_OOM;

    status = cf_math_cpy_d2h(x, host_data, x->metadata->len);
    if(status != CF_OK)
    {
      allocator.free(allocator.ctx, host_data);
      return status;
    }

    cf_math_print_data_values(x, host_data);
    allocator.free(allocator.ctx, host_data);
  }
  printf("}\n");

  return CF_OK;
}

cf_status cf_math_print_tensor(const cf_math *x)
{
  void *host_data = CF_NULL;
  cf_alloc allocator;
  cf_status status = CF_OK;

  if(x == CF_NULL) return CF_ERR_NULL;
  if(x->metadata == CF_NULL || x->handler == CF_NULL) return CF_ERR_STATE;

  if(x->metadata->rank == 0 || x->metadata->len == 0 || x->byte_size == 0)
  {
    printf("[]\n");
    return CF_OK;
  }

  cf_alloc_new(&allocator);
  host_data = allocator.alloc(allocator.ctx, x->byte_size);
  if(host_data == CF_NULL) return CF_ERR_OOM;

  status = cf_math_cpy_d2h(x, host_data, x->metadata->len);
  if(status != CF_OK)
  {
    allocator.free(allocator.ctx, host_data);
    return status;
  }

  cf_math_print_tensor_level(x->metadata, x->handler->storage.dtype, host_data, 0, 0, 0);
  printf("\n");

  allocator.free(allocator.ctx, host_data);
  return CF_OK;
}
