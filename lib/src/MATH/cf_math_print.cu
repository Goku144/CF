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

cf_status cf_math_print_shape(const cf_math *x)
{
  const cf_math_metadata *metadata = CF_NULL;

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
  printf("}\n");

  return CF_OK;
}