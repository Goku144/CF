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

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdio.h>
#include <stdlib.h>

static cf_usize cf_math_print_type_size(cf_math_dtype dtype)
{
  switch (dtype)
  {
    case CF_MATH_DTYPE_BOOL: return sizeof (cf_bool);
    case CF_MATH_DTYPE_I8: return sizeof (cf_i8);
    case CF_MATH_DTYPE_U8: return sizeof (cf_u8);
    case CF_MATH_DTYPE_I32: return sizeof (cf_i32);
    case CF_MATH_DTYPE_FP8E5M2: return sizeof (cf_u8);
    case CF_MATH_DTYPE_FP8E4M3: return sizeof (cf_u8);
    case CF_MATH_DTYPE_BF16: return sizeof (cf_u16);
    case CF_MATH_DTYPE_F16: return sizeof (cf_u16);
    case CF_MATH_DTYPE_F32: return sizeof (float);
    case CF_MATH_DTYPE_F64: return sizeof (double);
  }
  return (cf_usize) -1;
}

static void cf_math_print_indent(int depth)
{
  for (int i = 0; i < depth; i++) printf("  ");
}

static void cf_math_print_value(const void *data, cf_math_dtype dtype, cf_usize index)
{
  switch (dtype)
  {
    case CF_MATH_DTYPE_BOOL: printf("%s", ((const cf_bool *) data)[index] ? "true" : "false"); break;
    case CF_MATH_DTYPE_I8: printf("%d", (int) ((const cf_i8 *) data)[index]); break;
    case CF_MATH_DTYPE_U8: printf("%u", (unsigned) ((const cf_u8 *) data)[index]); break;
    case CF_MATH_DTYPE_I32: printf("%d", ((const cf_i32 *) data)[index]); break;
    case CF_MATH_DTYPE_FP8E5M2: printf("0x%02x", (unsigned) ((const cf_u8 *) data)[index]); break;
    case CF_MATH_DTYPE_FP8E4M3: printf("0x%02x", (unsigned) ((const cf_u8 *) data)[index]); break;
    case CF_MATH_DTYPE_BF16: printf("%g", (double) __bfloat162float(((__nv_bfloat16 *) data)[index])); break;
    case CF_MATH_DTYPE_F16: printf("%g", (double) __half2float(((__half *) data)[index])); break;
    case CF_MATH_DTYPE_F32: printf("%g", (double) ((const float *) data)[index]); break;
    case CF_MATH_DTYPE_F64: printf("%g", ((const double *) data)[index]); break;
  }
}

static void cf_math_print_tensor(const void *data, const cf_math_desc *desc, int axis, cf_usize base, int depth)
{
  if(axis == desc->rank - 1)
  {
    printf("[");
    for (int i = 0; i < desc->dim[axis]; i++)
    {
      if(i != 0) printf(", ");
      cf_math_print_value(data, desc->dtype, base + (cf_usize) i * (cf_usize) desc->strides[axis]);
    }
    printf("]");
    return;
  }

  printf("[\n");
  for (int i = 0; i < desc->dim[axis]; i++)
  {
    cf_math_print_indent(depth + 1);
    cf_math_print_tensor(data, desc, axis + 1, base + (cf_usize) i * (cf_usize) desc->strides[axis], depth + 1);
    if(i != desc->dim[axis] - 1) printf(",");
    printf("\n");
  }
  cf_math_print_indent(depth);
  printf("]");
}

extern "C" {

cf_status cf_math_print(cf_math_handle *handle, const cf_math *math)
{
  if(handle == CF_NULL || math == CF_NULL || math->desc == CF_NULL) return CF_ERR_NULL;

  cf_usize elem_size = cf_math_print_type_size(math->desc->dtype);
  cf_usize bytes = (cf_usize) math->desc->dim[0] * (cf_usize) math->desc->strides[0] * elem_size;
  const cf_u8 *src = (const cf_u8 *) handle->storage.backend + math->byte_offset;
  void *host = CF_NULL;
  const void *data = src;
  cf_status state = CF_OK;

  if(handle->device == CF_MATH_DEVICE_CUDA)
  {
    host = malloc(bytes);
    if(host == CF_NULL) return CF_ERR_OOM;
    if(cudaMemcpy(host, src, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) { state = CF_ERR_CUDA; goto done; }
    data = host;
  }

  cf_math_print_tensor(data, math->desc, 0, 0, 0);
  printf("\n");

done:
  if(host != CF_NULL) free(host);
  return state;
}

}
