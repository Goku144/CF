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
#include "MATH/cf_math_storage.h"
#include "RUNTIME/cf_random.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

cf_u8 cf_math_g8_mul_mod(cf_u8 p, cf_u8 q)
{
  cf_u8 res = 0;
  do
  {
    if(q & 0x01) res ^= p;
    if(p & 0x80) p = (p << 1) ^ 0x1B;
    else p <<= 1;
  } while(q >>= 1);
  return res;
}

cf_u8 cf_math_rotl8(cf_u8 x, cf_u8 n)
{
  n &= 7;
  return n == 0 ? x : (cf_u8)((x << n) | (x >> (8 - n)));
}

cf_u8 cf_math_rotr8(cf_u8 x, cf_u8 n)
{
  n &= 7;
  return n == 0 ? x : (cf_u8)((x >> n) | (x << (8 - n)));
}

cf_u32 cf_math_rotl32(cf_u32 x, cf_u8 n)
{
  n &= 31;
  return n == 0 ? x : (x << n) | (x >> (32 - n));
}

cf_u32 cf_math_rotr32(cf_u32 x, cf_u8 n)
{
  n &= 31;
  return n == 0 ? x : (x >> n) | (x << (32 - n));
}

cf_usize cf_math_min_usize(cf_usize a, cf_usize b)
{
  return a <= b ? a : b;
}

cf_usize cf_math_max_usize(cf_usize a, cf_usize b)
{
  return a >= b ? a : b;
}

static cf_usize cf_math_dtype_size(cf_math_dtype dtype)
{
  switch (dtype)
  {
    case CF_MATH_DTYPE_BOOL: return sizeof(cf_bool);
    case CF_MATH_DTYPE_I8: return sizeof(cf_i8);
    case CF_MATH_DTYPE_U8: return sizeof(cf_u8);
    case CF_MATH_DTYPE_I32: return sizeof(cf_i32);
    case CF_MATH_DTYPE_F64: return sizeof(double);
    case CF_MATH_DTYPE_F32: return sizeof(float);
    case CF_MATH_DTYPE_F16: return sizeof(cf_u16);
    case CF_MATH_DTYPE_BF16: return sizeof(cf_u16);
    case CF_MATH_DTYPE_FP8E4M3: return sizeof(cf_u8);
    case CF_MATH_DTYPE_FP8E5M2: return sizeof(cf_u8);
  }
  return 0;
}

cf_status cf_math_metadata_init(cf_math_metadata *metadata, cf_usize dim[CF_MATH_MAX_RANK], cf_usize rank, cf_math_shape shape, cf_math_layout layout)
{
  cf_usize len = 1;

  if(metadata == CF_NULL) return CF_ERR_NULL;
  if(rank > CF_MATH_MAX_RANK) return CF_ERR_INVALID;
  if(dim == CF_NULL && rank != 0) return CF_ERR_INVALID;

  memset(metadata, 0, sizeof(*metadata));
  if(rank != 0) memcpy(metadata->dim, dim, rank * sizeof(cf_usize));
  metadata->rank = rank;
  metadata->shape = shape;
  metadata->layout = layout;

  if(rank == 0)
  {
    metadata->len = 1;
    return CF_OK;
  }

  for(cf_usize i = 0; i < rank; ++i)
  {
    if(dim[i] != 0 && len > (cf_usize)-1 / dim[i]) return CF_ERR_OVERFLOW;
    len *= dim[i];
  }

  if(layout == CF_MATH_LAYOUT_COL_MAJOR)
  {
    metadata->strides[0] = 1;
    for(cf_usize i = 1; i < rank; ++i)
    {
      if(metadata->strides[i - 1] != 0 && dim[i - 1] > (cf_usize)-1 / metadata->strides[i - 1])
        return CF_ERR_OVERFLOW;
      metadata->strides[i] = metadata->strides[i - 1] * dim[i - 1];
    }
  }
  else
  {
    metadata->strides[rank - 1] = 1;
    for(cf_usize i = rank - 1; i > 0; --i)
    {
      if(metadata->strides[i] != 0 && dim[i] > (cf_usize)-1 / metadata->strides[i])
        return CF_ERR_OVERFLOW;
      metadata->strides[i - 1] = metadata->strides[i] * dim[i];
    }
  }

  metadata->len = len;
  return CF_OK;
}

cf_status cf_math_bind(cf_math *x, cf_math_handle_t *handler, cf_math_metadata *metadata)
{
  void *data = CF_NULL;
  cf_usize elem_size = 0;
  cf_usize bytes = 0;
  cf_usize offset = 0;
  cf_usize required = 0;
  cf_status status = CF_OK;

  if(x == CF_NULL || handler == CF_NULL || metadata == CF_NULL) return CF_ERR_NULL;

  elem_size = cf_math_dtype_size(handler->storage.dtype);
  if(elem_size == 0) return CF_ERR_INVALID;
  if(metadata->len > (cf_usize)-1 / elem_size) return CF_ERR_OVERFLOW;

  bytes = metadata->len * elem_size;
  offset = handler->storage.arena.offset;
  if((handler->storage.allocator.mem_flag & CF_MATH_MEM_ALIGNED128) != 0)
  {
    if(offset > (cf_usize)-1 - 127U) return CF_ERR_OVERFLOW;
    offset = (offset + 127U) & ~((cf_usize)127U);
  }
  if(offset > (cf_usize)-1 - bytes) return CF_ERR_OVERFLOW;

  required = offset + bytes;
  if(required > handler->storage.arena.capacity)
  {
    status = cf_math_handle_reserve(handler, required);
    if(status != CF_OK) return status;
  }

  status = cf_math_handle_alloc(handler, bytes, &data);
  if(status != CF_OK) return status;

  x->data = data;
  x->byte_offset = data != CF_NULL ? (cf_usize)((cf_u8 *)data - (cf_u8 *)handler->storage.allocator.backend) : 0;
  x->byte_size = bytes;
  x->metadata = metadata;
  x->handler = handler;
  x->grad = CF_NULL;
  x->grad_fn = CF_NULL;
  x->grad_state = CF_MATH_GRAD_NONE;

  return CF_OK;
}

#if defined(CF_CUDA_AVAILABLE)
static cf_status cf_math_copy_sync(const cf_math_handle_t *handler)
{
  if(handler != CF_NULL && handler->cuda_ctx != CF_NULL && handler->cuda_ctx->stream != CF_NULL)
    return cudaStreamSynchronize(handler->cuda_ctx->stream) == cudaSuccess ? CF_OK : CF_ERR_CUDA_SYNC;
  return cudaDeviceSynchronize() == cudaSuccess ? CF_OK : CF_ERR_CUDA_SYNC;
}
#endif

static cf_status cf_math_copy_bytes(const cf_math *x, void *dst, const void *src, cf_usize bytes)
{
  if(bytes == 0) return CF_OK;
  if(x == CF_NULL || dst == CF_NULL || src == CF_NULL) return CF_ERR_NULL;
  if(x->handler == CF_NULL) return CF_ERR_STATE;

  if(x->handler->storage.device == CF_MATH_DEVICE_CPU || (x->handler->storage.allocator.mem_flag & CF_MATH_MEM_PINNED) != 0)
  {
    memcpy(dst, src, bytes);
    return CF_OK;
  }

#if defined(CF_CUDA_AVAILABLE)
  if(cudaMemcpy(dst, src, (size_t)bytes, cudaMemcpyDefault) != cudaSuccess)
    return CF_ERR_CUDA_COPY;
  return cf_math_copy_sync(x->handler);
#else
  return CF_ERR_UNSUPPORTED;
#endif
}

cf_status cf_math_cpy_h2d(cf_math *dst, const void *host_data, cf_usize count)
{
  cf_usize elem_size = 0;
  cf_usize bytes = 0;

  if(dst == CF_NULL || host_data == CF_NULL) return CF_ERR_NULL;
  if(dst->handler == CF_NULL || dst->metadata == CF_NULL || dst->data == CF_NULL) return CF_ERR_STATE;
  if(count > dst->metadata->len) return CF_ERR_BOUNDS;

  elem_size = cf_math_dtype_size(dst->handler->storage.dtype);
  if(elem_size == 0) return CF_ERR_INVALID;
  if(count > (cf_usize)-1 / elem_size) return CF_ERR_OVERFLOW;

  bytes = count * elem_size;
  if(bytes > dst->byte_size) return CF_ERR_BOUNDS;

  return cf_math_copy_bytes(dst, dst->data, host_data, bytes);
}

cf_status cf_math_cpy_d2h(const cf_math *src, void *host_data, cf_usize count)
{
  cf_usize elem_size = 0;
  cf_usize bytes = 0;

  if(src == CF_NULL || host_data == CF_NULL) return CF_ERR_NULL;
  if(src->handler == CF_NULL || src->metadata == CF_NULL || src->data == CF_NULL) return CF_ERR_STATE;
  if(count > src->metadata->len) return CF_ERR_BOUNDS;

  elem_size = cf_math_dtype_size(src->handler->storage.dtype);
  if(elem_size == 0) return CF_ERR_INVALID;
  if(count > (cf_usize)-1 / elem_size) return CF_ERR_OVERFLOW;

  bytes = count * elem_size;
  if(bytes > src->byte_size) return CF_ERR_BOUNDS;

  return cf_math_copy_bytes(src, host_data, src->data, bytes);
}

cf_status cf_math_rebind(cf_math *x, cf_math_handle_t *handler, cf_math_metadata *metadata)
{
  cf_status status = CF_OK;

  if(x == CF_NULL || handler == CF_NULL || metadata == CF_NULL) return CF_ERR_NULL;

  status = cf_math_unbind(x);
  if(status != CF_OK) return status;

  return cf_math_bind(x, handler, metadata);
}

cf_status cf_math_unbind(cf_math *x)
{
  cf_status status = CF_OK;
  cf_bool released = CF_FALSE;

  if(x == CF_NULL) return CF_ERR_NULL;

  if(x->handler != CF_NULL && x->byte_size != 0)
  {
    status = cf_math_storage_release_slice(&x->handler->storage, x->byte_offset, x->byte_size, &released);
    if(status != CF_OK) return status;
  }

  x->data = CF_NULL;
  x->byte_offset = 0;
  x->byte_size = 0;
  x->metadata = CF_NULL;
  x->handler = CF_NULL;
  x->grad = CF_NULL;
  x->grad_fn = CF_NULL;
  x->grad_state = CF_MATH_GRAD_NONE;

  return CF_OK;
}
