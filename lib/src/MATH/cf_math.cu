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
#include "RUNTIME/cf_random.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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

static cf_status cf_math_cuda_storage_validate(cf_math_device device, cf_math_mem_flags flags)
{
  if((flags & CF_MATH_MEM_PINNED) != 0 && (flags & (CF_MATH_MEM_MANAGED | CF_MATH_MEM_POOLED)) != 0)
    return CF_ERR_INVALID;
  if((flags & CF_MATH_MEM_MANAGED) != 0 && (flags & CF_MATH_MEM_POOLED) != 0)
    return CF_ERR_INVALID;
  if((flags & CF_MATH_MEM_PEER_MAPPED) != 0)
    return CF_ERR_UNSUPPORTED;
  if((flags & CF_MATH_MEM_PINNED) != 0)
    return device == CF_MATH_DEVICE_CPU ? CF_OK : CF_ERR_INVALID;
  return device == CF_MATH_DEVICE_CUDA ? CF_OK : CF_ERR_UNSUPPORTED;
}

static cf_status cf_math_cuda_storage_free(cf_math_cuda_context *ctx, cf_math_device device, cf_math_mem_flags flags, void *ptr)
{
#if defined(CF_CUDA_AVAILABLE)
  if(ptr == CF_NULL) return CF_OK;
  if(ctx == CF_NULL) return CF_ERR_NULL;
  if(device == CF_MATH_DEVICE_CUDA && cudaSetDevice(ctx->device_id) != cudaSuccess) return CF_ERR_CUDA_DEVICE;

  if((flags & CF_MATH_MEM_PINNED) != 0)
    return cudaFreeHost(ptr) == cudaSuccess ? CF_OK : CF_ERR_CUDA_MEMORY;

  if((flags & CF_MATH_MEM_POOLED) != 0)
  {
    if(cudaFreeAsync(ptr, ctx->stream) != cudaSuccess) return CF_ERR_CUDA_MEMORY;
    return cudaStreamSynchronize(ctx->stream) == cudaSuccess ? CF_OK : CF_ERR_CUDA_SYNC;
  }

  return cudaFree(ptr) == cudaSuccess ? CF_OK : CF_ERR_CUDA_MEMORY;
#else
  (void)ctx;
  (void)device;
  (void)flags;
  if(ptr == CF_NULL) return CF_OK;
  return CF_ERR_UNSUPPORTED;
#endif
}

static cf_status cf_math_cuda_storage_alloc(cf_math_cuda_context *ctx, cf_math_device device, cf_math_mem_flags flags, cf_usize bytes, void **ptr)
{
#if defined(CF_CUDA_AVAILABLE)
  cf_status status = CF_OK;

  if(ctx == CF_NULL || ptr == CF_NULL) return CF_ERR_NULL;
  *ptr = CF_NULL;
  if(bytes == 0) return CF_OK;

  status = cf_math_cuda_storage_validate(device, flags);
  if(status != CF_OK) return status;
  if(cudaSetDevice(ctx->device_id) != cudaSuccess) return CF_ERR_CUDA_DEVICE;

  if((flags & CF_MATH_MEM_PINNED) != 0)
  {
    if(cudaHostAlloc(ptr, bytes, cudaHostAllocDefault) != cudaSuccess) return CF_ERR_CUDA_MEMORY;
  }
  else if((flags & CF_MATH_MEM_MANAGED) != 0)
  {
    if(cudaMallocManaged(ptr, bytes, cudaMemAttachGlobal) != cudaSuccess) return CF_ERR_CUDA_MEMORY;
    if((flags & CF_MATH_MEM_READ_ONLY) != 0)
    {
      struct cudaMemLocation location = {cudaMemLocationTypeDevice, ctx->device_id};
      (void)cudaMemAdvise(*ptr, bytes, cudaMemAdviseSetReadMostly, location);
    }
  }
  else if((flags & CF_MATH_MEM_POOLED) != 0)
  {
    if(cudaMallocAsync(ptr, bytes, ctx->stream) != cudaSuccess) return CF_ERR_CUDA_MEMORY;
    if(cudaStreamSynchronize(ctx->stream) != cudaSuccess)
    {
      (void)cudaFreeAsync(*ptr, ctx->stream);
      *ptr = CF_NULL;
      return CF_ERR_CUDA_SYNC;
    }
  }
  else
  {
    if(cudaMalloc(ptr, bytes) != cudaSuccess) return CF_ERR_CUDA_MEMORY;
  }

  if((flags & CF_MATH_MEM_ALIGNED128) != 0 && (((cf_uptr)*ptr) & 127U) != 0)
  {
    status = cf_math_cuda_storage_free(ctx, device, flags, *ptr);
    *ptr = CF_NULL;
    return status == CF_OK ? CF_ERR_CUDA_MEMORY : status;
  }

  return CF_OK;
#else
  (void)ctx;
  (void)device;
  (void)flags;
  if(ptr == CF_NULL) return CF_ERR_NULL;
  *ptr = CF_NULL;
  return bytes == 0 ? CF_OK : CF_ERR_UNSUPPORTED;
#endif
}

static cf_status cf_math_cuda_storage_copy(cf_math_cuda_context *ctx, cf_math_mem_flags flags, void *dst, const void *src, cf_usize bytes)
{
#if defined(CF_CUDA_AVAILABLE)
  if(bytes == 0) return CF_OK;
  if(ctx == CF_NULL || dst == CF_NULL || src == CF_NULL) return CF_ERR_NULL;

  if((flags & CF_MATH_MEM_PINNED) != 0)
  {
    memcpy(dst, src, bytes);
    return CF_OK;
  }

  if((flags & CF_MATH_MEM_POOLED) != 0)
  {
    if(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, ctx->stream) != cudaSuccess) return CF_ERR_CUDA_COPY;
    return cudaStreamSynchronize(ctx->stream) == cudaSuccess ? CF_OK : CF_ERR_CUDA_SYNC;
  }

  return cudaMemcpy(dst, src, bytes, cudaMemcpyDefault) == cudaSuccess ? CF_OK : CF_ERR_CUDA_COPY;
#else
  (void)ctx;
  (void)flags;
  (void)dst;
  (void)src;
  return bytes == 0 ? CF_OK : CF_ERR_UNSUPPORTED;
#endif
}

static void cf_math_storage_remove_free_block(cf_math_storage *storage, cf_usize index)
{
  if(storage == CF_NULL || index >= storage->free_count) return;
  for(cf_usize i = index + 1; i < storage->free_count; ++i)
    storage->free_blocks[i - 1] = storage->free_blocks[i];
  storage->free_count--;
}

static void cf_math_storage_remove_active_block(cf_math_storage *storage, cf_usize index)
{
  if(storage == CF_NULL || index >= storage->active_count) return;
  for(cf_usize i = index + 1; i < storage->active_count; ++i)
    storage->active_blocks[i - 1] = storage->active_blocks[i];
  storage->active_count--;
}

static cf_usize cf_math_storage_find_active_block(cf_math_storage *storage, cf_usize offset, cf_usize size)
{
  if(storage == CF_NULL) return (cf_usize)-1;
  for(cf_usize i = 0; i < storage->active_count; ++i)
  {
    if(storage->active_blocks[i].offset == offset && storage->active_blocks[i].size == size)
      return i;
  }
  return (cf_usize)-1;
}

static cf_status cf_math_storage_add_active_block(cf_math_storage *storage, cf_usize offset, cf_usize size)
{
  cf_usize index = 0;

  if(storage == CF_NULL) return CF_ERR_NULL;
  if(size == 0) return CF_OK;
  if(offset > storage->capacity || size > storage->capacity - offset) return CF_ERR_BOUNDS;

  index = cf_math_storage_find_active_block(storage, offset, size);
  if(index != (cf_usize)-1)
  {
    if(storage->active_blocks[index].ref_count == (cf_usize)-1) return CF_ERR_OVERFLOW;
    storage->active_blocks[index].ref_count++;
    return CF_OK;
  }

  if(storage->active_count >= CF_MATH_MAX_ACTIVE_BLOCKS) return CF_ERR_BOUNDS;
  storage->active_blocks[storage->active_count].offset = offset;
  storage->active_blocks[storage->active_count].size = size;
  storage->active_blocks[storage->active_count].ref_count = 1;
  storage->active_count++;

  return CF_OK;
}

static cf_status cf_math_storage_add_free_block(cf_math_storage *storage, cf_usize offset, cf_usize size)
{
  if(storage == CF_NULL) return CF_ERR_NULL;
  if(size == 0) return CF_OK;
  if(offset > storage->capacity || size > storage->capacity - offset) return CF_ERR_BOUNDS;

  if(offset + size == storage->offset)
  {
    storage->offset = offset;
    for(cf_bool merged = CF_TRUE; merged == CF_TRUE;)
    {
      merged = CF_FALSE;
      for(cf_usize i = 0; i < storage->free_count; ++i)
      {
        if(storage->free_blocks[i].offset + storage->free_blocks[i].size == storage->offset)
        {
          storage->offset = storage->free_blocks[i].offset;
          cf_math_storage_remove_free_block(storage, i);
          merged = CF_TRUE;
          break;
        }
      }
    }
    return CF_OK;
  }

  for(cf_usize i = 0; i < storage->free_count; ++i)
  {
    cf_math_memory_block *block = &storage->free_blocks[i];
    if(offset + size == block->offset)
    {
      block->offset = offset;
      block->size += size;
      return CF_OK;
    }
    if(block->offset + block->size == offset)
    {
      block->size += size;
      return CF_OK;
    }
  }

  if(storage->free_count >= CF_MATH_MAX_FREE_BLOCKS) return CF_ERR_BOUNDS;
  storage->free_blocks[storage->free_count].offset = offset;
  storage->free_blocks[storage->free_count].size = size;
  storage->free_blocks[storage->free_count].ref_count = 0;
  storage->free_count++;

  return CF_OK;
}

static cf_status cf_math_storage_release_active_block(cf_math_storage *storage, cf_usize offset, cf_usize size, cf_bool *released)
{
  cf_usize index = 0;

  if(storage == CF_NULL || released == CF_NULL) return CF_ERR_NULL;
  *released = CF_FALSE;
  if(size == 0) return CF_OK;

  index = cf_math_storage_find_active_block(storage, offset, size);
  if(index == (cf_usize)-1) return CF_ERR_STATE;

  if(storage->active_blocks[index].ref_count > 1)
  {
    storage->active_blocks[index].ref_count--;
    return CF_OK;
  }

  cf_math_storage_remove_active_block(storage, index);
  *released = CF_TRUE;
  return cf_math_storage_add_free_block(storage, offset, size);
}

cf_status cf_math_metadata_init
(
  cf_math_metadata *metadata,
  cf_usize dim[CF_MATH_MAX_RANK],
  cf_usize rank,
  cf_math_shape shape,
  cf_math_layout layout
)
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

cf_status cf_math_cuda_context_init(cf_math_cuda_context *ctx, int device_id)
{
  if(ctx == CF_NULL) return CF_ERR_NULL;
  if(device_id < 0) return CF_ERR_INVALID;

  memset(ctx, 0, sizeof(*ctx));
  ctx->device_id = device_id;

#if !defined(CF_CUDA_AVAILABLE)
  return CF_ERR_UNSUPPORTED;
#else
  if(cudaSetDevice(device_id) != cudaSuccess)
  {
    memset(ctx, 0, sizeof(*ctx));
    return CF_ERR_CUDA_DEVICE;
  }

  if(cudaStreamCreate(&ctx->stream) != cudaSuccess)
  {
    cf_math_cuda_context_destroy(ctx);
    return CF_ERR_CUDA_RUNTIME;
  }

  if(cublasCreate(&ctx->cublas) != CUBLAS_STATUS_SUCCESS)
  {
    cf_math_cuda_context_destroy(ctx);
    return CF_ERR_CUDA;
  }
  if(cublasSetStream(ctx->cublas, ctx->stream) != CUBLAS_STATUS_SUCCESS)
  {
    cf_math_cuda_context_destroy(ctx);
    return CF_ERR_CUDA;
  }

  if(cublasLtCreate(&ctx->cublasLt) != CUBLAS_STATUS_SUCCESS)
  {
    cf_math_cuda_context_destroy(ctx);
    return CF_ERR_CUDA;
  }

  if(cudnnCreate(&ctx->cudnn) != CUDNN_STATUS_SUCCESS)
  {
    cf_math_cuda_context_destroy(ctx);
    return CF_ERR_CUDA;
  }
  if(cudnnSetStream(ctx->cudnn, ctx->stream) != CUDNN_STATUS_SUCCESS)
  {
    cf_math_cuda_context_destroy(ctx);
    return CF_ERR_CUDA;
  }

  if(cusparseCreate(&ctx->cusparse) != CUSPARSE_STATUS_SUCCESS)
  {
    cf_math_cuda_context_destroy(ctx);
    return CF_ERR_CUDA;
  }
  if(cusparseSetStream(ctx->cusparse, ctx->stream) != CUSPARSE_STATUS_SUCCESS)
  {
    cf_math_cuda_context_destroy(ctx);
    return CF_ERR_CUDA;
  }

  if(cusolverDnCreate(&ctx->cusolverDn) != CUSOLVER_STATUS_SUCCESS)
  {
    cf_math_cuda_context_destroy(ctx);
    return CF_ERR_CUDA;
  }
  if(cusolverDnSetStream(ctx->cusolverDn, ctx->stream) != CUSOLVER_STATUS_SUCCESS)
  {
    cf_math_cuda_context_destroy(ctx);
    return CF_ERR_CUDA;
  }

  if(curandCreateGenerator(&ctx->curand, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS)
  {
    cf_math_cuda_context_destroy(ctx);
    return CF_ERR_CUDA;
  }
  if(curandSetStream(ctx->curand, ctx->stream) != CURAND_STATUS_SUCCESS)
  {
    cf_math_cuda_context_destroy(ctx);
    return CF_ERR_CUDA;
  }

  return CF_OK;
#endif
}

cf_status cf_math_cuda_context_destroy(cf_math_cuda_context *ctx)
{
  if(ctx == CF_NULL) return CF_ERR_NULL;

#if !defined(CF_CUDA_AVAILABLE)
  memset(ctx, 0, sizeof(*ctx));
  return CF_OK;
#else
  cf_status status = CF_OK;

  if(ctx->device_id >= 0 && cudaSetDevice(ctx->device_id) != cudaSuccess)
    status = CF_ERR_CUDA_DEVICE;

  if(ctx->cuda_workspace.ptr != CF_NULL)
  {
    if(cudaFree(ctx->cuda_workspace.ptr) != cudaSuccess && status == CF_OK)
      status = CF_ERR_CUDA_MEMORY;
    ctx->cuda_workspace.ptr = CF_NULL;
  }

  if(ctx->curand != CF_NULL)
  {
    if(curandDestroyGenerator(ctx->curand) != CURAND_STATUS_SUCCESS && status == CF_OK)
      status = CF_ERR_CUDA;
    ctx->curand = CF_NULL;
  }

  if(ctx->cusolverDn != CF_NULL)
  {
    if(cusolverDnDestroy(ctx->cusolverDn) != CUSOLVER_STATUS_SUCCESS && status == CF_OK)
      status = CF_ERR_CUDA;
    ctx->cusolverDn = CF_NULL;
  }

  if(ctx->cusparse != CF_NULL)
  {
    if(cusparseDestroy(ctx->cusparse) != CUSPARSE_STATUS_SUCCESS && status == CF_OK)
      status = CF_ERR_CUDA;
    ctx->cusparse = CF_NULL;
  }

  if(ctx->cudnn != CF_NULL)
  {
    if(cudnnDestroy(ctx->cudnn) != CUDNN_STATUS_SUCCESS && status == CF_OK)
      status = CF_ERR_CUDA;
    ctx->cudnn = CF_NULL;
  }

  if(ctx->cublasLt != CF_NULL)
  {
    if(cublasLtDestroy(ctx->cublasLt) != CUBLAS_STATUS_SUCCESS && status == CF_OK)
      status = CF_ERR_CUDA;
    ctx->cublasLt = CF_NULL;
  }

  if(ctx->cublas != CF_NULL)
  {
    if(cublasDestroy(ctx->cublas) != CUBLAS_STATUS_SUCCESS && status == CF_OK)
      status = CF_ERR_CUDA;
    ctx->cublas = CF_NULL;
  }

  if(ctx->stream != CF_NULL)
  {
    if(cudaStreamDestroy(ctx->stream) != cudaSuccess && status == CF_OK)
      status = CF_ERR_CUDA_RUNTIME;
    ctx->stream = CF_NULL;
  }

  ctx->device_id = 0;
  ctx->cuda_workspace.size = 0;
  ctx->cuda_workspace.max_size = 0;

  return status;
#endif
}

cf_status cf_math_cuda_workspace_reserve(cf_math_cuda_context *ctx, cf_usize bytes)
{
#if defined(CF_CUDA_AVAILABLE)
  void *ptr = CF_NULL;

  if(ctx == CF_NULL) return CF_ERR_NULL;
  if(bytes == 0 || ctx->cuda_workspace.size >= bytes) return CF_OK;
  if(cudaSetDevice(ctx->device_id) != cudaSuccess) return CF_ERR_CUDA_DEVICE;
  if(cudaMalloc(&ptr, bytes) != cudaSuccess) return CF_ERR_CUDA_MEMORY;

  if(ctx->cuda_workspace.ptr != CF_NULL)
  {
    if(cudaFree(ctx->cuda_workspace.ptr) != cudaSuccess)
    {
      cudaFree(ptr);
      return CF_ERR_CUDA_MEMORY;
    }
  }

  ctx->cuda_workspace.ptr = ptr;
  ctx->cuda_workspace.size = bytes;
  if(bytes > ctx->cuda_workspace.max_size) ctx->cuda_workspace.max_size = bytes;

  return CF_OK;
#else
  if(ctx == CF_NULL) return CF_ERR_NULL;
  return bytes == 0 ? CF_OK : CF_ERR_UNSUPPORTED;
#endif
}

cf_status cf_math_handle_init
(
  cf_math_handle_t *handler,
  cf_math_cuda_context *ctx,
  cf_math_dtype dtype,
  cf_math_device device,
  cf_math_mem_flags flags,
  cf_math_handle_opt optimized_for,
  cf_usize capacity
)
{
  cf_status status = CF_OK;

  if(handler == CF_NULL || ctx == CF_NULL) return CF_ERR_NULL;
  status = cf_math_cuda_storage_validate(device, flags);
  if(status != CF_OK) return status;

  memset(handler, 0, sizeof(*handler));
  handler->optimized_for = optimized_for;
  handler->cuda_ctx = ctx;
  handler->storage.dtype = dtype;
  handler->storage.device = device;
  handler->storage.allocator.backend = ctx;
  handler->storage.allocator.mem_flag = flags;

  if(capacity != 0)
  {
    status = cf_math_handle_reserve(handler, capacity);
    if(status != CF_OK) memset(handler, 0, sizeof(*handler));
  }

  return status;
}

cf_status cf_math_handle_reserve(cf_math_handle_t *handler, cf_usize bytes)
{
  void *ptr = CF_NULL;
  cf_usize copy_bytes = 0;
  cf_status status = CF_OK;

  if(handler == CF_NULL) return CF_ERR_NULL;
  if(handler->cuda_ctx == CF_NULL) return CF_ERR_STATE;
  if(bytes == 0 || handler->storage.capacity >= bytes) return CF_OK;

  status = cf_math_cuda_storage_alloc(handler->cuda_ctx, handler->storage.device, handler->storage.allocator.mem_flag, bytes, &ptr);
  if(status != CF_OK) return status;

  if(handler->storage.data_ptr != CF_NULL)
  {
    copy_bytes = handler->storage.offset < handler->storage.capacity ? handler->storage.offset : handler->storage.capacity;
    status = cf_math_cuda_storage_copy(handler->cuda_ctx, handler->storage.allocator.mem_flag, ptr, handler->storage.data_ptr, copy_bytes);
    if(status != CF_OK)
    {
      (void)cf_math_cuda_storage_free(handler->cuda_ctx, handler->storage.device, handler->storage.allocator.mem_flag, ptr);
      return status;
    }
    status = cf_math_cuda_storage_free(handler->cuda_ctx, handler->storage.device, handler->storage.allocator.mem_flag, handler->storage.data_ptr);
    if(status != CF_OK)
    {
      (void)cf_math_cuda_storage_free(handler->cuda_ctx, handler->storage.device, handler->storage.allocator.mem_flag, ptr);
      return status;
    }
  }

  handler->storage.data_ptr = ptr;
  handler->storage.capacity = bytes;
  handler->storage.allocator.total_reserved = bytes;

  return CF_OK;
}

cf_status cf_math_handle_alloc(cf_math_handle_t *handler, cf_usize bytes, void **ptr)
{
  cf_usize offset = 0;
  cf_status status = CF_OK;

  if(handler == CF_NULL || ptr == CF_NULL) return CF_ERR_NULL;
  if(bytes == 0)
  {
    *ptr = CF_NULL;
    return CF_OK;
  }

  for(cf_usize i = 0; i < handler->storage.free_count; ++i)
  {
    cf_math_memory_block block = handler->storage.free_blocks[i];
    cf_usize aligned = block.offset;
    cf_usize prefix = 0;
    cf_usize suffix_offset = 0;
    cf_usize suffix_size = 0;

    if(handler->storage.active_count >= CF_MATH_MAX_ACTIVE_BLOCKS) return CF_ERR_BOUNDS;

    if((handler->storage.allocator.mem_flag & CF_MATH_MEM_ALIGNED128) != 0)
    {
      if(aligned > (cf_usize)-1 - 127U) continue;
      aligned = (aligned + 127U) & ~((cf_usize)127U);
    }
    if(aligned < block.offset || aligned > block.offset + block.size) continue;

    prefix = aligned - block.offset;
    if(prefix > block.size || block.size - prefix < bytes) continue;

    suffix_offset = aligned + bytes;
    suffix_size = block.offset + block.size - suffix_offset;

    if(prefix == 0 && suffix_size == 0)
    {
      cf_math_storage_remove_free_block(&handler->storage, i);
    }
    else if(prefix == 0)
    {
      handler->storage.free_blocks[i].offset = suffix_offset;
      handler->storage.free_blocks[i].size = suffix_size;
    }
    else
    {
      if(suffix_size != 0 && handler->storage.free_count >= CF_MATH_MAX_FREE_BLOCKS) continue;
      handler->storage.free_blocks[i].size = prefix;
      if(suffix_size != 0)
      {
        handler->storage.free_blocks[handler->storage.free_count].offset = suffix_offset;
        handler->storage.free_blocks[handler->storage.free_count].size = suffix_size;
        handler->storage.free_blocks[handler->storage.free_count].ref_count = 0;
        handler->storage.free_count++;
      }
    }

    status = cf_math_storage_add_active_block(&handler->storage, aligned, bytes);
    if(status != CF_OK) return status;

    *ptr = (void *)((cf_u8 *)handler->storage.data_ptr + aligned);
    handler->storage.allocator.total_allocated += bytes;
    handler->storage.allocator.allocation_count++;
    return CF_OK;
  }

  offset = handler->storage.offset;
  if((handler->storage.allocator.mem_flag & CF_MATH_MEM_ALIGNED128) != 0)
  {
    if(offset > (cf_usize)-1 - 127U) return CF_ERR_OVERFLOW;
    offset = (offset + 127U) & ~((cf_usize)127U);
  }

  if(offset > (cf_usize)-1 - bytes) return CF_ERR_OVERFLOW;
  if(offset + bytes > handler->storage.capacity) return CF_ERR_BOUNDS;

  status = cf_math_storage_add_active_block(&handler->storage, offset, bytes);
  if(status != CF_OK) return status;

  *ptr = (void *)((cf_u8 *)handler->storage.data_ptr + offset);
  handler->storage.offset = offset + bytes;
  handler->storage.allocator.total_allocated += bytes;
  handler->storage.allocator.allocation_count++;

  return CF_OK;
}

void cf_math_handle_reset(cf_math_handle_t *handler)
{
  if(handler == CF_NULL) return;
  handler->storage.offset = 0;
  handler->storage.free_count = 0;
  handler->storage.active_count = 0;
  handler->storage.allocator.total_allocated = 0;
  handler->storage.allocator.allocation_count = 0;
}

cf_status cf_math_handle_destroy(cf_math_handle_t *handler)
{
  cf_status status = CF_OK;

  if(handler == CF_NULL) return CF_ERR_NULL;

#if defined(CF_CUDA_AVAILABLE)
  if(handler->storage.device == CF_MATH_DEVICE_CUDA && handler->cuda_ctx == CF_NULL)
    status = CF_ERR_STATE;
  if(status == CF_OK && handler->storage.device == CF_MATH_DEVICE_CUDA && cudaSetDevice(handler->cuda_ctx->device_id) != cudaSuccess)
    status = CF_ERR_CUDA_DEVICE;

  if(handler->desc_cache.has_lt_layout == CF_TRUE)
  {
    if(cublasLtMatrixLayoutDestroy(handler->desc_cache.lt_layout) != CUBLAS_STATUS_SUCCESS && status == CF_OK)
      status = CF_ERR_CUDA;
  }

  if(handler->desc_cache.has_cudnn_tensor == CF_TRUE)
  {
    if(cudnnDestroyTensorDescriptor(handler->desc_cache.cudnn_tensor) != CUDNN_STATUS_SUCCESS && status == CF_OK)
      status = CF_ERR_CUDA;
  }
#else
  if(handler->desc_cache.has_lt_layout == CF_TRUE || handler->desc_cache.has_cudnn_tensor == CF_TRUE)
    status = CF_ERR_UNSUPPORTED;
#endif

  if(handler->storage.data_ptr != CF_NULL)
  {
    cf_status free_status = cf_math_cuda_storage_free(handler->cuda_ctx, handler->storage.device, handler->storage.allocator.mem_flag, handler->storage.data_ptr);
    if(free_status != CF_OK && status == CF_OK) status = free_status;
  }

  memset(handler, 0, sizeof(*handler));
  return status;
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
  offset = handler->storage.offset;
  if((handler->storage.allocator.mem_flag & CF_MATH_MEM_ALIGNED128) != 0)
  {
    if(offset > (cf_usize)-1 - 127U) return CF_ERR_OVERFLOW;
    offset = (offset + 127U) & ~((cf_usize)127U);
  }
  if(offset > (cf_usize)-1 - bytes) return CF_ERR_OVERFLOW;

  required = offset + bytes;
  if(required > handler->storage.capacity)
  {
    status = cf_math_handle_reserve(handler, required);
    if(status != CF_OK) return status;
  }

  status = cf_math_handle_alloc(handler, bytes, &data);
  if(status != CF_OK) return status;

  x->data = data;
  x->byte_offset = data != CF_NULL ? (cf_usize)((cf_u8 *)data - (cf_u8 *)handler->storage.data_ptr) : 0;
  x->byte_size = bytes;
  x->metadata = metadata;
  x->handler = handler;
  x->grad = CF_NULL;
  x->grad_fn = CF_NULL;
  x->grad_state = CF_MATH_GRAD_NONE;

  return CF_OK;
}

cf_status cf_math_unbind(cf_math *x)
{
  cf_status status = CF_OK;
  cf_bool released = CF_FALSE;

  if(x == CF_NULL) return CF_ERR_NULL;

  if(x->handler != CF_NULL && x->byte_size != 0)
  {
    status = cf_math_storage_release_active_block(&x->handler->storage, x->byte_offset, x->byte_size, &released);
    if(status != CF_OK) return status;
    if(released == CF_TRUE && x->handler->storage.allocator.total_allocated >= x->byte_size)
      x->handler->storage.allocator.total_allocated -= x->byte_size;
    else if(released == CF_TRUE)
      x->handler->storage.allocator.total_allocated = 0;
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

cf_status cf_math_rebind(cf_math *x, cf_math_handle_t *handler, cf_math_metadata *metadata)
{
  cf_status status = CF_OK;

  if(x == CF_NULL || handler == CF_NULL || metadata == CF_NULL) return CF_ERR_NULL;

  status = cf_math_unbind(x);
  if(status != CF_OK) return status;

  return cf_math_bind(x, handler, metadata);
}
