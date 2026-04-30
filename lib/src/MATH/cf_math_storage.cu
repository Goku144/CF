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

#include "MATH/cf_math_storage.h"

#include <stdlib.h>
#include <string.h>

static cf_status cf_math_storage_validate(cf_math_device device, cf_math_mem_flags flags);

static cf_status cf_math_storage_backend_alloc(cf_math_cuda_context *ctx, cf_math_device device, cf_math_mem_flags flags, cf_usize bytes, void **ptr);

static cf_status cf_math_storage_backend_copy(cf_math_cuda_context *ctx, cf_math_mem_flags flags, void *dst, const void *src, cf_usize bytes);

static cf_status cf_math_storage_backend_free(cf_math_cuda_context *ctx, cf_math_device device, cf_math_mem_flags flags, void *ptr);

static cf_status cf_math_arena_add_free_block(cf_math_arena *arena, cf_usize offset, cf_usize size);

static void cf_math_arena_remove_free_block(cf_math_arena *arena, cf_usize index);

static cf_status cf_math_arena_add_active_block(cf_math_arena *arena, cf_usize offset, cf_usize size);

static cf_usize cf_math_arena_find_active_block(cf_math_arena *arena, cf_usize offset, cf_usize size);

static void cf_math_arena_remove_active_block(cf_math_arena *arena, cf_usize index);

static cf_status cf_math_storage_validate(cf_math_device device, cf_math_mem_flags flags)
{
  if((flags & CF_MATH_MEM_PINNED) != 0 && (flags & (CF_MATH_MEM_MANAGED | CF_MATH_MEM_POOLED)) != 0)
    return CF_ERR_INVALID;
  if((flags & CF_MATH_MEM_MANAGED) != 0 && (flags & CF_MATH_MEM_POOLED) != 0)
    return CF_ERR_INVALID;
  if((flags & CF_MATH_MEM_PEER_MAPPED) != 0)
    return CF_ERR_UNSUPPORTED;
  if((flags & CF_MATH_MEM_PINNED) != 0)
    return device == CF_MATH_DEVICE_CPU ? CF_OK : CF_ERR_INVALID;
  if(device == CF_MATH_DEVICE_CPU)
    return (flags & (CF_MATH_MEM_MANAGED | CF_MATH_MEM_POOLED)) == 0 ? CF_OK : CF_ERR_INVALID;
  return device == CF_MATH_DEVICE_CUDA ? CF_OK : CF_ERR_UNSUPPORTED;
}

static cf_status cf_math_storage_backend_alloc(cf_math_cuda_context *ctx, cf_math_device device, cf_math_mem_flags flags, cf_usize bytes, void **ptr)
{
  cf_status status = CF_OK;

  if(ptr == CF_NULL) return CF_ERR_NULL;
  *ptr = CF_NULL;
  if(bytes == 0) return CF_OK;

  status = cf_math_storage_validate(device, flags);
  if(status != CF_OK) return status;

  if(device == CF_MATH_DEVICE_CPU && (flags & CF_MATH_MEM_PINNED) == 0)
  {
    if((flags & CF_MATH_MEM_ALIGNED128) != 0)
    {
      cf_usize aligned_bytes = 0;
      if(bytes > (cf_usize)-1 - 127U) return CF_ERR_OVERFLOW;
      aligned_bytes = (bytes + 127U) & ~((cf_usize)127U);
      *ptr = aligned_alloc(128U, (size_t)aligned_bytes);
    }
    else
    {
      *ptr = malloc((size_t)bytes);
    }
    return *ptr != CF_NULL ? CF_OK : CF_ERR_OOM;
  }

#if defined(CF_CUDA_AVAILABLE)
  if(ctx == CF_NULL) return CF_ERR_NULL;
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
      CF_UNUSED(cudaMemAdvise(*ptr, bytes, cudaMemAdviseSetReadMostly, location));
    }
  }
  else if((flags & CF_MATH_MEM_POOLED) != 0)
  {
    if(cudaMallocAsync(ptr, bytes, ctx->stream) != cudaSuccess) return CF_ERR_CUDA_MEMORY;
    if(cudaStreamSynchronize(ctx->stream) != cudaSuccess)
    {
      CF_UNUSED(cudaFreeAsync(*ptr, ctx->stream));
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
    status = cf_math_storage_backend_free(ctx, device, flags, *ptr);
    *ptr = CF_NULL;
    return status == CF_OK ? CF_ERR_CUDA_MEMORY : status;
  }

  return CF_OK;
#else
  CF_UNUSED(ctx);
  CF_UNUSED(device);
  CF_UNUSED(flags);
  if(ptr == CF_NULL) return CF_ERR_NULL;
  *ptr = CF_NULL;
  return bytes == 0 ? CF_OK : CF_ERR_UNSUPPORTED;
#endif
}

static cf_status cf_math_storage_backend_copy(cf_math_cuda_context *ctx, cf_math_mem_flags flags, void *dst, const void *src, cf_usize bytes)
{
  if(bytes == 0) return CF_OK;
  if(dst == CF_NULL || src == CF_NULL) return CF_ERR_NULL;

  if(ctx == CF_NULL || (flags & CF_MATH_MEM_PINNED) != 0)
  {
    memcpy(dst, src, bytes);
    return CF_OK;
  }

#if defined(CF_CUDA_AVAILABLE)
  if((flags & CF_MATH_MEM_POOLED) != 0)
  {
    if(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, ctx->stream) != cudaSuccess) return CF_ERR_CUDA_COPY;
    return cudaStreamSynchronize(ctx->stream) == cudaSuccess ? CF_OK : CF_ERR_CUDA_SYNC;
  }

  return cudaMemcpy(dst, src, bytes, cudaMemcpyDefault) == cudaSuccess ? CF_OK : CF_ERR_CUDA_COPY;
#else
  CF_UNUSED(ctx);
  CF_UNUSED(flags);
  CF_UNUSED(dst);
  CF_UNUSED(src);
  return bytes == 0 ? CF_OK : CF_ERR_UNSUPPORTED;
#endif
}

static cf_status cf_math_storage_backend_free(cf_math_cuda_context *ctx, cf_math_device device, cf_math_mem_flags flags, void *ptr)
{
  if(ptr == CF_NULL) return CF_OK;

  if(device == CF_MATH_DEVICE_CPU && (flags & CF_MATH_MEM_PINNED) == 0)
  {
    free(ptr);
    return CF_OK;
  }

#if defined(CF_CUDA_AVAILABLE)
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
  CF_UNUSED(ctx);
  CF_UNUSED(device);
  CF_UNUSED(flags);
  if(ptr == CF_NULL) return CF_OK;
  return CF_ERR_UNSUPPORTED;
#endif
}

static cf_status cf_math_arena_add_active_block(cf_math_arena *arena, cf_usize offset, cf_usize size)
{
  cf_usize index = cf_math_arena_find_active_block(arena, offset, size);

  if(index != (cf_usize)-1)
  {
    if(arena->active_blocks[index].ref_count == (cf_usize)-1) return CF_ERR_OVERFLOW;
    arena->active_blocks[index].ref_count++;
    return CF_OK;
  }

  if(arena->active_count >= CF_MATH_MAX_ACTIVE_BLOCKS) return CF_ERR_BOUNDS;
  arena->active_blocks[arena->active_count].offset = offset;
  arena->active_blocks[arena->active_count].size = size;
  arena->active_blocks[arena->active_count].ref_count = 1;
  arena->active_count++;

  return CF_OK;
}

static cf_usize cf_math_arena_find_active_block(cf_math_arena *arena, cf_usize offset, cf_usize size)
{
  for(cf_usize i = 0; i < arena->active_count; ++i)
  {
    if(arena->active_blocks[i].offset == offset && arena->active_blocks[i].size == size)
      return i;
  }
  return (cf_usize)-1;
}

static void cf_math_arena_remove_active_block(cf_math_arena *arena, cf_usize index)
{
  for(cf_usize i = index + 1; i < arena->active_count; ++i)
    arena->active_blocks[i - 1] = arena->active_blocks[i];
  arena->active_count--;
}

static cf_status cf_math_arena_add_free_block(cf_math_arena *arena, cf_usize offset, cf_usize size)
{
  if(offset + size == arena->offset)
  {
    arena->offset = offset;
    for(cf_bool merged = CF_TRUE; merged == CF_TRUE;)
    {
      merged = CF_FALSE;
      for(cf_usize i = 0; i < arena->free_count; ++i)
      {
        if(arena->free_blocks[i].offset + arena->free_blocks[i].size == arena->offset)
        {
          arena->offset = arena->free_blocks[i].offset;
          cf_math_arena_remove_free_block(arena, i);
          merged = CF_TRUE;
          break;
        }
      }
    }
    return CF_OK;
  }

  for(cf_usize i = 0; i < arena->free_count; ++i)
  {
    cf_math_memory_block *block = &arena->free_blocks[i];
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

  if(arena->free_count >= CF_MATH_MAX_FREE_BLOCKS) return CF_ERR_BOUNDS;
  arena->free_blocks[arena->free_count].offset = offset;
  arena->free_blocks[arena->free_count].size = size;
  arena->free_blocks[arena->free_count].ref_count = 0;
  arena->free_count++;

  return CF_OK;
}

static void cf_math_arena_remove_free_block(cf_math_arena *arena, cf_usize index)
{
  for(cf_usize i = index + 1; i < arena->free_count; ++i)
    arena->free_blocks[i - 1] = arena->free_blocks[i];
  arena->free_count--;
}

cf_status cf_math_storage_release_slice(cf_math_storage *storage, cf_usize offset, cf_usize size, cf_bool *released)
{
  cf_math_arena *arena = CF_NULL;
  cf_usize index = 0;

  if(storage == CF_NULL || released == CF_NULL) return CF_ERR_NULL;
  *released = CF_FALSE;
  if(size == 0) return CF_OK;
  arena = &storage->arena;

  index = cf_math_arena_find_active_block(arena, offset, size);
  if(index == (cf_usize)-1) return CF_ERR_STATE;

  if(arena->active_blocks[index].ref_count > 1)
  {
    arena->active_blocks[index].ref_count--;
    return CF_OK;
  }

  cf_math_arena_remove_active_block(arena, index);
  *released = CF_TRUE;
  return cf_math_arena_add_free_block(arena, offset, size);
}

cf_status cf_math_cuda_context_init(cf_math_cuda_context *ctx, cf_usize bytes, int device_id)
{
  if(ctx == CF_NULL) return CF_ERR_NULL;
  if(device_id < 0) return CF_ERR_INVALID;

  memset(ctx, 0, sizeof(*ctx));
  ctx->device_id = device_id;

#if !defined(CF_CUDA_AVAILABLE)
  CF_UNUSED(bytes);
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

  if(bytes != 0)
  {
    void *ptr = CF_NULL;
    if(cudaMalloc(&ptr, bytes) != cudaSuccess) return CF_ERR_CUDA_MEMORY;

    if(ctx->cuda_workspace.ptr != CF_NULL)
    {
      cudaFree(ptr);
      return CF_ERR_CUDA_MEMORY;
    }

    ctx->cuda_workspace.ptr = ptr;
    ctx->cuda_workspace.size = bytes;
    if(bytes > ctx->cuda_workspace.high_water) ctx->cuda_workspace.high_water = bytes;
  }

  return CF_OK;
#endif
}

cf_status cf_math_cuda_context_reserve(cf_math_cuda_context *ctx, cf_usize bytes)
{
#if defined(CF_CUDA_AVAILABLE)
  void* new_ptr = nullptr;

  if(ctx == CF_NULL) return CF_ERR_NULL;
  if(bytes == 0 || ctx->cuda_workspace.size >= bytes) return CF_OK;
  if(cudaSetDevice(ctx->device_id) != cudaSuccess) return CF_ERR_CUDA_DEVICE;
  if(cudaMalloc(&new_ptr, bytes) != cudaSuccess) return CF_ERR_CUDA_MEMORY;
    
  if (ctx->cuda_workspace.ptr != CF_NULL)
  {
    cudaMemcpy(new_ptr, ctx->cuda_workspace.ptr, ctx->cuda_workspace.size < bytes ? ctx->cuda_workspace.size : bytes, cudaMemcpyDeviceToDevice);
    if(cudaFree(ctx->cuda_workspace.ptr) != cudaSuccess)
    {
      cudaFree(new_ptr);
      return CF_ERR_CUDA_MEMORY;
    }
  }
  else 
    return cf_math_cuda_context_init(ctx, bytes, 0);

  ctx->cuda_workspace.ptr = new_ptr;
  ctx->cuda_workspace.size = bytes;
  if(bytes > ctx->cuda_workspace.high_water) ctx->cuda_workspace.high_water = bytes;

  return CF_OK;
#else
  if(ctx == CF_NULL) return CF_ERR_NULL;
  return bytes == 0 ? CF_OK : CF_ERR_UNSUPPORTED;
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
  ctx->cuda_workspace.ptr = CF_NULL;
  ctx->cuda_workspace.size = 0;
  ctx->cuda_workspace.high_water = 0;

  return status;
#endif
}

cf_status cf_math_handle_init(cf_math_handle_t *handler, cf_math_cuda_context *ctx, cf_math_dtype dtype, cf_math_device device, cf_math_mem_flags flags, cf_math_handle_opt optimized_for, cf_usize capacity)
{
  cf_status status = CF_OK;

  if(handler == CF_NULL) return CF_ERR_NULL;
  if(device == CF_MATH_DEVICE_CUDA && ctx == CF_NULL) return CF_ERR_NULL;
  if(device == CF_MATH_DEVICE_CPU && (flags & CF_MATH_MEM_PINNED) != 0 && ctx == CF_NULL) return CF_ERR_NULL;
  status = cf_math_storage_validate(device, flags);
  if(status != CF_OK) return status;

  memset(handler, 0, sizeof(*handler));
  handler->optimized_for = optimized_for;
  handler->cuda_ctx = ctx;
  handler->storage.dtype = dtype;
  handler->storage.device = device;
  handler->storage.allocator.backend = CF_NULL;
  handler->storage.allocator.mem_flag = flags;

  if(capacity != 0)
  {
    status = cf_math_handle_reserve(handler, capacity);
    if(status != CF_OK) memset(handler, 0, sizeof(*handler));
  }
  return status;
}

cf_status cf_math_handle_alloc(cf_math_handle_t *handler, cf_usize bytes, void **ptr)
{
  cf_math_arena *arena = CF_NULL;
  cf_usize offset = 0;
  cf_status status = CF_OK;

  if(handler == CF_NULL || ptr == CF_NULL) return CF_ERR_NULL;
  if(bytes == 0)
  {
    *ptr = CF_NULL;
    return CF_OK;
  }

  arena = &handler->storage.arena;

  for(cf_usize i = 0; i < arena->free_count; ++i)
  {
    cf_math_memory_block block = arena->free_blocks[i];
    cf_usize aligned = block.offset;
    cf_usize prefix = 0;
    cf_usize suffix_offset = 0;
    cf_usize suffix_size = 0;

    if(arena->active_count >= CF_MATH_MAX_ACTIVE_BLOCKS) return CF_ERR_BOUNDS;

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
      cf_math_arena_remove_free_block(arena, i);
    }
    else if(prefix == 0)
    {
      arena->free_blocks[i].offset = suffix_offset;
      arena->free_blocks[i].size = suffix_size;
    }
    else
    {
      if(suffix_size != 0 && arena->free_count >= CF_MATH_MAX_FREE_BLOCKS) continue;
      arena->free_blocks[i].size = prefix;
      if(suffix_size != 0)
      {
        arena->free_blocks[arena->free_count].offset = suffix_offset;
        arena->free_blocks[arena->free_count].size = suffix_size;
        arena->free_blocks[arena->free_count].ref_count = 0;
        arena->free_count++;
      }
    }

    status = cf_math_arena_add_active_block(arena, aligned, bytes);
    if(status != CF_OK) return status;

    *ptr = (void *)((cf_u8 *)handler->storage.allocator.backend + aligned);
    return CF_OK;
  }

  offset = arena->offset;
  if((handler->storage.allocator.mem_flag & CF_MATH_MEM_ALIGNED128) != 0)
  {
    if(offset > (cf_usize)-1 - 127U) return CF_ERR_OVERFLOW;
    offset = (offset + 127U) & ~((cf_usize)127U);
  }

  if(offset > (cf_usize)-1 - bytes) return CF_ERR_OVERFLOW;
  if(offset + bytes > arena->capacity) return CF_ERR_BOUNDS;

  status = cf_math_arena_add_active_block(arena, offset, bytes);
  if(status != CF_OK) return status;

  *ptr = (void *)((cf_u8 *)handler->storage.allocator.backend + offset);
  arena->offset = offset + bytes;

  return CF_OK;
}

cf_status cf_math_handle_reserve(cf_math_handle_t *handler, cf_usize bytes)
{
  void *ptr = CF_NULL;
  cf_usize copy_bytes = 0;
  cf_status status = CF_OK;

  if(handler == CF_NULL) return CF_ERR_NULL;
  if(handler->storage.device == CF_MATH_DEVICE_CUDA && handler->cuda_ctx == CF_NULL) return CF_ERR_STATE;
  if(handler->storage.device == CF_MATH_DEVICE_CPU && (handler->storage.allocator.mem_flag & CF_MATH_MEM_PINNED) != 0 && handler->cuda_ctx == CF_NULL) return CF_ERR_STATE;
  if(bytes == 0 || handler->storage.arena.capacity >= bytes) return CF_OK;

  status = cf_math_storage_backend_alloc(handler->cuda_ctx, handler->storage.device, handler->storage.allocator.mem_flag, bytes, &ptr);
  if(status != CF_OK) return status;

  if(handler->storage.allocator.backend != CF_NULL)
  {
    copy_bytes = handler->storage.arena.offset < handler->storage.arena.capacity ? handler->storage.arena.offset : handler->storage.arena.capacity;
    status = cf_math_storage_backend_copy(handler->cuda_ctx, handler->storage.allocator.mem_flag, ptr, handler->storage.allocator.backend, copy_bytes);
    if(status != CF_OK)
    {
      CF_UNUSED(cf_math_storage_backend_free(handler->cuda_ctx, handler->storage.device, handler->storage.allocator.mem_flag, ptr));
      return status;
    }
    status = cf_math_storage_backend_free(handler->cuda_ctx, handler->storage.device, handler->storage.allocator.mem_flag, handler->storage.allocator.backend);
    if(status != CF_OK)
    {
      CF_UNUSED(cf_math_storage_backend_free(handler->cuda_ctx, handler->storage.device, handler->storage.allocator.mem_flag, ptr));
      return status;
    }
  }

  handler->storage.allocator.backend = ptr;
  handler->storage.arena.capacity = bytes;

  return CF_OK;
}

void cf_math_handle_reset(cf_math_handle_t *handler)
{
  if(handler == CF_NULL) return;
  handler->storage.arena.offset = 0;
  handler->storage.arena.free_count = 0;
  handler->storage.arena.active_count = 0;
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

  if(handler->storage.allocator.backend != CF_NULL)
  {
    cf_status free_status = cf_math_storage_backend_free(handler->cuda_ctx, handler->storage.device, handler->storage.allocator.mem_flag, handler->storage.allocator.backend);
    if(free_status != CF_OK && status == CF_OK) status = free_status;
  }

  memset(handler, 0, sizeof(*handler));
  return status;
}
