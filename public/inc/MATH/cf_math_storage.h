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

#if !defined(CF_MATH_STORAGE_H)
#define CF_MATH_STORAGE_H

#include "RUNTIME/cf_status.h"
#include "RUNTIME/cf_types.h"

#if defined(CF_CUDA_AVAILABLE)
  #include <cuda_runtime.h>
  #include <cublas_v2.h>
  #include <cublasLt.h>
  #include <cudnn.h>
  #include <cusparse_v2.h>
  #include <cusolverDn.h>
  #include <curand.h>
#else
  typedef void *cudaStream_t;
  typedef void *cudaMemPool_t;
  typedef void *cublasHandle_t;
  typedef void *cublasLtHandle_t;
  typedef void *cudnnHandle_t;
  typedef void *cusparseHandle_t;
  typedef void *cusolverDnHandle_t;
  typedef void *curandGenerator_t;

  typedef void *cudnnTensorDescriptor_t;
  typedef void *cudnnFilterDescriptor_t;
  typedef void *cudnnConvolutionDescriptor_t;
  typedef void *cudnnRNNDataDescriptor_t;
  typedef void *cublasLtMatrixLayout_t;
#endif

#define CF_MATH_MAX_FREE_BLOCKS 64
#define CF_MATH_MAX_ACTIVE_BLOCKS 256

typedef struct cf_math_cuda_workspace cf_math_cuda_workspace;
typedef struct cf_math_cuda_context cf_math_cuda_context;
typedef struct cf_math_allocator cf_math_allocator;
typedef struct cf_math_desc_cache cf_math_desc_cache;
typedef struct cf_math_memory_block cf_math_memory_block;
typedef struct cf_math_arena cf_math_arena;
typedef struct cf_math_storage cf_math_storage;
typedef struct cf_math_handle cf_math_handle_t;

typedef enum cf_math_dtype
{
  CF_MATH_DTYPE_BOOL = 0,
  CF_MATH_DTYPE_I8,
  CF_MATH_DTYPE_U8,
  CF_MATH_DTYPE_I32,
  CF_MATH_DTYPE_F64,
  CF_MATH_DTYPE_F32,
  CF_MATH_DTYPE_F16,
  CF_MATH_DTYPE_BF16,
  CF_MATH_DTYPE_FP8E4M3,
  CF_MATH_DTYPE_FP8E5M2,
} cf_math_dtype;
typedef enum cf_math_device
{
  CF_MATH_DEVICE_CPU = 0,
  CF_MATH_DEVICE_CUDA
} cf_math_device;
typedef enum cf_math_mem_flags
{
  CF_MATH_MEM_DEFAULT     = 0,
  CF_MATH_MEM_PINNED      = 1 << 0,
  CF_MATH_MEM_MANAGED     = 1 << 1,
  CF_MATH_MEM_POOLED      = 1 << 2,
  CF_MATH_MEM_ALIGNED128  = 1 << 3,
  CF_MATH_MEM_READ_ONLY   = 1 << 4,
  CF_MATH_MEM_PEER_MAPPED = 1 << 5,
} cf_math_mem_flags;

typedef enum cf_math_handle_opt
{
  CF_MATH_HANDLE_OPT_NONE        = 0,
  CF_MATH_HANDLE_OPT_ELEMENTWISE = 1 << 0,
  CF_MATH_HANDLE_OPT_REDUCTION   = 1 << 1,
  CF_MATH_HANDLE_OPT_MATMUL      = 1 << 2,
  CF_MATH_HANDLE_OPT_LINEAR      = 1 << 3,
  CF_MATH_HANDLE_OPT_CONV        = 1 << 4,
  CF_MATH_HANDLE_OPT_ATTENTION   = 1 << 5,
  CF_MATH_HANDLE_OPT_NORM        = 1 << 6,
  CF_MATH_HANDLE_OPT_RANDOM      = 1 << 7,
  CF_MATH_HANDLE_OPT_TRANSFER    = 1 << 8,
} cf_math_handle_opt;

struct cf_math_cuda_workspace
{
  void     *ptr;
  cf_usize  size;
  cf_usize  high_water;
};

struct cf_math_cuda_context
{
  int device_id;

  cudaStream_t stream;

  cublasHandle_t      cublas;
  cublasLtHandle_t    cublasLt;
  cudnnHandle_t       cudnn;
  cusparseHandle_t    cusparse;
  cusolverDnHandle_t  cusolverDn;
  curandGenerator_t   curand;

  cf_math_cuda_workspace cuda_workspace;
};

struct cf_math_allocator
{
  void *backend;
  cf_math_mem_flags mem_flag;
};

struct cf_math_memory_block
{
  cf_usize offset;
  cf_usize size;
  cf_usize ref_count;
};

struct cf_math_arena
{
  cf_usize offset;
  cf_usize capacity;

  cf_math_memory_block free_blocks[CF_MATH_MAX_FREE_BLOCKS];
  cf_usize free_count;

  cf_math_memory_block active_blocks[CF_MATH_MAX_ACTIVE_BLOCKS];
  cf_usize active_count;
};

struct cf_math_storage
{
  cf_math_arena arena;
  cf_math_dtype dtype;
  cf_math_device device;
  cf_math_allocator allocator;
};

struct cf_math_desc_cache
{
  cf_bool dirty;

  cf_bool has_cudnn_tensor;
  cudnnTensorDescriptor_t cudnn_tensor;

  cf_bool has_lt_layout;
  cublasLtMatrixLayout_t  lt_layout;
};

struct cf_math_handle
{
  cf_math_handle_opt optimized_for;
  cf_math_desc_cache desc_cache;
  cf_math_cuda_context *cuda_ctx;
  cf_math_storage storage;
};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Release an active byte slice back to storage tracking.
 * @param storage Storage arena that owns the active slice.
 * @param offset Slice byte offset inside the arena.
 * @param size Slice byte size.
 * @param released Receives whether the slice reached refcount zero.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_STATE`, or `CF_ERR_BOUNDS`.
 */
cf_status cf_math_storage_release_slice(cf_math_storage *storage, cf_usize offset, cf_usize size, cf_bool *released);

/**
 * @brief Initialize CUDA runtime handles for a device.
 * @param ctx Context object to initialize.
 * @param bytes Minimum workspace capacity in bytes.
 * @param device_id CUDA device index.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_INVALID`, `CF_ERR_UNSUPPORTED`, or a CUDA init error.
 */
cf_status cf_math_cuda_context_init(cf_math_cuda_context *ctx, cf_usize bytes, int device_id);

/**
 * @brief Reserve reusable CUDA operation workspace.
 * @param ctx CUDA context that owns the workspace.
 * @param bytes Minimum workspace capacity in bytes.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_UNSUPPORTED`, or a CUDA memory error.
 */
cf_status cf_math_cuda_context_reserve(cf_math_cuda_context *ctx, cf_usize bytes);

/**
 * @brief Destroy CUDA runtime handles and workspace owned by a context.
 * @param ctx Context object to destroy.
 * @return `CF_OK`, `CF_ERR_NULL`, or a CUDA cleanup error.
 */
cf_status cf_math_cuda_context_destroy(cf_math_cuda_context *ctx);

/**
 * @brief Initialize a CUDA-backed math handler and optionally reserve storage.
 * @param handler Handler to initialize.
 * @param ctx Shared CUDA context used by the handler; it must outlive handler.
 * @param dtype Storage element dtype.
 * @param device Storage device; currently `CF_MATH_DEVICE_CUDA`.
 * @param flags Allocation flags.
 * @param optimized_for Operation classes this handler is optimized for.
 * @param capacity Initial storage capacity in bytes.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_INVALID`, `CF_ERR_UNSUPPORTED`, or CUDA memory errors.
 */
cf_status cf_math_handle_init(cf_math_handle_t *handler, cf_math_cuda_context *ctx, cf_math_dtype dtype, cf_math_device device, cf_math_mem_flags flags, cf_math_handle_opt optimized_for, cf_usize capacity);

/**
 * @brief Allocate a byte slice from a handler storage arena.
 * @param handler Handler whose arena owns the allocation.
 * @param bytes Requested slice size in bytes.
 * @param ptr Receives the start of the allocated slice.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_BOUNDS`, `CF_ERR_OVERFLOW`, or a CUDA memory error.
 */
cf_status cf_math_handle_alloc(cf_math_handle_t *handler, cf_usize bytes, void **ptr);

/**
 * @brief Ensure handler storage has at least the requested byte capacity.
 * @param handler Handler whose storage arena should grow.
 * @param bytes Required storage capacity in bytes.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_STATE`, `CF_ERR_UNSUPPORTED`, or a CUDA memory error.
 */
cf_status cf_math_handle_reserve(cf_math_handle_t *handler, cf_usize bytes);

/**
 * @brief Reset handler arena slice tracking without freeing the base allocation.
 * @param handler Handler whose storage arena should reset.
 */
void cf_math_handle_reset(cf_math_handle_t *handler);

/**
 * @brief Destroy handler-owned storage and descriptor cache.
 * @param handler Handler to destroy.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_STATE`, `CF_ERR_UNSUPPORTED`, or a CUDA cleanup error.
 */
cf_status cf_math_handle_destroy(cf_math_handle_t *handler);

#ifdef __cplusplus
}
#endif

#endif /* CF_MATH_STORAGE_H */
