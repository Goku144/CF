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

#if !defined(CF_MATH_H)
#define CF_MATH_H

#include "RUNTIME/cf_status.h"
#include "RUNTIME/cf_types.h"

#define CF_MATH_MAX_NODE_INPUTS 8
#define CF_MATH_MAX_RANK 8
#define CF_MATH_HIGHEST_RANK CF_MATH_MAX_RANK
#define CF_MATH_MAX_FREE_BLOCKS 64
#define CF_MATH_MAX_ACTIVE_BLOCKS 256

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

typedef struct cf_math_cuda_workspace cf_math_cuda_workspace;
typedef struct cf_math_cuda_context cf_math_cuda_context;
typedef struct cf_math_allocator cf_math_allocator;
typedef struct cf_math_metadata cf_math_metadata;
typedef struct cf_math_desc_cache cf_math_desc_cache;
typedef struct cf_math_memory_block cf_math_memory_block;
typedef struct cf_math_storage cf_math_storage;
typedef struct cf_math_node cf_math_node;
typedef struct cf_math_handle cf_math_handle_t;
typedef struct cf_math cf_math;

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

typedef enum cf_math_layout
{
  CF_MATH_LAYOUT_ROW_MAJOR = 0,
  CF_MATH_LAYOUT_COL_MAJOR,
  CF_MATH_LAYOUT_NCHW,
  CF_MATH_LAYOUT_NHWC,
  CF_MATH_LAYOUT_STRIDED,
} cf_math_layout;

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

typedef enum cf_math_shape
{
  CF_MATH_SHAPE_SCALAR = 0,
  CF_MATH_SHAPE_VECTOR,
  CF_MATH_SHAPE_MATRIX,
  CF_MATH_SHAPE_TENSOR,
} cf_math_shape; 

typedef enum cf_math_grad_state
{
  CF_MATH_GRAD_NONE = 0,
  CF_MATH_GRAD_LEAF,
  CF_MATH_GRAD_INTERIOR,
  CF_MATH_GRAD_DETACHED
} cf_math_grad_state;

typedef enum cf_math_op_kind
{
  CF_MATH_OP_NONE = 0,

  CF_MATH_OP_ADD,
  CF_MATH_OP_SUB,
  CF_MATH_OP_MUL,
  CF_MATH_OP_DIV,

  CF_MATH_OP_MATMUL,
  CF_MATH_OP_LINEAR,

  CF_MATH_OP_RELU,
  CF_MATH_OP_GELU,
  CF_MATH_OP_SOFTMAX,
  CF_MATH_OP_CROSS_ENTROPY,

  CF_MATH_OP_LAYER_NORM,
  CF_MATH_OP_ATTENTION
} cf_math_op_kind;

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
  CF_MATH_HANDLE_OPT_TRANSFER    = 1 << 8
} cf_math_handle_opt;

struct cf_math_cuda_workspace
{
  void     *ptr;
  cf_usize  size;
  cf_usize  max_size;
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

  cf_usize total_allocated;
  cf_usize total_reserved;
  cf_usize allocation_count;

  cf_math_mem_flags mem_flag;
};

struct cf_math_memory_block
{
  cf_usize offset;
  cf_usize size;
  cf_usize ref_count;
};

struct cf_math_storage
{
  void *data_ptr;
  cf_usize offset;
  cf_usize capacity;
  cf_math_memory_block free_blocks[CF_MATH_MAX_FREE_BLOCKS];
  cf_usize free_count;
  cf_math_memory_block active_blocks[CF_MATH_MAX_ACTIVE_BLOCKS];
  cf_usize active_count;

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

/**
 * @brief Runtime/storage handler shared by one or more non-owning math views.
 *
 * The handler owns storage and descriptor state, but it does not own the CUDA
 * context. `cuda_ctx` points to a context whose lifetime must outlive the
 * handler.
 */
struct cf_math_handle
{
  cf_math_handle_opt optimized_for;
  cf_math_desc_cache desc_cache;
  cf_math_cuda_context *cuda_ctx;
  cf_math_storage storage;
};

struct cf_math_node
{
  cf_math_op_kind op;
  cf_math *inputs[CF_MATH_MAX_NODE_INPUTS];
  cf_usize input_count;
};

struct cf_math_metadata
{
  cf_usize rank;
  cf_usize dim[CF_MATH_MAX_RANK];
  cf_usize strides[CF_MATH_MAX_RANK];

  cf_usize len;
  cf_math_shape shape;
  cf_math_layout layout;
};

struct cf_math
{
  void *data;
  cf_usize byte_offset;
  cf_usize byte_size;

  cf_math_metadata *metadata;
  cf_math_handle_t *handler;

  cf_math *grad;
  cf_math_node *grad_fn;
  cf_math_grad_state grad_state;
};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Multiply two bytes in the AES GF(2^8) finite field.
 * @param p First finite-field byte operand.
 * @param q Second finite-field byte operand.
 * @return Computed 8-bit value.
 */
cf_u8 cf_math_g8_mul_mod(cf_u8 p, cf_u8 q);

/**
 * @brief Rotate an 8-bit value left.
 * @param x Input value or tensor.
 * @param n Bit count, scalar exponent, or batch index.
 * @return Computed 8-bit value.
 */
cf_u8 cf_math_rotl8(cf_u8 x, cf_u8 n);

/**
 * @brief Rotate an 8-bit value right.
 * @param x Input value or tensor.
 * @param n Bit count, scalar exponent, or batch index.
 * @return Computed 8-bit value.
 */
cf_u8 cf_math_rotr8(cf_u8 x, cf_u8 n);

/**
 * @brief Rotate a 32-bit value left.
 * @param x Input value or tensor.
 * @param n Bit count, scalar exponent, or batch index.
 * @return Computed 32-bit value.
 */
cf_u32 cf_math_rotl32(cf_u32 x, cf_u8 n);

/**
 * @brief Rotate a 32-bit value right.
 * @param x Input value or tensor.
 * @param n Bit count, scalar exponent, or batch index.
 * @return Computed 32-bit value.
 */
cf_u32 cf_math_rotr32(cf_u32 x, cf_u8 n);

/**
 * @brief Return the smaller of two cf_usize values.
 * @param a First input value, matrix, or sparse matrix.
 * @param b Second input value, tensor, bias tensor, or dense matrix.
 * @return Computed size value.
 */
cf_usize cf_math_min_usize(cf_usize a, cf_usize b);

/**
 * @brief Return the larger of two cf_usize values.
 * @param a First input value, matrix, or sparse matrix.
 * @param b Second input value, tensor, bias tensor, or dense matrix.
 * @return Computed size value.
 */
cf_usize cf_math_max_usize(cf_usize a, cf_usize b);

/**
 * @brief Initialize CUDA runtime handles for a device.
 * @param ctx Context object to initialize.
 * @param device_id CUDA device index.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_INVALID`, `CF_ERR_UNSUPPORTED`, or a CUDA init error.
 */
cf_status cf_math_cuda_context_init(cf_math_cuda_context *ctx, int device_id);

/**
 * @brief Destroy CUDA runtime handles and workspace owned by a context.
 * @param ctx Context object to destroy.
 * @return `CF_OK`, `CF_ERR_NULL`, or a CUDA cleanup error.
 */
cf_status cf_math_cuda_context_destroy(cf_math_cuda_context *ctx);

/**
 * @brief Initialize reusable shape metadata.
 * @param metadata Metadata object to initialize.
 * @param dim Shape dimensions, required when rank is nonzero.
 * @param rank Number of active dimensions.
 * @param shape Coarse shape kind.
 * @param layout Memory layout interpretation.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_INVALID`, or `CF_ERR_OVERFLOW`.
 */
cf_status cf_math_metadata_init(cf_math_metadata *metadata, cf_usize dim[CF_MATH_MAX_RANK], cf_usize rank, cf_math_shape shape, cf_math_layout layout);

/**
 * @brief Reserve reusable CUDA operation workspace.
 * @param ctx CUDA context that owns the workspace.
 * @param bytes Minimum workspace capacity in bytes.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_UNSUPPORTED`, or a CUDA memory error.
 */
cf_status cf_math_cuda_workspace_reserve(cf_math_cuda_context *ctx, cf_usize bytes);

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
 * @brief Ensure handler storage has at least the requested byte capacity.
 * @param handler Handler whose storage should grow.
 * @param bytes Required storage capacity in bytes.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_STATE`, `CF_ERR_UNSUPPORTED`, or CUDA memory errors.
 */
cf_status cf_math_handle_reserve(cf_math_handle_t *handler, cf_usize bytes);

/**
 * @brief Allocate a byte range from handler storage.
 * @param handler Handler arena to allocate from.
 * @param bytes Requested byte count.
 * @param ptr Receives the allocated range start.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_BOUNDS`, `CF_ERR_OVERFLOW`, or CUDA memory errors.
 */
cf_status cf_math_handle_alloc(cf_math_handle_t *handler, cf_usize bytes, void **ptr);

/**
 * @brief Reset handler arena offset without freeing storage.
 * @param handler Handler to reset.
 */
void cf_math_handle_reset(cf_math_handle_t *handler);

/**
 * @brief Destroy handler-owned storage and descriptor cache.
 * @param handler Handler to destroy.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_STATE`, `CF_ERR_UNSUPPORTED`, or a CUDA cleanup error.
 */
cf_status cf_math_handle_destroy(cf_math_handle_t *handler);

/**
 * @brief Bind a non-owning math view to a handler and metadata.
 * @param x Math view to bind.
 * @param handler Runtime/storage handler.
 * @param metadata Shape/layout metadata.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_INVALID`, `CF_ERR_BOUNDS`, `CF_ERR_OVERFLOW`, or CUDA memory errors.
 */
cf_status cf_math_bind(cf_math *x, cf_math_handle_t *handler, cf_math_metadata *metadata);

/**
 * @brief Unbind a math view and release its slice when no other view uses it.
 * @param x Math view to unbind.
 * @return `CF_OK`, `CF_ERR_NULL`, `CF_ERR_STATE`, or `CF_ERR_BOUNDS`.
 */
cf_status cf_math_unbind(cf_math *x);

/**
 * @brief Rebind a math view to a new handler and metadata.
 * @param x Math view to rebind.
 * @param handler New runtime/storage handler.
 * @param metadata New shape/layout metadata.
 * @return Status from unbind or bind.
 */
cf_status cf_math_rebind(cf_math *x, cf_math_handle_t *handler, cf_math_metadata *metadata);

#ifdef __cplusplus
}
#endif

#endif /* CF_MATH_H */
