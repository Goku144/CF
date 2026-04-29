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

#define CF_MATH_HIGHEST_RANK 8

#if defined(CF_CUDA_AVAILABLE)
  #if defined(__has_include)
    #if __has_include(<cuda_runtime_api.h>)
      #include <cuda_runtime_api.h>
      #define CF_MATH_HAVE_CUDA_RUNTIME 1
    #endif
    #if __has_include(<cublas_v2.h>)
      #include <cublas_v2.h>
      #define CF_MATH_HAVE_CUBLAS 1
    #endif
    #if __has_include(<cublasLt.h>)
      #include <cublasLt.h>
      #define CF_MATH_HAVE_CUBLASLT 1
    #endif
    #if __has_include(<cudnn.h>)
      #include <cudnn.h>
      #define CF_MATH_HAVE_CUDNN 1
    #endif
    #if __has_include(<cusparse.h>)
      #include <cusparse.h>
      #define CF_MATH_HAVE_CUSPARSE 1
    #endif
    #if __has_include(<cusolverDn.h>)
      #include <cusolverDn.h>
      #define CF_MATH_HAVE_CUSOLVER 1
    #endif
    #if __has_include(<curand.h>)
      #include <curand.h>
      #define CF_MATH_HAVE_CURAND 1
    #endif
    #if __has_include(<nccl.h>)
      #include <nccl.h>
      #define CF_MATH_HAVE_NCCL 1
    #endif
  #else
    #include <cuda_runtime_api.h>
    #include <cublas_v2.h>
    #include <cublasLt.h>
    #include <cudnn.h>
    #include <cusparse.h>
    #include <cusolverDn.h>
    #include <curand.h>
    #define CF_MATH_HAVE_CUDA_RUNTIME 1
    #define CF_MATH_HAVE_CUBLAS 1
    #define CF_MATH_HAVE_CUBLASLT 1
    #define CF_MATH_HAVE_CUDNN 1
    #define CF_MATH_HAVE_CUSPARSE 1
    #define CF_MATH_HAVE_CUSOLVER 1
    #define CF_MATH_HAVE_CURAND 1
  #endif
#endif

#if !defined(CF_MATH_HAVE_CUDA_RUNTIME)
typedef void *cudaStream_t;
typedef void *cudaMemPool_t;
#endif
#if !defined(CF_MATH_HAVE_CUBLAS)
typedef void *cublasHandle_t;
#endif
#if !defined(CF_MATH_HAVE_CUBLASLT)
typedef void *cublasLtHandle_t;
typedef void *cublasLtMatrixLayout_t;
#endif
#if !defined(CF_MATH_HAVE_CUDNN)
typedef void *cudnnHandle_t;
typedef void *cudnnTensorDescriptor_t;
typedef void *cudnnFilterDescriptor_t;
typedef void *cudnnRNNDataDescriptor_t;
#endif
#if !defined(CF_MATH_HAVE_CUSPARSE)
typedef void *cusparseHandle_t;
#endif
#if !defined(CF_MATH_HAVE_CUSOLVER)
typedef void *cusolverDnHandle_t;
#endif
#if !defined(CF_MATH_HAVE_CURAND)
typedef void *curandGenerator_t;
#endif
#if !defined(CF_MATH_HAVE_NCCL)
typedef void *ncclComm_t;
#endif

typedef enum cf_math_shape
{
  CF_SHAPE_SCALAR = 0,
  CF_SHAPE_MATRIX,
  CF_SHAPE_TENSOR,
} cf_math_shape;

typedef enum cf_math_device
{
  CF_DEVICE_CPU = 0,
  CF_DEVICE_CUDA,
} cf_math_device;

typedef enum cf_math_dtype
{
  CF_DTYPE_F64 = 0,
  CF_DTYPE_F32 = 1,
  CF_DTYPE_F16 = 2,
  CF_DTYPE_BF16 = 3,
  CF_DTYPE_FP8E4M3 = 4,
  CF_DTYPE_FP8E5M2 = 5,
  CF_DTYPE_I32 = 6,
  CF_DTYPE_I8 = 7,
  CF_DTYPE_U8 = 8,
  CF_DTYPE_BOOL = 9,
} cf_math_dtype;

typedef enum cf_math_layout
{
  CF_LAYOUT_ROW_MAJOR = 0,
  CF_LAYOUT_COL_MAJOR = 1,
  CF_LAYOUT_NHWC = 2,
  CF_LAYOUT_NCHW = 3,
  CF_LAYOUT_STRIDED = 4,
} cf_math_layout;

typedef enum cf_math_mem_flags
{
  CF_MEM_DEFAULT = 0x00,
  CF_MEM_PINNED = 0x01,
  CF_MEM_MANAGED = 0x02,
  CF_MEM_POOLED = 0x04,
  CF_MEM_ALIGNED_128 = 0x08,
  CF_MEM_READ_ONLY = 0x10,
  CF_MEM_PEER_MAPPED = 0x20,
} cf_math_mem_flags;

typedef enum cf_math_grad_state
{
  CF_GRAD_NONE = 0,
  CF_GRAD_LEAF = 1,
  CF_GRAD_INTERIOR = 2,
  CF_GRAD_DETACHED = 3,
} cf_math_grad_state;

typedef enum cf_math_softmax_mode
{
  CF_SOFTMAX_CHANNEL = 0,
  CF_SOFTMAX_INSTANCE = 1,
} cf_math_softmax_mode;

typedef enum cf_math_rnn_mode
{
  CF_RNN_RELU = 0,
  CF_RNN_TANH = 1,
  CF_RNN_LSTM = 2,
  CF_RNN_GRU = 3,
} cf_math_rnn_mode;

typedef struct cf_math_node cf_math_node;

typedef struct cf_math_workspace
{
  void *ptr;
  cf_usize size;
  cf_usize high_water;
} cf_math_workspace;

typedef struct cf_math_cuda_context
{
  int device_id;
  cudaStream_t stream;
  cudaStream_t h2d_stream;
  cudaStream_t d2h_stream;
  cublasHandle_t cublas;
  cublasLtHandle_t cublasLt;
  cudnnHandle_t cudnn;
  cusparseHandle_t cusparse;
  cusolverDnHandle_t cusolver;
  curandGenerator_t curand;
  ncclComm_t nccl;
  cf_math_workspace workspace;
} cf_math_cuda_context;

typedef struct cf_math_desc_cache
{
  cudnnTensorDescriptor_t tensor_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnRNNDataDescriptor_t rnn_data_desc;
  cublasLtMatrixLayout_t lt_layout;
  cf_u8 valid;
} cf_math_desc_cache;

typedef struct cf_math_storage
{
  void *data_ptr;
  cf_usize capacity;
  cf_u32 refcount;
  cf_math_mem_flags mem_flags;
  int device_id;
  cf_math_device device;
} cf_math_storage;

typedef struct cf_math_metadata
{
  cf_usize len;
  cf_usize batch;
  cf_usize strides[CF_MATH_HIGHEST_RANK];
  cf_math_shape shape;
  cf_math_layout layout;
  cf_math_device device;
  cf_math_dtype dtype;
  cf_math_mem_flags mem_flags;
  cf_math_cuda_context ctx;
} cf_math_metadata;

typedef struct cf_math
{
  cf_math_storage *storage;
  void *data;
  cf_usize byte_offset;
  cf_usize rank;
  cf_usize dim[CF_MATH_HIGHEST_RANK];
  cf_math_grad_state grad_state;
  struct cf_math *grad;
  cf_math_node *grad_fn;
  cf_math_desc_cache desc_cache;
  cf_math_metadata metadata;
} cf_math;

typedef struct cf_math_conv2d_params
{
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int dilation_h;
  int dilation_w;
  int groups;
} cf_math_conv2d_params;

typedef struct cf_math_dropout_state
{
  void *descriptor;
  void *reserve;
  cf_usize reserve_size;
  float probability;
  cf_u64 seed;
} cf_math_dropout_state;

typedef struct cf_math_rnn_state
{
  void *descriptor;
  void *weights;
  void *workspace;
  void *reserve;
  cf_usize weights_size;
  cf_usize workspace_size;
  cf_usize reserve_size;
  cf_math_rnn_mode mode;
} cf_math_rnn_state;

typedef struct cf_math_sparse
{
  void *values;
  cf_i32 *row_offsets;
  cf_i32 *col_indices;
  cf_usize rows;
  cf_usize cols;
  cf_usize nnz;
  cf_math_dtype dtype;
  cf_math_device device;
} cf_math_sparse;

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
 * @brief Return the byte size of one element of a math dtype.
 * @param dtype Tensor element data type.
 * @return Computed size value.
 */
cf_usize cf_math_dtype_size(cf_math_dtype dtype);
/**
 * @brief Initialize a CUDA math context for one device.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @param device_id CUDA device id.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_context_init(cf_math_cuda_context *ctx, int device_id);
/**
 * @brief Destroy handles and workspace owned by a CUDA math context.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_context_destroy(cf_math_cuda_context *ctx);
/**
 * @brief Grow the reusable CUDA workspace to at least the requested byte size.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @param bytes Requested workspace size in bytes.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_workspace_reserve(cf_math_cuda_context *ctx, cf_usize bytes);

/**
 * @brief Allocate a tensor with shape, dtype, device, and memory flags.
 * @param out Output tensor to write or allocate.
 * @param dim Shape dimension array.
 * @param rank Number of active dimensions.
 * @param dtype Tensor element data type.
 * @param device Target memory device.
 * @param flags Memory allocation flags.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_alloc(cf_math *out, const cf_usize dim[CF_MATH_HIGHEST_RANK], cf_usize rank, cf_math_dtype dtype, cf_math_device device, cf_math_mem_flags flags, cf_math_cuda_context *ctx);
/**
 * @brief Release a tensor storage reference and reset the tensor object.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_free(cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Allocate a CPU tensor using pinned host-memory metadata.
 * @param out Output tensor to write or allocate.
 * @param dim Shape dimension array.
 * @param rank Number of active dimensions.
 * @param dtype Tensor element data type.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_alloc_pinned(cf_math *out, const cf_usize dim[CF_MATH_HIGHEST_RANK], cf_usize rank, cf_math_dtype dtype, cf_math_cuda_context *ctx);
/**
 * @brief Allocate a tensor using CUDA managed-memory metadata when supported.
 * @param out Output tensor to write or allocate.
 * @param dim Shape dimension array.
 * @param rank Number of active dimensions.
 * @param dtype Tensor element data type.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_alloc_managed(cf_math *out, const cf_usize dim[CF_MATH_HIGHEST_RANK], cf_usize rank, cf_math_dtype dtype, cf_math_cuda_context *ctx);
/**
 * @brief Create a zero-copy tensor view into another tensor storage buffer.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param offset_elems Element offset into the source tensor storage.
 * @param dim Shape dimension array.
 * @param rank Number of active dimensions.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_view(cf_math *out, const cf_math *x, cf_usize offset_elems, const cf_usize dim[CF_MATH_HIGHEST_RANK], cf_usize rank);
/**
 * @brief Create a contiguous copy of a tensor.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_contiguous(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Copy or move a tensor into CUDA device storage.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param device_id CUDA device id.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_to_device(cf_math *out, const cf_math *x, int device_id, cf_math_cuda_context *ctx);
/**
 * @brief Copy or move a tensor into CPU host storage.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_to_host(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Clone a tensor into independent storage.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_clone(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);

/**
 * @brief Fill every tensor element with one scalar value.
 * @param out Output tensor to write or allocate.
 * @param value Scalar value to write.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_fill(cf_math *out, double value, cf_math_cuda_context *ctx);
/**
 * @brief Set every tensor element to zero.
 * @param out Output tensor to write or allocate.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_zeros(cf_math *out, cf_math_cuda_context *ctx);
/**
 * @brief Set every tensor element to one.
 * @param out Output tensor to write or allocate.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_ones(cf_math *out, cf_math_cuda_context *ctx);
/**
 * @brief Fill a tensor with deterministic uniform random values.
 * @param out Output tensor to write or allocate.
 * @param lo Lower scalar bound.
 * @param hi Upper scalar bound.
 * @param seed Deterministic random seed.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_rand_uniform(cf_math *out, double lo, double hi, cf_u64 seed, cf_math_cuda_context *ctx);
/**
 * @brief Fill a tensor with deterministic normal random values.
 * @param out Output tensor to write or allocate.
 * @param mean Mean value tensor or scalar distribution mean.
 * @param stddev Standard deviation for normal random values.
 * @param seed Deterministic random seed.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_rand_normal(cf_math *out, double mean, double stddev, cf_u64 seed, cf_math_cuda_context *ctx);
/**
 * @brief Fill a tensor with deterministic Bernoulli random values.
 * @param out Output tensor to write or allocate.
 * @param p Function-specific probability, convolution parameters, or byte operand.
 * @param seed Deterministic random seed.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_rand_bernoulli(cf_math *out, double p, cf_u64 seed, cf_math_cuda_context *ctx);
/**
 * @brief Initialize weights with Xavier uniform values.
 * @param out Output tensor to write or allocate.
 * @param fan_in Number of input features used by initializer scaling.
 * @param fan_out Number of output features used by initializer scaling.
 * @param seed Deterministic random seed.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_init_xavier_uniform(cf_math *out, cf_usize fan_in, cf_usize fan_out, cf_u64 seed, cf_math_cuda_context *ctx);
/**
 * @brief Initialize weights with Xavier normal values.
 * @param out Output tensor to write or allocate.
 * @param fan_in Number of input features used by initializer scaling.
 * @param fan_out Number of output features used by initializer scaling.
 * @param seed Deterministic random seed.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_init_xavier_normal(cf_math *out, cf_usize fan_in, cf_usize fan_out, cf_u64 seed, cf_math_cuda_context *ctx);
/**
 * @brief Initialize weights with Kaiming normal values.
 * @param out Output tensor to write or allocate.
 * @param fan_in Number of input features used by initializer scaling.
 * @param seed Deterministic random seed.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_init_kaiming_normal(cf_math *out, cf_usize fan_in, cf_u64 seed, cf_math_cuda_context *ctx);
/**
 * @brief Initialize weights with Kaiming uniform values.
 * @param out Output tensor to write or allocate.
 * @param fan_in Number of input features used by initializer scaling.
 * @param seed Deterministic random seed.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_init_kaiming_uniform(cf_math *out, cf_usize fan_in, cf_u64 seed, cf_math_cuda_context *ctx);
/**
 * @brief Initialize a rank-2 tensor with an orthogonal basis approximation.
 * @param out Output tensor to write or allocate.
 * @param seed Deterministic random seed.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_init_orthogonal(cf_math *out, cf_u64 seed, cf_math_cuda_context *ctx);
/**
 * @brief Initialize a matrix as an identity-like matrix.
 * @param out Output tensor to write or allocate.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_init_eye(cf_math *out, cf_math_cuda_context *ctx);

/**
 * @brief Add two tensors element by element.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param y Second input tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_add(cf_math *out, const cf_math *x, const cf_math *y, cf_math_cuda_context *ctx);
/**
 * @brief Add a scalar to every tensor element.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param c Scalar operand or LSTM cell-state tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_add_scalar(cf_math *out, const cf_math *x, double c, cf_math_cuda_context *ctx);
/**
 * @brief Subtract one tensor from another element by element.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param y Second input tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_sub(cf_math *out, const cf_math *x, const cf_math *y, cf_math_cuda_context *ctx);
/**
 * @brief Multiply two tensors element by element.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param y Second input tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_mul(cf_math *out, const cf_math *x, const cf_math *y, cf_math_cuda_context *ctx);
/**
 * @brief Multiply every tensor element by a scalar.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param c Scalar operand or LSTM cell-state tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_mul_scalar(cf_math *out, const cf_math *x, double c, cf_math_cuda_context *ctx);
/**
 * @brief Divide one tensor by another element by element.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param y Second input tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_div(cf_math *out, const cf_math *x, const cf_math *y, cf_math_cuda_context *ctx);
/**
 * @brief Divide every tensor element by a scalar.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param c Scalar operand or LSTM cell-state tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_div_scalar(cf_math *out, const cf_math *x, double c, cf_math_cuda_context *ctx);
/**
 * @brief Raise every tensor element to a scalar power.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param n Bit count, scalar exponent, or batch index.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_pow(cf_math *out, const cf_math *x, double n, cf_math_cuda_context *ctx);
/**
 * @brief Compute elementwise square root.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_sqrt(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Compute elementwise reciprocal square root.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_rsqrt(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Compute elementwise exponential.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_exp(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Compute elementwise natural logarithm.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_log(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Compute elementwise absolute value.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_abs(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Compute elementwise arithmetic negation.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_neg(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Clamp every tensor element into a scalar range.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param lo Lower scalar bound.
 * @param hi Upper scalar bound.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_clamp(cf_math *out, const cf_math *x, double lo, double hi, cf_math_cuda_context *ctx);
/**
 * @brief Compute elementwise sign values.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_sign(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);

/**
 * @brief Reduce all tensor elements to their sum.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_sum(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Reduce one tensor axis to sums.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param axis Axis index used by the operation.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_sum_axis(cf_math *out, const cf_math *x, cf_usize axis, cf_math_cuda_context *ctx);
/**
 * @brief Reduce all tensor elements to their mean.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_mean(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Reduce one tensor axis to means.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param axis Axis index used by the operation.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_mean_axis(cf_math *out, const cf_math *x, cf_usize axis, cf_math_cuda_context *ctx);
/**
 * @brief Compute population variance across all tensor elements.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_var(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Compute population standard deviation across all tensor elements.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_std(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Compute the L2 norm of a tensor.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_norm2(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Compute the L1 norm of a tensor.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_norm1(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Reduce all tensor elements to their maximum value.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_max(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Reduce all tensor elements to their minimum value.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_min(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Return the index of the largest tensor element.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_argmax(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Return the index of the smallest tensor element.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_argmin(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Compute the dot product of two tensors interpreted as vectors.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param y Second input tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_dot(cf_math *out, const cf_math *x, const cf_math *y, cf_math_cuda_context *ctx);
/**
 * @brief Compute the inclusive prefix sum of a tensor.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_cumsum(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);

/**
 * @brief Multiply two rank-2 matrices.
 * @param out Output tensor to write or allocate.
 * @param a First input value, matrix, or sparse matrix.
 * @param b Second input value, tensor, bias tensor, or dense matrix.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_matmul(cf_math *out, const cf_math *a, const cf_math *b, cf_math_cuda_context *ctx);
/**
 * @brief Multiply two matrices with optional logical transposition.
 * @param out Output tensor to write or allocate.
 * @param a First input value, matrix, or sparse matrix.
 * @param b Second input value, tensor, bias tensor, or dense matrix.
 * @param trans_a Whether to logically transpose the left matrix.
 * @param trans_b Whether to logically transpose the right matrix.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_matmul_t(cf_math *out, const cf_math *a, const cf_math *b, cf_bool trans_a, cf_bool trans_b, cf_math_cuda_context *ctx);
/**
 * @brief Multiply batches of matrices.
 * @param out Output tensor to write or allocate.
 * @param a First input value, matrix, or sparse matrix.
 * @param b Second input value, tensor, bias tensor, or dense matrix.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_matmul_batched(cf_math *out, const cf_math *a, const cf_math *b, cf_math_cuda_context *ctx);
/**
 * @brief Compute a dense linear layer.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param w Weight or parameter tensor.
 * @param b Second input value, tensor, bias tensor, or dense matrix.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_linear(cf_math *out, const cf_math *x, const cf_math *w, const cf_math *b, cf_math_cuda_context *ctx);
/**
 * @brief Compute a dense linear layer followed by ReLU.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param w Weight or parameter tensor.
 * @param b Second input value, tensor, bias tensor, or dense matrix.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_linear_fused_relu(cf_math *out, const cf_math *x, const cf_math *w, const cf_math *b, cf_math_cuda_context *ctx);
/**
 * @brief Compute a dense linear layer followed by GELU.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param w Weight or parameter tensor.
 * @param b Second input value, tensor, bias tensor, or dense matrix.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_linear_fused_gelu(cf_math *out, const cf_math *x, const cf_math *w, const cf_math *b, cf_math_cuda_context *ctx);
/**
 * @brief Compute dense linear-layer weight gradients.
 * @param dW Output weight-gradient tensor.
 * @param dL Incoming loss-gradient tensor.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_linear_backward_W(cf_math *dW, const cf_math *dL, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Compute dense linear-layer input gradients.
 * @param dx Output input-gradient tensor.
 * @param dL Incoming loss-gradient tensor.
 * @param W Weight tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_linear_backward_x(cf_math *dx, const cf_math *dL, const cf_math *W, cf_math_cuda_context *ctx);
/**
 * @brief Compute dense linear-layer bias gradients.
 * @param db Output bias-gradient tensor.
 * @param dL Incoming loss-gradient tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_linear_backward_b(cf_math *db, const cf_math *dL, cf_math_cuda_context *ctx);
/**
 * @brief Compute the outer product of two vectors.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param y Second input tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_outer(cf_math *out, const cf_math *x, const cf_math *y, cf_math_cuda_context *ctx);
/**
 * @brief Multiply a matrix by a vector.
 * @param out Output tensor to write or allocate.
 * @param a First input value, matrix, or sparse matrix.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_matvec(cf_math *out, const cf_math *a, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Transpose a rank-2 matrix into an output tensor.
 * @param out Output tensor to write or allocate.
 * @param a First input value, matrix, or sparse matrix.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_transpose(cf_math *out, const cf_math *a, cf_math_cuda_context *ctx);
/**
 * @brief Scale every matrix or tensor element by a scalar.
 * @param out Output tensor to write or allocate.
 * @param a First input value, matrix, or sparse matrix.
 * @param alpha Activation or loss scalar alpha.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_scale(cf_math *out, const cf_math *a, double alpha, cf_math_cuda_context *ctx);

/**
 * @brief Compute a 2D convolution forward pass.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param w Weight or parameter tensor.
 * @param b Second input value, tensor, bias tensor, or dense matrix.
 * @param p Function-specific probability, convolution parameters, or byte operand.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_conv2d_fwd(cf_math *out, const cf_math *x, const cf_math *w, const cf_math *b, cf_math_conv2d_params p, cf_math_cuda_context *ctx);
/**
 * @brief Compute a 2D convolution gradient with respect to input data.
 * @param dx Output input-gradient tensor.
 * @param dL Incoming loss-gradient tensor.
 * @param w Weight or parameter tensor.
 * @param p Function-specific probability, convolution parameters, or byte operand.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_conv2d_bwd_data(cf_math *dx, const cf_math *dL, const cf_math *w, cf_math_conv2d_params p, cf_math_cuda_context *ctx);
/**
 * @brief Compute a 2D convolution gradient with respect to filters.
 * @param dW Output weight-gradient tensor.
 * @param dL Incoming loss-gradient tensor.
 * @param x Input value or tensor.
 * @param p Function-specific probability, convolution parameters, or byte operand.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_conv2d_bwd_filter(cf_math *dW, const cf_math *dL, const cf_math *x, cf_math_conv2d_params p, cf_math_cuda_context *ctx);
/**
 * @brief Compute a 2D convolution bias gradient.
 * @param db Output bias-gradient tensor.
 * @param dL Incoming loss-gradient tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_conv2d_bwd_bias(cf_math *db, const cf_math *dL, cf_math_cuda_context *ctx);
/**
 * @brief Compute a depthwise 2D convolution forward pass.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param w Weight or parameter tensor.
 * @param p Function-specific probability, convolution parameters, or byte operand.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_conv2d_depthwise_fwd(cf_math *out, const cf_math *x, const cf_math *w, cf_math_conv2d_params p, cf_math_cuda_context *ctx);
/**
 * @brief Compute a dilated 2D convolution forward pass.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param w Weight or parameter tensor.
 * @param b Second input value, tensor, bias tensor, or dense matrix.
 * @param p Function-specific probability, convolution parameters, or byte operand.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_conv2d_dilated_fwd(cf_math *out, const cf_math *x, const cf_math *w, const cf_math *b, cf_math_conv2d_params p, cf_math_cuda_context *ctx);
/**
 * @brief Compute a transposed 2D convolution forward pass.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param w Weight or parameter tensor.
 * @param p Function-specific probability, convolution parameters, or byte operand.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_conv2d_transpose_fwd(cf_math *out, const cf_math *x, const cf_math *w, cf_math_conv2d_params p, cf_math_cuda_context *ctx);
/**
 * @brief Compute a 1D convolution forward pass through the convolution surface.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param w Weight or parameter tensor.
 * @param b Second input value, tensor, bias tensor, or dense matrix.
 * @param p Function-specific probability, convolution parameters, or byte operand.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_conv1d_fwd(cf_math *out, const cf_math *x, const cf_math *w, const cf_math *b, cf_math_conv2d_params p, cf_math_cuda_context *ctx);
/**
 * @brief Compute a 3D convolution forward pass.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param w Weight or parameter tensor.
 * @param b Second input value, tensor, bias tensor, or dense matrix.
 * @param p Function-specific probability, convolution parameters, or byte operand.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_conv3d_fwd(cf_math *out, const cf_math *x, const cf_math *w, const cf_math *b, cf_math_conv2d_params p, cf_math_cuda_context *ctx);

/**
 * @brief Compute batch normalization forward training output and saved statistics.
 * @param out Output tensor to write or allocate.
 * @param saved_mean saved_mean parameter.
 * @param saved_inv_var saved_inv_var parameter.
 * @param x Input value or tensor.
 * @param gamma Normalization scale tensor.
 * @param beta Normalization offset tensor or scalar beta coefficient.
 * @param eps Numerical stability epsilon.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_bn_fwd_train(cf_math *out, cf_math *saved_mean, cf_math *saved_inv_var, const cf_math *x, const cf_math *gamma, const cf_math *beta, double eps, cf_math_cuda_context *ctx);
/**
 * @brief Compute batch normalization forward inference output.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param gamma Normalization scale tensor.
 * @param beta Normalization offset tensor or scalar beta coefficient.
 * @param mean Mean value tensor or scalar distribution mean.
 * @param var var parameter.
 * @param eps Numerical stability epsilon.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_bn_fwd_infer(cf_math *out, const cf_math *x, const cf_math *gamma, const cf_math *beta, const cf_math *mean, const cf_math *var, double eps, cf_math_cuda_context *ctx);
/**
 * @brief Compute batch normalization backward outputs.
 * @param dx Output input-gradient tensor.
 * @param dgamma Output scale-gradient tensor.
 * @param dbeta Output bias/offset-gradient tensor.
 * @param dL Incoming loss-gradient tensor.
 * @param x Input value or tensor.
 * @param gamma Normalization scale tensor.
 * @param saved_mean saved_mean parameter.
 * @param saved_inv_var saved_inv_var parameter.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_bn_bwd(cf_math *dx, cf_math *dgamma, cf_math *dbeta, const cf_math *dL, const cf_math *x, const cf_math *gamma, const cf_math *saved_mean, const cf_math *saved_inv_var, cf_math_cuda_context *ctx);
/**
 * @brief Compute layer normalization forward output.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param gamma Normalization scale tensor.
 * @param beta Normalization offset tensor or scalar beta coefficient.
 * @param eps Numerical stability epsilon.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_ln_fwd(cf_math *out, const cf_math *x, const cf_math *gamma, const cf_math *beta, double eps, cf_math_cuda_context *ctx);
/**
 * @brief Compute layer normalization backward outputs.
 * @param dx Output input-gradient tensor.
 * @param dgamma Output scale-gradient tensor.
 * @param dbeta Output bias/offset-gradient tensor.
 * @param dL Incoming loss-gradient tensor.
 * @param x Input value or tensor.
 * @param gamma Normalization scale tensor.
 * @param eps Numerical stability epsilon.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_ln_bwd(cf_math *dx, cf_math *dgamma, cf_math *dbeta, const cf_math *dL, const cf_math *x, const cf_math *gamma, double eps, cf_math_cuda_context *ctx);
/**
 * @brief Compute instance normalization forward output.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param gamma Normalization scale tensor.
 * @param beta Normalization offset tensor or scalar beta coefficient.
 * @param eps Numerical stability epsilon.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_in_fwd(cf_math *out, const cf_math *x, const cf_math *gamma, const cf_math *beta, double eps, cf_math_cuda_context *ctx);
/**
 * @brief Compute group normalization forward output.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param gamma Normalization scale tensor.
 * @param beta Normalization offset tensor or scalar beta coefficient.
 * @param groups groups parameter.
 * @param eps Numerical stability epsilon.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_gn_fwd(cf_math *out, const cf_math *x, const cf_math *gamma, const cf_math *beta, cf_usize groups, double eps, cf_math_cuda_context *ctx);
/**
 * @brief Compute RMS normalization forward output.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param gamma Normalization scale tensor.
 * @param eps Numerical stability epsilon.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_rms_norm_fwd(cf_math *out, const cf_math *x, const cf_math *gamma, double eps, cf_math_cuda_context *ctx);
/**
 * @brief Compute RMS normalization backward outputs.
 * @param dx Output input-gradient tensor.
 * @param dgamma Output scale-gradient tensor.
 * @param dL Incoming loss-gradient tensor.
 * @param x Input value or tensor.
 * @param gamma Normalization scale tensor.
 * @param eps Numerical stability epsilon.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_rms_norm_bwd(cf_math *dx, cf_math *dgamma, const cf_math *dL, const cf_math *x, const cf_math *gamma, double eps, cf_math_cuda_context *ctx);

/**
 * @brief Apply ReLU activation.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_relu(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Compute ReLU activation backward gradient.
 * @param dx Output input-gradient tensor.
 * @param dy Incoming output-gradient tensor.
 * @param y Second input tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_relu_bwd(cf_math *dx, const cf_math *dy, const cf_math *y, cf_math_cuda_context *ctx);
/**
 * @brief Apply leaky ReLU activation.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param alpha Activation or loss scalar alpha.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_leaky_relu(cf_math *out, const cf_math *x, double alpha, cf_math_cuda_context *ctx);
/**
 * @brief Apply ELU activation.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param alpha Activation or loss scalar alpha.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_elu(cf_math *out, const cf_math *x, double alpha, cf_math_cuda_context *ctx);
/**
 * @brief Apply sigmoid activation.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_sigmoid(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Compute sigmoid activation backward gradient.
 * @param dx Output input-gradient tensor.
 * @param dy Incoming output-gradient tensor.
 * @param y Second input tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_sigmoid_bwd(cf_math *dx, const cf_math *dy, const cf_math *y, cf_math_cuda_context *ctx);
/**
 * @brief Apply hyperbolic tangent activation.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_tanh(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Compute tanh activation backward gradient.
 * @param dx Output input-gradient tensor.
 * @param dy Incoming output-gradient tensor.
 * @param y Second input tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_tanh_bwd(cf_math *dx, const cf_math *dy, const cf_math *y, cf_math_cuda_context *ctx);
/**
 * @brief Apply exact GELU activation.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_gelu(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Apply approximate GELU activation.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_gelu_approx(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Compute GELU activation backward gradient.
 * @param dx Output input-gradient tensor.
 * @param dy Incoming output-gradient tensor.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_gelu_bwd(cf_math *dx, const cf_math *dy, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Apply Swish activation.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param beta Normalization offset tensor or scalar beta coefficient.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_swish(cf_math *out, const cf_math *x, double beta, cf_math_cuda_context *ctx);
/**
 * @brief Apply SiLU activation.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_silu(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Apply softplus activation.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_softplus(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Apply Mish activation.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_mish(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);

/**
 * @brief Compute stable softmax along an axis.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param axis Axis index used by the operation.
 * @param mode Softmax interpretation mode.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_softmax_fwd(cf_math *out, const cf_math *x, cf_usize axis, cf_math_softmax_mode mode, cf_math_cuda_context *ctx);
/**
 * @brief Compute softmax backward gradient along an axis.
 * @param dx Output input-gradient tensor.
 * @param dy Incoming output-gradient tensor.
 * @param y Second input tensor.
 * @param axis Axis index used by the operation.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_softmax_bwd(cf_math *dx, const cf_math *dy, const cf_math *y, cf_usize axis, cf_math_cuda_context *ctx);
/**
 * @brief Compute stable log-softmax along an axis.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param axis Axis index used by the operation.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_log_softmax_fwd(cf_math *out, const cf_math *x, cf_usize axis, cf_math_cuda_context *ctx);
/**
 * @brief Compute log-softmax backward gradient along an axis.
 * @param dx Output input-gradient tensor.
 * @param dy Incoming output-gradient tensor.
 * @param y Second input tensor.
 * @param axis Axis index used by the operation.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_log_softmax_bwd(cf_math *dx, const cf_math *dy, const cf_math *y, cf_usize axis, cf_math_cuda_context *ctx);
/**
 * @brief Compute cross-entropy loss and optionally its fused gradient.
 * @param loss Output scalar loss tensor.
 * @param dx Output input-gradient tensor.
 * @param logits Input logits tensor.
 * @param target Target labels or probabilities tensor.
 * @param axis Axis index used by the operation.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_cross_entropy(cf_math *loss, cf_math *dx, const cf_math *logits, const cf_math *target, cf_usize axis, cf_math_cuda_context *ctx);
/**
 * @brief Compute cross-entropy gradient from probabilities and targets.
 * @param dx Output input-gradient tensor.
 * @param prob Input probability tensor.
 * @param target Target labels or probabilities tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_cross_entropy_bwd(cf_math *dx, const cf_math *prob, const cf_math *target, cf_math_cuda_context *ctx);
/**
 * @brief Compute negative log-likelihood loss.
 * @param loss Output scalar loss tensor.
 * @param log_prob Input log-probability tensor.
 * @param labels Input class-label tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_nll_loss(cf_math *loss, const cf_math *log_prob, const cf_math *labels, cf_math_cuda_context *ctx);
/**
 * @brief Compute mean squared error loss.
 * @param loss Output scalar loss tensor.
 * @param y Second input tensor.
 * @param target Target labels or probabilities tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_mse_loss(cf_math *loss, const cf_math *y, const cf_math *target, cf_math_cuda_context *ctx);
/**
 * @brief Compute mean squared error backward gradient.
 * @param dx Output input-gradient tensor.
 * @param y Second input tensor.
 * @param target Target labels or probabilities tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_mse_loss_bwd(cf_math *dx, const cf_math *y, const cf_math *target, cf_math_cuda_context *ctx);
/**
 * @brief Compute binary cross-entropy loss.
 * @param loss Output scalar loss tensor.
 * @param prob Input probability tensor.
 * @param target Target labels or probabilities tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_bce_loss(cf_math *loss, const cf_math *prob, const cf_math *target, cf_math_cuda_context *ctx);
/**
 * @brief Compute Huber loss.
 * @param loss Output scalar loss tensor.
 * @param y Second input tensor.
 * @param target Target labels or probabilities tensor.
 * @param delta Huber transition threshold.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_huber_loss(cf_math *loss, const cf_math *y, const cf_math *target, double delta, cf_math_cuda_context *ctx);
/**
 * @brief Compute focal loss.
 * @param loss Output scalar loss tensor.
 * @param prob Input probability tensor.
 * @param target Target labels or probabilities tensor.
 * @param alpha Activation or loss scalar alpha.
 * @param gamma Normalization scale tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_focal_loss(cf_math *loss, const cf_math *prob, const cf_math *target, double alpha, double gamma, cf_math_cuda_context *ctx);

/**
 * @brief Compute scaled attention scores from query and key tensors.
 * @param out Output tensor to write or allocate.
 * @param q Query tensor or byte operand.
 * @param k Key tensor.
 * @param scale Scalar multiplier or attention scale.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_attn_scores(cf_math *out, const cf_math *q, const cf_math *k, double scale, cf_math_cuda_context *ctx);
/**
 * @brief Add an attention mask to attention scores.
 * @param out Output tensor to write or allocate.
 * @param scores scores parameter.
 * @param mask mask parameter.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_attn_mask_add(cf_math *out, const cf_math *scores, const cf_math *mask, cf_math_cuda_context *ctx);
/**
 * @brief Apply attention softmax over the final axis.
 * @param out Output tensor to write or allocate.
 * @param scores scores parameter.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_attn_softmax(cf_math *out, const cf_math *scores, cf_math_cuda_context *ctx);
/**
 * @brief Compute attention context from attention probabilities and value tensor.
 * @param out Output tensor to write or allocate.
 * @param attn attn parameter.
 * @param v Value tensor, optimizer velocity tensor, or dense value tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_attn_context(cf_math *out, const cf_math *attn, const cf_math *v, cf_math_cuda_context *ctx);
/**
 * @brief Apply the attention output projection.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param wo Attention output-projection weight tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_attn_proj(cf_math *out, const cf_math *x, const cf_math *wo, cf_math_cuda_context *ctx);
/**
 * @brief Compute the public multi-head attention forward surface.
 * @param out Output tensor to write or allocate.
 * @param q Query tensor or byte operand.
 * @param k Key tensor.
 * @param v Value tensor, optimizer velocity tensor, or dense value tensor.
 * @param wo Attention output-projection weight tensor.
 * @param heads Number of attention heads.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_mha_fwd(cf_math *out, const cf_math *q, const cf_math *k, const cf_math *v, const cf_math *wo, cf_usize heads, cf_math_cuda_context *ctx);
/**
 * @brief Compute the public multi-head attention backward surface.
 * @param dq Output query-gradient tensor.
 * @param dk Output key-gradient tensor.
 * @param dv Output value-gradient tensor.
 * @param dwo Output projection-weight-gradient tensor.
 * @param dL Incoming loss-gradient tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_mha_bwd(cf_math *dq, cf_math *dk, cf_math *dv, cf_math *dwo, const cf_math *dL, cf_math_cuda_context *ctx);
/**
 * @brief Apply dropout to attention probabilities.
 * @param out Output tensor to write or allocate.
 * @param state Layer or RNG state object.
 * @param x Input value or tensor.
 * @param p Function-specific probability, convolution parameters, or byte operand.
 * @param seed Deterministic random seed.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_attn_dropout_fwd(cf_math *out, cf_math_dropout_state *state, const cf_math *x, double p, cf_u64 seed, cf_math_cuda_context *ctx);
/**
 * @brief Apply rotary position embedding forward rotation.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param cos_table Cosine lookup tensor for RoPE.
 * @param sin_table Sine lookup tensor for RoPE.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_rope_fwd(cf_math *out, const cf_math *x, const cf_math *cos_table, const cf_math *sin_table, cf_math_cuda_context *ctx);
/**
 * @brief Apply rotary position embedding backward inverse rotation.
 * @param dx Output input-gradient tensor.
 * @param dy Incoming output-gradient tensor.
 * @param cos_table Cosine lookup tensor for RoPE.
 * @param sin_table Sine lookup tensor for RoPE.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_rope_bwd(cf_math *dx, const cf_math *dy, const cf_math *cos_table, const cf_math *sin_table, cf_math_cuda_context *ctx);
/**
 * @brief Fill a tensor with a causal attention mask.
 * @param out Output tensor to write or allocate.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_causal_mask(cf_math *out, cf_math_cuda_context *ctx);

/**
 * @brief Apply dropout forward and store reusable mask state.
 * @param out Output tensor to write or allocate.
 * @param state Layer or RNG state object.
 * @param x Input value or tensor.
 * @param p Function-specific probability, convolution parameters, or byte operand.
 * @param seed Deterministic random seed.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_dropout_fwd(cf_math *out, cf_math_dropout_state *state, const cf_math *x, double p, cf_u64 seed, cf_math_cuda_context *ctx);
/**
 * @brief Apply dropout backward using saved mask state.
 * @param dx Output input-gradient tensor.
 * @param state Layer or RNG state object.
 * @param dy Incoming output-gradient tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_dropout_bwd(cf_math *dx, const cf_math_dropout_state *state, const cf_math *dy, cf_math_cuda_context *ctx);
/**
 * @brief Configure dropout probability for training or inference.
 * @param state Layer or RNG state object.
 * @param p Function-specific probability, convolution parameters, or byte operand.
 * @param training training parameter.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_dropout_train_set(cf_math_dropout_state *state, double p, cf_bool training, cf_math_cuda_context *ctx);

/**
 * @brief Gather embedding rows by index.
 * @param out Output tensor to write or allocate.
 * @param w Weight or parameter tensor.
 * @param idx Input index tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_embed_fwd(cf_math *out, const cf_math *w, const cf_math *idx, cf_math_cuda_context *ctx);
/**
 * @brief Accumulate embedding gradients by index.
 * @param dW Output weight-gradient tensor.
 * @param idx Input index tensor.
 * @param dy Incoming output-gradient tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_embed_bwd(cf_math *dW, const cf_math *idx, const cf_math *dy, cf_math_cuda_context *ctx);
/**
 * @brief Accumulate embedding gradients through the atomic-style public surface.
 * @param dW Output weight-gradient tensor.
 * @param idx Input index tensor.
 * @param dy Incoming output-gradient tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_embed_bwd_atomic(cf_math *dW, const cf_math *idx, const cf_math *dy, cf_math_cuda_context *ctx);

/**
 * @brief Run the RNN training forward surface.
 * @param out Output tensor to write or allocate.
 * @param state Layer or RNG state object.
 * @param x Input value or tensor.
 * @param h0 Initial hidden-state tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_rnn_fwd_train(cf_math *out, cf_math_rnn_state *state, const cf_math *x, const cf_math *h0, cf_math_cuda_context *ctx);
/**
 * @brief Run the RNN inference forward surface.
 * @param out Output tensor to write or allocate.
 * @param state Layer or RNG state object.
 * @param x Input value or tensor.
 * @param h Hidden-state tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_rnn_fwd_infer(cf_math *out, cf_math_rnn_state *state, const cf_math *x, const cf_math *h, cf_math_cuda_context *ctx);
/**
 * @brief Run the RNN backward-data surface.
 * @param dx Output input-gradient tensor.
 * @param dh Output hidden-state gradient tensor.
 * @param state Layer or RNG state object.
 * @param dL Incoming loss-gradient tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_rnn_bwd_data(cf_math *dx, cf_math *dh, cf_math_rnn_state *state, const cf_math *dL, cf_math_cuda_context *ctx);
/**
 * @brief Run the RNN backward-weights surface.
 * @param dW Output weight-gradient tensor.
 * @param state Layer or RNG state object.
 * @param x Input value or tensor.
 * @param dL Incoming loss-gradient tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_rnn_bwd_weights(cf_math *dW, cf_math_rnn_state *state, const cf_math *x, const cf_math *dL, cf_math_cuda_context *ctx);
/**
 * @brief Run the LSTM training forward surface.
 * @param out Output tensor to write or allocate.
 * @param state Layer or RNG state object.
 * @param x Input value or tensor.
 * @param h0 Initial hidden-state tensor.
 * @param c0 Initial LSTM cell-state tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_lstm_fwd_train(cf_math *out, cf_math_rnn_state *state, const cf_math *x, const cf_math *h0, const cf_math *c0, cf_math_cuda_context *ctx);
/**
 * @brief Run the LSTM backward-data surface.
 * @param dx Output input-gradient tensor.
 * @param dh Output hidden-state gradient tensor.
 * @param dc Output LSTM cell-state gradient tensor.
 * @param state Layer or RNG state object.
 * @param dL Incoming loss-gradient tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_lstm_bwd_data(cf_math *dx, cf_math *dh, cf_math *dc, cf_math_rnn_state *state, const cf_math *dL, cf_math_cuda_context *ctx);
/**
 * @brief Run the GRU training forward surface.
 * @param out Output tensor to write or allocate.
 * @param state Layer or RNG state object.
 * @param x Input value or tensor.
 * @param h0 Initial hidden-state tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_gru_fwd_train(cf_math *out, cf_math_rnn_state *state, const cf_math *x, const cf_math *h0, cf_math_cuda_context *ctx);

/**
 * @brief Multiply a CSR sparse matrix by a dense vector.
 * @param out Output tensor to write or allocate.
 * @param a First input value, matrix, or sparse matrix.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_spmv(cf_math *out, const cf_math_sparse *a, const cf_math *x, cf_math_cuda_context *ctx);
/**
 * @brief Multiply a CSR sparse matrix by a dense matrix.
 * @param out Output tensor to write or allocate.
 * @param a First input value, matrix, or sparse matrix.
 * @param b Second input value, tensor, bias tensor, or dense matrix.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_spmm(cf_math *out, const cf_math_sparse *a, const cf_math *b, cf_math_cuda_context *ctx);
/**
 * @brief Multiply two CSR sparse matrices.
 * @param out Output tensor to write or allocate.
 * @param a First input value, matrix, or sparse matrix.
 * @param b Second input value, tensor, bias tensor, or dense matrix.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_spgemm(cf_math_sparse *out, const cf_math_sparse *a, const cf_math_sparse *b, cf_math_cuda_context *ctx);
/**
 * @brief Convert a dense matrix to CSR sparse storage.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param threshold Sparse conversion threshold.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_dense_to_csr(cf_math_sparse *out, const cf_math *x, double threshold, cf_math_cuda_context *ctx);
/**
 * @brief Convert CSR sparse storage to a dense matrix.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_csr_to_dense(cf_math *out, const cf_math_sparse *x, cf_math_cuda_context *ctx);
/**
 * @brief Apply a sparse attention matrix to a dense value tensor.
 * @param out Output tensor to write or allocate.
 * @param a First input value, matrix, or sparse matrix.
 * @param v Value tensor, optimizer velocity tensor, or dense value tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_sparse_attn(cf_math *out, const cf_math_sparse *a, const cf_math *v, cf_math_cuda_context *ctx);

/**
 * @brief Apply one SGD optimizer step.
 * @param w Weight or parameter tensor.
 * @param g Gradient tensor.
 * @param lr Learning rate.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_sgd_step(cf_math *w, const cf_math *g, double lr, cf_math_cuda_context *ctx);
/**
 * @brief Apply one SGD optimizer step with momentum.
 * @param w Weight or parameter tensor.
 * @param v Value tensor, optimizer velocity tensor, or dense value tensor.
 * @param g Gradient tensor.
 * @param lr Learning rate.
 * @param momentum Momentum coefficient.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_sgd_momentum(cf_math *w, cf_math *v, const cf_math *g, double lr, double momentum, cf_math_cuda_context *ctx);
/**
 * @brief Apply one Adam optimizer step.
 * @param w Weight or parameter tensor.
 * @param m Adam first-moment tensor.
 * @param v Value tensor, optimizer velocity tensor, or dense value tensor.
 * @param g Gradient tensor.
 * @param lr Learning rate.
 * @param beta1 Adam first-moment decay.
 * @param beta2 Adam second-moment decay.
 * @param eps Numerical stability epsilon.
 * @param step step parameter.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_adam_step(cf_math *w, cf_math *m, cf_math *v, const cf_math *g, double lr, double beta1, double beta2, double eps, cf_u64 step, cf_math_cuda_context *ctx);
/**
 * @brief Apply one AdamW optimizer step.
 * @param w Weight or parameter tensor.
 * @param m Adam first-moment tensor.
 * @param v Value tensor, optimizer velocity tensor, or dense value tensor.
 * @param g Gradient tensor.
 * @param lr Learning rate.
 * @param beta1 Adam first-moment decay.
 * @param beta2 Adam second-moment decay.
 * @param eps Numerical stability epsilon.
 * @param decay Weight-decay coefficient.
 * @param step step parameter.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_adamw_step(cf_math *w, cf_math *m, cf_math *v, const cf_math *g, double lr, double beta1, double beta2, double eps, double decay, cf_u64 step, cf_math_cuda_context *ctx);
/**
 * @brief Apply one RMSProp optimizer step.
 * @param w Weight or parameter tensor.
 * @param v Value tensor, optimizer velocity tensor, or dense value tensor.
 * @param g Gradient tensor.
 * @param lr Learning rate.
 * @param beta Normalization offset tensor or scalar beta coefficient.
 * @param eps Numerical stability epsilon.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_rmsprop_step(cf_math *w, cf_math *v, const cf_math *g, double lr, double beta, double eps, cf_math_cuda_context *ctx);
/**
 * @brief Clip a gradient tensor by L2 norm.
 * @param g Gradient tensor.
 * @param max_norm Maximum allowed L2 norm.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_grad_clip_norm(cf_math *g, double max_norm, cf_math_cuda_context *ctx);
/**
 * @brief Clip each gradient value to a scalar range.
 * @param g Gradient tensor.
 * @param clip Maximum absolute gradient value.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_grad_clip_value(cf_math *g, double clip, cf_math_cuda_context *ctx);
/**
 * @brief Add weight decay contribution to gradients.
 * @param g Gradient tensor.
 * @param w Weight or parameter tensor.
 * @param decay Weight-decay coefficient.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_weight_decay(cf_math *g, const cf_math *w, double decay, cf_math_cuda_context *ctx);
/**
 * @brief Scale a gradient tensor by a learning-rate factor.
 * @param g Gradient tensor.
 * @param scale Scalar multiplier or attention scale.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_lr_scale(cf_math *g, double scale, cf_math_cuda_context *ctx);
/**
 * @brief All-reduce gradients across workers when backend support exists.
 * @param g Gradient tensor.
 * @param world_size Number of workers participating in all-reduce.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_grad_allreduce(cf_math *g, cf_usize world_size, cf_math_cuda_context *ctx);
/**
 * @brief Zero a gradient tensor.
 * @param g Gradient tensor.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_grad_zero(cf_math *g, cf_math_cuda_context *ctx);

/**
 * @brief Create a zero-copy tensor view with a new shape.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param dim Shape dimension array.
 * @param rank Number of active dimensions.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_reshape(cf_math *out, const cf_math *x, const cf_usize dim[CF_MATH_HIGHEST_RANK], cf_usize rank);
/**
 * @brief Permute tensor axes into an output tensor.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param axes axes parameter.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_permute(cf_math *out, const cf_math *x, const cf_usize axes[CF_MATH_HIGHEST_RANK], cf_math_cuda_context *ctx);
/**
 * @brief Create a view with size-one dimensions removed.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_squeeze(cf_math *out, const cf_math *x);
/**
 * @brief Create a view with a size-one dimension inserted.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param axis Axis index used by the operation.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_unsqueeze(cf_math *out, const cf_math *x, cf_usize axis);
/**
 * @brief Create a broadcast view with expanded shape metadata.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param dim Shape dimension array.
 * @param rank Number of active dimensions.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_expand(cf_math *out, const cf_math *x, const cf_usize dim[CF_MATH_HIGHEST_RANK], cf_usize rank);
/**
 * @brief Concatenate tensors along an axis.
 * @param out Output tensor to write or allocate.
 * @param xs Input tensor pointer array.
 * @param count Number of tensors or outputs.
 * @param axis Axis index used by the operation.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_concat(cf_math *out, const cf_math **xs, cf_usize count, cf_usize axis, cf_math_cuda_context *ctx);
/**
 * @brief Split a tensor into equal zero-copy views along an axis.
 * @param outs Output tensor array.
 * @param count Number of tensors or outputs.
 * @param x Input value or tensor.
 * @param axis Axis index used by the operation.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_split(cf_math *outs, cf_usize count, const cf_math *x, cf_usize axis);
/**
 * @brief Create a zero-copy slice view.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param start Slice start-coordinate array or optimizer step.
 * @param len Slice length array.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_slice(cf_math *out, const cf_math *x, const cf_usize start[CF_MATH_HIGHEST_RANK], const cf_usize len[CF_MATH_HIGHEST_RANK]);
/**
 * @brief Copy a tensor into a zero-padded output tensor.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param before Padding before each axis.
 * @param after Padding after each axis.
 * @param ctx Optional CUDA context; pass CF_NULL for CPU/reference paths.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_pad(cf_math *out, const cf_math *x, const cf_usize before[CF_MATH_HIGHEST_RANK], const cf_usize after[CF_MATH_HIGHEST_RANK], cf_math_cuda_context *ctx);
/**
 * @brief Create a view with a contiguous axis range flattened.
 * @param out Output tensor to write or allocate.
 * @param x Input value or tensor.
 * @param start_axis First axis included in flattening.
 * @param end_axis Last axis included in flattening.
 * @return `CF_OK` on success, or a `cf_status` error code on failure.
 */
cf_status cf_math_flatten(cf_math *out, const cf_math *x, cf_usize start_axis, cf_usize end_axis);

#ifdef __cplusplus
}
#endif

#endif /* CF_MATH_H */
