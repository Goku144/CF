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

#if !defined(CF_TENSOR_H)
#define CF_TENSOR_H

#include "RUNTIME/cf_status.h"
#include "RUNTIME/cf_types.h"

#define CF_TENSOR_HIGHEST_RANK 8

/**
 * Storage backend that owns the newest tensor values.
 */
typedef enum cf_tensor_device
{
  CF_TENSOR_DEVICE_CPU = 0,
  CF_TENSOR_DEVICE_CUDA,
} cf_tensor_device;

/**
 * Element type stored by a tensor.
 */
typedef enum cf_tensor_type
{
  CF_TENSOR_CHAR = 0,
  CF_TENSOR_SHORT,
  CF_TENSOR_INT,
  CF_TENSOR_LONG,
  CF_TENSOR_LL,
  CF_TENSOR_FLOAT,
  CF_TENSOR_DOUBLE,
  CF_TENSOR_LD,
  CF_TENSOR_U8,
  CF_TENSOR_U16,
  CF_TENSOR_U32,
  CF_TENSOR_U64,
  CF_TENSOR_U128,
} cf_tensor_type;

/**
 * Dense tensor metadata.
 *
 * `len` is the active element count. `capacity` is the allocated element count
 * available for future reshapes/resizes. `stride` is row-major and expressed in
 * elements, not bytes.
 */
typedef struct cf_tensor_metadata
{
  cf_usize len;
  cf_usize capacity;
  cf_usize stride[CF_TENSOR_HIGHEST_RANK];
  cf_usize elem_size;
  cf_tensor_type elem_type;
} cf_tensor_metadata;

/**
 * Dense tensor with optional CPU and CUDA storage mirrors.
 *
 * Math operations mutate `op1` directly. The caller owns shape/type/device
 * compatibility for hot operations; helper functions validate slower setup and
 * data movement paths.
 */
typedef struct cf_tensor
{
  void *data;
  void *device_data;
  void *backend_cache;
  cf_usize dim[CF_TENSOR_HIGHEST_RANK];
  cf_usize rank;
  cf_tensor_device device;
  cf_tensor_metadata metadata;
} cf_tensor;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Return the byte size of one tensor element.
 * @param elem_type Tensor element type.
 * @return Element byte size, or 0 when elem_type is unknown.
 */
cf_usize cf_tensor_element_size(cf_tensor_type elem_type);

/**
 * @brief Validate tensor metadata and active storage.
 * @param tensor Tensor to inspect.
 * @return CF_TRUE when valid, otherwise CF_FALSE.
 */
cf_bool cf_tensor_is_valid(const cf_tensor *tensor);

/**
 * @brief Initialize one zeroed CPU tensor.
 * @param tensor Tensor object to initialize.
 * @param dim Shape dimensions.
 * @param rank Number of active dimensions.
 * @param elem_type Element type.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_init_cpu(cf_tensor *tensor, const cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank, cf_tensor_type elem_type);

/**
 * @brief Initialize many CPU tensors with the same shape and element type.
 * @param tensors Array of tensor pointers.
 * @param count Number of tensors in the array.
 * @param dim Shape dimensions.
 * @param rank Number of active dimensions.
 * @param elem_type Element type.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_init_many_cpu(cf_tensor **tensors, cf_usize count, const cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank, cf_tensor_type elem_type);

/**
 * @brief Release CPU storage and reset the tensor object.
 * @param tensor Tensor to destroy.
 */
void cf_tensor_destroy_cpu(cf_tensor *tensor);

/**
 * @brief Destroy many CPU tensors from an array of tensor pointers.
 * @param tensors Array of tensor pointers.
 * @param count Number of tensors in the array.
 */
void cf_tensor_destroy_many_cpu(cf_tensor **tensors, cf_usize count);

/**
 * @brief Reserve CPU capacity without changing the active shape.
 * @param tensor Tensor whose capacity should grow.
 * @param capacity Minimum element capacity.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_reserve_cpu(cf_tensor *tensor, cf_usize capacity);

/**
 * @brief Change CPU tensor shape without allocation.
 * @param tensor Tensor to reshape.
 * @param dim New shape dimensions.
 * @param rank New rank.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_reshape_cpu(cf_tensor *tensor, const cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank);

/**
 * @brief Change CPU tensor shape, growing storage when needed.
 * @param tensor Tensor to resize.
 * @param dim New shape dimensions.
 * @param rank New rank.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_resize_cpu(cf_tensor *tensor, const cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank);

/**
 * @brief Copy data and shape from one CPU tensor to another.
 * @param dst Destination tensor.
 * @param src Source tensor.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_copy_cpu(cf_tensor *dst, const cf_tensor *src);

/**
 * @brief Copy a plain array into a CPU tensor as a rank-1 vector.
 * @param tensor Destination tensor.
 * @param array Source array.
 * @param count Number of elements to copy.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_copy_from_array_cpu(cf_tensor *tensor, const void *array, cf_usize count);

/**
 * @brief Copy CPU tensor data into a plain array.
 * @param array Destination array.
 * @param tensor Source tensor.
 * @param count Destination capacity in elements.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_copy_to_array_cpu(void *array, const cf_tensor *tensor, cf_usize count);

/**
 * @brief Read one CPU tensor element by logical indices.
 * @param out_value Destination value pointer.
 * @param tensor Source tensor.
 * @param indexs Logical coordinates.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_get_cpu(void *out_value, const cf_tensor *tensor, const cf_usize indexs[CF_TENSOR_HIGHEST_RANK]);

/**
 * @brief Write one CPU tensor element by logical indices.
 * @param tensor Destination tensor.
 * @param indexs Logical coordinates.
 * @param value Source value pointer.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_set_cpu(cf_tensor *tensor, const cf_usize indexs[CF_TENSOR_HIGHEST_RANK], const void *value);

/**
 * @brief Print a CPU-visible tensor in nested row-major form.
 * @param tensor Tensor to print.
 */
void cf_tensor_print(const cf_tensor *tensor);

/**
 * @brief In-place CPU elementwise addition.
 * @param op1 Left operand and destination tensor.
 * @param op2 Right operand tensor.
 * @note Hot-path preconditions are caller-owned.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_add_cpu(cf_tensor *op1, const cf_tensor *op2);

/**
 * @brief In-place CPU elementwise multiplication.
 * @param op1 Left operand and destination tensor.
 * @param op2 Right operand tensor.
 * @note Hot-path preconditions are caller-owned.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_mul_cpu(cf_tensor *op1, const cf_tensor *op2);

/**
 * @brief In-place CPU scalar multiplication.
 * @param op1 Operand and destination tensor.
 * @param scalar Pointer to one value of the same element type as op1.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_scalar_mul_cpu(cf_tensor *op1, const void *scalar);

/**
 * @brief In-place CPU batched matrix multiplication.
 * @param op1 Left operand and destination tensor.
 * @param op2 Right operand tensor.
 * @details Supports `[..., M, K] @ [..., K, N] -> [..., M, N]`.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_batch_mul_cpu(cf_tensor *op1, const cf_tensor *op2);

/**
 * @brief In-place CPU matrix multiplication.
 * @param op1 Left operand and destination tensor.
 * @param op2 Right operand tensor.
 * @details This calls cf_tensor_batch_mul_cpu.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_matrix_mul_cpu(cf_tensor *op1, const cf_tensor *op2);

#ifdef CF_CUDA_AVAILABLE

/**
 * @brief Initialize one zeroed CUDA tensor.
 * @param tensor Tensor object to initialize.
 * @param dim Shape dimensions.
 * @param rank Number of active dimensions.
 * @param elem_type Element type.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_init_gpu(cf_tensor *tensor, const cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank, cf_tensor_type elem_type);

/**
 * @brief Initialize many CUDA tensors with the same shape and element type.
 * @param tensors Array of tensor pointers.
 * @param count Number of tensors in the array.
 * @param dim Shape dimensions.
 * @param rank Number of active dimensions.
 * @param elem_type Element type.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_init_many_gpu(cf_tensor **tensors, cf_usize count, const cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank, cf_tensor_type elem_type);

/**
 * @brief Release CUDA and CPU mirror storage, then reset the tensor object.
 * @param tensor Tensor to destroy.
 */
void cf_tensor_destroy_gpu(cf_tensor *tensor);

/**
 * @brief Destroy many CUDA tensors from an array of tensor pointers.
 * @param tensors Array of tensor pointers.
 * @param count Number of tensors in the array.
 */
void cf_tensor_destroy_many_gpu(cf_tensor **tensors, cf_usize count);

/**
 * @brief Reserve CUDA capacity without changing the active shape.
 * @param tensor Tensor whose capacity should grow.
 * @param capacity Minimum element capacity.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_reserve_gpu(cf_tensor *tensor, cf_usize capacity);

/**
 * @brief Change CUDA tensor shape without allocation.
 * @param tensor Tensor to reshape.
 * @param dim New shape dimensions.
 * @param rank New rank.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_reshape_gpu(cf_tensor *tensor, const cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank);

/**
 * @brief Change CUDA tensor shape, growing storage when needed.
 * @param tensor Tensor to resize.
 * @param dim New shape dimensions.
 * @param rank New rank.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_resize_gpu(cf_tensor *tensor, const cf_usize dim[CF_TENSOR_HIGHEST_RANK], cf_usize rank);

/**
 * @brief Copy data and shape between CUDA tensors.
 * @param dst Destination tensor.
 * @param src Source tensor.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_copy_gpu(cf_tensor *dst, const cf_tensor *src);

/**
 * @brief Copy a host array into a CUDA tensor as a rank-1 vector.
 * @param tensor Destination tensor.
 * @param array Source array.
 * @param count Number of elements to copy.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_copy_from_array_gpu(cf_tensor *tensor, const void *array, cf_usize count);

/**
 * @brief Copy CUDA tensor data into a host array.
 * @param array Destination host array.
 * @param tensor Source CUDA tensor.
 * @param count Destination capacity in elements.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_copy_to_array_gpu(void *array, const cf_tensor *tensor, cf_usize count);

/**
 * @brief Read one CUDA tensor element into host memory.
 * @param out_value Destination host value pointer.
 * @param tensor Source CUDA tensor.
 * @param indexs Logical coordinates.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_get_gpu(void *out_value, const cf_tensor *tensor, const cf_usize indexs[CF_TENSOR_HIGHEST_RANK]);

/**
 * @brief Write one CUDA tensor element from host memory.
 * @param tensor Destination CUDA tensor.
 * @param indexs Logical coordinates.
 * @param value Source host value pointer.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_set_gpu(cf_tensor *tensor, const cf_usize indexs[CF_TENSOR_HIGHEST_RANK], const void *value);

/**
 * @brief Upload CPU storage into CUDA storage and make CUDA the active backend.
 * @param tensor Tensor to upload.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_to_gpu(cf_tensor *tensor);

/**
 * @brief Download CUDA storage into CPU storage and make CPU the active backend.
 * @param tensor Tensor to download.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_to_cpu(cf_tensor *tensor);

/**
 * @brief Free only CUDA storage, preserving CPU storage when it exists.
 * @param tensor Tensor whose CUDA storage should be released.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_free_gpu(cf_tensor *tensor);

/**
 * @brief Synchronize the current CUDA device.
 * @return CF_OK when all queued CUDA work completed, otherwise CF_ERR_CUDA_SYNC.
 */
cf_status cf_tensor_sync_gpu(void);

/**
 * @brief In-place CUDA elementwise addition.
 * @param op1 Left operand and destination tensor.
 * @param op2 Right operand tensor.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_add_gpu(cf_tensor *op1, const cf_tensor *op2);

/**
 * @brief In-place CUDA elementwise multiplication.
 * @param op1 Left operand and destination tensor.
 * @param op2 Right operand tensor.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_mul_gpu(cf_tensor *op1, const cf_tensor *op2);

/**
 * @brief In-place CUDA scalar multiplication.
 * @param op1 Operand and destination tensor.
 * @param scalar Pointer to one host value of the same element type as op1.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_scalar_mul_gpu(cf_tensor *op1, const void *scalar);

/**
 * @brief In-place CUDA batched matrix multiplication through cuBLASLt.
 * @param op1 Left operand and destination tensor.
 * @param op2 Right operand tensor.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_batch_mul_gpu(cf_tensor *op1, const cf_tensor *op2);

/**
 * @brief In-place CUDA matrix multiplication through cuBLASLt.
 * @param op1 Left operand and destination tensor.
 * @param op2 Right operand tensor.
 * @return CF_OK on success, otherwise a cf_status error.
 */
cf_status cf_tensor_matrix_mul_gpu(cf_tensor *op1, const cf_tensor *op2);

#define cf_tensor_init(tensor, dim, rank, elem_type) cf_tensor_init_gpu((tensor), (dim), (rank), (elem_type))
#define cf_tensor_init_many(tensors, count, dim, rank, elem_type) cf_tensor_init_many_gpu((tensors), (count), (dim), (rank), (elem_type))
#define cf_tensor_destroy(tensor) cf_tensor_destroy_gpu((tensor))
#define cf_tensor_destroy_many(tensors, count) cf_tensor_destroy_many_gpu((tensors), (count))
#define cf_tensor_reserve(tensor, capacity) cf_tensor_reserve_gpu((tensor), (capacity))
#define cf_tensor_reshape(tensor, dim, rank) cf_tensor_reshape_gpu((tensor), (dim), (rank))
#define cf_tensor_resize(tensor, dim, rank) cf_tensor_resize_gpu((tensor), (dim), (rank))
#define cf_tensor_copy(dst, src) cf_tensor_copy_gpu((dst), (src))
#define cf_tensor_copy_from_array(tensor, array, count) cf_tensor_copy_from_array_gpu((tensor), (array), (count))
#define cf_tensor_copy_to_array(array, tensor, count) cf_tensor_copy_to_array_gpu((array), (tensor), (count))
#define cf_tensor_get(out_value, tensor, indexs) cf_tensor_get_gpu((out_value), (tensor), (indexs))
#define cf_tensor_set(tensor, indexs, value) cf_tensor_set_gpu((tensor), (indexs), (value))
#define cf_tensor_add(op1, op2) cf_tensor_add_gpu((op1), (op2))
#define cf_tensor_mul(op1, op2) cf_tensor_mul_gpu((op1), (op2))
#define cf_tensor_scalar_mul(op1, scalar) cf_tensor_scalar_mul_gpu((op1), (scalar))
#define cf_tensor_matrix_mul(op1, op2) cf_tensor_matrix_mul_gpu((op1), (op2))
#define cf_tensor_batch_mul(op1, op2) cf_tensor_batch_mul_gpu((op1), (op2))

#else

#define cf_tensor_init(tensor, dim, rank, elem_type) cf_tensor_init_cpu((tensor), (dim), (rank), (elem_type))
#define cf_tensor_init_many(tensors, count, dim, rank, elem_type) cf_tensor_init_many_cpu((tensors), (count), (dim), (rank), (elem_type))
#define cf_tensor_destroy(tensor) cf_tensor_destroy_cpu((tensor))
#define cf_tensor_destroy_many(tensors, count) cf_tensor_destroy_many_cpu((tensors), (count))
#define cf_tensor_reserve(tensor, capacity) cf_tensor_reserve_cpu((tensor), (capacity))
#define cf_tensor_reshape(tensor, dim, rank) cf_tensor_reshape_cpu((tensor), (dim), (rank))
#define cf_tensor_resize(tensor, dim, rank) cf_tensor_resize_cpu((tensor), (dim), (rank))
#define cf_tensor_copy(dst, src) cf_tensor_copy_cpu((dst), (src))
#define cf_tensor_copy_from_array(tensor, array, count) cf_tensor_copy_from_array_cpu((tensor), (array), (count))
#define cf_tensor_copy_to_array(array, tensor, count) cf_tensor_copy_to_array_cpu((array), (tensor), (count))
#define cf_tensor_get(out_value, tensor, indexs) cf_tensor_get_cpu((out_value), (tensor), (indexs))
#define cf_tensor_set(tensor, indexs, value) cf_tensor_set_cpu((tensor), (indexs), (value))
#define cf_tensor_add(op1, op2) cf_tensor_add_cpu((op1), (op2))
#define cf_tensor_mul(op1, op2) cf_tensor_mul_cpu((op1), (op2))
#define cf_tensor_scalar_mul(op1, scalar) cf_tensor_scalar_mul_cpu((op1), (scalar))
#define cf_tensor_matrix_mul(op1, op2) cf_tensor_matrix_mul_cpu((op1), (op2))
#define cf_tensor_batch_mul(op1, op2) cf_tensor_batch_mul_cpu((op1), (op2))

#endif

#ifdef __cplusplus
}
#endif

#endif /* CF_TENSOR_H */
