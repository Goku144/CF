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

#include <string.h>

#if(CF_MATH_USE_DNNL == 1)
static cudnnDataType_t cf_math_cudnn_dtype(cf_math_dtype dtype)
{
  switch (dtype)
  {
    case CF_MATH_DTYPE_I8: return CUDNN_DATA_INT8;
    case CF_MATH_DTYPE_U8: return CUDNN_DATA_UINT8;
    case CF_MATH_DTYPE_I32: return CUDNN_DATA_INT32;
    case CF_MATH_DTYPE_BF16: return CUDNN_DATA_BFLOAT16;
    case CF_MATH_DTYPE_F16: return CUDNN_DATA_HALF;
    case CF_MATH_DTYPE_F32: return CUDNN_DATA_FLOAT;
    case CF_MATH_DTYPE_F64: return CUDNN_DATA_DOUBLE;
    default: return CUDNN_DATA_FLOAT;
  }
}
#endif

static cudaDataType_t cf_math_cuda_dtype(cf_math_dtype dtype)
{
  switch (dtype)
  {
    case CF_MATH_DTYPE_BOOL: return CUDA_R_8U;
    case CF_MATH_DTYPE_I8: return CUDA_R_8I;
    case CF_MATH_DTYPE_U8: return CUDA_R_8U;
    case CF_MATH_DTYPE_I32: return CUDA_R_32I;
    case CF_MATH_DTYPE_BF16: return CUDA_R_16BF;
    case CF_MATH_DTYPE_F16: return CUDA_R_16F;
    case CF_MATH_DTYPE_F32: return CUDA_R_32F;
    case CF_MATH_DTYPE_F64: return CUDA_R_64F;
    default: return CUDA_R_32F;
  }
}

static cudnnDataType_t cf_math_cudnn_dtype(cf_math_dtype dtype)
{
  switch (dtype)
  {
    case CF_MATH_DTYPE_I8:   return CUDNN_DATA_INT8;
    case CF_MATH_DTYPE_I32:  return CUDNN_DATA_INT32;
    case CF_MATH_DTYPE_BF16: return CUDNN_DATA_BFLOAT16;
    case CF_MATH_DTYPE_F16:  return CUDNN_DATA_HALF;
    case CF_MATH_DTYPE_F32:  return CUDNN_DATA_FLOAT;
    case CF_MATH_DTYPE_F64:  return CUDNN_DATA_DOUBLE;
    default: return CUDNN_DATA_FLOAT;
  }
}

static cf_usize cf_math_dtype_size(cf_math_dtype dtype)
{
  switch (dtype)
  {
    case CF_MATH_DTYPE_BOOL: return sizeof(cf_bool);
    case CF_MATH_DTYPE_I8: return sizeof(cf_i8);
    case CF_MATH_DTYPE_U8: return sizeof(cf_u8);
    case CF_MATH_DTYPE_I32: return sizeof(cf_i32);
    case CF_MATH_DTYPE_FP8E5M2: return sizeof(cf_u8);
    case CF_MATH_DTYPE_FP8E4M3: return sizeof(cf_u8);
    case CF_MATH_DTYPE_BF16: return sizeof(cf_u16);
    case CF_MATH_DTYPE_F16: return sizeof(cf_u16);
    case CF_MATH_DTYPE_F32: return sizeof(float);
    case CF_MATH_DTYPE_F64: return sizeof(double);
  }
  return 0;
}

#if(CF_MATH_USE_DNNL == 1)
static dnnl_data_type_t cf_math_dnnl_dtype(cf_math_dtype dtype)
{
  switch (dtype)
  {
    case CF_MATH_DTYPE_I8: return dnnl_s8;
    case CF_MATH_DTYPE_U8: return dnnl_u8;
    case CF_MATH_DTYPE_I32: return dnnl_s32;
    case CF_MATH_DTYPE_BF16: return dnnl_bf16;
    case CF_MATH_DTYPE_F16: return dnnl_f16;
    case CF_MATH_DTYPE_F32: return dnnl_f32;
    case CF_MATH_DTYPE_F64: return dnnl_f64;
    default: return dnnl_f32;
  }
}
#endif

static cf_status cf_math_cublaslt_desc_create(cf_math_cublaslt_desc *desc, int rank, const int *dim, const int *strides, cf_math_dtype dtype)
{
  if (desc == CF_NULL || dim == CF_NULL || strides == CF_NULL) return CF_ERR_NULL;

  *desc = (cf_math_cublaslt_desc) {0};

  if (rank < 2) return CF_OK;

  cublasStatus_t state;
  cudaDataType_t cuda_dtype = cf_math_cuda_dtype(dtype);

  int64_t rows = dim[rank - 2];
  int64_t cols = dim[rank - 1];
  int64_t ld = strides[rank - 2];

  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  cublasOperation_t trans = CUBLAS_OP_N;

  state = cublasLtMatrixLayoutCreate(&desc->layout, cuda_dtype, rows, cols, ld);
  if (state != CUBLAS_STATUS_SUCCESS) goto fail;

  state = cublasLtMatrixLayoutSetAttribute(desc->layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (state != CUBLAS_STATUS_SUCCESS) goto fail;

  state = cublasLtMatmulDescCreate( &desc->op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (state != CUBLAS_STATUS_SUCCESS) goto fail;

  state = cublasLtMatmulDescSetAttribute( desc->op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans));
  if (state != CUBLAS_STATUS_SUCCESS) goto fail;

  state = cublasLtMatmulDescSetAttribute( desc->op, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans));
  if (state != CUBLAS_STATUS_SUCCESS) goto fail;

  state = cublasLtMatmulPreferenceCreate(&desc->preference);
  if (state != CUBLAS_STATUS_SUCCESS) goto fail;

  return CF_OK;

fail:
  if (desc->preference != CF_NULL)
    cublasLtMatmulPreferenceDestroy(desc->preference);

  if (desc->op != CF_NULL)
    cublasLtMatmulDescDestroy(desc->op);

  if (desc->layout != CF_NULL)
    cublasLtMatrixLayoutDestroy(desc->layout);

  *desc = (cf_math_cublaslt_desc) {0};
  return CF_ERR_CUDA;
}


static cf_status cf_math_cudnn_desc_create(cf_math_cudnn_desc *desc, int rank, const int *dim, const int *strides, cf_math_dtype dtype)
{
  if (desc == CF_NULL || dim == CF_NULL || strides == CF_NULL) return CF_ERR_NULL;

  *desc = (cf_math_cudnn_desc) {0};

  cudnnStatus_t state;
  cudnnDataType_t cudnn_dtype = cf_math_cudnn_dtype(dtype);

  state = cudnnCreateTensorDescriptor(&desc->tensor);
  if (state != CUDNN_STATUS_SUCCESS) goto fail;

  state = cudnnSetTensorNdDescriptor(desc->tensor, cudnn_dtype, rank, dim, strides);
  if (state != CUDNN_STATUS_SUCCESS) goto fail;

  state = cudnnCreateFilterDescriptor(&desc->filter);
  if (state != CUDNN_STATUS_SUCCESS) goto fail;

  if (rank == 4 || rank == 5)
  {
    state = cudnnSetFilterNdDescriptor(desc->filter, cudnn_dtype, CUDNN_TENSOR_NCHW, rank, dim);
    if (state != CUDNN_STATUS_SUCCESS) goto fail;
  }

  state = cudnnCreateConvolutionDescriptor(&desc->conv);
  if (state != CUDNN_STATUS_SUCCESS) goto fail;

  state = cudnnCreateActivationDescriptor(&desc->activation);
  if (state != CUDNN_STATUS_SUCCESS) goto fail;

  state = cudnnCreatePoolingDescriptor(&desc->pooling);
  if (state != CUDNN_STATUS_SUCCESS) goto fail;

  state = cudnnCreateReduceTensorDescriptor(&desc->reduce);
  if (state != CUDNN_STATUS_SUCCESS) goto fail;

  state = cudnnCreateOpTensorDescriptor(&desc->opTensor);
  if (state != CUDNN_STATUS_SUCCESS) goto fail;

  return CF_OK;

fail:
  if (desc->opTensor != CF_NULL)
    cudnnDestroyOpTensorDescriptor(desc->opTensor);

  if (desc->reduce != CF_NULL)
    cudnnDestroyReduceTensorDescriptor(desc->reduce);

  if (desc->pooling != CF_NULL)
    cudnnDestroyPoolingDescriptor(desc->pooling);

  if (desc->activation != CF_NULL)
    cudnnDestroyActivationDescriptor(desc->activation);

  if (desc->conv != CF_NULL)
    cudnnDestroyConvolutionDescriptor(desc->conv);

  if (desc->filter != CF_NULL)
    cudnnDestroyFilterDescriptor(desc->filter);

  if (desc->tensor != CF_NULL)
    cudnnDestroyTensorDescriptor(desc->tensor);

  *desc = (cf_math_cudnn_desc) {0};
  return CF_ERR_CUDA;
}

static void cf_math_cublaslt_desc_destroy(cf_math_cublaslt_desc *desc)
{
  if (desc == CF_NULL) return;

  if (desc->preference != CF_NULL) cublasLtMatmulPreferenceDestroy(desc->preference);

  if (desc->op != CF_NULL) cublasLtMatmulDescDestroy(desc->op);

  if (desc->layout != CF_NULL) cublasLtMatrixLayoutDestroy(desc->layout);

  *desc = (cf_math_cublaslt_desc) {0};
}

static void cf_math_cudnn_desc_destroy(cf_math_cudnn_desc *desc)
{
  if (desc == CF_NULL) return;

  if (desc->opTensor != CF_NULL) cudnnDestroyOpTensorDescriptor(desc->opTensor);

  if (desc->reduce != CF_NULL) cudnnDestroyReduceTensorDescriptor(desc->reduce);

  if (desc->pooling != CF_NULL) cudnnDestroyPoolingDescriptor(desc->pooling);

  if (desc->activation != CF_NULL) cudnnDestroyActivationDescriptor(desc->activation);

  if (desc->conv != CF_NULL) cudnnDestroyConvolutionDescriptor(desc->conv);

  if (desc->filter != CF_NULL) cudnnDestroyFilterDescriptor(desc->filter);

  if (desc->tensor != CF_NULL) cudnnDestroyTensorDescriptor(desc->tensor);

  *desc = (cf_math_cudnn_desc) {0};
}

cf_status cf_math_desc_create(cf_math_desc *desc, int rank, const int *dim, cf_math_dtype dtype)
{
  if(desc == CF_NULL || dim == CF_NULL) return CF_ERR_NULL;
  if(rank <= 0 || rank > CF_MATH_MAX_RANK) return CF_ERR_INVALID;

  cf_status state = CF_OK;
  *desc = {0};

  desc->rank = rank;
  desc->dtype = dtype;

  for (int i = 0; i < rank; i++) desc->dim[i] = dim[i];

  desc->strides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; i--) desc->strides[i] = desc->strides[i + 1] * desc->dim[i + 1];

  state = cf_math_cublaslt_desc_create(&desc->cublastlt, rank, desc->dim, desc->strides, dtype);
  if (state != CF_OK) goto fail;

  state = cf_math_cudnn_desc_create(&desc->cudnn, rank, desc->dim, desc->strides, dtype);
  if (state != CF_OK) goto fail;

  return CF_OK;

fail:
  cf_math_desc_destroy(desc);
  return state;
}

void cf_math_desc_destroy(cf_math_desc *desc)
{
  if(desc == CF_NULL) return;
  cf_math_cublaslt_desc_destroy(&desc->cublastlt);
  cf_math_cudnn_desc_destroy(&desc->cudnn);
  *desc = (cf_math_desc) {0};
}

cf_status cf_math_bind(cf_math_handle *handle, cf_math *math, cf_math_desc *desc)
{
  if(handle == CF_NULL || math == CF_NULL || desc == CF_NULL) return CF_ERR_NULL;

  *math = (cf_math) {0};
  math->desc = desc;

  cf_status state = cf_math_handle_add(handle, math);
  if(state != CF_OK) *math = (cf_math) {0};
  return state;
}

cf_status cf_math_rebind(cf_math_handle *handle, cf_math *math, cf_math_desc *desc)
{
  return cf_math_bind(handle, math, desc);
}

cf_status cf_math_copy_to_host(cf_math_handle *handle, const cf_math *src, void *dst, cf_usize dst_bytes)
{
  if(handle == CF_NULL || src == CF_NULL || src->desc == CF_NULL || dst == CF_NULL) return CF_ERR_NULL;

  cf_usize elem_size = cf_math_dtype_size(src->desc->dtype);
  if(elem_size == 0) return CF_ERR_INVALID;
  if(src->elem_len > SIZE_MAX / elem_size) return CF_ERR_OVERFLOW;

  cf_usize bytes = src->elem_len * elem_size;
  if(dst_bytes < bytes) return CF_ERR_BOUNDS;

  const cf_u8 *src_ptr = (const cf_u8 *)handle->storage.backend + src->byte_offset;

  switch(handle->device)
  {
    case CF_MATH_DEVICE_CPU:
      memcpy(dst, src_ptr, bytes);
      return CF_OK;

    case CF_MATH_DEVICE_CUDA:
      if(cudaMemcpyAsync(dst, src_ptr, bytes, cudaMemcpyDeviceToHost, handle->workspace->stream) != cudaSuccess) return CF_ERR_CUDA_COPY;
      if(cudaStreamSynchronize(handle->workspace->stream) != cudaSuccess) return CF_ERR_CUDA;
      return CF_OK;
  }

  return CF_ERR_INVALID;
}
