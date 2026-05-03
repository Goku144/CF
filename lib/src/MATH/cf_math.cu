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

cf_status cf_math_desc_create(cf_math_desc *desc, int rank, const int *dim, cf_math_dtype dtype, cf_math_desc_type desc_type)
{
  if(desc == CF_NULL || dim == CF_NULL) return CF_ERR_NULL;
  cf_status state = CF_OK;
  *desc = {0};

  desc->rank = rank;
  desc->dtype = dtype;
  desc->desc_type = desc_type;

  for (int i = 0; i < rank; i++)
  {
    desc->dim[i] = dim[i];
  }
  desc->strides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; i--)
    desc->strides[i] = desc->strides[i + 1] * desc->dim[i + 1];

  switch (desc_type)
  {
    case CF_MATH_DESC_NONE:
    break;

    case CF_MATH_DESC_CUDNN:
      if(cudnnCreateTensorDescriptor(&desc->desc.cudnn_tensor) != CUDNN_STATUS_SUCCESS) { state = CF_ERR_CUDA; goto fail; }
      if(cudnnSetTensorNdDescriptor(desc->desc.cudnn_tensor, cf_math_cudnn_dtype(dtype), rank, desc->dim, desc->strides) != CUDNN_STATUS_SUCCESS) { state = CF_ERR_CUDA; goto fail; }
    break;

    case CF_MATH_DESC_LT:
      if(cublasLtMatrixLayoutCreate(&desc->desc.lt_layout, cf_math_cuda_dtype(dtype), dim[0], dim[1], desc->strides[0]) != CUBLAS_STATUS_SUCCESS) { state = CF_ERR_CUDA; goto fail; }
    break;

    case CF_MATH_DESC_DNNL:
    {
      dnnl_dims_t dnnl_dim = {0};
      dnnl_dims_t dnnl_strides = {0};
      for (int i = 0; i < rank; i++)
      {
        dnnl_dim[i] = (dnnl_dim_t) dim[i];
        dnnl_strides[i] = (dnnl_dim_t) desc->strides[i];
      }
      if(dnnl_memory_desc_create_with_strides(&desc->desc.dnnl_desc, rank, dnnl_dim, cf_math_dnnl_dtype(dtype), dnnl_strides) != dnnl_success) { state = CF_ERR_INTERNAL; goto fail; }
    }
    break;

    default: state = CF_ERR_INVALID;
  }

  return state;

fail:
  cf_math_desc_destroy(desc);
  return state;
}

void cf_math_desc_destroy(cf_math_desc *desc)
{
  if(desc == CF_NULL) return;

  switch (desc->desc_type)
  {
    case CF_MATH_DESC_CUDNN:
      if(desc->desc.cudnn_tensor != CF_NULL) cudnnDestroyTensorDescriptor(desc->desc.cudnn_tensor);
    break;

    case CF_MATH_DESC_LT:
      if(desc->desc.lt_layout != CF_NULL) cublasLtMatrixLayoutDestroy(desc->desc.lt_layout);
    break;

    case CF_MATH_DESC_DNNL:
      if(desc->desc.dnnl_desc != CF_NULL) dnnl_memory_desc_destroy(desc->desc.dnnl_desc);
    break;

    default:
    break;
  }

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

void cf_math_unbind(cf_math *math)
{
  if(math == CF_NULL) return;
  *math = (cf_math) {0};
}
