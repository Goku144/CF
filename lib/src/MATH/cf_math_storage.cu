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

static cf_usize cf_math_type_size(cf_math_dtype dtype)
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

cf_status cf_math_context_create(cf_math_context *ctx, int id_or_tnum, cf_math_device device)
{
  if(ctx == CF_NULL) return CF_ERR_NULL;
  cf_status state = CF_OK;
  *ctx = {0};
  ctx->device = device;

  switch (device)
  {
    case CF_MATH_DEVICE_CPU:
      ctx->context.cpu.num_threads = id_or_tnum;
      if(dnnl_engine_create(&ctx->context.cpu.engine, dnnl_cpu, 0) != dnnl_success) { state = CF_ERR_INTERNAL; goto fail; }
      if(dnnl_stream_create(&ctx->context.cpu.stream, ctx->context.cpu.engine, dnnl_stream_default_flags) != dnnl_success) { state = CF_ERR_INTERNAL; goto fail; }
    break;

    case CF_MATH_DEVICE_CUDA:
      ctx->context.cuda.device_id = id_or_tnum;
      if(cudaSetDevice(ctx->context.cuda.device_id) != cudaSuccess) { state = CF_ERR_CUDA; goto fail; }
      if(cublasCreate(&ctx->context.cuda.cublas) != CUBLAS_STATUS_SUCCESS) { state = CF_ERR_CUDA; goto fail; }
      if(cublasLtCreate(&ctx->context.cuda.cublasLt) != CUBLAS_STATUS_SUCCESS) { state = CF_ERR_CUDA; goto fail; }
      if(cudnnCreate(&ctx->context.cuda.cudnn) != CUDNN_STATUS_SUCCESS) { state = CF_ERR_CUDA; goto fail; }
      if(curandCreateGenerator(&ctx->context.cuda.curand, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) { state = CF_ERR_CUDA; goto fail; }
      if(cusolverDnCreate(&ctx->context.cuda.cusolverDn) != CUSOLVER_STATUS_SUCCESS) { state = CF_ERR_CUDA; goto fail; }
      if(cusparseCreate(&ctx->context.cuda.cusparse) != CUSPARSE_STATUS_SUCCESS) { state = CF_ERR_CUDA; goto fail; }
    break;
    
    default: state = CF_ERR_INVALID;
  }
  return state;

fail:
  cf_math_context_destroy(ctx);
  return state;
}

void cf_math_context_destroy(cf_math_context *ctx)
{
  if(ctx == CF_NULL) return;

  switch (ctx->device)
  {
    case CF_MATH_DEVICE_CPU:
      if(ctx->context.cpu.stream != CF_NULL) dnnl_stream_destroy(ctx->context.cpu.stream);
      if(ctx->context.cpu.engine != CF_NULL) dnnl_engine_destroy(ctx->context.cpu.engine);
    break;

    case CF_MATH_DEVICE_CUDA:
      if(ctx->context.cuda.cublas != CF_NULL) cublasDestroy(ctx->context.cuda.cublas);
      if(ctx->context.cuda.cublasLt != CF_NULL) cublasLtDestroy(ctx->context.cuda.cublasLt);
      if(ctx->context.cuda.cudnn != CF_NULL) cudnnDestroy(ctx->context.cuda.cudnn);
      if(ctx->context.cuda.curand != CF_NULL) curandDestroyGenerator(ctx->context.cuda.curand);
      if(ctx->context.cuda.cusolverDn != CF_NULL) cusolverDnDestroy(ctx->context.cuda.cusolverDn);
      if(ctx->context.cuda.cusparse != CF_NULL) cusparseDestroy(ctx->context.cuda.cusparse);
    break;
  }
  *ctx = (cf_math_context) {0};
}

cf_status cf_math_workspace_create(cf_math_workspace *workspace, cf_usize capacity, cf_math_device device)
{
  if(workspace == CF_NULL) return CF_ERR_NULL;
  cf_status state = CF_OK;
  *workspace = {0};

  if(capacity == 0) return state;
  if(capacity > SIZE_MAX - 15) return CF_ERR_OVERFLOW;
  capacity = (capacity + 15) & ~(cf_usize)15;
  workspace->device = device;

  switch (device)
  {
    case CF_MATH_DEVICE_CPU:
      workspace->scratchpad = mi_malloc_aligned(capacity, 16);
      workspace->scratchpad_size = workspace->scratchpad == CF_NULL ? 0 : capacity;
      if(!workspace->scratchpad_size) state = CF_ERR_OOM;
    break;

    case CF_MATH_DEVICE_CUDA:
      if(cudaStreamCreate(&workspace->stream) != cudaSuccess) return CF_ERR_CUDA;
      workspace->scratchpad_size = cudaMallocAsync(&workspace->scratchpad, capacity, workspace->stream) != cudaSuccess ? 0 : capacity;
      if(!workspace->scratchpad_size) { state = CF_ERR_CUDA_MEMORY; goto fail; }
    break;
    
    default: state = CF_ERR_INVALID;
  }
  return state;

fail:
  cf_math_workspace_destroy(workspace);
  return state;
}

void cf_math_workspace_destroy(cf_math_workspace *workspace)
{
  if(workspace == CF_NULL) return;

  switch (workspace->device)
  {
    case CF_MATH_DEVICE_CPU:
      mi_free(workspace->scratchpad);
    break;
      
    case CF_MATH_DEVICE_CUDA:
      if(workspace->scratchpad != CF_NULL) cudaFreeAsync(workspace->scratchpad, workspace->stream);
      if(workspace->stream != CF_NULL) cudaStreamDestroy(workspace->stream);
    break;
  }
  *workspace = (cf_math_workspace) {0};
}

cf_status cf_math_handle_create(cf_math_handle *handle, cf_math_context *ctx, cf_math_workspace *workspace, cf_usize capacity, cf_math_device device)
{
  if(handle == CF_NULL) return CF_ERR_NULL;
  cf_status state = CF_OK;
  *handle = {0};
  handle->ctx = ctx;
  handle->device = device;
  handle->workspace = workspace;
  
  if(capacity > SIZE_MAX - 15) return CF_ERR_OVERFLOW;
  capacity = (capacity + 15) & ~(cf_usize)15;

  switch (device)
  {
    case CF_MATH_DEVICE_CPU:
      handle->storage.heap = mi_heap_new();
      if(handle->storage.heap == CF_NULL) { state = CF_ERR_OOM; goto fail; }
      handle->storage.backend = mi_heap_malloc_aligned(handle->storage.heap, capacity, 16);
      handle->storage.byte_capacity = handle->storage.backend == CF_NULL ? 0 : capacity;
      if(!handle->storage.byte_capacity) { state = CF_ERR_OOM; goto fail; }
    break;

    case CF_MATH_DEVICE_CUDA:
      handle->storage.byte_capacity = cudaMallocAsync(&handle->storage.backend, capacity, workspace->stream) != cudaSuccess ? 0 : capacity;
      if(!handle->storage.byte_capacity) { state = CF_ERR_CUDA_MEMORY; goto fail; }
    break;

    default: state = CF_ERR_INVALID;
  }

  return state;

fail:
  cf_math_handle_destroy(handle);
  return state;
}

cf_status cf_math_handle_add(cf_math_handle *handle, cf_math *math)
{
  if(handle == CF_NULL || math == CF_NULL) return CF_ERR_NULL;
  if(math->desc == CF_NULL) return CF_ERR_STATE;

  cf_usize len = (math->desc->dim[0] * math->desc->strides[0] * cf_math_type_size(math->desc->dtype) + 15) & ~(cf_usize)15;

  if(len > SIZE_MAX - handle->storage.offset) return CF_ERR_OVERFLOW;
  if(len + handle->storage.offset > handle->storage.byte_capacity) return CF_ERR_BOUNDS;

  math->byte_offset = handle->storage.offset;
  handle->storage.offset += ((len ? len : 1) + 15) & ~(cf_usize)15;
  return CF_OK;
}

void cf_math_handle_reset(cf_math_handle *handle)
{
  if(handle == CF_NULL) return;
  handle->storage.offset = 0;
}

void cf_math_handle_destroy(cf_math_handle *handle)
{
  if(handle == CF_NULL) return;

  switch (handle->device)
  {
    case CF_MATH_DEVICE_CPU:
      if(handle->storage.heap != CF_NULL) mi_heap_destroy(handle->storage.heap);
    break;

    case CF_MATH_DEVICE_CUDA:
      if(handle->storage.backend != CF_NULL && handle->workspace != CF_NULL) cudaFreeAsync(handle->storage.backend, handle->workspace->stream);
    break;
  }

  *handle = (cf_math_handle) {0};
}
