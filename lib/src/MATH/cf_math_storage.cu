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

cf_status cf_math_context_create(cf_math_context *ctx, int id_or_tnum, cf_math_device device)
{
  if(ctx == CF_NULL) return CF_ERR_NULL;
  cf_status state = CF_OK;

  switch (device)
  {
    case CF_MATH_DEVICE_CPU:
      ctx->context.cpu.num_threads = id_or_tnum;
      if(dnnl_engine_create(&ctx->context.cpu.engine, dnnl_cpu, 0) != dnnl_success) state = CF_ERR_INTERNAL;
      if(dnnl_stream_create(&ctx->context.cpu.stream, ctx->context.cpu.engine, dnnl_stream_default_flags) != dnnl_success) state = CF_ERR_INTERNAL;
    break;

    case CF_MATH_DEVICE_CUDA:
      ctx->context.cuda.device_id = id_or_tnum;
      if(cublasCreate(&ctx->context.cuda.cublas) != cudaSuccess) state = CF_ERR_CUDA;
      if(cublasLtCreate(&ctx->context.cuda.cublasLt) != cudaSuccess) state = CF_ERR_CUDA;
      if(cudnnCreate(&ctx->context.cuda.cudnn) != cudaSuccess) state = CF_ERR_CUDA;
      if(curandCreateGenerator(&ctx->context.cuda.curand, CURAND_RNG_PSEUDO_DEFAULT) != cudaSuccess) state = CF_ERR_CUDA;
      if(cusolverDnCreate(&ctx->context.cuda.cusolverDn) != cudaSuccess) state = CF_ERR_CUDA;
      if(cusparseCreate(&ctx->context.cuda.cusparse) != cudaSuccess) state = CF_ERR_CUDA;
    break;
    
    default: state = CF_ERR_INVALID;
  }

  ctx->device = device;
  return state;
}

void cf_math_context_destroy(cf_math_context *ctx)
{
  if(ctx == CF_NULL) return;

  switch (ctx->device)
  {
    case CF_MATH_DEVICE_CPU:
      dnnl_engine_destroy(ctx->context.cpu.engine);
      dnnl_stream_destroy(ctx->context.cpu.stream);
    break;

    case CF_MATH_DEVICE_CUDA:
      
      cublasDestroy(ctx->context.cuda.cublas);
      cublasLtDestroy(ctx->context.cuda.cublasLt);
      cudnnDestroy(ctx->context.cuda.cudnn);
      curandDestroyGenerator(ctx->context.cuda.curand);
      cusolverDnDestroy(ctx->context.cuda.cusolverDn);
      cusparseDestroy(ctx->context.cuda.cusparse);
    break;
  }
  *ctx = (cf_math_context) {0};
}

cf_status cf_math_workspace_create(cf_math_workspace *workspace, cf_math_context ctx, cf_usize capacity, cf_math_device device)
{
  if(workspace == CF_NULL) return CF_ERR_NULL;
  cf_status state = CF_OK;
  *workspace = {0};

  if(capacity == 0) return state;
  switch (device)
  {
    case CF_MATH_DEVICE_CPU:
      workspace->scratchpad = mi_malloc(capacity);
      workspace->scratchpad_size = workspace->scratchpad == CF_NULL ? 0 : capacity;
      if(!capacity) state = CF_ERR_OOM;
    break;

    case CF_MATH_DEVICE_CUDA:
      if(cudaStreamCreate(&workspace->stream) != cudaSuccess) state = CF_ERR_CUDA;
      cudaMallocAsync(&workspace->scratchpad, capacity, workspace->stream);
      workspace->scratchpad_size = workspace->scratchpad == CF_NULL ? 0 : capacity;
      if(!capacity) state = CF_ERR_CUDA_MEMORY;
    break;
    
    default: state = CF_ERR_INVALID;
  }

  workspace->device = device;
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
      cudaFreeAsync(workspace->scratchpad, workspace->stream);
      cudaStreamDestroy(workspace->stream);
    case CF_MATH_DEVICE_CUDA:

    break;
  }
  *workspace = (cf_math_workspace) {0};
}

cf_status cf_math_handle_create(cf_math_handle *handle, cf_math_context *ctx, cf_math_workspace *workspace,cf_math_device device, cf_usize capacity)
{
  if(handle == CF_NULL) return CF_ERR_NULL;
  cf_status state = CF_OK;

  switch (device)
  {
    case CF_MATH_DEVICE_CPU:
      handle->storage.heap = mi_heap_new();
      handle->storage.backend = mi_heap_malloc(handle->storage.heap, capacity);
      handle->storage.byte_capacity = handle->storage.backend == CF_NULL ? 0 : capacity;
      if(!capacity) state = CF_ERR_CUDA_MEMORY;
    break;

    case CF_MATH_DEVICE_CUDA:
      cudaMallocAsync(&handle->storage.backend, capacity, handle->workspace->stream);
      handle->storage.byte_capacity = handle->storage.backend == CF_NULL ? 0 : capacity;
      if(!capacity) state = CF_ERR_CUDA_MEMORY;
    break;
  }

  handle->ctx = ctx;
  handle->device = device;
  handle->workspace = workspace;
  return state;
}

void cf_math_handle_destroy(cf_math_handle *handle)
{
  if(handle == CF_NULL) return;

  switch (handle->device)
  {
    case CF_MATH_DEVICE_CPU:
      mi_heap_destroy(handle->storage.heap);
    break;

    case CF_MATH_DEVICE_CUDA:
      cudaFreeAsync(handle->storage.backend, handle->workspace->stream);
    break;
  }

  *handle = (cf_math_handle) {0};
}