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

#include <mimalloc.h>
#include <dnnl.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cudnn.h>
#include <cusparse_v2.h>
#include <cusolverDn.h>
#include <curand.h>

typedef struct cf_math cf_math;
typedef struct cf_math_context cf_math_context;
typedef struct cf_math_workspace cf_math_workspace;
typedef struct cf_math_cuda_context cf_math_cuda_context;
typedef struct cf_math_cpu_context cf_math_cpu_context;
typedef struct cf_math_arena cf_math_arena;
typedef struct cf_math_handle cf_math_handle;

typedef enum cf_math_device
{
  CF_MATH_DEVICE_CPU = 0,
  CF_MATH_DEVICE_CUDA
} cf_math_device;

struct cf_math_workspace
{
  void *scratchpad;
  cudaStream_t stream;
  cf_usize scratchpad_size;
  cf_math_device device;
};

struct cf_math_cuda_context
{
  int device_id;

  cublasHandle_t      cublas;
  cublasLtHandle_t    cublasLt;
  cudnnHandle_t       cudnn;
  cusparseHandle_t    cusparse;
  cusolverDnHandle_t  cusolverDn;
  curandGenerator_t   curand;
};

struct cf_math_cpu_context
{
  int num_threads;
  dnnl_stream_t stream;
  dnnl_engine_t engine;
};

struct cf_math_context
{
  union
  {
    cf_math_cuda_context cuda;
    cf_math_cpu_context cpu;
  } context;
  cf_math_device device;
};

struct cf_math_arena
{
  void *backend;
  mi_heap_t *heap;

  cf_usize offset;
  cf_usize byte_capacity;
};

struct cf_math_handle
{
  cf_math_arena storage;
  cf_math_context *ctx;
  cf_math_workspace *workspace;
  cf_math_device device;
};

#ifdef __cplusplus
extern "C" {
#endif

cf_status cf_math_context_create(cf_math_context *ctx, int id_or_tnum, cf_math_device device);

void cf_math_context_destroy(cf_math_context *ctx);

cf_status cf_math_workspace_create(cf_math_workspace *workspace, cf_usize capacity, cf_math_device device);

void cf_math_workspace_destroy(cf_math_workspace *workspace);

cf_status cf_math_handle_create(cf_math_handle *handle, cf_math_context *ctx, cf_math_workspace *workspace, cf_usize capacity, cf_math_device device);

cf_status cf_math_handle_add(cf_math_handle *handle, cf_math *math);

void cf_math_handle_reset(cf_math_handle *handle);

void cf_math_handle_destroy(cf_math_handle *handle);

#ifdef __cplusplus
}
#endif

#endif /* CF_MATH_STORAGE_H */
