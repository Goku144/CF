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

#include "MEMORY/cf_memory.h"
#include "MEMORY/cf_array.h"

#include "RUNTIME/cf_status.h"
#include "RUNTIME/cf_types.h"

#define CF_TENSOR_HIGHEST_RANK 8

typedef enum cf_tensor_device {
  CF_TENSOR_DEVICE_CPU,
  CF_TENSOR_DEVICE_CUDA,
  CF_TENSOR_DEVICE_OPENCL,
  CF_TENSOR_DEVICE_VULKAN
} cf_tensor_device;

typedef struct cf_tensor
{
  cf_array array;
  cf_usize dim[CF_TENSOR_HIGHEST_RANK];
  cf_usize stride[CF_TENSOR_HIGHEST_RANK];
  cf_usize rank;
  cf_tensor_device device;
}cf_tensor;


#ifdef CF_CUDA_AVAILABLE

#ifdef __cplusplus
extern "C" {
#endif

cf_status cf_tensor_init_gpu(cf_tensor *tensor, cf_usize dim, cf_usize rank, cf_usize elem_size);

cf_status cf_tensor_add_gpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out);

cf_status cf_tensor_mul_gpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out);

#ifdef __cplusplus
}
#endif

#define cf_tensor_init(tp, rank, dim, elem_size) cf_tensor_init_gpu(tp, rank, dim, elem_size)

#define cf_tensor_add(t1, t2, t_out) cf_tensor_add_gpu(t1, t2, t_out)

#define cf_tensor_mul(t1, t2, t_out) cf_tensor_mul_gpu(t1, t2, t_out)

#else

cf_status cf_tensor_init_cpu(cf_tensor *tensor, cf_usize dim, cf_usize rank, cf_usize elem_size);

cf_status cf_tensor_add_cpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out);

cf_status cf_tensor_mul_cpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out);

#define cf_tensor_init(tp, rank, dim, elem_size) cf_tensor_init_cpu(tp, dim, rank, elem_size)

#define cf_tensor_add(t1, t2, t_out) cf_tensor_add_cpu(t1, t2, t_out)

#define cf_tensor_mul(t1, t2, t_out) cf_tensor_mul_cpu(t1, t2, t_out)

#endif

cf_status cf_tensor_destroy(cf_tensor *tensor);

cf_status cf_tensor_get(cf_tensor *tensor, cf_usize *indexs);

#endif /* CF_TENSOR_H */
