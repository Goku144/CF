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

#include "MATH/cf_tensor.h"

#include <cuda_runtime.h>

__global__ void cf_tensor_kernel_add(void)
{
  
}

extern "C" cf_status cf_tensor_add_gpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out)
{
  CF_UNUSED(t1);
  CF_UNUSED(t2);
  CF_UNUSED(t_out);
  return CF_ERR_UNSUPPORTED;
}

extern "C" cf_status cf_tensor_scalar_mul_gpu(cf_tensor *t1, void *scalar, cf_tensor *t_out)
{
  CF_UNUSED(t1);
  CF_UNUSED(scalar);
  CF_UNUSED(t_out);
  return CF_ERR_UNSUPPORTED;
}

extern "C" cf_status cf_tensor_mul_gpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out)
{
  CF_UNUSED(t1);
  CF_UNUSED(t2);
  CF_UNUSED(t_out);
  return CF_ERR_UNSUPPORTED;
}

extern "C" cf_status cf_tensor_matrix_mul_gpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out)
{
  CF_UNUSED(t1);
  CF_UNUSED(t2);
  CF_UNUSED(t_out);
  return CF_ERR_UNSUPPORTED;
}
