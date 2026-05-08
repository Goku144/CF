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

#include "AI/cf_tokenizer.h"

#define STB_IMAGE_IMPLEMENTATION
#include "RUNTIME/stb_image.h"

#include "MATH/cf_math.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

static cf_usize cf_tokenizer_align16(cf_usize size)
{
  return (size + 15) & ~(cf_usize)15;
}

static cf_status cf_tokenizer_reserve_image_bytes(cf_math_handle *handle, cf_math *raw_image, cf_usize byte_len)
{
  cf_usize aligned_len = 0;

  if(byte_len > SIZE_MAX - 15) return CF_ERR_OVERFLOW;
  aligned_len = cf_tokenizer_align16(byte_len ? byte_len : 1);

  if(aligned_len > SIZE_MAX - handle->storage.offset) return CF_ERR_OVERFLOW;
  if(aligned_len + handle->storage.offset > handle->storage.byte_capacity) return CF_ERR_BOUNDS;

  raw_image->byte_offset = handle->storage.offset;
  handle->storage.offset += aligned_len;
  return CF_OK;
}

static void CUDART_CB cf_tokenizer_free_pinned_memory(void *user_data)
{
  cudaFreeHost(user_data);
}

cf_status cf_tokenizer_load_and_transfer_image_u16(cf_math_handle *handle, cf_math *raw_image, const char *filename)
{
  int width = 0;
  int height = 0;
  int channels = 0;
  cf_u16 *pixels = CF_NULL;
  cf_u16 *pinned_pixels = CF_NULL;
  cf_u8 *device_pixels = CF_NULL;
  cf_usize pixel_count = 0;
  cf_usize byte_len = 0;
  cf_usize original_offset = 0;
  cf_status state = CF_OK;
  cf_bool copy_queued = CF_FALSE;
  cudaError_t cuda_state = cudaSuccess;

  if(handle == CF_NULL || raw_image == CF_NULL || filename == CF_NULL) return CF_ERR_NULL;
  if(handle->device != CF_MATH_DEVICE_CUDA || handle->workspace == CF_NULL) return CF_ERR_UNSUPPORTED;

  *raw_image = (cf_math) {0};

  pixels = stbi_load_16(filename, &width, &height, &channels, 1);
  if(pixels == CF_NULL) return CF_ERR_IO_READ;
  if(width <= 0 || height <= 0) { state = CF_ERR_PARSE; goto done; }

  pixel_count = (cf_usize)width * (cf_usize)height;
  if(width != 0 && pixel_count / (cf_usize)width != (cf_usize)height) { state = CF_ERR_OVERFLOW; goto done; }
  if(pixel_count > SIZE_MAX / sizeof(cf_u16)) { state = CF_ERR_OVERFLOW; goto done; }

  byte_len = pixel_count * sizeof(cf_u16);
  raw_image->elem_len = pixel_count;
  original_offset = handle->storage.offset;

  state = cf_tokenizer_reserve_image_bytes(handle, raw_image, byte_len);
  if(state != CF_OK) goto done;

  cuda_state = cudaMallocHost((void **)&pinned_pixels, byte_len);
  if(cuda_state != cudaSuccess) { state = CF_ERR_CUDA_MEMORY; goto done; }

  memcpy(pinned_pixels, pixels, byte_len);
  device_pixels = (cf_u8 *)handle->storage.backend + raw_image->byte_offset;

  cuda_state = cudaMemcpyAsync(device_pixels,
                               pinned_pixels,
                               byte_len,
                               cudaMemcpyHostToDevice,
                               handle->workspace->stream);
  if(cuda_state != cudaSuccess) { state = CF_ERR_CUDA_COPY; goto done; }
  copy_queued = CF_TRUE;

  cuda_state = cudaLaunchHostFunc(handle->workspace->stream,
                                  cf_tokenizer_free_pinned_memory,
                                  pinned_pixels);
  if(cuda_state != cudaSuccess) { state = CF_ERR_CUDA_LAUNCH; goto done; }

  pinned_pixels = CF_NULL;

done:
  if(pixels != CF_NULL) stbi_image_free(pixels);
  if(state != CF_OK && copy_queued) cudaStreamSynchronize(handle->workspace->stream);
  if(pinned_pixels != CF_NULL) cudaFreeHost(pinned_pixels);
  if(state != CF_OK) {
    handle->storage.offset = original_offset;
    *raw_image = (cf_math) {0};
  }
  (void)channels;
  return state;
}
