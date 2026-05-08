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

// The callback function that the GPU stream will trigger
void CUDART_CB free_pinned_memory_callback(void *userData) {
    cudaFreeHost(userData);
}

// Pass the cf_math object directly so the function updates it internally
void cf_load_and_transfer_image_u16(cf_math_handle *handle, cf_math *RawImage, const char* filename) 
{
    int width, height, channels;
    
    // 1. Load high-fidelity pixels from disk
    cf_u16* raw_pixels = stbi_load_16(filename, &width, &height, &channels, 1);
    if (!raw_pixels) {
        printf("Failed to load image: %s\n", filename);
        return;
    }

    // 2. Update the cf_math object's metadata directly
    RawImage->elem_len = width * height * 1; 
    size_t img_size = RawImage->elem_len * sizeof(cf_u16); 

    // 3. Allocate "Pinned" CPU memory (The VIP Dock)
    cf_u16* pinned_cpu_mem;
    cudaMallocHost((void**)&pinned_cpu_mem, img_size);
    memcpy(pinned_cpu_mem, raw_pixels, img_size);

    // Free the slow stb memory immediately
    stbi_image_free(raw_pixels);

    // 4. The Competitive Move: Async GPU Memory Allocation
    // This tells the GPU to allocate memory on this specific stream without stopping other work.
    cf_u8* device_ptr;
    cudaMallocAsync((void**)&device_ptr, img_size, handle->workspace->stream);
    
    // Assign this direct pointer to your object. 
    // (Note: If your backend strictly uses byte_offsets from a pre-allocated arena pool, 
    // you would instead update the offset here. But for dynamic fast memory, this is the way).
    RawImage->byte_offset = (size_t)(device_ptr - (cf_u8*)handle->storage.backend); // Example adaptation

    // 5. Fire across the PCIe bus asynchronously!
    cudaMemcpyAsync(device_ptr, pinned_cpu_mem, img_size, cudaMemcpyHostToDevice, handle->workspace->stream);

    // 6. The Cleanup Callback
    // Insert the callback into the stream queue. The GPU will execute it ONLY after the copy finishes.
    cudaLaunchHostFunc(handle->workspace->stream, free_pinned_memory_callback, (void*)pinned_cpu_mem);
}