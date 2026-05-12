#include "HANDLER/HandleCuda.hpp"
#include "HANDLER/HandleImage.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "HANDLER/stb_image.h"

#include "VIEW/Layout.hpp"
#include "VIEW/Math.hpp"

#include <iostream>
#include <cstring>
#include <stdexcept>

#include <cuda_fp16.h>

namespace fs = std::filesystem;

HandleImage::HandleImage()
{
  for (int label = 0; label < 10; label++)
  {
    fs::path folder = fs::path(IMAGES_DEFAULT_PATH) / std::to_string(label);

    for(const auto& entry : fs::directory_iterator(folder))
    {
      if(!entry.is_regular_file()) continue;
      if(entry.path().extension() != ".png") continue;
      this->samples.push_back({entry.path().string(), label});
    }
  }

  std::mt19937 rng(std::random_device{}());
  std::shuffle(this->samples.begin(), this->samples.end(), rng);
}

HandleImage::~HandleImage()
{}

void HandleImage::getTensorAndTargets(HandleCpu& handle, Math& tensor, Math& targets)
{
  if (this->samples.size() < 64)
    throw std::runtime_error("FATAL ERROR: Not enough images to build a batch!");

  if (this->index + 64 > this->samples.size())
  {
    std::mt19937 rng(std::random_device{}());
    std::shuffle(this->samples.begin(), this->samples.end(), rng);
    this->index = 0;
  }

  handle.reset();
  size_t offset = 0;

  /*
   * fp16 storage: load image as float via stbi, convert to __half
   * and store in the handler's memory arena.
   */
  __half target[64] = {};
  for (int n = 0; n < 64; n++)
  {
    float *image = stbi_loadf(this->samples[n + this->index].path.c_str(), &this->width, &this->height, &this->original_channels, desired_channels);

    int pixels = this->desired_channels * this->height * this->width;
    size_t bytes = pixels * sizeof(__half);

    /* Convert float pixels → __half and write into handler arena */
    __half *dst = (__half *)((uint8_t *)handle.getData() + offset);
    for (int p = 0; p < pixels; p++)
      dst[p] = __float2half(image[p]);

    target[n] = __float2half((float)this->samples[n + this->index].label);
    offset += bytes;
    stbi_image_free(image);
  }

  int tensorDim[HIGHEST_RANK] = {64, 1, this->height, this->width};
  tensor.createLayout(tensorDim, 4, LT_TENSOR, OT_CONV);
  handle.bind(tensor);

  /* Targets also stored as __half */
  std::memcpy((uint8_t *)handle.getData() + handle.getCurrentOffset(), target, 64 * sizeof(__half));
  int targetDim[HIGHEST_RANK] = {64, 0, 0, 0};
  targets.createLayout(targetDim, 1);
  handle.bind(targets);

  this->index += 64;
}
