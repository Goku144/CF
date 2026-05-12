#include "HANDLER/HandleCuda.hpp"
#include "HANDLER/HandleImage.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "HANDLER/stb_image.h"

#include "VIEW/Layout.hpp"
#include "VIEW/Math.hpp"

#include <iostream>
#include <cstring>

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
  if(this->index > this->samples.size() - 64) return;
  handle.reset();
  size_t offset = 0;
  float target[64] = {0};
  for (int n = 0; n < 64; n++)
  {
    float *image = stbi_loadf(this->samples[n + this->index].path.c_str(), &this->width, &this->height, &this->original_channels, desired_channels);
    size_t bytes = this->desired_channels * this->height * this->width * sizeof(float);
    std::memcpy((uint8_t *)handle.getData() + offset, image, bytes);
    target[n] = (float) this->samples[n + this->index].label;
    offset += bytes;
    stbi_image_free(image);
  }

#ifdef use_old
  tensor.createLayout((int[]) {64, 1, this->height, this->width}, 4, LT_TENSOR, OT_CONV);
#else
  int tensorDim[HIGHEST_RANK] = {64, 1, this->height, this->width};
  tensor.createLayout(tensorDim, 4, LT_TENSOR, OT_CONV);
#endif
  handle.bind(tensor);

  std::memcpy((uint8_t*) handle.getData() + handle.getCurrentOffset(), target, 64 * sizeof(float));
#ifdef use_old
  targets.createLayout((int[]){64, 0, 0, 0}, 1);
#else
  int targetDim[HIGHEST_RANK] = {64, 0, 0, 0};
  targets.createLayout(targetDim, 1);
#endif
  handle.bind(targets);

  this->index += 64;
}
