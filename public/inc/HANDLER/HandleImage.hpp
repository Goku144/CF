#if !defined(DL_HANDLE_IMAGE_HPP)
#define DL_HANDLE_IMAGE_HPP

#include <stdint.h>

#include <filesystem>
#include <random>
#include <vector>
#include <string>
#include <algorithm>

#define IMAGES_DEFAULT_PATH "public/test"

class HandleCuda;

class HandleCpu;

class Layout;

class Math;

class HandleImage
{
private:
  int width, height, original_channels;
  const int desired_channels = 1;

  typedef struct Sample
  {
    std::string path;
    int label;
  } Sample;

  std::vector<Sample> samples;

  size_t index = 0;
public:
  HandleImage();
  ~HandleImage();

  void getTensorAndTargets(HandleCpu& handle, Math& tensor, Math& targets);
};

#endif /* DL_HANDLE_IMAGE_HPP */
