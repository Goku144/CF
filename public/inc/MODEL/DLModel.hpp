#if !defined(DL_MODEL_HPP)
#define DL_MODEL_HPP

#include "MODEL/Layer.hpp"

class ConvLayer;
class PoolLayer;
class ExtractFeaturesLayer;
class DenseLayer;

class HandleCpu;
class HandleCuda;
class HandleImage;

class DLModel
{
private:
  HandleCpu *cpu;
  HandleCuda *gpu;
  HandleImage *images;

  Math hostTensor, hostTargets, deviceTargets;

  ConvLayer *convL;
  PoolLayer *poolL;
  ExtractFeaturesLayer *exfL;
  DenseLayer *dL;

public:
  DLModel();
  ~DLModel();

  void train(int iterations);
  void estimate(const char *convW, const char *convB,
                const char *exfW, const char *exfB,
                const char *denseW, const char *denseB);
};


#endif /* DL_MODEL_HPP */