#if !defined(DL_POOL_LAYER_HPP)
#define DL_POOL_LAYER_HPP

#include "MODEL/Layer.hpp"

/**
 * PoolLayer — 2×2 max-pooling layer using cuDNN.
 *
 * Architecture:
 *   Input:  [64, 1, 28, 28]  →  Output: [64, 1, 14, 14]
 *   Window: 2×2,  Stride: 2×2,  No padding
 *
 * Forward:  z = maxpool(x)
 * Backward: da = maxpool_backward(dz, z, x)
 */
class PoolLayer : public Layer
{
public:
  PoolLayer(HandleCuda& handle);

  void maxPool(Math& a1);
  void backPool(Math& daUpperLayer);
};

#endif /* DL_POOL_LAYER_HPP */
