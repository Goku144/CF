#if !defined(DL_CONV_LAYER_HPP)
#define DL_CONV_LAYER_HPP

#include "MODEL/Layer.hpp"

/**
 * ConvLayer — 2D convolution layer using cuDNN.
 *
 * Architecture:
 *   Input:  [64, 1, 28, 28]  (batch × channels × H × W)
 *   Filter: [1, 1, 3, 3]     (K × C × kH × kW)
 *   Bias:   [1, 1, 1, 1]
 *   Output: [64, 1, 28, 28]  (same-padding via pad=1, stride=1)
 *
 * Forward:  z = conv(x, w) + b  →  act = ReLU(z)
 * Backward: dz = ReLU'(z) ⊙ da  →  dw, db, (propagate da to previous layer)
 * Update:   w -= lr * dw,  b -= lr * db
 */
class ConvLayer : public Layer
{
public:
  ConvLayer(HandleCuda& handle);

  void convXWpB();
  void activationReLU();
  void backActivation(Math& daUpperLayer);
  void learnFunc(float lr);
};

#endif /* DL_CONV_LAYER_HPP */
