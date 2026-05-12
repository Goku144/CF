#if !defined(DL_EXTRACT_FEATURES_LAYER_HPP)
#define DL_EXTRACT_FEATURES_LAYER_HPP

#include "MODEL/Layer.hpp"

/**
 * ExtractFeaturesLayer — Fully-connected feature extraction layer.
 *
 * Takes the flattened pooling output and projects it to a lower-dim space.
 *
 * Architecture:
 *   Input:   [64, 196]  (batch × flattened 1×14×14)
 *   Weights: [196, 128]
 *   Bias:    [1, 128]
 *   Output:  [64, 128]
 *
 * Forward:  z = x·W + b  →  act = ReLU(z)
 * Backward: dz = ReLU'(z) ⊙ da  →  dw = xᵀ·dz,  db = sum(dz),  da_prev = dz·Wᵀ
 * Update:   w -= lr * dw,  b -= lr * db
 */
class ExtractFeaturesLayer : public Layer
{
public:
  ExtractFeaturesLayer(HandleCuda& handle);

  void linear(Math& a1);
  void activationReLU();
  void backActivation(Math& daUpperLayer);
  void backLinear(Math& a1);
  void learnFunc(float lr);
};

#endif /* DL_EXTRACT_FEATURES_LAYER_HPP */
