#if !defined(DL_DENSE_LAYER_HPP)
#define DL_DENSE_LAYER_HPP

#include "MODEL/Layer.hpp"

/**
 * DenseLayer — Output classification layer.
 *
 * Architecture:
 *   Input:   [64, 128]
 *   Weights: [128, 10]
 *   Bias:    [1, 10]
 *   Output:  [64, 10]  (10 digit classes)
 *
 * Forward:  z = x·W + b  →  act = softmax(z),  loss = cross_entropy(act, targets)
 * Backward: dz = (softmax - one_hot) / batch  →  dw, db, da_prev
 * Update:   w -= lr * dw,  b -= lr * db
 */
class DenseLayer : public Layer
{
public:
  DenseLayer(HandleCuda& handle);

  void linear(Math& a1);
  float *softmaxCrossEntropy(Math& targets);
  void backLinear(Math& a1);
  void learnFunc(float lr);
};

#endif /* DL_DENSE_LAYER_HPP */
