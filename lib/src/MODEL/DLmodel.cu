/******************************************************************************\
 * DLmodel.cu — Model orchestrator.
 *
 * Owns the handlers (CPU, GPU, Image) and the layer stack.
 * Implements the train loop and estimation entry points.
 * All layer logic lives in the per-layer .cu files.
\******************************************************************************/

#include "MODEL/DLModel.hpp"
#include "Kernels.cuh"

#include "HANDLER/HandleCpu.hpp"
#include "HANDLER/HandleCuda.hpp"
#include "HANDLER/HandleImage.hpp"

#include <filesystem>
#include <iostream>
#include <cmath>
#include <string>

#include <cuda_fp16.h>


/* ═══════════════════════════════════════════════════════════════════════════ *
 *  Construction / Destruction
 * ═══════════════════════════════════════════════════════════════════════════ */

DLModel::DLModel()
{
  this->cpu = new HandleCpu();
  this->gpu = new HandleCuda();
  this->images = new HandleImage();

  this->convL = new ConvLayer(*this->gpu);
  this->poolL = new PoolLayer(*this->gpu);
  this->exfL = new ExtractFeaturesLayer(*this->gpu);
  this->dL = new DenseLayer(*this->gpu);

  /* He initialization for conv weights, zero for biases */
  initMathRandom(*this->gpu, this->convL->w, std::sqrt(2.0f / 9.0f));
  initMathZero(*this->gpu, this->convL->b);

  /* He initialization for feature-extraction weights */
  initMathRandom(*this->gpu, this->exfL->w, std::sqrt(2.0f / 196.0f));
  initMathZero(*this->gpu, this->exfL->b);

  /* He initialization for dense weights */
  initMathRandom(*this->gpu, this->dL->w, std::sqrt(2.0f / 128.0f));
  initMathZero(*this->gpu, this->dL->b);
}

DLModel::~DLModel()
{
  delete this->dL;
  delete this->exfL;
  delete this->poolL;
  delete this->convL;

  delete this->images;
  delete this->gpu;
  delete this->cpu;
}


/* ═══════════════════════════════════════════════════════════════════════════ *
 *  Training Loop
 * ═══════════════════════════════════════════════════════════════════════════ */

void DLModel::train(int iterations)
{
  std::filesystem::create_directories("public/checkpoints");

  for (int i = 0; i < iterations; i++)
  {
    /* ── Load batch ───────────────────────────────────────────────────── */
    this->images->getTensorAndTargets(*this->cpu, this->hostTensor, this->hostTargets);

    this->cpu->copyToDevice(*this->gpu, this->convL->x, this->hostTensor);
    this->cpu->copyToDevice(*this->gpu, this->deviceTargets, this->hostTargets);

    /* ── Forward pass ─────────────────────────────────────────────────── */
    this->convL->convXWpB();
    this->convL->activationReLU();

    this->poolL->maxPool(this->convL->act);

    this->exfL->linear(this->poolL->z);
    this->exfL->activationReLU();

    this->dL->linear(this->exfL->act);
    float *loss = this->dL->softmaxCrossEntropy(this->deviceTargets);

    /* ── Diagnostics ──────────────────────────────────────────────────── */
    printTrainingProgress(*this->gpu, i + 1, loss, this->dL->act, this->deviceTargets);

    /* ── Backward pass + SGD update ───────────────────────────────────── */
    float lr = 0.01f;

    this->dL->backLinear(this->exfL->act);
    this->dL->learnFunc(lr);

    this->exfL->backActivation(this->dL->da);
    this->exfL->backLinear(this->poolL->z);
    this->exfL->learnFunc(lr);

    this->poolL->backPool(this->exfL->da);

    this->convL->backActivation(this->poolL->da);
    this->convL->learnFunc(lr);

    /* ── Checkpoint ───────────────────────────────────────────────────── */
    if ((i + 1) % 32 == 0)
    {
      std::string base = "public/checkpoints/iter_" + std::to_string(i + 1) + "_";

      saveMath(*this->gpu, this->convL->w, (base + "conv_w.bin").c_str());
      saveMath(*this->gpu, this->convL->b, (base + "conv_b.bin").c_str());

      saveMath(*this->gpu, this->exfL->w, (base + "exf_w.bin").c_str());
      saveMath(*this->gpu, this->exfL->b, (base + "exf_b.bin").c_str());

      saveMath(*this->gpu, this->dL->w, (base + "dense_w.bin").c_str());
      saveMath(*this->gpu, this->dL->b, (base + "dense_b.bin").c_str());
    }
  }
}


/* ═══════════════════════════════════════════════════════════════════════════ *
 *  Estimation / Inference
 * ═══════════════════════════════════════════════════════════════════════════ */

void DLModel::estimate(const char *convW, const char *convB,
                       const char *exfW, const char *exfB,
                       const char *denseW, const char *denseB)
{
  /* Load saved weights */
  loadMath(*this->gpu, this->convL->w, convW);
  loadMath(*this->gpu, this->convL->b, convB);

  loadMath(*this->gpu, this->exfL->w, exfW);
  loadMath(*this->gpu, this->exfL->b, exfB);

  loadMath(*this->gpu, this->dL->w, denseW);
  loadMath(*this->gpu, this->dL->b, denseB);

  /* Load one batch */
  this->images->getTensorAndTargets(*this->cpu, this->hostTensor, this->hostTargets);

  this->cpu->copyToDevice(*this->gpu, this->convL->x, this->hostTensor);
  this->cpu->copyToDevice(*this->gpu, this->deviceTargets, this->hostTargets);

  /* Forward only */
  this->convL->convXWpB();
  this->convL->activationReLU();

  this->poolL->maxPool(this->convL->act);

  this->exfL->linear(this->poolL->z);
  this->exfL->activationReLU();

  this->dL->linear(this->exfL->act);
  this->dL->softmaxCrossEntropy(this->deviceTargets);

  cudaStreamSynchronize(this->gpu->getWorkspaceStream());

  /* Copy results back to host */
  Math hostProb;
  Math hostTarget;

  this->gpu->copyToHost(*this->cpu, hostProb, this->dL->act);
  this->gpu->copyToHost(*this->cpu, hostTarget, this->deviceTargets);

  /* Data is fp16 in the arena — cast to __half */
  __half *probH = (__half *)((uint8_t *)this->cpu->getData() + hostProb.getOffset());
  __half *targetH = (__half *)((uint8_t *)this->cpu->getData() + hostTarget.getOffset());

  printEstimationByNumber(probH);

  /* Convert to fp32 for accuracy calculation */
  float prob[64 * 10];
  float target[64];
  for (int i = 0; i < 64 * 10; i++) prob[i] = __half2float(probH[i]);
  for (int i = 0; i < 64; i++) target[i] = __half2float(targetH[i]);

  int correct = 0;
  float confidence = 0.0f;

  for (int n = 0; n < 64; n++)
  {
    int best = 0;

    for (int c = 1; c < 10; c++)
      if (prob[n * 10 + c] > prob[n * 10 + best])
        best = c;

    confidence += prob[n * 10 + best];

    if (best == (int)target[n])
      correct++;
  }

  std::cout << "accuracy: " << (float)correct / 64.0f << std::endl;
  std::cout << "avg confidence: " << confidence / 64.0f << std::endl;
}
