#include "MODEL/ConvLayer.hpp"
#include "Kernels.cuh"

#include "HANDLER/HandleCuda.hpp"

#include <cuda_fp16.h>
#include <cudnn.h>

ConvLayer::ConvLayer(HandleCuda& handle) : Layer(handle)
{
  int xDim[HIGHEST_RANK]   = {64, 1, 28, 28};
  int wDim[HIGHEST_RANK]   = {1, 1, 3, 3};
  int bDim[HIGHEST_RANK]   = {1, 1, 1, 1};
  int zDim[HIGHEST_RANK]   = {64, 1, 28, 28};
  int actDim[HIGHEST_RANK] = {64, 1, 28, 28};
  int dzDim[HIGHEST_RANK]  = {64, 1, 28, 28};
  int dwDim[HIGHEST_RANK]  = {1, 1, 3, 3};
  int dbDim[HIGHEST_RANK]  = {1, 1, 1, 1};

  this->setX(xDim,     4, LT_TENSOR, OT_CONV);
  this->setW(wDim,     4, LT_FILTER, OT_CONV);
  this->setB(bDim,     4, LT_TENSOR, OT_CONV);

  this->setZ(zDim,     4, LT_TENSOR, OT_CONV);
  this->setAct(actDim, 4, LT_TENSOR, OT_CONV);

  this->setDz(dzDim,   4, LT_TENSOR, OT_CONV);
  this->setDw(dwDim,   4, LT_FILTER, OT_CONV);
  this->setDb(dbDim,   4, LT_TENSOR, OT_CONV);
}

void ConvLayer::convXWpB()
{
  /*
   * cuDNN convolution with fp16 tensors and fp32 compute.
   * alpha/beta are fp32 because the convolution descriptor
   * uses CUDNN_DATA_FLOAT as compute type.
   */
  float alpha = 1.0f;
  float beta = 0.0f;

  void *x = (uint8_t *)this->handle->getData() + this->x.getOffset();
  void *w = (uint8_t *)this->handle->getData() + this->w.getOffset();
  void *b = (uint8_t *)this->handle->getData() + this->b.getOffset();
  void *z = (uint8_t *)this->handle->getData() + this->z.getOffset();

  cudnnConvolutionForward(
    this->handle->getContextCudnnHanler(),
    &alpha,
    this->x.getLayout()->getLayoutDescriptor().tensor,
    x,
    this->w.getLayout()->getLayoutDescriptor().filter,
    w,
    this->x.getLayout()->getOpDescriptor().conv,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
    this->handle->getWorkspaceScratchpad(),
    this->handle->getWorkspaceScratchpadSize(),
    &beta,
    this->z.getLayout()->getLayoutDescriptor().tensor,
    z
  );

  beta = 1.0f;

  cudnnAddTensor(
    this->handle->getContextCudnnHanler(),
    &alpha,
    this->b.getLayout()->getLayoutDescriptor().tensor,
    b,
    &beta,
    this->z.getLayout()->getLayoutDescriptor().tensor,
    z
  );
}

void ConvLayer::activationReLU()
{
  cudnnActivationDescriptor_t activDesc;
  cudnnCreateActivationDescriptor(&activDesc);
  cudnnSetActivationDescriptor(activDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);

  float alpha = 1.0f;
  float beta = 0.0f;

  void *z   = (uint8_t *)this->handle->getData() + this->z.getOffset();
  void *act = (uint8_t *)this->handle->getData() + this->act.getOffset();

  cudnnActivationForward
  (
    this->handle->getContextCudnnHanler(),
    activDesc,
    &alpha,
    this->z.getLayout()->getLayoutDescriptor().tensor,
    z,
    &beta,
    this->act.getLayout()->getLayoutDescriptor().tensor,
    act
  );

  cudnnDestroyActivationDescriptor(activDesc);
}

void ConvLayer::backActivation(Math& daUpperLayer)
{
  float alpha = 1.0f;
  float beta = 0.0f;

  void *x   = (uint8_t *)this->handle->getData() + this->x.getOffset();
  void *z   = (uint8_t *)this->handle->getData() + this->z.getOffset();
  void *act = (uint8_t *)this->handle->getData() + this->act.getOffset();

  void *da  = (uint8_t *)this->handle->getData() + daUpperLayer.getOffset();
  void *dz  = (uint8_t *)this->handle->getData() + this->dz.getOffset();
  void *dw  = (uint8_t *)this->handle->getData() + this->dw.getOffset();
  void *db  = (uint8_t *)this->handle->getData() + this->db.getOffset();

  cudnnActivationDescriptor_t activDesc;
  cudnnCreateActivationDescriptor(&activDesc);
  cudnnSetActivationDescriptor(activDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);

  cudnnActivationBackward
  (
    this->handle->getContextCudnnHanler(),
    activDesc,
    &alpha,
    this->act.getLayout()->getLayoutDescriptor().tensor,
    act,
    daUpperLayer.getLayout()->getLayoutDescriptor().tensor,
    da,
    this->z.getLayout()->getLayoutDescriptor().tensor,
    z,
    &beta,
    this->dz.getLayout()->getLayoutDescriptor().tensor,
    dz
  );

  cudnnDestroyActivationDescriptor(activDesc);

  cudnnConvolutionBackwardBias
  (
    this->handle->getContextCudnnHanler(),
    &alpha,
    this->dz.getLayout()->getLayoutDescriptor().tensor,
    dz,
    &beta,
    this->db.getLayout()->getLayoutDescriptor().tensor,
    db
  );

  cudnnConvolutionBackwardFilter
  (
    this->handle->getContextCudnnHanler(),
    &alpha,
    this->x.getLayout()->getLayoutDescriptor().tensor,
    x,
    this->dz.getLayout()->getLayoutDescriptor().tensor,
    dz,
    this->x.getLayout()->getOpDescriptor().conv,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
    this->handle->getWorkspaceScratchpad(),
    this->handle->getWorkspaceScratchpadSize(),
    &beta,
    this->dw.getLayout()->getLayoutDescriptor().filter,
    dw
  );
}

void ConvLayer::learnFunc(float lr)
{
  __half *w  = (__half *)((uint8_t *)this->handle->getData() + this->w.getOffset());
  __half *b  = (__half *)((uint8_t *)this->handle->getData() + this->b.getOffset());
  __half *dw = (__half *)((uint8_t *)this->handle->getData() + this->dw.getOffset());
  __half *db = (__half *)((uint8_t *)this->handle->getData() + this->db.getOffset());

  cudaStream_t stream = this->handle->getWorkspaceStream();

  sgdUpdateHalf(w, dw, lr, 1 * 1 * 3 * 3, stream);
  sgdUpdateHalf(b, db, lr, 1, stream);
}
