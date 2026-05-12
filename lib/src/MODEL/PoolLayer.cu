#include "MODEL/PoolLayer.hpp"
#include "Kernels.cuh"

#include "HANDLER/HandleCuda.hpp"

#include <cuda_fp16.h>
#include <cudnn.h>

PoolLayer::PoolLayer(HandleCuda& handle) : Layer(handle)
{
  int xDim[HIGHEST_RANK]  = {64, 1, 28, 28};
  int zDim[HIGHEST_RANK]  = {64, 1, 14, 14};
  int dzDim[HIGHEST_RANK] = {64, 1, 14, 14};
  int daDim[HIGHEST_RANK] = {64, 1, 28, 28};

  this->setX(xDim,   4, LT_TENSOR, OT_POOL);
  this->setZ(zDim,   4, LT_TENSOR, OT_POOL);
  this->setDz(dzDim, 4, LT_TENSOR, OT_POOL);
  this->setDa(daDim, 4, LT_TENSOR, OT_POOL);
}

void PoolLayer::maxPool(Math& a1)
{
  float alpha = 1.0f;
  float beta = 0.0f;

  void *x = (uint8_t *)this->handle->getData() + a1.getOffset();
  void *poolX = (uint8_t *)this->handle->getData() + this->x.getOffset();
  void *z = (uint8_t *)this->handle->getData() + this->z.getOffset();

  cudaMemcpyAsync(poolX, x, a1.getCapacity(), cudaMemcpyDeviceToDevice, this->handle->getWorkspaceStream());

  cudnnPoolingForward(
    this->handle->getContextCudnnHanler(),
    this->x.getLayout()->getOpDescriptor().pool,
    &alpha,

    a1.getLayout()->getLayoutDescriptor().tensor,
    x,

    &beta,

    this->z.getLayout()->getLayoutDescriptor().tensor,
    z
  );
}

void PoolLayer::backPool(Math& daUpperLayer)
{
  float alpha = 1.0f;
  float beta = 0.0f;

  void *x  = (uint8_t *)this->handle->getData() + this->x.getOffset();
  void *z  = (uint8_t *)this->handle->getData() + this->z.getOffset();
  void *dz = (uint8_t *)this->handle->getData() + daUpperLayer.getOffset();
  void *da = (uint8_t *)this->handle->getData() + this->da.getOffset();

  cudnnPoolingBackward(
    this->handle->getContextCudnnHanler(),
    this->x.getLayout()->getOpDescriptor().pool,
    &alpha,

    this->z.getLayout()->getLayoutDescriptor().tensor,
    z,

    daUpperLayer.getLayout()->getLayoutDescriptor().tensor,
    dz,

    this->x.getLayout()->getLayoutDescriptor().tensor,
    x,

    &beta,

    this->da.getLayout()->getLayoutDescriptor().tensor,
    da
  );
}
