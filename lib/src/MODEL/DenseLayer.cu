#include "MODEL/DenseLayer.hpp"
#include "Kernels.cuh"

#include "HANDLER/HandleCuda.hpp"

#include <cuda_fp16.h>

DenseLayer::DenseLayer(HandleCuda& handle) : Layer(handle)
{
  int xDim[HIGHEST_RANK]   = {64, 128, 0, 0};
  int wDim[HIGHEST_RANK]   = {128, 10, 0, 0};
  int bDim[HIGHEST_RANK]   = {1, 10, 0, 0};
  int zDim[HIGHEST_RANK]   = {64, 10, 0, 0};
  int actDim[HIGHEST_RANK] = {64, 10, 0, 0};
  int dzDim[HIGHEST_RANK]  = {64, 10, 0, 0};
  int dwDim[HIGHEST_RANK]  = {128, 10, 0, 0};
  int dbDim[HIGHEST_RANK]  = {1, 10, 0, 0};
  int daDim[HIGHEST_RANK]  = {64, 128, 0, 0};

  this->setX(xDim,     2, LT_MATRIX, OT_MATRIX);
  this->setW(wDim,     2, LT_MATRIX, OT_MATRIX);
  this->setB(bDim,     2, LT_MATRIX, OT_MATRIX);

  this->setZ(zDim,     2, LT_MATRIX, OT_MATRIX);
  this->setAct(actDim, 2, LT_MATRIX, OT_MATRIX);

  this->setDz(dzDim,   2, LT_MATRIX, OT_MATRIX);
  this->setDw(dwDim,   2, LT_MATRIX, OT_MATRIX);
  this->setDb(dbDim,   2, LT_MATRIX, OT_MATRIX);

  this->setDa(daDim,   2, LT_MATRIX, OT_MATRIX);
}

void DenseLayer::linear(Math& a1)
{
  void *src = (uint8_t *)this->handle->getData() + a1.getOffset();
  void *x = (uint8_t *)this->handle->getData() + this->x.getOffset();
  __half *w = (__half *)((uint8_t *)this->handle->getData() + this->w.getOffset());
  __half *b = (__half *)((uint8_t *)this->handle->getData() + this->b.getOffset());
  __half *z = (__half *)((uint8_t *)this->handle->getData() + this->z.getOffset());

  cudaMemcpyAsync(x, src, a1.getCapacity(), cudaMemcpyDeviceToDevice, this->handle->getWorkspaceStream());

  /* z = x · W  (fp16 storage, fp32 accumulate via Tensor Cores) */
  rowMajorGemmHalf(
    this->handle->getContextCublasHanler(),
    CUBLAS_OP_N,
    CUBLAS_OP_N,
    64,
    10,
    128,
    (__half *)x,
    128,
    w,
    10,
    z
  );

  /* z += b  (fp16 bias add) */
  addBiasKernel<<<3, 256, 0, this->handle->getWorkspaceStream()>>>(
    z, b, 64, 10
  );
}

float *DenseLayer::softmaxCrossEntropy(Math& targets)
{
  __half *z = (__half *)((uint8_t *)this->handle->getData() + this->z.getOffset());
  __half *act = (__half *)((uint8_t *)this->handle->getData() + this->act.getOffset());
  __half *dz = (__half *)((uint8_t *)this->handle->getData() + this->dz.getOffset());
  __half *target = (__half *)((uint8_t *)this->handle->getData() + targets.getOffset());

  /* Loss vectors live in fp32 scratchpad workspace */
  float *lossOut = (float *)this->handle->getWorkspaceScratchpad();
  float *lossVec = lossOut + 1;

  softmaxCrossEntropyKernel<<<64, 1, 0, this->handle->getWorkspaceStream()>>>(
    z, target, act, dz, lossVec
  );

  reduceLoss64Kernel<<<1, 64, 0, this->handle->getWorkspaceStream()>>>(
    lossVec, lossOut
  );

  return lossOut;
}

void DenseLayer::backLinear(Math& a1)
{
  __half *x  = (__half *)((uint8_t *)this->handle->getData() + a1.getOffset());
  __half *w  = (__half *)((uint8_t *)this->handle->getData() + this->w.getOffset());
  __half *dz = (__half *)((uint8_t *)this->handle->getData() + this->dz.getOffset());

  __half *da = (__half *)((uint8_t *)this->handle->getData() + this->da.getOffset());
  __half *dw = (__half *)((uint8_t *)this->handle->getData() + this->dw.getOffset());
  __half *db = (__half *)((uint8_t *)this->handle->getData() + this->db.getOffset());

  /* da = dz · Wᵀ */
  rowMajorGemmHalf(this->handle->getContextCublasHanler(), CUBLAS_OP_N, CUBLAS_OP_T, 64, 128, 10, dz, 10, w, 10, da);

  /* dw = xᵀ · dz */
  rowMajorGemmHalf(this->handle->getContextCublasHanler(), CUBLAS_OP_T, CUBLAS_OP_N, 128, 10, 64, x, 128, dz, 10, dw);

  /* db = sum(dz, axis=0)  — fp32 accumulate inside kernel */
  biasBackwardKernel<<<1, 32, 0, this->handle->getWorkspaceStream()>>>(dz, db, 64, 10);
}

void DenseLayer::learnFunc(float lr)
{
  __half *w  = (__half *)((uint8_t *)this->handle->getData() + this->w.getOffset());
  __half *b  = (__half *)((uint8_t *)this->handle->getData() + this->b.getOffset());
  __half *dw = (__half *)((uint8_t *)this->handle->getData() + this->dw.getOffset());
  __half *db = (__half *)((uint8_t *)this->handle->getData() + this->db.getOffset());

  cudaStream_t stream = this->handle->getWorkspaceStream();

  sgdUpdateHalf(w, dw, lr, 128 * 10, stream);
  sgdUpdateHalf(b, db, lr, 10, stream);
}
