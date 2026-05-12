#include "MODEL/ExtractFeaturesLayer.hpp"
#include "Kernels.cuh"

#include "HANDLER/HandleCuda.hpp"

#include <cuda_fp16.h>

ExtractFeaturesLayer::ExtractFeaturesLayer(HandleCuda& handle) : Layer(handle)
{
  int xDim[HIGHEST_RANK]   = {64, 196, 0, 0};
  int wDim[HIGHEST_RANK]   = {196, 128, 0, 0};
  int bDim[HIGHEST_RANK]   = {1, 128, 0, 0};
  int zDim[HIGHEST_RANK]   = {64, 128, 0, 0};
  int actDim[HIGHEST_RANK] = {64, 128, 0, 0};
  int dzDim[HIGHEST_RANK]  = {64, 128, 0, 0};
  int dwDim[HIGHEST_RANK]  = {196, 128, 0, 0};
  int dbDim[HIGHEST_RANK]  = {1, 128, 0, 0};
  int daDim[HIGHEST_RANK]  = {64, 196, 0, 0};

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

void ExtractFeaturesLayer::linear(Math& a1)
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
    128,
    196,
    (__half *)x,
    196,
    w,
    128,
    z
  );

  /* z += b  (fp16 bias add) */
  addBiasKernel<<<32, 256, 0, this->handle->getWorkspaceStream()>>>(
    z, b, 64, 128
  );
}

void ExtractFeaturesLayer::activationReLU()
{
  void *z = (uint8_t *)this->handle->getData() + this->z.getOffset();
  void *act = (uint8_t *)this->handle->getData() + this->act.getOffset();

  /* 8 halfs per uint4 → N/8 uint4 chunks */
  int N = this->act.getCapacity() / sizeof(__half);
  int N8 = N / 8;
  int threads = 256;
  int blocks = (N8 + threads - 1) / threads;

  reluForwardHalf2Kernel<<<blocks, threads, 0, this->handle->getWorkspaceStream()>>>(
    (const uint4 *)z,
    (uint4 *)act,
    N8
  );
}

void ExtractFeaturesLayer::backActivation(Math& daUpperLayer)
{
  void *z = (uint8_t *)this->handle->getData() + this->z.getOffset();
  void *da = (uint8_t *)this->handle->getData() + daUpperLayer.getOffset();
  void *dz = (uint8_t *)this->handle->getData() + this->dz.getOffset();

  int N = this->dz.getCapacity() / sizeof(__half);
  int N8 = N / 8;
  int threads = 256;
  int blocks = (N8 + threads - 1) / threads;

  reluBackwardHalf2Kernel<<<blocks, threads, 0, this->handle->getWorkspaceStream()>>>(
    (const uint4 *)z,
    (const uint4 *)da,
    (uint4 *)dz,
    N8
  );
}

void ExtractFeaturesLayer::backLinear(Math& a1)
{
  __half *x  = (__half *)((uint8_t *)this->handle->getData() + a1.getOffset());
  __half *w  = (__half *)((uint8_t *)this->handle->getData() + this->w.getOffset());
  __half *dz = (__half *)((uint8_t *)this->handle->getData() + this->dz.getOffset());

  __half *da = (__half *)((uint8_t *)this->handle->getData() + this->da.getOffset());
  __half *dw = (__half *)((uint8_t *)this->handle->getData() + this->dw.getOffset());
  __half *db = (__half *)((uint8_t *)this->handle->getData() + this->db.getOffset());

  /* da = dz · Wᵀ */
  rowMajorGemmHalf(this->handle->getContextCublasHanler(), CUBLAS_OP_N, CUBLAS_OP_T, 64, 196, 128, dz, 128, w, 128, da);

  /* dw = xᵀ · dz */
  rowMajorGemmHalf(this->handle->getContextCublasHanler(), CUBLAS_OP_T, CUBLAS_OP_N, 196, 128, 64, x, 196, dz, 128, dw);

  /* db = sum(dz, axis=0)  — fp32 accumulate inside kernel */
  biasBackwardKernel<<<1, 128, 0, this->handle->getWorkspaceStream()>>>(dz, db, 64, 128);
}

void ExtractFeaturesLayer::learnFunc(float lr)
{
  __half *w  = (__half *)((uint8_t *)this->handle->getData() + this->w.getOffset());
  __half *b  = (__half *)((uint8_t *)this->handle->getData() + this->b.getOffset());
  __half *dw = (__half *)((uint8_t *)this->handle->getData() + this->dw.getOffset());
  __half *db = (__half *)((uint8_t *)this->handle->getData() + this->db.getOffset());

  cudaStream_t stream = this->handle->getWorkspaceStream();

  sgdUpdateHalf(w, dw, lr, 196 * 128, stream);
  sgdUpdateHalf(b, db, lr, 128, stream);
}
