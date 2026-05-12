#include "VIEW/Layout.hpp"

#include <iostream>
#include <string>
#include <stdexcept>

#include <cuda_fp16.h>

#define ALIGN_128(x) (x + 0x0F) & ~0x0F

Layout::Layout(int dim[HIGHEST_RANK], int rank, layoutType lt, operationType ot)
{
  this->lt = lt;
  this->ot = ot;

  if (dim != NULL && rank > 0)
    this->setDim(dim, rank);
}

Layout::~Layout()
{
  this->destroyDescriptors();
}

void Layout::setDim(int dim[HIGHEST_RANK], int rank)
{
  if (rank > HIGHEST_RANK)
    throw std::runtime_error("FATAL ERROR: RANK OVERFLOW");

  this->rank = rank;

  int tmp = 1;
  for (int i = rank - 1; i >= 0; --i)
  {
    this->dim[i] = dim[i];
    this->strides[i] = tmp;
    tmp *= dim[i];
  }

  for (int i = rank; i < HIGHEST_RANK; ++i)
  {
    this->dim[i] = 0;
    this->strides[i] = 0;
  }

  this->update();
}

void Layout::update(void)
{
  this->destroyDescriptors();
  this->createDescriptors();
  this->setDescriptors();
  this->ready = true;
}

void Layout::createDescriptors(void)
{
  /* ── Layout descriptors ──────────────────────────────────────────── */

  if (this->lt == LT_FILTER)
    if (cudnnCreateFilterDescriptor(&this->ly.filter) != CUDNN_STATUS_SUCCESS)
      throw std::runtime_error("FATAL ERROR: Filter descriptor creation failed!");

  if (this->lt == LT_MATRIX)
  {
    /*
     * fp16 storage: CUDA_R_16F
     * Leading dimension = number of columns (row-major)
     */
    int ld = this->dim[1];
    if (cublasLtMatrixLayoutCreate(&this->ly.matrix, CUDA_R_16F, this->dim[0], this->dim[1], ld) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("FATAL ERROR: Matrix layout creation failed!");
  }

  if (this->lt == LT_TENSOR)
    if (cudnnCreateTensorDescriptor(&this->ly.tensor) != CUDNN_STATUS_SUCCESS)
      throw std::runtime_error("FATAL ERROR: Tensor descriptor creation failed!");

  /* ── Operation descriptors ───────────────────────────────────────── */

  if (this->ot == OT_CONV)
    if (cudnnCreateConvolutionDescriptor(&this->op.conv) != CUDNN_STATUS_SUCCESS)
      throw std::runtime_error("FATAL ERROR: Convolution descriptor creation failed!");

  if (this->ot == OT_POOL)
    if (cudnnCreatePoolingDescriptor(&this->op.pool) != CUDNN_STATUS_SUCCESS)
      throw std::runtime_error("FATAL ERROR: Pooling descriptor creation failed!");

  if (this->ot == OT_MATRIX)
    if (cublasLtMatmulDescCreate(&this->op.matrix, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("FATAL ERROR: Matmul descriptor creation failed!");
}

void Layout::setDescriptors(void)
{
  /* ── Layout descriptors ──────────────────────────────────────────── */

  if (this->lt == LT_FILTER)
    if (cudnnSetFilter4dDescriptor(this->ly.filter, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, this->dim[0], this->dim[1], this->dim[2], this->dim[3]) != CUDNN_STATUS_SUCCESS)
      throw std::runtime_error("FATAL ERROR: Filter descriptor setup failed!");

  if (this->lt == LT_MATRIX)
  {
    int32_t order = CUBLASLT_ORDER_ROW;
    if (cublasLtMatrixLayoutSetAttribute(this->ly.matrix, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("FATAL ERROR: Matrix layout attribute setup failed!");
  }

  if (this->lt == LT_TENSOR)
    if (cudnnSetTensor4dDescriptor(this->ly.tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, this->dim[0], this->dim[1], this->dim[2], this->dim[3]) != CUDNN_STATUS_SUCCESS)
      throw std::runtime_error("FATAL ERROR: Tensor descriptor setup failed!");

  /* ── Operation descriptors ───────────────────────────────────────── */

  if (this->ot == OT_CONV)
  {
    /*
     * fp16 storage, fp32 compute:
     *   CUDNN_DATA_FLOAT as compute type lets cuDNN accumulate in fp32
     *   while reading/writing fp16 tensors.
     */
    if (cudnnSetConvolution2dDescriptor(this->op.conv, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT) != CUDNN_STATUS_SUCCESS)
      throw std::runtime_error("FATAL ERROR: Convolution descriptor setup failed!");

    if (cudnnSetConvolutionMathType(this->op.conv, CUDNN_TENSOR_OP_MATH) != CUDNN_STATUS_SUCCESS)
      throw std::runtime_error("FATAL ERROR: Convolution math type setup failed!");
  }

  if (this->ot == OT_POOL)
    if (cudnnSetPooling2dDescriptor(this->op.pool, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2) != CUDNN_STATUS_SUCCESS)
      throw std::runtime_error("FATAL ERROR: Pooling descriptor setup failed!");

  if (this->ot == OT_MATRIX)
  {
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    if (cublasLtMatmulDescSetAttribute(this->op.matrix, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("FATAL ERROR: Matmul epilogue attribute setup failed!");
  }
}

void Layout::destroyDescriptors(void)
{
  if (!this->ready)
    return;

  if (this->ot == OT_CONV)   cudnnDestroyConvolutionDescriptor(this->op.conv);
  if (this->ot == OT_POOL)   cudnnDestroyPoolingDescriptor(this->op.pool);
  if (this->ot == OT_MATRIX) cublasLtMatmulDescDestroy(this->op.matrix);

  if (this->lt == LT_FILTER) cudnnDestroyFilterDescriptor(this->ly.filter);
  if (this->lt == LT_MATRIX) cublasLtMatrixLayoutDestroy(this->ly.matrix);
  if (this->lt == LT_TENSOR) cudnnDestroyTensorDescriptor(this->ly.tensor);

  this->ready = false;
}

layoutType Layout::getLayoutType() const
{
  return this->lt;
}

operationType Layout::getOperationType() const
{
  return this->ot;
}

operation Layout::getOpDescriptor() const
{
  return this->op;
}
layout Layout::getLayoutDescriptor() const
{
  return this->ly;
}

std::array<int, HIGHEST_RANK> Layout::getDim() const
{
  std::array<int, HIGHEST_RANK> copy;

  for (int i = 0; i < HIGHEST_RANK; i++)
    copy[i] = this->dim[i];

  return copy;
}

std::array<int, HIGHEST_RANK> Layout::getStrides() const
{
  std::array<int, HIGHEST_RANK> copy;

  for (int i = 0; i < HIGHEST_RANK; i++)
    copy[i] = this->strides[i];

  return copy;
}

int Layout::getRank() const
{
  return this->rank;
}
