#include "VIEW/Math.hpp"
#include "HANDLER/HandleCuda.hpp"
#include "HANDLER/HandleCpu.hpp"

#include <iostream>
#include <string>
#include <stdexcept>
#include <cstring>
#include <cstdint>

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#define ALIGN_128(x) (x + 0x0F) & ~0x0F

HandleCpu::HandleCpu(size_t capacity)
{
  std::string err_msg = "";
  this->capacity = ALIGN_128(capacity * sizeof(__half));

#if (CUDA_CPU == 1)
  if(cudaMallocHost(&this->backend, this->capacity) != cudaSuccess)
  {err_msg = "FATAL ERROR: Handler Memory was not initialized before execution!"; goto _throw;}
#else
  if((this->backend = malloc(this->capacity)) == NULL)
  {err_msg = "FATAL ERROR: Handler Memory was not initialized before execution!"; goto _throw;}
#endif

  return;
_throw:
    throw std::runtime_error(err_msg);
}


HandleCpu::~HandleCpu()
{ 
#if (CUDA_CPU == 1)
  if(cudaFree(this->backend) != cudaSuccess)
    std::cerr << "[WARNING] Backend Memory Failed to be Freed!\n";
#else
  free(this->backend);
#endif
}


void *HandleCpu::getData()
{
  return this->backend;
}

size_t HandleCpu::getCurrentOffset()
{
  return this->offset;
}

size_t HandleCpu::getCapacity()
{
  return this->capacity;
}

void HandleCpu::copyToDevice(HandleCuda& handle, Math& deviceMath, Math& hostMath)
{
  auto dim = hostMath.getLayout()->getDim();
  int rank = hostMath.getLayout()->getRank();

  deviceMath.createLayout(dim.data(), rank, hostMath.getLayout()->getLayoutType(), hostMath.getLayout()->getOperationType());

  handle.bind(deviceMath);

  size_t count = hostMath.getCapacity();

  void *hostPtr = (void *)((uint8_t *)this->backend + hostMath.getOffset());
  void *cudaPtr = (void *)((uint8_t *)handle.getData() + deviceMath.getOffset());

  if (cudaMemcpy(cudaPtr, hostPtr, count, cudaMemcpyHostToDevice) != cudaSuccess)
    std::cerr << "[WARNING] CPU Memory Failed to be copied to Device!\n";
}


void HandleCpu::bind(Math& math)
{
  /* fp16 storage: sizeof(__half) = 2 bytes per element */
  size_t len = ALIGN_128(math.getLayout()->getStrides().at(0) * math.getLayout()->getDim().at(0) * sizeof(__half));

  if (this->offset > this->capacity || len > this->capacity - this->offset)
  {
    std::cerr << "[WARNING] CPU Handler Memory Not Enough to allocate requested Capacity, Not Allocating!\n";
    return;
  }

  math.setOffset(this->offset);
  math.setCapacity(len);

  this->offset += math.getCapacity();
}

void HandleCpu::unbind(Math& math)
{
  math.setOffset(0);
  math.setCapacity(0);
}

void HandleCpu::reset()
{
  this->offset = 0;
}