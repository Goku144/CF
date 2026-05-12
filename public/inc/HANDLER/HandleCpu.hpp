#if !defined(DL_HANDLE_CPU_HPP)
#define DL_HANDLE_CPU_HPP

#include <stdint.h>

#define HANDLE_CPU_MEMORY_SPACE 2ULL * 1024 * 1024 * 1024 /* 2GB */
#define CUDA_CPU 0

class Math;
class HandleCuda;

class HandleCpu
{
private:
  void *backend = NULL;
  size_t offset = 0;
  size_t capacity = 0;

public:
  HandleCpu(size_t capacity = HANDLE_CPU_MEMORY_SPACE);
  ~HandleCpu();

  HandleCpu(const HandleCpu&) = delete;
  HandleCpu& operator=(const HandleCpu&) = delete;


  void *getData();
  size_t getCurrentOffset();
  size_t getCapacity();

  void copyToDevice(HandleCuda& handle, Math& deviceMath, Math& hostMath);

  void bind(Math& math);

  void unbind(Math& math);
  
  void reset();
};

#endif /* DL_HANDLE_CPU_HPP */