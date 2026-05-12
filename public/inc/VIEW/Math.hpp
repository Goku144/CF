#if !defined(DL_MATH_HPP)
#define DL_MATH_HPP

#include "HANDLER/HandleCpu.hpp"
#include "HANDLER/HandleCuda.hpp"
#include "VIEW/Layout.hpp"

#include <stdint.h>

class Math
{
private:
  size_t offset = 0;
  size_t capacity = 0;
  Layout *layout = NULL;

public:
  Math(size_t offset = 0, size_t capacity = 0, Layout *layout = NULL);
  ~Math();

  Math(const Math&) = delete;
  Math& operator=(const Math&) = delete;

  size_t getOffset();
  size_t getCapacity();
  Layout *getLayout();

  void setOffset(size_t offset);
  void setCapacity(size_t capavity);
  
  void createLayout(int dim[HIGHEST_RANK], int rank, layoutType lt = LT_MATRIX, operationType ot = OT_MATRIX);

  void setLayout(int dim[HIGHEST_RANK], int rank);

  void deleteLayout();
};

#endif /* DL_MATH_HPP */