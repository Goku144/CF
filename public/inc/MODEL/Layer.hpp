#if !defined(DL_LAYER_HPP)
#define DL_LAYER_HPP

#include "VIEW/Math.hpp"

class Layer
{
protected:
  HandleCuda *handle = NULL;
public:
  Math da = {0}, dw = {0}, db = {0};
  Math x = {0}, w = {0}, b = {0};
  Math z = {0}, act = {0}, dz = {0};

  Layer(HandleCuda& handle);
  virtual ~Layer();

  void setZ(int dim[HIGHEST_RANK], int rank, layoutType lt, operationType ot);

  void setX(int dim[HIGHEST_RANK], int rank, layoutType lt, operationType ot);

  void setW(int dim[HIGHEST_RANK], int rank, layoutType lt, operationType ot);

  void setB(int dim[HIGHEST_RANK], int rank, layoutType lt, operationType ot);

  void setAct(int dim[HIGHEST_RANK], int rank, layoutType lt, operationType ot);

  void setDa(int dim[HIGHEST_RANK], int rank, layoutType lt, operationType ot);

  void setDz(int dim[HIGHEST_RANK], int rank, layoutType lt, operationType ot);

  void setDw(int dim[HIGHEST_RANK], int rank, layoutType lt, operationType ot);

  void setDb(int dim[HIGHEST_RANK], int rank, layoutType lt, operationType ot);
};


#endif /* DL_LAYER_HPP */