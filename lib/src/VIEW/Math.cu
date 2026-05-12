#include "VIEW/Math.hpp"


Math::Math(size_t offset, size_t capacity, Layout *layout)
{
  this->offset = offset;
  this->capacity = capacity;
  this->layout = layout;
}

Math::~Math() 
{
  delete this->layout;
}

size_t Math::getOffset()
{
  return this->offset;
}

size_t Math::getCapacity()
{
  return this->capacity;
}

Layout *Math::getLayout()
{
  return this->layout;
}

void Math::setOffset(size_t offset)
{
  this->offset = offset;
}

void Math::setCapacity(size_t capacity)
{
  this->capacity = capacity;
}

void Math::createLayout(int dim[HIGHEST_RANK], int rank,
                        layoutType lt, operationType ot)
{
  this->deleteLayout();
  this->layout = new Layout(dim, rank, lt, ot);
}

void Math::setLayout(int dim[HIGHEST_RANK], int rank)
{
  if (this->layout == NULL)
    this->createLayout(dim, rank);
  else
    this->layout->setDim(dim, rank);
}

void Math::deleteLayout()
{
  delete this->layout;
  this->layout = NULL;
}
