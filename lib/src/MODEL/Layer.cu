#include "MODEL/Layer.hpp"


Layer::Layer(HandleCuda& handle)
{
  this->handle = &handle;
}

Layer::~Layer()
{}

void Layer::setZ(int dim[HIGHEST_RANK], int rank, layoutType lt, operationType ot)
{
  this->z.createLayout(dim, rank, lt, ot);
  this->handle->bind(this->z);
}

void Layer::setX(int dim[HIGHEST_RANK], int rank, layoutType lt, operationType ot)
{
  this->x.createLayout(dim, rank, lt, ot);
  this->handle->bind(this->x);
}

void Layer::setW(int dim[HIGHEST_RANK], int rank, layoutType lt, operationType ot)
{
  this->w.createLayout(dim, rank, lt, ot);
  this->handle->bind(this->w);
}

void Layer::setB(int dim[HIGHEST_RANK], int rank, layoutType lt, operationType ot)
{
  this->b.createLayout(dim, rank, lt, ot);
  this->handle->bind(this->b);
}

void Layer::setAct(int dim[HIGHEST_RANK], int rank, layoutType lt, operationType ot)
{
  this->act.createLayout(dim, rank, lt, ot);
  this->handle->bind(this->act);
}

void Layer::setDa(int dim[HIGHEST_RANK], int rank, layoutType lt, operationType ot)
{
  this->da.createLayout(dim, rank, lt, ot);
  this->handle->bind(this->da);
}

void Layer::setDz(int dim[HIGHEST_RANK], int rank, layoutType lt, operationType ot)
{
  this->dz.createLayout(dim, rank, lt, ot);
  this->handle->bind(this->dz);
}

void Layer::setDw(int dim[HIGHEST_RANK], int rank, layoutType lt, operationType ot)
{
  this->dw.createLayout(dim, rank, lt, ot);
  this->handle->bind(this->dw);
}

void Layer::setDb(int dim[HIGHEST_RANK], int rank, layoutType lt, operationType ot)
{
  this->db.createLayout(dim, rank, lt, ot);
  this->handle->bind(this->db);
}