#if !defined(DL_LAYOUT_HPP)
#define DL_LAYOUT_HPP

#include <stdint.h>
#include <array>

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cudnn.h>

#define HIGHEST_RANK 4

enum layoutType
{
  LT_FILTER = 0,
  LT_MATRIX,
  LT_TENSOR,
};

enum operationType
{
  OT_CONV = 0,
  OT_POOL,
  OT_MATRIX,
};

union layout
  {
    /******************************************\
     * @cuDNN (layer 1) (layout)
     * when we want to process N = 64 images
     * in the first layer we need this shap
     * it describe how everything is stored
     * in memory and wath is the format
     * @format CUDNN_TENSOR_NCHW
     * @datatype CUDNN_DATA_HALF  (fp16 storage)
     * @N the batch number of images
     * @C chanel of the image grayscale/rgb/...
     * @H the height of the image
     * @W the width of the image
     * @func cudnnSetTensor4dDescriptor
    \******************************************/
    cudnnTensorDescriptor_t tensor;

    /************************************************\
     * @cuDNN (layer 1) (layout)
     * this the kernl descriptor that
     * will slide over the Tensor based
     * on the convolution description
     * @format CUDNN_TENSOR_NCHW
     * @datatype CUDNN_DATA_HALF  (fp16 storage)
     * @K number of output 16 because we use 16 class
     * @C chanel of the image grayscale/rgb/...
     * @H the height of the image 3
     * @W the width of the image 3
     * @func cudnnSetFilter4dDescriptor
    \************************************************/
    cudnnFilterDescriptor_t filter;

    /*************************************\
     * @cuBlastLt (layer 2,3) (layout)
     * we use this to describ the matrix 
     * how it is in the memory and wath 
     * type with row columns and how much
     * padding of the leading dim (column)
     * @type CUDA_R_16F  (fp16 storage)
     * @rows
     * @colums
     * @ld will be 16
     * @func cuBlastLtMatrixLayoutCreate
    \*************************************/
    cublasLtMatrixLayout_t matrix;
  };

  union operation
  {
    /*******************************************\
     * @cuDNN (layer 1) (op)
     * it describs the convolution form
     * do we need to add a padding to
     * the Tensor we created for the N
     * images, do we need padding the 
     * steps we will use for the filter
     * using the stride height and width
     * with dilation 1 so no space between
     * the kenel element, mathematical mode
     * that is usefull for AI training without
     * needing the full convolution kernel
     * fliping the AI learn weigth auto
     * he can flip the kernel if needed 
     * using the weights
     * @pad h,w padd to keep the same size
     * @strides u,v
     * @dilation d_h,d_w
     * @mode CUDNN_CROSS_CORRELATION
     * @computeType CUDNN_DATA_FLOAT (fp32 compute)
     * @func cudnnSetConvolution2dDescriptor
    \*******************************************/
    cudnnConvolutionDescriptor_t conv;

    /***********************************\
     * @cuDNN (layer 1 -> 2) (op)
     * dimention reduction of the image
     * for smoth memory utilization
     * @mode CUDNN_POOLING_MAX
     * @window w_h,w_w 2,2
     * @stride s_h,s_w 2,2
     * @func cudnnSetPooling2dDescriptor
    \***********************************/
    cudnnPoolingDescriptor_t pool;
    
    /*************************************\
     * @cuBlastLt (layer 2,3) (op)
     * we will use it do describe how
     * the matrix mul will be the type
     * too do the math and do we add bias
     * @computeType CUBLAS_COMPUTE_32F (fp32 compute)
     * @epilogue CUBLASLT_EPILOGUE_BIAS
     * @func cublasLtMatmulDescCreate
    \*************************************/
    cublasLtMatmulDesc_t matrix;
  };

class Layout
{
private:
  int dim[HIGHEST_RANK] = {0};
  int strides[HIGHEST_RANK] = {0};
  int rank = 0;

  layout ly = {0};
  layoutType lt;
  operation op = {0};
  operationType ot;

  bool ready = false;

  void createDescriptors(void);
  void setDescriptors(void);
  void destroyDescriptors(void);

public:
  Layout(int dim[HIGHEST_RANK] = NULL, int rank = 0,
         layoutType lt = LT_MATRIX, operationType ot = OT_MATRIX);
  ~Layout();

  Layout(const Layout&) = delete;
  Layout& operator=(const Layout&) = delete;

  void setDim(int dim[HIGHEST_RANK], int rank);
  void update(void);

  layoutType getLayoutType() const;
  operationType getOperationType() const;

  operation getOpDescriptor() const;
  layout getLayoutDescriptor() const;

  std::array<int, HIGHEST_RANK> getDim() const;
  std::array<int, HIGHEST_RANK> getStrides() const;
  int getRank() const;
};

#endif /* DL_LAYOUT_HPP */