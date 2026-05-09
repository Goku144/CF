/*
 * CF Framework
 * Copyright (C) 2026 Orion
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#if !defined(CF_MATH_F16_H)
#define CF_MATH_F16_H

#define CF_GELU_COEFF_A 0.79788456f
#define CF_GELU_COEFF_B 0.03567741f

#define CF_MATH_POOLING_MAX 0
#define CF_MATH_POOLING_AVG 1

typedef struct cf_math_handle cf_math_handle;
typedef struct cf_math cf_math;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Add two f16 tensors elementwise.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param C Output tensor.
 * @param A Left input tensor.
 * @param B Right input tensor.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_add_f16(cf_math_handle *handle, cf_math *C, cf_math *A, cf_math *B);

/**
 * @brief Subtract two f16 tensors elementwise.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param C Output tensor.
 * @param A Left input tensor.
 * @param B Right input tensor.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_sub_f16(cf_math_handle *handle, cf_math *C, cf_math *A, cf_math *B);

/**
 * @brief Multiply two f16 tensors elementwise.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param C Output tensor.
 * @param A Left input tensor.
 * @param B Right input tensor.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_mul_f16(cf_math_handle *handle, cf_math *C, cf_math *A, cf_math *B);

/**
 * @brief Divide two f16 tensors elementwise.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param C Output tensor.
 * @param A Left input tensor.
 * @param B Right input tensor.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_div_f16(cf_math_handle *handle, cf_math *C, cf_math *A, cf_math *B);

/**
 * @brief Negate an f16 tensor elementwise.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param C Output tensor.
 * @param A Input tensor.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_neg_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

/**
 * @brief Compute square root over an f16 tensor elementwise.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param C Output tensor.
 * @param A Input tensor.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_sqrt_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

/**
 * @brief Compute exponential over an f16 tensor elementwise.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param C Output tensor.
 * @param A Input tensor.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_exp_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

/**
 * @brief Compute natural logarithm over an f16 tensor elementwise.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param C Output tensor.
 * @param A Input tensor.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_log_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

/**
 * @brief Compute tanh over an f16 tensor elementwise.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param C Output tensor.
 * @param A Input tensor.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_tanh_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

/**
 * @brief Apply ReLU to an f16 tensor elementwise.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param C Output tensor.
 * @param A Input tensor.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_relu_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

/**
 * @brief Apply sigmoid to an f16 tensor elementwise.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param C Output tensor.
 * @param A Input tensor.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_sigmoid_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

/**
 * @brief Apply GELU to an f16 tensor elementwise.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param C Output tensor.
 * @param A Input tensor.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_gelu_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

/**
 * @brief Reduce all elements of an f16 tensor into one f16 sum.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param C One-element output tensor.
 * @param A Input tensor.
 * @return Nothing. This hot-path function assumes valid shapes, storage, and workspace.
 */
void cf_math_reduce_sum_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

/**
 * @brief Reduce all elements of an f16 tensor into one f16 mean.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param C One-element output tensor.
 * @param A Input tensor.
 * @return Nothing. This hot-path function assumes valid shapes, storage, and workspace.
 */
void cf_math_reduce_mean_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

/**
 * @brief Scale an f16 tensor by the reciprocal of a scalar.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param C Output tensor.
 * @param A Input tensor.
 * @param scalar Scalar divisor.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_norm_f16(cf_math_handle *handle, cf_math *C, cf_math *A, const float scalar);

/**
 * @brief Return the index of the largest element in an f16 tensor.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param A Input tensor.
 * @return Index of the maximum element, or -1 when the tensor/workspace is invalid.
 */
int cf_math_argmax_f16(cf_math_handle *handle, cf_math *A);

/**
 * @brief Set an f16 tensor to zero asynchronously.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param C Tensor to clear.
 * @return Nothing.
 */
void cf_math_zero_f16(cf_math_handle *handle, cf_math *C);

/**
 * @brief Run f16 matrix multiplication with the existing cuBLASLt descriptors.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param C Output matrix view.
 * @param A Left input matrix view.
 * @param B Right input matrix view.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_matmul_f16(cf_math_handle *handle, cf_math *C, cf_math *A, cf_math *B);

/**
 * @brief Run f16 matrix multiplication with A interpreted as transposed.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param C Output matrix view.
 * @param A Left input matrix view to read as transposed.
 * @param B Right input matrix view.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_matmul_trans_a_f16(cf_math_handle *handle, cf_math *C, cf_math *A, cf_math *B);

/**
 * @brief Run f16 matrix multiplication with B interpreted as transposed.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param C Output matrix view.
 * @param A Left input matrix view.
 * @param B Right input matrix view to read as transposed.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_matmul_trans_b_f16(cf_math_handle *handle, cf_math *C, cf_math *A, cf_math *B);

/**
 * @brief Compute `Output = Input @ Weight + Bias` using cuBLASLt bias epilogue.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param Output Output matrix view.
 * @param Input Input matrix view.
 * @param Weight Weight matrix view.
 * @param Bias Row-broadcast bias vector view.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_linear_bias_f16(cf_math_handle *handle, cf_math *Output, cf_math *Input, cf_math *Weight, cf_math *Bias);

/**
 * @brief Compute f16 softmax over the last dimension.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param Out Output tensor with the same shape as `In`.
 * @param In Input logits tensor.
 * @param dim Must be the last dimension in this implementation.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_softmax_f16(cf_math_handle *handle, cf_math *Out, cf_math *In, int dim);

/**
 * @brief Compute fused cross entropy loss and softmax gradient.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param dY Output tensor that receives `(P - one_hot(T)) / batch_size`.
 * @param batch_L Per-row loss output tensor.
 * @param loss One-element output tensor that receives mean loss.
 * @param P Row-major softmax probabilities tensor.
 * @param T One u8 target class index per row.
 * @return Nothing. Current implementation infers batch size from `P` and expects 16 padded classes.
 */
void cf_math_fused_cross_entropy(cf_math_handle *handle, cf_math *dY, cf_math *batch_L, cf_math *loss, cf_math *P, cf_math *T);

/**
 * @brief Reduce a row-major f16 matrix over rows into one value per column.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param C Output vector with one element per column.
 * @param A Input row-major matrix.
 * @return Nothing. This hot-path function assumes `A->elem_len` is divisible by `C->elem_len`.
 */
void cf_math_reduce_sum_rows_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

/**
 * @brief Backpropagate through ReLU over an f16 tensor.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param dIn Output gradient with respect to the ReLU input.
 * @param dOut Incoming gradient with respect to the ReLU output.
 * @param In Forward ReLU input tensor.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_relu_backward_f16(cf_math_handle *handle, cf_math *dIn, cf_math *dOut, cf_math *In);

/**
 * @brief Apply an in-place SGD update to an f16 tensor.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param Weight Parameter tensor updated in place.
 * @param Grad Gradient tensor.
 * @param lr Learning rate.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_sgd_update_f16(cf_math_handle *handle, cf_math *Weight, cf_math *Grad, float lr);

/**
 * @brief Set an f16 gradient tensor to zero asynchronously.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param Grad Gradient tensor to clear.
 * @return Nothing.
 */
void cf_math_zero_grad_f16(cf_math_handle *handle, cf_math *Grad);

/**
 * @brief Apply f16 2D convolution over NCHW tensors using cuDNN.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param Out Output tensor.
 * @param In Input tensor.
 * @param Weight Convolution filter tensor.
 * @param pad_h Height padding.
 * @param pad_w Width padding.
 * @param stride_h Height stride.
 * @param stride_w Width stride.
 * @param dilation_h Height dilation.
 * @param dilation_w Width dilation.
 * @return Nothing. This hot-path function assumes valid shapes, descriptors, and storage.
 */
void cf_math_conv2d_f16(
  cf_math_handle *handle,
  cf_math *Out,
  cf_math *In,
  cf_math *Weight,
  int pad_h,
  int pad_w,
  int stride_h,
  int stride_w,
  int dilation_h,
  int dilation_w
);

/**
 * @brief Apply f16 2D pooling over NCHW tensors using cuDNN.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param Out Output tensor shaped [N, C, OH, OW].
 * @param In Input tensor shaped [N, C, H, W].
 * @param mode CF_MATH_POOLING_MAX or CF_MATH_POOLING_AVG.
 * @param window_h Pooling window height.
 * @param window_w Pooling window width.
 * @param pad_h Height padding.
 * @param pad_w Width padding.
 * @param stride_h Height stride.
 * @param stride_w Width stride.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_pooling_f16(
  cf_math_handle *handle,
  cf_math *Out,
  cf_math *In,
  int mode,
  int window_h,
  int window_w,
  int pad_h,
  int pad_w,
  int stride_h,
  int stride_w
);

/**
 * @brief Backpropagate f16 2D pooling over NCHW tensors using cuDNN.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param dIn Output gradient with respect to `In`.
 * @param dOut Incoming gradient with respect to `Out`.
 * @param Out Forward pooling output tensor.
 * @param In Forward pooling input tensor.
 * @param mode CF_MATH_POOLING_MAX or CF_MATH_POOLING_AVG.
 * @param window_h Pooling window height.
 * @param window_w Pooling window width.
 * @param pad_h Height padding.
 * @param pad_w Width padding.
 * @param stride_h Height stride.
 * @param stride_w Width stride.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_pooling_backward_f16(
  cf_math_handle *handle,
  cf_math *dIn,
  cf_math *dOut,
  cf_math *Out,
  cf_math *In,
  int mode,
  int window_h,
  int window_w,
  int pad_h,
  int pad_w,
  int stride_h,
  int stride_w
);

/**
 * @brief Compute f16 convolution input gradients using cuDNN.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param dIn Output gradient with respect to convolution input.
 * @param dOut Incoming gradient with respect to convolution output.
 * @param Weight Forward convolution filter tensor.
 * @param pad_h Height padding.
 * @param pad_w Width padding.
 * @param stride_h Height stride.
 * @param stride_w Width stride.
 * @param dilation_h Height dilation.
 * @param dilation_w Width dilation.
 * @return Nothing. This hot-path function assumes valid shapes, descriptors, and storage.
 */
void cf_math_conv2d_backward_data_f16(
  cf_math_handle *handle,
  cf_math *dIn,
  cf_math *dOut,
  cf_math *Weight,
  int pad_h,
  int pad_w,
  int stride_h,
  int stride_w,
  int dilation_h,
  int dilation_w
);

/**
 * @brief Compute f16 convolution filter gradients using cuDNN.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param dWeight Output gradient with respect to convolution filters.
 * @param dOut Incoming gradient with respect to convolution output.
 * @param In Forward convolution input tensor.
 * @param pad_h Height padding.
 * @param pad_w Width padding.
 * @param stride_h Height stride.
 * @param stride_w Width stride.
 * @param dilation_h Height dilation.
 * @param dilation_w Width dilation.
 * @return Nothing. This hot-path function assumes valid shapes, descriptors, and storage.
 */
void cf_math_conv2d_backward_filter_f16(
  cf_math_handle *handle,
  cf_math *dWeight,
  cf_math *dOut,
  cf_math *In,
  int pad_h,
  int pad_w,
  int stride_h,
  int stride_w,
  int dilation_h,
  int dilation_w
);

/**
 * @brief Compute f16 convolution bias gradients using cuDNN.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param dBias Output gradient with respect to convolution bias.
 * @param dOut Incoming gradient with respect to convolution output.
 * @return Nothing. This hot-path function assumes valid shapes, descriptors, and storage.
 */
void cf_math_conv2d_backward_bias_f16(cf_math_handle *handle, cf_math *dBias, cf_math *dOut);

#ifdef __cplusplus
}
#endif

#endif /* CF_MATH_F16_H */
