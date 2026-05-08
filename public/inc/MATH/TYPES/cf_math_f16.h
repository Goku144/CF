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

typedef struct cf_math_handle cf_math_handle;
typedef struct cf_math cf_math;

#ifdef __cplusplus
extern "C" {
#endif

void cf_math_add_f16(cf_math_handle *handle, cf_math *C, cf_math *A, cf_math *B);

void cf_math_sub_f16(cf_math_handle *handle, cf_math *C, cf_math *A, cf_math *B);

void cf_math_mul_f16(cf_math_handle *handle, cf_math *C, cf_math *A, cf_math *B);

void cf_math_div_f16(cf_math_handle *handle, cf_math *C, cf_math *A, cf_math *B);

void cf_math_neg_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

void cf_math_sqrt_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

void cf_math_exp_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

void cf_math_log_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

void cf_math_tanh_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

void cf_math_relu_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

void cf_math_sigmoid_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

void cf_math_gelu_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

void cf_math_reduce_sum_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

void cf_math_reduce_mean_f16(cf_math_handle *handle, cf_math *C, cf_math *A);

void cf_math_norm_f16(cf_math_handle *handle, cf_math *C, cf_math *A, const float scalar);

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
 * @brief Apply last-dimension f16 layer normalization.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param Out Output tensor with the same shape as `In`.
 * @param In Input tensor, normalized over its last dimension.
 * @param Weight Scale vector with length equal to the last dimension.
 * @param Bias Bias vector with length equal to the last dimension.
 * @param eps Small value added to variance before reciprocal square root.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_layer_norm_f16(cf_math_handle *handle, cf_math *Out, cf_math *In, cf_math *Weight, cf_math *Bias, float eps);

/**
 * @brief Apply f16 layer normalization and cache per-row mean and variance.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param Out Output tensor with the same shape as `In`.
 * @param Mean Per-row mean tensor with `In->elem_len / cols` elements.
 * @param Var Per-row variance tensor with `In->elem_len / cols` elements.
 * @param In Input tensor, normalized over its last dimension.
 * @param Weight Scale vector with length equal to the last dimension.
 * @param Bias Bias vector with length equal to the last dimension.
 * @param eps Small value added to variance before reciprocal square root.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_layer_norm_stats_f16(cf_math_handle *handle, cf_math *Out, cf_math *Mean, cf_math *Var, cf_math *In, cf_math *Weight, cf_math *Bias, float eps);

/**
 * @brief Backpropagate f16 layer normalization using cached row mean/variance.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param dIn Output gradient with respect to `In`.
 * @param dWeight Output gradient with respect to `Weight`.
 * @param dBias Output gradient with respect to `Bias`.
 * @param dOut Incoming output gradient.
 * @param In Forward input tensor.
 * @param Weight Forward scale vector.
 * @param Mean Cached per-row mean from `cf_math_layer_norm_stats_f16`.
 * @param Var Cached per-row variance from `cf_math_layer_norm_stats_f16`.
 * @param eps Small value added to variance before reciprocal square root.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_layer_norm_backward_f16(cf_math_handle *handle, cf_math *dIn, cf_math *dWeight, cf_math *dBias, cf_math *dOut, cf_math *In, cf_math *Weight, cf_math *Mean, cf_math *Var, float eps);

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
 * @brief Compute mean cross entropy from f16 logits and f16 class indices.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param Loss One-element output tensor that receives mean loss.
 * @param Logits Row-major logits tensor, classes on the last dimension.
 * @param Targets One f16 class index per row.
 * @return Nothing. This hot-path function assumes valid shapes, storage, and workspace.
 */
void cf_math_cross_entropy_f16(cf_math_handle *handle, cf_math *Loss, cf_math *Logits, cf_math *Targets);

/**
 * @brief Compute f16 cross entropy gradient with respect to logits.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param dLogits Output tensor with the same shape as `Logits`.
 * @param Logits Row-major logits tensor, classes on the last dimension.
 * @param Targets One f16 class index per row.
 * @return Nothing. Writes `(softmax(Logits) - one_hot(Targets)) / rows`.
 */
void cf_math_cross_entropy_backward_f16(cf_math_handle *handle, cf_math *dLogits, cf_math *Logits, cf_math *Targets);

/**
 * @brief Apply an in-place f16 AdamW optimizer step.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param Weight Parameter tensor updated in place.
 * @param Grad Gradient tensor.
 * @param M First moment tensor updated in place.
 * @param V Second moment tensor updated in place.
 * @param lr Learning rate.
 * @param beta1 First moment decay.
 * @param beta2 Second moment decay.
 * @param eps Denominator epsilon.
 * @param weight_decay Decoupled weight decay.
 * @param step One-based optimizer step used for bias correction.
 * @return Nothing. This hot-path function assumes valid shapes and storage.
 */
void cf_math_adamw_update_f16(cf_math_handle *handle, cf_math *Weight, cf_math *Grad, cf_math *M, cf_math *V, float lr, float beta1, float beta2, float eps, float weight_decay, int step);

/**
 * @brief Set an f16 gradient tensor to zero asynchronously.
 * @param handle CUDA math handle that owns storage, workspace, and stream.
 * @param Grad Gradient tensor to clear.
 * @return Nothing.
 */
void cf_math_zero_grad_f16(cf_math_handle *handle, cf_math *Grad);

#ifdef __cplusplus
}
#endif

#endif /* CF_MATH_F16_H */
