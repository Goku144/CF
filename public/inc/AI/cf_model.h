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

#if !defined(CF_MODEL_H)
#define CF_MODEL_H

#include "MATH/cf_math.h"
#include "RUNTIME/cf_status.h"
#include "RUNTIME/cf_types.h"

typedef enum cf_ai_activation
{
  CF_AI_ACT_NONE = 0,
  CF_AI_ACT_RELU,
  CF_AI_ACT_GELU,
  CF_AI_ACT_SIGMOID,
  CF_AI_ACT_TANH
} cf_ai_activation;

typedef enum cf_ai_loss_kind
{
  CF_AI_LOSS_MSE = 0,
  CF_AI_LOSS_BINARY_CROSS_ENTROPY
} cf_ai_loss_kind;

typedef struct cf_ai_dense
{
  cf_math weights;
  cf_math bias;
  cf_math output;
  cf_math_metadata weights_meta;
  cf_math_metadata bias_meta;
  cf_math_metadata output_meta;
  cf_ai_activation activation;
} cf_ai_dense;

typedef struct cf_ai_model
{
  cf_ai_dense *layers;
  cf_usize layer_count;
  cf_math_handle_t *parameter_handler;
  cf_math_handle_t *activation_handler;
  cf_math_device device;
} cf_ai_model;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Bind a dense layer's parameter and output views from existing handlers.
 *
 * Weights are `[in_features, out_features]`, bias is `[out_features]`, and
 * output is `[batch, out_features]`. Handlers must use the same dtype/device.
 */
cf_status cf_ai_dense_init(cf_ai_dense *layer, cf_math_handle_t *parameter_handler, cf_math_handle_t *activation_handler, cf_usize batch, cf_usize in_features, cf_usize out_features, cf_ai_activation activation);

/**
 * @brief Run dense forward: `output = activation(input @ weights + bias)`.
 *
 * The input contract is `[batch, in_features]`. No storage is allocated and no
 * host/device copies are performed.
 */
cf_status cf_ai_dense_forward(cf_ai_dense *layer, const cf_math *input);

/**
 * @brief Unbind dense layer views without destroying the handlers.
 */
cf_status cf_ai_dense_destroy(cf_ai_dense *layer);

/**
 * @brief Attach a caller-owned dense layer array to a sequential model.
 */
cf_status cf_ai_model_init(cf_ai_model *model, cf_ai_dense *layers, cf_usize layer_count, cf_math_handle_t *parameter_handler, cf_math_handle_t *activation_handler, cf_math_device device);

/**
 * @brief Run layers sequentially and return the final layer output view.
 */
cf_status cf_ai_model_forward(cf_ai_model *model, const cf_math *input, cf_math **output);

/**
 * @brief Destroy all layer view bindings and clear the model descriptor.
 */
cf_status cf_ai_model_destroy(cf_ai_model *model);

/**
 * @brief Compute one-element MSE or binary cross entropy loss.
 */
cf_status cf_ai_loss_forward(cf_ai_loss_kind loss, cf_math *out, const cf_math *prediction, const cf_math *target);

#ifdef __cplusplus
}
#endif

#endif /* CF_MODEL_H */
