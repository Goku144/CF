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

#if !defined(CF_GRADIENT_H)
#define CF_GRADIENT_H

#include "AI/cf_model.h"
#include "RUNTIME/cf_status.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Manual dense backward API boundary for the next AI batch.
 *
 * V1 currently returns `CF_ERR_UNSUPPORTED`.
 */
cf_status cf_ai_dense_backward(cf_ai_dense *layer, const cf_math *input, const cf_math *grad_output);

/**
 * @brief Manual loss backward API boundary for the next AI batch.
 *
 * V1 currently returns `CF_ERR_UNSUPPORTED`.
 */
cf_status cf_ai_loss_backward(cf_ai_loss_kind loss, cf_math *grad_prediction, const cf_math *prediction, const cf_math *target);

#ifdef __cplusplus
}
#endif

#endif /* CF_GRADIENT_H */
