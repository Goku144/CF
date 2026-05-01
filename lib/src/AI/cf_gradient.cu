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

#include "AI/cf_gradient.h"

cf_status cf_ai_dense_backward(cf_ai_dense *layer, const cf_math *input, const cf_math *grad_output)
{
  CF_UNUSED(layer);
  CF_UNUSED(input);
  CF_UNUSED(grad_output);
  return CF_ERR_UNSUPPORTED;
}

cf_status cf_ai_loss_backward(cf_ai_loss_kind loss, cf_math *grad_prediction, const cf_math *prediction, const cf_math *target)
{
  CF_UNUSED(loss);
  CF_UNUSED(grad_prediction);
  CF_UNUSED(prediction);
  CF_UNUSED(target);
  return CF_ERR_UNSUPPORTED;
}
