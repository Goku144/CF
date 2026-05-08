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

#if !defined(CF_TOKENIZER_H)
#define CF_TOKENIZER_H

#include "RUNTIME/cf_status.h"

typedef struct cf_math_handle cf_math_handle;
typedef struct cf_math cf_math;

#ifdef __cplusplus
extern "C" {
#endif

cf_status cf_tokenizer_load_and_transfer_image_u16(cf_math_handle *handle, cf_math *raw_image, const char *filename);

#ifdef __cplusplus
}
#endif

#endif /* CF_TOKENIZER_H */
