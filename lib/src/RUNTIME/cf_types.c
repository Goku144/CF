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

#include "RUNTIME/cf_types.h"

/*
 * Classify a byte width into the native size groups used by diagnostics and
 * generic container metadata.
 */
cf_native_group cf_types_type_size(cf_usize type_size)
{
  switch(type_size)
  {
    case CF_NATIVE_1_BYTE: return CF_NATIVE_1_BYTE;
    case CF_NATIVE_2_BYTE: return CF_NATIVE_2_BYTE;
    case CF_NATIVE_4_BYTE: return CF_NATIVE_4_BYTE;
    case CF_NATIVE_8_BYTE: return CF_NATIVE_8_BYTE;
    case CF_NATIVE_16_BYTE: return CF_NATIVE_16_BYTE;
    default: return CF_NATIVE_UNKNOWN;
  }
}

/*
 * Return a readable description of a native size group for debug printers.
 */
const char *cf_types_as_char(cf_usize type_size)
{
  switch(cf_types_type_size(type_size))
  {
    case CF_NATIVE_1_BYTE:
      return "1 byte: char/bool-sized primitive value or struct with 1-byte total size.";
    case CF_NATIVE_2_BYTE:
      return "2 bytes: short-sized primitive value or struct with 2-byte total size.";
    case CF_NATIVE_4_BYTE:
      return "4 bytes: int/float-sized primitive value or struct with 4-byte total size.";
    case CF_NATIVE_8_BYTE:
      return "8 bytes: long/pointer/double-sized primitive value or struct with 8-byte total size.";
    case CF_NATIVE_16_BYTE:
      return "16 bytes: 128-bit/long-double-sized primitive value or struct with 16-byte total size.";
    default:
      return "Not a primitive type size or a struct with a native primitive-sized total width.";
  }
}
