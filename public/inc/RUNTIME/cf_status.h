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

#if !defined(CF_STATUS_H)
#define CF_STATUS_H
 
/* Shared status codes used across the framework public APIs. */
typedef enum cf_status
{
  /* Operation completed successfully. */
  CF_OK = 0x00,

  /* A required pointer argument was NULL. */
  CF_ERR_NULL = 0x01,

  /* An argument value was invalid for the operation. */
  CF_ERR_INVALID = 0x02,

  /* The object or subsystem is in the wrong state. */
  CF_ERR_STATE = 0x04,

  /* A size, offset, or numeric conversion exceeded allowed bounds. */
  CF_ERR_BOUNDS = 0x08,
  CF_ERR_OVERFLOW = 0x10,

  /* Memory allocation or reservation failed. */
  CF_ERR_OOM = 0x20,

  /* Input or output operation failed. */
  CF_ERR_IO = 0x40,

  /* Parsing, decoding, or validation of input data failed. */
  CF_ERR_PARSE = 0x80,

  /* Requested behavior or platform capability is not supported. */
  CF_ERR_UNSUPPORTED = 0x100,

  /* Authentication, integrity, or other security checks failed. */
  CF_ERR_SECURITY = 0x200,

  /* An unexpected internal failure occurred. */
  CF_ERR_INTERNAL = 0x400,

  /* A fallback value for undefined or unmapped states. */
  CF_ERR_UNDEFINED = 0x800,
} cf_status;

/**
 * Return the symbolic string name of a framework status code.
 *
 * The returned pointer always refers to a static string literal and must not
 * be modified or freed by the caller.
 *
 * @param state
 *   The status code to convert to its symbolic string form.
 *
 * @return
 *   A stable null-terminated string such as "CF_OK" or "CF_ERR_IO".
 *   Unknown values return "CF_ERR_UNKNOWN".
 */
const char *cf_state_as_char(cf_status state);

/**
 * Print a framework status code in a readable diagnostic form.
 *
 * The printed output includes the caller-provided line number, the symbolic
 * status code, and a human-readable explanation, followed by a trailing
 * newline.
 *
 * @param state
 *   The status code to print.
 *
 * @param line
 *   A caller-provided source line number, typically the __LINE__ macro.
 *
 * @return
 *   This function does not return a value.
 */
void cf_state_print(cf_status state, const int line);

 
#endif /* CF_STATUS_H */
