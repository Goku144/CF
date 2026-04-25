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

#if !defined(CF_IO_H)
#define CF_IO_H

#include "MEMORY/cf_memory.h"
#include "TEXT/cf_string.h"

#define CF_IO_INIT_SIZE 32
#define CF_IO_RESERVE_SIZE 2048

/**
 * Runtime IO helpers assume that pointer arguments and framework objects are
 * valid. Argument validation belongs to higher layers.
 */

/**
 * @brief Check whether a filesystem path currently exists.
 *
 * @param path Null-terminated path string.
 * @return `CF_TRUE` when the path exists, otherwise `CF_FALSE`.
 */
cf_bool cf_io_exists(const char *path);

/**
 * @brief Query the size of a filesystem object in bytes.
 *
 * @param path Null-terminated path string.
 * @param size Output pointer receiving the byte size reported by `stat`.
 * @return `CF_OK` on success or `CF_ERR_IO_METADATA` when the path cannot be
 * queried.
 */
cf_status cf_io_file_size(const char *path, cf_usize *size);

/**
 * @brief Read a whole file into a buffer.
 *
 * When `dst->data` is `CF_NULL`, the buffer is initialized first. Otherwise,
 * file bytes are appended at the current `dst->len`.
 *
 * @param dst Valid buffer receiving file bytes.
 * @param path Null-terminated source path string.
 * @return `CF_OK` on success, `CF_ERR_OOM` when allocation or growth fails,
 * `CF_ERR_IO_OPEN` when opening fails, `CF_ERR_IO_READ` when reading fails,
 * or `CF_ERR_IO_CLOSE` when closing fails.
 */
cf_status cf_io_read_file(cf_buffer *dst, const char *path);

/**
 * @brief Write raw bytes to a file, creating or truncating it.
 *
 * @param path Null-terminated destination path string.
 * @param src Byte view providing file contents.
 * @return `CF_OK` on success, `CF_ERR_OVERFLOW` when byte count arithmetic
 * overflows, `CF_ERR_IO_OPEN` when opening fails, `CF_ERR_IO_WRITE` when
 * writing fails, or `CF_ERR_IO_CLOSE` when closing fails.
 */
cf_status cf_io_write_file(const char *path, cf_bytes src);

/**
 * @brief Append raw bytes to a file, creating it when missing.
 *
 * @param path Null-terminated destination path string.
 * @param src Byte view providing bytes to append.
 * @return `CF_OK` on success, `CF_ERR_OVERFLOW` when byte count arithmetic
 * overflows, `CF_ERR_IO_OPEN` when opening fails, `CF_ERR_IO_WRITE` when
 * writing fails, or `CF_ERR_IO_CLOSE` when closing fails.
 */
cf_status cf_io_append_file(const char *path, cf_bytes src);

/**
 * @brief Read a whole file into a string.
 *
 * When `dst->data` is `CF_NULL`, the string is initialized first. Otherwise,
 * file bytes are appended at the current `dst->len`. The result is always
 * null-terminated on success.
 *
 * @param dst Valid string receiving file text.
 * @param path Null-terminated source path string.
 * @return `CF_OK` on success, `CF_ERR_OOM` when allocation or growth fails,
 * `CF_ERR_IO_OPEN` when opening fails, `CF_ERR_IO_READ` when reading fails,
 * or `CF_ERR_IO_CLOSE` when closing fails.
 */
cf_status cf_io_read_text(cf_string *dst, const char *path);

/**
 * @brief Write string contents to a file, creating or truncating it.
 *
 * @param path Null-terminated destination path string.
 * @param src Source string to write.
 * @return `CF_OK` on success, `CF_ERR_OVERFLOW` when byte count arithmetic
 * overflows, `CF_ERR_IO_OPEN` when opening fails, `CF_ERR_IO_WRITE` when
 * writing fails, or `CF_ERR_IO_CLOSE` when closing fails.
 */
cf_status cf_io_write_text(const char *path, cf_string *src);

/**
 * @brief Append string contents to a file, creating it when missing.
 *
 * @param path Null-terminated destination path string.
 * @param src Source string to append.
 * @return `CF_OK` on success, `CF_ERR_OVERFLOW` when byte count arithmetic
 * overflows, `CF_ERR_IO_OPEN` when opening fails, `CF_ERR_IO_WRITE` when
 * writing fails, or `CF_ERR_IO_CLOSE` when closing fails.
 */
cf_status cf_io_append_text(const char *path, cf_string *src);

#endif /* CF_IO_H */
