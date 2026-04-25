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

#if !defined(CF_LOG_H)
#define CF_LOG_H

#include "RUNTIME/cf_status.h"
#include "RUNTIME/cf_types.h"

typedef enum cf_log_level
{
  CF_LOG_LEVEL_TRACE = 0,
  CF_LOG_LEVEL_DEBUG = 1,
  CF_LOG_LEVEL_INFO = 2,
  CF_LOG_LEVEL_WARN = 3,
  CF_LOG_LEVEL_ERROR = 4,
  CF_LOG_LEVEL_FATAL = 5,
  CF_LOG_LEVEL_OFF = 6,
} cf_log_level;

/**
 * @brief Return the symbolic name for a log level.
 *
 * @param level Log level to convert.
 * @return Stable null-terminated level name, or "UNKNOWN" for unmapped values.
 */
const char *cf_log_level_as_char(cf_log_level level);

/**
 * @brief Set the global minimum log level.
 *
 * Messages below this level are ignored by `cf_log_write`. Passing
 * `CF_LOG_LEVEL_OFF` suppresses all log output.
 *
 * @param level Minimum level to emit.
 */
void cf_log_set_level(cf_log_level level);

/**
 * @brief Read the current global minimum log level.
 *
 * @return Current minimum level.
 */
cf_log_level cf_log_get_level(void);

/**
 * @brief Check whether a message at `level` would be emitted.
 *
 * @param level Log level to test.
 * @return `CF_TRUE` if the level passes the current filter, otherwise
 * `CF_FALSE`.
 */
cf_bool cf_log_should_write(cf_log_level level);

/**
 * @brief Write a formatted log message to standard error.
 *
 * The emitted line includes the level name, source file, source line, formatted
 * message, and a trailing newline. Messages below the current global level are
 * ignored.
 *
 * @param level Log level for the message.
 * @param file Source file name, usually `__FILE__`.
 * @param line Source line number, usually `__LINE__`.
 * @param fmt printf-style format string.
 */
void cf_log_write(cf_log_level level, const char *file, int line, const char *fmt, ...);

#define CF_LOG_TRACE(...) cf_log_write(CF_LOG_LEVEL_TRACE, __FILE__, __LINE__, __VA_ARGS__)
#define CF_LOG_DEBUG(...) cf_log_write(CF_LOG_LEVEL_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define CF_LOG_INFO(...) cf_log_write(CF_LOG_LEVEL_INFO, __FILE__, __LINE__, __VA_ARGS__)
#define CF_LOG_WARN(...) cf_log_write(CF_LOG_LEVEL_WARN, __FILE__, __LINE__, __VA_ARGS__)
#define CF_LOG_ERROR(...) cf_log_write(CF_LOG_LEVEL_ERROR, __FILE__, __LINE__, __VA_ARGS__)
#define CF_LOG_FATAL(...) cf_log_write(CF_LOG_LEVEL_FATAL, __FILE__, __LINE__, __VA_ARGS__)
#define CF_LOG_STATUS(level, status) cf_log_write(level, __FILE__, __LINE__, "%s", cf_status_as_char(status))

#endif /* CF_LOG_H */
