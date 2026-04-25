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

#include "RUNTIME/cf_log.h"

#include <stdarg.h>
#include <stdio.h>

static cf_log_level g_cf_log_level = CF_LOG_LEVEL_INFO;

const char *cf_log_level_as_char(cf_log_level level)
{
  switch(level)
  {
    case CF_LOG_LEVEL_TRACE: return "TRACE";
    case CF_LOG_LEVEL_DEBUG: return "DEBUG";
    case CF_LOG_LEVEL_INFO: return "INFO";
    case CF_LOG_LEVEL_WARN: return "WARN";
    case CF_LOG_LEVEL_ERROR: return "ERROR";
    case CF_LOG_LEVEL_FATAL: return "FATAL";
    case CF_LOG_LEVEL_OFF: return "OFF";
    default: return "UNKNOWN";
  }
}

void cf_log_set_level(cf_log_level level)
{
  if(level < CF_LOG_LEVEL_TRACE || level > CF_LOG_LEVEL_OFF) return;
  g_cf_log_level = level;
}

cf_log_level cf_log_get_level(void)
{
  return g_cf_log_level;
}

cf_bool cf_log_should_write(cf_log_level level)
{
  if(level < CF_LOG_LEVEL_TRACE || level >= CF_LOG_LEVEL_OFF) return CF_FALSE;
  if(g_cf_log_level == CF_LOG_LEVEL_OFF) return CF_FALSE;
  return level >= g_cf_log_level;
}

void cf_log_write(cf_log_level level, const char *file, int line, const char *fmt, ...)
{
  if(fmt == CF_NULL) return;
  if(cf_log_should_write(level) == CF_FALSE) return;

  va_list args;
  va_start(args, fmt);
  (void)fprintf(stderr, "[%s] %s:%d: ", cf_log_level_as_char(level), file == CF_NULL ? "<unknown>" : file, line);
  (void)vfprintf(stderr, fmt, args);
  (void)fputc('\n', stderr);
  va_end(args);
}
