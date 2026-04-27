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
#define _POSIX_C_SOURCE 199309L
#include "RUNTIME/cf_time.h"

#include <errno.h>
#include <time.h>

/*
 * Read the wall clock for timestamps that should track real calendar time.
 * This belongs to the runtime layer because logging, IO metadata, and app code
 * need a shared status-returning clock wrapper.
 */
cf_status cf_time_now_wall(cf_time_point *out)
{
  if(out == CF_NULL) return CF_ERR_NULL;

  struct timespec ts;

  if(clock_gettime(CLOCK_REALTIME, &ts) != 0)
    return CF_ERR_TIME_CLOCK;

  out->ns = (cf_i64)ts.tv_sec * 1000000000 + ts.tv_nsec;
  return CF_OK;
}

/*
 * Read the monotonic clock for elapsed-time measurements. Framework benchmarks
 * and performance checks should use this instead of wall time.
 */
cf_status cf_time_now_mono(cf_time_point *out)
{
  if(out == CF_NULL) return CF_ERR_NULL;

  struct timespec ts;

  if(clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
    return CF_ERR_TIME_CLOCK;

  out->ns = (cf_i64)ts.tv_sec * 1000000000 + ts.tv_nsec;
  return CF_OK;
}


/*
 * Construct a framework duration from nanoseconds without conversion.
 */
cf_time cf_time_from_ns(cf_i64 ns)
{
  return (cf_time) {.ns = ns};
}
/*
 * Construct a framework duration from milliseconds.
 */
cf_time cf_time_from_ms(cf_i64 ms)
{
  return (cf_time) {.ns = ms * 1000000};
}
/*
 * Construct a framework duration from whole seconds.
 */
cf_time cf_time_from_sec(cf_i64 sec)
{
  return (cf_time) {.ns = sec * 1000000000};
}

/*
 * Return a duration in nanoseconds for precise runtime accounting.
 */
cf_i64 cf_time_as_ns(cf_time d)
{
  return d.ns;
}
/*
 * Return a duration in whole milliseconds for coarse reporting.
 */
cf_i64 cf_time_as_ms(cf_time d)
{
  return d.ns / 1000000;
}

/*
 * Return a duration in whole seconds for human-scale reporting.
 */
cf_i64 cf_time_as_sec(cf_time d)
{
  return d.ns / 1000000000;
}

/*
 * Compute elapsed duration between two framework time points.
 */
cf_time cf_time_elapsed(cf_time_point start, cf_time_point end)
{
  return (cf_time) {.ns = end.ns - start.ns};
}

/*
 * Sleep for a millisecond duration through the nanosecond sleep primitive.
 */
cf_status cf_time_sleep_ms(cf_u64 ms)
{
  return cf_time_sleep_ns(ms * 1000000);
}


/*
 * Sleep for a nanosecond duration and resume correctly after EINTR. Returning
 * cf_status keeps OS sleep failures visible to framework callers.
 */
cf_status cf_time_sleep_ns(cf_u64 ns)
{
  struct timespec req;
  struct timespec rem;

  req.tv_sec = ns / 1000000000;
  req.tv_nsec = ns % 1000000000;

  while(nanosleep(&req, &rem) != 0)
  {
    if(errno != EINTR)
      return CF_ERR_TIME_SLEEP;

    req = rem;
  }

  return CF_OK;
}
