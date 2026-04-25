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

#if !defined(CF_TIME_H)
#define CF_TIME_H

#include "RUNTIME/cf_status.h"
#include "RUNTIME/cf_types.h"

typedef struct cf_time
{
  cf_i64 ns;
}cf_time, cf_time_point;

/**
 * @brief Read the current wall-clock time.
 *
 * Wall time represents real system/calendar time and may move forward or
 * backward if the system clock is adjusted.
 *
 * @param out Time point receiving the current wall time in nanoseconds.
 * @return `CF_OK` on success or `CF_ERR_TIME_CLOCK` when the clock read fails.
 */
cf_status cf_time_now_wall(cf_time_point *out);

/**
 * @brief Read the current monotonic time.
 *
 * Monotonic time is intended for elapsed-time measurement and is not a
 * calendar timestamp.
 *
 * @param out Time point receiving the current monotonic time in nanoseconds.
 * @return `CF_OK` on success or `CF_ERR_TIME_CLOCK` when the clock read fails.
 */
cf_status cf_time_now_mono(cf_time_point *out);

/**
 * @brief Create a time duration from nanoseconds.
 *
 * @param ns Duration value in nanoseconds.
 * @return Time value storing `ns` directly.
 */
cf_time cf_time_from_ns(cf_i64 ns);

/**
 * @brief Create a time duration from milliseconds.
 *
 * @param ms Duration value in milliseconds.
 * @return Time value converted to nanoseconds.
 */
cf_time cf_time_from_ms(cf_i64 ms);

/**
 * @brief Create a time duration from seconds.
 *
 * @param sec Duration value in seconds.
 * @return Time value converted to nanoseconds.
 */
cf_time cf_time_from_sec(cf_i64 sec);

/**
 * @brief Return a time value as nanoseconds.
 *
 * @param d Time value to convert.
 * @return Nanosecond representation.
 */
cf_i64 cf_time_as_ns(cf_time d);

/**
 * @brief Return a time value as whole milliseconds.
 *
 * @param d Time value to convert.
 * @return Millisecond representation, truncated toward zero.
 */
cf_i64 cf_time_as_ms(cf_time d);

/**
 * @brief Return a time value as whole seconds.
 *
 * @param d Time value to convert.
 * @return Second representation, truncated toward zero.
 */
cf_i64 cf_time_as_sec(cf_time d);

/**
 * @brief Compute elapsed time between two time points.
 *
 * @param start Starting time point.
 * @param end Ending time point.
 * @return Duration equal to `end - start`, stored in nanoseconds.
 */
cf_time cf_time_elapsed(cf_time_point start, cf_time_point end);

/**
 * @brief Sleep for at least the requested number of milliseconds.
 *
 * @param ms Duration to sleep in milliseconds.
 * @return `CF_OK` on success or `CF_ERR_TIME_SLEEP` when sleeping fails.
 */
cf_status cf_time_sleep_ms(cf_u64 ms);

/**
 * @brief Sleep for at least the requested number of nanoseconds.
 *
 * Interrupted sleeps are resumed until the requested duration has elapsed or
 * a non-interrupt sleep failure occurs.
 *
 * @param ns Duration to sleep in nanoseconds.
 * @return `CF_OK` on success or `CF_ERR_TIME_SLEEP` when sleeping fails.
 */
cf_status cf_time_sleep_ns(cf_u64 ns);

#endif /* CF_TIME_H */
