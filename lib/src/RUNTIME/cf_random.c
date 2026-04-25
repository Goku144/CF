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

#include "RUNTIME/cf_random.h"

#include <errno.h>
#include <sys/random.h>

cf_status cf_random_bytes(void *dst, cf_usize len)
{
  if(len == 0) return CF_OK;
  if(dst == CF_NULL) return CF_ERR_NULL;

  cf_u8 *out = (cf_u8 *)dst;
  cf_usize offset = 0;
  while (offset < len)
  {
    cf_isize n = getrandom(out + offset, len - offset, 0);
    if(n < 0)
    {
      if(errno == EINTR) continue;
      return CF_ERR_RANDOM;
    }

    if(n == 0) return CF_ERR_RANDOM;

    offset += n;
  }
  return CF_OK;
}

cf_status cf_random_u32(cf_u32 *dst)
{
  return cf_random_bytes(dst, sizeof(*dst));
}

cf_status cf_random_u64(cf_u64 *dst)
{
  return cf_random_bytes(dst, sizeof(*dst));
}
