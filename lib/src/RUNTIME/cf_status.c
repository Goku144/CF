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

#include "RUNTIME/cf_status.h"

#include <stdio.h>

/********************************************************************/
/* diagnostics                                                      */
/********************************************************************/

const char *cf_status_str(cf_status status)
{
    switch(status)
    {
        case CF_OK:             return "CF_OK";
        case CF_ERR_INVALID:    return "CF_ERR_INVALID";
        case CF_ERR_NULL:       return "CF_ERR_NULL";
        case CF_ERR_OOM:        return "CF_ERR_OOM";
        case CF_ERR_OVERFLOW:   return "CF_ERR_OVERFLOW";
        case CF_ERR_BOUNDS:     return "CF_ERR_BOUNDS";
        case CF_ERR_STATE:      return "CF_ERR_STATE";
        case CF_ERR_UNSUPPORTED:return "CF_ERR_UNSUPPORTED";
        case CF_ERR_DENIED:     return "CF_ERR_DENIED";
        case CF_ERR_INTERNAL:   return "CF_ERR_INTERNAL";
        default:                return "CF_ERR_UNKNOWN";
    }
}

const char *cf_status_desc(cf_status status)
{
    switch(status)
    {
        case CF_OK:             return "Success.";
        case CF_ERR_INVALID:    return "Invalid argument or value.";
        case CF_ERR_NULL:       return "NULL pointer passed where not allowed.";
        case CF_ERR_OOM:        return "Out of memory.";
        case CF_ERR_OVERFLOW:   return "Arithmetic or size overflow.";
        case CF_ERR_BOUNDS:     return "Index or range out of bounds.";
        case CF_ERR_STATE:      return "Object state is invalid for this operation.";
        case CF_ERR_UNSUPPORTED:return "Feature, platform, or operation not supported.";
        case CF_ERR_DENIED:     return "Permission or policy denied.";
        case CF_ERR_INTERNAL:   return "Unexpected internal failure.";
        default:                return "Unknown status code.";
    }
}

void cf_status_print(cf_status status)
{
    printf("[%s] %s\n", cf_status_str(status), cf_status_desc(status));
}

void cf_status_fprint(void *stream, cf_status status)
{
    if(stream == CF_NULL) return;
    fprintf((FILE *) stream, "[%s] %s\n", cf_status_str(status), cf_status_desc(status));
}