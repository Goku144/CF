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