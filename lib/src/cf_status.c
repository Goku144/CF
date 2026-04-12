#include "cf_status.h"
#include "cf_types.h"

#include <stdio.h>

/* ------------------------------------------------------------------ */
/* Internal state                                                      */
/* ------------------------------------------------------------------ */

static cf_status_handler g_handler = CF_NULL;

/* ------------------------------------------------------------------ */
/* Handler registration                                                */
/* ------------------------------------------------------------------ */

void cf_status_set_handler(cf_status_handler handler)
{
  g_handler = handler;
}

/* ------------------------------------------------------------------ */
/* Status printing                                                     */
/* ------------------------------------------------------------------ */

static const char *cf_status_message(cf_status status)
{
  switch (status)
  {
    case CF_OK:             return "ok";
    case CF_ERR_INVALID:    return "invalid argument or value";
    case CF_ERR_NULL:       return "null pointer passed where not allowed";
    case CF_ERR_OOM:        return "out of memory";
    case CF_ERR_OVERFLOW:   return "arithmetic or size overflow";
    case CF_ERR_BOUNDS:     return "index or range out of bounds";
    case CF_ERR_STATE:      return "object is not in a valid state for this operation";
    case CF_ERR_UNSUPPORTED:return "feature or platform not supported";
    case CF_ERR_DENIED:     return "permission or policy denied";
    case CF_ERR_INTERNAL:   return "unexpected internal failure";
    default:                return CF_NULL;
  }
}

void cf_status_print(cf_status status, const char *context)
{
  /* let the custom handler go first */
  if (g_handler != CF_NULL && g_handler(status, context) == CF_TRUE)
    return;

  const char *msg = cf_status_message(status);

  if (context != CF_NULL)
    fprintf(stderr, "%s: %s\n", context, msg != CF_NULL ? msg : "unknown status");
  else
    fprintf(stderr, "%s\n", msg != CF_NULL ? msg : "unknown status");
}