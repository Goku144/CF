#if !defined(CF_STATUS_H)
#define CF_STATUS_H

#include "cf_types.h"

typedef enum cf_status
{
  CF_OK = 0,          /* SUCCESS */
  CF_ERR_INVALID,     /* Invalid argument/value */
  CF_ERR_NULL,        /* Null pointer passed where not allowed */
  CF_ERR_OOM,         /* Out of memory */
  CF_ERR_OVERFLOW,    /* Arithmetic/Size overflow */
  CF_ERR_BOUNDS,      /* Index/Range out of bounds */
  CF_ERR_STATE,       /* Object/State not valid for this operation */
  CF_ERR_UNSUPPORTED, /* Feature/Platform/Op not supported */
  CF_ERR_DENIED,      /* Permission/Policy denied */
  CF_ERR_INTERNAL     /* Unexpected internal failure */
} cf_status;

/* ------------------------------------------------------------------ */
/* Status printing                                                     */
/* ------------------------------------------------------------------ */

/*
 * Custom handler signature.
 * Called by cf_status_print when no built-in message covers the status.
 * Return CF_TRUE if your handler handled it, CF_FALSE to fall through
 * to the default "unknown status" message.
 */
typedef cf_bool (*cf_status_handler)(cf_status status, const char *context);

/**
 * Registers a custom status handler for extended/user-defined status codes.
 * Pass CF_NULL to remove the current handler and restore default behavior.
 * @param handler  Function pointer to your custom handler, or CF_NULL.
 */
void cf_status_set_handler(cf_status_handler handler);

/**
 * Prints a human-readable message for the given status to stderr.
 * If context is non-null it is prepended as "context: message".
 * If a custom handler is registered and claims the status, it runs instead.
 * @param status   The status code to describe.
 * @param context  Optional caller label, e.g. "cf_buffer_reserve". May be CF_NULL.
 */
void cf_status_print(cf_status status, const char *context);

#endif // CF_STATUS_H
