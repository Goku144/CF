#if !defined(CF_STATUS_H)
#define CF_STATUS_H
 
#include "RUNTIME/cf_types.h"
 
/**
 * cf_status
 *
 * Return code used by every function in the library that can fail.
 * Functions return CF_OK on success and one of the CF_ERR_* values
 * on failure. The specific errors each function may return are listed
 * in that function's documentation.
 *
 * Convention: always check the return value. A result of CF_OK means
 * the output parameters are valid; any other value means they are not.
 */
typedef enum cf_status
{
    CF_OK             = 0,  /* Success                                    */
    CF_ERR_INVALID,         /* Invalid argument or value                  */
    CF_ERR_NULL,            /* NULL pointer passed where not allowed      */
    CF_ERR_OOM,             /* Out of memory                              */
    CF_ERR_OVERFLOW,        /* Arithmetic or size overflow                */
    CF_ERR_BOUNDS,          /* Index or range out of bounds               */
    CF_ERR_STATE,           /* Object state is invalid for this operation */
    CF_ERR_UNSUPPORTED,     /* Feature, platform, or op not supported     */
    CF_ERR_DENIED,          /* Permission or policy denied                */
    CF_ERR_INTERNAL         /* Unexpected internal failure                */
} cf_status;
 
/********************************************************************/
/* diagnostics                                                      */
/********************************************************************/
 
/**
 * cf_status_str
 *
 * Return a short, human-readable string that names @p status.
 * The returned pointer is a string literal with static storage
 * duration — do not free it.
 * Unknown values return the string "CF_ERR_UNKNOWN".
 *
 * @param status  The status code to name.
 * @return  A null-terminated string literal, never CF_NULL.
 */
const char *cf_status_str(cf_status status);
 
/**
 * cf_status_desc
 *
 * Return a one-sentence description of what @p status means.
 * The returned pointer is a string literal with static storage
 * duration — do not free it.
 * Unknown values return a generic description.
 *
 * @param status  The status code to describe.
 * @return  A null-terminated string literal, never CF_NULL.
 */
const char *cf_status_desc(cf_status status);
 
/**
 * cf_status_print
 *
 * Print a single formatted line to stdout in the form:
 *   [CF_ERR_NULL] NULL pointer passed where not allowed
 * Appends a newline. Passing any valid or unknown status is safe.
 *
 * @param status  The status code to print.
 */
void cf_status_print(cf_status status);
 
/**
 * cf_status_fprint
 *
 * Same as cf_status_print() but writes to @p stream instead of stdout.
 * Passing CF_NULL for @p stream is safe and has no effect.
 *
 * @param stream  Output FILE stream. May be CF_NULL.
 * @param status  The status code to print.
 */
void cf_status_fprint(void *stream, cf_status status);
 
#endif /* CF_STATUS_H */