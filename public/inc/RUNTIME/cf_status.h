#if !defined(CF_STATUS_H)
#define CF_STATUS_H

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

#endif /* CF_STATUS_H */