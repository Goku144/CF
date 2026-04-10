#include "cf_string.h"

cf_str cf_str_empty(void) {return (cf_str) {CF_NULL, 0};}

cf_string cf_string_empty(void) {return (cf_string) {CF_NULL, 0, 0};}

cf_str cf_str_from(const char *data, cf_usize len) {return (cf_str) {data, len};}