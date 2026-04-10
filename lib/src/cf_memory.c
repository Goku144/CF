#include "cf_memory.h"

cf_bytes cf_bytes_empty(void) {return (cf_bytes) {CF_NULL, 0};}

cf_bytes_mut cf_bytes_mut_empty(void) {return (cf_bytes_mut) {CF_NULL, 0};}

cf_buffer cf_buffer_empty(void) {return (cf_buffer) {CF_NULL, 0, 0};}

cf_bytes cf_bytes_from(const cf_u8 *data, cf_usize len) {return (cf_bytes) {data, len};}

cf_bytes_mut cf_bytes_mut_from(cf_u8 *data, cf_usize len) {return (cf_bytes_mut) {data, len};}