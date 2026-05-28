#ifndef VECTRAPARSE_H
#define VECTRAPARSE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VectraParseHandle VectraParseHandle;

typedef struct VectraParseOptions {
  uint32_t timeout_ms;
  size_t max_bytes;
} VectraParseOptions;

typedef struct VectraParseResult {
  uint8_t *data;
  size_t len;
} VectraParseResult;

typedef enum VectraParseError {
  VECTRAPARSE_OK = 0,
  VECTRAPARSE_NULL_POINTER = 1,
  VECTRAPARSE_INVALID_UTF8 = 2,
  VECTRAPARSE_INTERNAL = 255
} VectraParseError;

VectraParseError vectraparse_create_handle(VectraParseHandle **out);
void vectraparse_destroy_handle(VectraParseHandle *handle);

VectraParseError vectraparse_detect(VectraParseHandle *handle,
                                    const uint8_t *input,
                                    size_t input_len,
                                    const VectraParseOptions *options,
                                    VectraParseResult *out);
VectraParseError vectraparse_detect_with_hints(
    VectraParseHandle *handle,
    const uint8_t *input,
    size_t input_len,
    const VectraParseOptions *options,
    const char *resource_name,
    const char *content_type_hint,
    const char *force_content_type,
    VectraParseResult *out);
VectraParseError vectraparse_detect_file(VectraParseHandle *handle,
                                         const char *file_path,
                                         const VectraParseOptions *options,
                                         VectraParseResult *out);

VectraParseError vectraparse_parse(VectraParseHandle *handle,
                                   const uint8_t *input,
                                   size_t input_len,
                                   const VectraParseOptions *options,
                                   VectraParseResult *out);

void vectraparse_result_free(VectraParseResult *result);
const char *vectraparse_version(void);
VectraParseError vectraparse_capabilities_json(VectraParseResult *out);

#ifdef __cplusplus
}
#endif

#endif  // VECTRAPARSE_H
