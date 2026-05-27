#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "vectraparse.h"

int main(void) {
  VectraParseHandle *handle = NULL;
  if (vectraparse_create_handle(&handle) != VECTRAPARSE_OK) {
    fprintf(stderr, "create_handle failed\n");
    return 1;
  }

  const uint8_t sample[] = "%PDF-1.7\n";
  VectraParseOptions options = {.timeout_ms = 1000, .max_bytes = 1024 * 1024};
  VectraParseResult out = {0};

  if (vectraparse_detect(handle, sample, sizeof(sample) - 1, &options, &out) !=
      VECTRAPARSE_OK) {
    fprintf(stderr, "detect failed\n");
    vectraparse_destroy_handle(handle);
    return 2;
  }
  printf("detect: %.*s\n", (int)out.len, (const char *)out.data);
  vectraparse_result_free(&out);

  if (vectraparse_parse(handle, sample, sizeof(sample) - 1, &options, &out) !=
      VECTRAPARSE_OK) {
    fprintf(stderr, "parse failed\n");
    vectraparse_destroy_handle(handle);
    return 3;
  }
  printf("parse: %.*s\n", (int)out.len, (const char *)out.data);
  vectraparse_result_free(&out);

  if (vectraparse_capabilities_json(&out) != VECTRAPARSE_OK) {
    fprintf(stderr, "capabilities failed\n");
    vectraparse_destroy_handle(handle);
    return 4;
  }
  printf("capabilities: %.*s\n", (int)out.len, (const char *)out.data);
  vectraparse_result_free(&out);

  printf("version: %s\n", vectraparse_version());
  vectraparse_destroy_handle(handle);
  return 0;
}
