#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vectraparse.h"

static char *extract_json_string_field(const uint8_t *json, size_t len,
                                       const char *field) {
  if (json == NULL || field == NULL) {
    return NULL;
  }
  const char *start = (const char *)json;
  const char *end = start + len;
  char needle[128] = {0};
  snprintf(needle, sizeof(needle), "\"%s\":\"", field);
  const char *p = strstr(start, needle);
  if (p == NULL || p >= end) {
    return NULL;
  }
  p += strlen(needle);
  const char *q = p;
  while (q < end) {
    if (*q == '"' && (q == p || *(q - 1) != '\\')) {
      break;
    }
    q++;
  }
  if (q >= end || q <= p) {
    return NULL;
  }
  size_t n = (size_t)(q - p);
  char *out = (char *)malloc(n + 1);
  if (out == NULL) {
    return NULL;
  }
  memcpy(out, p, n);
  out[n] = '\0';
  return out;
}

static int read_file_bytes(const char *path, uint8_t **data, size_t *len) {
  FILE *fp = fopen(path, "rb");
  if (fp == NULL) {
    return -1;
  }
  if (fseek(fp, 0, SEEK_END) != 0) {
    fclose(fp);
    return -1;
  }
  long sz = ftell(fp);
  if (sz < 0) {
    fclose(fp);
    return -1;
  }
  if (fseek(fp, 0, SEEK_SET) != 0) {
    fclose(fp);
    return -1;
  }
  uint8_t *buf = (uint8_t *)malloc((size_t)sz);
  if (buf == NULL) {
    fclose(fp);
    return -1;
  }
  size_t got = fread(buf, 1, (size_t)sz, fp);
  fclose(fp);
  if (got != (size_t)sz) {
    free(buf);
    return -1;
  }
  *data = buf;
  *len = got;
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <absolute_file_path>\n", argv[0]);
    return 1;
  }
  const char *path = argv[1];
  if (path[0] != '/') {
    fprintf(stderr, "error: only absolute path is accepted\n");
    return 1;
  }

  VectraParseHandle *handle = NULL;
  if (vectraparse_create_handle(&handle) != VECTRAPARSE_OK) {
    fprintf(stderr, "error: vectraparse_create_handle failed\n");
    return 2;
  }

  VectraParseOptions options = {.timeout_ms = 30000, .max_bytes = 64 * 1024 * 1024};
  VectraParseResult detect_out = {0};
  VectraParseResult parse_out = {0};

  if (vectraparse_detect_file(handle, path, &options, &detect_out) != VECTRAPARSE_OK) {
    fprintf(stderr, "error: detect_file failed: %s\n", path);
    vectraparse_destroy_handle(handle);
    return 3;
  }

  uint8_t *bytes = NULL;
  size_t bytes_len = 0;
  if (read_file_bytes(path, &bytes, &bytes_len) != 0) {
    fprintf(stderr, "error: failed to read file bytes: %s\n", path);
    vectraparse_result_free(&detect_out);
    vectraparse_destroy_handle(handle);
    return 4;
  }

  if (vectraparse_parse(handle, bytes, bytes_len, &options, &parse_out) != VECTRAPARSE_OK) {
    fprintf(stderr, "error: parse failed: %s\n", path);
    free(bytes);
    vectraparse_result_free(&detect_out);
    vectraparse_destroy_handle(handle);
    return 5;
  }

  char *mime = extract_json_string_field(detect_out.data, detect_out.len, "mime_type");
  char *content = extract_json_string_field(parse_out.data, parse_out.len, "content");

  printf("File Type: %s\n\n\n", mime != NULL ? mime : "(unknown)");
  printf("Content:\n%s\n", content != NULL ? content : "");

  free(mime);
  free(content);
  free(bytes);
  vectraparse_result_free(&detect_out);
  vectraparse_result_free(&parse_out);
  vectraparse_destroy_handle(handle);
  return 0;
}
