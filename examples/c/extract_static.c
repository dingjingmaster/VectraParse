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

static void append_utf8(char **buf, size_t *len, size_t *cap, unsigned cp) {
  if (cp > 0x10FFFF) {
    cp = 0xFFFD;
  }
  char tmp[4];
  size_t n = 0;
  if (cp < 0x80) {
    tmp[n++] = (char)cp;
  } else if (cp < 0x800) {
    tmp[n++] = (char)(0xC0 | (cp >> 6));
    tmp[n++] = (char)(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    tmp[n++] = (char)(0xE0 | (cp >> 12));
    tmp[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
    tmp[n++] = (char)(0x80 | (cp & 0x3F));
  } else {
    tmp[n++] = (char)(0xF0 | (cp >> 18));
    tmp[n++] = (char)(0x80 | ((cp >> 12) & 0x3F));
    tmp[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
    tmp[n++] = (char)(0x80 | (cp & 0x3F));
  }
  if (*len + n + 1 > *cap) {
    size_t new_cap = (*cap == 0) ? 64 : (*cap * 2);
    while (new_cap < *len + n + 1) {
      new_cap *= 2;
    }
    char *new_buf = (char *)realloc(*buf, new_cap);
    if (new_buf == NULL) {
      return;
    }
    *buf = new_buf;
    *cap = new_cap;
  }
  memcpy(*buf + *len, tmp, n);
  *len += n;
  (*buf)[*len] = '\0';
}

static int hex4_to_u32(const char *s, unsigned *out) {
  unsigned v = 0;
  for (int i = 0; i < 4; i++) {
    char c = s[i];
    v <<= 4;
    if (c >= '0' && c <= '9') {
      v |= (unsigned)(c - '0');
    } else if (c >= 'a' && c <= 'f') {
      v |= (unsigned)(c - 'a' + 10);
    } else if (c >= 'A' && c <= 'F') {
      v |= (unsigned)(c - 'A' + 10);
    } else {
      return -1;
    }
  }
  *out = v;
  return 0;
}

static char *json_unescape_utf8(const char *in) {
  if (in == NULL) {
    return NULL;
  }
  char *out = NULL;
  size_t len = 0, cap = 0;
  for (const char *p = in; *p != '\0'; p++) {
    if (*p != '\\') {
      /* Keep original UTF-8 bytes as-is. */
      if (len + 2 > cap) {
        size_t new_cap = (cap == 0) ? 64 : (cap * 2);
        while (new_cap < len + 2) {
          new_cap *= 2;
        }
        char *new_buf = (char *)realloc(out, new_cap);
        if (new_buf == NULL) {
          break;
        }
        out = new_buf;
        cap = new_cap;
      }
      out[len++] = *p;
      out[len] = '\0';
      continue;
    }
    p++;
    if (*p == '\0') {
      break;
    }
    switch (*p) {
    case 'n':
      append_utf8(&out, &len, &cap, '\n');
      break;
    case 't':
      append_utf8(&out, &len, &cap, '\t');
      break;
    case 'r':
      append_utf8(&out, &len, &cap, '\r');
      break;
    case '"':
      append_utf8(&out, &len, &cap, '"');
      break;
    case '\\':
      append_utf8(&out, &len, &cap, '\\');
      break;
    case '/':
      append_utf8(&out, &len, &cap, '/');
      break;
    case 'u': {
      if (p[1] == '\0' || p[2] == '\0' || p[3] == '\0' || p[4] == '\0') {
        append_utf8(&out, &len, &cap, 0xFFFD);
        break;
      }
      unsigned cp = 0;
      if (hex4_to_u32(p + 1, &cp) != 0) {
        append_utf8(&out, &len, &cap, 0xFFFD);
        break;
      }
      append_utf8(&out, &len, &cap, cp);
      p += 4;
      break;
    }
    default:
      append_utf8(&out, &len, &cap, (unsigned char)*p);
      break;
    }
  }
  if (out == NULL) {
    out = (char *)malloc(1);
    if (out != NULL) {
      out[0] = '\0';
    }
  }
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
  char *content_json = extract_json_string_field(parse_out.data, parse_out.len, "content");
  char *content = json_unescape_utf8(content_json);

  printf("File Type: %s\n\n\n", mime != NULL ? mime : "(unknown)");
  printf("Content:\n%s\n", content != NULL ? content : "");

  free(mime);
  free(content_json);
  free(content);
  free(bytes);
  vectraparse_result_free(&detect_out);
  vectraparse_result_free(&parse_out);
  vectraparse_destroy_handle(handle);
  return 0;
}
