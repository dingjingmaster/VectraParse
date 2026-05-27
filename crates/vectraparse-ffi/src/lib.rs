use std::ffi::c_char;
use std::ptr;
use std::slice;

use vectraparse_core::{detect_json, parse_json, CAPABILITIES_JSON};

#[repr(C)]
pub struct VectraParseHandle {
    _private: u8,
}

#[repr(C)]
pub struct VectraParseOptions {
    pub timeout_ms: u32,
    pub max_bytes: usize,
}

#[repr(C)]
pub struct VectraParseResult {
    pub data: *mut u8,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub enum VectraParseError {
    Ok = 0,
    NullPointer = 1,
    InvalidUtf8 = 2,
    Internal = 255,
}

fn alloc_json_result(json: String, out: *mut VectraParseResult) -> VectraParseError {
    if out.is_null() {
        return VectraParseError::NullPointer;
    }
    let mut bytes = json.into_bytes();
    bytes.shrink_to_fit();
    let len = bytes.len();
    let ptr = bytes.as_mut_ptr();
    std::mem::forget(bytes);
    // SAFETY: out was checked for null and points to caller-provided writable storage.
    unsafe {
        (*out).data = ptr;
        (*out).len = len;
    }
    VectraParseError::Ok
}

#[unsafe(no_mangle)]
pub extern "C" fn vectraparse_create_handle(out: *mut *mut VectraParseHandle) -> VectraParseError {
    if out.is_null() {
        return VectraParseError::NullPointer;
    }
    let handle = Box::new(VectraParseHandle { _private: 0 });
    // SAFETY: out was checked for null and points to caller-provided writable storage.
    unsafe {
        *out = Box::into_raw(handle);
    }
    VectraParseError::Ok
}

#[unsafe(no_mangle)]
pub extern "C" fn vectraparse_destroy_handle(handle: *mut VectraParseHandle) {
    if handle.is_null() {
        return;
    }
    // SAFETY: handle came from Box::into_raw in vectraparse_create_handle and is consumed once.
    unsafe {
        drop(Box::from_raw(handle));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn vectraparse_detect(
    handle: *mut VectraParseHandle,
    input: *const u8,
    input_len: usize,
    _options: *const VectraParseOptions,
    out: *mut VectraParseResult,
) -> VectraParseError {
    if handle.is_null() || input.is_null() {
        return VectraParseError::NullPointer;
    }
    // SAFETY: input pointer and length are provided by caller and validated for non-null above.
    let bytes = unsafe { slice::from_raw_parts(input, input_len) };
    alloc_json_result(detect_json(bytes), out)
}

#[unsafe(no_mangle)]
pub extern "C" fn vectraparse_parse(
    handle: *mut VectraParseHandle,
    input: *const u8,
    input_len: usize,
    _options: *const VectraParseOptions,
    out: *mut VectraParseResult,
) -> VectraParseError {
    if handle.is_null() || input.is_null() {
        return VectraParseError::NullPointer;
    }
    // SAFETY: input pointer and length are provided by caller and validated for non-null above.
    let bytes = unsafe { slice::from_raw_parts(input, input_len) };
    alloc_json_result(parse_json(bytes), out)
}

#[unsafe(no_mangle)]
pub extern "C" fn vectraparse_result_free(result: *mut VectraParseResult) {
    if result.is_null() {
        return;
    }
    // SAFETY: result points to caller-owned struct; buffer was allocated by Rust and reclaimed once.
    unsafe {
        let data = (*result).data;
        let len = (*result).len;
        if !data.is_null() && len > 0 {
            let _ = Vec::from_raw_parts(data, len, len);
        }
        (*result).data = ptr::null_mut();
        (*result).len = 0;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn vectraparse_version() -> *const c_char {
    static VERSION: &[u8] = b"0.1.0\0";
    VERSION.as_ptr().cast()
}

#[unsafe(no_mangle)]
pub extern "C" fn vectraparse_capabilities_json(out: *mut VectraParseResult) -> VectraParseError {
    alloc_json_result(CAPABILITIES_JSON.to_string(), out)
}
