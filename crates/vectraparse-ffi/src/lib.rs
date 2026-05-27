use std::ffi::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;

use vectraparse_core::{detect_with_limits_json, parse_with_limits_json, CAPABILITIES_JSON};

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

fn resolve_limit(options: *const VectraParseOptions) -> usize {
    if options.is_null() {
        64 * 1024 * 1024
    } else {
        // SAFETY: options is non-null in this branch and only read.
        let max_bytes = unsafe { (*options).max_bytes };
        if max_bytes == 0 {
            64 * 1024 * 1024
        } else {
            max_bytes
        }
    }
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
    options: *const VectraParseOptions,
    out: *mut VectraParseResult,
) -> VectraParseError {
    if handle.is_null() || input.is_null() {
        return VectraParseError::NullPointer;
    }
    let limit = resolve_limit(options);
    let run = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: input pointer and length are provided by caller and validated for non-null above.
        let bytes = unsafe { slice::from_raw_parts(input, input_len) };
        detect_with_limits_json(bytes, limit)
    }));
    match run {
        Ok(Ok(json)) => alloc_json_result(json, out),
        Ok(Err(_)) => VectraParseError::Internal,
        Err(_) => VectraParseError::Internal,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn vectraparse_parse(
    handle: *mut VectraParseHandle,
    input: *const u8,
    input_len: usize,
    options: *const VectraParseOptions,
    out: *mut VectraParseResult,
) -> VectraParseError {
    if handle.is_null() || input.is_null() {
        return VectraParseError::NullPointer;
    }
    let limit = resolve_limit(options);
    let run = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: input pointer and length are provided by caller and validated for non-null above.
        let bytes = unsafe { slice::from_raw_parts(input, input_len) };
        parse_with_limits_json(bytes, limit)
    }));
    match run {
        Ok(Ok(json)) => alloc_json_result(json, out),
        Ok(Err(_)) => VectraParseError::Internal,
        Err(_) => VectraParseError::Internal,
    }
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

#[cfg(test)]
mod tests {
    use super::{
        vectraparse_create_handle, vectraparse_destroy_handle, vectraparse_detect,
        vectraparse_result_free, VectraParseError, VectraParseHandle, VectraParseOptions,
        VectraParseResult,
    };
    use std::ptr;

    #[test]
    fn detect_respects_max_bytes_limit() {
        let mut handle: *mut VectraParseHandle = ptr::null_mut();
        let rc = vectraparse_create_handle(&mut handle as *mut *mut VectraParseHandle);
        assert!(matches!(rc, VectraParseError::Ok));

        let input = b"abcdef";
        let options = VectraParseOptions {
            timeout_ms: 10,
            max_bytes: 4,
        };
        let mut out = VectraParseResult {
            data: ptr::null_mut(),
            len: 0,
        };
        let rc = vectraparse_detect(
            handle,
            input.as_ptr(),
            input.len(),
            &options as *const VectraParseOptions,
            &mut out as *mut VectraParseResult,
        );
        assert!(matches!(rc, VectraParseError::Internal));
        vectraparse_result_free(&mut out as *mut VectraParseResult);
        vectraparse_destroy_handle(handle);
    }
}
