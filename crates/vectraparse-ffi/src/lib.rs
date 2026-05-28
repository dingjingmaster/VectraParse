use std::ffi::c_char;
use std::ffi::CStr;
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;

use vectraparse_core::{detect_with_limits_json, parse_with_limits_json, CAPABILITIES_JSON};
use vectraparse_mime::{DetectHints, detect_media_type};

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

fn cstr_opt<'a>(ptr: *const c_char) -> Result<Option<&'a str>, VectraParseError> {
    if ptr.is_null() {
        return Ok(None);
    }
    // SAFETY: caller provides NUL-terminated string pointer when non-null.
    let s = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|_| VectraParseError::InvalidUtf8)?;
    Ok(Some(s))
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
pub extern "C" fn vectraparse_detect_with_hints(
    handle: *mut VectraParseHandle,
    input: *const u8,
    input_len: usize,
    options: *const VectraParseOptions,
    resource_name: *const c_char,
    content_type_hint: *const c_char,
    force_content_type: *const c_char,
    out: *mut VectraParseResult,
) -> VectraParseError {
    if handle.is_null() || input.is_null() {
        return VectraParseError::NullPointer;
    }
    let res_name = match cstr_opt(resource_name) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let type_hint = match cstr_opt(content_type_hint) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let forced = match cstr_opt(force_content_type) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let limit = resolve_limit(options);
    let run = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: input pointer and length are provided by caller and validated for non-null above.
        let bytes = unsafe { slice::from_raw_parts(input, input_len) };
        vectraparse_core::runtime::validate_input_size(
            bytes.len(),
            &vectraparse_core::runtime::ResourceLimits {
                max_input_bytes: limit,
                ..vectraparse_core::runtime::ResourceLimits::default()
            },
        )
        .map_err(|e| format!("{e:?}"))?;
        let mime = detect_media_type(
            bytes,
            &DetectHints {
                resource_name: res_name,
                content_type_hint: type_hint,
                force_content_type: forced,
            },
        );
        Ok::<String, String>(format!(
            "{{\"mime_type\":\"{mime}\",\"metadata\":{{}},\"content\":null,\"embedded\":[],\"warnings\":[],\"error\":null}}"
        ))
    }));
    match run {
        Ok(Ok(json)) => alloc_json_result(json, out),
        Ok(Err(_)) => VectraParseError::Internal,
        Err(_) => VectraParseError::Internal,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn vectraparse_detect_file(
    handle: *mut VectraParseHandle,
    file_path: *const c_char,
    options: *const VectraParseOptions,
    out: *mut VectraParseResult,
) -> VectraParseError {
    if handle.is_null() || file_path.is_null() {
        return VectraParseError::NullPointer;
    }
    let path = match cstr_opt(file_path) {
        Ok(Some(v)) => v,
        Ok(None) => return VectraParseError::NullPointer,
        Err(e) => return e,
    };
    let limit = resolve_limit(options);
    let run = catch_unwind(AssertUnwindSafe(|| {
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        detect_with_limits_json(&bytes, limit)
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
        vectraparse_capabilities_json, vectraparse_create_handle, vectraparse_destroy_handle,
        vectraparse_detect, vectraparse_detect_with_hints, vectraparse_parse, vectraparse_result_free,
        VectraParseError, VectraParseHandle, VectraParseOptions, VectraParseResult,
    };
    use std::ffi::CString;
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

    #[test]
    fn ffi_detect_parse_hints_and_capabilities_roundtrip() {
        let mut handle: *mut VectraParseHandle = ptr::null_mut();
        assert!(matches!(
            vectraparse_create_handle(&mut handle as *mut *mut VectraParseHandle),
            VectraParseError::Ok
        ));
        let input = b"<html><title>x</title></html>";
        let mut out = VectraParseResult {
            data: ptr::null_mut(),
            len: 0,
        };
        assert!(matches!(
            vectraparse_detect(
                handle,
                input.as_ptr(),
                input.len(),
                ptr::null(),
                &mut out as *mut VectraParseResult
            ),
            VectraParseError::Ok
        ));
        vectraparse_result_free(&mut out as *mut VectraParseResult);
        // double free must be safe
        vectraparse_result_free(&mut out as *mut VectraParseResult);

        let rn = CString::new("a.docx").expect("cstr");
        let hint = CString::new("application/octet-stream").expect("cstr");
        let forced = CString::new("text/plain").expect("cstr");
        assert!(matches!(
            vectraparse_detect_with_hints(
                handle,
                input.as_ptr(),
                input.len(),
                ptr::null(),
                rn.as_ptr(),
                hint.as_ptr(),
                forced.as_ptr(),
                &mut out as *mut VectraParseResult
            ),
            VectraParseError::Ok
        ));
        vectraparse_result_free(&mut out as *mut VectraParseResult);

        assert!(matches!(
            vectraparse_parse(
                handle,
                input.as_ptr(),
                input.len(),
                ptr::null(),
                &mut out as *mut VectraParseResult
            ),
            VectraParseError::Ok
        ));
        vectraparse_result_free(&mut out as *mut VectraParseResult);

        assert!(matches!(
            vectraparse_capabilities_json(&mut out as *mut VectraParseResult),
            VectraParseError::Ok
        ));
        vectraparse_result_free(&mut out as *mut VectraParseResult);
        vectraparse_destroy_handle(handle);
    }

    #[test]
    fn ffi_returns_null_pointer_error_on_null_inputs() {
        let mut out = VectraParseResult {
            data: ptr::null_mut(),
            len: 0,
        };
        let rc = vectraparse_detect(
            ptr::null_mut(),
            b"x".as_ptr(),
            1,
            ptr::null(),
            &mut out as *mut VectraParseResult,
        );
        assert!(matches!(rc, VectraParseError::NullPointer));
    }
}
