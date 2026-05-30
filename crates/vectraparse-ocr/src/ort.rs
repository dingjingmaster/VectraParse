use std::ffi::{c_char, c_void, CString};
use std::sync::OnceLock;

#[derive(Debug, Copy, Clone)]
struct ApiPtr(*const ());
unsafe impl Send for ApiPtr {}
unsafe impl Sync for ApiPtr {}

static API_PTR: OnceLock<ApiPtr> = OnceLock::new();
static LAST_SHAPE: OnceLock<Vec<usize>> = OnceLock::new();

pub(crate) fn last_known_shape() -> Vec<usize> {
    LAST_SHAPE.get().cloned().unwrap_or_default()
}

pub(crate) fn ensure_initialized() -> Result<(), String> {
    if API_PTR.get().is_some() {
        return Ok(());
    }
    let ptr = get_api_ptr()?;
    API_PTR.set(ptr).map_err(|_| "already initialized".to_string())
}

fn get_api_ptr() -> Result<ApiPtr, String> {
    let base = unsafe { OrtGetApiBase() };
    if base.is_null() {
        return Err("OrtGetApiBase returned null".to_string());
    }
    let get_api_fn = unsafe { (*base).GetApi };
    let get_api = get_api_fn.ok_or("GetApi is null")?;
    let api = unsafe { get_api(ORT_API_VERSION as u32) };
    if api.is_null() {
        return Err(format!("GetApi(version {}) returned null", ORT_API_VERSION));
    }
    Ok(ApiPtr(api))
}

fn api_fn<T>(offset: usize) -> T {
    let ApiPtr(ptr) = *API_PTR.get().expect("ort not initialized");
    unsafe {
        let byte_ptr = ptr as usize;
        let field_ptr = (byte_ptr + offset * std::mem::size_of::<usize>()) as *const T;
        std::ptr::read(field_ptr)
    }
}

fn check_status(status: *mut c_void) -> Result<(), String> {
    if status.is_null() {
        return Ok(());
    }
    let get_error_msg: unsafe extern "C" fn(*const c_void) -> *const c_char = api_fn(2);
    let release_status: unsafe extern "C" fn(*mut c_void) = api_fn(93);
    let msg_ptr = unsafe { get_error_msg(status) };
    let msg = if msg_ptr.is_null() {
        "unknown error".to_string()
    } else {
        unsafe { std::ffi::CStr::from_ptr(msg_ptr) }
            .to_string_lossy()
            .into_owned()
    };
    unsafe { release_status(status) };
    Err(msg)
}

fn char_p_to_string(ptr: *const c_char) -> String {
    unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

pub(crate) fn create_session_from_memory(model_bytes: &[u8]) -> Result<*mut c_void, String> {
    let create_session_options: unsafe extern "C" fn(*mut *mut c_void) -> *mut c_void = api_fn(10);
    let set_graph_opt: unsafe extern "C" fn(*mut c_void, u32) -> *mut c_void = api_fn(23);
    let set_intra_threads: unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void = api_fn(24);
    let create_env: unsafe extern "C" fn(u32, *const c_char, *mut *mut c_void) -> *mut c_void =
        api_fn(3);
    let create_session_from_array: unsafe extern "C" fn(
        *const c_void, *const c_void, usize, *mut c_void, *mut *mut c_void,
    ) -> *mut c_void = api_fn(8);
    let release_session_options: unsafe extern "C" fn(*mut c_void) = api_fn(100);
    let release_env: unsafe extern "C" fn(*mut c_void) = api_fn(92);

    let mut session_options: *mut c_void = std::ptr::null_mut();
    let status = unsafe { create_session_options(&mut session_options) };
    check_status(status)?;

    unsafe {
        set_graph_opt(session_options, 0);
        set_intra_threads(session_options, 1);
    }

    let cname = CString::new("vectraparse-ocr").unwrap();
    let mut env_ptr: *mut c_void = std::ptr::null_mut();
    let status = unsafe { create_env(2, cname.as_ptr(), &mut env_ptr) };
    check_status(status)?;

    let mut session_ptr: *mut c_void = std::ptr::null_mut();
    let status = unsafe {
        create_session_from_array(
            env_ptr,
            model_bytes.as_ptr() as *const c_void,
            model_bytes.len(),
            session_options,
            &mut session_ptr,
        )
    };
    let result = check_status(status);
    unsafe { release_session_options(session_options) };
    unsafe { release_env(env_ptr) };
    result?;

    if session_ptr.is_null() {
        return Err("CreateSessionFromArray returned null".into());
    }
    Ok(session_ptr)
}

pub(crate) fn create_allocator() -> Result<*mut c_void, String> {
    let get_allocator: unsafe extern "C" fn(*mut *mut c_void) -> *mut c_void = api_fn(78);
    let mut allocator: *mut c_void = std::ptr::null_mut();
    let status = unsafe { get_allocator(&mut allocator) };
    check_status(status)?;
    if allocator.is_null() {
        return Err("GetAllocatorWithDefaultOptions returned null".into());
    }
    Ok(allocator)
}

pub(crate) fn create_memory_info() -> Result<*mut c_void, String> {
    let create_cpu: unsafe extern "C" fn(u32, u32, *mut *mut c_void) -> *mut c_void = api_fn(69);
    let mut mem_info: *mut c_void = std::ptr::null_mut();
    let status = unsafe { create_cpu(1, 0, &mut mem_info) };
    check_status(status)?;
    if mem_info.is_null() {
        return Err("CreateCpuMemoryInfo returned null".into());
    }
    Ok(mem_info)
}

pub(crate) fn release_session(session: *mut c_void) {
    if !session.is_null() {
        let rel: unsafe extern "C" fn(*mut c_void) = api_fn(95);
        unsafe { rel(session) };
    }
}

pub(crate) fn release_allocator(_allocator: *mut c_void) {}

pub(crate) fn release_memory_info(mem_info: *mut c_void) {
    if !mem_info.is_null() {
        let rel: unsafe extern "C" fn(*mut c_void) = api_fn(94);
        unsafe { rel(mem_info) };
    }
}

pub(crate) fn run_session(
    session: &super::OrtSession,
    inputs: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, String> {
    let sptr = session.session_ptr;
    let alloc = session.allocator_ptr;
    let meminfo = session.memory_info_ptr;

    let fn_input_count: unsafe extern "C" fn(*const c_void, *mut usize) -> *mut c_void = api_fn(30);
    let fn_input_name: unsafe extern "C" fn(
        *const c_void, usize, *mut c_void, *mut *mut c_char,
    ) -> *mut c_void = api_fn(36);
    let fn_input_type: unsafe extern "C" fn(
        *const c_void, usize, *mut *mut c_void,
    ) -> *mut c_void = api_fn(33);
    let fn_alloc_free: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void = api_fn(76);
    let fn_cast_tensor_info: unsafe extern "C" fn(
        *const c_void, *mut *const c_void,
    ) -> *mut c_void = api_fn(55);
    let fn_dim_count: unsafe extern "C" fn(*const c_void, *mut usize) -> *mut c_void = api_fn(61);
    let fn_dim: unsafe extern "C" fn(*const c_void, *mut i64, usize) -> *mut c_void = api_fn(62);
    let fn_rel_type_info: unsafe extern "C" fn(*mut c_void) = api_fn(98);
    let fn_create_tensor: unsafe extern "C" fn(
        *const c_void, *mut c_void, usize, *const i64, usize, u32, *mut *mut c_void,
    ) -> *mut c_void = api_fn(49);
    let fn_output_count: unsafe extern "C" fn(*const c_void, *mut usize) -> *mut c_void =
        api_fn(31);
    let fn_output_name: unsafe extern "C" fn(
        *const c_void, usize, *mut c_void, *mut *mut c_char,
    ) -> *mut c_void = api_fn(37);
    let fn_run: unsafe extern "C" fn(
        *mut c_void, *const c_void, *const *const c_char, *const *const c_void, usize,
        *const *const c_char, usize, *mut *mut c_void,
    ) -> *mut c_void = api_fn(9);
    let fn_mutable_data: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> *mut c_void =
        api_fn(51);
    let fn_tensor_shape: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> *mut c_void =
        api_fn(65);
    let fn_rel_tensor_shape: unsafe extern "C" fn(*mut c_void) = api_fn(99);
    let fn_release_value: unsafe extern "C" fn(*mut c_void) = api_fn(96);

    let num_inputs = {
        let mut count: usize = 0;
        let status = unsafe { fn_input_count(sptr, &mut count) };
        check_status(status)?;
        count
    };
    if num_inputs != inputs.len() {
        return Err(format!(
            "model expects {num_inputs} inputs, got {}",
            inputs.len()
        ));
    }

    let mut input_tensors: Vec<*mut c_void> = Vec::new();
    let mut input_name_cstrs: Vec<CString> = Vec::new();
    let mut input_name_ptrs: Vec<*const c_char> = Vec::new();

    for i in 0..num_inputs {
        let mut name_ptr: *mut c_char = std::ptr::null_mut();
        let status = unsafe { fn_input_name(sptr, i, alloc, &mut name_ptr) };
        check_status(status)?;
        let cname = CString::new(char_p_to_string(name_ptr)).unwrap();
        unsafe { fn_alloc_free(alloc, name_ptr as *mut c_void) };
        let ptr = cname.as_ptr();
        input_name_cstrs.push(cname);
        input_name_ptrs.push(ptr);
    }

    for (i, data) in inputs.iter().enumerate() {
        let shape = read_input_shape(
            sptr,
            i,
            fn_input_type,
            fn_cast_tensor_info,
            fn_dim_count,
            fn_dim,
            fn_rel_type_info,
        )?;
        let flat: Vec<i64> = shape.iter().map(|d| *d as i64).collect();
        let mut tensor: *mut c_void = std::ptr::null_mut();
        let status = unsafe {
            fn_create_tensor(
                meminfo,
                data.as_ptr() as *mut c_void,
                data.len() * 4,
                flat.as_ptr(),
                flat.len(),
                1,
                &mut tensor,
            )
        };
        check_status(status)?;
        input_tensors.push(tensor);
    }

    let input_vals: Vec<*const c_void> = input_tensors.iter().map(|t| *t as *const c_void).collect();

    let num_outputs = {
        let mut count: usize = 0;
        let status = unsafe { fn_output_count(sptr, &mut count) };
        check_status(status)?;
        count
    };

    let mut output_name_cstrs: Vec<CString> = Vec::new();
    let mut output_name_ptrs: Vec<*const c_char> = Vec::new();
    for i in 0..num_outputs {
        let mut name_ptr: *mut c_char = std::ptr::null_mut();
        let status = unsafe { fn_output_name(sptr, i, alloc, &mut name_ptr) };
        check_status(status)?;
        let cname = CString::new(char_p_to_string(name_ptr)).unwrap();
        unsafe { fn_alloc_free(alloc, name_ptr as *mut c_void) };
        let ptr = cname.as_ptr();
        output_name_cstrs.push(cname);
        output_name_ptrs.push(ptr);
    }

    let mut output_tensors: Vec<*mut c_void> = vec![std::ptr::null_mut(); num_outputs];

    let status = unsafe {
        fn_run(
            sptr,
            std::ptr::null(),
            input_name_ptrs.as_ptr(),
            input_vals.as_ptr(),
            input_vals.len(),
            output_name_ptrs.as_ptr(),
            output_name_ptrs.len(),
            output_tensors.as_mut_ptr(),
        )
    };
    check_status(status)?;

    let mut results = Vec::new();
    for tensor in output_tensors.iter() {
        let data = extract_f32(
            *tensor,
            fn_mutable_data,
            fn_tensor_shape,
            fn_dim_count,
            fn_dim,
            fn_rel_tensor_shape,
        )?;
        record_shape(*tensor, fn_tensor_shape, fn_dim_count, fn_dim, fn_rel_tensor_shape)?;
        unsafe { fn_release_value(*tensor) };
        results.push(data);
    }

    for tensor in input_tensors {
        unsafe { fn_release_value(tensor) };
    }

    Ok(results)
}

fn record_shape(
    tensor: *mut c_void,
    gts: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> *mut c_void,
    gdc: unsafe extern "C" fn(*const c_void, *mut usize) -> *mut c_void,
    gd: unsafe extern "C" fn(*const c_void, *mut i64, usize) -> *mut c_void,
    rel: unsafe extern "C" fn(*mut c_void),
) -> Result<(), String> {
    let mut ti: *mut c_void = std::ptr::null_mut();
    let status = unsafe { gts(tensor, &mut ti) };
    check_status(status)?;
    let mut nd: usize = 0;
    unsafe { gdc(ti, &mut nd) };
    let mut dims: Vec<i64> = vec![0; nd];
    unsafe { gd(ti, dims.as_mut_ptr(), nd) };
    let shape: Vec<usize> = dims.iter().map(|d| *d as usize).collect();
    unsafe {
        rel(ti);
        let _ = LAST_SHAPE.set(shape);
    }
    Ok(())
}

fn extract_f32(
    tensor: *mut c_void,
    gmd: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> *mut c_void,
    gts: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> *mut c_void,
    gdc: unsafe extern "C" fn(*const c_void, *mut usize) -> *mut c_void,
    gd: unsafe extern "C" fn(*const c_void, *mut i64, usize) -> *mut c_void,
    rel: unsafe extern "C" fn(*mut c_void),
) -> Result<Vec<f32>, String> {
    let mut data_ptr: *mut c_void = std::ptr::null_mut();
    let status = unsafe { gmd(tensor, &mut data_ptr) };
    check_status(status)?;

    let mut ti: *mut c_void = std::ptr::null_mut();
    let status = unsafe { gts(tensor, &mut ti) };
    check_status(status)?;

    let mut nd: usize = 0;
    unsafe { gdc(ti, &mut nd) };
    let mut dims: Vec<i64> = vec![0; nd];
    unsafe { gd(ti, dims.as_mut_ptr(), nd) };
    let total: usize = dims.iter().map(|d| *d as usize).product();
    unsafe { rel(ti) };

    let slice = unsafe { std::slice::from_raw_parts(data_ptr as *const f32, total) };
    Ok(slice.to_vec())
}

fn read_input_shape(
    sptr: *mut c_void,
    idx: usize,
    get_type: unsafe extern "C" fn(*const c_void, usize, *mut *mut c_void) -> *mut c_void,
    cast: unsafe extern "C" fn(*const c_void, *mut *const c_void) -> *mut c_void,
    gdc: unsafe extern "C" fn(*const c_void, *mut usize) -> *mut c_void,
    gd: unsafe extern "C" fn(*const c_void, *mut i64, usize) -> *mut c_void,
    rel: unsafe extern "C" fn(*mut c_void),
) -> Result<Vec<usize>, String> {
    let mut type_info: *mut c_void = std::ptr::null_mut();
    let status = unsafe { get_type(sptr, idx, &mut type_info) };
    check_status(status)?;

    let mut tensor_info: *const c_void = std::ptr::null_mut();
    let status = unsafe { cast(type_info, &mut tensor_info) };
    check_status(status)?;

    let mut nd: usize = 0;
    unsafe { gdc(tensor_info, &mut nd) };
    let mut dims: Vec<i64> = vec![0; nd];
    unsafe { gd(tensor_info, dims.as_mut_ptr(), nd) };
    let shape: Vec<usize> = dims.iter().map(|d| if *d < 0 { 1 } else { *d as usize }).collect();
    unsafe { rel(type_info) };
    Ok(shape)
}

pub(crate) type OrtSession = c_void;
pub(crate) type OrtAllocator = c_void;
pub(crate) type OrtMemoryInfo = c_void;
pub(crate) type _OrtValue = c_void;

const ORT_API_VERSION: u32 = 26;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
#[allow(non_snake_case)]
struct OrtApiBase {
    GetApi: Option<unsafe extern "C" fn(u32) -> *const ()>,
    GetVersionString: Option<unsafe extern "C" fn() -> *const c_char>,
}

unsafe extern "C" {
    fn OrtGetApiBase() -> *const OrtApiBase;
}
