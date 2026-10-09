//! Optional native inference bridges over resident CUDA tensors.
//! TensorRT and the deterministic test double have separate constructors/DLLs;
//! a missing or invalid trained engine never selects the mock backend.
#![allow(unsafe_code)]

use super::Error;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use libloading::Library;
#[cfg(feature = "neuro-trt")]
use std::ffi::CString;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
#[cfg(feature = "neuro-trt")]
use std::path::Path;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::Arc;

const FEATURES: usize = 32;
const CANDIDATES: usize = 32;
type Create = unsafe extern "C" fn(*const c_char, *mut c_void, *mut *mut c_void) -> c_int;
type Bind =
    unsafe extern "C" fn(*mut c_void, c_uint, *mut c_void, usize, *mut c_void, usize) -> c_int;
type Enqueue = unsafe extern "C" fn(*mut c_void, c_uint, *mut c_void) -> c_int;
type Destroy = unsafe extern "C" fn(*mut c_void) -> c_int;
type LastError = unsafe extern "C" fn() -> *const c_char;

#[derive(Clone, Copy)]
struct Api {
    create: Create,
    bind: Bind,
    enqueue: Enqueue,
    destroy: Destroy,
    last_error: LastError,
    backend: &'static str,
}
impl Api {
    fn load(library: &Library, backend: &'static str) -> Result<Self, Error> {
        // SAFETY: this DLL is built from our C ABI header; pointer types exactly
        // match its exported declarations. RiskEngine retains it while in use.
        unsafe {
            Ok(Self {
                create: *library
                    .get(b"risk_create\0")
                    .map_err(|error| load_error(backend, error))?,
                bind: *library
                    .get(b"risk_bind_slot\0")
                    .map_err(|error| load_error(backend, error))?,
                enqueue: *library
                    .get(b"risk_enqueue\0")
                    .map_err(|error| load_error(backend, error))?,
                destroy: *library
                    .get(b"risk_destroy\0")
                    .map_err(|error| load_error(backend, error))?,
                last_error: *library
                    .get(b"risk_last_error\0")
                    .map_err(|error| load_error(backend, error))?,
                backend,
            })
        }
    }
    fn check(self, status: c_int, operation: &str) -> Result<(), Error> {
        if status == 0 {
            return Ok(());
        }
        // SAFETY: bridge error text is NUL-terminated thread-local storage and
        // is copied immediately, before another bridge call can change it.
        let detail = unsafe {
            let pointer = (self.last_error)();
            if pointer.is_null() {
                "sin detalle".into()
            } else {
                CStr::from_ptr(pointer).to_string_lossy().into_owned()
            }
        };
        Err(Error::Device(format!(
            "{} {operation}: {detail} (código {status})",
            self.backend
        )))
    }
}
fn load_error(backend: &str, error: libloading::Error) -> Error {
    Error::Unavailable(format!("puente {backend}: {error}"))
}
fn cuda_error(error: cudarc::driver::DriverError) -> Error {
    Error::Device(format!("inferencia CUDA: {error}"))
}

#[cfg(feature = "neuro-trt")]
fn model_path(path: &Path) -> Result<CString, Error> {
    let absolute = path
        .canonicalize()
        .map_err(|error| Error::Input(format!("modelo TensorRT {}: {error}", path.display())))?;
    let metadata = absolute.metadata().map_err(|error| {
        Error::Input(format!("modelo TensorRT {}: {error}", absolute.display()))
    })?;
    if !metadata.is_file() {
        return Err(Error::Input(
            "modelo TensorRT debe ser un archivo ONNX, plan o engine".into(),
        ));
    }
    match absolute
        .extension()
        .and_then(|extension| extension.to_str())
    {
        Some(extension)
            if extension.eq_ignore_ascii_case("onnx")
                || extension.eq_ignore_ascii_case("plan")
                || extension.eq_ignore_ascii_case("engine") => {}
        _ => {
            return Err(Error::Input(
                "modelo TensorRT debe terminar en .onnx, .plan o .engine".into(),
            ))
        }
    }
    let utf8 = absolute.to_str().ok_or_else(|| {
        Error::Input("ruta del modelo TensorRT no puede representarse como UTF-8".into())
    })?;
    CString::new(utf8)
        .map_err(|_| Error::Input("ruta del modelo TensorRT contiene un byte NUL".into()))
}

pub struct RiskEngine {
    // Retain all device resources if synchronization/destroy fails. Keeping only
    // the DLL would still permit freeing tensors used by an unfinished launch.
    input: ManuallyDrop<CudaSlice<u16>>,
    output: ManuallyDrop<CudaSlice<f32>>,
    stream: ManuallyDrop<Arc<CudaStream>>,
    context: ManuallyDrop<Arc<CudaContext>>,
    handle: NonNull<c_void>,
    api: Api,
    rows: usize,
    // Unloaded in Drop after destroy, while the CUDA context is still retained.
    library: Option<Library>,
    // One creating host thread owns the mutable execution context and FFI state.
    _single_thread: PhantomData<Rc<()>>,
}
impl RiskEngine {
    #[cfg(feature = "neuro-mock")]
    pub fn mock(context: Arc<CudaContext>, batch_capacity: u32) -> Result<Self, Error> {
        Self::create(
            context,
            batch_capacity,
            env!("RAMSEY43_RISK_DLL"),
            c"mock",
            "mock CUDA",
        )
    }
    /// Read/build the requested ONNX or deserialize a plan/engine once. Subsequent
    /// enqueues reuse this handle and its fixed buffers. The bridge converts X
    /// from FP16 to the model's FP32 input entirely in VRAM on this stream, and
    /// writes FP32 scores directly into Y; Rust never downloads either tensor.
    #[cfg(feature = "neuro-trt")]
    pub fn tensorrt(
        context: Arc<CudaContext>,
        batch_capacity: u32,
        path: &Path,
    ) -> Result<Self, Error> {
        let path = model_path(path)?;
        let dll = option_env!("RAMSEY43_TRT_DLL")
            .or(option_env!("RAMSEY43_RISK_TRT_DLL"))
            .ok_or_else(|| {
                Error::Unavailable("puente TensorRT no configurado al compilar".into())
            })?;
        Self::create(context, batch_capacity, dll, &path, "TensorRT")
    }
    fn create(
        context: Arc<CudaContext>,
        batch_capacity: u32,
        dll: &str,
        path: &CStr,
        backend: &'static str,
    ) -> Result<Self, Error> {
        if !(1..=512).contains(&batch_capacity) {
            return Err(Error::Input(
                "inferencia batch_capacity debe pertenecer a 1..512".into(),
            ));
        }
        context.bind_to_thread().map_err(cuda_error)?;
        let stream = context.new_stream().map_err(cuda_error)?;
        let rows = batch_capacity as usize * CANDIDATES;
        let input = stream.alloc_zeros(rows * FEATURES).map_err(cuda_error)?;
        let output = stream.alloc_zeros(rows * 2).map_err(cuda_error)?;
        // Setup-only synchronization makes allocations available for pointer
        // attribute validation. Per-inference dependency tracking stays async.
        stream.synchronize().map_err(cuda_error)?;
        // SAFETY: locally built bridge selected by a compile-time Cargo OUT_DIR
        // path. Mock and TensorRT use separate DLLs and neither auto-falls back.
        let library = unsafe { Library::new(dll) }.map_err(|error| load_error(backend, error))?;
        let api = Api::load(&library, backend)?;
        let mut handle = std::ptr::null_mut();
        // SAFETY: context remains retained by Arc and is current on this thread.
        // The NUL-terminated path is borrowed only for this call; risk_create
        // loads the model synchronously and owns the resulting runtime/engine.
        let status = unsafe { (api.create)(path.as_ptr(), context.cu_ctx().cast(), &mut handle) };
        api.check(status, "create")?;
        let handle = NonNull::new(handle)
            .ok_or_else(|| Error::Device(format!("{backend} create devolvió handle nulo")))?;
        Ok(Self {
            input: ManuallyDrop::new(input),
            output: ManuallyDrop::new(output),
            stream: ManuallyDrop::new(stream),
            context: ManuallyDrop::new(context),
            handle,
            api,
            rows,
            library: Some(library),
            _single_thread: PhantomData,
        })
    }
    pub fn input_mut(&mut self) -> &mut CudaSlice<u16> {
        &mut self.input
    }
    pub fn output(&self) -> &CudaSlice<f32> {
        &self.output
    }
    pub fn inference_stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
    #[cfg(feature = "neuro-trt")]
    pub fn batch_capacity(&self) -> u32 {
        (self.rows / CANDIDATES) as u32
    }
    /// CUDA self-test entry point; all tested FFI calls reject before launching.
    #[cfg(feature = "neuro-mock")]
    pub fn validate_mock_contracts(context: Arc<CudaContext>) -> Result<usize, Error> {
        let mut engine = Self::mock(context, 1)?;
        let api = engine.api;
        let handle = engine.handle.as_ptr();
        let (input, read_guard) = engine.input.device_ptr(&engine.stream);
        let (output, write_guard) = engine.output.device_ptr_mut(&engine.stream);
        let input = input as *mut c_void;
        let output = output as *mut c_void;
        let input_bytes = CANDIDATES * FEATURES * std::mem::size_of::<u16>();
        let output_bytes = CANDIDATES * 2 * std::mem::size_of::<f32>();
        let mut tested = 0;
        let mut reject = |status, expected, operation: &str| -> Result<(), Error> {
            if status != expected {
                let detail = api
                    .check(status, operation)
                    .err()
                    .map(|error| error.to_string())
                    .unwrap_or_else(|| "aceptación inesperada".into());
                return Err(Error::Device(format!(
                    "contrato {operation}: se esperaba código {expected}, llegó {status}: {detail}"
                )));
            }
            tested += 1;
            Ok(())
        };
        // SAFETY: valid owned handle/context and borrowed device ranges. Invalid
        // values below are specifically rejected before dereference or launch.
        unsafe {
            let mut rejected_handle = std::ptr::null_mut();
            let status = (api.create)(
                b"missing.plan\0".as_ptr().cast(),
                engine.context.cu_ctx().cast(),
                &mut rejected_handle,
            );
            if !rejected_handle.is_null() {
                let _ = (api.destroy)(rejected_handle);
                return Err(Error::Device("engine no mock creó un handle".into()));
            }
            reject(status, 8, "engine no mock")?;
            reject(
                (api.create)(
                    b"mock\0".as_ptr().cast(),
                    std::ptr::null_mut(),
                    &mut rejected_handle,
                ),
                2,
                "contexto nulo",
            )?;
            reject(
                (api.bind)(handle, 2, input, input_bytes, output, output_bytes),
                10,
                "slot fuera de rango",
            )?;
            reject(
                (api.bind)(handle, 0, input, input_bytes - 1, output, output_bytes),
                11,
                "fila FP16 incompleta",
            )?;
            reject(
                (api.bind)(handle, 0, input, input_bytes, output, output_bytes - 4),
                11,
                "shape Y incorrecta",
            )?;
            reject(
                (api.bind)(
                    handle,
                    0,
                    (input as usize + 2) as *mut c_void,
                    input_bytes,
                    output,
                    output_bytes,
                ),
                5,
                "puntero desalineado",
            )?;
            reject(
                (api.bind)(
                    handle,
                    0,
                    std::ptr::null_mut(),
                    input_bytes,
                    output,
                    output_bytes,
                ),
                5,
                "puntero X nulo",
            )?;
            reject(
                (api.bind)(handle, 0, input, input_bytes, input, output_bytes),
                12,
                "tensores superpuestos",
            )?;
            api.check(
                (api.bind)(handle, 0, input, input_bytes, output, output_bytes),
                "bind válido del self-test",
            )?;
            reject(
                (api.bind)(handle, 0, input, input_bytes * 2, output, output_bytes * 2),
                11,
                "batch variable",
            )?;
            reject(
                (api.enqueue)(handle, 0, std::ptr::null_mut()),
                13,
                "stream por defecto",
            )?;
            reject(
                (api.enqueue)(handle, 1, engine.stream.cu_stream().cast()),
                10,
                "slot sin binding",
            )?;
        }
        drop(reject);
        drop(write_guard);
        drop(read_guard);
        Ok(tested)
    }
    pub fn enqueue(&mut self) -> Result<(), Error> {
        self.context.bind_to_thread().map_err(cuda_error)?;
        if self.input.len() != self.rows * FEATURES || self.output.len() != self.rows * 2 {
            return Err(Error::Input(
                "buffers de inferencia cambiaron su shape fija".into(),
            ));
        }
        if self.input.context().cu_ctx() != self.context.cu_ctx()
            || self.output.context().cu_ctx() != self.context.cu_ctx()
        {
            return Err(Error::Input(
                "buffers de inferencia pertenecen a otro contexto CUDA".into(),
            ));
        }
        let input_bytes = self.input.len() * std::mem::size_of::<u16>();
        let output_bytes = self.output.len() * std::mem::size_of::<f32>();
        let (input_pointer, read_guard) = self.input.device_ptr(&self.stream);
        let (output_pointer, write_guard) = self.output.device_ptr_mut(&self.stream);
        // Rebind each time: input_mut permits replacing its CudaSlice in safe
        // Rust. Never retain a stale device address after such a replacement.
        // SAFETY: both ranges are borrowed for this scope and the guard events
        // are recorded only after the kernel read/write has been scheduled.
        let result = unsafe {
            self.api.check(
                (self.api.bind)(
                    self.handle.as_ptr(),
                    0,
                    input_pointer as *mut c_void,
                    input_bytes,
                    output_pointer as *mut c_void,
                    output_bytes,
                ),
                "bind_slot",
            )?;
            self.api.check(
                (self.api.enqueue)(self.handle.as_ptr(), 0, self.stream.cu_stream().cast()),
                "enqueue",
            )
        };
        drop(write_guard);
        drop(read_guard);
        result
    }
}
impl Drop for RiskEngine {
    fn drop(&mut self) {
        // Keep the CUDA code/DLL alive until all its launches are complete.
        if let Err(error) = self
            .context
            .bind_to_thread()
            .and_then(|_| self.stream.synchronize())
        {
            eprintln!(
                "c {} teardown: CUDA synchronize falló: {error}; se conservan recursos CUDA y DLL hasta salir", self.api.backend
            );
            if let Some(library) = self.library.take() {
                std::mem::forget(library);
            }
            return;
        }
        // SAFETY: unique owned handle, correct current context and live stream;
        // all inference work completed and the DLL remains loaded during this call.
        let status = unsafe { (self.api.destroy)(self.handle.as_ptr()) };
        if let Err(error) = self.api.check(status, "destroy") {
            eprintln!(
                "c {} teardown: {error}; se conservan recursos CUDA y DLL hasta salir",
                self.api.backend
            );
            if let Some(library) = self.library.take() {
                std::mem::forget(library);
            }
        } else {
            // Unregister the bridge module while context/stream are retained.
            drop(self.library.take());
            // SAFETY: these fields are manually dropped exactly once, only after
            // a completed stream and successful bridge destroy. On either error
            // above they intentionally remain alive for process lifetime.
            unsafe {
                ManuallyDrop::drop(&mut self.input);
                ManuallyDrop::drop(&mut self.output);
                ManuallyDrop::drop(&mut self.stream);
                ManuallyDrop::drop(&mut self.context);
            }
        }
    }
}

#[cfg(all(test, feature = "neuro-trt"))]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct PathFixture(PathBuf);
    impl PathFixture {
        fn new(extension: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "ramsey43-risk-path-{}-{}.{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed),
                extension
            ));
            // Only path validation is exercised; the bridge must still parse
            // and validate a real network before permitting inference.
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .expect("crear fixture de ruta");
            Self(path)
        }
    }
    impl Drop for PathFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn trained_model_paths_resolve_before_native_setup() {
        for extension in ["onnx", "plan", "engine", "ONNX"] {
            let fixture = PathFixture::new(extension);
            let encoded = model_path(&fixture.0).expect("ruta TensorRT válida");
            let resolved = Path::new(encoded.to_str().expect("ruta UTF-8"));
            assert!(resolved.is_absolute());
            assert_eq!(resolved, fixture.0.canonicalize().unwrap());
        }
    }

    #[test]
    fn invalid_model_paths_reject_before_native_setup() {
        let fixture = PathFixture::new("sat");
        assert!(matches!(model_path(&fixture.0), Err(Error::Input(_))));
        assert!(matches!(
            model_path(&std::env::temp_dir()),
            Err(Error::Input(_))
        ));
        let missing = PathFixture::new("onnx");
        std::fs::remove_file(&missing.0).unwrap();
        assert!(matches!(model_path(&missing.0), Err(Error::Input(_))));
        // The production constructor must never interpret the mock sentinel as
        // an alternate backend, including when that pathname does not exist.
        assert!(matches!(
            model_path(Path::new("mock")),
            Err(Error::Input(_))
        ));
    }
}
