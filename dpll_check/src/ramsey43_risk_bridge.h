#pragma once
#include <stddef.h>

#ifdef _WIN32
#define RAMSEY43_RISK_API __declspec(dllexport)
#else
#define RAMSEY43_RISK_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Host API only. Handles/pointers/streams belong to the caller's current CUDA
// context. No context is created, retained, reset or destroyed by this bridge.
// TensorRT backend: UTF-8 path ending .onnx (build once), .plan or .engine (load
// once). Model/engine I/O must be FP32 features[rows,32] -> log_cost[rows,2].
// Separate test-only mock DLL accepts only engine_path="mock"; no TRT fallback.
RAMSEY43_RISK_API int risk_create(
    const char* engine_path, void* expected_context, void** handle);
// X is fixed [B*32,32] FP16 and Y [B*32,2] FP32, B=1..512.
// Both borrowed device ranges must stay alive until the enqueue stream finishes.
// Calls for one handle are serialized on its creating host thread.
// A slot retains its first explicit nonblocking stream for its lifetime.
// Identical rebinds are idempotent; changed addresses require completed work.
// Each TRT slot owns its FP32 conversion buffer and execution context. Profile
// index equals slot index: external serialized plans need profile 1 for slot 1.
RAMSEY43_RISK_API int risk_bind_slot(
    void* handle, unsigned slot, void* x_device, size_t x_bytes,
    void* y_device, size_t y_bytes);
RAMSEY43_RISK_API int risk_enqueue(void* handle, unsigned slot, void* cuda_stream);
// Caller must keep any streams passed to enqueue alive until destroy succeeds.
// Failed destroy preserves the handle; retain tensors, streams, context and DLL.
// Failed enqueue poisons the slot; destroy still fences all attempted work.
RAMSEY43_RISK_API int risk_destroy(void* handle);
// Thread-local error text, valid until the next bridge call on this host thread.
RAMSEY43_RISK_API const char* risk_last_error(void);

#ifdef __cplusplus
}
#endif
