// Native TensorRT inference, with a separately compiled test-only CUDA mock.
#include "ramsey43_risk_bridge.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(RAMSEY43_RISK_TENSORRT) == defined(RAMSEY43_RISK_MOCK_ONLY)
#error "Select exactly one native risk backend"
#endif
#ifdef RAMSEY43_RISK_TENSORRT
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>
#if NV_TENSORRT_MAJOR != 10 && NV_TENSORRT_MAJOR != 11
#error "The risk bridge supports the TensorRT 10.x and 11.x nvinfer APIs"
#endif
#endif

namespace {
constexpr unsigned kSlots = 2;
constexpr unsigned kFeatures = 32;
constexpr unsigned kCandidates = 32;
constexpr unsigned kMaxRows = 512 * kCandidates;
constexpr uint64_t kMagic = 0x52414d5345593433ull;
thread_local char last_error[1024] = {};

int fail(const char* message, int code = 1) noexcept {
    snprintf(last_error, sizeof(last_error), "%s", message);
    return code;
}
int driver_fail(CUresult result, const char* operation) noexcept {
    const char* message = nullptr;
    cuGetErrorString(result, &message);
    snprintf(last_error, sizeof(last_error), "%s: %s (%d)", operation,
        message ? message : "CUDA driver error", static_cast<int>(result));
    return 1000 + static_cast<int>(result);
}
int runtime_fail(cudaError_t result, const char* operation) noexcept {
    snprintf(last_error, sizeof(last_error), "%s: %s (%d)", operation,
        cudaGetErrorString(result), static_cast<int>(result));
    return 2000 + static_cast<int>(result);
}
template <typename F> int boundary(const char* operation, F&& work) noexcept {
    last_error[0] = '\0';
    try { return work(); }
    catch (const std::exception& error) {
        snprintf(last_error, sizeof(last_error), "%s: %s", operation, error.what());
        return 9;
    } catch (...) {
        snprintf(last_error, sizeof(last_error), "unexpected C++ exception in %s", operation);
        return 9;
    }
}
#ifdef RAMSEY43_RISK_TENSORRT
// TensorRT may log on worker threads. A fixed diagnostic buffer avoids throwing
// allocation exceptions from ILogger::log; the logger outlives all TRT objects.
class Logger final : public nvinfer1::ILogger {
    std::mutex mutex_;
    char message_[768] = {};
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity > Severity::kWARNING) return;
        try {
            std::lock_guard<std::mutex> guard(mutex_);
            snprintf(message_, sizeof(message_), "%s", message ? message : "TensorRT error");
        } catch (...) {}
    }
    void clear() {
        std::lock_guard<std::mutex> guard(mutex_);
        message_[0] = '\0';
    }
    int error(const char* operation) {
        std::lock_guard<std::mutex> guard(mutex_);
        snprintf(last_error, sizeof(last_error), "%s: %s", operation,
            message_[0] ? message_ : "TensorRT returned failure without a diagnostic");
        return 15;
    }
};
struct DeviceBuffer {
    CUdeviceptr pointer = 0;
    ~DeviceBuffer() { if (pointer) cuMemFree(pointer); }
};
#endif
struct CompletionEvent {
    CUevent event = nullptr;
    ~CompletionEvent() { if (event) cuEventDestroy(event); }
};
struct Slot {
    const __half* input = nullptr;
    float* output = nullptr;
    unsigned rows = 0;
    CUstream last_stream = nullptr;
    CompletionEvent completion;
    bool completion_recorded = false;
    bool failed_enqueue = false;
#ifdef RAMSEY43_RISK_TENSORRT
    DeviceBuffer float_input;
    std::unique_ptr<nvinfer1::IExecutionContext> execution;
    bool configured = false;
#endif
};
struct RiskHandle {
    uint64_t magic = kMagic;
    CUcontext context = nullptr;
    CUdevice device = 0;
    unsigned fixed_rows = 0;
    std::thread::id owner = std::this_thread::get_id();
#ifdef RAMSEY43_RISK_TENSORRT
    // Reverse member destruction: slots/contexts, engine, runtime, logger.
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
#endif
    Slot slots[kSlots];
};
int validate_context(CUcontext expected) noexcept {
    CUcontext current = nullptr;
    CUresult result = cuCtxGetCurrent(&current);
    if (result != CUDA_SUCCESS) return driver_fail(result, "cuCtxGetCurrent");
    if (!expected || current != expected)
        return fail("risk bridge requires the caller's expected current CUDA context", 2);
    return 0;
}
int validate_handle(void* opaque, RiskHandle*& handle) noexcept {
    if (!opaque) return fail("null risk handle", 3);
    handle = static_cast<RiskHandle*>(opaque);
    if (handle->magic != kMagic) return fail("invalid risk handle", 3);
    if (handle->owner != std::this_thread::get_id())
        return fail("risk handle used from a different host thread", 4);
    return validate_context(handle->context);
}
int validate_pointer(RiskHandle* handle, void* pointer, size_t bytes) noexcept {
    uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    if (!pointer || !bytes || address % 256 != 0 || bytes > UINTPTR_MAX - address)
        return fail("risk tensor pointer/range is null, misaligned or overflowing", 5);
    CUdeviceptr device_pointer = static_cast<CUdeviceptr>(address);
    CUcontext allocation_context = nullptr;
    unsigned memory_type = 0;
    int ordinal = -1;
    CUdeviceptr range_start = 0;
    size_t range_size = 0;
    CUmemoryPool pool = nullptr;
    CUpointer_attribute attributes[] = {
        CU_POINTER_ATTRIBUTE_CONTEXT, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
        CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
        CU_POINTER_ATTRIBUTE_RANGE_SIZE, CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
    };
    void* values[] = { &allocation_context, &memory_type, &ordinal,
        &range_start, &range_size, &pool };
    CUresult result = cuPointerGetAttributes(6, attributes, values, device_pointer);
    if (result != CUDA_SUCCESS) return driver_fail(result, "cuPointerGetAttributes");
    // Async allocator ranges have a null CONTEXT; require their mempool and device.
    if (memory_type != CU_MEMORYTYPE_DEVICE || ordinal != handle->device
        || (allocation_context && allocation_context != handle->context)
        || (!allocation_context && !pool))
        return fail("risk tensor is not device memory accessible in the expected context", 6);
    if (device_pointer < range_start || device_pointer - range_start > range_size
        || bytes > range_size - static_cast<size_t>(device_pointer - range_start))
        return fail("risk tensor bytes exceed its device allocation", 7);
    return 0;
}
bool overlap(uintptr_t a, size_t a_bytes, uintptr_t b, size_t b_bytes) noexcept {
    return a < b + b_bytes && b < a + a_bytes;
}
int require_idle(Slot& slot) noexcept {
    if (!slot.last_stream) return 0;
    CUresult result = slot.completion_recorded
        ? cuEventQuery(slot.completion.event) : cuStreamQuery(slot.last_stream);
    if (result == CUDA_ERROR_NOT_READY)
        return fail("risk slot tensors cannot change while inference is in flight", 14);
    return result == CUDA_SUCCESS ? 0 : driver_fail(result, "risk rebind completion query");
}
// Record after all attempted enqueues, including failures: the conversion,
// profile setup or part of TensorRT enqueue may already be queued. If recording
// fails, retain the stream and require a stream query/synchronize for teardown.
int record_completion(Slot& slot, int status) noexcept {
    char diagnostic[sizeof(last_error)];
    if (status) memcpy(diagnostic, last_error, sizeof(diagnostic));
    CUresult result = cuEventRecord(slot.completion.event, slot.last_stream);
    slot.completion_recorded = result == CUDA_SUCCESS;
    if (status) {
        memcpy(last_error, diagnostic, sizeof(diagnostic));
        return status;
    }
    return result == CUDA_SUCCESS ? 0 : driver_fail(result, "risk completion event record");
}

#ifdef RAMSEY43_RISK_TENSORRT
constexpr const char* kInputName = "features";
constexpr const char* kOutputName = "log_cost";
constexpr size_t kMaxModelBytes = 512ull * 1024 * 1024;
std::vector<char> read_model(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open TensorRT model/engine file");
    const auto end = file.tellg();
    if (end <= 0 || static_cast<uint64_t>(end) > kMaxModelBytes)
        throw std::runtime_error("model/engine file must contain 1..512 MiB");
    std::vector<char> bytes(static_cast<size_t>(end));
    file.seekg(0, std::ios::beg);
    if (!file.read(bytes.data(), static_cast<std::streamsize>(bytes.size())))
        throw std::runtime_error("failed to read complete model/engine file");
    return bytes;
}
bool dimensions(nvinfer1::Dims shape, int64_t rows, int64_t columns) noexcept {
    return shape.nbDims == 2 && shape.d[0] == rows && shape.d[1] == columns;
}
int validate_engine(RiskHandle* handle) {
    auto& engine = *handle->engine;
    if (engine.getNbIOTensors() != 2 || engine.getNbOptimizationProfiles() < 1)
        return fail("risk engine requires exactly one input, one output and a dynamic profile", 16);
    for (int i = 0; i < 2; ++i) {
        const char* name = engine.getIOTensorName(i);
        if (!name || (strcmp(name, kInputName) && strcmp(name, kOutputName)))
            return fail("risk engine tensor names must be features and log_cost", 16);
        if (engine.getTensorDataType(name) != nvinfer1::DataType::kFLOAT
            || engine.getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE
            || engine.isShapeInferenceIO(name))
            return fail("risk engine tensors must be execution FP32 tensors in device memory", 16);
        for (int p = 0; p < engine.getNbOptimizationProfiles(); ++p) {
            const auto format = engine.getTensorFormat(name, p);
            const int vectorized = engine.getTensorVectorizedDim(name, p);
            const int components = engine.getTensorComponentsPerElement(name, p);
            const int bytes = engine.getTensorBytesPerComponent(name, p);
            // For nonvectorized tensors TensorRT documents -1 for component
            // metadata; some versions report the scalar values 1 / 4 instead.
            // kFLOAT + kLINEAR + no vectorized dimension defines our FP32 rows.
            if (format != nvinfer1::TensorFormat::kLINEAR || vectorized != -1
                || (components != -1 && components != 1)
                || (bytes != -1 && bytes != static_cast<int>(sizeof(float)))) {
                snprintf(last_error, sizeof(last_error),
                    "risk tensor %s profile %d must be linear scalar FP32: format=%d vectorized_dim=%d components=%d bytes=%d",
                    name, p, static_cast<int>(format), vectorized, components, bytes);
                return 16;
            }
        }
    }
    if (engine.getTensorIOMode(kInputName) != nvinfer1::TensorIOMode::kINPUT
        || engine.getTensorIOMode(kOutputName) != nvinfer1::TensorIOMode::kOUTPUT
        || !dimensions(engine.getTensorShape(kInputName), -1, kFeatures)
        || !dimensions(engine.getTensorShape(kOutputName), -1, 2))
        return fail("risk engine requires dynamic features[-1,32] and log_cost[-1,2]", 16);
    for (int p = 0; p < engine.getNbOptimizationProfiles(); ++p) {
        auto low = engine.getProfileShape(kInputName, p, nvinfer1::OptProfileSelector::kMIN);
        auto opt = engine.getProfileShape(kInputName, p, nvinfer1::OptProfileSelector::kOPT);
        auto high = engine.getProfileShape(kInputName, p, nvinfer1::OptProfileSelector::kMAX);
        if (low.nbDims != 2 || opt.nbDims != 2 || high.nbDims != 2
            || low.d[1] != kFeatures || opt.d[1] != kFeatures || high.d[1] != kFeatures
            || low.d[0] < 1 || low.d[0] > opt.d[0] || opt.d[0] > high.d[0]
            || high.d[0] > kMaxRows)
            return fail("risk engine profile rows must remain within 1..16384", 16);
    }
    return 0;
}
int create_engine(RiskHandle* handle, const char* engine_path) {
    // u8path yields a wide Windows path, including canonical \\?\ prefixes.
    const auto path = std::filesystem::u8path(engine_path);
    std::string extension = path.extension().u8string();
    for (char& c : extension) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
    if (extension != ".onnx" && extension != ".plan" && extension != ".engine")
        return fail("TensorRT engine path must end in .onnx, .plan or .engine", 8);
    auto bytes = read_model(path);
    handle->runtime.reset(nvinfer1::createInferRuntime(handle->logger));
    if (!handle->runtime) return handle->logger.error("createInferRuntime");
    // Do not enable engine host code or dynamically load custom plugin libraries.
    if (extension == ".onnx") {
        std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(handle->logger));
        if (!builder) return handle->logger.error("createInferBuilder");
        // TensorRT 10 allows mixed precision through kFP16. TensorRT 11 is
        // always strongly typed: the ONNX model specifies internal precision.
        // Both versions preserve the bridge's existing FP32 I/O contract.
        std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
        if (!network) return handle->logger.error("createNetworkV2");
        // Parser owns weights: keep it alive through buildSerializedNetwork.
        std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, handle->logger));
        if (!parser) return handle->logger.error("createParser");
        if (!parser->parse(bytes.data(), bytes.size())) {
            if (parser->getNbErrors() && parser->getError(0))
                return fail(parser->getError(0)->desc(), 15);
            return handle->logger.error("ONNX parse");
        }
        if (network->getNbInputs() != 1 || network->getNbOutputs() != 1)
            return fail("risk ONNX requires one features input and one log_cost output", 16);
        auto* input = network->getInput(0);
        auto* output = network->getOutput(0);
        if (!input || !output || strcmp(input->getName(), kInputName)
            || strcmp(output->getName(), kOutputName)
            || input->getType() != nvinfer1::DataType::kFLOAT
            || output->getType() != nvinfer1::DataType::kFLOAT
            || !dimensions(input->getDimensions(), -1, kFeatures)
            || !dimensions(output->getDimensions(), -1, 2))
            return fail("risk ONNX must export FP32 features[-1,32] -> log_cost[-1,2]", 16);
        input->setAllowedFormats(1U << static_cast<unsigned>(nvinfer1::TensorFormat::kLINEAR));
        output->setAllowedFormats(1U << static_cast<unsigned>(nvinfer1::TensorFormat::kLINEAR));
        std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        if (!config) return handle->logger.error("createBuilderConfig");
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 256ull * 1024 * 1024);
#if NV_TENSORRT_MAJOR == 10
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
#endif
        config->clearFlag(nvinfer1::BuilderFlag::kTF32);
        // One profile per slot avoids sharing a profile between concurrent
        // contexts. Profiles are builder-owned; they outlive the config/build.
        for (unsigned p = 0; p < kSlots; ++p) {
            auto* profile = builder->createOptimizationProfile();
            if (!profile
                || !profile->setDimensions(kInputName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(32, 32))
                || !profile->setDimensions(kInputName, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(1024, 32))
                || !profile->setDimensions(kInputName, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(kMaxRows, 32))
                || !profile->isValid() || config->addOptimizationProfile(profile) != static_cast<int>(p))
                return handle->logger.error("risk optimization profile");
        }
        std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
        if (!plan) return handle->logger.error("buildSerializedNetwork");
        handle->engine.reset(handle->runtime->deserializeCudaEngine(plan->data(), plan->size()));
    } else {
        handle->engine.reset(handle->runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
    }
    if (!handle->engine) return handle->logger.error("deserializeCudaEngine");
    return validate_engine(handle);
}
__global__ void risk_fp16_to_fp32(const __half* input, float* output, unsigned elements) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = __half2float(input[index]);
}
int infer(RiskHandle* handle, unsigned index, CUstream stream) {
    Slot& slot = handle->slots[index];
    handle->logger.clear();
    if (!slot.configured) {
        if (!slot.execution->setOptimizationProfileAsync(static_cast<int>(index),
                reinterpret_cast<cudaStream_t>(stream)))
            return handle->logger.error("setOptimizationProfileAsync");
        if (!slot.execution->setInputShape(kInputName, nvinfer1::Dims2(slot.rows, kFeatures)))
            return handle->logger.error("setInputShape");
        if (!dimensions(slot.execution->getTensorShape(kOutputName), slot.rows, 2))
            return fail("risk TensorRT context output shape differs from bound Y[rows,2]", 16);
        if (!slot.execution->setTensorAddress(kInputName,
                reinterpret_cast<void*>(static_cast<uintptr_t>(slot.float_input.pointer)))
            || !slot.execution->setTensorAddress(kOutputName, slot.output))
            return handle->logger.error("setTensorAddress");
        slot.configured = true;
    }
    const unsigned elements = slot.rows * kFeatures;
    risk_fp16_to_fp32<<<(elements + 255) / 256, 256, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(slot.input,
            reinterpret_cast<float*>(static_cast<uintptr_t>(slot.float_input.pointer)), elements);
    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) return runtime_fail(result, "risk_fp16_to_fp32 launch");
    if (!slot.execution->enqueueV3(reinterpret_cast<cudaStream_t>(stream)))
        return handle->logger.error("enqueueV3");
    result = cudaGetLastError();
    return result == cudaSuccess ? 0 : runtime_fail(result, "TensorRT enqueueV3 CUDA launch");
}
#else
__global__ void risk_mock_kernel(const __half* input, float* output, unsigned rows) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const __half* features = input + static_cast<size_t>(row) * kFeatures;
    float score = 0.0f;
    bool valid = true;
    for (unsigned feature = 0; feature < kFeatures; ++feature) {
        float value = __half2float(features[feature]);
        valid = valid && isfinite(value) && (feature < 20 || value == 0.0f);
        if (feature < 20) score += fabsf(value) * (0.001f / (feature + 1));
    }
    // Deterministic plumbing fixture; this is never compiled into the TRT DLL.
    score -= 0.001f * static_cast<float>(row % kCandidates);
    output[2 * row] = valid ? score + 0.25f : nanf("");
    output[2 * row + 1] = valid ? score : nanf("");
}
int infer(RiskHandle* handle, unsigned index, CUstream stream) {
    Slot& slot = handle->slots[index];
    risk_mock_kernel<<<(slot.rows + 127) / 128, 128, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(slot.input, slot.output, slot.rows);
    cudaError_t result = cudaGetLastError();
    return result == cudaSuccess ? 0 : runtime_fail(result, "risk_mock_kernel launch");
}
#endif
} // namespace

extern "C" int risk_create(const char* engine_path, void* expected_context, void** out_handle) {
    return boundary("risk_create", [&]() -> int {
        if (!out_handle) return fail("null risk handle output", 3);
        *out_handle = nullptr;
#ifdef RAMSEY43_RISK_MOCK_ONLY
        if (!engine_path || strcmp(engine_path, "mock"))
            return fail("test-only mock bridge requires engine_path=mock", 8);
#else
        if (!engine_path || !engine_path[0] || !strcmp(engine_path, "mock"))
            return fail("production TensorRT bridge requires a real .onnx/.plan/.engine path", 8);
#endif
        CUcontext context = static_cast<CUcontext>(expected_context);
        int error = validate_context(context);
        if (error) return error;
        auto handle = std::make_unique<RiskHandle>();
        handle->context = context;
        CUresult result = cuCtxGetDevice(&handle->device);
        if (result != CUDA_SUCCESS) return driver_fail(result, "cuCtxGetDevice");
#ifdef RAMSEY43_RISK_TENSORRT
        error = create_engine(handle.get(), engine_path);
        if (error) return error;
#endif
        *out_handle = handle.release();
        return 0;
    });
}

extern "C" int risk_bind_slot(void* opaque, unsigned index, void* x_device, size_t x_bytes,
    void* y_device, size_t y_bytes) {
    return boundary("risk_bind_slot", [&]() -> int {
        RiskHandle* handle = nullptr;
        int error = validate_handle(opaque, handle);
        if (error) return error;
        if (index >= kSlots) return fail("risk slot must be 0 or 1", 10);
        if (!x_bytes || x_bytes % (kFeatures * sizeof(__half)))
            return fail("risk X must contain complete 32-feature FP16 rows", 11);
        size_t rows = x_bytes / (kFeatures * sizeof(__half));
        if (rows % kCandidates || rows > kMaxRows || y_bytes != rows * 2 * sizeof(float))
            return fail("risk shapes require X[B*32,32] and Y[B*32,2], B=1..512", 11);
        if (handle->fixed_rows && handle->fixed_rows != rows)
            return fail("risk batch shape is fixed for the handle lifetime", 11);
        error = validate_pointer(handle, x_device, x_bytes);
        if (error) return error;
        error = validate_pointer(handle, y_device, y_bytes);
        if (error) return error;
        uintptr_t x = reinterpret_cast<uintptr_t>(x_device), y = reinterpret_cast<uintptr_t>(y_device);
        if (overlap(x, x_bytes, y, y_bytes))
            return fail("risk input and output ranges must not overlap", 12);
        for (unsigned other = 0; other < kSlots; ++other) {
            if (other == index || !handle->slots[other].input) continue;
            auto& binding = handle->slots[other];
            auto ox = reinterpret_cast<uintptr_t>(binding.input), oy = reinterpret_cast<uintptr_t>(binding.output);
            if (overlap(y, y_bytes, ox, binding.rows * kFeatures * sizeof(__half))
                || overlap(y, y_bytes, oy, binding.rows * 2 * sizeof(float))
                || overlap(x, x_bytes, oy, binding.rows * 2 * sizeof(float)))
                return fail("risk slots must not alias writable device tensor ranges", 12);
        }
        Slot& slot = handle->slots[index];
        // Repeated Rust bind of identical owned buffers does not mutate an
        // execution context while it is running; ordered same-stream reuse is safe.
        if (slot.input == x_device && slot.output == y_device && slot.rows == rows) return 0;
        error = require_idle(slot);
        if (error) return error;
#ifdef RAMSEY43_RISK_TENSORRT
        if (index >= static_cast<unsigned>(handle->engine->getNbOptimizationProfiles()))
            return fail("serialized risk engine needs a separate optimization profile for this slot", 16);
        auto low = handle->engine->getProfileShape(kInputName, index, nvinfer1::OptProfileSelector::kMIN);
        auto high = handle->engine->getProfileShape(kInputName, index, nvinfer1::OptProfileSelector::kMAX);
        if (rows < static_cast<size_t>(low.d[0]) || rows > static_cast<size_t>(high.d[0]))
            return fail("bound risk rows are outside the serialized optimization profile", 11);
        if (!slot.execution) {
            slot.execution.reset(handle->engine->createExecutionContext());
            if (!slot.execution) return handle->logger.error("createExecutionContext");
        }
        if (!slot.float_input.pointer) {
            // Reserve the bounded profile maximum, including retries after a
            // partially failed bind. A changed first-bind size cannot overflow it.
            CUresult result = cuMemAlloc(&slot.float_input.pointer, kMaxRows * kFeatures * sizeof(float));
            if (result != CUDA_SUCCESS) return driver_fail(result, "risk FP32 staging allocation");
        }
        // Changing a pointer happens only after the completion query above.
        // Defer tensor-address/profile setup to the caller's validated stream.
        slot.configured = false;
#endif
        if (!slot.completion.event) {
            CUresult result = cuEventCreate(&slot.completion.event, CU_EVENT_DISABLE_TIMING);
            if (result != CUDA_SUCCESS) return driver_fail(result, "risk completion event create");
        }
        handle->fixed_rows = static_cast<unsigned>(rows);
        slot.input = static_cast<const __half*>(x_device);
        slot.output = static_cast<float*>(y_device);
        slot.rows = static_cast<unsigned>(rows);
        return 0;
    });
}

extern "C" int risk_enqueue(void* opaque, unsigned index, void* cuda_stream) {
    return boundary("risk_enqueue", [&]() -> int {
        RiskHandle* handle = nullptr;
        int error = validate_handle(opaque, handle);
        if (error) return error;
        if (index >= kSlots || !handle->slots[index].input || !handle->slots[index].output)
            return fail("risk slot is invalid or has no bound tensors", 10);
        CUstream stream = static_cast<CUstream>(cuda_stream);
        if (!stream) return fail("risk inference needs an explicit nonblocking stream", 13);
        CUcontext stream_context = nullptr;
        CUresult result = cuStreamGetCtx(stream, &stream_context);
        if (result != CUDA_SUCCESS) return driver_fail(result, "cuStreamGetCtx");
        unsigned flags = 0;
        result = cuStreamGetFlags(stream, &flags);
        if (result != CUDA_SUCCESS) return driver_fail(result, "cuStreamGetFlags");
        if (stream_context != handle->context || !(flags & CU_STREAM_NON_BLOCKING))
            return fail("risk stream must be nonblocking and belong to the expected context", 13);
        Slot& slot = handle->slots[index];
        if (slot.failed_enqueue)
            return fail("risk slot had a failed enqueue and must be destroyed", 14);
        if (slot.last_stream && slot.last_stream != stream)
            return fail("risk slot must keep the same inference stream for its lifetime", 13);
        slot.last_stream = stream;
        slot.completion_recorded = false;
        slot.failed_enqueue = true;
        // If a C++ exception follows partial scheduling, the stream still stays
        // retained. destroy fences it; require_idle falls back to cuStreamQuery.
        const int status = record_completion(slot, infer(handle, index, stream));
        slot.failed_enqueue = status != 0;
        return status;
    });
}

extern "C" int risk_destroy(void* opaque) {
    return boundary("risk_destroy", [&]() -> int {
        RiskHandle* handle = nullptr;
        int error = validate_handle(opaque, handle);
        if (error) return error;
        for (unsigned index = 0; index < kSlots; ++index) {
            if (!handle->slots[index].last_stream) continue;
            CUresult result = cuStreamSynchronize(handle->slots[index].last_stream);
            if (result != CUDA_SUCCESS) return driver_fail(result, "risk_destroy synchronize");
        }
        // Failed fencing above deliberately leaves the entire handle intact.
        // Caller must retain context, streams, tensors and DLL in that case.
        handle->magic = 0;
        delete handle;
        return 0;
    });
}
extern "C" const char* risk_last_error(void) { return last_error; }
