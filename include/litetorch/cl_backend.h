#ifndef LITETORCH_CL_BACKEND_H
#define LITETORCH_CL_BACKEND_H

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <mutex>
#include <memory>
#include <list>

typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_mem;
typedef void* cl_program;
typedef void* cl_kernel;
typedef void* cl_event;
typedef unsigned long cl_mem_flags;
typedef unsigned int cl_uint;
typedef int cl_int;
typedef size_t cl_device_info;

struct cl_buffer_region {
    size_t origin;
    size_t size;
};

namespace litetorch {

constexpr cl_int CL_SUCCESS = 0;
constexpr cl_mem_flags CL_MEM_READ_WRITE = (1 << 0);
constexpr cl_mem_flags CL_MEM_COPY_HOST_PTR = (1 << 5);
constexpr cl_int CL_DEVICE_TYPE_GPU = (1 << 2);
constexpr cl_uint CL_PROGRAM_BUILD_LOG = 0x1183;

enum class KernelID {
    Add,
    AddInplace,
    Sub,
    Mul,
    Div,
    Pow,
    Sigmoid,
    Tanh,
    ReLU,
    GELU,
    Clamp,
    Sin,
    Cos,
    Exp,
    Log,
    Sqrt,
    Abs,
    Neg,
    MatMul,
    BMM,
    EmbeddingForward,
    EmbeddingBackward,
    Conv2dForward,
    Conv2dBackwardInput,
    Conv2dBackwardWeight,
    Conv2dBackwardBias,
    Conv3dForward,
    Conv3dBackwardInput,
    Conv3dBackwardWeight,
    Conv3dBackwardBias,
    MaxPool2dForward,
    MaxPool2dBackward,
    MaxPool3dForward,
    MaxPool3dBackward,
    AdaptiveAvgPool2dForward,
    AdaptiveAvgPool2dBackward,
    FillZero,
    MakeContiguous,
    SGDStep,
    AdamStep,
    RMSpropStep,
    ReluBackward,
    SigmoidBackward,
    TanhBackward,
    LeakyReluForward,
    LeakyReluBackward,
    SoftmaxForward,
    SoftmaxBackward,
    GeluForward,
    GeluBackward,
    LayerNormForward,
    LayerNormBackwardDx,
    LayerNormBackwardDw,
    LayerNormBackwardDb,
    BatchNorm2dForwardStats,
    BatchNorm2dForwardNorm,
    BatchNorm2dBackwardStats,
    BatchNorm2dBackwardDx,
    MseLossForward,
    MseLossBackward,
    L1LossForward,
    L1LossBackward,
    BceLossForward,
    BceLossBackward,
    CrossEntropyLossForward,
    CrossEntropyLossBackward,
    SumForward,
    SumBackward,
    ReduceBroadcastPrepended,
    ReduceBroadcastDim,
    Im2col,
    AddBias2d,
    FusedAddLayerNormForward,
    FakeQuantizeForward,
    FlashAttentionForward,
    RopeForward,
    RopeBackward,
    PagedAttentionForward,
    W8A8MatMul,
    CastFP32ToFP16,
    CastFP16ToFP32,
    CastFP32ToBF16,
    CastBF16ToFP32,
    CastFP32ToNF4,
    CastNF4ToFP32,
    CastFP32ToINT8,
    CastINT8ToFP32,
    CastFP32ToINT4,
    CastINT4ToFP32,
    CastFP32ToFP8E4M3,
    CastFP8E4M3ToFP32,
    CastFP32ToFP8E5M2,
    CastFP8E5M2ToFP32,
    COUNT
};

struct OpenCLCommand {
    cl_kernel kernel;
    std::vector<size_t> global_work_size;
    std::vector<size_t> local_work_size;
    std::vector<std::vector<uint8_t>> arg_data;
    std::vector<size_t> arg_sizes;
};

class OpenCLCommandGraph {
public:
    std::vector<OpenCLCommand> commands;
    bool recording = false;

    ~OpenCLCommandGraph();
    void start_recording();
    void stop_recording();
    void replay();
};

class CLBackend {
public:
    static CLBackend& get();

    std::shared_ptr<OpenCLCommandGraph> active_graph = nullptr;

    bool is_available() const { return available_; }
    cl_command_queue get_queue() const { return queue_; }
    
    cl_mem allocate(size_t size);
    void free(cl_mem buffer);
    void read(cl_mem buffer, size_t size, void* host_ptr, size_t buffer_offset = 0);
    void write(cl_mem buffer, size_t size, const void* host_ptr, size_t buffer_offset = 0);
    void read_async(cl_mem buffer, size_t size, void* host_ptr, size_t buffer_offset = 0);
    void write_async(cl_mem buffer, size_t size, const void* host_ptr, size_t buffer_offset = 0);
    void copy(cl_mem src, cl_mem dst, size_t size, size_t src_offset = 0, size_t dst_offset = 0);

    cl_kernel get_kernel(KernelID id);
    cl_kernel get_kernel(const std::string& program_name, const std::string& program_source, const std::string& kernel_name);
    void launch(cl_kernel kernel, const std::vector<size_t>& global_work_size, const std::vector<size_t>& local_work_size, const std::vector<void*>& args, const std::vector<size_t>& arg_sizes);
    std::shared_ptr<std::mutex> get_kernel_mutex(cl_kernel kernel);
    void finish();
    void clear_cache();

private:
    CLBackend();
    ~CLBackend();
    bool init();
    void shutdown();

    bool available_ = false;
    void* lib_handle_ = nullptr;
    
    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;

    std::unordered_map<std::string, cl_program> programs_;
    std::unordered_map<std::string, cl_kernel> kernels_;
    cl_kernel precompiled_kernels_[static_cast<size_t>(KernelID::COUNT)];
    mutable std::mutex mutex_;
    std::unordered_map<cl_kernel, std::shared_ptr<std::mutex>> kernel_mutexes_;

    struct SuperBlock;
    struct Block {
        size_t offset;
        size_t size;
        bool is_free;
        cl_mem sub_buffer = nullptr;
        SuperBlock* parent = nullptr;
    };
    struct SuperBlock {
        cl_mem buffer = nullptr;
        size_t size = 0;
        std::list<Block> blocks;
    };

    std::vector<std::unique_ptr<SuperBlock>> super_blocks_;
    std::unordered_map<cl_mem, Block*> active_sub_buffers_;
    size_t mem_alignment_bytes_ = 128;

    std::multimap<size_t, cl_mem> free_buffers_;
    std::unordered_map<cl_mem, size_t> allocated_sizes_;
    size_t total_device_allocated_ = 0;
    void evict_cache_to_limit();
};

constexpr cl_uint CL_BUFFER_CREATE_TYPE_REGION = 0x1220;
constexpr cl_device_info CL_DEVICE_MEM_BASE_ADDR_ALIGN = 0x1019;

}

#endif
