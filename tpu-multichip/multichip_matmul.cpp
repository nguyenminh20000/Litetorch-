#include "pjrt_c_api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <dlfcn.h>
#include <glob.h>
#include <unistd.h>

static const PJRT_Api* g_api = nullptr;

static const unsigned char kCompileOptionsProto18[] = {0x1A, 0x04, 0x20, 0x01, 0x28, 0x08};
static const unsigned char kCompileOptionsProto11[] = {0x1A, 0x04, 0x20, 0x01, 0x28, 0x01};

static bool check_err(PJRT_Error* err, const char* what) {
    if (!err) return true;
    std::string msg = "unknown";
    if (g_api && g_api->PJRT_Error_Message) {
        PJRT_Error_Message_Args margs{};
        margs.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
        margs.error = err;
        g_api->PJRT_Error_Message(&margs);
        if (margs.message) msg.assign(margs.message, margs.message_size);
    }
    std::fprintf(stderr, "MULTICHIP_FAIL %s: %s\n", what, msg.c_str());
    if (g_api && g_api->PJRT_Error_Destroy) {
        PJRT_Error_Destroy_Args dargs{};
        dargs.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
        dargs.error = err;
        g_api->PJRT_Error_Destroy(&dargs);
    }
    return false;
}

static bool await_event(PJRT_Event* ev, const char* what) {
    if (!ev) return true;
    PJRT_Event_Await_Args aargs{};
    aargs.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    aargs.event = ev;
    bool ok = check_err(g_api->PJRT_Event_Await(&aargs), what);
    PJRT_Event_Destroy_Args dargs{};
    dargs.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    dargs.event = ev;
    PJRT_Error* derr = g_api->PJRT_Event_Destroy(&dargs);
    if (derr) check_err(derr, "Event_Destroy");
    return ok;
}

static void destroy_buffer(PJRT_Buffer* buf) {
    if (!buf) return;
    PJRT_Buffer_Destroy_Args args{};
    args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    args.buffer = buf;
    PJRT_Error* err = g_api->PJRT_Buffer_Destroy(&args);
    if (err) check_err(err, "Buffer_Destroy");
}

static std::string find_libtpu(const std::string& explicit_path) {
    if (!explicit_path.empty()) return explicit_path;
    const char* env_names[] = {"TPU_LIBRARY_PATH", "LIBTPU_PATH", nullptr};
    for (int i = 0; env_names[i]; ++i) {
        const char* p = std::getenv(env_names[i]);
        if (p && p[0]) return std::string(p);
    }
    const char* candidates[] = {
        "/lib/libtpu.so",
        "/usr/lib/libtpu.so",
        "/usr/local/lib/libtpu.so",
        "libtpu.so",
        nullptr};
    for (int i = 0; candidates[i]; ++i) {
        if (access(candidates[i], R_OK) == 0) return candidates[i];
        void* h = dlopen(candidates[i], RTLD_NOW | RTLD_NOLOAD);
        if (h) { dlclose(h); return std::string(candidates[i]); }
    }
    const char* home = std::getenv("HOME");
    std::vector<std::string> patterns = {
        "/usr/local/lib/python3.*/dist-packages/libtpu/libtpu.so*",
        "/usr/local/lib/python3.*/site-packages/libtpu/libtpu.so*",
        "/usr/lib/python3/dist-packages/libtpu/libtpu.so*",
        "/opt/conda/lib/python3.*/site-packages/libtpu/libtpu.so*"};
    if (home && home[0]) {
        patterns.push_back(std::string(home) + "/.local/lib/python3.*/site-packages/libtpu/libtpu.so*");
    }
    for (const auto& pat : patterns) {
        glob_t g{};
        if (glob(pat.c_str(), GLOB_NOSORT, nullptr, &g) != 0) continue;
        std::string hit;
        for (size_t j = 0; j < g.gl_pathc; ++j) {
            if (access(g.gl_pathv[j], R_OK) == 0) { hit = g.gl_pathv[j]; break; }
        }
        globfree(&g);
        if (!hit.empty()) return hit;
    }
    return "";
}

static uint16_t float_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t rounding_bias = 0x7FFFu + lsb;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

static float bf16_to_float(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static double now_ms() {
    auto t = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t.time_since_epoch()).count();
}

static bool load_f32_file(const std::string& path, std::vector<float>& dst) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    size_t got = std::fread(dst.data(), sizeof(float), dst.size(), f);
    std::fclose(f);
    return got == dst.size();
}

static bool save_f32_file(const std::string& path, const float* data, size_t n) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t put = std::fwrite(data, sizeof(float), n, f);
    std::fclose(f);
    return put == n;
}

static std::string build_mlir_allreduce(int64_t s, int partitions) {
    std::string groups;
    for (int i = 0; i < partitions; ++i) {
        if (i) groups += ", ";
        char tmp[16];
        std::snprintf(tmp, sizeof(tmp), "%d", i);
        groups += tmp;
    }
    char buf[4096];
    std::snprintf(buf, sizeof(buf),
        "module @multichip_allreduce {\n"
        "  func.func public @main(%%arg0: tensor<%lldx%lldxf32> {mhlo.sharding = \"{replicated}\"}) -> tensor<%lldx%lldxf32> {\n"
        "    %%0 = \"stablehlo.all_reduce\"(%%arg0) ({\n"
        "      ^bb0(%%a: tensor<f32>, %%b: tensor<f32>):\n"
        "        %%s = stablehlo.add %%a, %%b : tensor<f32>\n"
        "        stablehlo.return %%s : tensor<f32>\n"
        "    }) {replica_groups = dense<[[%s]]> : tensor<1x%dxi64>,\n"
        "        channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>} : (tensor<%lldx%lldxf32>) -> tensor<%lldx%lldxf32>\n"
        "    func.return %%0 : tensor<%lldx%lldxf32>\n"
        "  }\n"
        "}\n",
        (long long)s, (long long)s,
        (long long)s, (long long)s,
        groups.c_str(), partitions,
        (long long)s, (long long)s,
        (long long)s, (long long)s,
        (long long)s, (long long)s);
    return std::string(buf);
}

static std::string build_mlir_matmul(int64_t m, int64_t k, int64_t n, const char* etype, int mode) {
    char buf[4096];
    if (mode == 0) {
        std::snprintf(buf, sizeof(buf),
            "module @multichip_matmul {\n"
            "  func.func public @main(%%arg0: tensor<%lldx%lldx%s>, %%arg1: tensor<%lldx%lldx%s>) -> tensor<%lldx%lldx%s> {\n"
            "    %%0 = stablehlo.dot_general %%arg0, %%arg1, contracting_dims = [1] x [0] : (tensor<%lldx%lldx%s>, tensor<%lldx%lldx%s>) -> tensor<%lldx%lldx%s>\n"
            "    func.return %%0 : tensor<%lldx%lldx%s>\n"
            "  }\n"
            "}\n",
            (long long)m, (long long)k, etype,
            (long long)k, (long long)n, etype,
            (long long)m, (long long)n, etype,
            (long long)m, (long long)k, etype,
            (long long)k, (long long)n, etype,
            (long long)m, (long long)n, etype,
            (long long)m, (long long)n, etype);
    } else if (mode == 1) {
        std::snprintf(buf, sizeof(buf),
            "module @multichip_matmul_rowshard {\n"
            "  func.func public @main(%%arg0: tensor<%lldx%lldx%s> {mhlo.sharding = \"{devices=[8,1]0,1,2,3,4,5,6,7}\"}, %%arg1: tensor<%lldx%lldx%s> {mhlo.sharding = \"{replicated}\"}) -> tensor<%lldx%lldx%s> {\n"
            "    %%0 = stablehlo.dot_general %%arg0, %%arg1, contracting_dims = [1] x [0] : (tensor<%lldx%lldx%s>, tensor<%lldx%lldx%s>) -> tensor<%lldx%lldx%s>\n"
            "    func.return %%0 : tensor<%lldx%lldx%s>\n"
            "  }\n"
            "}\n",
            (long long)m, (long long)k, etype,
            (long long)k, (long long)n, etype,
            (long long)m, (long long)n, etype,
            (long long)m, (long long)k, etype,
            (long long)k, (long long)n, etype,
            (long long)m, (long long)n, etype,
            (long long)m, (long long)n, etype);
    } else {
        std::snprintf(buf, sizeof(buf),
            "module @multichip_matmul_kshard {\n"
            "  func.func public @main(%%arg0: tensor<%lldx%lldx%s> {mhlo.sharding = \"{devices=[1,8]0,1,2,3,4,5,6,7}\"}, %%arg1: tensor<%lldx%lldx%s> {mhlo.sharding = \"{devices=[8,1]0,1,2,3,4,5,6,7}\"}) -> tensor<%lldx%lldx%s> {\n"
            "    %%0 = stablehlo.dot_general %%arg0, %%arg1, contracting_dims = [1] x [0] : (tensor<%lldx%lldx%s>, tensor<%lldx%lldx%s>) -> tensor<%lldx%lldx%s>\n"
            "    func.return %%0 : tensor<%lldx%lldx%s>\n"
            "  }\n"
            "}\n",
            (long long)m, (long long)k, etype,
            (long long)k, (long long)n, etype,
            (long long)m, (long long)n, etype,
            (long long)m, (long long)k, etype,
            (long long)k, (long long)n, etype,
            (long long)m, (long long)n, etype,
            (long long)m, (long long)n, etype);
    }
    return std::string(buf);
}

static int compile_module(PJRT_Client* client, const std::string& mlir,
                          const unsigned char* opts_bytes, size_t opts_size,
                          PJRT_LoadedExecutable** exe_out, double* compile_ms_out) {
    std::printf("program_format=mlir\n");
    std::printf("mlir_module_begin\n%smlir_module_end\n", mlir.c_str());
    PJRT_Program prog{};
    prog.struct_size = PJRT_Program_STRUCT_SIZE;
    prog.code = const_cast<char*>(mlir.data());
    prog.code_size = mlir.size();
    prog.format = "mlir";
    prog.format_size = 4;
    PJRT_Client_Compile_Args cargs{};
    cargs.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    cargs.client = client;
    cargs.program = &prog;
    cargs.compile_options = reinterpret_cast<const char*>(opts_bytes);
    cargs.compile_options_size = opts_size;
    std::printf("compile_options_hex=");
    for (size_t i = 0; i < opts_size; ++i) {
        std::printf("%02X", (unsigned)opts_bytes[i]);
    }
    std::printf("\n");
    double t0 = now_ms();
    PJRT_Error* err = g_api->PJRT_Client_Compile(&cargs);
    *compile_ms_out = now_ms() - t0;
    if (!check_err(err, "Client_Compile")) return 5;
    if (!cargs.executable) {
        std::fprintf(stderr, "MULTICHIP_FAIL Client_Compile returned null executable\n");
        return 5;
    }
    *exe_out = cargs.executable;
    std::printf("compile_ms=%.3f\n", *compile_ms_out);
    return 0;
}

static int get_num_outputs(PJRT_LoadedExecutable* exe, size_t* num_out) {
    PJRT_LoadedExecutable_GetExecutable_Args gargs{};
    gargs.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
    gargs.loaded_executable = exe;
    if (!check_err(g_api->PJRT_LoadedExecutable_GetExecutable(&gargs), "LoadedExecutable_GetExecutable")) return 5;
    PJRT_Executable* pexe = gargs.executable;
    PJRT_Executable_NumOutputs_Args nargs{};
    nargs.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
    nargs.executable = pexe;
    if (!check_err(g_api->PJRT_Executable_NumOutputs(&nargs), "Executable_NumOutputs")) return 5;
    *num_out = nargs.num_outputs;
    PJRT_Executable_Destroy_Args dargs{};
    dargs.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
    dargs.executable = pexe;
    PJRT_Error* derr = g_api->PJRT_Executable_Destroy(&dargs);
    if (derr) check_err(derr, "Executable_Destroy");
    return 0;
}

static PJRT_Buffer* make_buffer(PJRT_Client* client, const void* data, int64_t d0, int64_t d1,
                                PJRT_Device* device, bool is_bf16) {
    int64_t dims[2] = {d0, d1};
    PJRT_Client_BufferFromHostBuffer_Args args{};
    args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    args.client = client;
    args.data = data;
    args.type = is_bf16 ? PJRT_Buffer_Type_BF16 : PJRT_Buffer_Type_F32;
    args.dims = dims;
    args.num_dims = 2;
    args.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    args.device = device;
    if (!check_err(g_api->PJRT_Client_BufferFromHostBuffer(&args), "BufferFromHostBuffer")) return nullptr;
    if (args.done_with_host_buffer) {
        if (!await_event(args.done_with_host_buffer, "BufferFromHostBuffer done event")) return nullptr;
    }
    return args.buffer;
}

static bool buffer_bytes(PJRT_Buffer* buf, size_t* out) {
    PJRT_Buffer_OnDeviceSizeInBytes_Args args{};
    args.struct_size = PJRT_Buffer_OnDeviceSizeInBytes_Args_STRUCT_SIZE;
    args.buffer = buf;
    if (!check_err(g_api->PJRT_Buffer_OnDeviceSizeInBytes(&args), "Buffer_OnDeviceSizeInBytes")) return false;
    *out = args.on_device_size_in_bytes;
    return true;
}

static bool run_once_multi(PJRT_LoadedExecutable* exe,
                           const std::vector<std::vector<PJRT_Buffer*>>& arg_bufs,
                           size_t num_outputs, std::vector<PJRT_Buffer*>* outs) {
    size_t nd = arg_bufs.size();
    size_t na = arg_bufs.empty() ? 0 : arg_bufs[0].size();
    std::vector<PJRT_Buffer* const*> arg_lists(nd);
    for (size_t d = 0; d < nd; ++d) arg_lists[d] = arg_bufs[d].data();
    std::vector<PJRT_Buffer*> out_flat(nd * num_outputs, nullptr);
    std::vector<PJRT_Buffer**> out_lists(nd);
    for (size_t d = 0; d < nd; ++d) out_lists[d] = out_flat.data() + d * num_outputs;
    std::vector<PJRT_Event*> events(nd, nullptr);
    std::vector<int64_t> non_donatable(na);
    for (size_t i = 0; i < na; ++i) non_donatable[i] = (int64_t)i;
    PJRT_ExecuteOptions opts{};
    opts.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    opts.non_donatable_input_indices = non_donatable.data();
    opts.num_non_donatable_input_indices = na;
    PJRT_LoadedExecutable_Execute_Args args{};
    args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    args.executable = exe;
    args.options = &opts;
    args.argument_lists = arg_lists.data();
    args.num_devices = nd;
    args.num_args = na;
    args.output_lists = out_lists.data();
    args.device_complete_events = events.data();
    args.execute_device = nullptr;
    if (!check_err(g_api->PJRT_LoadedExecutable_Execute(&args), "LoadedExecutable_Execute")) return false;
    for (size_t d = 0; d < nd; ++d) {
        if (!await_event(events[d], "Execute complete event")) return false;
    }
    for (size_t i = 0; i < out_flat.size(); ++i) {
        if (!out_flat[i]) return false;
    }
    *outs = out_flat;
    return true;
}

static bool download_buffer(PJRT_Buffer* buf, void* dst, size_t bytes) {
    PJRT_Buffer_ToHostBuffer_Args targs{};
    targs.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    targs.src = buf;
    targs.dst = dst;
    targs.dst_size = bytes;
    if (!check_err(g_api->PJRT_Buffer_ToHostBuffer(&targs), "Buffer_ToHostBuffer")) return false;
    if (!await_event(targs.event, "ToHostBuffer event")) return false;
    return true;
}

int main(int argc, char** argv) {
    std::string lib_path, load_a, load_b, save_out, load_inputs, save_inputs;
    std::string sub;
    std::vector<std::string> pos;
    int partitions = 8;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--lib=", 0) == 0) lib_path = a.substr(6);
        else if (a == "--lib" && i + 1 < argc) lib_path = argv[++i];
        else if (a.rfind("--load-a=", 0) == 0) load_a = a.substr(9);
        else if (a.rfind("--load-b=", 0) == 0) load_b = a.substr(9);
        else if (a.rfind("--save-out=", 0) == 0) save_out = a.substr(11);
        else if (a.rfind("--load-inputs=", 0) == 0) load_inputs = a.substr(14);
        else if (a.rfind("--save-inputs=", 0) == 0) save_inputs = a.substr(14);
        else if (a.rfind("--partitions=", 0) == 0) partitions = std::atoi(a.substr(13).c_str());
        else if (a.rfind("--", 0) == 0) {
            std::fprintf(stderr, "MULTICHIP_FAIL unknown flag %s\n", a.c_str());
            return 2;
        } else if (sub.empty()) sub = a;
        else pos.push_back(a);
    }
    bool is_allreduce = (sub == "allreduce");
    bool is_row = (sub == "matmul-rowshard");
    bool is_k = (sub == "matmul-kshard");
    if (!is_allreduce && !is_row && !is_k) {
        std::fprintf(stderr, "usage: multichip_matmul [--lib=PATH] [--partitions=8|1] [--load-a=F32BIN] [--load-b=F32BIN] [--save-out=F32BIN] allreduce SIZE f32 iters warmup verify\n");
        std::fprintf(stderr, "       multichip_matmul [--lib=PATH] [--partitions=8|1] [--load-a=F32BIN] [--load-b=F32BIN] [--save-out=F32BIN] matmul-rowshard|matmul-kshard M K N f32|bf16 iters warmup verify\n");
        return 2;
    }
    if (partitions != 8 && partitions != 1) {
        std::fprintf(stderr, "MULTICHIP_FAIL --partitions must be 8 or 1, got %d\n", partitions);
        return 2;
    }
    if (is_allreduce && pos.size() < 5) {
        std::fprintf(stderr, "MULTICHIP_FAIL allreduce needs SIZE f32 iters warmup verify\n");
        return 2;
    }
    if (!is_allreduce && pos.size() < 7) {
        std::fprintf(stderr, "MULTICHIP_FAIL matmul needs M K N f32|bf16 iters warmup verify\n");
        return 2;
    }
    int64_t S = 0, M = 0, K = 0, N = 0;
    std::string dtype;
    int iters = 0, warmup = 0, verify = 0;
    if (is_allreduce) {
        S = std::atoll(pos[0].c_str());
        dtype = pos[1];
        iters = std::atoi(pos[2].c_str());
        warmup = std::atoi(pos[3].c_str());
        verify = std::atoi(pos[4].c_str());
        if (S <= 0 || iters <= 0 || warmup < 0) {
            std::fprintf(stderr, "MULTICHIP_FAIL bad shape or iteration arguments\n");
            return 2;
        }
        if (dtype != "f32") {
            std::fprintf(stderr, "MULTICHIP_FAIL allreduce is FP32 only in this proof\n");
            return 2;
        }
    } else {
        M = std::atoll(pos[0].c_str());
        K = std::atoll(pos[1].c_str());
        N = std::atoll(pos[2].c_str());
        dtype = pos[3];
        iters = std::atoi(pos[4].c_str());
        warmup = std::atoi(pos[5].c_str());
        verify = std::atoi(pos[6].c_str());
        if (M <= 0 || K <= 0 || N <= 0 || iters <= 0 || warmup < 0) {
            std::fprintf(stderr, "MULTICHIP_FAIL bad shape or iteration arguments\n");
            return 2;
        }
        if (dtype != "f32" && dtype != "bf16") {
            std::fprintf(stderr, "MULTICHIP_FAIL dtype must be f32 or bf16\n");
            return 2;
        }
        if (partitions == 8 && (M % 8 != 0 || K % 8 != 0)) {
            std::fprintf(stderr, "MULTICHIP_FAIL sharded matmul needs M and K divisible by 8\n");
            return 2;
        }
    }
    bool is_bf16 = (dtype == "bf16");

    std::string lib = find_libtpu(lib_path);
    if (lib.empty()) {
        std::fprintf(stderr, "MULTICHIP_FAIL libtpu.so not found (set TPU_LIBRARY_PATH or pass --lib=PATH)\n");
        return 3;
    }
    std::printf("libtpu_path=%s\n", lib.c_str());
    void* handle = dlopen(lib.c_str(), RTLD_NOW);
    if (!handle) {
        std::fprintf(stderr, "MULTICHIP_FAIL dlopen(%s): %s\n", lib.c_str(), dlerror());
        return 3;
    }
    using GetApiFn = const PJRT_Api* (*)();
    dlerror();
    auto get_api = reinterpret_cast<GetApiFn>(dlsym(handle, "GetPjrtApi"));
    if (!get_api) {
        std::fprintf(stderr, "MULTICHIP_FAIL dlsym(GetPjrtApi): %s\n", dlerror());
        return 3;
    }
    g_api = get_api();
    if (!g_api) {
        std::fprintf(stderr, "MULTICHIP_FAIL GetPjrtApi returned null\n");
        return 3;
    }
    std::printf("pjrt_api_version=%d.%d header=%d.%d\n",
                g_api->pjrt_api_version.major_version, g_api->pjrt_api_version.minor_version,
                PJRT_API_MAJOR, PJRT_API_MINOR);
    if (g_api->pjrt_api_version.major_version != PJRT_API_MAJOR) {
        std::fprintf(stderr, "MULTICHIP_FAIL PJRT API major version mismatch\n");
        return 3;
    }

    {
        PJRT_Plugin_Initialize_Args iargs{};
        iargs.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
        if (!check_err(g_api->PJRT_Plugin_Initialize(&iargs), "Plugin_Initialize")) return 4;
    }
    if (g_api->PJRT_Plugin_Attributes) {
        PJRT_Plugin_Attributes_Args aargs{};
        aargs.struct_size = PJRT_Plugin_Attributes_Args_STRUCT_SIZE;
        if (check_err(g_api->PJRT_Plugin_Attributes(&aargs), "Plugin_Attributes")) {
            for (size_t i = 0; i < aargs.num_attributes; ++i) {
                const PJRT_NamedValue& v = aargs.attributes[i];
                std::string name(v.name, v.name_size);
                if (v.type == PJRT_NamedValue_kString) {
                    std::printf("plugin_attribute %s=%.*s\n", name.c_str(), (int)v.value_size, v.string_value);
                } else if (v.type == PJRT_NamedValue_kInt64) {
                    std::printf("plugin_attribute %s=%lld\n", name.c_str(), (long long)v.int64_value);
                }
            }
        }
    }

    PJRT_Client* client = nullptr;
    {
        PJRT_Client_Create_Args cargs{};
        cargs.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
        if (!check_err(g_api->PJRT_Client_Create(&cargs), "Client_Create")) return 4;
        client = cargs.client;
    }
    if (!client) { std::fprintf(stderr, "MULTICHIP_FAIL Client_Create returned null client\n"); return 4; }

    {
        PJRT_Client_PlatformName_Args pargs{};
        pargs.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
        pargs.client = client;
        if (check_err(g_api->PJRT_Client_PlatformName(&pargs), "PlatformName")) {
            std::printf("platform_name=%.*s\n", (int)pargs.platform_name_size, pargs.platform_name);
        }
        PJRT_Client_PlatformVersion_Args vargs{};
        vargs.struct_size = PJRT_Client_PlatformVersion_Args_STRUCT_SIZE;
        vargs.client = client;
        if (check_err(g_api->PJRT_Client_PlatformVersion(&vargs), "PlatformVersion")) {
            std::printf("platform_version=%.*s\n", (int)vargs.platform_version_size, vargs.platform_version);
        }
    }

    std::vector<PJRT_Device*> devices;
    int num_devices = 0;
    {
        PJRT_Client_AddressableDevices_Args dargs{};
        dargs.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
        dargs.client = client;
        if (!check_err(g_api->PJRT_Client_AddressableDevices(&dargs), "AddressableDevices")) return 4;
        num_devices = (int)dargs.num_addressable_devices;
        std::printf("num_addressable_devices=%d\n", num_devices);
        for (size_t i = 0; i < dargs.num_addressable_devices; ++i) {
            PJRT_Device* dev = dargs.addressable_devices[i];
            PJRT_Device_GetDescription_Args ddargs{};
            ddargs.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
            ddargs.device = dev;
            std::string kind = "unknown";
            int dev_id = -1;
            if (check_err(g_api->PJRT_Device_GetDescription(&ddargs), "Device_GetDescription")) {
                PJRT_DeviceDescription_Kind_Args kargs{};
                kargs.struct_size = PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE;
                kargs.device_description = ddargs.device_description;
                if (check_err(g_api->PJRT_DeviceDescription_Kind(&kargs), "DeviceDescription_Kind")) {
                    kind.assign(kargs.device_kind, kargs.device_kind_size);
                }
                PJRT_DeviceDescription_Id_Args idargs{};
                idargs.struct_size = PJRT_DeviceDescription_Id_Args_STRUCT_SIZE;
                idargs.device_description = ddargs.device_description;
                if (check_err(g_api->PJRT_DeviceDescription_Id(&idargs), "DeviceDescription_Id")) {
                    dev_id = idargs.id;
                }
            }
            std::printf("device[%zu] id=%d kind=%s\n", i, dev_id, kind.c_str());
            devices.push_back(dev);
        }
    }
    if (devices.empty()) { std::fprintf(stderr, "MULTICHIP_FAIL no addressable device\n"); return 4; }
    if (partitions == 8 && num_devices != 8) {
        std::fprintf(stderr, "MULTICHIP_FAIL sharded run needs 8 addressable devices, got %d; refusing to run degraded\n", num_devices);
        return 4;
    }
    size_t nd = (size_t)partitions;
    const unsigned char* opts_bytes = (partitions == 8) ? kCompileOptionsProto18 : kCompileOptionsProto11;
    size_t opts_size = 6;

    if (is_allreduce) {
        std::string mlir = build_mlir_allreduce(S, partitions);
        PJRT_LoadedExecutable* exe = nullptr;
        double compile_ms = 0.0;
        int crc = compile_module(client, mlir, opts_bytes, opts_size, &exe, &compile_ms);
        if (crc != 0) return crc;
        size_t num_outputs = 0;
        if (get_num_outputs(exe, &num_outputs) != 0) return 5;
        if (num_outputs != 1) {
            std::fprintf(stderr, "MULTICHIP_FAIL expected 1 output per device, got %zu\n", num_outputs);
            return 5;
        }

        size_t nelem = (size_t)(S * S);
        std::vector<std::vector<float>> in_f(nd, std::vector<float>(nelem));
        if (!load_inputs.empty()) {
            std::vector<float> flat(nd * nelem);
            if (!load_f32_file(load_inputs, flat)) {
                std::fprintf(stderr, "MULTICHIP_FAIL cannot load inputs from %s\n", load_inputs.c_str());
                return 2;
            }
            for (size_t d = 0; d < nd; ++d) {
                std::memcpy(in_f[d].data(), flat.data() + d * nelem, nelem * sizeof(float));
            }
        } else {
            uint32_t seed = 20261119u;
            auto next_rand = [&seed]() {
                seed = seed * 1664525u + 1013904223u;
                return (float)((seed >> 8) & 0xFFFF) / 32768.0f - 1.0f;
            };
            for (size_t d = 0; d < nd; ++d) {
                for (size_t i = 0; i < nelem; ++i) in_f[d][i] = next_rand() * 0.5f;
            }
        }
        if (!save_inputs.empty()) {
            std::vector<float> flat(nd * nelem);
            for (size_t d = 0; d < nd; ++d) {
                std::memcpy(flat.data() + d * nelem, in_f[d].data(), nelem * sizeof(float));
            }
            if (!save_f32_file(save_inputs, flat.data(), flat.size())) {
                std::fprintf(stderr, "MULTICHIP_FAIL cannot open save-inputs %s\n", save_inputs.c_str());
                return 2;
            }
            std::printf("saved_inputs=%s\n", save_inputs.c_str());
        }

        std::vector<std::vector<PJRT_Buffer*>> arg_bufs(nd);
        std::vector<size_t> bytes_a(nd, 0);
        for (size_t d = 0; d < nd; ++d) {
            PJRT_Buffer* b = make_buffer(client, in_f[d].data(), S, S, devices[d], false);
            if (!b) { std::fprintf(stderr, "MULTICHIP_FAIL input buffer creation\n"); return 6; }
            arg_bufs[d].push_back(b);
            if (!buffer_bytes(b, &bytes_a[d])) return 6;
            std::printf("on_device_bytes dev=%zu role=a bytes=%zu\n", d, bytes_a[d]);
        }

        for (int i = 0; i < warmup; ++i) {
            std::vector<PJRT_Buffer*> outs;
            if (!run_once_multi(exe, arg_bufs, num_outputs, &outs)) return 7;
            for (size_t j = 0; j < outs.size(); ++j) destroy_buffer(outs[j]);
        }
        std::vector<double> times;
        for (int i = 0; i < iters; ++i) {
            std::vector<PJRT_Buffer*> outs;
            double t0 = now_ms();
            if (!run_once_multi(exe, arg_bufs, num_outputs, &outs)) return 7;
            times.push_back(now_ms() - t0);
            for (size_t j = 0; j < outs.size(); ++j) destroy_buffer(outs[j]);
        }
        std::sort(times.begin(), times.end());
        double median_ms = times[times.size() / 2];
        double min_ms = times.front();
        double gbps = (double)(nelem * 4) / median_ms / 1e6;
        std::printf("execute_median_ms=%.3f execute_min_ms=%.3f iters=%d warmup=%d approx_gbps=%.3f\n",
                    median_ms, min_ms, iters, warmup, gbps);

        std::vector<size_t> bytes_c(nd, 0);
        {
            std::vector<PJRT_Buffer*> outs;
            if (!run_once_multi(exe, arg_bufs, num_outputs, &outs)) return 7;
            for (size_t d = 0; d < nd; ++d) {
                if (!buffer_bytes(outs[d], &bytes_c[d])) return 7;
                std::printf("on_device_bytes dev=%zu role=c bytes=%zu\n", d, bytes_c[d]);
            }
            for (size_t j = 0; j < outs.size(); ++j) destroy_buffer(outs[j]);
        }

        double max_abs_diff = -1.0;
        double cross_device_max_abs_diff = -1.0;
        int verify_pass = -1;
        if (verify) {
            std::vector<PJRT_Buffer*> outs;
            if (!run_once_multi(exe, arg_bufs, num_outputs, &outs)) return 7;
            std::vector<std::vector<float>> out_f(nd, std::vector<float>(nelem));
            for (size_t d = 0; d < nd; ++d) {
                if (!download_buffer(outs[d], out_f[d].data(), nelem * sizeof(float))) return 8;
            }
            std::vector<float> ref(nelem, 0.0f);
            for (size_t d = 0; d < nd; ++d) {
                for (size_t i = 0; i < nelem; ++i) ref[i] += in_f[d][i];
            }
            max_abs_diff = 0.0;
            cross_device_max_abs_diff = 0.0;
            for (size_t d = 0; d < nd; ++d) {
                for (size_t i = 0; i < nelem; ++i) {
                    double dd = std::fabs((double)out_f[d][i] - (double)ref[i]);
                    if (dd > max_abs_diff) max_abs_diff = dd;
                    double cd = std::fabs((double)out_f[d][i] - (double)out_f[0][i]);
                    if (cd > cross_device_max_abs_diff) cross_device_max_abs_diff = cd;
                }
            }
            verify_pass = (max_abs_diff < 1e-4) ? 1 : 0;
            std::printf("verify_max_abs_diff=%.6g cross_device_max_abs_diff=%.6g verify_pass=%d\n",
                        max_abs_diff, cross_device_max_abs_diff, verify_pass);
            if (!save_out.empty()) {
                std::vector<float> flat(nd * nelem);
                for (size_t d = 0; d < nd; ++d) {
                    std::memcpy(flat.data() + d * nelem, out_f[d].data(), nelem * sizeof(float));
                }
                if (!save_f32_file(save_out, flat.data(), flat.size())) {
                    std::fprintf(stderr, "MULTICHIP_FAIL cannot open save-out %s\n", save_out.c_str());
                    return 8;
                }
                std::printf("saved_out=%s\n", save_out.c_str());
            }
            for (size_t j = 0; j < outs.size(); ++j) destroy_buffer(outs[j]);
        }

        std::printf("RESULT_JSON {\"op\": \"allreduce\", \"partitions\": %d, \"size\": %lld, \"dtype\": \"f32\", \"compile_ms\": %.3f, \"execute_median_ms\": %.3f, \"execute_min_ms\": %.3f, \"approx_gbps\": %.3f, \"num_devices\": %d, \"num_outputs_per_device\": %zu, \"on_device_bytes\": {\"a\": [",
                    partitions, (long long)S, compile_ms, median_ms, min_ms, gbps, num_devices, num_outputs);
        for (size_t d = 0; d < nd; ++d) {
            std::printf("%s%zu", d ? ", " : "", bytes_a[d]);
        }
        std::printf("], \"c\": [");
        for (size_t d = 0; d < nd; ++d) {
            std::printf("%s%zu", d ? ", " : "", bytes_c[d]);
        }
        std::printf("]}, \"global_bytes\": {\"a\": %zu, \"c\": %zu}, \"verify\": %d, \"max_abs_diff\": %.6g, \"max_rel_err\": -1, \"verify_pass\": %d, \"cross_device_max_abs_diff\": %.6g}\n",
                    nelem * 4, nelem * 4, verify, max_abs_diff, verify_pass, cross_device_max_abs_diff);

        for (size_t d = 0; d < nd; ++d) {
            for (size_t j = 0; j < arg_bufs[d].size(); ++j) destroy_buffer(arg_bufs[d][j]);
        }
        {
            PJRT_LoadedExecutable_Destroy_Args args{};
            args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
            args.executable = exe;
            PJRT_Error* err = g_api->PJRT_LoadedExecutable_Destroy(&args);
            if (err) check_err(err, "LoadedExecutable_Destroy");
        }
        {
            PJRT_Client_Destroy_Args args{};
            args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
            args.client = client;
            PJRT_Error* err = g_api->PJRT_Client_Destroy(&args);
            if (err) check_err(err, "Client_Destroy");
        }
        dlclose(handle);
        if (verify && verify_pass == 0) {
            std::fprintf(stderr, "MULTICHIP_FAIL numerics verification failed\n");
            return 9;
        }
        std::printf("MULTICHIP_OK\n");
        return 0;
    }

    int mode = 0;
    if (partitions == 8) mode = is_row ? 1 : 2;
    std::string mlir = build_mlir_matmul(M, K, N, is_bf16 ? "bf16" : "f32", mode);
    PJRT_LoadedExecutable* exe = nullptr;
    double compile_ms = 0.0;
    int crc = compile_module(client, mlir, opts_bytes, opts_size, &exe, &compile_ms);
    if (crc != 0) return crc;
    size_t num_outputs = 0;
    if (get_num_outputs(exe, &num_outputs) != 0) return 5;
    if (num_outputs != 1) {
        std::fprintf(stderr, "MULTICHIP_FAIL expected 1 output per device, got %zu\n", num_outputs);
        return 5;
    }

    size_t na = (size_t)(M * K);
    size_t nb = (size_t)(K * N);
    size_t nc = (size_t)(M * N);
    std::vector<float> a_f(na), b_f(nb);
    uint32_t seed = 12345;
    auto next_rand = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return (float)((seed >> 8) & 0xFFFF) / 32768.0f - 1.0f;
    };
    for (size_t i = 0; i < na; ++i) a_f[i] = next_rand() * 0.5f;
    for (size_t i = 0; i < nb; ++i) b_f[i] = next_rand() * 0.5f;
    if (!load_a.empty() && !load_f32_file(load_a, a_f)) {
        std::fprintf(stderr, "MULTICHIP_FAIL cannot load A from %s\n", load_a.c_str());
        return 2;
    }
    if (!load_b.empty() && !load_f32_file(load_b, b_f)) {
        std::fprintf(stderr, "MULTICHIP_FAIL cannot load B from %s\n", load_b.c_str());
        return 2;
    }

    std::vector<std::vector<float>> a_stage(nd), b_stage(nd);
    std::vector<int64_t> a_d0(nd), a_d1(nd), b_d0(nd), b_d1(nd);
    if (mode == 0) {
        a_stage[0] = a_f;
        b_stage[0] = b_f;
        a_d0[0] = M; a_d1[0] = K;
        b_d0[0] = K; b_d1[0] = N;
    } else if (mode == 1) {
        int64_t mr = M / 8;
        for (size_t d = 0; d < nd; ++d) {
            a_stage[d].resize((size_t)(mr * K));
            for (int64_t r = 0; r < mr; ++r) {
                std::memcpy(a_stage[d].data() + (size_t)(r * K),
                            a_f.data() + (size_t)((d * (size_t)mr + (size_t)r) * (size_t)K),
                            (size_t)K * sizeof(float));
            }
            b_stage[d] = b_f;
            a_d0[d] = mr; a_d1[d] = K;
            b_d0[d] = K; b_d1[d] = N;
        }
    } else {
        int64_t kc = K / 8;
        for (size_t d = 0; d < nd; ++d) {
            a_stage[d].resize((size_t)(M * kc));
            for (int64_t r = 0; r < M; ++r) {
                for (int64_t c = 0; c < kc; ++c) {
                    a_stage[d][(size_t)(r * kc + c)] = a_f[(size_t)(r * K + (int64_t)d * kc + c)];
                }
            }
            b_stage[d].resize((size_t)(kc * N));
            for (int64_t r = 0; r < kc; ++r) {
                std::memcpy(b_stage[d].data() + (size_t)(r * N),
                            b_f.data() + (size_t)(((int64_t)d * kc + r) * N),
                            (size_t)N * sizeof(float));
            }
            a_d0[d] = M; a_d1[d] = kc;
            b_d0[d] = kc; b_d1[d] = N;
        }
    }
    std::vector<std::vector<uint16_t>> a_stage_b(nd), b_stage_b(nd);
    if (is_bf16) {
        for (size_t i = 0; i < na; ++i) { uint16_t v = float_to_bf16(a_f[i]); a_f[i] = bf16_to_float(v); }
        for (size_t i = 0; i < nb; ++i) { uint16_t v = float_to_bf16(b_f[i]); b_f[i] = bf16_to_float(v); }
        for (size_t d = 0; d < nd; ++d) {
            a_stage_b[d].resize(a_stage[d].size());
            b_stage_b[d].resize(b_stage[d].size());
            for (size_t i = 0; i < a_stage[d].size(); ++i) a_stage_b[d][i] = float_to_bf16(a_stage[d][i]);
            for (size_t i = 0; i < b_stage[d].size(); ++i) b_stage_b[d][i] = float_to_bf16(b_stage[d][i]);
        }
    }

    std::vector<std::vector<PJRT_Buffer*>> arg_bufs(nd);
    std::vector<size_t> bytes_a(nd, 0), bytes_b(nd, 0);
    for (size_t d = 0; d < nd; ++d) {
        const void* ap = is_bf16 ? (const void*)a_stage_b[d].data() : (const void*)a_stage[d].data();
        const void* bp = is_bf16 ? (const void*)b_stage_b[d].data() : (const void*)b_stage[d].data();
        PJRT_Buffer* ba = make_buffer(client, ap, a_d0[d], a_d1[d], devices[d], is_bf16);
        PJRT_Buffer* bb = make_buffer(client, bp, b_d0[d], b_d1[d], devices[d], is_bf16);
        if (!ba || !bb) { std::fprintf(stderr, "MULTICHIP_FAIL input buffer creation\n"); return 6; }
        arg_bufs[d].push_back(ba);
        arg_bufs[d].push_back(bb);
        if (!buffer_bytes(ba, &bytes_a[d])) return 6;
        if (!buffer_bytes(bb, &bytes_b[d])) return 6;
        std::printf("on_device_bytes dev=%zu role=a bytes=%zu\n", d, bytes_a[d]);
        std::printf("on_device_bytes dev=%zu role=b bytes=%zu\n", d, bytes_b[d]);
    }

    for (int i = 0; i < warmup; ++i) {
        std::vector<PJRT_Buffer*> outs;
        if (!run_once_multi(exe, arg_bufs, num_outputs, &outs)) return 7;
        for (size_t j = 0; j < outs.size(); ++j) destroy_buffer(outs[j]);
    }
    std::vector<double> times;
    for (int i = 0; i < iters; ++i) {
        std::vector<PJRT_Buffer*> outs;
        double t0 = now_ms();
        if (!run_once_multi(exe, arg_bufs, num_outputs, &outs)) return 7;
        times.push_back(now_ms() - t0);
        for (size_t j = 0; j < outs.size(); ++j) destroy_buffer(outs[j]);
    }
    std::sort(times.begin(), times.end());
    double median_ms = times[times.size() / 2];
    double min_ms = times.front();
    double gflops = 2.0 * (double)M * (double)N * (double)K / (median_ms * 1e6);
    std::printf("execute_median_ms=%.3f execute_min_ms=%.3f iters=%d warmup=%d approx_gflops=%.1f\n",
                median_ms, min_ms, iters, warmup, gflops);

    std::vector<size_t> bytes_c(nd, 0);
    {
        std::vector<PJRT_Buffer*> outs;
        if (!run_once_multi(exe, arg_bufs, num_outputs, &outs)) return 7;
        for (size_t d = 0; d < nd; ++d) {
            if (!buffer_bytes(outs[d], &bytes_c[d])) return 7;
            std::printf("on_device_bytes dev=%zu role=c bytes=%zu\n", d, bytes_c[d]);
        }
        for (size_t j = 0; j < outs.size(); ++j) destroy_buffer(outs[j]);
    }

    double max_abs_diff = -1.0;
    double max_rel_err = -1.0;
    double cross_device_max_abs_diff = -1.0;
    int verify_pass = -1;
    if (verify) {
        std::vector<PJRT_Buffer*> outs;
        if (!run_once_multi(exe, arg_bufs, num_outputs, &outs)) return 7;
        size_t esz = is_bf16 ? 2 : 4;
        std::vector<std::vector<float>> out_f(nd);
        for (size_t d = 0; d < nd; ++d) {
            size_t out_elems = (mode == 1) ? (size_t)((M / 8) * N) : nc;
            out_f[d].resize(out_elems);
            if (is_bf16) {
                std::vector<uint16_t> tmp(out_elems);
                if (!download_buffer(outs[d], tmp.data(), out_elems * esz)) return 8;
                for (size_t i = 0; i < out_elems; ++i) out_f[d][i] = bf16_to_float(tmp[i]);
            } else {
                if (!download_buffer(outs[d], out_f[d].data(), out_elems * esz)) return 8;
            }
            size_t obytes = 0;
            if (!buffer_bytes(outs[d], &obytes)) return 8;
            std::printf("on_device_bytes dev=%zu role=c bytes=%zu\n", d, obytes);
        }
        std::vector<float> c_full(nc, 0.0f);
        if (mode == 1) {
            int64_t mr = M / 8;
            for (size_t d = 0; d < nd; ++d) {
                std::memcpy(c_full.data() + d * (size_t)(mr * N), out_f[d].data(), (size_t)(mr * N) * sizeof(float));
            }
        } else {
            c_full = out_f[0];
            cross_device_max_abs_diff = 0.0;
            for (size_t d = 1; d < nd; ++d) {
                for (size_t i = 0; i < nc; ++i) {
                    double cd = std::fabs((double)out_f[d][i] - (double)out_f[0][i]);
                    if (cd > cross_device_max_abs_diff) cross_device_max_abs_diff = cd;
                }
            }
        }
        std::vector<float> ref(nc, 0.0f);
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t k = 0; k < K; ++k) {
                float av = a_f[(size_t)(i * K + k)];
                if (av == 0.0f) continue;
                for (int64_t j = 0; j < N; ++j) {
                    ref[(size_t)(i * N + j)] += av * b_f[(size_t)(k * N + j)];
                }
            }
        }
        max_abs_diff = 0.0;
        max_rel_err = 0.0;
        for (size_t i = 0; i < nc; ++i) {
            double dd = std::fabs((double)c_full[i] - (double)ref[i]);
            if (dd > max_abs_diff) max_abs_diff = dd;
            double denom = std::max(1.0, std::fabs((double)ref[i]));
            double r = dd / denom;
            if (r > max_rel_err) max_rel_err = r;
        }
        if (is_bf16) verify_pass = (max_rel_err < 1e-2) ? 1 : 0;
        else verify_pass = (max_abs_diff < 1e-4) ? 1 : 0;
        std::printf("verify_max_abs_diff=%.6g verify_max_rel_err=%.6g cross_device_max_abs_diff=%.6g verify_pass=%d\n",
                    max_abs_diff, max_rel_err, cross_device_max_abs_diff, verify_pass);
        if (!save_out.empty()) {
            if (!save_f32_file(save_out, c_full.data(), c_full.size())) {
                std::fprintf(stderr, "MULTICHIP_FAIL cannot open save-out %s\n", save_out.c_str());
                return 8;
            }
            std::printf("saved_out=%s\n", save_out.c_str());
        }
        for (size_t j = 0; j < outs.size(); ++j) destroy_buffer(outs[j]);
    }

    std::printf("RESULT_JSON {\"op\": \"%s\", \"partitions\": %d, \"M\": %lld, \"K\": %lld, \"N\": %lld, \"dtype\": \"%s\", \"compile_ms\": %.3f, \"execute_median_ms\": %.3f, \"execute_min_ms\": %.3f, \"approx_gflops\": %.1f, \"num_devices\": %d, \"num_outputs_per_device\": %zu, \"on_device_bytes\": {\"a\": [",
                is_row ? "matmul-rowshard" : "matmul-kshard", partitions,
                (long long)M, (long long)K, (long long)N, dtype.c_str(),
                compile_ms, median_ms, min_ms, gflops, num_devices, num_outputs);
    for (size_t d = 0; d < nd; ++d) {
        std::printf("%s%zu", d ? ", " : "", bytes_a[d]);
    }
    std::printf("], \"b\": [");
    for (size_t d = 0; d < nd; ++d) {
        std::printf("%s%zu", d ? ", " : "", bytes_b[d]);
    }
    std::printf("], \"c\": [");
    for (size_t d = 0; d < nd; ++d) {
        std::printf("%s%zu", d ? ", " : "", bytes_c[d]);
    }
    size_t esz = is_bf16 ? 2 : 4;
    std::printf("]}, \"global_bytes\": {\"a\": %zu, \"b\": %zu, \"c\": %zu}, \"verify\": %d, \"max_abs_diff\": %.6g, \"max_rel_err\": %.6g, \"verify_pass\": %d, \"cross_device_max_abs_diff\": %.6g}\n",
                na * esz, nb * esz, nc * esz, verify, max_abs_diff, max_rel_err, verify_pass, cross_device_max_abs_diff);

    for (size_t d = 0; d < nd; ++d) {
        for (size_t j = 0; j < arg_bufs[d].size(); ++j) destroy_buffer(arg_bufs[d][j]);
    }
    {
        PJRT_LoadedExecutable_Destroy_Args args{};
        args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        args.executable = exe;
        PJRT_Error* err = g_api->PJRT_LoadedExecutable_Destroy(&args);
        if (err) check_err(err, "LoadedExecutable_Destroy");
    }
    {
        PJRT_Client_Destroy_Args args{};
        args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
        args.client = client;
        PJRT_Error* err = g_api->PJRT_Client_Destroy(&args);
        if (err) check_err(err, "Client_Destroy");
    }
    dlclose(handle);
    if (verify && verify_pass == 0) {
        std::fprintf(stderr, "MULTICHIP_FAIL numerics verification failed\n");
        return 9;
    }
    std::printf("MULTICHIP_OK\n");
    return 0;
}
