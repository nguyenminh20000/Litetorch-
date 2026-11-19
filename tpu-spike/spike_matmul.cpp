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
    std::fprintf(stderr, "SPIKE_FAIL %s: %s\n", what, msg.c_str());
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
        if (access(candidates[i], R_OK) == 0) return std::string(candidates[i]);
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

static std::string build_mlir(int64_t m, int64_t k, int64_t n, const char* etype) {
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
        "module @spike_matmul {\n"
        "  func.func public @main(%%arg0: tensor<%lldx%lldx%s>, %%arg1: tensor<%lldx%lldx%s>) -> tensor<%lldx%lldx%s> {\n"
        "    %%0 = stablehlo.dot_general %%arg0, %%arg1, contracting_dims = [1] x [0] : (tensor<%lldx%lldx%s>, tensor<%lldx%lldx%s>) -> tensor<%lldx%lldx%s>\n"
        "    return %%0 : tensor<%lldx%lldx%s>\n"
        "  }\n"
        "}\n",
        (long long)m, (long long)k, etype,
        (long long)k, (long long)n, etype,
        (long long)m, (long long)n, etype,
        (long long)m, (long long)k, etype,
        (long long)k, (long long)n, etype,
        (long long)m, (long long)n, etype,
        (long long)m, (long long)n, etype);
    return std::string(buf);
}

int main(int argc, char** argv) {
    std::string lib_path;
    std::string load_a, load_b, save_out;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--lib=", 0) == 0) lib_path = a.substr(6);
        else if (a == "--lib" && i + 1 < argc) lib_path = argv[++i];
        else if (a.rfind("--load-a=", 0) == 0) load_a = a.substr(9);
        else if (a.rfind("--load-b=", 0) == 0) load_b = a.substr(9);
        else if (a.rfind("--save-out=", 0) == 0) save_out = a.substr(11);
        else pos.push_back(a);
    }
    if (pos.size() < 7) {
        std::fprintf(stderr, "usage: spike_matmul [--lib=PATH] [--load-a=F32BIN] [--load-b=F32BIN] [--save-out=F32BIN] M K N f32|bf16 iters warmup verify\n");
        return 2;
    }
    int64_t M = std::atoll(pos[0].c_str());
    int64_t K = std::atoll(pos[1].c_str());
    int64_t N = std::atoll(pos[2].c_str());
    std::string dtype = pos[3];
    int iters = std::atoi(pos[4].c_str());
    int warmup = std::atoi(pos[5].c_str());
    int verify = std::atoi(pos[6].c_str());
    if (M <= 0 || K <= 0 || N <= 0 || iters <= 0 || warmup < 0) {
        std::fprintf(stderr, "SPIKE_FAIL bad shape or iteration arguments\n");
        return 2;
    }
    bool is_bf16;
    if (dtype == "f32") is_bf16 = false;
    else if (dtype == "bf16") is_bf16 = true;
    else { std::fprintf(stderr, "SPIKE_FAIL dtype must be f32 or bf16\n"); return 2; }

    std::string lib = find_libtpu(lib_path);
    if (lib.empty()) {
        std::fprintf(stderr, "SPIKE_FAIL libtpu.so not found (set TPU_LIBRARY_PATH or pass --lib=PATH)\n");
        return 3;
    }
    std::printf("libtpu_path=%s\n", lib.c_str());
    void* handle = dlopen(lib.c_str(), RTLD_NOW);
    if (!handle) {
        std::fprintf(stderr, "SPIKE_FAIL dlopen(%s): %s\n", lib.c_str(), dlerror());
        return 3;
    }
    using GetApiFn = const PJRT_Api* (*)();
    dlerror();
    auto get_api = reinterpret_cast<GetApiFn>(dlsym(handle, "GetPjrtApi"));
    if (!get_api) {
        std::fprintf(stderr, "SPIKE_FAIL dlsym(GetPjrtApi): %s\n", dlerror());
        return 3;
    }
    g_api = get_api();
    if (!g_api) {
        std::fprintf(stderr, "SPIKE_FAIL GetPjrtApi returned null\n");
        return 3;
    }
    std::printf("pjrt_api_version=%d.%d header=%d.%d\n",
                g_api->pjrt_api_version.major_version, g_api->pjrt_api_version.minor_version,
                PJRT_API_MAJOR, PJRT_API_MINOR);
    if (g_api->pjrt_api_version.major_version != PJRT_API_MAJOR) {
        std::fprintf(stderr, "SPIKE_FAIL PJRT API major version mismatch\n");
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
    if (!client) { std::fprintf(stderr, "SPIKE_FAIL Client_Create returned null client\n"); return 4; }

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

    PJRT_Device* device = nullptr;
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
            if (i == 0) device = dev;
        }
    }
    if (!device) { std::fprintf(stderr, "SPIKE_FAIL no addressable device\n"); return 4; }

    std::string mlir = build_mlir(M, K, N, is_bf16 ? "bf16" : "f32");
    std::printf("program_format=mlir\n");
    std::printf("mlir_module_begin\n%smlir_module_end\n", mlir.c_str());

    PJRT_LoadedExecutable* exe = nullptr;
    double compile_ms = 0.0;
    {
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
        double t0 = now_ms();
        PJRT_Error* err = g_api->PJRT_Client_Compile(&cargs);
        compile_ms = now_ms() - t0;
        if (!check_err(err, "Client_Compile")) return 5;
        exe = cargs.executable;
    }
    if (!exe) { std::fprintf(stderr, "SPIKE_FAIL Client_Compile returned null executable\n"); return 5; }
    std::printf("compile_ms=%.3f\n", compile_ms);

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
    auto load_f32 = [](const std::string& path, std::vector<float>& dst) -> bool {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        size_t got = std::fread(dst.data(), sizeof(float), dst.size(), f);
        std::fclose(f);
        return got == dst.size();
    };
    if (!load_a.empty() && !load_f32(load_a, a_f)) {
        std::fprintf(stderr, "SPIKE_FAIL cannot load A from %s\n", load_a.c_str());
        return 2;
    }
    if (!load_b.empty() && !load_f32(load_b, b_f)) {
        std::fprintf(stderr, "SPIKE_FAIL cannot load B from %s\n", load_b.c_str());
        return 2;
    }

    std::vector<uint16_t> a_b, b_b;
    const void* a_ptr = a_f.data();
    const void* b_ptr = b_f.data();
    if (is_bf16) {
        a_b.resize(na);
        b_b.resize(nb);
        for (size_t i = 0; i < na; ++i) { a_b[i] = float_to_bf16(a_f[i]); a_f[i] = bf16_to_float(a_b[i]); }
        for (size_t i = 0; i < nb; ++i) { b_b[i] = float_to_bf16(b_f[i]); b_f[i] = bf16_to_float(b_b[i]); }
        a_ptr = a_b.data();
        b_ptr = b_b.data();
    }

    auto make_buffer = [&](const void* data, int64_t d0, int64_t d1) -> PJRT_Buffer* {
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
    };

    PJRT_Buffer* buf_a = make_buffer(a_ptr, M, K);
    PJRT_Buffer* buf_b = make_buffer(b_ptr, K, N);
    if (!buf_a || !buf_b) { std::fprintf(stderr, "SPIKE_FAIL input buffer creation\n"); return 6; }

    int64_t non_donatable[2] = {0, 1};
    auto run_once = [&](PJRT_Buffer** out_buf) -> bool {
        PJRT_Buffer* inputs[2] = {buf_a, buf_b};
        PJRT_Buffer* const* arg_lists[1] = {inputs};
        PJRT_Buffer* outputs[1] = {nullptr};
        PJRT_Buffer** output_lists[1] = {outputs};
        PJRT_Event* events[1] = {nullptr};
        PJRT_ExecuteOptions opts{};
        opts.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
        opts.non_donatable_input_indices = non_donatable;
        opts.num_non_donatable_input_indices = 2;
        PJRT_LoadedExecutable_Execute_Args args{};
        args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
        args.executable = exe;
        args.options = &opts;
        args.argument_lists = arg_lists;
        args.num_devices = 1;
        args.num_args = 2;
        args.output_lists = output_lists;
        args.device_complete_events = events;
        args.execute_device = device;
        if (!check_err(g_api->PJRT_LoadedExecutable_Execute(&args), "LoadedExecutable_Execute")) return false;
        if (!await_event(events[0], "Execute complete event")) return false;
        *out_buf = outputs[0];
        return outputs[0] != nullptr;
    };

    for (int i = 0; i < warmup; ++i) {
        PJRT_Buffer* out = nullptr;
        if (!run_once(&out)) return 7;
        destroy_buffer(out);
    }
    std::vector<double> times;
    for (int i = 0; i < iters; ++i) {
        PJRT_Buffer* out = nullptr;
        double t0 = now_ms();
        if (!run_once(&out)) return 7;
        times.push_back(now_ms() - t0);
        destroy_buffer(out);
    }
    std::sort(times.begin(), times.end());
    double median_ms = times[times.size() / 2];
    double min_ms = times.front();
    double gflops = 2.0 * (double)M * (double)N * (double)K / (median_ms * 1e6);
    std::printf("execute_median_ms=%.3f execute_min_ms=%.3f iters=%d warmup=%d approx_gflops=%.1f\n",
                median_ms, min_ms, iters, warmup, gflops);

    double max_abs_diff = -1.0;
    double max_rel_err = -1.0;
    int verify_pass = -1;
    if (verify) {
        PJRT_Buffer* out = nullptr;
        if (!run_once(&out)) return 7;
        std::vector<uint16_t> c_b;
        std::vector<float> c_f(nc);
        void* dst = is_bf16 ? (void*)nullptr : (void*)c_f.data();
        if (is_bf16) { c_b.resize(nc); dst = c_b.data(); }
        size_t dst_bytes = nc * (is_bf16 ? 2 : 4);
        PJRT_Buffer_ToHostBuffer_Args targs{};
        targs.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
        targs.src = out;
        targs.dst = dst;
        targs.dst_size = dst_bytes;
        if (!check_err(g_api->PJRT_Buffer_ToHostBuffer(&targs), "Buffer_ToHostBuffer")) return 8;
        if (!await_event(targs.event, "ToHostBuffer event")) return 8;
        if (is_bf16) for (size_t i = 0; i < nc; ++i) c_f[i] = bf16_to_float(c_b[i]);
        if (!save_out.empty()) {
            std::FILE* f = std::fopen(save_out.c_str(), "wb");
            if (!f) { std::fprintf(stderr, "SPIKE_FAIL cannot open save-out %s\n", save_out.c_str()); return 8; }
            std::fwrite(c_f.data(), sizeof(float), c_f.size(), f);
            std::fclose(f);
            std::printf("saved_out=%s\n", save_out.c_str());
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
            double d = std::fabs((double)c_f[i] - (double)ref[i]);
            if (d > max_abs_diff) max_abs_diff = d;
            double denom = std::max(1.0, std::fabs((double)ref[i]));
            double r = d / denom;
            if (r > max_rel_err) max_rel_err = r;
        }
        if (is_bf16) verify_pass = (max_rel_err < 1e-2) ? 1 : 0;
        else verify_pass = (max_abs_diff < 1e-4) ? 1 : 0;
        std::printf("verify_max_abs_diff=%.6g verify_max_rel_err=%.6g verify_pass=%d\n",
                    max_abs_diff, max_rel_err, verify_pass);
        destroy_buffer(out);
    }

    std::printf("RESULT_JSON {\"M\": %lld, \"K\": %lld, \"N\": %lld, \"dtype\": \"%s\", \"compile_ms\": %.3f, \"execute_median_ms\": %.3f, \"execute_min_ms\": %.3f, \"approx_gflops\": %.1f, \"num_devices\": %d, \"verify\": %d, \"max_abs_diff\": %.6g, \"max_rel_err\": %.6g, \"verify_pass\": %d}\n",
                (long long)M, (long long)K, (long long)N, dtype.c_str(), compile_ms, median_ms, min_ms, gflops,
                num_devices, verify, max_abs_diff, max_rel_err, verify_pass);

    destroy_buffer(buf_a);
    destroy_buffer(buf_b);
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
        std::fprintf(stderr, "SPIKE_FAIL numerics verification failed\n");
        return 9;
    }
    std::printf("SPIKE_OK\n");
    return 0;
}
