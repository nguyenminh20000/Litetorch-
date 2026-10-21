#include "litetorch/allocator.h"
#include "litetorch/backend.h"
#include <iostream>
#include <string>
#include <stdexcept>

using namespace litetorch;

static int g_pass = 0;
static int g_fail = 0;

static void check(const std::string& name, bool ok, const std::string& msg = "") {
    if (ok) { ++g_pass; std::cout << "PASS " << name << "\n"; }
    else { ++g_fail; std::cout << "FAIL " << name << " :: " << msg << "\n"; }
}

static bool gpu_available() {
    auto b = BackendDispatcher::get().get_backend();
    return b && b->is_available();
}

static void t_free_gpu_now_unknown_throws() {
    bool threw = false;
    std::string what;
    try {
        CachingAllocator::get().free_gpu_now((void*)0x12345678);
    } catch (const std::runtime_error& e) { threw = true; what = e.what(); }
    bool msg_ok = threw && what.find("unknown pointer") != std::string::npos;
    check("free_gpu_now_unknown_throws", msg_ok, threw ? what : "no throw");
}

static void t_free_gpu_now_null_noop() {
    bool threw = false;
    try { CachingAllocator::get().free_gpu_now(nullptr); }
    catch (...) { threw = true; }
    check("free_gpu_now_null_noop", !threw, "threw on nullptr");
}

static void t_free_gpu_now_known_pointer() {
    if (!gpu_available()) {
        std::cout << "SKIP free_gpu_now_known_pointer (KAGGLE-GATED: requires T4 GPU)\n";
        return;
    }
    auto& a = CachingAllocator::get();
    size_t before = a.get_cached_gpu_bytes();
    void* p = a.allocate_gpu(1 << 20);
    check("free_gpu_now_alloc", p != nullptr, "allocate_gpu returned null on GPU");
    if (!p) return;
    a.free_gpu_now(p);
    check("free_gpu_now_no_cache_hoard", a.get_cached_gpu_bytes() == before,
          "cached bytes grew after free_gpu_now");
    bool threw = false;
    try { a.free_gpu_now(p); } catch (const std::runtime_error&) { threw = true; }
    check("free_gpu_now_double_free_throws", threw, "second free_gpu_now did not throw");
}

static void kaggle_instructions() {
    std::cout << "KAGGLE-GATED free_gpu_now eviction verification (mandatory QA-phase item for H11):\n";
    std::cout << "  1. On Kaggle T4 x2: run this binary. free_gpu_now_known_pointer must PASS.\n";
    std::cout << "  2. nvidia-smi before/after: allocate 400MB via Tensor on GPU, evict via MemoryManager evict path (or swap), run `nvidia-smi --query-gpu=memory.used --format=csv,noheader` before and after. Expect device footprint to drop by ~400MB (true release), not stay parked in cache.\n";
    std::cout << "  3. Repeat alloc/evict 50x: footprint must not grow (no leak), cached bytes stay 0.\n";
}

int main() {
    t_free_gpu_now_unknown_throws();
    t_free_gpu_now_null_noop();
    t_free_gpu_now_known_pointer();
    kaggle_instructions();
    std::cout << "----\nRESULT test_free_gpu_now: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
