#include "litetorch/memory_manager.h"
#include <iostream>
#include <string>

using namespace litetorch;

static int g_pass = 0;
static int g_fail = 0;

static void check(const std::string& name, bool ok, const std::string& msg = "") {
    if (ok) { ++g_pass; std::cout << "PASS " << name << "\n"; }
    else { ++g_fail; std::cout << "FAIL " << name << " :: " << msg << "\n"; }
}

static void t_eviction_accounting_sane() {
    auto& mm = MemoryManager::get();
    size_t used0 = mm.get_gpu_used();
    bool threw = false;
    try {
        mm.evict_to_free(1024);
    } catch (...) { threw = true; }
    check("eviction_accounting_no_crash", !threw && mm.get_gpu_used() == used0,
          "evict_to_free crashed or changed gpu-side accounting");
}

int main() {
    t_eviction_accounting_sane();
    std::cout << "----\nRESULT test_eviction_accounting: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
