#include "litetorch/flat_param.h"
#include "litetorch/fsdp_wrapper.h"
#include "litetorch/fsdp_tp.h"
#include "litetorch/distributed.h"
#include "litetorch/allocator.h"
#include "litetorch/nn.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <unistd.h>
#include <sys/wait.h>

using namespace litetorch;

static int failures = 0;
static void check(bool cond, const std::string& name) {
    std::cout << (cond ? "PASS " : "FAIL ") << name << std::endl;
    if (!cond) ++failures;
}

static bool vec_eq(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) return false;
    return true;
}

static long rss_kb() {
    FILE* f = fopen("/proc/self/statm", "r");
    if (!f) return -1;
    long size, resident;
    if (fscanf(f, "%ld %ld", &size, &resident) != 2) { fclose(f); return -1; }
    fclose(f);
    return resident * (sysconf(_SC_PAGESIZE) / 1024);
}

static std::shared_ptr<Tensor> fvec(const std::vector<float>& v) {
    return Tensor::from_vector(v, {(int64_t)v.size()}, Device(DeviceType::CPU, 0));
}

static void test_single() {
    CachingAllocator::get().empty_cache();
    const int64_t N = 2000000;
    std::vector<float> data((size_t)N, 1.5f);
    auto p = fvec(data);
    distributed::FlatParameter fp({p});
    int64_t flat = fp.flat_numel();
    check(flat == N, "single: flat_numel");
#ifdef DIST_W4_NEW
    check(!fp.has_full_buffer(), "single: full buffer lazy (not allocated in ctor)");
    check(fp.full_buffer() == nullptr, "single: full_buffer() null before gather");
#else
    check(fp.full_buffer() != nullptr, "old: full buffer eagerly allocated in ctor");
#endif
    bool threw = false;
    try { fp.param_views(); } catch (const std::exception&) { threw = true; }
    check(threw, "single: param_views throws before gather");
    long rss0 = rss_kb();
    fp.all_gather_flat();
    check(fp.full_buffer() != nullptr, "single: full buffer allocated on gather");
    check(vec_eq(fp.full_buffer()->to_vector(), data), "single: gather values correct");
    {
        auto views = fp.param_views();
        check(views.size() == 1 && vec_eq(views[0]->to_vector(), data), "single: param_views after gather");
    }
    size_t c1 = CachingAllocator::get().get_cached_cpu_bytes();
#ifdef DIST_W4_NEW
    fp.release_full_buffer();
#endif
    long rss2 = rss_kb();
    size_t c2 = CachingAllocator::get().get_cached_cpu_bytes();
#ifdef DIST_W4_NEW
    check(!fp.has_full_buffer(), "single: buffer freed on release");
    check(fp.full_buffer() == nullptr, "single: full_buffer() null after release");
    check(c2 == c1 + (size_t)flat * 4, "single: full buffer bytes returned to allocator on release");
#else
    (void)c1; (void)c2;
#endif
    std::cout << "INFO single: flat_numel=" << flat << " rss_ctor_kb=" << rss0
              << " cache_before_release=" << c1 << " cache_after_release=" << c2
              << " rss_after_release_kb=" << rss2 << std::endl;
    fp.all_gather_flat();
    check(vec_eq(fp.full_buffer()->to_vector(), data), "single: re-gather after release works");
    auto gs = fp.reduce_scatter_flat(fvec(std::vector<float>((size_t)flat, 2.0f)));
    check(vec_eq(gs->to_vector(), std::vector<float>((size_t)flat, 2.0f)), "single: reduce_scatter world=1");
}

static void test_cache_path() {
#ifdef DIST_W4_NEW
    CachingAllocator::get().empty_cache();
    const int64_t N = 10000;
    auto p = fvec(std::vector<float>((size_t)N, 2.5f));
    distributed::FlatParameter fp({p});
    fp.all_gather_flat();
    size_t c1 = CachingAllocator::get().get_cached_cpu_bytes();
    fp.release_full_buffer();
    size_t c2 = CachingAllocator::get().get_cached_cpu_bytes();
    std::cout << "INFO cache: flat_numel=" << fp.flat_numel() << " cache_after_gather=" << c1
              << " cache_after_release=" << c2 << std::endl;
    check(c2 >= c1 + (size_t)fp.flat_numel() * 4, "cache: freed bytes returned to allocator cache");
#endif
}

static void test_fsdp_unit() {
    CachingAllocator::get().empty_cache();
    auto lin = std::make_shared<nn::Linear>(8, 4, true);
    auto w0 = lin->weight->to_vector();
    auto b0 = lin->bias->to_vector();
    distributed::FSDP::fully_shard(lin);
    auto units = distributed::FSDP::active_units();
    check(units.size() == 1, "fsdp: one unit created");
    auto u = units.back();
#ifdef DIST_W4_NEW
    check(!u->flat->has_full_buffer(), "fsdp: full buffer lazy after fully_shard");
#endif
    u->ensure_gathered();
#ifdef DIST_W4_NEW
    check(u->flat->has_full_buffer(), "fsdp: buffer present after ensure_gathered");
#endif
    check(u->is_gathered(), "fsdp: is_gathered true");
    u->copy_flat_to_params();
    check(vec_eq(lin->weight->to_vector(), w0) && vec_eq(lin->bias->to_vector(), b0), "fsdp: params intact after copy_flat_to_params");
    size_t c1 = CachingAllocator::get().get_cached_cpu_bytes();
    u->release_gathered();
    size_t c2 = CachingAllocator::get().get_cached_cpu_bytes();
#ifdef DIST_W4_NEW
    check(!u->flat->has_full_buffer(), "fsdp: release_gathered frees full buffer");
#endif
    check(!u->is_gathered(), "fsdp: is_gathered false after release");
#ifdef DIST_W4_NEW
    check(c2 > c1, "fsdp: release_gathered returns bytes to cache");
#else
    (void)c1; (void)c2;
#endif
    std::cout << "INFO fsdp: flat_numel=" << u->flat->flat_numel() << " cache_delta=" << (c2 - c1) << std::endl;
    u->ensure_gathered();
#ifdef DIST_W4_NEW
    check(u->flat->has_full_buffer(), "fsdp: re-gather after release works");
#endif
    u->copy_flat_to_params();
    check(vec_eq(lin->weight->to_vector(), w0), "fsdp: params intact after re-gather");
    u->release_gathered();
    u->prefetch_gather();
    u->release_gathered();
#ifdef DIST_W4_NEW
    check(!u->flat->has_full_buffer(), "fsdp: prefetch then release is safe (no hang/crash)");
#endif
    u->ensure_gathered();
    check(u->is_gathered(), "fsdp: gather works after prefetch-release cycle");
    u->release_gathered();
}

static void run_rank2(int rank) {
    distributed::init_process_group(rank, 2, "127.0.0.1", 9981);
    std::vector<float> data;
    for (int i = 1; i <= 11; ++i) data.push_back((float)i);
    distributed::FlatParameter fp({fvec(data)});
    check(fp.shard_numel() == 6 && fp.flat_numel() == 12, "m2: shard math (11 elems, world 2)");
    auto sv = fp.local_shard()->to_vector();
    bool ok = sv.size() == 6;
    for (int i = 0; i < 6 && ok; ++i) {
        int gi = rank * 6 + i;
        float expect = gi < 11 ? (float)(gi + 1) : 0.0f;
        ok = sv[i] == expect;
    }
    check(ok, "m2: local shard contents (padded)");
    fp.all_gather_flat();
    std::vector<float> padded = data;
    padded.push_back(0.0f);
    check(vec_eq(fp.full_buffer()->to_vector(), padded), "m2: all_gather values");
    {
        auto views = fp.param_views();
        check(views.size() == 1 && vec_eq(views[0]->to_vector(), data), "m2: param_views");
    }
    auto gs = fp.reduce_scatter_flat(fvec(std::vector<float>(12, (float)(rank + 1))));
    check(vec_eq(gs->to_vector(), std::vector<float>(6, 3.0f)), "m2: reduce_scatter SUM across ranks");
#ifdef DIST_W4_NEW
    fp.release_full_buffer();
    check(!fp.has_full_buffer(), "m2: released");
#endif
    fp.all_gather_flat();
    check(vec_eq(fp.full_buffer()->to_vector(), padded), "m2: re-gather after release");
    {
        distributed::FlatParameter fq({fvec({7.0f})});
        check(fq.shard_numel() == 1 && fq.flat_numel() == 2, "m2: world>numel shard math");
        float expect_shard = rank == 0 ? 7.0f : 0.0f;
        check(fq.local_shard()->to_vector()[0] == expect_shard, "m2: world>numel shard contents");
        fq.all_gather_flat();
        check(vec_eq(fq.full_buffer()->to_vector(), {7.0f, 0.0f}), "m2: world>numel all_gather");
        fq.release_full_buffer();
#ifdef DIST_W4_NEW
        check(!fq.has_full_buffer(), "m2: world>numel released");
#endif
    }
    distributed::shutdown();
}

static void run_rank4(int rank) {
    distributed::init_process_group(rank, 4, "127.0.0.1", 9982);
    distributed::FSDPTP::init(2, 2);
    distributed::FlatParameter fp({fvec({1, 2, 3, 4, 5})});
    bool threw = false;
    std::string msg;
    try { fp.all_gather_flat(); } catch (const std::exception& e) { threw = true; msg = e.what(); }
    check(threw && msg.find("hybrid TP+DP") != std::string::npos, "m4: hybrid CPU all_gather throws fail-closed");
    bool threw2 = false;
    try { fp.reduce_scatter_flat(fvec(std::vector<float>(8, 1.0f))); } catch (const std::exception&) { threw2 = true; }
    check(threw2, "m4: hybrid CPU reduce_scatter throws fail-closed");
    distributed::shutdown();
}

static int run_forked(int world, void (*fn)(int), int port_base) {
    (void)port_base;
    std::vector<pid_t> kids;
    for (int r = 1; r < world; ++r) {
        pid_t pid = fork();
        if (pid == 0) { fn(r); _exit(failures ? 1 : 0); }
        kids.push_back(pid);
    }
    fn(0);
    int rc = failures ? 1 : 0;
    for (auto pid : kids) {
        int st = 0;
        waitpid(pid, &st, 0);
        if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) rc = 1;
    }
    return rc;
}

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "single";
    if (mode == "single") {
        test_single();
        test_cache_path();
        test_fsdp_unit();
    } else if (mode == "m2") {
        return run_forked(2, run_rank2, 9981);
    } else if (mode == "m4") {
        return run_forked(4, run_rank4, 9982);
    } else {
        std::cerr << "unknown mode" << std::endl;
        return 2;
    }
    std::cout << (failures ? "FAILURES: " : "ALL PASS, failures=") << failures << std::endl;
    return failures ? 1 : 0;
}
