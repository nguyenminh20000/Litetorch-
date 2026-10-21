#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/nn.h"
#include "litetorch/distributed.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>

using namespace litetorch;

static int g_pass = 0;
static int g_fail = 0;

static void verdict(const std::string& name, bool ok, const std::string& detail = "") {
    if (ok) { ++g_pass; std::cout << "PASS " << name << std::endl; }
    else { ++g_fail; std::cout << "FAIL " << name << " :: " << detail << std::endl; }
}

static int wait_timeout(pid_t pid, int secs) {
    int status = 0;
    for (int i = 0; i < secs; ++i) {
        pid_t r = waitpid(pid, &status, WNOHANG);
        if (r == pid) return WIFEXITED(status) ? WEXITSTATUS(status) : -2;
        if (r < 0) return -3;
        sleep(1);
    }
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
    return -99;
}

struct TwoRank {
    int port;
    int (*rank_fn)(int);
    std::string name;
};

static bool run_tworank(TwoRank c, int timeout_s, std::string& detail) {
    pid_t pid = fork();
    if (pid < 0) { detail = "fork failed"; return false; }
    if (pid == 0) { _exit(c.rank_fn(1)); }
    int rc0 = c.rank_fn(0);
    int rc1 = wait_timeout(pid, timeout_s);
    if (rc0 == -99 || rc1 == -99) { detail = "HANG detected (timeout)"; return false; }
    if (rc0 == 0 && rc1 == 0) return true;
    detail = "rc0=" + std::to_string(rc0) + " rc1=" + std::to_string(rc1);
    return false;
}

static void init_rank(int rank, int port) {
    distributed::init_process_group(rank, 2, "127.0.0.1", port);
}

static int case_shape_mismatch(int rank) {
    init_rank(rank, 21011);
    bool threw = false;
    std::string msg;
    try {
        auto t = (rank == 0)
            ? Tensor::from_vector({1.0f, 2.0f}, {2})
            : Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
        distributed::all_reduce(t);
    } catch (const std::exception& e) { threw = true; msg = e.what(); }
    distributed::shutdown();
    if (!threw) { std::cerr << "rank " << rank << " did NOT throw on shape mismatch" << std::endl; return 2; }
    if (msg.find("mismatch") == std::string::npos) { std::cerr << "rank " << rank << " threw without 'mismatch': " << msg << std::endl; return 3; }
    return 0;
}

static int case_peer_abort(int rank) {
    init_rank(rank, 21012);
    if (rank == 1) { distributed::shutdown(); return 0; }
    bool threw = false;
    std::string msg;
    try {
        auto t = Tensor::from_vector({1.0f, 2.0f}, {2});
        distributed::all_reduce(t);
    } catch (const std::exception& e) { threw = true; msg = e.what(); }
    distributed::shutdown();
    if (!threw) { std::cerr << "rank 0 did NOT throw after peer exited (silent hang risk)" << std::endl; return 2; }
    return 0;
}

static int case_empty_allreduce(int rank) {
    init_rank(rank, 21013);
    bool ok = false;
    try {
        auto t = Tensor::create({0});
        distributed::all_reduce(t);
        ok = (t->numel() == 0);
    } catch (const std::exception& e) { std::cerr << "rank " << rank << " threw: " << e.what() << std::endl; }
    distributed::shutdown();
    return ok ? 0 : 4;
}

static int case_default_sum(int rank) {
    init_rank(rank, 21014);
    bool ok = false;
    try {
        auto t = (rank == 0) ? Tensor::from_vector({1.0f, 2.0f}, {2})
                             : Tensor::from_vector({3.0f, 4.0f}, {2});
        distributed::all_reduce(t);
        float* d = t->data_ptr();
        ok = (d[0] == 4.0f && d[1] == 6.0f);
        if (!ok) std::cerr << "rank " << rank << " got [" << d[0] << "," << d[1] << "] expected [4,6]" << std::endl;
    } catch (const std::exception& e) { std::cerr << "rank " << rank << " threw: " << e.what() << std::endl; }
    distributed::shutdown();
    return ok ? 0 : 5;
}

class TwoLinear : public nn::Module {
public:
    std::shared_ptr<nn::Linear> used = std::make_shared<nn::Linear>(4, 4);
    std::shared_ptr<nn::Linear> unused = std::make_shared<nn::Linear>(4, 4);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override { return used->forward(input); }
    std::vector<std::shared_ptr<Tensor>> parameters() override {
        auto p = used->parameters();
        auto q = unused->parameters();
        p.insert(p.end(), q.begin(), q.end());
        return p;
    }
    std::vector<std::shared_ptr<nn::Module>> children() override { return {used, unused}; }
    void to(const Device& d) override { used->to(d); unused->to(d); }
    void train() override {} void eval() override {}
};

static int case_find_unused_throw(int rank) {
    init_rank(rank, 21015);
    bool threw = false;
    std::string msg;
    try {
        auto model = std::make_shared<TwoLinear>();
        distributed::DistributedDataParallel ddp(model);
        ddp.set_find_unused_parameters(true);
        auto x = Tensor::from_vector({1,2,3,4}, {1, 4});
        auto out = ddp.forward(x);
        auto loss = Ops::sum(out);
        ddp.backward(loss);
    } catch (const std::exception& e) { threw = true; msg = e.what(); }
    distributed::shutdown();
    if (!threw) { std::cerr << "rank " << rank << " did NOT throw on unused params" << std::endl; return 2; }
    if (msg.find("no gradient") == std::string::npos) { std::cerr << "rank " << rank << " threw without 'no gradient': " << msg << std::endl; return 3; }
    return 0;
}

int main() {
    std::string detail;
    verdict("F1 shape-mismatch both-ranks-throw", run_tworank({21011, case_shape_mismatch, ""}, 60, detail), detail);
    verdict("F2 peer-exit must throw not hang", run_tworank({21012, case_peer_abort, ""}, 60, detail), detail);
    verdict("F3 empty-tensor all_reduce completes", run_tworank({21013, case_empty_allreduce, ""}, 60, detail), detail);
    verdict("F5b all_reduce default op is SUM", run_tworank({21014, case_default_sum, ""}, 60, detail), detail);
    verdict("F5c DDP find_unused throws", run_tworank({21015, case_find_unused_throw, ""}, 60, detail), detail);
    {
        int v = static_cast<int>(distributed::DistributedDataParallel::kDefaultDDPReduceOp);
        verdict("F5a DDP default reduce op is AVG", v == static_cast<int>(distributed::ReduceOp::AVG), "value=" + std::to_string(v));
    }
    std::cout << "----\nRESULT qa_fuzz4: " << g_pass << " passed, " << g_fail << " failed" << std::endl;
    return g_fail == 0 ? 0 : 1;
}
