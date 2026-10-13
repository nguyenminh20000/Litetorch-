#include "litetorch/profiler.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace litetorch {
namespace profiler {

namespace {

struct Agg {
    int64_t calls = 0;
    double total_ms = 0.0;
    double max_ms = 0.0;
};

std::mutex& mtx() {
    static std::mutex m;
    return m;
}

std::unordered_map<std::string, Agg>& table() {
    static std::unordered_map<std::string, Agg> t;
    return t;
}

bool& enabled_flag() {
    static bool e = false;
    return e;
}

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

}

void enable() {
    std::lock_guard<std::mutex> l(mtx());
    enabled_flag() = true;
}

void disable() {
    std::lock_guard<std::mutex> l(mtx());
    enabled_flag() = false;
}

bool is_enabled() {
    std::lock_guard<std::mutex> l(mtx());
    return enabled_flag();
}

void reset() {
    std::lock_guard<std::mutex> l(mtx());
    table().clear();
}

void record(const std::string& name, double ms) {
    std::lock_guard<std::mutex> l(mtx());
    if (!enabled_flag()) return;
    Agg& a = table()[name];
    a.calls += 1;
    a.total_ms += ms;
    if (ms > a.max_ms) a.max_ms = ms;
}

std::vector<Event> events() {
    std::lock_guard<std::mutex> l(mtx());
    std::vector<Event> out;
    for (auto& kv : table()) {
        Event e;
        e.name = kv.first;
        e.calls = kv.second.calls;
        e.total_ms = kv.second.total_ms;
        e.max_ms = kv.second.max_ms;
        out.push_back(e);
    }
    return out;
}

std::string report() {
    auto evs = events();
    std::sort(evs.begin(), evs.end(),
              [](const Event& a, const Event& b) { return a.total_ms > b.total_ms; });
    std::ostringstream os;
    os << std::left << std::setw(40) << "name" << std::right
       << std::setw(10) << "calls" << std::setw(14) << "total_ms"
       << std::setw(14) << "avg_ms" << std::setw(14) << "max_ms" << "\n";
    os << std::string(92, '-') << "\n";
    os << std::fixed << std::setprecision(3);
    for (auto& e : evs) {
        os << std::left << std::setw(40) << e.name.substr(0, 39) << std::right
           << std::setw(10) << e.calls << std::setw(14) << e.total_ms
           << std::setw(14) << e.avg_ms() << std::setw(14) << e.max_ms << "\n";
    }
    return os.str();
}

ScopedTimer::ScopedTimer(const std::string& name)
    : name_(name), start_ns_(now_ns()), active_(is_enabled()) {}

ScopedTimer::~ScopedTimer() {
    if (active_) record(name_, (now_ns() - start_ns_) / 1e6);
}

}
}
