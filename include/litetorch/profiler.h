#ifndef LITETORCH_PROFILER_H
#define LITETORCH_PROFILER_H

#include <cstdint>
#include <string>
#include <vector>

namespace litetorch {
namespace profiler {

void enable();
void disable();
bool is_enabled();
void reset();

struct Event {
    std::string name;
    int64_t calls = 0;
    double total_ms = 0.0;
    double max_ms = 0.0;
    double avg_ms() const { return calls ? total_ms / (double)calls : 0.0; }
};

std::vector<Event> events();
std::string report();
void record(const std::string& name, double ms);

class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name);
    ~ScopedTimer();
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
private:
    std::string name_;
    int64_t start_ns_;
    bool active_;
};

}
}

#endif
