#ifndef LITETORCH_RENDEZVOUS_H
#define LITETORCH_RENDEZVOUS_H

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace litetorch {
namespace distributed {
namespace rendezvous {

class RendezvousStore {
public:
    RendezvousStore(const std::string& bind_addr, int port, int backlog = 64);
    ~RendezvousStore();
    RendezvousStore(const RendezvousStore&) = delete;
    RendezvousStore& operator=(const RendezvousStore&) = delete;
    int port() const { return port_; }
    void run();
    void stop();
private:
    void serve_one(int fd);
    std::string handle(const std::string& body, bool& close_after);
    int listen_fd_ = -1;
    int port_ = 0;
    std::atomic<bool> running_{false};
    std::mutex mu_;
    std::map<std::string, std::string> kv_;
    std::map<int, int64_t> heartbeats_;
};

class RendezvousClient {
public:
    RendezvousClient(const std::string& addr, int port, int timeout_sec = 30);
    void set(const std::string& key, const std::string& value);
    std::string get(const std::string& key, int timeout_sec = -1);
    int64_t add(const std::string& key, int64_t delta);
    void heartbeat(int rank);
    std::map<int, int64_t> get_heartbeats();
private:
    std::string roundtrip(const std::string& body, int timeout_sec);
    std::string addr_;
    int port_;
    int timeout_sec_;
};

int free_port();

}
}
}

#endif
