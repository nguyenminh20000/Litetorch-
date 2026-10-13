#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace litetorch {
namespace distributed {
namespace launcher {

struct RendezvousInfo {
    int rank_offset = 0;
    int world_size = 0;
    int node_rank = 0;
    int nnodes = 0;
    std::string master_addr;
    int master_port = 0;
};

struct RendezvousResult {
    std::map<int, int> rank_offsets;
    int world_size = 0;
};

namespace {

void send_all(int fd, const char* buf, size_t n) {
    size_t sent = 0;
    while (sent < n) {
        ssize_t r = ::send(fd, buf + sent, n - sent, 0);
        if (r <= 0) throw std::runtime_error("launcher: send failed");
        sent += (size_t)r;
    }
}

void recv_all(int fd, char* buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = ::recv(fd, buf + got, n - got, 0);
        if (r <= 0) throw std::runtime_error("launcher: recv failed");
        got += (size_t)r;
    }
}

void send_msg(int fd, const std::string& body) {
    uint32_t n = htonl((uint32_t)body.size());
    send_all(fd, (const char*)&n, 4);
    if (!body.empty()) send_all(fd, body.data(), body.size());
}

std::string recv_msg(int fd) {
    uint32_t nnet = 0;
    recv_all(fd, (char*)&nnet, 4);
    uint32_t n = ntohl(nnet);
    if (n > 16 * 1024 * 1024) throw std::runtime_error("launcher: message too large");
    std::string body(n, '\0');
    if (n) recv_all(fd, body.data(), n);
    return body;
}

std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c == '\n') out += "\\n";
        else out += c;
    }
    return out;
}

long json_int(const std::string& body, const char* key) {
    std::string k = std::string("\"") + key + "\"";
    size_t p = body.find(k);
    if (p == std::string::npos) throw std::runtime_error("launcher: missing json key");
    p = body.find(':', p + k.size());
    if (p == std::string::npos) throw std::runtime_error("launcher: bad json");
    return std::stol(body.substr(p + 1));
}

std::string json_str(const std::string& body, const char* key) {
    std::string k = std::string("\"") + key + "\"";
    size_t p = body.find(k);
    if (p == std::string::npos) throw std::runtime_error("launcher: missing json key");
    p = body.find(':', p + k.size());
    if (p == std::string::npos) throw std::runtime_error("launcher: bad json");
    size_t q1 = body.find('"', p + 1);
    size_t q2 = body.find('"', q1 + 1);
    if (q1 == std::string::npos || q2 == std::string::npos) throw std::runtime_error("launcher: bad json");
    return body.substr(q1 + 1, q2 - q1 - 1);
}

int connect_with_retry(const std::string& addr, int port, int timeout_sec) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
    while (std::chrono::steady_clock::now() < deadline) {
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) throw std::runtime_error("launcher: socket failed");
        sockaddr_in sa{};
        sa.sin_family = AF_INET;
        sa.sin_port = htons((uint16_t)port);
        if (::inet_pton(AF_INET, addr.c_str(), &sa.sin_addr) != 1) {
            ::close(fd);
            throw std::runtime_error("launcher: bad master_addr");
        }
        if (::connect(fd, (sockaddr*)&sa, sizeof(sa)) == 0) return fd;
        ::close(fd);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    throw std::runtime_error("launcher: connect to rendezvous timed out");
}

}

class RendezvousServer {
public:
    RendezvousServer(const std::string& bind_addr, int port, int nnodes, int timeout_sec)
        : nnodes_(nnodes), timeout_sec_(timeout_sec) {
        if (nnodes_ < 1) throw std::runtime_error("launcher: nnodes must be >= 1");
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0) throw std::runtime_error("launcher: socket failed");
        int one = 1;
        ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in sa{};
        sa.sin_family = AF_INET;
        sa.sin_port = htons((uint16_t)port);
        if (::inet_pton(AF_INET, bind_addr.c_str(), &sa.sin_addr) != 1) {
            ::close(listen_fd_);
            throw std::runtime_error("launcher: bad bind_addr");
        }
        if (::bind(listen_fd_, (sockaddr*)&sa, sizeof(sa)) != 0) {
            ::close(listen_fd_);
            throw std::runtime_error("launcher: bind failed");
        }
        if (::listen(listen_fd_, 64) != 0) {
            ::close(listen_fd_);
            throw std::runtime_error("launcher: listen failed");
        }
        sockaddr_in got{};
        socklen_t len = sizeof(got);
        if (::getsockname(listen_fd_, (sockaddr*)&got, &len) == 0) port_ = ntohs(got.sin_port);
        else port_ = port;
    }

    ~RendezvousServer() {
        if (listen_fd_ >= 0) ::close(listen_fd_);
    }

    int port() const { return port_; }

    RendezvousResult run() {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec_);
        std::map<int, int> nprocs;
        std::map<int, int> fds;
        while ((int)nprocs.size() < nnodes_) {
            if (std::chrono::steady_clock::now() > deadline)
                throw std::runtime_error("launcher: rendezvous join timed out");
            fd_set rfds;
            FD_ZERO(&rfds);
            FD_SET(listen_fd_, &rfds);
            timeval tv{0, 200000};
            int r = ::select(listen_fd_ + 1, &rfds, nullptr, nullptr, &tv);
            if (r <= 0) continue;
            int fd = ::accept(listen_fd_, nullptr, nullptr);
            if (fd < 0) continue;
            try {
                std::string msg = recv_msg(fd);
                int nr = (int)json_int(msg.c_str(), "node_rank");
                int np = (int)json_int(msg.c_str(), "nproc");
                if (nprocs.count(nr)) {
                    send_msg(fd, "{\"op\":\"error\",\"msg\":\"duplicate node_rank\"}");
                    ::close(fd);
                    continue;
                }
                nprocs[nr] = np;
                fds[nr] = fd;
            } catch (...) {
                ::close(fd);
            }
        }
        std::map<int, int> offsets;
        int offset = 0;
        for (auto& kv : nprocs) {
            offsets[kv.first] = offset;
            offset += kv.second;
        }
        int world = offset;
        for (auto& kv : fds) {
            int nr = kv.first;
            std::string a = "{\"op\":\"assign\",\"rank_offset\":" + std::to_string(offsets[nr]) +
                ",\"world_size\":" + std::to_string(world) +
                ",\"node_rank\":" + std::to_string(nr) +
                ",\"nnodes\":" + std::to_string(nnodes_) + "}";
            send_msg(kv.second, a);
        }
        int ready = 0;
        for (auto& kv : fds) {
            std::string m = recv_msg(kv.second);
            if (json_str(m.c_str(), "op") == "ready") ready++;
        }
        if (ready != nnodes_) throw std::runtime_error("launcher: ready barrier failed");
        for (auto& kv : fds) {
            send_msg(kv.second, "{\"op\":\"go\"}");
            ::close(kv.second);
        }
        ::close(listen_fd_);
        listen_fd_ = -1;
        RendezvousResult res;
        res.rank_offsets = offsets;
        res.world_size = world;
        return res;
    }

private:
    int nnodes_;
    int timeout_sec_;
    int listen_fd_ = -1;
    int port_ = 0;
};

RendezvousInfo rendezvous_join(const std::string& master_addr, int master_port,
                              int node_rank, int nproc, int nnodes, int timeout_sec) {
    if (nnodes < 1 || nproc < 1) throw std::runtime_error("launcher: nnodes/nproc must be >= 1");
    if (node_rank < 0 || node_rank >= nnodes) throw std::runtime_error("launcher: bad node_rank");
    int fd = connect_with_retry(master_addr, master_port, timeout_sec);
    char host[256] = {0};
    ::gethostname(host, sizeof(host) - 1);
    std::string join = "{\"op\":\"join\",\"node_rank\":" + std::to_string(node_rank) +
        ",\"nproc\":" + std::to_string(nproc) +
        ",\"nnodes\":" + std::to_string(nnodes) +
        ",\"hostname\":\"" + json_escape(host) + "\"}";
    send_msg(fd, join);
    std::string a = recv_msg(fd);
    if (json_str(a.c_str(), "op") == "error")
        throw std::runtime_error("launcher: rendezvous rejected");
    RendezvousInfo info;
    info.rank_offset = (int)json_int(a.c_str(), "rank_offset");
    info.world_size = (int)json_int(a.c_str(), "world_size");
    info.node_rank = (int)json_int(a.c_str(), "node_rank");
    info.nnodes = (int)json_int(a.c_str(), "nnodes");
    info.master_addr = master_addr;
    info.master_port = master_port;
    std::string ready = "{\"op\":\"ready\",\"node_rank\":" + std::to_string(node_rank) + "}";
    send_msg(fd, ready);
    std::string go = recv_msg(fd);
    if (json_str(go.c_str(), "op") != "go") throw std::runtime_error("launcher: expected go");
    ::close(fd);
    return info;
}

int free_port() {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("launcher: socket failed");
    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    sa.sin_port = 0;
    if (::bind(fd, (sockaddr*)&sa, sizeof(sa)) != 0) { ::close(fd); throw std::runtime_error("launcher: bind failed"); }
    sockaddr_in got{};
    socklen_t len = sizeof(got);
    int p = 0;
    if (::getsockname(fd, (sockaddr*)&got, &len) == 0) p = ntohs(got.sin_port);
    ::close(fd);
    return p;
}

}
}
}
