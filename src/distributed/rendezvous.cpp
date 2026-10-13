#include "litetorch/rendezvous.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <stdexcept>

namespace litetorch {
namespace distributed {
namespace rendezvous {

namespace {

void send_all(int fd, const char* buf, size_t n) {
    size_t sent = 0;
    while (sent < n) {
        ssize_t r = ::send(fd, buf + sent, n - sent, 0);
        if (r <= 0) throw std::runtime_error("[litetorch Error] rendezvous: send failed");
        sent += (size_t)r;
    }
}

void recv_all(int fd, char* buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = ::recv(fd, buf + got, n - got, 0);
        if (r <= 0) throw std::runtime_error("[litetorch Error] rendezvous: recv failed");
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
    if (n > 16 * 1024 * 1024) throw std::runtime_error("[litetorch Error] rendezvous: message too large");
    std::string body(n, '\0');
    if (n) recv_all(fd, &body[0], n);
    return body;
}

std::string jesc(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 2);
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else o += c;
    }
    return o;
}

std::string jget_str(const std::string& b, const char* key) {
    std::string k = std::string("\"") + key + "\"";
    size_t p = b.find(k);
    if (p == std::string::npos) throw std::runtime_error("[litetorch Error] rendezvous: missing key");
    p = b.find(':', p + k.size());
    if (p == std::string::npos) throw std::runtime_error("[litetorch Error] rendezvous: bad json");
    size_t q1 = b.find('"', p + 1);
    size_t q2 = b.find('"', q1 + 1);
    if (q1 == std::string::npos || q2 == std::string::npos) throw std::runtime_error("[litetorch Error] rendezvous: bad json str");
    return b.substr(q1 + 1, q2 - q1 - 1);
}

long jget_int(const std::string& b, const char* key) {
    std::string k = std::string("\"") + key + "\"";
    size_t p = b.find(k);
    if (p == std::string::npos) throw std::runtime_error("[litetorch Error] rendezvous: missing key");
    p = b.find(':', p + k.size());
    if (p == std::string::npos) throw std::runtime_error("[litetorch Error] rendezvous: bad json");
    return std::stol(b.substr(p + 1));
}

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int connect_nb(const std::string& addr, int port, int timeout_sec) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
    while (std::chrono::steady_clock::now() < deadline) {
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) throw std::runtime_error("[litetorch Error] rendezvous: socket failed");
        sockaddr_in sa{};
        sa.sin_family = AF_INET;
        sa.sin_port = htons((uint16_t)port);
        if (::inet_pton(AF_INET, addr.c_str(), &sa.sin_addr) != 1) {
            ::close(fd);
            throw std::runtime_error("[litetorch Error] rendezvous: bad addr");
        }
        if (::connect(fd, (sockaddr*)&sa, sizeof(sa)) == 0) return fd;
        ::close(fd);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    throw std::runtime_error("[litetorch Error] rendezvous: connect timed out");
}

}

RendezvousStore::RendezvousStore(const std::string& bind_addr, int port, int backlog) {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) throw std::runtime_error("[litetorch Error] rendezvous: socket failed");
    int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)port);
    if (::inet_pton(AF_INET, bind_addr.c_str(), &sa.sin_addr) != 1) {
        ::close(listen_fd_);
        throw std::runtime_error("[litetorch Error] rendezvous: bad bind_addr");
    }
    if (::bind(listen_fd_, (sockaddr*)&sa, sizeof(sa)) != 0) {
        ::close(listen_fd_);
        throw std::runtime_error("[litetorch Error] rendezvous: bind failed");
    }
    if (::listen(listen_fd_, backlog) != 0) {
        ::close(listen_fd_);
        throw std::runtime_error("[litetorch Error] rendezvous: listen failed");
    }
    sockaddr_in got{};
    socklen_t len = sizeof(got);
    if (::getsockname(listen_fd_, (sockaddr*)&got, &len) == 0) port_ = ntohs(got.sin_port);
    else port_ = port;
}

RendezvousStore::~RendezvousStore() {
    stop();
}

void RendezvousStore::stop() {
    running_.store(false);
    if (listen_fd_ >= 0) {
        ::shutdown(listen_fd_, SHUT_RDWR);
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
}

void RendezvousStore::run() {
    running_.store(true);
    while (running_.load()) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(listen_fd_, &rfds);
        timeval tv{0, 200000};
        int r = ::select(listen_fd_ + 1, &rfds, nullptr, nullptr, &tv);
        if (!running_.load()) break;
        if (r <= 0) continue;
        int fd = ::accept(listen_fd_, nullptr, nullptr);
        if (fd < 0) continue;
        std::thread([this, fd]() {
            try { serve_one(fd); } catch (...) {}
            ::close(fd);
        }).detach();
    }
}

void RendezvousStore::serve_one(int fd) {
    while (running_.load()) {
        std::string body;
        try {
            body = recv_msg(fd);
        } catch (...) {
            return;
        }
        bool close_after = false;
        std::string resp = handle(body, close_after);
        try {
            send_msg(fd, resp);
        } catch (...) {
            return;
        }
        if (close_after) return;
    }
}

std::string RendezvousStore::handle(const std::string& body, bool& close_after) {
    close_after = true;
    std::string op = jget_str(body, "op");
    if (op == "set") {
        std::string k = jget_str(body, "key");
        std::string v = jget_str(body, "value");
        {
            std::lock_guard<std::mutex> lk(mu_);
            kv_[k] = v;
        }
        return "{\"op\":\"ok\"}";
    }
    if (op == "get") {
        std::string k = jget_str(body, "key");
        long timeout = 30;
        try { timeout = jget_int(body, "timeout"); } catch (...) {}
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout);
        std::unique_lock<std::mutex> lk(mu_);
        while (running_.load()) {
            auto it = kv_.find(k);
            if (it != kv_.end())
                return "{\"op\":\"value\",\"value\":\"" + jesc(it->second) + "\"}";
            if (std::chrono::steady_clock::now() >= deadline)
                return "{\"op\":\"timeout\"}";
            lk.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            lk.lock();
        }
        return "{\"op\":\"timeout\"}";
    }
    if (op == "add") {
        std::string k = jget_str(body, "key");
        long d = jget_int(body, "delta");
        int64_t nv = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = kv_.find(k);
            int64_t cur = (it == kv_.end()) ? 0 : std::stoll(it->second);
            nv = cur + d;
            kv_[k] = std::to_string(nv);
        }
        return "{\"op\":\"value\",\"value\":\"" + std::to_string(nv) + "\"}";
    }
    if (op == "heartbeat") {
        int rank = (int)jget_int(body, "rank");
        {
            std::lock_guard<std::mutex> lk(mu_);
            heartbeats_[rank] = now_ms();
        }
        return "{\"op\":\"ok\"}";
    }
    if (op == "heartbeats") {
        std::string out = "{\"op\":\"heartbeats\"";
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (auto& kv : heartbeats_)
                out += ",\"r" + std::to_string(kv.first) + "\":" + std::to_string(kv.second);
        }
        out += "}";
        return out;
    }
    return "{\"op\":\"error\",\"msg\":\"unknown op\"}";
}

RendezvousClient::RendezvousClient(const std::string& addr, int port, int timeout_sec)
    : addr_(addr), port_(port), timeout_sec_(timeout_sec) {}

std::string RendezvousClient::roundtrip(const std::string& body, int timeout_sec) {
    int fd = connect_nb(addr_, port_, timeout_sec_);
    try {
        send_msg(fd, body);
        std::string resp = recv_msg(fd);
        ::close(fd);
        return resp;
    } catch (...) {
        ::close(fd);
        throw;
    }
}

void RendezvousClient::set(const std::string& key, const std::string& value) {
    std::string resp = roundtrip(
        "{\"op\":\"set\",\"key\":\"" + jesc(key) + "\",\"value\":\"" + jesc(value) + "\"}", timeout_sec_);
    if (jget_str(resp, "op") != "ok")
        throw std::runtime_error("[litetorch Error] rendezvous: set failed");
}

std::string RendezvousClient::get(const std::string& key, int timeout_sec) {
    long t = (timeout_sec < 0) ? timeout_sec_ : timeout_sec;
    std::string resp = roundtrip(
        "{\"op\":\"get\",\"key\":\"" + jesc(key) + "\",\"timeout\":" + std::to_string(t) + "}",
        (int)t + 5);
    std::string op = jget_str(resp, "op");
    if (op == "timeout")
        throw std::runtime_error("[litetorch Error] rendezvous: get timed out for key " + key);
    return jget_str(resp, "value");
}

int64_t RendezvousClient::add(const std::string& key, int64_t delta) {
    std::string resp = roundtrip(
        "{\"op\":\"add\",\"key\":\"" + jesc(key) + "\",\"delta\":" + std::to_string(delta) + "}",
        timeout_sec_);
    return std::stoll(jget_str(resp, "value"));
}

void RendezvousClient::heartbeat(int rank) {
    std::string resp = roundtrip(
        "{\"op\":\"heartbeat\",\"rank\":" + std::to_string(rank) + "}", timeout_sec_);
    if (jget_str(resp, "op") != "ok")
        throw std::runtime_error("[litetorch Error] rendezvous: heartbeat failed");
}

std::map<int, int64_t> RendezvousClient::get_heartbeats() {
    std::string resp = roundtrip("{\"op\":\"heartbeats\"}", timeout_sec_);
    std::map<int, int64_t> out;
    size_t p = 0;
    while (true) {
        size_t rk = resp.find("\"r", p);
        if (rk == std::string::npos) break;
        size_t qe = resp.find('"', rk + 2);
        if (qe == std::string::npos) break;
        int rank = std::stoi(resp.substr(rk + 2, qe - rk - 2));
        size_t c = resp.find(':', qe + 1);
        if (c == std::string::npos) break;
        size_t e = resp.find_first_of(",}", c + 1);
        if (e == std::string::npos) break;
        out[rank] = std::stoll(resp.substr(c + 1, e - c - 1));
        p = e + 1;
    }
    return out;
}

int free_port() {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("[litetorch Error] rendezvous: socket failed");
    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    sa.sin_port = 0;
    if (::bind(fd, (sockaddr*)&sa, sizeof(sa)) != 0) {
        ::close(fd);
        throw std::runtime_error("[litetorch Error] rendezvous: bind failed");
    }
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
