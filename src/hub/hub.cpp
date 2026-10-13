#include "litetorch/hub.h"
#include "litetorch/serialization.h"
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <stdexcept>
#include <sys/stat.h>

namespace litetorch {
namespace hub {

namespace {

std::string& dir_override() {
    static std::string d;
    return d;
}

bool file_exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

void ensure_dir(const std::string& p) {
    std::string cmd = std::string("mkdir -p '") + p + "'";
    if (std::system(cmd.c_str()) != 0)
        throw std::runtime_error("[litetorch Error] hub: cannot create dir " + p);
}

typedef void CURL;
typedef int CURLcode;
typedef size_t (*curl_write_cb)(char*, size_t, size_t, void*);
struct CurlApi {
    void* handle = nullptr;
    CURL* (*easy_init)() = nullptr;
    CURLcode (*easy_setopt)(CURL*, int, ...) = nullptr;
    CURLcode (*easy_perform)(CURL*) = nullptr;
    void (*easy_cleanup)(CURL*) = nullptr;
    const char* (*easy_strerror)(CURLcode) = nullptr;
};

CurlApi& api() {
    static CurlApi a;
    static bool loaded = false;
    if (!loaded) {
        loaded = true;
        a.handle = dlopen("libcurl.so.4", RTLD_LAZY);
        if (!a.handle) a.handle = dlopen("libcurl.so", RTLD_LAZY);
        if (a.handle) {
            a.easy_init = (CURL*(*)())dlsym(a.handle, "curl_easy_init");
            a.easy_setopt = (CURLcode(*)(CURL*, int, ...))dlsym(a.handle, "curl_easy_setopt");
            a.easy_perform = (CURLcode(*)(CURL*))dlsym(a.handle, "curl_easy_perform");
            a.easy_cleanup = (void(*)(CURL*))dlsym(a.handle, "curl_easy_cleanup");
            a.easy_strerror = (const char*(*)(CURLcode))dlsym(a.handle, "curl_easy_strerror");
        }
    }
    return a;
}

size_t write_file_cb(char* ptr, size_t size, size_t nmemb, void* userdata) {
    FILE* f = static_cast<FILE*>(userdata);
    return fwrite(ptr, size, nmemb, f);
}

int progress_cb(void*, double dltotal, double dlnow, double, double) {
    if (dltotal > 0) {
        int pct = (int)(100.0 * dlnow / dltotal);
        fprintf(stderr, "\r[litetorch hub] downloading %d%%", pct);
        fflush(stderr);
    }
    return 0;
}

}

std::string get_dir() {
    if (!dir_override().empty()) return dir_override();
    const char* home = std::getenv("HOME");
    std::string base = home ? home : "/tmp";
    return base + "/.cache/litetorch/hub";
}

void set_dir(const std::string& dir) {
    dir_override() = dir;
}

bool is_url(const std::string& path) {
    return path.rfind("http://", 0) == 0 || path.rfind("https://", 0) == 0;
}

std::string filename_from_url(const std::string& url) {
    size_t q = url.find('?');
    std::string clean = (q == std::string::npos) ? url : url.substr(0, q);
    size_t s = clean.rfind('/');
    std::string name = (s == std::string::npos) ? clean : clean.substr(s + 1);
    if (name.empty()) name = "download.bin";
    return name;
}

std::string download_url_to_file(const std::string& url, const std::string& dst,
                                 bool show_progress) {
    CurlApi& c = api();
    if (!c.handle || !c.easy_init)
        throw std::runtime_error("[litetorch Error] hub: libcurl not available (dlopen failed)");
    std::string out = dst.empty() ? get_dir() + "/" + filename_from_url(url) : dst;
    size_t slash = out.rfind('/');
    if (slash != std::string::npos) ensure_dir(out.substr(0, slash));
    FILE* f = fopen(out.c_str(), "wb");
    if (!f) throw std::runtime_error("[litetorch Error] hub: cannot write " + out);
    CURL* h = c.easy_init();
    if (!h) { fclose(f); throw std::runtime_error("[litetorch Error] hub: curl init failed"); }
    c.easy_setopt(h, 10002, url.c_str());
    c.easy_setopt(h, 20011, write_file_cb);
    c.easy_setopt(h, 10001, f);
    c.easy_setopt(h, 52, 1L);
    c.easy_setopt(h, 43, show_progress ? 0L : 1L);
    if (show_progress) c.easy_setopt(h, 20056, progress_cb);
    CURLcode rc = c.easy_perform(h);
    c.easy_cleanup(h);
    fclose(f);
    if (show_progress) fprintf(stderr, "\n");
    if (rc != 0) {
        std::string err = c.easy_strerror ? c.easy_strerror(rc) : "unknown";
        throw std::runtime_error("[litetorch Error] hub: download failed: " + err);
    }
    return out;
}

std::map<std::string, std::shared_ptr<Tensor>> load_state_dict_from_url(
    const std::string& url, bool force_download) {
    std::string cached = get_dir() + "/" + filename_from_url(url);
    if (force_download || !file_exists(cached)) cached = download_url_to_file(url, cached);
    return load_safetensors(cached);
}

}
}
