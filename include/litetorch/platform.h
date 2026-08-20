#ifndef LITETORCH_PLATFORM_H
#define LITETORCH_PLATFORM_H

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#include <direct.h>

#define RTLD_LAZY 0
#define RTLD_NOW 0
#define RTLD_GLOBAL 0

inline void* dlopen(const char* filename, int) {
    if (!filename) return (void*)GetModuleHandleA(NULL);
    return (void*)LoadLibraryA(filename);
}

inline void* dlsym(void* handle, const char* symbol) {
    return (void*)GetProcAddress((HMODULE)handle, symbol);
}

inline int dlclose(void* handle) {
    return FreeLibrary((HMODULE)handle) ? 0 : -1;
}

inline const char* dlerror() {
    return "Dynamic load error";
}

#else
#include <dlfcn.h>
#include <unistd.h>
#endif

#endif
