#include "litetorch/optim.h"
#include "litetorch/thread_pool.h"
#include "litetorch/cl_backend.h"
#include "litetorch/backend.h"
#include "litetorch/backend_route.h"
#include "optim_utils.h"
#include <atomic>
#include <cmath>

namespace litetorch {
namespace optim {

namespace {
std::atomic<int>& fused_launches() {
    static std::atomic<int> c{0};
    return c;
}
std::atomic<int>& single_launches() {
    static std::atomic<int> c{0};
    return c;
}
}

int AdamW::fused_launch_count() { return fused_launches().load(); }
int AdamW::single_launch_count() { return single_launches().load(); }
void AdamW::reset_launch_counts() {
    fused_launches().store(0);
    single_launches().store(0);
}

AdamW::AdamW(const std::vector<std::shared_ptr<Tensor>>& params, double lr, float beta1, float beta2, float eps, float weight_decay, bool offload_to_cpu)
    : Optimizer(params), lr(lr), beta1(beta1), beta2(beta2), eps(eps), weight_decay(weight_decay), offload_to_cpu(offload_to_cpu) {
    for (auto& p : params) {
        Device target_device = offload_to_cpu ? Device(DeviceType::CPU, 0) : p->device;
        m.push_back(Tensor::zeros(p->shape, target_device));
        v.push_back(Tensor::zeros(p->shape, target_device));
    }
}

void AdamW::step_single(size_t i, const std::shared_ptr<Tensor>& p,
                        const std::shared_ptr<Tensor>& g_c,
                        float bias_correction1, float bias_correction2) {
    single_launches().fetch_add(1);
    auto g = p->grad;

    if (offload_to_cpu) {
        auto p_cpu = p->device.type == DeviceType::CPU ? p : p->to(Device(DeviceType::CPU, 0));
        auto g_cpu = g_c->device.type == DeviceType::CPU ? g_c : g_c->to(Device(DeviceType::CPU, 0));

        float* p_ptr = p_cpu->data_ptr();
        float* g_ptr = g_cpu->data_ptr();
        float* m_ptr = m[i]->data_ptr();
        float* v_ptr = v[i]->data_ptr();
        size_t size = p->numel();

        ThreadPool::get().parallel_for(0, size, [&](int64_t j) {
            float grad_val = g_ptr[j];
            m_ptr[j] = beta1 * m_ptr[j] + (1.0f - beta1) * grad_val;
            v_ptr[j] = beta2 * v_ptr[j] + (1.0f - beta2) * grad_val * grad_val;

            float m_hat = m_ptr[j] / bias_correction1;
            float v_hat = v_ptr[j] / bias_correction2;
            float update = m_hat / (std::sqrt(v_hat) + eps);
            if (weight_decay != 0.0f) {
                p_ptr[j] -= lr * (weight_decay * p_ptr[j] + update);
            } else {
                p_ptr[j] -= lr * update;
            }
        });

        if (p->device.type != DeviceType::CPU) {
            p->copy_(p_cpu);
        }
    } else if (p->device.type == DeviceType::GPU) {
        StorageUseGuard guard({p->storage, g_c->storage, m[i]->storage, v[i]->storage});
        auto native = BackendDispatcher::get().get_backend();
        if (native && native->is_available()) {
            native->adamw_step(p->gpu_data(), p->offset, g_c->gpu_data(), g_c->offset,
                               m[i]->gpu_data(), m[i]->offset, v[i]->gpu_data(), v[i]->offset,
                               p->numel(), lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);
        } else {
            cl_mem p_mem = p->gpu_data();
            int p_off = p->offset;
            cl_mem g_mem = g_c->gpu_data();
            int g_off = g_c->offset;
            cl_mem m_mem = m[i]->gpu_data();
            int m_off = m[i]->offset;
            cl_mem v_mem = v[i]->gpu_data();
            int v_off = v[i]->offset;
            int size = p->numel();

            float lr_f = static_cast<float>(lr);
        auto kernel = rt_gpu_get_kernel("litetorch_kernels", litetorch_kernels_src, "adamw_step_kernel");
            if (!kernel) {
                throw std::runtime_error("[litetorch Error] AdamW: adamw_step_kernel not available");
            }
            rt_gpu_launch(kernel, {static_cast<size_t>(size)}, {},
                {&p_mem, &p_off, &g_mem, &g_off, &m_mem, &m_off, &v_mem, &v_off, &beta1, &beta2, &lr_f, &eps, &weight_decay, &bias_correction1, &bias_correction2, &size},
                {sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(cl_mem), sizeof(int), sizeof(float), sizeof(float), sizeof(float), sizeof(float), sizeof(float), sizeof(float), sizeof(float), sizeof(int)});
        }
    } else if (p->device.type == DeviceType::TPU) {
        auto tpu = BackendDispatcher::get().get_tpu_backend();
        if (tpu && tpu->is_available()) {
            tpu->adamw_step(p->gpu_data(), p->offset, g_c->gpu_data(), g_c->offset,
                            m[i]->gpu_data(), m[i]->offset, v[i]->gpu_data(), v[i]->offset,
                            p->numel(), lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);
        } else {
            throw std::runtime_error("[litetorch Error] AdamW: TPU backend not available for TPU param");
        }
    } else {
        float* p_ptr = p->data_ptr();
        float* g_ptr = g_c->data_ptr();
        float* m_ptr = m[i]->data_ptr();
        float* v_ptr = v[i]->data_ptr();
        size_t size = p->numel();

        ThreadPool::get().parallel_for(0, size, [&](int64_t j) {
            float grad_val = g_ptr[j];
            m_ptr[j] = beta1 * m_ptr[j] + (1.0f - beta1) * grad_val;
            v_ptr[j] = beta2 * v_ptr[j] + (1.0f - beta2) * grad_val * grad_val;

            float m_hat = m_ptr[j] / bias_correction1;
            float v_hat = v_ptr[j] / bias_correction2;
            float update = m_hat / (std::sqrt(v_hat) + eps);
            if (weight_decay != 0.0f) {
                p_ptr[j] -= lr * (weight_decay * p_ptr[j] + update);
            } else {
                p_ptr[j] -= lr * update;
            }
        });
    }
}

void AdamW::step() {
    step_count++;
    float bias_correction1 = 1.0f - std::pow(beta1, step_count);
    float bias_correction2 = 1.0f - std::pow(beta2, step_count);

    struct FusedSlot {
        float* p;
        float* g;
        float* m;
        float* v;
        size_t numel;
    };
    std::vector<FusedSlot> fused;
    std::vector<std::shared_ptr<Tensor>> keepalive;
    fused.reserve(params.size());
    keepalive.reserve(params.size());

    for (size_t i = 0; i < params.size(); ++i) {
        auto p = params[i];
        if (!p || !p->grad) continue;

        if (!p->is_contiguous()) {
            throw std::runtime_error("[litetorch Error] Optimizer parameter must be contiguous");
        }
        auto g = p->grad;
        auto g_c = g->is_contiguous() ? g : g->contiguous();

        bool cpu_fusable = !offload_to_cpu
            && p->device.type == DeviceType::CPU
            && g_c->device.type == DeviceType::CPU
            && m[i]->device.type == DeviceType::CPU
            && v[i]->device.type == DeviceType::CPU
            && p->dtype == g_c->dtype;

        if (cpu_fusable) {
            fused.push_back({p->data_ptr(), g_c->data_ptr(),
                             m[i]->data_ptr(), v[i]->data_ptr(),
                             p->numel()});
            keepalive.push_back(g_c);
        } else {
            step_single(i, p, g_c, bias_correction1, bias_correction2);
        }
    }

    if (fused.empty()) return;
    fused_launches().fetch_add(1);

    std::vector<size_t> prefix(fused.size() + 1, 0);
    for (size_t k = 0; k < fused.size(); ++k) {
        prefix[k + 1] = prefix[k] + fused[k].numel;
    }
    size_t total = prefix.back();
    if (total == 0) return;

    float b1 = beta1;
    float b2 = beta2;
    float ep = eps;
    float wd = weight_decay;
    double lrate = lr;

    ThreadPool::get().parallel_for(0, static_cast<int64_t>(total), [&](int64_t j) {
        size_t uj = static_cast<size_t>(j);
        size_t lo = 0, hi = fused.size();
        while (lo + 1 < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (prefix[mid] <= uj) lo = mid;
            else hi = mid;
        }
        const FusedSlot& s = fused[lo];
        size_t local = uj - prefix[lo];

        float grad_val = s.g[local];
        s.m[local] = b1 * s.m[local] + (1.0f - b1) * grad_val;
        s.v[local] = b2 * s.v[local] + (1.0f - b2) * grad_val * grad_val;

        float m_hat = s.m[local] / bias_correction1;
        float v_hat = s.v[local] / bias_correction2;
        float update = m_hat / (std::sqrt(v_hat) + ep);
        if (wd != 0.0f) {
            s.p[local] -= lrate * (wd * s.p[local] + update);
        } else {
            s.p[local] -= lrate * update;
        }
    });
}

}
}
