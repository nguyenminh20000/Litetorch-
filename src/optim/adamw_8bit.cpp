#include "litetorch/optim.h"
#include "litetorch/thread_pool.h"
#include "litetorch/cl_backend.h"
#include "litetorch/backend.h"
#include "optim_utils.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace litetorch {
namespace optim {

namespace {
inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            f = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f, 4);
    return result;
}

inline uint16_t fp32_to_fp16(float x) {
    uint32_t f;
    std::memcpy(&f, &x, 4);
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t exp = (int32_t)((f >> 23) & 0xFF) - 112;
    uint32_t mant = f & 0x7FFFFF;
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7BFF);
    } else if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000;
        uint32_t t = mant >> (14 - exp);
        if ((mant >> (13 - exp)) & 1) t++;
        return (uint16_t)(sign | t);
    }
    uint32_t h = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000) h++;
    return (uint16_t)h;
}

inline float read_param(const void* ptr, DataType dtype, size_t idx) {
    if (dtype == DataType::FP16) {
        return fp16_to_fp32(static_cast<const uint16_t*>(ptr)[idx]);
    }
    return static_cast<const float*>(ptr)[idx];
}

inline void write_param(void* ptr, DataType dtype, size_t idx, float val) {
    if (dtype == DataType::FP16) {
        static_cast<uint16_t*>(ptr)[idx] = fp32_to_fp16(val);
    } else {
        static_cast<float*>(ptr)[idx] = val;
    }
}
}

AdamW8bit::AdamW8bit(const std::vector<std::shared_ptr<Tensor>>& params, double lr, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(params), lr(lr), beta1(beta1), beta2(beta2), eps(eps), weight_decay(weight_decay) {
    for (auto& p : params) {
        Device dev = p->device.type == DeviceType::CPU ? Device(DeviceType::CPU, 0) : p->device;
        auto m_tensor = Tensor::create(p->shape, dev, false, DataType::INT8);
        auto v_tensor = Tensor::create(p->shape, dev, false, DataType::INT8);
        size_t n = p->numel();
        if (dev.type == DeviceType::CPU) {
            std::memset(m_tensor->storage->get_cpu_ptr(), 0, n);
            std::memset(v_tensor->storage->get_cpu_ptr(), 0, n);
        } else {
            auto z = Tensor::create(p->shape, Device(DeviceType::CPU, 0), false, DataType::INT8);
            std::memset(z->storage->get_cpu_ptr(), 0, n);
            m_tensor->copy_(z);
            v_tensor->copy_(z);
        }
        m.push_back(m_tensor);
        v.push_back(v_tensor);
        size_t nblocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        m_scales.emplace_back(nblocks, 1.0f);
        v_scales.emplace_back(nblocks, 1.0f);
    }
}

void AdamW8bit::step() {
    step_count++;
    float bias_correction1 = 1.0f - std::pow(beta1, step_count);
    float bias_correction2 = 1.0f - std::pow(beta2, step_count);

    for (size_t i = 0; i < params.size(); ++i) {
        auto p = params[i];
        if (!p || !p->grad) continue;
        auto g = p->grad;

        if (!p->is_contiguous()) {
            throw std::runtime_error("[litetorch Error] Optimizer parameter must be contiguous");
        }
        auto g_c = g->is_contiguous() ? g : g->contiguous();
        if (p->dtype != DataType::FP32 && p->dtype != DataType::FP16) {
            throw std::runtime_error("[litetorch Error] AdamW8bit only supports FP32 and FP16 params");
        }
        if (g_c->dtype != DataType::FP32 && g_c->dtype != DataType::FP16) {
            throw std::runtime_error("[litetorch Error] AdamW8bit only supports FP32 and FP16 grads");
        }

        if (p->device.type == DeviceType::GPU) {
            auto native = BackendDispatcher::get().get_backend();
            if (native && native->is_available()) {
                size_t nblocks = m_scales[i].size();
                auto m_scales_t = Tensor::from_vector(m_scales[i], {(int64_t)nblocks}, Device(DeviceType::CPU, 0), false);
                auto v_scales_t = Tensor::from_vector(v_scales[i], {(int64_t)nblocks}, Device(DeviceType::CPU, 0), false);
                auto m_scales_gpu = m_scales_t->to(p->device);
                auto v_scales_gpu = v_scales_t->to(p->device);
                native->adamw_8bit_step(
                    p->gpu_data(), p->offset, g_c->gpu_data(), g_c->offset,
                    m[i]->gpu_data(), m[i]->offset, v[i]->gpu_data(), v[i]->offset,
                    m_scales_gpu->gpu_data(), v_scales_gpu->gpu_data(),
                    p->numel(), (int)(p->dtype == DataType::FP16),
                    lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);
                auto m_scales_back = m_scales_gpu->to(Device(DeviceType::CPU, 0));
                auto v_scales_back = v_scales_gpu->to(Device(DeviceType::CPU, 0));
                auto m_sv = m_scales_back->to_vector();
                auto v_sv = v_scales_back->to_vector();
                for (size_t b = 0; b < nblocks; ++b) {
                    m_scales[i][b] = m_sv[b];
                    v_scales[i][b] = v_sv[b];
                }
                continue;
            }
        }

        auto p_cpu = p->device.type == DeviceType::CPU ? p : p->to(Device(DeviceType::CPU, 0));
        auto g_cpu = g_c->device.type == DeviceType::CPU ? g_c : g_c->to(Device(DeviceType::CPU, 0));
        auto m_cpu = m[i]->device.type == DeviceType::CPU ? m[i] : m[i]->to(Device(DeviceType::CPU, 0));
        auto v_cpu = v[i]->device.type == DeviceType::CPU ? v[i] : v[i]->to(Device(DeviceType::CPU, 0));

        size_t p_elem = p_cpu->storage->element_size();
        size_t g_elem = g_cpu->storage->element_size();
        void* p_ptr = (char*)p_cpu->storage->get_cpu_ptr() + p_cpu->offset * p_elem;
        void* g_ptr = (char*)g_cpu->storage->get_cpu_ptr() + g_cpu->offset * g_elem;
        int8_t* m_ptr = reinterpret_cast<int8_t*>((char*)m_cpu->storage->get_cpu_ptr() + m_cpu->offset);
        uint8_t* v_ptr = reinterpret_cast<uint8_t*>((char*)v_cpu->storage->get_cpu_ptr() + v_cpu->offset);
        DataType dtype = p->dtype;
        DataType g_dtype = g_cpu->dtype;
        size_t size = p->numel();
        size_t nblocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

        ThreadPool::get().parallel_for(0, (int64_t)nblocks, [&](int64_t bb) {
            size_t b = (size_t)bb;
            size_t start = b * BLOCK_SIZE;
            size_t end = std::min(start + BLOCK_SIZE, size);
            size_t len = end - start;
            float m_scale = m_scales[i][b];
            float v_scale = v_scales[i][b];
            float tmp_m[BLOCK_SIZE];
            float tmp_v[BLOCK_SIZE];
            float max_m = 0.0f, max_v = 0.0f;
            for (size_t k = 0; k < len; ++k) {
                size_t j = start + k;
                float grad_val = read_param(g_ptr, g_dtype, j);
                float m_val = static_cast<float>(m_ptr[j]) * m_scale;
                float v_val = static_cast<float>(v_ptr[j]) * v_scale;
                m_val = beta1 * m_val + (1.0f - beta1) * grad_val;
                v_val = beta2 * v_val + (1.0f - beta2) * grad_val * grad_val;
                tmp_m[k] = m_val;
                tmp_v[k] = v_val;
                float a = std::abs(m_val);
                if (a > max_m) max_m = a;
                if (v_val > max_v) max_v = v_val;
                float m_hat = m_val / bias_correction1;
                float v_hat = v_val / bias_correction2;
                float update = m_hat / (std::sqrt(v_hat) + eps);
                float p_val = read_param(p_ptr, dtype, j);
                float new_p = weight_decay != 0.0f ? p_val - lr * (weight_decay * p_val + update) : p_val - lr * update;
                write_param(p_ptr, dtype, j, new_p);
            }
            float new_m_scale = max_m > 0.0f ? max_m / 127.0f : m_scale;
            float new_v_scale = max_v > 0.0f ? max_v / 255.0f : v_scale;
            m_scales[i][b] = new_m_scale;
            v_scales[i][b] = new_v_scale;
            for (size_t k = 0; k < len; ++k) {
                size_t j = start + k;
                m_ptr[j] = static_cast<int8_t>(std::round(tmp_m[k] / new_m_scale));
                v_ptr[j] = static_cast<uint8_t>(std::round(tmp_v[k] / new_v_scale));
            }
        });

        if (m[i]->device.type != DeviceType::CPU) m[i]->copy_(m_cpu);
        if (v[i]->device.type != DeviceType::CPU) v[i]->copy_(v_cpu);
        if (p->device.type != DeviceType::CPU) p->copy_(p_cpu);
    }
}

}
}
