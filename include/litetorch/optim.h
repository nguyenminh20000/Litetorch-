#ifndef LITETORCH_OPTIM_H
#define LITETORCH_OPTIM_H

#include "litetorch/tensor.h"
#include <vector>
#include <memory>

namespace litetorch {
namespace optim {

class Optimizer {
public:
    std::vector<std::shared_ptr<Tensor>> params;

    Optimizer(const std::vector<std::shared_ptr<Tensor>>& params);
    virtual ~Optimizer() = default;

    virtual void step() = 0;
    void zero_grad();
    virtual float get_lr() const { return 0.0f; }
    virtual void set_lr(float) {}
};

class SGD : public Optimizer {
public:
    float lr;
    float momentum;
    float weight_decay;
    std::vector<std::shared_ptr<Tensor>> velocity;

    SGD(const std::vector<std::shared_ptr<Tensor>>& params, float lr = 1e-2f, float momentum = 0.0f, float weight_decay = 0.0f);
    void step() override;
    float get_lr() const override { return lr; }
    void set_lr(float new_lr) override { lr = new_lr; }
};

class Adam : public Optimizer {
public:
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    int step_count = 0;
    std::vector<std::shared_ptr<Tensor>> m;
    std::vector<std::shared_ptr<Tensor>> v;

    Adam(const std::vector<std::shared_ptr<Tensor>>& params, float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.0f);
    void step() override;
    float get_lr() const override { return lr; }
    void set_lr(float new_lr) override { lr = new_lr; }
};

class AdamW : public Optimizer {
public:
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    bool offload_to_cpu;
    int step_count = 0;
    std::vector<std::shared_ptr<Tensor>> m;
    std::vector<std::shared_ptr<Tensor>> v;

    AdamW(const std::vector<std::shared_ptr<Tensor>>& params, float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.0f, bool offload_to_cpu = false);
    void step() override;
    float get_lr() const override { return lr; }
    void set_lr(float new_lr) override { lr = new_lr; }
};

class AdamW8bit : public Optimizer {
public:
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    int step_count = 0;
    std::vector<std::shared_ptr<Tensor>> m;
    std::vector<std::shared_ptr<Tensor>> v;
    std::vector<float> m_scales;
    std::vector<float> v_scales;

    AdamW8bit(const std::vector<std::shared_ptr<Tensor>>& params, float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.0f);
    void step() override;
    float get_lr() const override { return lr; }
    void set_lr(float new_lr) override { lr = new_lr; }
};

class AdamWFP8 : public Optimizer {
public:
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    int step_count = 0;
    std::vector<std::shared_ptr<Tensor>> m;
    std::vector<std::shared_ptr<Tensor>> v;
    std::vector<float> m_scales;
    std::vector<float> v_scales;
    std::vector<std::vector<float>> m_amax_history;
    std::vector<std::vector<float>> v_amax_history;

    AdamWFP8(const std::vector<std::shared_ptr<Tensor>>& params, float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.0f);
    void step() override;
    float get_lr() const override { return lr; }
    void set_lr(float new_lr) override { lr = new_lr; }
};

class RMSprop : public Optimizer {
public:
    float lr;
    float alpha;
    float eps;
    float weight_decay;
    std::vector<std::shared_ptr<Tensor>> square_avg;

    RMSprop(const std::vector<std::shared_ptr<Tensor>>& params, float lr = 1e-2f, float alpha = 0.99f, float eps = 1e-8f, float weight_decay = 0.0f);
    void step() override;
    float get_lr() const override { return lr; }
    void set_lr(float new_lr) override { lr = new_lr; }
};

class StepLR {
public:
    Optimizer* optimizer;
    int step_size;
    float gamma;
    int last_epoch = 0;

    StepLR(Optimizer* optimizer, int step_size, float gamma = 0.1f);
    void step();
};

class CosineAnnealingLR {
public:
    Optimizer* optimizer;
    int T_max;
    float eta_min;
    float base_lr;
    int last_epoch = 0;

    CosineAnnealingLR(Optimizer* optimizer, int T_max, float eta_min = 0.0f);
    void step();
};

}
}

#endif
