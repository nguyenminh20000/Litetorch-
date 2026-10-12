#ifndef LITETORCH_OPTIM_H
#define LITETORCH_OPTIM_H

#include "litetorch/tensor.h"
#include <vector>
#include <memory>
#include <map>
#include <string>

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
    float dampening;
    bool nesterov;
    float weight_decay;
    std::vector<std::shared_ptr<Tensor>> velocity;

    SGD(const std::vector<std::shared_ptr<Tensor>>& params, float lr = 1e-2f, float momentum = 0.0f,
        float dampening = 0.0f, float weight_decay = 0.0f, bool nesterov = false);
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
    static constexpr size_t BLOCK_SIZE = 2048;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    int step_count = 0;
    std::vector<std::shared_ptr<Tensor>> m;
    std::vector<std::shared_ptr<Tensor>> v;
    std::vector<std::vector<float>> m_scales;
    std::vector<std::vector<float>> v_scales;

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

class LinearWarmup {
public:
    Optimizer* optimizer;
    int warmup_steps;
    int total_steps;
    float base_lr;
    float min_lr;
    int last_step = 0;

    LinearWarmup(Optimizer* optimizer, int warmup_steps, int total_steps, float min_lr = 0.0f);
    void step();
    float get_lr() const;
};

struct ParamGroup {
    std::vector<std::shared_ptr<Tensor>> params;
    float lr;
    float weight_decay;
};

class Adagrad : public Optimizer {
public:
    float lr;
    float lr_decay;
    float weight_decay;
    float eps;
    int step_count = 0;
    std::vector<std::shared_ptr<Tensor>> sum;
    std::vector<ParamGroup> param_groups;

    Adagrad(const std::vector<std::shared_ptr<Tensor>>& params, float lr = 1e-2f, float lr_decay = 0.0f,
            float weight_decay = 0.0f, float eps = 1e-10f);
    void step() override;
    float get_lr() const override;
    void set_lr(float new_lr) override;
    void add_param_group(const std::vector<std::shared_ptr<Tensor>>& params, float group_lr, float group_weight_decay);
};

class NAdam : public Optimizer {
public:
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float momentum_decay;
    int step_count = 0;
    std::vector<std::shared_ptr<Tensor>> exp_avg;
    std::vector<std::shared_ptr<Tensor>> exp_avg_sq;
    std::vector<ParamGroup> param_groups;

    NAdam(const std::vector<std::shared_ptr<Tensor>>& params, float lr = 2e-3f,
          std::pair<float, float> betas = {0.9f, 0.999f}, float eps = 1e-8f,
          float weight_decay = 0.0f, float momentum_decay = 4e-3f);
    void step() override;
    float get_lr() const override;
    void set_lr(float new_lr) override;
    void add_param_group(const std::vector<std::shared_ptr<Tensor>>& params, float group_lr, float group_weight_decay);
};

class MultiStepLR {
public:
    Optimizer* optimizer;
    std::vector<int> milestones;
    float gamma;
    float base_lr;
    int last_epoch = 0;

    MultiStepLR(Optimizer* optimizer, const std::vector<int>& milestones, float gamma = 0.1f);
    void step();
    float get_last_lr() const;
    std::map<std::string, double> state_dict() const;
    void load_state_dict(const std::map<std::string, double>& state);
};

class ExponentialLR {
public:
    Optimizer* optimizer;
    float gamma;
    float base_lr;
    int last_epoch = 0;

    ExponentialLR(Optimizer* optimizer, float gamma);
    void step();
    float get_last_lr() const;
    std::map<std::string, double> state_dict() const;
    void load_state_dict(const std::map<std::string, double>& state);
};

class ReduceLROnPlateau {
public:
    Optimizer* optimizer;
    std::string mode;
    float factor;
    int patience;
    float threshold;
    std::string threshold_mode;
    int cooldown;
    float min_lr;
    float eps;
    float base_lr;
    float best;
    int cooldown_counter = 0;
    int num_bad_epochs = 0;
    int last_epoch = 0;

    ReduceLROnPlateau(Optimizer* optimizer, const std::string& mode = "min", float factor = 0.1f,
                      int patience = 10, float threshold = 1e-4f, const std::string& threshold_mode = "rel",
                      int cooldown = 0, float min_lr = 0.0f, float eps = 1e-8f);
    void step(float metric);
    float get_last_lr() const;
    bool is_better(float a, float best_val) const;
    std::map<std::string, double> state_dict() const;
    void load_state_dict(const std::map<std::string, double>& state);
};

}
}

#endif
