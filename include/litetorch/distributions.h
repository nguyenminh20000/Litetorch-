#ifndef LITETORCH_DISTRIBUTIONS_H
#define LITETORCH_DISTRIBUTIONS_H

#include "litetorch/tensor.h"
#include <vector>
#include <memory>
#include <string>

namespace litetorch {
namespace distributions {

std::vector<int64_t> broadcast_shapes(const std::vector<std::shared_ptr<Tensor>>& params);
std::shared_ptr<Tensor> broadcast_param(std::shared_ptr<Tensor> p, const std::vector<int64_t>& shape);

class Distribution {
public:
    virtual ~Distribution() = default;
    virtual std::shared_ptr<Tensor> sample(const std::vector<int64_t>& sample_shape = {}) = 0;
    virtual std::shared_ptr<Tensor> log_prob(std::shared_ptr<Tensor> value) = 0;
    virtual std::shared_ptr<Tensor> entropy() = 0;
    virtual std::shared_ptr<Tensor> mean() = 0;
    virtual std::shared_ptr<Tensor> variance() = 0;
    virtual std::vector<int64_t> batch_shape() const = 0;
    virtual std::string name() const = 0;
};

class Normal : public Distribution {
public:
    std::shared_ptr<Tensor> loc, scale;
    Normal(std::shared_ptr<Tensor> loc, std::shared_ptr<Tensor> scale);
    std::shared_ptr<Tensor> sample(const std::vector<int64_t>& sample_shape = {}) override;
    std::shared_ptr<Tensor> log_prob(std::shared_ptr<Tensor> value) override;
    std::shared_ptr<Tensor> entropy() override;
    std::shared_ptr<Tensor> mean() override;
    std::shared_ptr<Tensor> variance() override;
    std::vector<int64_t> batch_shape() const override;
    std::string name() const override { return "Normal"; }
};

class Uniform : public Distribution {
public:
    std::shared_ptr<Tensor> low, high;
    Uniform(std::shared_ptr<Tensor> low, std::shared_ptr<Tensor> high);
    std::shared_ptr<Tensor> sample(const std::vector<int64_t>& sample_shape = {}) override;
    std::shared_ptr<Tensor> log_prob(std::shared_ptr<Tensor> value) override;
    std::shared_ptr<Tensor> entropy() override;
    std::shared_ptr<Tensor> mean() override;
    std::shared_ptr<Tensor> variance() override;
    std::vector<int64_t> batch_shape() const override;
    std::string name() const override { return "Uniform"; }
};

class Exponential : public Distribution {
public:
    std::shared_ptr<Tensor> rate;
    Exponential(std::shared_ptr<Tensor> rate);
    std::shared_ptr<Tensor> sample(const std::vector<int64_t>& sample_shape = {}) override;
    std::shared_ptr<Tensor> log_prob(std::shared_ptr<Tensor> value) override;
    std::shared_ptr<Tensor> entropy() override;
    std::shared_ptr<Tensor> mean() override;
    std::shared_ptr<Tensor> variance() override;
    std::vector<int64_t> batch_shape() const override;
    std::string name() const override { return "Exponential"; }
};

class Laplace : public Distribution {
public:
    std::shared_ptr<Tensor> loc, scale;
    Laplace(std::shared_ptr<Tensor> loc, std::shared_ptr<Tensor> scale);
    std::shared_ptr<Tensor> sample(const std::vector<int64_t>& sample_shape = {}) override;
    std::shared_ptr<Tensor> log_prob(std::shared_ptr<Tensor> value) override;
    std::shared_ptr<Tensor> entropy() override;
    std::shared_ptr<Tensor> mean() override;
    std::shared_ptr<Tensor> variance() override;
    std::vector<int64_t> batch_shape() const override;
    std::string name() const override { return "Laplace"; }
};

class Cauchy : public Distribution {
public:
    std::shared_ptr<Tensor> loc, scale;
    Cauchy(std::shared_ptr<Tensor> loc, std::shared_ptr<Tensor> scale);
    std::shared_ptr<Tensor> sample(const std::vector<int64_t>& sample_shape = {}) override;
    std::shared_ptr<Tensor> log_prob(std::shared_ptr<Tensor> value) override;
    std::shared_ptr<Tensor> entropy() override;
    std::shared_ptr<Tensor> mean() override;
    std::shared_ptr<Tensor> variance() override;
    std::vector<int64_t> batch_shape() const override;
    std::string name() const override { return "Cauchy"; }
};

class Beta : public Distribution {
public:
    std::shared_ptr<Tensor> alpha, beta;
    Beta(std::shared_ptr<Tensor> alpha, std::shared_ptr<Tensor> beta);
    std::shared_ptr<Tensor> sample(const std::vector<int64_t>& sample_shape = {}) override;
    std::shared_ptr<Tensor> log_prob(std::shared_ptr<Tensor> value) override;
    std::shared_ptr<Tensor> entropy() override;
    std::shared_ptr<Tensor> mean() override;
    std::shared_ptr<Tensor> variance() override;
    std::vector<int64_t> batch_shape() const override;
    std::string name() const override { return "Beta"; }
};

class Gamma : public Distribution {
public:
    std::shared_ptr<Tensor> alpha, beta;
    Gamma(std::shared_ptr<Tensor> alpha, std::shared_ptr<Tensor> beta);
    std::shared_ptr<Tensor> sample(const std::vector<int64_t>& sample_shape = {}) override;
    std::shared_ptr<Tensor> log_prob(std::shared_ptr<Tensor> value) override;
    std::shared_ptr<Tensor> entropy() override;
    std::shared_ptr<Tensor> mean() override;
    std::shared_ptr<Tensor> variance() override;
    std::vector<int64_t> batch_shape() const override;
    std::string name() const override { return "Gamma"; }
};

class Dirichlet : public Distribution {
public:
    std::shared_ptr<Tensor> concentration;
    Dirichlet(std::shared_ptr<Tensor> concentration);
    std::shared_ptr<Tensor> sample(const std::vector<int64_t>& sample_shape = {}) override;
    std::shared_ptr<Tensor> log_prob(std::shared_ptr<Tensor> value) override;
    std::shared_ptr<Tensor> entropy() override;
    std::shared_ptr<Tensor> mean() override;
    std::shared_ptr<Tensor> variance() override;
    std::vector<int64_t> batch_shape() const override;
    std::string name() const override { return "Dirichlet"; }
};

class Bernoulli : public Distribution {
public:
    std::shared_ptr<Tensor> probs;
    Bernoulli(std::shared_ptr<Tensor> probs, bool from_logits = false);
    std::shared_ptr<Tensor> sample(const std::vector<int64_t>& sample_shape = {}) override;
    std::shared_ptr<Tensor> log_prob(std::shared_ptr<Tensor> value) override;
    std::shared_ptr<Tensor> entropy() override;
    std::shared_ptr<Tensor> mean() override;
    std::shared_ptr<Tensor> variance() override;
    std::vector<int64_t> batch_shape() const override;
    std::string name() const override { return "Bernoulli"; }
};

class Categorical : public Distribution {
public:
    std::shared_ptr<Tensor> probs;
    Categorical(std::shared_ptr<Tensor> probs, bool from_logits = false);
    std::shared_ptr<Tensor> sample(const std::vector<int64_t>& sample_shape = {}) override;
    std::shared_ptr<Tensor> log_prob(std::shared_ptr<Tensor> value) override;
    std::shared_ptr<Tensor> entropy() override;
    std::shared_ptr<Tensor> mean() override;
    std::shared_ptr<Tensor> variance() override;
    std::vector<int64_t> batch_shape() const override;
    std::string name() const override { return "Categorical"; }
};

class Poisson : public Distribution {
public:
    std::shared_ptr<Tensor> rate;
    Poisson(std::shared_ptr<Tensor> rate);
    std::shared_ptr<Tensor> sample(const std::vector<int64_t>& sample_shape = {}) override;
    std::shared_ptr<Tensor> log_prob(std::shared_ptr<Tensor> value) override;
    std::shared_ptr<Tensor> entropy() override;
    std::shared_ptr<Tensor> mean() override;
    std::shared_ptr<Tensor> variance() override;
    std::vector<int64_t> batch_shape() const override;
    std::string name() const override { return "Poisson"; }
};

class Geometric : public Distribution {
public:
    std::shared_ptr<Tensor> probs;
    Geometric(std::shared_ptr<Tensor> probs);
    std::shared_ptr<Tensor> sample(const std::vector<int64_t>& sample_shape = {}) override;
    std::shared_ptr<Tensor> log_prob(std::shared_ptr<Tensor> value) override;
    std::shared_ptr<Tensor> entropy() override;
    std::shared_ptr<Tensor> mean() override;
    std::shared_ptr<Tensor> variance() override;
    std::vector<int64_t> batch_shape() const override;
    std::string name() const override { return "Geometric"; }
};

class Binomial : public Distribution {
public:
    std::shared_ptr<Tensor> total_count, probs;
    Binomial(std::shared_ptr<Tensor> total_count, std::shared_ptr<Tensor> probs);
    std::shared_ptr<Tensor> sample(const std::vector<int64_t>& sample_shape = {}) override;
    std::shared_ptr<Tensor> log_prob(std::shared_ptr<Tensor> value) override;
    std::shared_ptr<Tensor> entropy() override;
    std::shared_ptr<Tensor> mean() override;
    std::shared_ptr<Tensor> variance() override;
    std::vector<int64_t> batch_shape() const override;
    std::string name() const override { return "Binomial"; }
};

class NegativeBinomial : public Distribution {
public:
    std::shared_ptr<Tensor> total_count, probs;
    NegativeBinomial(std::shared_ptr<Tensor> total_count, std::shared_ptr<Tensor> probs);
    std::shared_ptr<Tensor> sample(const std::vector<int64_t>& sample_shape = {}) override;
    std::shared_ptr<Tensor> log_prob(std::shared_ptr<Tensor> value) override;
    std::shared_ptr<Tensor> entropy() override;
    std::shared_ptr<Tensor> mean() override;
    std::shared_ptr<Tensor> variance() override;
    std::vector<int64_t> batch_shape() const override;
    std::string name() const override { return "NegativeBinomial"; }
};

class StudentT : public Distribution {
public:
    std::shared_ptr<Tensor> df;
    StudentT(std::shared_ptr<Tensor> df);
    std::shared_ptr<Tensor> sample(const std::vector<int64_t>& sample_shape = {}) override;
    std::shared_ptr<Tensor> log_prob(std::shared_ptr<Tensor> value) override;
    std::shared_ptr<Tensor> entropy() override;
    std::shared_ptr<Tensor> mean() override;
    std::shared_ptr<Tensor> variance() override;
    std::vector<int64_t> batch_shape() const override;
    std::string name() const override { return "StudentT"; }
};

class Chi2 : public Distribution {
public:
    std::shared_ptr<Tensor> df;
    Chi2(std::shared_ptr<Tensor> df);
    std::shared_ptr<Tensor> sample(const std::vector<int64_t>& sample_shape = {}) override;
    std::shared_ptr<Tensor> log_prob(std::shared_ptr<Tensor> value) override;
    std::shared_ptr<Tensor> entropy() override;
    std::shared_ptr<Tensor> mean() override;
    std::shared_ptr<Tensor> variance() override;
    std::vector<int64_t> batch_shape() const override;
    std::string name() const override { return "Chi2"; }
};

class LogNormal : public Distribution {
public:
    std::shared_ptr<Tensor> loc, scale;
    LogNormal(std::shared_ptr<Tensor> loc, std::shared_ptr<Tensor> scale);
    std::shared_ptr<Tensor> sample(const std::vector<int64_t>& sample_shape = {}) override;
    std::shared_ptr<Tensor> log_prob(std::shared_ptr<Tensor> value) override;
    std::shared_ptr<Tensor> entropy() override;
    std::shared_ptr<Tensor> mean() override;
    std::shared_ptr<Tensor> variance() override;
    std::vector<int64_t> batch_shape() const override;
    std::string name() const override { return "LogNormal"; }
};

class Gumbel : public Distribution {
public:
    std::shared_ptr<Tensor> loc, scale;
    Gumbel(std::shared_ptr<Tensor> loc, std::shared_ptr<Tensor> scale);
    std::shared_ptr<Tensor> sample(const std::vector<int64_t>& sample_shape = {}) override;
    std::shared_ptr<Tensor> log_prob(std::shared_ptr<Tensor> value) override;
    std::shared_ptr<Tensor> entropy() override;
    std::shared_ptr<Tensor> mean() override;
    std::shared_ptr<Tensor> variance() override;
    std::vector<int64_t> batch_shape() const override;
    std::string name() const override { return "Gumbel"; }
};

class Weibull : public Distribution {
public:
    std::shared_ptr<Tensor> scale, concentration;
    Weibull(std::shared_ptr<Tensor> scale, std::shared_ptr<Tensor> concentration);
    std::shared_ptr<Tensor> sample(const std::vector<int64_t>& sample_shape = {}) override;
    std::shared_ptr<Tensor> log_prob(std::shared_ptr<Tensor> value) override;
    std::shared_ptr<Tensor> entropy() override;
    std::shared_ptr<Tensor> mean() override;
    std::shared_ptr<Tensor> variance() override;
    std::vector<int64_t> batch_shape() const override;
    std::string name() const override { return "Weibull"; }
};

class MultivariateNormal : public Distribution {
public:
    std::shared_ptr<Tensor> loc, covariance_matrix;
    MultivariateNormal(std::shared_ptr<Tensor> loc, std::shared_ptr<Tensor> covariance_matrix);
    std::shared_ptr<Tensor> sample(const std::vector<int64_t>& sample_shape = {}) override;
    std::shared_ptr<Tensor> log_prob(std::shared_ptr<Tensor> value) override;
    std::shared_ptr<Tensor> entropy() override;
    std::shared_ptr<Tensor> mean() override;
    std::shared_ptr<Tensor> variance() override;
    std::vector<int64_t> batch_shape() const override;
    std::string name() const override { return "MultivariateNormal"; }
};

std::shared_ptr<Tensor> kl_divergence(std::shared_ptr<Distribution> p, std::shared_ptr<Distribution> q);

}
}

#endif
