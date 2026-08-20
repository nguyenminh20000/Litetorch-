#include "litetorch/tensor.h"
#include "litetorch/ops.h"
#include "litetorch/nn.h"
#include "litetorch/optim.h"
#include "litetorch/data.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <memory>
#include <cstdio>
#include <sys/resource.h>
#include <chrono>
#include "litetorch/backend.h"

static std::string exec_cmd(const std::string& cmd) {
    std::string out;
    char buffer[256];
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return out;
    while (fgets(buffer, sizeof(buffer), pipe)) out += buffer;
    pclose(pipe);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ' || out.back() == '\t')) out.pop_back();
    return out;
}

static size_t current_rss_kb() {
    std::ifstream file("/proc/self/status");
    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::stringstream ss(line.substr(6));
            size_t kb = 0;
            ss >> kb;
            return kb;
        }
    }
    return 0;
}

static size_t peak_rss_kb() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return usage.ru_maxrss;
    }
    return 0;
}

struct GPUInfo {
    bool ok = false;
    double util = 0.0;
    double mem_used = 0.0;
    double mem_total = 0.0;
};

static GPUInfo get_gpu_info() {
    GPUInfo info;
    return info;
}

static double gpu_time_ms() {
    auto backend = litetorch::BackendDispatcher::get().get_backend();
    if (backend && backend->is_available()) {
        backend->finish();
    }
    static auto start_time = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(now - start_time).count();
}

static int argmax_class(const float* p, int classes) {
    int idx = 0;
    for (int i = 1; i < classes; ++i) {
        if (p[i] > p[idx]) idx = i;
    }
    return idx;
}

static std::pair<double, double> eval_model(std::shared_ptr<litetorch::nn::Sequential> model,
                                            const std::shared_ptr<litetorch::Tensor>& x,
                                            const std::shared_ptr<litetorch::Tensor>& y,
                                            int classes) {
    auto out = model->forward(x);
    auto loss = litetorch::Ops::cross_entropy_loss(out, y);

    const float* op = out->data_ptr();
    const float* yp = y->data_ptr();

    int n = static_cast<int>(x->shape[0]);

    int correct = 0;
    for (int i = 0; i < n; ++i) {
        int pred = argmax_class(op + i * classes, classes);
        int truth = static_cast<int>(yp[i]);
        if (pred == truth) ++correct;
    }

    return std::make_pair(loss->item(), 100.0 * correct / n);
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    litetorch::manual_seed(13);

    litetorch::Device dev(litetorch::DeviceType::CPU, 0);
    if (litetorch::CLBackend::get().is_available()) {
        dev = litetorch::Device(litetorch::DeviceType::GPU, 0);
    }

    const int classes = 3;
    const int per_class = 200;
    const int n = classes * per_class;
    const float PI = 3.14159265358979323846f;

    std::vector<float> xv;
    std::vector<float> yv;
    xv.reserve(n * 2);
    yv.reserve(n);

    for (int c = 0; c < classes; ++c) {
        for (int i = 0; i < per_class; ++i) {
            float t = static_cast<float>(i) / static_cast<float>(per_class - 1);
            float angle = t * 4.0f * PI + c * 2.0f * PI / classes;
            float radius = 0.20f + 0.80f * t;
            float noise = 0.03f * std::sin(11.0f * t + c * 1.7f);
            float r = radius + noise;
            float x1 = r * std::cos(angle);
            float x2 = r * std::sin(angle);
            xv.push_back(x1);
            xv.push_back(x2);
            yv.push_back(static_cast<float>(c));
        }
    }

    auto x_cpu = litetorch::Tensor::from_vector(xv, {n, 2}, litetorch::Device(litetorch::DeviceType::CPU, 0));
    auto y_cpu = litetorch::Tensor::from_vector(yv, {n}, litetorch::Device(litetorch::DeviceType::CPU, 0));

    std::cout << "--- DATASET ---\n";
    std::cout << "Samples: " << n << " | Classes: " << classes << "\n";
    std::cout << "First 8 points:\n";
    for (int i = 0; i < 8; ++i) {
        std::cout << "(" << x_cpu->data_ptr()[i * 2] << ", " << x_cpu->data_ptr()[i * 2 + 1] << ") ";
    }
    std::cout << "\n\n";

    auto dataset = std::make_shared<litetorch::data::TensorDataset>(x_cpu, y_cpu);
    size_t prefetch_limit = (dev.type == litetorch::DeviceType::GPU) ? 4 : 2;
    litetorch::data::DataLoader loader(dataset, 64, true, dev, prefetch_limit);

    auto model = std::make_shared<litetorch::nn::Sequential>();
    model->add(std::make_shared<litetorch::nn::Linear>(2, 128));
    model->add(std::make_shared<litetorch::nn::LayerNorm>(std::vector<int64_t>{128}));
    model->add(std::make_shared<litetorch::nn::GELU>());
    model->add(std::make_shared<litetorch::nn::Linear>(128, 128));
    model->add(std::make_shared<litetorch::nn::LayerNorm>(std::vector<int64_t>{128}));
    model->add(std::make_shared<litetorch::nn::GELU>());
    model->add(std::make_shared<litetorch::nn::Linear>(128, 64));
    model->add(std::make_shared<litetorch::nn::GELU>());
    model->add(std::make_shared<litetorch::nn::Linear>(64, classes));

    model->to(dev);
    model->compile();

    auto optimizer = std::make_unique<litetorch::optim::Adam>(model->parameters(), 0.01f);
    litetorch::optim::StepLR scheduler(optimizer.get(), 300, 0.8f);

    auto x_eval = x_cpu->to(dev);
    auto y_eval = y_cpu->to(dev);
    std::pair<double, double> init_stats = eval_model(model, x_eval, y_eval, classes);
    std::cout << "--- INITIAL ---\n";
    std::cout << "Loss: " << init_stats.first << " | Accuracy: " << init_stats.second << "%\n";
    std::cout << "RAM: " << current_rss_kb() / 1024.0 << " MB | Peak RAM: " << peak_rss_kb() / 1024.0 << " MB | GPU Time: " << gpu_time_ms() << " ms";

    GPUInfo gi0 = get_gpu_info();
    if (gi0.ok) {
        std::cout << " | GPU: " << gi0.util << "% | VRAM: " << gi0.mem_used << "/" << gi0.mem_total << " MB";
    } else {
        std::cout << " | GPU: N/A";
    }
    std::cout << "\n\n";

    std::cout << "--- TRAINING ---\n";
    for (int epoch = 1; epoch <= 300; ++epoch) {
        loader.reset();
        std::shared_ptr<litetorch::Tensor> batch_x;
        std::shared_ptr<litetorch::Tensor> batch_y;
        while (loader.next(batch_x, batch_y)) {
            optimizer->zero_grad();
            auto out = model->forward(batch_x);
            auto loss = litetorch::Ops::cross_entropy_loss(out, batch_y);
            loss->backward();
            optimizer->step();
        }
        scheduler.step();

        if (epoch == 1 || epoch % 50 == 0) {
            auto x_eval_cur = x_cpu->to(dev);
            auto y_eval_cur = y_cpu->to(dev);
            std::pair<double, double> cur_stats = eval_model(model, x_eval_cur, y_eval_cur, classes);
            std::cout << "Epoch " << epoch
                      << " | Loss: " << cur_stats.first
                      << " | Acc: " << cur_stats.second << "%"
                      << " | LR: " << optimizer->get_lr()
                      << " | RAM: " << current_rss_kb() / 1024.0 << " MB"
                      << " | Peak RAM: " << peak_rss_kb() / 1024.0 << " MB"
                      << " | GPU Time: " << gpu_time_ms() << " ms";
            GPUInfo gi = get_gpu_info();
            if (gi.ok) {
                std::cout << " | GPU: " << gi.util << "% | VRAM: " << gi.mem_used << "/" << gi.mem_total << " MB";
            } else {
                std::cout << " | GPU: N/A";
            }
            std::cout << "\n";
        }
    }

    auto x_final = x_cpu->to(dev);
    auto y_final = y_cpu->to(dev);
    std::pair<double, double> final_stats = eval_model(model, x_final, y_final, classes);
    auto final_out = model->forward(x_final);

    std::cout << "\n--- FINAL PREDICTIONS ---\n";
    for (int i = 0; i < 12; ++i) {
        const float* p = final_out->data_ptr() + i * classes;
        std::cout << "[" << p[0] << ", " << p[1] << ", " << p[2] << "] ";
    }
    std::cout << "\n\n";
    std::cout << "Final Loss: " << final_stats.first << "\n";
    std::cout << "Final Accuracy: " << final_stats.second << "%\n";
    std::cout << "RAM: " << current_rss_kb() / 1024.0 << " MB | Peak RAM: " << peak_rss_kb() / 1024.0 << " MB | GPU Time: " << gpu_time_ms() << " ms";

    GPUInfo gif = get_gpu_info();
    if (gif.ok) {
        std::cout << " | GPU: " << gif.util << "% | VRAM: " << gif.mem_used << "/" << gif.mem_total << " MB";
    } else {
        std::cout << " | GPU: N/A";
    }
    std::cout << "\n";

    return 0;
}