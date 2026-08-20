#include "litetorch/tensor.h"
#include "litetorch/nn.h"
#include "litetorch/ops.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <sys/resource.h>

using namespace litetorch;

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

void run_leak_test(const Device& dev) {
    std::cout << "\n==================================================\n";
    std::cout << "Running memory leak test on " << (dev.type == DeviceType::GPU ? "GPU" : "CPU") << "\n";
    std::cout << "==================================================\n";

    int B = 4;
    int T_tgt = 8;
    int T_mem = 12;
    int C = 16;
    int num_heads = 4;
    int dim_feedforward = 32;

    auto layer = std::make_shared<nn::TransformerDecoderLayer>(C, num_heads, dim_feedforward);
    layer->to(dev);

    std::vector<float> tgt_data(B * T_tgt * C, 0.5f);
    std::vector<float> mem_data(B * T_mem * C, -0.5f);

    size_t start_rss = current_rss_kb();
    std::cout << "Iteration 0   | RAM RSS: " << start_rss / 1024.0 << " MB\n";

    for (int iter = 1; iter <= 1000; ++iter) {
        auto tgt = Tensor::from_vector(tgt_data, {B, T_tgt, C}, dev, true);
        auto memory = Tensor::from_vector(mem_data, {B, T_mem, C}, dev, true);

        auto out = layer->forward(tgt, memory);
        auto loss = Ops::sum(out);
        loss->backward();

        for (auto& p : layer->parameters()) {
            if (p->grad) p->grad = nullptr;
        }
        tgt->grad = nullptr;
        memory->grad = nullptr;

        if (iter % 100 == 0 || iter == 10) {
            std::cout << "Iteration " << iter << " | RAM RSS: " << current_rss_kb() / 1024.0 << " MB\n";
        }
    }

    size_t end_rss = current_rss_kb();
    double diff_mb = (static_cast<double>(end_rss) - static_cast<double>(start_rss)) / 1024.0;
    std::cout << "--------------------------------------------------\n";
    std::cout << "Final diff RAM: " << diff_mb << " MB\n";
    std::cout << "==================================================\n";
}

int main() {
    std::vector<Device> devices = { Device(DeviceType::CPU, 0) };
    if (CLBackend::get().is_available()) {
        devices.push_back(Device(DeviceType::GPU, 0));
    }

    for (const auto& dev : devices) {
        run_leak_test(dev);
    }

    return 0;
}
