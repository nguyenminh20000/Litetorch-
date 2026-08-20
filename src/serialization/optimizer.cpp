#include "litetorch/serialization.h"
#include "litetorch/optim.h"
#include <fstream>
#include <stdexcept>
#include <vector>

namespace litetorch {

namespace {
uint32_t calculate_crc32(const void* data, size_t size, uint32_t initial_crc = 0xFFFFFFFF) {
    static uint32_t table[256];
    static bool table_initialized = false;
    if (!table_initialized) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t crc = i;
            for (uint32_t j = 0; j < 8; ++j) {
                if (crc & 1) {
                    crc = (crc >> 1) ^ 0xEDB88320;
                } else {
                    crc >>= 1;
                }
            }
            table[i] = crc;
        }
        table_initialized = true;
    }

    uint32_t crc = initial_crc;
    const uint8_t* byte_ptr = reinterpret_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        crc = (crc >> 8) ^ table[(crc ^ byte_ptr[i]) & 0xFF];
    }
    return crc;
}
}

void save_optimizer_state(optim::Optimizer* opt, const std::string& filepath) {
    if (!opt) {
        throw std::runtime_error("[litetorch Error] Cannot save null optimizer state");
    }
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("[litetorch Error] Failed to open file for saving optimizer state: " + filepath);
    }

    uint32_t running_crc = 0xFFFFFFFF;

    if (auto adam = dynamic_cast<optim::Adam*>(opt)) {
        const char magic[4] = {'L', 'T', 'A', 'D'};
        out.write(magic, 4);
        running_crc = calculate_crc32(magic, 4, running_crc);
        
        int32_t step_count = adam->step_count;
        out.write(reinterpret_cast<const char*>(&step_count), sizeof(step_count));
        running_crc = calculate_crc32(&step_count, sizeof(step_count), running_crc);

        uint32_t num_tensors = static_cast<uint32_t>(adam->m.size());
        out.write(reinterpret_cast<const char*>(&num_tensors), sizeof(num_tensors));
        running_crc = calculate_crc32(&num_tensors, sizeof(num_tensors), running_crc);

        for (const auto& param : adam->m) {
            uint32_t ndims = static_cast<uint32_t>(param->shape.size());
            out.write(reinterpret_cast<const char*>(&ndims), sizeof(ndims));
            running_crc = calculate_crc32(&ndims, sizeof(ndims), running_crc);
            for (int64_t dim : param->shape) {
                out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
                running_crc = calculate_crc32(&dim, sizeof(dim), running_crc);
            }
            uint64_t numel = static_cast<uint64_t>(param->numel());
            out.write(reinterpret_cast<const char*>(&numel), sizeof(numel));
            running_crc = calculate_crc32(&numel, sizeof(numel), running_crc);
            std::vector<float> data = param->to_vector();
            out.write(reinterpret_cast<const char*>(data.data()), numel * sizeof(float));
            running_crc = calculate_crc32(data.data(), numel * sizeof(float), running_crc);
        }

        for (const auto& param : adam->v) {
            uint32_t ndims = static_cast<uint32_t>(param->shape.size());
            out.write(reinterpret_cast<const char*>(&ndims), sizeof(ndims));
            running_crc = calculate_crc32(&ndims, sizeof(ndims), running_crc);
            for (int64_t dim : param->shape) {
                out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
                running_crc = calculate_crc32(&dim, sizeof(dim), running_crc);
            }
            uint64_t numel = static_cast<uint64_t>(param->numel());
            out.write(reinterpret_cast<const char*>(&numel), sizeof(numel));
            running_crc = calculate_crc32(&numel, sizeof(numel), running_crc);
            std::vector<float> data = param->to_vector();
            out.write(reinterpret_cast<const char*>(data.data()), numel * sizeof(float));
            running_crc = calculate_crc32(data.data(), numel * sizeof(float), running_crc);
        }
    } else if (auto sgd = dynamic_cast<optim::SGD*>(opt)) {
        const char magic[4] = {'L', 'T', 'S', 'G'};
        out.write(magic, 4);
        running_crc = calculate_crc32(magic, 4, running_crc);

        uint32_t num_tensors = static_cast<uint32_t>(sgd->velocity.size());
        out.write(reinterpret_cast<const char*>(&num_tensors), sizeof(num_tensors));
        running_crc = calculate_crc32(&num_tensors, sizeof(num_tensors), running_crc);

        for (const auto& param : sgd->velocity) {
            uint32_t ndims = static_cast<uint32_t>(param->shape.size());
            out.write(reinterpret_cast<const char*>(&ndims), sizeof(ndims));
            running_crc = calculate_crc32(&ndims, sizeof(ndims), running_crc);
            for (int64_t dim : param->shape) {
                out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
                running_crc = calculate_crc32(&dim, sizeof(dim), running_crc);
            }
            uint64_t numel = static_cast<uint64_t>(param->numel());
            out.write(reinterpret_cast<const char*>(&numel), sizeof(numel));
            running_crc = calculate_crc32(&numel, sizeof(numel), running_crc);
            std::vector<float> data = param->to_vector();
            out.write(reinterpret_cast<const char*>(data.data()), numel * sizeof(float));
            running_crc = calculate_crc32(data.data(), numel * sizeof(float), running_crc);
        }
    } else if (auto rmsprop = dynamic_cast<optim::RMSprop*>(opt)) {
        const char magic[4] = {'L', 'T', 'R', 'P'};
        out.write(magic, 4);
        running_crc = calculate_crc32(magic, 4, running_crc);

        uint32_t num_tensors = static_cast<uint32_t>(rmsprop->square_avg.size());
        out.write(reinterpret_cast<const char*>(&num_tensors), sizeof(num_tensors));
        running_crc = calculate_crc32(&num_tensors, sizeof(num_tensors), running_crc);

        for (const auto& param : rmsprop->square_avg) {
            uint32_t ndims = static_cast<uint32_t>(param->shape.size());
            out.write(reinterpret_cast<const char*>(&ndims), sizeof(ndims));
            running_crc = calculate_crc32(&ndims, sizeof(ndims), running_crc);
            for (int64_t dim : param->shape) {
                out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
                running_crc = calculate_crc32(&dim, sizeof(dim), running_crc);
            }
            uint64_t numel = static_cast<uint64_t>(param->numel());
            out.write(reinterpret_cast<const char*>(&numel), sizeof(numel));
            running_crc = calculate_crc32(&numel, sizeof(numel), running_crc);
            std::vector<float> data = param->to_vector();
            out.write(reinterpret_cast<const char*>(data.data()), numel * sizeof(float));
            running_crc = calculate_crc32(data.data(), numel * sizeof(float), running_crc);
        }
    } else {
        throw std::runtime_error("[litetorch Error] Unsupported optimizer type for saving state");
    }

    uint32_t final_crc = running_crc ^ 0xFFFFFFFF;
    out.write(reinterpret_cast<const char*>(&final_crc), sizeof(final_crc));
    out.flush();
    if (out.fail()) {
        throw std::runtime_error("[litetorch Error] Write failure or disk full during saving optimizer state");
    }
    out.close();
}

void load_optimizer_state(optim::Optimizer* opt, const std::string& filepath) {
    if (!opt) {
        throw std::runtime_error("[litetorch Error] Cannot load null optimizer state");
    }
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("[litetorch Error] Failed to open file for loading optimizer state: " + filepath);
    }

    uint32_t running_crc = 0xFFFFFFFF;

    char magic[4];
    in.read(magic, 4);
    running_crc = calculate_crc32(magic, 4, running_crc);

    if (magic[0] == 'L' && magic[1] == 'T' && magic[2] == 'A' && magic[3] == 'D') {
        auto adam = dynamic_cast<optim::Adam*>(opt);
        if (!adam) {
            throw std::runtime_error("[litetorch Error] Loaded Adam state but optimizer is not Adam");
        }
        
        int32_t step_count;
        in.read(reinterpret_cast<char*>(&step_count), sizeof(step_count));
        running_crc = calculate_crc32(&step_count, sizeof(step_count), running_crc);

        uint32_t num_tensors;
        in.read(reinterpret_cast<char*>(&num_tensors), sizeof(num_tensors));
        running_crc = calculate_crc32(&num_tensors, sizeof(num_tensors), running_crc);
        if (num_tensors != adam->m.size()) {
            throw std::runtime_error("[litetorch Error] Adam state size mismatch");
        }

        std::vector<std::vector<float>> loaded_m(num_tensors);
        for (size_t i = 0; i < adam->m.size(); ++i) {
            auto param = adam->m[i];
            uint32_t ndims;
            in.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));
            running_crc = calculate_crc32(&ndims, sizeof(ndims), running_crc);
            if (ndims != param->shape.size()) {
                throw std::runtime_error("[litetorch Error] Adam state ndims mismatch");
            }
            std::vector<int64_t> shape(ndims);
            for (uint32_t d = 0; d < ndims; ++d) {
                in.read(reinterpret_cast<char*>(&shape[d]), sizeof(int64_t));
                running_crc = calculate_crc32(&shape[d], sizeof(int64_t), running_crc);
                if (shape[d] != param->shape[d]) {
                    throw std::runtime_error("[litetorch Error] Adam state shape mismatch");
                }
            }
            uint64_t numel;
            in.read(reinterpret_cast<char*>(&numel), sizeof(numel));
            running_crc = calculate_crc32(&numel, sizeof(numel), running_crc);
            if (numel != param->numel()) {
                throw std::runtime_error("[litetorch Error] Adam state numel mismatch");
            }
            std::vector<float> cpu_data(numel);
            in.read(reinterpret_cast<char*>(cpu_data.data()), numel * sizeof(float));
            running_crc = calculate_crc32(cpu_data.data(), numel * sizeof(float), running_crc);
            if (!in) {
                throw std::runtime_error("[litetorch Error] Unexpected end of file or read error in Adam state file");
            }
            loaded_m[i] = std::move(cpu_data);
        }

        std::vector<std::vector<float>> loaded_v(num_tensors);
        for (size_t i = 0; i < adam->v.size(); ++i) {
            auto param = adam->v[i];
            uint32_t ndims;
            in.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));
            running_crc = calculate_crc32(&ndims, sizeof(ndims), running_crc);
            if (ndims != param->shape.size()) {
                throw std::runtime_error("[litetorch Error] Adam state ndims mismatch");
            }
            std::vector<int64_t> shape(ndims);
            for (uint32_t d = 0; d < ndims; ++d) {
                in.read(reinterpret_cast<char*>(&shape[d]), sizeof(int64_t));
                running_crc = calculate_crc32(&shape[d], sizeof(int64_t), running_crc);
                if (shape[d] != param->shape[d]) {
                    throw std::runtime_error("[litetorch Error] Adam state shape mismatch");
                }
            }
            uint64_t numel;
            in.read(reinterpret_cast<char*>(&numel), sizeof(numel));
            running_crc = calculate_crc32(&numel, sizeof(numel), running_crc);
            if (numel != param->numel()) {
                throw std::runtime_error("[litetorch Error] Adam state numel mismatch");
            }
            std::vector<float> cpu_data(numel);
            in.read(reinterpret_cast<char*>(cpu_data.data()), numel * sizeof(float));
            running_crc = calculate_crc32(cpu_data.data(), numel * sizeof(float), running_crc);
            if (!in) {
                throw std::runtime_error("[litetorch Error] Unexpected end of file or read error in Adam state file");
            }
            loaded_v[i] = std::move(cpu_data);
        }

        uint32_t stored_crc;
        in.read(reinterpret_cast<char*>(&stored_crc), sizeof(stored_crc));
        if (!in) {
            throw std::runtime_error("[litetorch Error] Failed to read CRC32 checksum from optimizer state file");
        }
        uint32_t expected_crc = running_crc ^ 0xFFFFFFFF;
        if (stored_crc != expected_crc) {
            throw std::runtime_error("[litetorch Error] Optimizer state file is corrupted (CRC32 checksum mismatch)");
        }

        
        adam->step_count = step_count;
        for (size_t i = 0; i < adam->m.size(); ++i) {
            auto temp_m = Tensor::from_vector(loaded_m[i], adam->m[i]->shape, Device(DeviceType::CPU, 0));
            adam->m[i]->copy_(temp_m);
            auto temp_v = Tensor::from_vector(loaded_v[i], adam->v[i]->shape, Device(DeviceType::CPU, 0));
            adam->v[i]->copy_(temp_v);
        }

    } else if (magic[0] == 'L' && magic[1] == 'T' && magic[2] == 'S' && magic[3] == 'G') {
        auto sgd = dynamic_cast<optim::SGD*>(opt);
        if (!sgd) {
            throw std::runtime_error("[litetorch Error] Loaded SGD state but optimizer is not SGD");
        }

        uint32_t num_tensors;
        in.read(reinterpret_cast<char*>(&num_tensors), sizeof(num_tensors));
        running_crc = calculate_crc32(&num_tensors, sizeof(num_tensors), running_crc);
        if (num_tensors != sgd->velocity.size()) {
            throw std::runtime_error("[litetorch Error] SGD state size mismatch");
        }

        std::vector<std::vector<float>> loaded_velocity(num_tensors);
        for (size_t i = 0; i < sgd->velocity.size(); ++i) {
            auto param = sgd->velocity[i];
            uint32_t ndims;
            in.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));
            running_crc = calculate_crc32(&ndims, sizeof(ndims), running_crc);
            if (ndims != param->shape.size()) {
                throw std::runtime_error("[litetorch Error] SGD state ndims mismatch");
            }
            std::vector<int64_t> shape(ndims);
            for (uint32_t d = 0; d < ndims; ++d) {
                in.read(reinterpret_cast<char*>(&shape[d]), sizeof(int64_t));
                running_crc = calculate_crc32(&shape[d], sizeof(int64_t), running_crc);
                if (shape[d] != param->shape[d]) {
                    throw std::runtime_error("[litetorch Error] SGD state shape mismatch");
                }
            }
            uint64_t numel;
            in.read(reinterpret_cast<char*>(&numel), sizeof(numel));
            running_crc = calculate_crc32(&numel, sizeof(numel), running_crc);
            if (numel != param->numel()) {
                throw std::runtime_error("[litetorch Error] SGD state numel mismatch");
            }
            std::vector<float> cpu_data(numel);
            in.read(reinterpret_cast<char*>(cpu_data.data()), numel * sizeof(float));
            running_crc = calculate_crc32(cpu_data.data(), numel * sizeof(float), running_crc);
            if (!in) {
                throw std::runtime_error("[litetorch Error] Unexpected end of file or read error in SGD state file");
            }
            loaded_velocity[i] = std::move(cpu_data);
        }

        uint32_t stored_crc;
        in.read(reinterpret_cast<char*>(&stored_crc), sizeof(stored_crc));
        if (!in) {
            throw std::runtime_error("[litetorch Error] Failed to read CRC32 checksum from optimizer state file");
        }
        uint32_t expected_crc = running_crc ^ 0xFFFFFFFF;
        if (stored_crc != expected_crc) {
            throw std::runtime_error("[litetorch Error] Optimizer state file is corrupted (CRC32 checksum mismatch)");
        }

        
        for (size_t i = 0; i < sgd->velocity.size(); ++i) {
            auto temp = Tensor::from_vector(loaded_velocity[i], sgd->velocity[i]->shape, Device(DeviceType::CPU, 0));
            sgd->velocity[i]->copy_(temp);
        }

    } else if (magic[0] == 'L' && magic[1] == 'T' && magic[2] == 'R' && magic[3] == 'P') {
        auto rmsprop = dynamic_cast<optim::RMSprop*>(opt);
        if (!rmsprop) {
            throw std::runtime_error("[litetorch Error] Loaded RMSprop state but optimizer is not RMSprop");
        }

        uint32_t num_tensors;
        in.read(reinterpret_cast<char*>(&num_tensors), sizeof(num_tensors));
        running_crc = calculate_crc32(&num_tensors, sizeof(num_tensors), running_crc);
        if (num_tensors != rmsprop->square_avg.size()) {
            throw std::runtime_error("[litetorch Error] RMSprop state size mismatch");
        }

        std::vector<std::vector<float>> loaded_square_avg(num_tensors);
        for (size_t i = 0; i < rmsprop->square_avg.size(); ++i) {
            auto param = rmsprop->square_avg[i];
            uint32_t ndims;
            in.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));
            running_crc = calculate_crc32(&ndims, sizeof(ndims), running_crc);
            if (ndims != param->shape.size()) {
                throw std::runtime_error("[litetorch Error] RMSprop state ndims mismatch");
            }
            std::vector<int64_t> shape(ndims);
            for (uint32_t d = 0; d < ndims; ++d) {
                in.read(reinterpret_cast<char*>(&shape[d]), sizeof(int64_t));
                running_crc = calculate_crc32(&shape[d], sizeof(int64_t), running_crc);
                if (shape[d] != param->shape[d]) {
                    throw std::runtime_error("[litetorch Error] RMSprop state shape mismatch");
                }
            }
            uint64_t numel;
            in.read(reinterpret_cast<char*>(&numel), sizeof(numel));
            running_crc = calculate_crc32(&numel, sizeof(numel), running_crc);
            if (numel != param->numel()) {
                throw std::runtime_error("[litetorch Error] RMSprop state numel mismatch");
            }
            std::vector<float> cpu_data(numel);
            in.read(reinterpret_cast<char*>(cpu_data.data()), numel * sizeof(float));
            running_crc = calculate_crc32(cpu_data.data(), numel * sizeof(float), running_crc);
            if (!in) {
                throw std::runtime_error("[litetorch Error] Unexpected end of file or read error in RMSprop state file");
            }
            loaded_square_avg[i] = std::move(cpu_data);
        }

        uint32_t stored_crc;
        in.read(reinterpret_cast<char*>(&stored_crc), sizeof(stored_crc));
        if (!in) {
            throw std::runtime_error("[litetorch Error] Failed to read CRC32 checksum from optimizer state file");
        }
        uint32_t expected_crc = running_crc ^ 0xFFFFFFFF;
        if (stored_crc != expected_crc) {
            throw std::runtime_error("[litetorch Error] Optimizer state file is corrupted (CRC32 checksum mismatch)");
        }

        
        for (size_t i = 0; i < rmsprop->square_avg.size(); ++i) {
            auto temp = Tensor::from_vector(loaded_square_avg[i], rmsprop->square_avg[i]->shape, Device(DeviceType::CPU, 0));
            rmsprop->square_avg[i]->copy_(temp);
        }

    } else {
        throw std::runtime_error("[litetorch Error] Unsupported or corrupted optimizer file magic");
    }
}

}
