#include "litetorch/serialization.h"
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

void save_parameters(const std::vector<std::shared_ptr<Tensor>>& params, const std::string& filepath) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("[litetorch Error] Failed to open file for saving weights: " + filepath);
    }

    uint32_t running_crc = 0xFFFFFFFF;

    const char magic[4] = {'L', 'T', 'W', '1'};
    out.write(magic, 4);
    running_crc = calculate_crc32(magic, 4, running_crc);

    uint32_t num_tensors = static_cast<uint32_t>(params.size());
    out.write(reinterpret_cast<const char*>(&num_tensors), sizeof(num_tensors));
    running_crc = calculate_crc32(&num_tensors, sizeof(num_tensors), running_crc);

    for (const auto& param : params) {
        if (!param) {
            throw std::runtime_error("[litetorch Error] Cannot save null parameter tensor");
        }
        
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

    uint32_t final_crc = running_crc ^ 0xFFFFFFFF;
    out.write(reinterpret_cast<const char*>(&final_crc), sizeof(final_crc));
    out.flush();
    if (out.fail()) {
        throw std::runtime_error("[litetorch Error] Write failure or disk full during saving weights");
    }
    out.close();
}

void load_parameters(const std::vector<std::shared_ptr<Tensor>>& params, const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("[litetorch Error] Failed to open file for loading weights: " + filepath);
    }

    uint32_t running_crc = 0xFFFFFFFF;

    char magic[4];
    in.read(magic, 4);
    if (magic[0] != 'L' || magic[1] != 'T' || magic[2] != 'W' || magic[3] != '1') {
        throw std::runtime_error("[litetorch Error] Invalid file format magic in weights file");
    }
    running_crc = calculate_crc32(magic, 4, running_crc);

    uint32_t num_tensors;
    in.read(reinterpret_cast<char*>(&num_tensors), sizeof(num_tensors));
    running_crc = calculate_crc32(&num_tensors, sizeof(num_tensors), running_crc);

    if (num_tensors != params.size()) {
        throw std::runtime_error("[litetorch Error] Weights file has " + std::to_string(num_tensors) + 
                                 " tensors, but model has " + std::to_string(params.size()) + " parameters");
    }

    std::vector<std::vector<float>> loaded_tensors_data(params.size());

    for (size_t i = 0; i < params.size(); ++i) {
        auto param = params[i];
        if (!param) {
            throw std::runtime_error("[litetorch Error] Cannot load into null parameter tensor");
        }

        uint32_t ndims;
        in.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));
        running_crc = calculate_crc32(&ndims, sizeof(ndims), running_crc);

        if (ndims != param->shape.size()) {
            throw std::runtime_error("[litetorch Error] Tensor dimension mismatch at parameter index " + std::to_string(i));
        }

        std::vector<int64_t> loaded_shape(ndims);
        for (uint32_t d = 0; d < ndims; ++d) {
            in.read(reinterpret_cast<char*>(&loaded_shape[d]), sizeof(int64_t));
            running_crc = calculate_crc32(&loaded_shape[d], sizeof(int64_t), running_crc);
            if (loaded_shape[d] != param->shape[d]) {
                throw std::runtime_error("[litetorch Error] Tensor shape mismatch at parameter index " + std::to_string(i));
            }
        }

        uint64_t numel;
        in.read(reinterpret_cast<char*>(&numel), sizeof(numel));
        running_crc = calculate_crc32(&numel, sizeof(numel), running_crc);
        if (numel != param->numel()) {
            throw std::runtime_error("[litetorch Error] Tensor numel mismatch at parameter index " + std::to_string(i));
        }

        std::vector<float> cpu_data(numel);
        in.read(reinterpret_cast<char*>(cpu_data.data()), numel * sizeof(float));
        running_crc = calculate_crc32(cpu_data.data(), numel * sizeof(float), running_crc);
        if (!in) {
            throw std::runtime_error("[litetorch Error] Unexpected end of file or read error in weights file");
        }

        loaded_tensors_data[i] = std::move(cpu_data);
    }

    uint32_t stored_crc;
    in.read(reinterpret_cast<char*>(&stored_crc), sizeof(stored_crc));
    if (!in) {
        throw std::runtime_error("[litetorch Error] Failed to read CRC32 checksum from weights file");
    }

    uint32_t expected_crc = running_crc ^ 0xFFFFFFFF;
    if (stored_crc != expected_crc) {
        throw std::runtime_error("[litetorch Error] Model weights file is corrupted (CRC32 checksum mismatch)");
    }

    for (size_t i = 0; i < params.size(); ++i) {
        auto temp_tensor = Tensor::from_vector(loaded_tensors_data[i], params[i]->shape, Device(DeviceType::CPU, 0));
        params[i]->copy_(temp_tensor);
    }
}

}
