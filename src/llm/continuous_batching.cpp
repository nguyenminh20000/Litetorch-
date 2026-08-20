#include "litetorch/continuous_batching.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace litetorch {
namespace nn {

ServingScheduler::ServingScheduler(int num_blocks, int block_size, int max_num_seqs, int max_num_batched_tokens)
    : block_size(block_size), max_num_seqs(max_num_seqs), max_num_batched_tokens(max_num_batched_tokens), num_blocks(num_blocks) {
    for (int i = 0; i < num_blocks; ++i) {
        free_blocks.push_back(i);
    }
}

void ServingScheduler::add_request(std::shared_ptr<Request> req) {
    pending_queue.push_back(req);
}

std::vector<std::shared_ptr<Request>> ServingScheduler::step(std::shared_ptr<Module> model) {
    std::vector<std::shared_ptr<Request>> completed;

    while (running_queue.size() < static_cast<size_t>(max_num_seqs) && !pending_queue.empty()) {
        auto req = pending_queue.front();
        size_t total_tokens = req->prompt_tokens.size() + req->max_new_tokens;
        size_t needed_blocks = (total_tokens + block_size - 1) / block_size;
        if (free_blocks.size() >= needed_blocks) {
            pending_queue.erase(pending_queue.begin());
            for (size_t i = 0; i < needed_blocks; ++i) {
                req->block_ids.push_back(free_blocks.back());
                free_blocks.pop_back();
            }
            running_queue.push_back(req);
        } else {
            break;
        }
    }

    if (running_queue.empty()) {
        return completed;
    }

    int budget = max_num_batched_tokens;
    std::vector<std::shared_ptr<Request>> active_this_step;

    std::unordered_map<uint64_t, int64_t> current_step_chunks;
    for (auto& req : running_queue) {
        if (budget <= 0) break;
        active_this_step.push_back(req);
        if (req->prefill_processed < static_cast<int64_t>(req->prompt_tokens.size())) {
            int64_t chunk = std::min(static_cast<int64_t>(req->prompt_tokens.size()) - req->prefill_processed, static_cast<int64_t>(budget));
            current_step_chunks[req->id] = chunk;
            req->prefill_processed += chunk;
            budget -= chunk;
        } else {
            current_step_chunks[req->id] = 0;
            budget -= 1;
        }
    }

    std::vector<std::shared_ptr<Request>> decode_reqs;
    std::vector<std::shared_ptr<Request>> prefill_reqs;
    for (auto& req : active_this_step) {
        int64_t chunk = current_step_chunks[req->id];
        if (chunk > 0) {
            prefill_reqs.push_back(req);
        } else {
            decode_reqs.push_back(req);
        }
    }

    for (auto& req : prefill_reqs) {
        int64_t chunk = current_step_chunks[req->id];
        int64_t start = req->prefill_processed - chunk;
        int64_t length = chunk;
        std::vector<float> vec(length);
        for (int64_t i = 0; i < length; ++i) {
            vec[i] = static_cast<float>(req->prompt_tokens[start + i]);
        }
        auto input = Tensor::from_vector(vec, {1, length});
        auto output = model->forward(input);

        float* out_ptr = output->data_ptr();
        int64_t vocab_size = output->shape[2];
        int64_t seq_len = output->shape[1];

        float max_logit = -1e9f;
        int next_token = 0;
        for (int v = 0; v < vocab_size; ++v) {
            float logit = out_ptr[(seq_len - 1) * vocab_size + v];
            if (logit > max_logit) {
                max_logit = logit;
                next_token = v;
            }
        }

        if (req->prefill_processed >= static_cast<int64_t>(req->prompt_tokens.size())) {
            req->generated_tokens.push_back(next_token);
        }

        if (next_token == 2 || req->generated_tokens.size() >= static_cast<size_t>(req->max_new_tokens)) {
            req->is_finished = true;
        }
    }

    if (!decode_reqs.empty()) {
        int64_t batch_size = decode_reqs.size();
        std::vector<float> batch_vec(batch_size);
        for (int64_t i = 0; i < batch_size; ++i) {
            auto& req = decode_reqs[i];
            float last_token = 0.0f;
            if (req->generated_tokens.empty()) {
                last_token = static_cast<float>(req->prompt_tokens.back());
            } else {
                last_token = static_cast<float>(req->generated_tokens.back());
            }
            batch_vec[i] = last_token;
        }
        auto input = Tensor::from_vector(batch_vec, {batch_size, 1});
        auto output = model->forward(input);

        float* out_ptr = output->data_ptr();
        int64_t vocab_size = output->shape[2];
        int64_t seq_len = output->shape[1];

        for (int64_t i = 0; i < batch_size; ++i) {
            auto& req = decode_reqs[i];
            float max_logit = -1e9f;
            int next_token = 0;
            for (int v = 0; v < vocab_size; ++v) {
                float logit = out_ptr[i * (seq_len * vocab_size) + (seq_len - 1) * vocab_size + v];
                if (logit > max_logit) {
                    max_logit = logit;
                    next_token = v;
                }
            }

            if (req->prefill_processed >= static_cast<int64_t>(req->prompt_tokens.size())) {
                req->generated_tokens.push_back(next_token);
            }

            if (next_token == 2 || req->generated_tokens.size() >= static_cast<size_t>(req->max_new_tokens)) {
                req->is_finished = true;
            }
        }
    }

    for (auto it = running_queue.begin(); it != running_queue.end();) {
        if ((*it)->is_finished) {
            for (int bid : (*it)->block_ids) {
                free_blocks.push_back(bid);
            }
            completed.push_back(*it);
            it = running_queue.erase(it);
        } else {
            ++it;
        }
    }

    return completed;
}

std::shared_ptr<Tensor> ServingScheduler::get_block_table_tensor() {
    int64_t num_seqs = running_queue.size();
    if (num_seqs == 0) return nullptr;

    int64_t max_blocks = 0;
    for (auto& req : running_queue) {
        max_blocks = std::max(max_blocks, static_cast<int64_t>(req->block_ids.size()));
    }

    auto tensor = Tensor::create({num_seqs, max_blocks}, Device(DeviceType::CPU), false, DataType::FP32);
    float* ptr = tensor->data_ptr();
    std::fill(ptr, ptr + num_seqs * max_blocks, 0.0f);

    for (int64_t i = 0; i < num_seqs; ++i) {
        auto& req = running_queue[i];
        for (size_t j = 0; j < req->block_ids.size(); ++j) {
            ptr[i * max_blocks + j] = static_cast<float>(req->block_ids[j]);
        }
    }
    return tensor;
}

std::shared_ptr<Tensor> ServingScheduler::get_context_lens_tensor() {
    int64_t num_seqs = running_queue.size();
    if (num_seqs == 0) return nullptr;

    auto tensor = Tensor::create({num_seqs}, Device(DeviceType::CPU), false, DataType::FP32);
    float* ptr = tensor->data_ptr();

    for (int64_t i = 0; i < num_seqs; ++i) {
        auto& req = running_queue[i];
        ptr[i] = static_cast<float>(req->prefill_processed + req->generated_tokens.size());
    }
    return tensor;
}

}
}
