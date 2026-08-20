#ifndef LITETORCH_CONTINUOUS_BATCHING_H
#define LITETORCH_CONTINUOUS_BATCHING_H

#include "litetorch/tensor.h"
#include "litetorch/nn.h"
#include <vector>
#include <memory>
#include <unordered_map>

namespace litetorch {
namespace nn {

struct Request {
    uint64_t id;
    std::vector<int64_t> prompt_tokens;
    std::vector<int64_t> generated_tokens;
    int64_t max_new_tokens;
    int64_t prefill_processed = 0;
    bool is_finished = false;
    std::vector<int> block_ids;

    Request(uint64_t id, const std::vector<int64_t>& prompt, int64_t max_new_tokens = 32)
        : id(id), prompt_tokens(prompt), max_new_tokens(max_new_tokens) {}
};

class ServingScheduler {
public:
    int block_size;
    int max_num_seqs;
    int max_num_batched_tokens;
    std::vector<std::shared_ptr<Request>> pending_queue;
    std::vector<std::shared_ptr<Request>> running_queue;
    std::vector<int> free_blocks;
    int num_blocks;

    ServingScheduler(int num_blocks, int block_size, int max_num_seqs, int max_num_batched_tokens);

    void add_request(std::shared_ptr<Request> req);
    std::vector<std::shared_ptr<Request>> step(std::shared_ptr<Module> model);
    std::shared_ptr<Tensor> get_block_table_tensor();
    std::shared_ptr<Tensor> get_context_lens_tensor();
};

}
}

#endif
