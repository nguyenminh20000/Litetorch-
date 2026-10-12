#include "litetorch/nn.h"
#include "litetorch/backend.h"
#include "litetorch/llm_serving.h"
#include "litetorch/quantization.h"
#include "litetorch/distributed.h"
#include <typeindex>
#include <unordered_set>

namespace litetorch {
namespace nn {

Sequential::Sequential(const std::vector<std::shared_ptr<Module>>& modules) : modules(modules) {}

void Sequential::add(std::shared_ptr<Module> module) {
    modules.push_back(module);
}

std::shared_ptr<Tensor> Sequential::forward(std::shared_ptr<Tensor> input) {
    if (is_compiled && input->device.type == DeviceType::GPU) {
        auto backend = BackendDispatcher::get().get_backend();
        if (backend && backend->is_available()) {
            if (compiled_graph_exec) {
                if (compiled_graph_input && compiled_graph_input->shape == input->shape &&
                    compiled_graph_input->dtype == input->dtype) {
                    compiled_graph_input->copy_(input);
                    backend->launch_graph(compiled_graph_exec);
                    return compiled_graph_output;
                }
            } else if (!is_graph_recorded) {
                backend->start_recording();
                auto out = input;
                for (auto& module : modules) {
                    out = module->forward(out);
                }
                void* exec = backend->stop_recording(nullptr);
                is_graph_recorded = true;
                if (exec) {
                    compiled_graph_exec = exec;
                    compiled_graph_input = input;
                    compiled_graph_output = out;
                }
                return out;
            }
        }
    }
    auto out = input;
    for (auto& module : modules) {
        out = module->forward(out);
    }
    return out;
}

std::vector<std::shared_ptr<Tensor>> Sequential::parameters() {
    std::vector<std::shared_ptr<Tensor>> params;
    for (auto& module : modules) {
        auto m_params = module->parameters();
        params.insert(params.end(), m_params.begin(), m_params.end());
    }
    return params;
}

void Sequential::to(const Device& device) {
    for (auto& module : modules) {
        module->to(device);
    }
}

void Sequential::train() {
    Module::train();
    for (auto& module : modules) {
        module->train();
    }
}

void Sequential::eval() {
    Module::eval();
    for (auto& module : modules) {
        module->eval();
    }
}

std::shared_ptr<Tensor> Flatten::forward(std::shared_ptr<Tensor> input) {
    int64_t batch_size = input->shape[0];
    int64_t remaining = input->numel() / batch_size;
    return input->view({batch_size, remaining});
}
ModuleList::ModuleList(const std::vector<std::shared_ptr<Module>>& modules) : modules(modules) {}

void ModuleList::append(std::shared_ptr<Module> module) {
    modules.push_back(module);
}

void ModuleList::extend(const std::vector<std::shared_ptr<Module>>& ms) {
    modules.insert(modules.end(), ms.begin(), ms.end());
}

std::shared_ptr<Module>& ModuleList::operator[](size_t i) {
    return modules[i];
}

const std::shared_ptr<Module>& ModuleList::operator[](size_t i) const {
    return modules[i];
}

size_t ModuleList::size() const {
    return modules.size();
}

std::shared_ptr<Tensor> ModuleList::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("[litetorch Error] ModuleList has no forward; iterate its modules manually");
}

std::vector<std::shared_ptr<Tensor>> ModuleList::parameters() {
    std::vector<std::shared_ptr<Tensor>> params;
    for (auto& module : modules) {
        auto m_params = module->parameters();
        params.insert(params.end(), m_params.begin(), m_params.end());
    }
    return params;
}

void ModuleList::to(const Device& device) {
    for (auto& module : modules) {
        module->to(device);
    }
}

void ModuleList::train() {
    Module::train();
    for (auto& module : modules) {
        module->train();
    }
}

void ModuleList::eval() {
    Module::eval();
    for (auto& module : modules) {
        module->eval();
    }
}

Parameter::Parameter(const std::vector<int64_t>& shape, const Device& device, DataType dtype)
    : Tensor(shape, device, true, dtype) {}

Buffer::Buffer(const std::vector<int64_t>& shape, const Device& device, DataType dtype)
    : Tensor(shape, device, false, dtype) {}

namespace {

void push_t(std::vector<std::pair<std::string, std::shared_ptr<Tensor>>>& v,
            const std::string& name, const std::shared_ptr<Tensor>& t) {
    if (t) v.push_back({name, t});
}

void push_m(std::vector<std::pair<std::string, std::shared_ptr<Module>>>& v,
            const std::string& name, const std::shared_ptr<Module>& m) {
    if (m) v.push_back({name, m});
}

}

const std::unordered_map<std::type_index, Module::LegacyFill>& Module::legacy_fill_map() {
    static const std::unordered_map<std::type_index, Module::LegacyFill> m = {
        {std::type_index(typeid(Linear)), [](const Module* b, LegacyLocal& l) {
            auto* s = static_cast<const Linear*>(b);
            push_t(l.params, "weight", s->weight);
            push_t(l.params, "bias", s->bias);
        }},
        {std::type_index(typeid(QLoRALinear)), [](const Module* b, LegacyLocal& l) {
            auto* s = static_cast<const QLoRALinear*>(b);
            push_t(l.params, "weight", s->weight);
            push_t(l.params, "bias", s->bias);
            push_t(l.params, "lora_A", s->lora_A);
            push_t(l.params, "lora_B", s->lora_B);
        }},
        {std::type_index(typeid(MoELinear)), [](const Module* b, LegacyLocal& l) {
            auto* s = static_cast<const MoELinear*>(b);
            push_t(l.params, "gate_weight", s->gate_weight);
            for (size_t i = 0; i < s->experts.size(); ++i)
                push_m(l.children, "experts." + std::to_string(i), s->experts[i]);
        }},
        {std::type_index(typeid(ColumnParallelLinear)), [](const Module* b, LegacyLocal& l) {
            auto* s = static_cast<const ColumnParallelLinear*>(b);
            push_t(l.params, "weight", s->weight);
            push_t(l.params, "bias", s->bias);
        }},
        {std::type_index(typeid(RowParallelLinear)), [](const Module* b, LegacyLocal& l) {
            auto* s = static_cast<const RowParallelLinear*>(b);
            push_t(l.params, "weight", s->weight);
            push_t(l.params, "bias", s->bias);
        }},
        {std::type_index(typeid(Conv2d)), [](const Module* b, LegacyLocal& l) {
            auto* s = static_cast<const Conv2d*>(b);
            push_t(l.params, "weight", s->weight);
            push_t(l.params, "bias", s->bias);
        }},
        {std::type_index(typeid(Conv3d)), [](const Module* b, LegacyLocal& l) {
            auto* s = static_cast<const Conv3d*>(b);
            push_t(l.params, "weight", s->weight);
            push_t(l.params, "bias", s->bias);
        }},
        {std::type_index(typeid(Sequential)), [](const Module* b, LegacyLocal& l) {
            auto* s = static_cast<const Sequential*>(b);
            for (size_t i = 0; i < s->modules.size(); ++i)
                push_m(l.children, std::to_string(i), s->modules[i]);
        }},
        {std::type_index(typeid(ModuleList)), [](const Module* b, LegacyLocal& l) {
            auto* s = static_cast<const ModuleList*>(b);
            for (size_t i = 0; i < s->modules.size(); ++i)
                push_m(l.children, std::to_string(i), s->modules[i]);
        }},
        {std::type_index(typeid(ModuleDict)), [](const Module* b, LegacyLocal& l) {
            auto* s = static_cast<const ModuleDict*>(b);
            for (auto& kv : s->items) push_m(l.children, kv.first, kv.second);
        }},
        {std::type_index(typeid(BatchNorm2d)), [](const Module* b, LegacyLocal& l) {
            auto* s = static_cast<const BatchNorm2d*>(b);
            push_t(l.params, "weight", s->weight);
            push_t(l.params, "bias", s->bias);
            push_t(l.buffers, "running_mean", s->running_mean);
            push_t(l.buffers, "running_var", s->running_var);
        }},
        {std::type_index(typeid(CrossEntropyLoss)), [](const Module* b, LegacyLocal& l) {
            auto* s = static_cast<const CrossEntropyLoss*>(b);
            push_t(l.params, "weight", s->weight);
        }},
        {std::type_index(typeid(Embedding)), [](const Module* b, LegacyLocal& l) {
            auto* s = static_cast<const Embedding*>(b);
            push_t(l.params, "weight", s->weight);
        }},
        {std::type_index(typeid(LayerNorm)), [](const Module* b, LegacyLocal& l) {
            auto* s = static_cast<const LayerNorm*>(b);
            push_t(l.params, "weight", s->weight);
            push_t(l.params, "bias", s->bias);
        }},
        {std::type_index(typeid(RMSNorm)), [](const Module* b, LegacyLocal& l) {
            auto* s = static_cast<const RMSNorm*>(b);
            push_t(l.params, "weight", s->weight);
        }},
        {std::type_index(typeid(MultiHeadAttention)), [](const Module* b, LegacyLocal& l) {
            auto* s = static_cast<const MultiHeadAttention*>(b);
            push_m(l.children, "qkv_proj", s->qkv_proj);
            push_m(l.children, "q_proj", s->q_proj);
            push_m(l.children, "k_proj", s->k_proj);
            push_m(l.children, "v_proj", s->v_proj);
            push_m(l.children, "out_proj", s->out_proj);
            push_m(l.children, "attn_dropout", s->attn_dropout);
        }},
        {std::type_index(typeid(TransformerDecoderLayer)), [](const Module* b, LegacyLocal& l) {
            auto* s = static_cast<const TransformerDecoderLayer*>(b);
            push_m(l.children, "self_attn", s->self_attn);
            push_m(l.children, "multihead_attn", s->multihead_attn);
            push_m(l.children, "linear1", s->linear1);
            push_m(l.children, "linear2", s->linear2);
            push_m(l.children, "norm1", s->norm1);
            push_m(l.children, "norm2", s->norm2);
            push_m(l.children, "norm3", s->norm3);
        }},
        {std::type_index(typeid(GPTBlock)), [](const Module* b, LegacyLocal& l) {
            auto* s = static_cast<const GPTBlock*>(b);
            push_m(l.children, "q_proj", s->q_proj);
            push_m(l.children, "k_proj", s->k_proj);
            push_m(l.children, "v_proj", s->v_proj);
            push_m(l.children, "o_proj", s->o_proj);
            push_m(l.children, "q_norm", s->q_norm);
            push_m(l.children, "k_norm", s->k_norm);
            push_m(l.children, "gate_proj", s->gate_proj);
            push_m(l.children, "up_proj", s->up_proj);
            push_m(l.children, "down_proj", s->down_proj);
            push_m(l.children, "attn_norm", s->attn_norm);
            push_m(l.children, "ffn_norm", s->ffn_norm);
        }},
        {std::type_index(typeid(GPT2B)), [](const Module* b, LegacyLocal& l) {
            auto* s = static_cast<const GPT2B*>(b);
            push_m(l.children, "tok_emb", s->tok_emb);
            for (size_t i = 0; i < s->blocks.size(); ++i)
                push_m(l.children, "blocks." + std::to_string(i), s->blocks[i]);
            push_m(l.children, "final_norm", s->final_norm);
            push_m(l.children, "lm_head", s->lm_head);
        }},
        {std::type_index(typeid(MedusaHead)), [](const Module* b, LegacyLocal& l) {
            auto* s = static_cast<const MedusaHead*>(b);
            push_t(l.params, "weight", s->weight);
            push_t(l.params, "bias", s->bias);
        }},
        {std::type_index(typeid(quantization::QATLinear)), [](const Module* b, LegacyLocal& l) {
            auto* s = static_cast<const quantization::QATLinear*>(b);
            push_t(l.params, "weight", s->weight);
            push_t(l.params, "bias", s->bias);
        }},
        {std::type_index(typeid(distributed::PipelineParallelModule)), [](const Module* b, LegacyLocal& l) {
            auto* s = static_cast<const distributed::PipelineParallelModule*>(b);
            push_m(l.children, "module", s->local_module);
        }},
    };
    return m;
}

Module::LegacyLocal Module::legacy_local() const {
    LegacyLocal l;
    auto it = legacy_fill_map().find(std::type_index(typeid(*this)));
    if (it != legacy_fill_map().end()) it->second(this, l);
    return l;
}

std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> Module::local_params() const {
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> out = named_params_;
    std::unordered_set<const Tensor*> seen;
    for (auto& kv : out) {
        if (kv.second) seen.insert(kv.second.get());
    }
    for (auto& kv : legacy_local().params) {
        if (kv.second && seen.insert(kv.second.get()).second) out.push_back(kv);
    }
    return out;
}

std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> Module::local_buffers() const {
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> out = named_buffers_;
    std::unordered_set<const Tensor*> seen;
    for (auto& kv : out) {
        if (kv.second) seen.insert(kv.second.get());
    }
    for (auto& kv : legacy_local().buffers) {
        if (kv.second && seen.insert(kv.second.get()).second) out.push_back(kv);
    }
    return out;
}

std::vector<std::shared_ptr<Tensor>> Module::parameters() {
    std::vector<std::shared_ptr<Tensor>> out;
    std::unordered_set<const Tensor*> seen;
    for (auto& kv : named_params_) {
        if (kv.second && seen.insert(kv.second.get()).second) out.push_back(kv.second);
    }
    for (auto& ck : named_children()) {
        if (!ck.second) continue;
        for (auto& t : ck.second->parameters()) {
            if (t && seen.insert(t.get()).second) out.push_back(t);
        }
    }
    return out;
}

void Module::to(const Device& device) {
    for (auto& kv : named_params_) {
        if (kv.second) kv.second = kv.second->to(device);
    }
    for (auto& kv : named_buffers_) {
        if (kv.second) kv.second = kv.second->to(device);
    }
    for (auto& ck : named_children()) {
        if (ck.second) ck.second->to(device);
    }
}

void Module::train() {
    training = true;
    for (auto& ck : named_children()) {
        if (ck.second) ck.second->train();
    }
}

void Module::eval() {
    training = false;
    for (auto& ck : named_children()) {
        if (ck.second) ck.second->eval();
    }
}

void Module::register_parameter(const std::string& name, std::shared_ptr<Tensor> tensor) {
    if (!tensor) throw std::runtime_error("[litetorch Error] register_parameter got null tensor for '" + name + "'");
    tensor->requires_grad = true;
    for (auto& kv : named_params_) {
        if (kv.first == name) {
            kv.second = tensor;
            return;
        }
    }
    named_params_.push_back({name, tensor});
}

void Module::register_buffer(const std::string& name, std::shared_ptr<Tensor> tensor) {
    if (!tensor) throw std::runtime_error("[litetorch Error] register_buffer got null tensor for '" + name + "'");
    for (auto& kv : named_buffers_) {
        if (kv.first == name) {
            kv.second = tensor;
            return;
        }
    }
    named_buffers_.push_back({name, tensor});
}

void Module::add_module(const std::string& name, std::shared_ptr<Module> module) {
    if (!module) throw std::runtime_error("[litetorch Error] add_module got null module for '" + name + "'");
    for (auto& kv : named_children_) {
        if (kv.first == name) {
            kv.second = module;
            return;
        }
    }
    named_children_.push_back({name, module});
}

std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> Module::named_parameters(const std::string& prefix) {
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> out;
    for (auto& kv : local_params()) {
        if (kv.second) out.push_back({prefix + kv.first, kv.second});
    }
    for (auto& ck : named_children()) {
        if (!ck.second) continue;
        std::string cp = prefix.empty() ? ck.first : prefix + "." + ck.first;
        for (auto& kv : ck.second->named_parameters(cp)) out.push_back(kv);
    }
    return out;
}

std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> Module::named_buffers(const std::string& prefix) {
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> out;
    for (auto& kv : local_buffers()) {
        if (kv.second) out.push_back({prefix + kv.first, kv.second});
    }
    for (auto& ck : named_children()) {
        if (!ck.second) continue;
        std::string cp = prefix.empty() ? ck.first : prefix + "." + ck.first;
        for (auto& kv : ck.second->named_buffers(cp)) out.push_back(kv);
    }
    return out;
}

std::vector<std::shared_ptr<Tensor>> Module::buffers() {
    std::vector<std::shared_ptr<Tensor>> out;
    for (auto& kv : named_buffers()) out.push_back(kv.second);
    return out;
}

std::vector<std::pair<std::string, std::shared_ptr<Module>>> Module::named_children() {
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> out = named_children_;
    std::unordered_set<const Module*> seen;
    for (auto& kv : out) {
        if (kv.second) seen.insert(kv.second.get());
    }
    for (auto& kv : legacy_local().children) {
        if (kv.second && seen.insert(kv.second.get()).second) out.push_back(kv);
    }
    if (out.empty()) {
        auto ch = children();
        for (size_t i = 0; i < ch.size(); ++i) {
            if (ch[i] && seen.insert(ch[i].get()).second) out.push_back({std::to_string(i), ch[i]});
        }
    }
    return out;
}

std::vector<std::pair<std::string, std::shared_ptr<Module>>> Module::named_modules(const std::string& prefix) {
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> out;
    out.push_back({prefix, shared_from_this()});
    for (auto& ck : named_children()) {
        if (!ck.second) continue;
        std::string cp = prefix.empty() ? ck.first : prefix + "." + ck.first;
        for (auto& kv : ck.second->named_modules(cp)) out.push_back(kv);
    }
    return out;
}

void Module::state_dict_rec(const std::string& prefix,
                            std::vector<std::pair<std::string, std::shared_ptr<Tensor>>>& out) {
    for (auto& kv : local_params()) {
        if (kv.second) out.push_back({prefix + kv.first, kv.second});
    }
    for (auto& kv : local_buffers()) {
        if (kv.second) out.push_back({prefix + kv.first, kv.second});
    }
    for (auto& ck : named_children()) {
        if (!ck.second) continue;
        std::string cp = prefix.empty() ? ck.first : prefix + "." + ck.first;
        ck.second->state_dict_rec(cp, out);
    }
}

std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> Module::state_dict() {
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> out;
    state_dict_rec("", out);
    return out;
}

std::pair<std::vector<std::string>, std::vector<std::string>> Module::load_state_dict(
    const std::vector<std::pair<std::string, std::shared_ptr<Tensor>>>& state, bool strict) {
    std::unordered_map<std::string, std::shared_ptr<Tensor>> own;
    std::vector<std::string> order;
    for (auto& kv : state_dict()) {
        if (!own.count(kv.first)) order.push_back(kv.first);
        own[kv.first] = kv.second;
    }
    std::vector<std::string> missing;
    std::vector<std::string> unexpected;
    std::unordered_set<std::string> seen;
    for (auto& kv : state) {
        auto it = own.find(kv.first);
        if (it == own.end()) {
            unexpected.push_back(kv.first);
            continue;
        }
        auto& dst = it->second;
        auto& src = kv.second;
        if (!src) throw std::runtime_error("[litetorch Error] load_state_dict: null tensor for key '" + kv.first + "'");
        if (!dst) throw std::runtime_error("[litetorch Error] load_state_dict: null target tensor for key '" + kv.first + "'");
        if (dst->shape != src->shape)
            throw std::runtime_error("[litetorch Error] load_state_dict: size mismatch for key '" + kv.first + "'");
        if (dst->dtype != src->dtype)
            throw std::runtime_error("[litetorch Error] load_state_dict: dtype mismatch for key '" + kv.first + "'");
        dst->copy_(src);
        seen.insert(kv.first);
    }
    for (auto& k : order) {
        if (!seen.count(k)) missing.push_back(k);
    }
    if (strict && (!missing.empty() || !unexpected.empty())) {
        std::string msg = "[litetorch Error] load_state_dict strict check failed.";
        if (!missing.empty()) {
            msg += " Missing keys:";
            for (auto& k : missing) msg += " " + k + ";";
        }
        if (!unexpected.empty()) {
            msg += " Unexpected keys:";
            for (auto& k : unexpected) msg += " " + k + ";";
        }
        throw std::runtime_error(msg);
    }
    return {missing, unexpected};
}

void Module::zero_grad() {
    for (auto& t : parameters()) {
        if (t && t->grad) t->grad = Tensor::zeros(t->grad->shape, t->grad->device, false, t->grad->dtype);
    }
}

void Module::apply(const std::function<void(std::shared_ptr<Module>)>& fn) {
    fn(shared_from_this());
    for (auto& ck : named_children()) {
        if (ck.second) ck.second->apply(fn);
    }
}

void ModuleDict::put(const std::string& key, std::shared_ptr<Module> module) {
    if (!module) throw std::runtime_error("[litetorch Error] ModuleDict::put got null module for key '" + key + "'");
    for (auto& kv : items) {
        if (kv.first == key) {
            kv.second = module;
            return;
        }
    }
    items.push_back({key, module});
}

std::shared_ptr<Module> ModuleDict::get(const std::string& key) const {
    for (auto& kv : items) {
        if (kv.first == key) return kv.second;
    }
    return nullptr;
}

bool ModuleDict::contains(const std::string& key) const {
    return get(key) != nullptr;
}

void ModuleDict::erase(const std::string& key) {
    for (auto it = items.begin(); it != items.end(); ++it) {
        if (it->first == key) {
            items.erase(it);
            return;
        }
    }
}

void ModuleDict::clear() {
    items.clear();
}

size_t ModuleDict::size() const {
    return items.size();
}

std::vector<std::string> ModuleDict::keys() const {
    std::vector<std::string> ks;
    for (auto& kv : items) ks.push_back(kv.first);
    return ks;
}

std::shared_ptr<Module>& ModuleDict::operator[](const std::string& key) {
    for (auto& kv : items) {
        if (kv.first == key) return kv.second;
    }
    throw std::out_of_range("[litetorch Error] ModuleDict key not found: '" + key + "'");
}

const std::shared_ptr<Module>& ModuleDict::operator[](const std::string& key) const {
    for (auto& kv : items) {
        if (kv.first == key) return kv.second;
    }
    throw std::out_of_range("[litetorch Error] ModuleDict key not found: '" + key + "'");
}

std::shared_ptr<Tensor> ModuleDict::forward(std::shared_ptr<Tensor>) {
    throw std::runtime_error("[litetorch Error] ModuleDict has no forward; index its modules manually");
}

std::vector<std::shared_ptr<Tensor>> ModuleDict::parameters() {
    std::vector<std::shared_ptr<Tensor>> params;
    for (auto& kv : items) {
        if (!kv.second) continue;
        auto m_params = kv.second->parameters();
        params.insert(params.end(), m_params.begin(), m_params.end());
    }
    return params;
}

void ModuleDict::to(const Device& device) {
    for (auto& kv : items) {
        if (kv.second) kv.second->to(device);
    }
}

void ModuleDict::train() {
    Module::train();
    for (auto& kv : items) {
        if (kv.second) kv.second->train();
    }
}

void ModuleDict::eval() {
    Module::eval();
    for (auto& kv : items) {
        if (kv.second) kv.second->eval();
    }
}

std::vector<std::shared_ptr<Module>> ModuleDict::children() {
    std::vector<std::shared_ptr<Module>> c;
    for (auto& kv : items) c.push_back(kv.second);
    return c;
}

}
}
