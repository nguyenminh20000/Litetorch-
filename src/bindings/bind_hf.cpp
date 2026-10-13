#include "bindings_common.h"
#include "litetorch/hf.h"

void bind_hf(py::module_& m) {
    auto hf_mod = m.def_submodule("hf", "LiteTorch HuggingFace integration");

    py::class_<litetorch::hf::ModelConfig>(hf_mod, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("model_type", &litetorch::hf::ModelConfig::model_type)
        .def_readwrite("hidden_size", &litetorch::hf::ModelConfig::hidden_size)
        .def_readwrite("num_hidden_layers", &litetorch::hf::ModelConfig::num_hidden_layers)
        .def_readwrite("num_attention_heads", &litetorch::hf::ModelConfig::num_attention_heads)
        .def_readwrite("num_key_value_heads", &litetorch::hf::ModelConfig::num_key_value_heads)
        .def_readwrite("intermediate_size", &litetorch::hf::ModelConfig::intermediate_size)
        .def_readwrite("vocab_size", &litetorch::hf::ModelConfig::vocab_size)
        .def_readwrite("max_position_embeddings",
                       &litetorch::hf::ModelConfig::max_position_embeddings)
        .def_readwrite("hidden_act", &litetorch::hf::ModelConfig::hidden_act)
        .def_readwrite("rms_norm_eps", &litetorch::hf::ModelConfig::rms_norm_eps)
        .def_readwrite("layer_norm_eps", &litetorch::hf::ModelConfig::layer_norm_eps)
        .def_readwrite("n_layer", &litetorch::hf::ModelConfig::n_layer)
        .def_readwrite("n_head", &litetorch::hf::ModelConfig::n_head)
        .def_readwrite("n_embd", &litetorch::hf::ModelConfig::n_embd)
        .def_readwrite("n_positions", &litetorch::hf::ModelConfig::n_positions);

    py::class_<litetorch::hf::PretrainedBundle>(hf_mod, "PretrainedBundle")
        .def(py::init<>())
        .def_readwrite("config", &litetorch::hf::PretrainedBundle::config)
        .def_readwrite("state_dict", &litetorch::hf::PretrainedBundle::state_dict);

    hf_mod.def("load_config", &litetorch::hf::load_config, py::arg("repo_or_dir"),
               py::arg("revision") = "main");
    hf_mod.def("from_pretrained", &litetorch::hf::from_pretrained, py::arg("repo_id"),
               py::arg("revision") = "main");
    hf_mod.def("weight_name_map", &litetorch::hf::weight_name_map, py::arg("model_type"));
    hf_mod.def("remap_state_dict", &litetorch::hf::remap_state_dict, py::arg("state_dict"),
               py::arg("model_type"));
}
