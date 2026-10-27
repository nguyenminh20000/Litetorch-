#include "bindings_common.h"
#include <cmath>

void bind_optim(py::module_& m) {
        auto optim_mod = m.def_submodule("optim", "LiteTorch Optimizers");

        py::class_<optim::Optimizer, std::shared_ptr<optim::Optimizer>>(optim_mod, "Optimizer")
            .def("step", &optim::Optimizer::step)
            .def("zero_grad", &optim::Optimizer::zero_grad)
            .def("get_lr", &optim::Optimizer::get_lr)
            .def("set_lr", &optim::Optimizer::set_lr, py::arg("new_lr"));

        py::class_<optim::SGD, optim::Optimizer, std::shared_ptr<optim::SGD>>(optim_mod, "SGD")
            .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, double, float, float, float, bool>(), py::arg("params"), py::arg("lr") = 0.01, py::arg("momentum") = 0.0f, py::arg("dampening") = 0.0f, py::arg("weight_decay") = 0.0f, py::arg("nesterov") = false)
            .def("step", &optim::SGD::step)
            .def("zero_grad", &optim::SGD::zero_grad)
            .def("get_lr", &optim::SGD::get_lr)
            .def("set_lr", &optim::SGD::set_lr, py::arg("new_lr"))
            .def_readwrite("lr", &optim::SGD::lr)
            .def_readwrite("momentum", &optim::SGD::momentum)
            .def_readwrite("dampening", &optim::SGD::dampening)
            .def_readwrite("nesterov", &optim::SGD::nesterov)
            .def_readwrite("weight_decay", &optim::SGD::weight_decay);

        py::class_<optim::Adam, optim::Optimizer, std::shared_ptr<optim::Adam>>(optim_mod, "Adam")
            .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, double, float, float, float, float>(), py::arg("params"), py::arg("lr") = 0.001, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f)
            .def("step", &optim::Adam::step)
            .def("zero_grad", &optim::Adam::zero_grad)
            .def("get_lr", &optim::Adam::get_lr)
            .def("set_lr", &optim::Adam::set_lr, py::arg("new_lr"))
            .def_readwrite("lr", &optim::Adam::lr)
            .def_readwrite("beta1", &optim::Adam::beta1)
            .def_readwrite("beta2", &optim::Adam::beta2)
            .def_readwrite("eps", &optim::Adam::eps)
            .def_readwrite("weight_decay", &optim::Adam::weight_decay);

        py::class_<optim::AdamW, optim::Optimizer, std::shared_ptr<optim::AdamW>>(optim_mod, "AdamW")
            .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, double, float, float, float, float, bool>(), py::arg("params"), py::arg("lr") = 0.001, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.01f, py::arg("offload_to_cpu") = false)
            .def("step", &optim::AdamW::step)
            .def("zero_grad", &optim::AdamW::zero_grad)
            .def("get_lr", &optim::AdamW::get_lr)
            .def("set_lr", &optim::AdamW::set_lr, py::arg("new_lr"))
            .def_readwrite("lr", &optim::AdamW::lr)
            .def_readwrite("beta1", &optim::AdamW::beta1)
            .def_readwrite("beta2", &optim::AdamW::beta2)
            .def_readwrite("eps", &optim::AdamW::eps)
            .def_readwrite("weight_decay", &optim::AdamW::weight_decay)
            .def_static("fused_launch_count", &optim::AdamW::fused_launch_count)
            .def_static("single_launch_count", &optim::AdamW::single_launch_count)
            .def_static("reset_launch_counts", &optim::AdamW::reset_launch_counts);

        py::class_<optim::AdamW8bit, optim::Optimizer, std::shared_ptr<optim::AdamW8bit>>(optim_mod, "AdamW8bit")
            .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, double, float, float, float, float>(), py::arg("params"), py::arg("lr") = 0.001, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f)
            .def("step", &optim::AdamW8bit::step)
            .def("zero_grad", &optim::AdamW8bit::zero_grad)
            .def("get_lr", &optim::AdamW8bit::get_lr)
            .def("set_lr", &optim::AdamW8bit::set_lr, py::arg("new_lr"))
            .def_readwrite("lr", &optim::AdamW8bit::lr)
            .def_readwrite("weight_decay", &optim::AdamW8bit::weight_decay);

        py::class_<optim::AdamWFP8, optim::Optimizer, std::shared_ptr<optim::AdamWFP8>>(optim_mod, "AdamWFP8")
            .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, double, float, float, float, float>(), py::arg("params"), py::arg("lr") = 0.001, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f)
            .def("step", &optim::AdamWFP8::step)
            .def("zero_grad", &optim::AdamWFP8::zero_grad)
            .def("get_lr", &optim::AdamWFP8::get_lr)
            .def("set_lr", &optim::AdamWFP8::set_lr, py::arg("new_lr"))
            .def_readwrite("lr", &optim::AdamWFP8::lr)
            .def_readwrite("weight_decay", &optim::AdamWFP8::weight_decay);

        py::class_<optim::RMSprop, optim::Optimizer, std::shared_ptr<optim::RMSprop>>(optim_mod, "RMSprop")
            .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, double, float, float, float>(), py::arg("params"), py::arg("lr") = 0.01, py::arg("alpha") = 0.99f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f)
            .def("step", &optim::RMSprop::step)
            .def("zero_grad", &optim::RMSprop::zero_grad)
            .def("get_lr", &optim::RMSprop::get_lr)
            .def("set_lr", &optim::RMSprop::set_lr, py::arg("new_lr"))
            .def_readwrite("lr", &optim::RMSprop::lr)
            .def_readwrite("weight_decay", &optim::RMSprop::weight_decay);

        py::class_<optim::Adadelta, optim::Optimizer, std::shared_ptr<optim::Adadelta>>(optim_mod, "Adadelta")
            .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, double, float, float, float>(), py::arg("params"), py::arg("lr") = 1.0, py::arg("rho") = 0.9f, py::arg("eps") = 1e-6f, py::arg("weight_decay") = 0.0f)
            .def("step", &optim::Adadelta::step)
            .def("zero_grad", &optim::Adadelta::zero_grad)
            .def("get_lr", &optim::Adadelta::get_lr)
            .def("set_lr", &optim::Adadelta::set_lr, py::arg("new_lr"))
            .def_readwrite("lr", &optim::Adadelta::lr)
            .def_readwrite("rho", &optim::Adadelta::rho)
            .def_readwrite("weight_decay", &optim::Adadelta::weight_decay);

        py::class_<optim::Adamax, optim::Optimizer, std::shared_ptr<optim::Adamax>>(optim_mod, "Adamax")
            .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, double, float, float, float, float>(), py::arg("params"), py::arg("lr") = 0.002, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f)
            .def("step", &optim::Adamax::step)
            .def("zero_grad", &optim::Adamax::zero_grad)
            .def("get_lr", &optim::Adamax::get_lr)
            .def("set_lr", &optim::Adamax::set_lr, py::arg("new_lr"))
            .def_readwrite("lr", &optim::Adamax::lr)
            .def_readwrite("weight_decay", &optim::Adamax::weight_decay);

        py::class_<optim::ZeRO3Optimizer, optim::Optimizer, std::shared_ptr<optim::ZeRO3Optimizer>>(optim_mod, "ZeRO3Optimizer")
            .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, double, float, float, float, float>(), py::arg("params"), py::arg("lr") = 0.001, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f)
            .def("step", &optim::ZeRO3Optimizer::step)
            .def("zero_grad", &optim::ZeRO3Optimizer::zero_grad)
            .def_readwrite("lr", &optim::ZeRO3Optimizer::lr)
            .def_readwrite("beta1", &optim::ZeRO3Optimizer::beta1)
            .def_readwrite("beta2", &optim::ZeRO3Optimizer::beta2)
            .def_readwrite("eps", &optim::ZeRO3Optimizer::eps)
            .def_readwrite("weight_decay", &optim::ZeRO3Optimizer::weight_decay);

        py::class_<optim::StepLR>(optim_mod, "StepLR")
            .def(py::init<optim::Optimizer*, int, float>(), py::keep_alive<1, 2>(), py::arg("optimizer"), py::arg("step_size"), py::arg("gamma") = 0.1f)
            .def("step", &optim::StepLR::step)
            .def_readwrite("step_size", &optim::StepLR::step_size)
            .def_readwrite("gamma", &optim::StepLR::gamma)
            .def_readwrite("last_epoch", &optim::StepLR::last_epoch);

        py::class_<optim::CosineAnnealingLR>(optim_mod, "CosineAnnealingLR")
            .def(py::init<optim::Optimizer*, int, float>(), py::keep_alive<1, 2>(), py::arg("optimizer"), py::arg("T_max"), py::arg("eta_min") = 0.0f)
            .def("step", &optim::CosineAnnealingLR::step)
            .def_readwrite("T_max", &optim::CosineAnnealingLR::T_max)
            .def_readwrite("eta_min", &optim::CosineAnnealingLR::eta_min)
            .def_readwrite("last_epoch", &optim::CosineAnnealingLR::last_epoch);

        py::class_<optim::CosineAnnealingWarmRestarts>(optim_mod, "CosineAnnealingWarmRestarts")
            .def(py::init<optim::Optimizer*, int, int, float>(), py::keep_alive<1, 2>(), py::arg("optimizer"), py::arg("T_0"), py::arg("T_mult") = 1, py::arg("eta_min") = 0.0f)
            .def("step", &optim::CosineAnnealingWarmRestarts::step)
            .def("get_last_lr", &optim::CosineAnnealingWarmRestarts::get_last_lr)
            .def_readwrite("T_0", &optim::CosineAnnealingWarmRestarts::T_0)
            .def_readwrite("T_mult", &optim::CosineAnnealingWarmRestarts::T_mult)
            .def_readwrite("eta_min", &optim::CosineAnnealingWarmRestarts::eta_min)
            .def_readwrite("last_epoch", &optim::CosineAnnealingWarmRestarts::last_epoch);

        py::class_<optim::LinearLR>(optim_mod, "LinearLR")
            .def(py::init<optim::Optimizer*, float, float, int>(), py::keep_alive<1, 2>(), py::arg("optimizer"), py::arg("start_factor") = 1.0f / 3, py::arg("end_factor") = 1.0f, py::arg("total_iters") = 5)
            .def("step", &optim::LinearLR::step)
            .def("get_last_lr", &optim::LinearLR::get_last_lr)
            .def_readwrite("start_factor", &optim::LinearLR::start_factor)
            .def_readwrite("end_factor", &optim::LinearLR::end_factor)
            .def_readwrite("total_iters", &optim::LinearLR::total_iters)
            .def_readwrite("last_epoch", &optim::LinearLR::last_epoch);

        py::class_<optim::PolynomialLR>(optim_mod, "PolynomialLR")
            .def(py::init<optim::Optimizer*, int, float>(), py::keep_alive<1, 2>(), py::arg("optimizer"), py::arg("total_iters") = 5, py::arg("power") = 1.0f)
            .def("step", &optim::PolynomialLR::step)
            .def("get_last_lr", &optim::PolynomialLR::get_last_lr)
            .def_readwrite("total_iters", &optim::PolynomialLR::total_iters)
            .def_readwrite("power", &optim::PolynomialLR::power)
            .def_readwrite("last_epoch", &optim::PolynomialLR::last_epoch);

        py::class_<optim::OneCycleLR>(optim_mod, "OneCycleLR")
            .def(py::init<optim::Optimizer*, double, int, int, int, float, const std::string&, float, float>(), py::keep_alive<1, 2>(), py::arg("optimizer"), py::arg("max_lr"), py::arg("total_steps") = -1, py::arg("epochs") = -1, py::arg("steps_per_epoch") = -1, py::arg("pct_start") = 0.3f, py::arg("anneal_strategy") = "cos", py::arg("div_factor") = 25.0f, py::arg("final_div_factor") = 10000.0f)
            .def("step", &optim::OneCycleLR::step)
            .def("get_last_lr", &optim::OneCycleLR::get_last_lr)
            .def_readwrite("max_lr", &optim::OneCycleLR::max_lr)
            .def_readwrite("total_steps", &optim::OneCycleLR::total_steps)
            .def_readwrite("last_epoch", &optim::OneCycleLR::last_epoch);

        py::class_<optim::LambdaLR>(optim_mod, "LambdaLR")
            .def(py::init([](optim::Optimizer* opt, py::function fn) {
                std::function<double(int)> cpp_fn = [fn](int epoch) -> double {
                    py::gil_scoped_acquire gil;
                    return fn(epoch).cast<double>();
                };
                return std::unique_ptr<optim::LambdaLR>(new optim::LambdaLR(opt, cpp_fn));
            }), py::keep_alive<1, 2>(), py::arg("optimizer"), py::arg("lr_lambda"))
            .def("step", &optim::LambdaLR::step)
            .def("get_last_lr", &optim::LambdaLR::get_last_lr)
            .def_readwrite("last_epoch", &optim::LambdaLR::last_epoch);

        auto lr_sched_mod = optim_mod.def_submodule("lr_scheduler", "Learning rate schedulers");
        lr_sched_mod.def("StepLR", [](optim::Optimizer* opt, int step_size, float gamma) {
            return std::unique_ptr<optim::StepLR>(new optim::StepLR(opt, step_size, gamma));
        }, py::keep_alive<0, 1>(), py::arg("optimizer"), py::arg("step_size"), py::arg("gamma") = 0.1f);
        lr_sched_mod.def("CosineAnnealingLR", [](optim::Optimizer* opt, int T_max, float eta_min) {
            return std::unique_ptr<optim::CosineAnnealingLR>(new optim::CosineAnnealingLR(opt, T_max, eta_min));
        }, py::keep_alive<0, 1>(), py::arg("optimizer"), py::arg("T_max"), py::arg("eta_min") = 0.0f);
        py::class_<optim::LinearWarmup>(lr_sched_mod, "LinearWarmup")
            .def(py::init<optim::Optimizer*, int, int, float>(), py::keep_alive<1, 2>(), py::arg("optimizer"), py::arg("warmup_steps"), py::arg("total_steps"), py::arg("min_lr") = 0.0f)
            .def("step", &optim::LinearWarmup::step)
            .def("get_lr", &optim::LinearWarmup::get_lr)
            .def_readwrite("warmup_steps", &optim::LinearWarmup::warmup_steps)
            .def_readwrite("total_steps", &optim::LinearWarmup::total_steps)
            .def_readwrite("last_step", &optim::LinearWarmup::last_step);

        py::class_<optim::Adagrad, optim::Optimizer, std::shared_ptr<optim::Adagrad>>(optim_mod, "Adagrad")
            .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, double, float, float, float>(), py::arg("params"), py::arg("lr") = 0.01, py::arg("lr_decay") = 0.0f, py::arg("weight_decay") = 0.0f, py::arg("eps") = 1e-10f)
            .def("step", &optim::Adagrad::step)
            .def("zero_grad", &optim::Adagrad::zero_grad)
            .def("get_lr", &optim::Adagrad::get_lr)
            .def("set_lr", &optim::Adagrad::set_lr, py::arg("new_lr"))
            .def("add_param_group", &optim::Adagrad::add_param_group, py::arg("params"), py::arg("lr"), py::arg("weight_decay") = 0.0f)
            .def("param_groups", [](optim::Adagrad& o) {
                py::list groups;
                for (auto& g : o.param_groups) {
                    py::dict gd;
                    py::list ps;
                    for (auto& p : g.params) ps.append(py::cast(p));
                    gd["params"] = ps;
                    gd["lr"] = g.lr;
                    gd["weight_decay"] = g.weight_decay;
                    groups.append(gd);
                }
                return groups;
            })
            .def("state_dict", [](optim::Adagrad& o) {
                py::dict d;
                d["lr"] = o.lr;
                d["lr_decay"] = o.lr_decay;
                d["weight_decay"] = o.weight_decay;
                d["eps"] = o.eps;
                py::list states;
                size_t idx = 0;
                for (auto& g : o.param_groups) {
                    for (size_t i = 0; i < g.params.size(); ++i, ++idx) {
                        py::dict s;
                        s["step"] = o.step_count;
                        s["sum"] = py::cast(o.sum[idx]->clone());
                        states.append(s);
                    }
                }
                d["state"] = states;
                py::list groups;
                for (auto& g : o.param_groups) {
                    py::dict gd;
                    gd["lr"] = g.lr;
                    gd["weight_decay"] = g.weight_decay;
                    gd["num_params"] = static_cast<int>(g.params.size());
                    groups.append(gd);
                }
                d["param_groups"] = groups;
                return d;
            })
            .def("load_state_dict", [](optim::Adagrad& o, py::dict d) {
                py::list groups = d["param_groups"];
                if (groups.size() != static_cast<py::ssize_t>(o.param_groups.size()))
                    throw std::runtime_error("[litetorch Error] Adagrad.load_state_dict: param_groups size mismatch");
                for (py::ssize_t gi = 0; gi < groups.size(); ++gi) {
                    py::dict gd = groups[gi];
                    o.param_groups[gi].lr = gd["lr"].cast<double>();
                    o.param_groups[gi].weight_decay = gd["weight_decay"].cast<float>();
                }
                o.lr = o.param_groups[0].lr;
                o.weight_decay = o.param_groups[0].weight_decay;
                if (d.contains("lr_decay")) o.lr_decay = d["lr_decay"].cast<float>();
                if (d.contains("eps")) o.eps = d["eps"].cast<float>();
                py::list states = d["state"];
                if (states.size() != static_cast<py::ssize_t>(o.sum.size()))
                    throw std::runtime_error("[litetorch Error] Adagrad.load_state_dict: state size mismatch");
                for (py::ssize_t i = 0; i < states.size(); ++i) {
                    py::dict s = states[i];
                    o.step_count = s["step"].cast<int>();
                    auto src = py::cast<std::shared_ptr<Tensor>>(s["sum"]);
                    auto dst = o.sum[i];
                    if (src->numel() != dst->numel())
                        throw std::runtime_error("[litetorch Error] Adagrad.load_state_dict: state tensor size mismatch");
                    std::memcpy(dst->data_ptr(), src->data_ptr(), static_cast<size_t>(src->numel()) * sizeof(float));
                }
            }, py::arg("state_dict"))
            .def_readwrite("lr", &optim::Adagrad::lr)
            .def_readwrite("lr_decay", &optim::Adagrad::lr_decay)
            .def_readwrite("weight_decay", &optim::Adagrad::weight_decay)
            .def_readwrite("eps", &optim::Adagrad::eps);

        py::class_<optim::NAdam, optim::Optimizer, std::shared_ptr<optim::NAdam>>(optim_mod, "NAdam")
            .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, double, std::pair<double, double>, float, float, float>(), py::arg("params"), py::arg("lr") = 0.002, py::arg("betas") = std::pair<double, double>(0.9, 0.999), py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f, py::arg("momentum_decay") = 0.004f)
            .def("step", &optim::NAdam::step)
            .def("zero_grad", &optim::NAdam::zero_grad)
            .def("get_lr", &optim::NAdam::get_lr)
            .def("set_lr", &optim::NAdam::set_lr, py::arg("new_lr"))
            .def("add_param_group", &optim::NAdam::add_param_group, py::arg("params"), py::arg("lr"), py::arg("weight_decay") = 0.0f)
            .def("param_groups", [](optim::NAdam& o) {
                py::list groups;
                for (auto& g : o.param_groups) {
                    py::dict gd;
                    py::list ps;
                    for (auto& p : g.params) ps.append(py::cast(p));
                    gd["params"] = ps;
                    gd["lr"] = g.lr;
                    gd["weight_decay"] = g.weight_decay;
                    groups.append(gd);
                }
                return groups;
            })
            .def("state_dict", [](optim::NAdam& o) {
                py::dict d;
                d["lr"] = o.lr;
                d["betas"] = py::make_tuple(o.beta1, o.beta2);
                d["eps"] = o.eps;
                d["weight_decay"] = o.weight_decay;
                d["momentum_decay"] = o.momentum_decay;
                py::list states;
                size_t idx = 0;
                for (auto& g : o.param_groups) {
                    for (size_t i = 0; i < g.params.size(); ++i, ++idx) {
                        py::dict s;
                        s["step"] = o.param_steps[idx];
                        s["exp_avg"] = py::cast(o.exp_avg[idx]->clone());
                        s["exp_avg_sq"] = py::cast(o.exp_avg_sq[idx]->clone());
                        s["mu_product"] = o.mu_products[idx];
                        states.append(s);
                    }
                }
                d["state"] = states;
                py::list groups;
                for (auto& g : o.param_groups) {
                    py::dict gd;
                    gd["lr"] = g.lr;
                    gd["weight_decay"] = g.weight_decay;
                    gd["num_params"] = static_cast<int>(g.params.size());
                    groups.append(gd);
                }
                d["param_groups"] = groups;
                return d;
            })
            .def("load_state_dict", [](optim::NAdam& o, py::dict d) {
                py::list groups = d["param_groups"];
                if (groups.size() != static_cast<py::ssize_t>(o.param_groups.size()))
                    throw std::runtime_error("[litetorch Error] NAdam.load_state_dict: param_groups size mismatch");
                for (py::ssize_t gi = 0; gi < groups.size(); ++gi) {
                    py::dict gd = groups[gi];
                    o.param_groups[gi].lr = gd["lr"].cast<double>();
                    o.param_groups[gi].weight_decay = gd["weight_decay"].cast<float>();
                }
                o.lr = o.param_groups[0].lr;
                o.weight_decay = o.param_groups[0].weight_decay;
                if (d.contains("betas")) {
                    auto b = d["betas"].cast<std::pair<double, double>>();
                    o.beta1 = b.first;
                    o.beta2 = b.second;
                }
                if (d.contains("eps")) o.eps = d["eps"].cast<float>();
                if (d.contains("momentum_decay")) o.momentum_decay = d["momentum_decay"].cast<float>();
                py::list states = d["state"];
                if (states.size() != static_cast<py::ssize_t>(o.exp_avg.size()))
                    throw std::runtime_error("[litetorch Error] NAdam.load_state_dict: state size mismatch");
                for (py::ssize_t i = 0; i < states.size(); ++i) {
                    py::dict s = states[i];
                    o.param_steps[i] = s["step"].cast<int>();
                    if (o.param_steps[i] > o.step_count) o.step_count = o.param_steps[i];
                    if (s.contains("mu_product")) {
                        o.mu_products[i] = s["mu_product"].cast<float>();
                    } else {
                        float mp = 1.0f;
                        float b1 = static_cast<float>(o.beta1);
                        for (int t = 1; t <= o.param_steps[i]; ++t) {
                            mp *= b1 * (1.0f - 0.5f * std::pow(0.96f, static_cast<float>(t) * o.momentum_decay));
                        }
                        o.mu_products[i] = mp;
                    }
                    auto m_src = py::cast<std::shared_ptr<Tensor>>(s["exp_avg"]);
                    auto v_src = py::cast<std::shared_ptr<Tensor>>(s["exp_avg_sq"]);
                    auto m_dst = o.exp_avg[i];
                    auto v_dst = o.exp_avg_sq[i];
                    if (m_src->numel() != m_dst->numel() || v_src->numel() != v_dst->numel())
                        throw std::runtime_error("[litetorch Error] NAdam.load_state_dict: state tensor size mismatch");
                    std::memcpy(m_dst->data_ptr(), m_src->data_ptr(), static_cast<size_t>(m_src->numel()) * sizeof(float));
                    std::memcpy(v_dst->data_ptr(), v_src->data_ptr(), static_cast<size_t>(v_src->numel()) * sizeof(float));
                }
            }, py::arg("state_dict"))
            .def_readwrite("lr", &optim::NAdam::lr)
            .def_readwrite("beta1", &optim::NAdam::beta1)
            .def_readwrite("beta2", &optim::NAdam::beta2)
            .def_readwrite("eps", &optim::NAdam::eps)
            .def_readwrite("weight_decay", &optim::NAdam::weight_decay)
            .def_readwrite("momentum_decay", &optim::NAdam::momentum_decay);

        py::class_<optim::MultiStepLR>(optim_mod, "MultiStepLR")
            .def(py::init<optim::Optimizer*, std::vector<int>, float>(), py::keep_alive<1, 2>(), py::arg("optimizer"), py::arg("milestones"), py::arg("gamma") = 0.1f)
            .def("step", &optim::MultiStepLR::step)
            .def("get_last_lr", &optim::MultiStepLR::get_last_lr)
            .def("state_dict", [](optim::MultiStepLR& s) {
                py::dict d;
                for (auto& kv : s.state_dict()) d[kv.first.c_str()] = kv.second;
                py::list ms;
                for (int m : s.milestones) ms.append(m);
                d["milestones"] = ms;
                return d;
            })
            .def("load_state_dict", [](optim::MultiStepLR& s, py::dict d) {
                std::map<std::string, double> m;
                for (auto item : d) {
                    std::string k = py::cast<std::string>(item.first);
                    if (k == "milestones") continue;
                    m[k] = py::cast<double>(item.second);
                }
                s.load_state_dict(m);
                if (d.contains("milestones")) {
                    s.milestones.clear();
                    for (auto v : d["milestones"]) s.milestones.push_back(py::cast<int>(v));
                }
            }, py::arg("state_dict"))
            .def_readwrite("gamma", &optim::MultiStepLR::gamma)
            .def_readwrite("last_epoch", &optim::MultiStepLR::last_epoch);

        py::class_<optim::ExponentialLR>(optim_mod, "ExponentialLR")
            .def(py::init<optim::Optimizer*, float>(), py::keep_alive<1, 2>(), py::arg("optimizer"), py::arg("gamma"))
            .def("step", &optim::ExponentialLR::step)
            .def("get_last_lr", &optim::ExponentialLR::get_last_lr)
            .def("state_dict", [](optim::ExponentialLR& s) {
                py::dict d;
                for (auto& kv : s.state_dict()) d[kv.first.c_str()] = kv.second;
                return d;
            })
            .def("load_state_dict", [](optim::ExponentialLR& s, py::dict d) {
                std::map<std::string, double> m;
                for (auto item : d) {
                    m[py::cast<std::string>(item.first)] = py::cast<double>(item.second);
                }
                s.load_state_dict(m);
            }, py::arg("state_dict"))
            .def_readwrite("gamma", &optim::ExponentialLR::gamma)
            .def_readwrite("last_epoch", &optim::ExponentialLR::last_epoch);

        py::class_<optim::ReduceLROnPlateau>(optim_mod, "ReduceLROnPlateau")
            .def(py::init<optim::Optimizer*, std::string, float, int, float, std::string, int, float, float>(), py::keep_alive<1, 2>(), py::arg("optimizer"), py::arg("mode") = "min", py::arg("factor") = 0.1f, py::arg("patience") = 10, py::arg("threshold") = 1e-4f, py::arg("threshold_mode") = "rel", py::arg("cooldown") = 0, py::arg("min_lr") = 0.0f, py::arg("eps") = 1e-8f)
            .def("step", &optim::ReduceLROnPlateau::step, py::arg("metric"))
            .def("get_last_lr", &optim::ReduceLROnPlateau::get_last_lr)
            .def("state_dict", [](optim::ReduceLROnPlateau& s) {
                py::dict d;
                for (auto& kv : s.state_dict()) d[kv.first.c_str()] = kv.second;
                d["mode"] = s.mode;
                d["threshold_mode"] = s.threshold_mode;
                return d;
            })
            .def("load_state_dict", [](optim::ReduceLROnPlateau& s, py::dict d) {
                std::map<std::string, double> m;
                for (auto item : d) {
                    std::string k = py::cast<std::string>(item.first);
                    if (k == "mode" || k == "threshold_mode") continue;
                    m[k] = py::cast<double>(item.second);
                }
                s.load_state_dict(m);
                if (d.contains("mode")) s.mode = py::cast<std::string>(d["mode"]);
                if (d.contains("threshold_mode")) s.threshold_mode = py::cast<std::string>(d["threshold_mode"]);
            }, py::arg("state_dict"))
            .def_readwrite("patience", &optim::ReduceLROnPlateau::patience)
            .def_readwrite("last_epoch", &optim::ReduceLROnPlateau::last_epoch)
            .def_readwrite("mode", &optim::ReduceLROnPlateau::mode)
            .def_readwrite("best", &optim::ReduceLROnPlateau::best);

        lr_sched_mod.def("MultiStepLR", [](optim::Optimizer* opt, std::vector<int> milestones, float gamma) {
            return std::unique_ptr<optim::MultiStepLR>(new optim::MultiStepLR(opt, milestones, gamma));
        }, py::keep_alive<0, 1>(), py::arg("optimizer"), py::arg("milestones"), py::arg("gamma") = 0.1f);
        lr_sched_mod.def("ExponentialLR", [](optim::Optimizer* opt, float gamma) {
            return std::unique_ptr<optim::ExponentialLR>(new optim::ExponentialLR(opt, gamma));
        }, py::keep_alive<0, 1>(), py::arg("optimizer"), py::arg("gamma"));
        lr_sched_mod.def("ReduceLROnPlateau", [](optim::Optimizer* opt, std::string mode, float factor, int patience, float threshold, std::string threshold_mode, int cooldown, float min_lr, float eps) {
            return std::unique_ptr<optim::ReduceLROnPlateau>(new optim::ReduceLROnPlateau(opt, mode, factor, patience, threshold, threshold_mode, cooldown, min_lr, eps));
        }, py::keep_alive<0, 1>(), py::arg("optimizer"), py::arg("mode") = "min", py::arg("factor") = 0.1f, py::arg("patience") = 10, py::arg("threshold") = 1e-4f, py::arg("threshold_mode") = "rel", py::arg("cooldown") = 0, py::arg("min_lr") = 0.0f, py::arg("eps") = 1e-8f);
        lr_sched_mod.def("LambdaLR", [](optim::Optimizer* opt, py::function fn) {
            std::function<double(int)> cpp_fn = [fn](int epoch) -> double {
                py::gil_scoped_acquire gil;
                return fn(epoch).cast<double>();
            };
            return std::unique_ptr<optim::LambdaLR>(new optim::LambdaLR(opt, cpp_fn));
        }, py::keep_alive<0, 1>(), py::arg("optimizer"), py::arg("lr_lambda"));
        lr_sched_mod.def("LinearLR", [](optim::Optimizer* opt, float start_factor, float end_factor, int total_iters) {
            return std::unique_ptr<optim::LinearLR>(new optim::LinearLR(opt, start_factor, end_factor, total_iters));
        }, py::keep_alive<0, 1>(), py::arg("optimizer"), py::arg("start_factor") = 1.0f / 3, py::arg("end_factor") = 1.0f, py::arg("total_iters") = 5);
        lr_sched_mod.def("PolynomialLR", [](optim::Optimizer* opt, int total_iters, float power) {
            return std::unique_ptr<optim::PolynomialLR>(new optim::PolynomialLR(opt, total_iters, power));
        }, py::keep_alive<0, 1>(), py::arg("optimizer"), py::arg("total_iters") = 5, py::arg("power") = 1.0f);
        lr_sched_mod.def("CosineAnnealingWarmRestarts", [](optim::Optimizer* opt, int T_0, int T_mult, float eta_min) {
            return std::unique_ptr<optim::CosineAnnealingWarmRestarts>(new optim::CosineAnnealingWarmRestarts(opt, T_0, T_mult, eta_min));
        }, py::keep_alive<0, 1>(), py::arg("optimizer"), py::arg("T_0"), py::arg("T_mult") = 1, py::arg("eta_min") = 0.0f);
        lr_sched_mod.def("OneCycleLR", [](optim::Optimizer* opt, double max_lr, int total_steps, float pct_start, std::string anneal_strategy, float div_factor, float final_div_factor) {
            return std::unique_ptr<optim::OneCycleLR>(new optim::OneCycleLR(opt, max_lr, total_steps, -1, -1, pct_start, anneal_strategy, div_factor, final_div_factor));
        }, py::keep_alive<0, 1>(), py::arg("optimizer"), py::arg("max_lr"), py::arg("total_steps"), py::arg("pct_start") = 0.3f, py::arg("anneal_strategy") = "cos", py::arg("div_factor") = 25.0f, py::arg("final_div_factor") = 10000.0f);

        py::class_<optim::GradScaler>(optim_mod, "GradScaler")
            .def(py::init<float, float, float, int, bool>(), py::arg("init_scale") = 65536.0f, py::arg("growth_factor") = 2.0f, py::arg("backoff_factor") = 0.5f, py::arg("growth_interval") = 2000, py::arg("enabled") = true)
            .def("scale", &optim::GradScaler::scale, py::arg("loss"))
            .def("scale_loss", &optim::GradScaler::scale_loss, py::arg("loss"))
            .def("unscale_", &optim::GradScaler::unscale_, py::arg("optimizer"))
            .def("step", &optim::GradScaler::step, py::arg("optimizer"))
            .def("update", &optim::GradScaler::update)
            .def("get_scale", &optim::GradScaler::get_scale)
            .def("set_scale", &optim::GradScaler::set_scale, py::arg("scale"))
            .def("is_enabled", &optim::GradScaler::is_enabled);
}
