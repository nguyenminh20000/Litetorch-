#include "bindings_common.h"
#include "litetorch/dist_checkpoint.h"
#include "litetorch/flat_param.h"

void bind_distributed(py::module_& m) {
        auto dist_mod = m.def_submodule("distributed", "LiteTorch Distributed Modules");

        // === DT-A2 (LT-ENG-019): TP group init helper ===
        dist_mod.def("init_tp_groups", [](int tp_size) {
            auto& pg = distributed::ProcessGroup::get();
            if (!pg.is_initialized())
                throw std::runtime_error("[litetorch Error] init_tp_groups: init_process_group first");
            distributed::NCCLBridge::get().init_groups(pg.get_rank(), pg.get_world_size(), tp_size, 1);
        }, py::arg("tp_size"));
        // === end DT-A2 ===

        py::class_<distributed::ProcessGroup>(dist_mod, "ProcessGroup")
            .def_static("get", &distributed::ProcessGroup::get, py::return_value_policy::reference)
            .def("init", &distributed::ProcessGroup::init, py::arg("rank"), py::arg("world_size"), py::arg("master_addr"), py::arg("master_port"))
            .def("shutdown", &distributed::ProcessGroup::shutdown)
            .def("all_reduce", py::overload_cast<std::shared_ptr<Tensor>>(&distributed::ProcessGroup::all_reduce), py::arg("tensor"))
            .def("all_reduce", py::overload_cast<std::shared_ptr<Tensor>, distributed::ReduceOp>(&distributed::ProcessGroup::all_reduce), py::arg("tensor"), py::arg("op"))
            .def("barrier", &distributed::ProcessGroup::barrier)
            .def("all_to_all", &distributed::ProcessGroup::all_to_all, py::arg("input"), py::arg("output"))
            .def("scatter", &distributed::ProcessGroup::scatter, py::arg("output"), py::arg("input"), py::arg("src"))
            .def("gather", &distributed::ProcessGroup::gather, py::arg("input"), py::arg("output"), py::arg("dst"))
            .def("broadcast", &distributed::ProcessGroup::broadcast, py::arg("tensor"), py::arg("src"))
            .def("all_gather", &distributed::ProcessGroup::all_gather, py::arg("shard"), py::arg("full"))
            .def("reduce_scatter", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&distributed::ProcessGroup::reduce_scatter), py::arg("shard"), py::arg("full"))
            .def("reduce_scatter", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, distributed::ReduceOp>(&distributed::ProcessGroup::reduce_scatter), py::arg("shard"), py::arg("full"), py::arg("op"))
            .def("send_tensor", &distributed::ProcessGroup::send_tensor, py::arg("tensor"), py::arg("dst"))
            .def("recv_tensor", &distributed::ProcessGroup::recv_tensor, py::arg("tensor"), py::arg("src"))
            .def("sync_comm", &distributed::ProcessGroup::sync_comm)
            .def("get_rank", &distributed::ProcessGroup::get_rank)
            .def("get_world_size", &distributed::ProcessGroup::get_world_size)
            .def("is_initialized", &distributed::ProcessGroup::is_initialized);

        py::class_<distributed::DeviceMesh, std::shared_ptr<distributed::DeviceMesh>>(dist_mod, "DeviceMesh")
            .def(py::init<const std::vector<int>&, const std::string&>(), py::arg("shape"), py::arg("dim_names"))
            .def("get_rank", &distributed::DeviceMesh::get_rank, py::arg("coords"))
            .def_readwrite("mesh_shape", &distributed::DeviceMesh::mesh_shape)
            .def_readwrite("mesh_dim_names", &distributed::DeviceMesh::mesh_dim_names)
            .def_readwrite("mesh_topology", &distributed::DeviceMesh::mesh_topology);

        py::enum_<distributed::PlacementType>(dist_mod, "PlacementType")
            .value("SHARD", distributed::PlacementType::SHARD)
            .value("REPLICATE", distributed::PlacementType::REPLICATE)
            .value("PARTIAL", distributed::PlacementType::PARTIAL)
            .export_values();

        py::class_<distributed::Placement>(dist_mod, "Placement")
            .def(py::init<>())
            .def_readwrite("type", &distributed::Placement::type)
            .def_readwrite("dim", &distributed::Placement::dim);

        py::class_<distributed::DTensor, std::shared_ptr<distributed::DTensor>>(dist_mod, "DTensor")
            .def(py::init<std::shared_ptr<Tensor>, std::shared_ptr<distributed::DeviceMesh>, const std::vector<distributed::Placement>&>(),
                 py::arg("local_tensor"), py::arg("mesh"), py::arg("placements"))
            .def("redistribute", &distributed::DTensor::redistribute, py::arg("new_mesh"), py::arg("new_placements"))
            .def_readwrite("device_mesh", &distributed::DTensor::device_mesh)
            .def_readwrite("placements", &distributed::DTensor::placements)
            .def_readwrite("local_tensor", &distributed::DTensor::local_tensor);

        py::class_<distributed::OverlappedAllReducer>(dist_mod, "OverlappedAllReducer")
            .def_static("get", &distributed::OverlappedAllReducer::get, py::return_value_policy::reference)
            .def("push_and_all_reduce", &distributed::OverlappedAllReducer::push_and_all_reduce, py::arg("tensor"))
            .def("push_and_reduce_scatter", &distributed::OverlappedAllReducer::push_and_reduce_scatter, py::arg("shard"), py::arg("full"))
            .def("sync", &distributed::OverlappedAllReducer::sync);

        dist_mod.def("overlapped_all_reduce", &distributed::overlapped_all_reduce, py::arg("tensor"));
        dist_mod.def("overlapped_reduce_scatter", &distributed::overlapped_reduce_scatter, py::arg("shard"), py::arg("full"));
        dist_mod.def("sync_overlapped_all_reduce", &distributed::sync_overlapped_all_reduce);

        py::class_<distributed::PipelineParallelModule, nn::Module, std::shared_ptr<distributed::PipelineParallelModule>>(dist_mod, "PipelineParallelModule")
            .def(py::init<std::shared_ptr<nn::Module>>(), py::arg("sub_module"))
            .def("forward", &distributed::PipelineParallelModule::forward, py::arg("input"))
            .def("forward_microbatches", &distributed::PipelineParallelModule::forward_microbatches, py::arg("microbatches"))
            .def("schedule_1f1b", &distributed::PipelineParallelModule::schedule_1f1b, py::arg("microbatches"))
            .def("train_step_1f1b", &distributed::PipelineParallelModule::train_step_1f1b, py::arg("microbatches"), py::arg("loss_fn") = nullptr)
            .def("run_1f1b_with_backward", &distributed::PipelineParallelModule::run_1f1b_with_backward, py::arg("microbatches"), py::arg("loss_fn"))
            .def("parameters", &distributed::PipelineParallelModule::parameters)
            .def("to", &distributed::PipelineParallelModule::to, py::arg("device"));

        py::enum_<distributed::ReduceOp>(dist_mod, "ReduceOp")
            .value("SUM", distributed::ReduceOp::SUM)
            .value("AVG", distributed::ReduceOp::AVG)
            .value("PRODUCT", distributed::ReduceOp::PRODUCT)
            .value("MIN", distributed::ReduceOp::MIN)
            .value("MAX", distributed::ReduceOp::MAX)
            .export_values();

        dist_mod.def("init_process_group", &distributed::init_process_group,
                     py::arg("rank"), py::arg("world_size"),
                     py::arg("master_addr") = "127.0.0.1", py::arg("master_port") = 29500);
        dist_mod.def("shutdown", &distributed::shutdown);
        dist_mod.def("barrier", &distributed::barrier);
        dist_mod.def("all_reduce",
                     py::overload_cast<std::shared_ptr<Tensor>>(&distributed::all_reduce),
                     py::arg("tensor"));
        dist_mod.def("all_reduce",
                     py::overload_cast<std::shared_ptr<Tensor>, distributed::ReduceOp>(&distributed::all_reduce),
                     py::arg("tensor"), py::arg("op") = distributed::ReduceOp::SUM);
        dist_mod.def("all_gather", &distributed::all_gather, py::arg("shard"), py::arg("full"));
        dist_mod.def("reduce_scatter",
                     py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&distributed::reduce_scatter),
                     py::arg("shard"), py::arg("full"));
        dist_mod.def("reduce_scatter",
                     py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, distributed::ReduceOp>(&distributed::reduce_scatter),
                     py::arg("shard"), py::arg("full"), py::arg("op"));
        dist_mod.def("broadcast", &distributed::broadcast, py::arg("tensor"), py::arg("src_rank"));
        dist_mod.def("all_to_all", &distributed::all_to_all, py::arg("input"), py::arg("output"));
        dist_mod.def("scatter", &distributed::scatter, py::arg("output"), py::arg("input"), py::arg("src"));
        dist_mod.def("gather", &distributed::gather, py::arg("input"), py::arg("output"), py::arg("dst"));
        dist_mod.def("send_tensor", &distributed::send_tensor, py::arg("tensor"), py::arg("dst"));
        dist_mod.def("recv_tensor", &distributed::recv_tensor, py::arg("tensor"), py::arg("src"));

        py::class_<distributed::NCCLBridge, std::unique_ptr<distributed::NCCLBridge, py::nodelete>>(dist_mod, "NCCLBridge")
            .def_static("get", &distributed::NCCLBridge::get, py::return_value_policy::reference)
            .def("is_available", &distributed::NCCLBridge::is_available)
            .def("init", &distributed::NCCLBridge::init, py::arg("rank"), py::arg("world_size"))
            .def("shutdown", &distributed::NCCLBridge::shutdown)
            .def("sync_comm", &distributed::NCCLBridge::sync_comm);

        py::class_<distributed::DistributedDataParallel, nn::Module, std::shared_ptr<distributed::DistributedDataParallel>>(dist_mod, "DistributedDataParallel")
            .def(py::init<std::shared_ptr<nn::Module>, distributed::ReduceOp>(), py::arg("module"), py::arg("op") = distributed::ReduceOp::AVG)
            .def("forward", &distributed::DistributedDataParallel::forward, py::arg("input"))
            .def("backward", py::overload_cast<std::shared_ptr<Tensor>>(&distributed::DistributedDataParallel::backward), py::arg("loss"))
            .def("backward", py::overload_cast<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(&distributed::DistributedDataParallel::backward), py::arg("loss"), py::arg("grad_output"))
            .def("sync_grads", &distributed::DistributedDataParallel::sync_grads)
            .def("parameters", &distributed::DistributedDataParallel::parameters)
            .def("to", &distributed::DistributedDataParallel::to, py::arg("device"))
            .def_readwrite("reduce_op", &distributed::DistributedDataParallel::reduce_op);

        // === DT-A1 (LT-ENG-018): tensor-parallel autograd collectives ===
        dist_mod.def("tp_all_reduce", &distributed::tp_all_reduce_autograd, py::arg("tensor"));
        dist_mod.def("tp_all_gather", &distributed::tp_all_gather_autograd, py::arg("shard"));
        // === end DT-A1 ===

        // === Phase D1 (LT-ENG-033): sharded distributed checkpoint ===
        py::class_<dist_checkpoint::TensorShardInfo>(dist_mod, "TensorShardInfo")
            .def(py::init<>())
            .def_readwrite("global_shape", &dist_checkpoint::TensorShardInfo::global_shape)
            .def_readwrite("dtype_str", &dist_checkpoint::TensorShardInfo::dtype_str)
            .def_readwrite("shard_dim", &dist_checkpoint::TensorShardInfo::shard_dim)
            .def_readwrite("shard_ranges", &dist_checkpoint::TensorShardInfo::shard_ranges);
        py::class_<dist_checkpoint::CheckpointMetadata>(dist_mod, "CheckpointMetadata")
            .def(py::init<>())
            .def_readwrite("version", &dist_checkpoint::CheckpointMetadata::version)
            .def_readwrite("world_size", &dist_checkpoint::CheckpointMetadata::world_size)
            .def_readwrite("step", &dist_checkpoint::CheckpointMetadata::step)
            .def_readwrite("timestamp", &dist_checkpoint::CheckpointMetadata::timestamp)
            .def_readwrite("tensors", &dist_checkpoint::CheckpointMetadata::tensors);
        dist_mod.def("make_sharded_spec", &dist_checkpoint::make_sharded_spec,
            py::arg("global_shape"), py::arg("dtype_str"), py::arg("shard_dim"), py::arg("world_size"));
        dist_mod.def("make_replicated_spec", &dist_checkpoint::make_replicated_spec,
            py::arg("global_shape"), py::arg("dtype_str"));
        dist_mod.def("save_distributed", [](py::dict local_shards, py::dict shard_specs,
                                            const std::string& ckpt_dir, int64_t step) {
            std::map<std::string, std::shared_ptr<Tensor>> shards;
            for (auto kv : local_shards)
                shards[kv.first.cast<std::string>()] = kv.second.cast<std::shared_ptr<Tensor>>();
            std::map<std::string, dist_checkpoint::TensorShardInfo> specs;
            for (auto kv : shard_specs)
                specs[kv.first.cast<std::string>()] = kv.second.cast<dist_checkpoint::TensorShardInfo>();
            dist_checkpoint::save_distributed(shards, specs, ckpt_dir, step);
        }, py::arg("local_shards"), py::arg("shard_specs"), py::arg("ckpt_dir"), py::arg("step"));
        dist_mod.def("load_distributed", [](py::dict local_shards, const std::string& ckpt_dir) {
            std::map<std::string, std::shared_ptr<Tensor>> shards;
            for (auto kv : local_shards)
                shards[kv.first.cast<std::string>()] = kv.second.cast<std::shared_ptr<Tensor>>();
            dist_checkpoint::load_distributed(shards, ckpt_dir);
        }, py::arg("local_shards"), py::arg("ckpt_dir"));
        dist_mod.def("read_dist_metadata", &dist_checkpoint::read_metadata, py::arg("ckpt_dir"));
        // === end Phase D1 ===

        py::class_<distributed::FlatParameter, std::shared_ptr<distributed::FlatParameter>>(dist_mod, "FlatParameter")
            .def(py::init<const std::vector<std::shared_ptr<Tensor>>&>(), py::arg("params"))
            .def("flat_numel", &distributed::FlatParameter::flat_numel)
            .def("shard_numel", &distributed::FlatParameter::shard_numel)
            .def("unpadded_numel", &distributed::FlatParameter::unpadded_numel)
            .def("world_size", &distributed::FlatParameter::world_size)
            .def("rank", &distributed::FlatParameter::rank)
            .def("local_shard", &distributed::FlatParameter::local_shard)
            .def("full_buffer", &distributed::FlatParameter::full_buffer)
            .def("param_views", &distributed::FlatParameter::param_views)
            .def("all_gather_flat", &distributed::FlatParameter::all_gather_flat)
            .def("reduce_scatter_flat", &distributed::FlatParameter::reduce_scatter_flat, py::arg("full_grad"))
            .def("metadata", [](const distributed::FlatParameter& fp) {
                py::list out;
                for (const auto& m : fp.metadata()) {
                    py::dict d;
                    d["offset"] = m.offset;
                    d["numel"] = m.numel;
                    d["shape"] = m.shape;
                    out.append(d);
                }
                return out;
            });
        dist_mod.def("make_flat_param", &distributed::make_flat_param, py::arg("params"));
}
