import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()

stage_module = lt.nn.Sequential([
    lt.nn.Linear(1024, 1024, True),
    lt.nn.GELU()
])
stage_module.to(device)

pp_module = lt.distributed.PipelineParallelModule(stage_module)

mb1 = lt.Tensor.from_vector([1.0] * 1024, [1, 1024], device, False)
mb2 = lt.Tensor.from_vector([2.0] * 1024, [1, 1024], device, False)
microbatches = [mb1, mb2]

outputs = pp_module.schedule_1f1b(microbatches)
