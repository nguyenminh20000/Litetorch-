import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

mesh = lt.distributed.DeviceMesh([2, 2], "tp,dp")

placement_shard = lt.distributed.Placement()
placement_shard.type = lt.distributed.PlacementType.SHARD
placement_shard.dim = 0

local_data = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], lt.auto_device(), False)
dtensor = lt.distributed.DTensor(local_data, mesh, [placement_shard])
