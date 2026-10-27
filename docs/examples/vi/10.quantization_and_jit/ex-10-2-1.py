import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()
calibrator = lt.quantization.Calibrator()

sample_act = lt.Tensor.from_vector([0.1, -0.5, 2.3, -1.8], [1, 4], device, False)
calibrator.collect("layer1_act", sample_act)

scale = calibrator.get_scale("layer1_act", bits=8)
params = calibrator.get_asymmetric_params("layer1_act", bits=8)
