import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
moe = lt.nn.MoELinear(in_features=1024, out_features=1024, num_experts=8, top_k=2)
