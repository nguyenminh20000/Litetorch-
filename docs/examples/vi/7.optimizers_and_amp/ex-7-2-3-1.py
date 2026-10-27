import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
optimizer = lt.optim.AdamW(model.parameters(), lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, offload_to_cpu=False)
