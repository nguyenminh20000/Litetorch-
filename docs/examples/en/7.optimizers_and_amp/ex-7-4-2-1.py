import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
scheduler = lt.optim.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
