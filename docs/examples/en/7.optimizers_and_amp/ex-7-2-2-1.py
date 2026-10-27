import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
optimizer = lt.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-8, weight_decay=0.0)
