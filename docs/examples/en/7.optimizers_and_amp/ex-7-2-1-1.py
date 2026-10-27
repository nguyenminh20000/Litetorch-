import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
optimizer = lt.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
