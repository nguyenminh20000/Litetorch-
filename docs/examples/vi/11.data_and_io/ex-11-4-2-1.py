import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
optimizer = lt.optim.AdamW(model.parameters(), lr=1e-3)

lt.save(optimizer, "optimizer.lt")
lt.load(optimizer, "optimizer.lt")
