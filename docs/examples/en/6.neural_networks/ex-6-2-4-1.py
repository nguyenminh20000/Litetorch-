import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
col_linear = lt.nn.ColumnParallelLinear(in_features=4096, out_features=2048, has_bias=True)
row_linear = lt.nn.RowParallelLinear(in_features=2048, out_features=4096, has_bias=True)
