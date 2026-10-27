import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

pg = lt.distributed.ProcessGroup.get()
pg.init(rank=0, world_size=4, master_addr="127.0.0.1", master_port=29500)

rank = pg.get_rank()
world_size = pg.get_world_size()

print(f"Khoi tao rank {rank} tren tong so {world_size} tien trinh")
