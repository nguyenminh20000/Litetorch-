import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

scheduler = lt.nn.ServingScheduler(
    num_blocks=1024,
    block_size=16,
    max_num_seqs=64,
    max_num_batched_tokens=2048
)

req1 = lt.nn.Request(id=1, prompt=[101, 2054, 2003, 102], max_new_tokens=64)
req2 = lt.nn.Request(id=2, prompt=[101, 7592, 102], max_new_tokens=32)

scheduler.add_request(req1)
scheduler.add_request(req2)

active_requests = scheduler.step()
block_table_tensor = scheduler.get_block_table_tensor()
context_lens_tensor = scheduler.get_context_lens_tensor()
