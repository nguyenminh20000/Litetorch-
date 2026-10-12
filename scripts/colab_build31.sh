#!/bin/bash
set -u
cd /tmp/lt-colab
git fetch origin -q && git reset --hard origin/agent -q
echo "HEAD=$(git log --oneline -1)"
pip install -q pybind11 2>&1 | tail -1
rm -rf build/
export LITETORCH_CUDA=1
python3 setup.py build_ext --inplace > /tmp/build31c.log 2>&1
echo "rc=$?"
ls -la /tmp/lt-colab/python/litetorch.cpython-3*-x86_64-linux-gnu.so 2>/dev/null | awk '{print "SO:", $6, $7, $8, $5}'
export PYTHONPATH=/tmp/lt-colab
echo "=== FP16 MFU ==="
timeout 600 python3 tests/bench_gptblock_mfu.py --fp16 2>&1 | tail -8
echo "=== gptblock ==="
timeout 300 python3 tests/test_gptblock.py 2>&1 | tail -4
echo "=== chunk ==="
timeout 300 python3 tests/test_chunk.py 2>&1 | tail -4
echo "=== grad ==="
timeout 600 python3 tests/test_grad_comprehensive.py 2>&1 | tail -3
echo "DONE31"
