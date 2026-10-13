#!/usr/bin/env bash
# Build tools/lt-serve against the compiled litetorch objects.
# Requires a completed `python3 setup.py build_ext --inplace` first.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OBJDIR="$ROOT/build/temp.linux-x86_64-cpython-312"
if [ ! -d "$OBJDIR" ]; then
    echo "lt-serve build: object dir $OBJDIR not found; run setup.py build_ext first" >&2
    exit 1
fi
OBJS=$(find "$OBJDIR" -name "*.o" | grep -v "/src/bindings/" | tr '\n' ' ')
PYLD=$(python3-config --ldflags --embed)
g++ -std=c++14 -O1 -I"$ROOT/include" -I"$ROOT/tools" -I"$ROOT/src" \
    "$ROOT/tools/lt-serve.cpp" $OBJS -o "$ROOT/tools/lt-serve" -lpthread $PYLD
echo "built $ROOT/tools/lt-serve"
