# Security Policy

## Supported Versions

Only the latest minor release receives security updates (currently 0.3.x).
Older versions are not patched; upgrade before reporting.

## Reporting a Vulnerability

Email: security@litetorch.dev (placeholder — set a real address before first release).
PGP key: to be published.

Do NOT open a public issue for a vulnerability.
Expect an acknowledgement within 72 hours and a fix timeline within 14 days.

## Scope

- Memory safety in C++ core, backends (CUDA/ROCm/OpenCL/CPU) and autograd.
- Deserialization: checkpoint loading, ONNX export/import.
- Python binding boundary (pybind11): type/dtype/device mismatches must fail closed, never corrupt.
- Distributed transports (socket/SHM/NCCL): malformed peer data must throw, never deadlock or corrupt memory.

## Out of Scope

- Third-party dependencies (report upstream).
- Physical access, social engineering.
- Benchmarks or performance claims.

## Known Security-Relevant Hardening

- ONNX export uses a hand-rolled wire writer; no libprotobuf linkage (verified 2026-10-17, qa-p8 packaging audit).
- CUDA/cuDNN symbols are loaded dynamically at runtime, never statically linked.
