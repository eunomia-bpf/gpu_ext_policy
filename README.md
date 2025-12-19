# gpu_ext

Extending Linux GPU drivers with eBPF for programmable memory offloading and scheduling.

## Overview

Modern GPU workloads (LLM inference, vector databases, DNN training) exhibit diverse memory access patterns and scheduling requirements. However, GPU drivers use fixed, one-size-fits-all policies that cannot adapt to workload-specific needs.

**gpu_ext** enables customizable GPU resource management through eBPF struct_ops:

- **Memory Management**: Pluggable eviction and prefetch policies at the driver level
- **Scheduling**: Per-process timeslice and priority control for multi-tenant GPU sharing
- **Observability**: Tracing tools for memory and scheduling events

Inspired by Linux kernel's `sched_ext`, gpu_ext brings the same extensibility to GPU drivers.

## Structure

```
├── src/              # eBPF policies and userspace loaders
├── kernel-module/    # Modified GPU kernel modules with eBPF hooks (NVIDIA, AMD, etc.)
├── libbpf/           # libbpf submodule
├── bpftool/          # bpftool submodule
├── vmlinux/          # vmlinux headers
├── docs/             # Documentation
└── microbench/       # Microbenchmarks
```

**Policies in src/**:
- Memory eviction policies (FIFO, LFU, MRU, PID-based, etc.)
- Prefetch policies (sequential, stride, adaptive, etc.)
- Scheduling policies (timeslice, priority)
- Tracing tools

## Build

```sh
# Install dependencies (Ubuntu)
make install

# Build all policies
make build
```

## Related

- [bpftime](https://github.com/eunomia-bpf/bpftime) - GPU device-side eBPF support

## License

MIT
