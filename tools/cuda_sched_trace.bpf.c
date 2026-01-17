// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2025 */
/*
 * CUDA + Scheduler Trace - Trace CUDA operations with CPU scheduler impact
 *
 * Strategy:
 * 1. Track all CUDA launch events to identify GPU processes
 * 2. Only trace sched_switch for identified GPU processes
 * 3. Track sync operations for those processes
 * 4. Output raw events for userspace analysis
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "cuda_launch_trace_event.h"

char LICENSE[] SEC("license") = "GPL";

// Track which PIDs are using GPU (set by launch events)
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u32);    // PID
    __type(value, __u8);   // 1 if GPU process
} gpu_pids SEC(".maps");

// Ring buffer for all events
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1024 * 1024);  // 1MB
} events SEC(".maps");

// Statistics
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 10);
    __type(key, __u32);
    __type(value, __u64);
} stats SEC(".maps");

#define STAT_CULAUNCH 0
#define STAT_CUDALAUNCH 1
#define STAT_SYNC_ENTER 2
#define STAT_SYNC_EXIT 3
#define STAT_SCHED_SWITCH 4
#define STAT_DROPPED 5
#define STAT_HARDIRQ 6
#define STAT_SOFTIRQ 7

// Per-CPU storage for IRQ entry timestamps
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u64);
} irq_start SEC(".maps");

static __always_inline void inc_stat(__u32 key)
{
    __u64 *val = bpf_map_lookup_elem(&stats, &key);
    if (val)
        __sync_fetch_and_add(val, 1);
}

static __always_inline void mark_gpu_process(__u32 pid)
{
    __u8 one = 1;
    bpf_map_update_elem(&gpu_pids, &pid, &one, BPF_ANY);
}

static __always_inline bool is_gpu_process(__u32 pid)
{
    return bpf_map_lookup_elem(&gpu_pids, &pid) != NULL;
}

/*
 * Tracepoint: sched_switch
 * Only trace if the process is a GPU process
 */
SEC("tp_btf/sched_switch")
int BPF_PROG(sched_switch, bool preempt, struct task_struct *prev, struct task_struct *next)
{
    __u32 prev_pid = BPF_CORE_READ(prev, pid);
    __u32 next_pid = BPF_CORE_READ(next, pid);
    __u64 now = bpf_ktime_get_ns();

    // Only track GPU processes being switched
    if (prev_pid && is_gpu_process(prev_pid)) {
        struct cuda_launch_event *e;

        e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
        if (!e) {
            inc_stat(STAT_DROPPED);
            return 0;
        }

        e->timestamp_ns = now;
        e->pid = prev_pid;
        e->tid = prev_pid;
        e->cpu_id = bpf_get_smp_processor_id();
        bpf_probe_read_kernel_str(&e->comm, sizeof(e->comm), BPF_CORE_READ(prev, comm));
        e->hook_type = HOOK_SCHED_SWITCH;

        // Mark as OFF-CPU (switched out)
        e->last_offcpu_ns = now;
        e->last_oncpu_ns = 0;

        bpf_ringbuf_submit(e, 0);
        inc_stat(STAT_SCHED_SWITCH);
    }

    // Track GPU processes being switched back ON
    if (next_pid && is_gpu_process(next_pid)) {
        struct cuda_launch_event *e;

        e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
        if (!e) {
            inc_stat(STAT_DROPPED);
            return 0;
        }

        e->timestamp_ns = now;
        e->pid = next_pid;
        e->tid = next_pid;
        e->cpu_id = bpf_get_smp_processor_id();
        bpf_probe_read_kernel_str(&e->comm, sizeof(e->comm), BPF_CORE_READ(next, comm));
        e->hook_type = HOOK_SCHED_SWITCH;

        // Mark as ON-CPU (switched in)
        e->last_offcpu_ns = 0;
        e->last_oncpu_ns = now;

        bpf_ringbuf_submit(e, 0);
        inc_stat(STAT_SCHED_SWITCH);
    }

    return 0;
}

/*
 * Uprobe: cuLaunchKernel (CUDA Driver API)
 */
SEC("uprobe/cuLaunchKernel")
int trace_cuLaunchKernel(struct pt_regs *ctx)
{
    struct cuda_launch_event *e;
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;

    // Mark this PID as GPU process
    mark_gpu_process(pid);
    inc_stat(STAT_CULAUNCH);

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return 0;
    }

    e->timestamp_ns = bpf_ktime_get_ns();
    e->pid = pid;
    e->tid = (__u32)pid_tgid;
    e->cpu_id = bpf_get_smp_processor_id();
    bpf_get_current_comm(&e->comm, sizeof(e->comm));
    e->hook_type = HOOK_CULAUNCHKERNEL;

    // Read CUDA parameters
    e->function = PT_REGS_PARM1(ctx);
    e->grid_dim_x = (__u32)PT_REGS_PARM2(ctx);
    e->grid_dim_y = (__u32)PT_REGS_PARM3(ctx);
    e->grid_dim_z = (__u32)PT_REGS_PARM4(ctx);
    e->block_dim_x = (__u32)PT_REGS_PARM5(ctx);
    e->block_dim_y = (__u32)PT_REGS_PARM6(ctx);

    void *sp = (void *)PT_REGS_SP(ctx);
    bpf_probe_read_user(&e->block_dim_z, sizeof(e->block_dim_z), sp + 8);
    bpf_probe_read_user(&e->shared_mem_bytes, sizeof(e->shared_mem_bytes), sp + 16);
    bpf_probe_read_user(&e->stream, sizeof(e->stream), sp + 24);

    e->last_offcpu_ns = 0;
    e->last_oncpu_ns = 0;
    e->total_offcpu_ns = 0;
    e->switch_count = 0;

    bpf_ringbuf_submit(e, 0);
    return 0;
}

/*
 * Uprobe: cudaLaunchKernel (CUDA Runtime API)
 */
SEC("uprobe/cudaLaunchKernel")
int trace_cudaLaunchKernel(struct pt_regs *ctx)
{
    struct cuda_launch_event *e;
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;

    // Mark this PID as GPU process
    mark_gpu_process(pid);
    inc_stat(STAT_CUDALAUNCH);

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return 0;
    }

    e->timestamp_ns = bpf_ktime_get_ns();
    e->pid = pid;
    e->tid = (__u32)pid_tgid;
    e->cpu_id = bpf_get_smp_processor_id();
    bpf_get_current_comm(&e->comm, sizeof(e->comm));
    e->hook_type = HOOK_CUDALAUNCHKERNEL;

    // Read CUDA parameters
    e->function = PT_REGS_PARM1(ctx);

    __u64 grid_packed = PT_REGS_PARM2(ctx);
    e->grid_dim_x = (__u32)(grid_packed & 0xFFFFFFFF);
    e->grid_dim_y = (__u32)(grid_packed >> 32);
    e->grid_dim_z = (__u32)PT_REGS_PARM3(ctx);

    __u64 block_packed = PT_REGS_PARM4(ctx);
    e->block_dim_x = (__u32)(block_packed & 0xFFFFFFFF);
    e->block_dim_y = (__u32)(block_packed >> 32);
    e->block_dim_z = (__u32)PT_REGS_PARM5(ctx);

    void *sp = (void *)PT_REGS_SP(ctx);
    bpf_probe_read_user(&e->shared_mem_bytes, sizeof(e->shared_mem_bytes), sp + 8);
    bpf_probe_read_user(&e->stream, sizeof(e->stream), sp + 16);

    e->last_offcpu_ns = 0;
    e->last_oncpu_ns = 0;
    e->total_offcpu_ns = 0;
    e->switch_count = 0;

    bpf_ringbuf_submit(e, 0);
    return 0;
}

/*
 * Uprobe: cudaDeviceSynchronize - Entry
 */
SEC("uprobe/cudaDeviceSynchronize")
int trace_cudaDeviceSynchronize_enter(struct pt_regs *ctx)
{
    struct sync_event *e;
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;

    inc_stat(STAT_SYNC_ENTER);

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return 0;
    }

    e->timestamp_ns = bpf_ktime_get_ns();
    e->pid = pid;
    e->tid = (__u32)pid_tgid;
    e->cpu_id = bpf_get_smp_processor_id();
    bpf_get_current_comm(&e->comm, sizeof(e->comm));
    e->hook_type = HOOK_SYNC_ENTER;

    e->duration_ns = 0;
    e->offcpu_time_ns = 0;
    e->switch_count = 0;

    bpf_ringbuf_submit(e, 0);
    return 0;
}

/*
 * Uretprobe: cudaDeviceSynchronize - Exit
 */
SEC("uretprobe/cudaDeviceSynchronize")
int trace_cudaDeviceSynchronize_exit(struct pt_regs *ctx)
{
    struct sync_event *e;
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;

    inc_stat(STAT_SYNC_EXIT);

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return 0;
    }

    e->timestamp_ns = bpf_ktime_get_ns();
    e->pid = pid;
    e->tid = (__u32)pid_tgid;
    e->cpu_id = bpf_get_smp_processor_id();
    bpf_get_current_comm(&e->comm, sizeof(e->comm));
    e->hook_type = HOOK_SYNC_EXIT;

    // Duration and analysis will be done in userspace
    e->duration_ns = 0;
    e->offcpu_time_ns = 0;
    e->switch_count = 0;

    bpf_ringbuf_submit(e, 0);
    return 0;
}

/*
 * Tracepoint: irq_handler_entry (Hard IRQ)
 * Only trace if interrupting a GPU process
 */
SEC("tp_btf/irq_handler_entry")
int BPF_PROG(irq_handler_entry, int irq, struct irqaction *action)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;

    // Only track if we're interrupting a GPU process
    if (!is_gpu_process(pid))
        return 0;

    __u64 now = bpf_ktime_get_ns();
    __u32 key = 0;
    bpf_map_update_elem(&irq_start, &key, &now, BPF_ANY);

    struct irq_event *e;
    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return 0;
    }

    e->timestamp_ns = now;
    e->pid = pid;
    e->tid = (__u32)pid_tgid;
    e->cpu_id = bpf_get_smp_processor_id();
    bpf_get_current_comm(&e->comm, sizeof(e->comm));
    e->hook_type = HOOK_HARDIRQ_ENTRY;
    e->irq = irq;
    e->duration_ns = 0;

    // Try to get IRQ handler name
    if (action) {
        const char *name = BPF_CORE_READ(action, name);
        if (name)
            bpf_probe_read_kernel_str(&e->irq_name, sizeof(e->irq_name), name);
        else
            __builtin_memset(&e->irq_name, 0, sizeof(e->irq_name));
    } else {
        __builtin_memset(&e->irq_name, 0, sizeof(e->irq_name));
    }

    bpf_ringbuf_submit(e, 0);
    inc_stat(STAT_HARDIRQ);
    return 0;
}

/*
 * Tracepoint: irq_handler_exit (Hard IRQ)
 */
SEC("tp_btf/irq_handler_exit")
int BPF_PROG(irq_handler_exit, int irq, struct irqaction *action, int ret)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;

    // Only track if we're interrupting a GPU process
    if (!is_gpu_process(pid))
        return 0;

    __u64 now = bpf_ktime_get_ns();
    __u32 key = 0;
    __u64 *tsp = bpf_map_lookup_elem(&irq_start, &key);
    if (!tsp)
        return 0;

    __u64 delta = now - *tsp;

    struct irq_event *e;
    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return 0;
    }

    e->timestamp_ns = now;
    e->pid = pid;
    e->tid = (__u32)pid_tgid;
    e->cpu_id = bpf_get_smp_processor_id();
    bpf_get_current_comm(&e->comm, sizeof(e->comm));
    e->hook_type = HOOK_HARDIRQ_EXIT;
    e->irq = irq;
    e->duration_ns = delta;

    // Try to get IRQ handler name
    if (action) {
        const char *name = BPF_CORE_READ(action, name);
        if (name)
            bpf_probe_read_kernel_str(&e->irq_name, sizeof(e->irq_name), name);
        else
            __builtin_memset(&e->irq_name, 0, sizeof(e->irq_name));
    } else {
        __builtin_memset(&e->irq_name, 0, sizeof(e->irq_name));
    }

    bpf_ringbuf_submit(e, 0);
    return 0;
}

/*
 * Tracepoint: softirq_entry (Soft IRQ)
 * Only trace if running in context of a GPU process
 */
SEC("tp_btf/softirq_entry")
int BPF_PROG(softirq_entry, unsigned int vec_nr)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;

    // Only track if we're in a GPU process context
    if (!is_gpu_process(pid))
        return 0;

    __u64 now = bpf_ktime_get_ns();
    __u32 key = 0;
    bpf_map_update_elem(&irq_start, &key, &now, BPF_ANY);

    struct irq_event *e;
    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return 0;
    }

    e->timestamp_ns = now;
    e->pid = pid;
    e->tid = (__u32)pid_tgid;
    e->cpu_id = bpf_get_smp_processor_id();
    bpf_get_current_comm(&e->comm, sizeof(e->comm));
    e->hook_type = HOOK_SOFTIRQ_ENTRY;
    e->irq = vec_nr;
    e->duration_ns = 0;
    __builtin_memset(&e->irq_name, 0, sizeof(e->irq_name));

    bpf_ringbuf_submit(e, 0);
    inc_stat(STAT_SOFTIRQ);
    return 0;
}

/*
 * Tracepoint: softirq_exit (Soft IRQ)
 */
SEC("tp_btf/softirq_exit")
int BPF_PROG(softirq_exit, unsigned int vec_nr)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;

    // Only track if we're in a GPU process context
    if (!is_gpu_process(pid))
        return 0;

    __u64 now = bpf_ktime_get_ns();
    __u32 key = 0;
    __u64 *tsp = bpf_map_lookup_elem(&irq_start, &key);
    if (!tsp)
        return 0;

    __u64 delta = now - *tsp;

    struct irq_event *e;
    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return 0;
    }

    e->timestamp_ns = now;
    e->pid = pid;
    e->tid = (__u32)pid_tgid;
    e->cpu_id = bpf_get_smp_processor_id();
    bpf_get_current_comm(&e->comm, sizeof(e->comm));
    e->hook_type = HOOK_SOFTIRQ_EXIT;
    e->irq = vec_nr;
    e->duration_ns = delta;
    __builtin_memset(&e->irq_name, 0, sizeof(e->irq_name));

    bpf_ringbuf_submit(e, 0);
    return 0;
}
