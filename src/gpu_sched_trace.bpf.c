// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2025 */
/*
 * GPU Scheduler Trace Tool - Trace GPU scheduling hook calls using kprobes
 *
 * Traces the nv_gpu_sched_* hook functions in nvidia.ko:
 * - nv_gpu_sched_task_init:    TSG (channel group) creation
 * - nv_gpu_sched_bind:         TSG bind to hardware runlist
 * - nv_gpu_sched_token_request: Work submit token request
 * - nv_gpu_sched_task_destroy: TSG destruction
 *
 * These hooks are implemented in kernel-open/nvidia/nv-gpu-sched-hooks.c
 * and called from src/nvidia/src/kernel/gpu/fifo/*.c
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "gpu_sched_trace_event.h"

char LICENSE[] SEC("license") = "GPL";

/*
 * Context structures matching those in nv-gpu-sched-hooks.h
 * These must match the kernel driver definitions exactly!
 */
struct nv_gpu_task_init_ctx {
    u64 tsg_id;
    u32 engine_type;
    u64 default_timeslice;
    u32 default_interleave;
    u32 runlist_id;
    u64 timeslice;
    u32 interleave_level;
};

struct nv_gpu_bind_ctx {
    u64 tsg_id;
    u32 runlist_id;
    u32 channel_count;
    u64 timeslice_us;
    u32 interleave_level;
    u32 allow;
};

struct nv_gpu_token_request_ctx {
    u32 channel_id;
    u64 tsg_id;
    u32 token;
};

struct nv_gpu_task_destroy_ctx {
    u64 tsg_id;
};

// Ring buffer for outputting events
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");

// Statistics counters
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 8);
    __type(key, u32);
    __type(value, u64);
} stats SEC(".maps");

#define STAT_TASK_INIT      0
#define STAT_BIND           1
#define STAT_TOKEN_REQUEST  2
#define STAT_TASK_DESTROY   3
#define STAT_DROPPED        4
#define STAT_READ_FAILED    5

static __always_inline void inc_stat(u32 key)
{
    u64 *val = bpf_map_lookup_elem(&stats, &key);
    if (val)
        __sync_fetch_and_add(val, 1);
}

/*
 * Hook 1: Task Init - TSG (channel group) creation
 *
 * Called when a new TSG is being created. This is the point where
 * scheduling parameters (timeslice, interleave level) are set.
 */
SEC("kprobe/nv_gpu_sched_task_init")
int BPF_KPROBE(trace_task_init, struct nv_gpu_task_init_ctx *hook_ctx)
{
    struct gpu_sched_event *e;
    struct nv_gpu_task_init_ctx local_ctx;

    inc_stat(STAT_TASK_INIT);

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return 0;
    }

    __u64 pid_tgid = bpf_get_current_pid_tgid();
    int read_ok = 0;

    // Read context from kernel memory
    if (bpf_probe_read_kernel(&local_ctx, sizeof(local_ctx), hook_ctx) == 0) {
        read_ok = 1;
    } else {
        inc_stat(STAT_READ_FAILED);
        __builtin_memset(&local_ctx, 0, sizeof(local_ctx));
    }

    e->timestamp_ns = bpf_ktime_get_ns();
    e->cpu = bpf_get_smp_processor_id();
    e->pid = pid_tgid;           // lower 32 bits = pid (thread id)
    e->tgid = pid_tgid >> 32;    // upper 32 bits = tgid (process id)
    bpf_get_current_comm(&e->comm, sizeof(e->comm));
    e->hook_type = HOOK_TASK_INIT;
    e->tsg_id = read_ok ? local_ctx.tsg_id : 0xFFFFFFFF;
    e->engine_type = local_ctx.engine_type;
    e->timeslice_us = local_ctx.default_timeslice;
    e->interleave_level = local_ctx.default_interleave;
    e->runlist_id = local_ctx.runlist_id;
    e->channel_count = 0;
    e->allow = 0;
    e->channel_id = 0;
    e->token = 0;

    bpf_ringbuf_submit(e, 0);
    return 0;
}

/*
 * Hook 2: Bind - TSG bind to hardware runlist
 *
 * Called when a TSG is being bound to the hardware runlist. This is the
 * admission control point where the scheduler can accept or reject binding.
 * This is a ONE-TIME operation, not called on every scheduling decision.
 */
SEC("kprobe/nv_gpu_sched_bind")
int BPF_KPROBE(trace_bind, struct nv_gpu_bind_ctx *hook_ctx)
{
    struct gpu_sched_event *e;
    struct nv_gpu_bind_ctx local_ctx;

    inc_stat(STAT_BIND);

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return 0;
    }

    __u64 pid_tgid = bpf_get_current_pid_tgid();
    int read_ok = 0;

    // Read context from kernel memory
    if (bpf_probe_read_kernel(&local_ctx, sizeof(local_ctx), hook_ctx) == 0) {
        read_ok = 1;
    } else {
        inc_stat(STAT_READ_FAILED);
        __builtin_memset(&local_ctx, 0, sizeof(local_ctx));
    }

    e->timestamp_ns = bpf_ktime_get_ns();
    e->cpu = bpf_get_smp_processor_id();
    e->pid = pid_tgid;
    e->tgid = pid_tgid >> 32;
    bpf_get_current_comm(&e->comm, sizeof(e->comm));
    e->hook_type = HOOK_BIND;
    e->tsg_id = read_ok ? local_ctx.tsg_id : 0xFFFFFFFF;
    e->engine_type = 0;
    e->timeslice_us = local_ctx.timeslice_us;
    e->interleave_level = local_ctx.interleave_level;
    e->runlist_id = local_ctx.runlist_id;
    e->channel_count = local_ctx.channel_count;
    e->allow = local_ctx.allow;
    e->channel_id = 0;
    e->token = 0;

    bpf_ringbuf_submit(e, 0);
    return 0;
}

/*
 * Hook 3: Token Request - Work submit token request (for sync)
 *
 * Called when user requests a work submit token via ioctl.
 * This is NOT called on every kernel launch - only when userspace
 * explicitly requests a token (typically for synchronization).
 *
 * Triggered by: NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN ioctl
 */
SEC("kprobe/nv_gpu_sched_token_request")
int BPF_KPROBE(trace_token_request, struct nv_gpu_token_request_ctx *hook_ctx)
{
    struct gpu_sched_event *e;
    struct nv_gpu_token_request_ctx local_ctx;

    inc_stat(STAT_TOKEN_REQUEST);

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return 0;
    }

    __u64 pid_tgid = bpf_get_current_pid_tgid();
    int read_ok = 0;

    // Read context from kernel memory
    if (bpf_probe_read_kernel(&local_ctx, sizeof(local_ctx), hook_ctx) == 0) {
        read_ok = 1;
    } else {
        inc_stat(STAT_READ_FAILED);
        __builtin_memset(&local_ctx, 0, sizeof(local_ctx));
    }

    e->timestamp_ns = bpf_ktime_get_ns();
    e->cpu = bpf_get_smp_processor_id();
    e->pid = pid_tgid;
    e->tgid = pid_tgid >> 32;
    bpf_get_current_comm(&e->comm, sizeof(e->comm));
    e->hook_type = HOOK_TOKEN_REQUEST;
    e->tsg_id = read_ok ? local_ctx.tsg_id : 0xFFFFFFFF;
    e->engine_type = 0;
    e->timeslice_us = 0;
    e->interleave_level = 0;
    e->runlist_id = 0;
    e->channel_count = 0;
    e->allow = 0;
    e->channel_id = local_ctx.channel_id;
    e->token = local_ctx.token;

    bpf_ringbuf_submit(e, 0);
    return 0;
}

/*
 * Hook 4: Task Destroy - TSG destruction
 *
 * Called when a TSG is being destroyed. This can be used to clean up
 * any BPF map entries associated with the TSG.
 */
SEC("kprobe/nv_gpu_sched_task_destroy")
int BPF_KPROBE(trace_task_destroy, struct nv_gpu_task_destroy_ctx *hook_ctx)
{
    struct gpu_sched_event *e;
    struct nv_gpu_task_destroy_ctx local_ctx;

    inc_stat(STAT_TASK_DESTROY);

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return 0;
    }

    __u64 pid_tgid = bpf_get_current_pid_tgid();
    int read_ok = 0;

    // Read context from kernel memory
    if (bpf_probe_read_kernel(&local_ctx, sizeof(local_ctx), hook_ctx) == 0) {
        read_ok = 1;
    } else {
        inc_stat(STAT_READ_FAILED);
        __builtin_memset(&local_ctx, 0, sizeof(local_ctx));
    }

    e->timestamp_ns = bpf_ktime_get_ns();
    e->cpu = bpf_get_smp_processor_id();
    e->pid = pid_tgid;
    e->tgid = pid_tgid >> 32;
    bpf_get_current_comm(&e->comm, sizeof(e->comm));
    e->hook_type = HOOK_TASK_DESTROY;
    e->tsg_id = read_ok ? local_ctx.tsg_id : 0xFFFFFFFF;
    e->engine_type = 0;
    e->timeslice_us = 0;
    e->interleave_level = 0;
    e->runlist_id = 0;
    e->channel_count = 0;
    e->allow = 0;
    e->channel_id = 0;
    e->token = 0;

    bpf_ringbuf_submit(e, 0);
    return 0;
}
