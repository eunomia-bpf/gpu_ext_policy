// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2025 */
/*
 * gpu_preempt_ctrl.bpf.c - eBPF program for GPU TSG preempt control
 *
 * Uses kernel tracepoints (instead of kprobes) to capture GPU scheduling
 * events including hClient/hTsg handles needed for preempt ioctl.
 *
 * Tracepoints used:
 * - tracepoint/nvidia/nvidia_gpu_tsg_create:  TSG creation with handles
 * - tracepoint/nvidia/nvidia_gpu_tsg_schedule: TSG scheduling
 * - tracepoint/nvidia/nvidia_gpu_tsg_destroy: TSG destruction
 *
 * The userspace component can use the captured handles to send
 * NVA06C_CTRL_CMD_PREEMPT ioctl to control GPU scheduling.
 *
 * This approach is similar to GPreempt patch but uses tracepoints
 * instead of modifying the kernel driver.
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "gpu_preempt_ctrl_event.h"

char LICENSE[] SEC("license") = "GPL";

/*
 * Tracepoint context structures
 *
 * These match the trace event definitions in nv-gpu-sched-tracepoint.h.
 * The field layout must match exactly what the kernel exposes.
 */

/* nvidia_gpu_tsg_create tracepoint args */
struct trace_nvidia_gpu_tsg_create {
    /* Common tracepoint fields (skipped via offset) */
    __u64 __do_not_use__;  /* common fields padding */

    /* Actual trace event fields - from TP_STRUCT__entry */
    __u32 hClient;
    __u32 hTsg;
    __u64 tsg_id;
    __u32 engine_type;
    __u64 timeslice_us;
    __u32 interleave_level;
    __u32 runlist_id;
    __u32 gpu_instance;
    __u32 pid;
};

/* nvidia_gpu_tsg_schedule tracepoint args */
struct trace_nvidia_gpu_tsg_schedule {
    __u64 __do_not_use__;

    __u32 hClient;
    __u32 hTsg;
    __u64 tsg_id;
    __u32 channel_count;
    __u64 timeslice_us;
    __u32 interleave_level;
    __u32 runlist_id;
    __u32 pid;
};

/* nvidia_gpu_tsg_destroy tracepoint args */
struct trace_nvidia_gpu_tsg_destroy {
    __u64 __do_not_use__;

    __u32 hClient;
    __u32 hTsg;
    __u64 tsg_id;
    __u32 pid;
};

/* Ring buffer for events to userspace */
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");

/*
 * Hash map to track active TSGs
 * Key: PID (process that owns the TSG)
 * Value: tsg_info struct with handles
 *
 * This allows userspace to look up TSG handles by PID.
 */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_TSG_ENTRIES);
    __type(key, __u32);  /* Use hTsg as key for quick lookup */
    __type(value, struct tsg_info);
} tsg_map SEC(".maps");

/*
 * Map to track TSGs by PID (for finding all TSGs of a process)
 * Key: PID
 * Value: Most recent tsg_info for that PID
 */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_TSG_ENTRIES);
    __type(key, __u32);  /* PID */
    __type(value, struct tsg_info);
} pid_tsg_map SEC(".maps");

/* Statistics counters */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 8);
    __type(key, __u32);
    __type(value, __u64);
} stats SEC(".maps");

static __always_inline void inc_stat(__u32 key)
{
    __u64 *val = bpf_map_lookup_elem(&stats, &key);
    if (val)
        __sync_fetch_and_add(val, 1);
}

/*
 * Tracepoint handler: nvidia_gpu_tsg_create
 *
 * Captures TSG creation events with hClient/hTsg handles.
 * These handles are required for the NVA06C_CTRL_CMD_PREEMPT ioctl.
 */
SEC("tracepoint/nvidia/nvidia_gpu_tsg_create")
int handle_tsg_create(struct trace_nvidia_gpu_tsg_create *ctx)
{
    struct gpu_ctrl_event *e;
    struct tsg_info info = {};
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;
    __u32 hTsg;

    inc_stat(STAT_TSG_CREATE);

    /* Allocate event for ringbuffer */
    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return 0;
    }

    /* Fill event */
    e->timestamp_ns = bpf_ktime_get_ns();
    e->event_type = EVENT_TSG_CREATE;
    e->cpu = bpf_get_smp_processor_id();
    e->pid = pid_tgid;
    e->tgid = pid;
    bpf_get_current_comm(&e->comm, sizeof(e->comm));

    /* Read tracepoint fields */
    e->hClient = ctx->hClient;
    e->hTsg = ctx->hTsg;
    e->tsg_id = ctx->tsg_id;
    e->engine_type = ctx->engine_type;
    e->timeslice_us = ctx->timeslice_us;
    e->interleave_level = ctx->interleave_level;
    e->runlist_id = ctx->runlist_id;
    e->gpu_instance = ctx->gpu_instance;
    e->channel_count = 0;

    bpf_ringbuf_submit(e, 0);

    /* Store TSG info in map for later lookup */
    info.hClient = ctx->hClient;
    info.hTsg = ctx->hTsg;
    info.tsg_id = ctx->tsg_id;
    info.engine_type = ctx->engine_type;
    info.runlist_id = ctx->runlist_id;
    info.timeslice_us = ctx->timeslice_us;
    info.interleave_level = ctx->interleave_level;
    info.pid = pid;
    bpf_get_current_comm(&info.comm, sizeof(info.comm));
    info.create_time_ns = bpf_ktime_get_ns();

    /* Store by hTsg for direct lookup */
    hTsg = ctx->hTsg;
    bpf_map_update_elem(&tsg_map, &hTsg, &info, BPF_ANY);

    /* Also store by PID for process-based lookup */
    bpf_map_update_elem(&pid_tsg_map, &pid, &info, BPF_ANY);

    return 0;
}

/*
 * Tracepoint handler: nvidia_gpu_tsg_schedule
 *
 * Captures TSG scheduling events. This is when the TSG is enabled
 * and ready for GPU execution.
 */
SEC("tracepoint/nvidia/nvidia_gpu_tsg_schedule")
int handle_tsg_schedule(struct trace_nvidia_gpu_tsg_schedule *ctx)
{
    struct gpu_ctrl_event *e;
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;

    inc_stat(STAT_TSG_SCHEDULE);

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return 0;
    }

    e->timestamp_ns = bpf_ktime_get_ns();
    e->event_type = EVENT_TSG_SCHEDULE;
    e->cpu = bpf_get_smp_processor_id();
    e->pid = pid_tgid;
    e->tgid = pid;
    bpf_get_current_comm(&e->comm, sizeof(e->comm));

    e->hClient = ctx->hClient;
    e->hTsg = ctx->hTsg;
    e->tsg_id = ctx->tsg_id;
    e->channel_count = ctx->channel_count;
    e->timeslice_us = ctx->timeslice_us;
    e->interleave_level = ctx->interleave_level;
    e->runlist_id = ctx->runlist_id;
    e->engine_type = 0;
    e->gpu_instance = 0;

    bpf_ringbuf_submit(e, 0);

    return 0;
}

/*
 * Tracepoint handler: nvidia_gpu_tsg_destroy
 *
 * Captures TSG destruction. Removes the TSG from tracking maps.
 */
SEC("tracepoint/nvidia/nvidia_gpu_tsg_destroy")
int handle_tsg_destroy(struct trace_nvidia_gpu_tsg_destroy *ctx)
{
    struct gpu_ctrl_event *e;
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;
    __u32 hTsg;

    inc_stat(STAT_TSG_DESTROY);

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        inc_stat(STAT_DROPPED);
        return 0;
    }

    e->timestamp_ns = bpf_ktime_get_ns();
    e->event_type = EVENT_TSG_DESTROY;
    e->cpu = bpf_get_smp_processor_id();
    e->pid = pid_tgid;
    e->tgid = pid;
    bpf_get_current_comm(&e->comm, sizeof(e->comm));

    e->hClient = ctx->hClient;
    e->hTsg = ctx->hTsg;
    e->tsg_id = ctx->tsg_id;
    e->engine_type = 0;
    e->timeslice_us = 0;
    e->interleave_level = 0;
    e->runlist_id = 0;
    e->gpu_instance = 0;
    e->channel_count = 0;

    bpf_ringbuf_submit(e, 0);

    /* Remove from tracking map */
    hTsg = ctx->hTsg;
    bpf_map_delete_elem(&tsg_map, &hTsg);

    return 0;
}
