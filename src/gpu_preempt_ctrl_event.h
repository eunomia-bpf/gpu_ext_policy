// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2025 */

#ifndef __GPU_PREEMPT_CTRL_EVENT_H
#define __GPU_PREEMPT_CTRL_EVENT_H

/*
 * gpu_preempt_ctrl_event.h - Event structures for GPU preempt control
 *
 * This header defines shared structures between eBPF and userspace for
 * monitoring NVIDIA GPU TSG creation and enabling userspace preempt control.
 *
 * Based on the tracepoints defined in nv-gpu-sched-tracepoint.h:
 * - nvidia_gpu_tsg_create: captures hClient/hTsg for preempt ioctl
 * - nvidia_gpu_tsg_schedule: TSG scheduling events
 * - nvidia_gpu_tsg_destroy: TSG destruction events
 */

/* Event types */
#define EVENT_TSG_CREATE    1
#define EVENT_TSG_SCHEDULE  2
#define EVENT_TSG_DESTROY   3
#define EVENT_PREEMPT       4  /* Preempt notification from userspace */

/* TSG creation event - contains handles needed for preempt ioctl */
struct tsg_create_event {
    __u64 timestamp_ns;
    __u32 event_type;
    __u32 cpu;

    /* Process info */
    __u32 pid;
    __u32 tgid;
    char comm[16];

    /* RM handles - needed for preempt ioctl */
    __u32 hClient;          /* RM client handle */
    __u32 hTsg;             /* TSG object handle */

    /* TSG info */
    __u64 tsg_id;           /* Internal TSG ID */
    __u32 engine_type;      /* Engine type (GRAPHICS, COPY, etc.) */
    __u64 timeslice_us;     /* Timeslice in microseconds */
    __u32 interleave_level; /* Interleave level (LOW/MEDIUM/HIGH) */
    __u32 runlist_id;       /* Runlist ID */
    __u32 gpu_instance;     /* GPU instance */
};

/* TSG schedule event */
struct tsg_schedule_event {
    __u64 timestamp_ns;
    __u32 event_type;
    __u32 cpu;

    /* Process info */
    __u32 pid;
    __u32 tgid;
    char comm[16];

    /* RM handles */
    __u32 hClient;
    __u32 hTsg;

    /* TSG info */
    __u64 tsg_id;
    __u32 channel_count;
    __u64 timeslice_us;
    __u32 interleave_level;
    __u32 runlist_id;
};

/* TSG destroy event */
struct tsg_destroy_event {
    __u64 timestamp_ns;
    __u32 event_type;
    __u32 cpu;

    /* Process info */
    __u32 pid;
    __u32 tgid;
    char comm[16];

    /* RM handles */
    __u32 hClient;
    __u32 hTsg;
    __u64 tsg_id;
};

/* Generic event union for ringbuffer */
struct gpu_ctrl_event {
    __u64 timestamp_ns;
    __u32 event_type;
    __u32 cpu;

    /* Process info */
    __u32 pid;
    __u32 tgid;
    char comm[16];

    /* RM handles */
    __u32 hClient;
    __u32 hTsg;

    /* TSG info */
    __u64 tsg_id;
    __u32 engine_type;
    __u64 timeslice_us;
    __u32 interleave_level;
    __u32 runlist_id;
    __u32 gpu_instance;
    __u32 channel_count;
};

/* TSG map key - for tracking active TSGs */
struct tsg_key {
    __u32 pid;              /* Process that created the TSG */
    __u32 hClient;          /* Client handle */
    __u32 hTsg;             /* TSG handle */
};

/* TSG map value - cached TSG info for preempt */
struct tsg_info {
    __u32 hClient;
    __u32 hTsg;
    __u64 tsg_id;
    __u32 engine_type;
    __u32 runlist_id;
    __u64 timeslice_us;
    __u32 interleave_level;
    __u32 pid;
    char comm[16];
    __u64 create_time_ns;
};

/* Engine types (from NVIDIA driver) */
#define ENGINE_TYPE_GRAPHICS    0
#define ENGINE_TYPE_COPY        1
#define ENGINE_TYPE_NVDEC       2
#define ENGINE_TYPE_NVENC       3
#define ENGINE_TYPE_NVJPEG      4

/* Interleave levels (from NVA06C_CTRL_INTERLEAVE_LEVEL_*) */
#define INTERLEAVE_LEVEL_LOW    0
#define INTERLEAVE_LEVEL_MEDIUM 1
#define INTERLEAVE_LEVEL_HIGH   2

/* NVIDIA ioctl escape codes */
#define NV_ESC_RM_CONTROL   0x2A

/* Control command IDs */
#define NVA06C_CTRL_CMD_PREEMPT              0xa06c0105
#define NVA06C_CTRL_CMD_SET_TIMESLICE        0xa06c0103
#define NVA06C_CTRL_CMD_GET_TIMESLICE        0xa06c0104
#define NVA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL 0xa06c0107

/* Max entries in TSG tracking map */
#define MAX_TSG_ENTRIES 1024

/* Statistics map keys */
#define STAT_TSG_CREATE     0
#define STAT_TSG_SCHEDULE   1
#define STAT_TSG_DESTROY    2
#define STAT_DROPPED        3

#endif /* __GPU_PREEMPT_CTRL_EVENT_H */
