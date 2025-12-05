/* SPDX-License-Identifier: GPL-2.0 */
/*
 * GPU Scheduler struct_ops BPF header
 *
 * This header defines the struct_ops interface and kfuncs for BPF programs
 * that want to customize GPU scheduling behavior.
 */

#ifndef _GPU_SCHED_STRUCT_OPS_H
#define _GPU_SCHED_STRUCT_OPS_H

/* Context structures - must match kernel definitions */
struct nv_gpu_task_init_ctx {
    __u64 tsg_id;
    __u32 engine_type;
    __u64 default_timeslice;
    __u32 default_interleave;
    __u32 runlist_id;
    __u64 timeslice;
    __u32 interleave_level;
};

struct nv_gpu_bind_ctx {
    __u64 tsg_id;
    __u32 runlist_id;
    __u32 channel_count;
    __u64 timeslice_us;
    __u32 interleave_level;
    __u32 allow;
};

struct nv_gpu_task_destroy_ctx {
    __u64 tsg_id;
};

/* GPU Scheduler struct_ops definition */
struct nv_gpu_sched_ops {
    int (*on_task_init)(struct nv_gpu_task_init_ctx *ctx);
    int (*on_bind)(struct nv_gpu_bind_ctx *ctx);
    int (*on_task_destroy)(struct nv_gpu_task_destroy_ctx *ctx);
};

/* Engine types */
#define NV_ENGINE_TYPE_GRAPHICS    0
#define NV_ENGINE_TYPE_COPY        1
#define NV_ENGINE_TYPE_NVDEC       2
#define NV_ENGINE_TYPE_NVENC       3
#define NV_ENGINE_TYPE_NVJPEG      4

/* Interleave levels */
#define NV_INTERLEAVE_LEVEL_LOW    1
#define NV_INTERLEAVE_LEVEL_MEDIUM 2
#define NV_INTERLEAVE_LEVEL_HIGH   3

/* kfunc declarations */
#ifndef BPF_NO_KFUNC_PROTOTYPES
#ifndef __ksym
#define __ksym __attribute__((section(".ksyms")))
#endif
#ifndef __weak
#define __weak __attribute__((weak))
#endif

/* Set timeslice for a TSG during task_init */
extern void bpf_nv_gpu_set_timeslice(struct nv_gpu_task_init_ctx *ctx,
                                     __u64 timeslice_us) __weak __ksym;

/* Set interleave level for a TSG during task_init */
extern void bpf_nv_gpu_set_interleave(struct nv_gpu_task_init_ctx *ctx,
                                      __u32 interleave_level) __weak __ksym;

/* Reject binding for a TSG during on_bind */
extern void bpf_nv_gpu_reject_bind(struct nv_gpu_bind_ctx *ctx) __weak __ksym;

#endif /* BPF_NO_KFUNC_PROTOTYPES */

#endif /* _GPU_SCHED_STRUCT_OPS_H */
