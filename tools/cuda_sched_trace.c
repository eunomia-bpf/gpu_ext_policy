// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2025 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include "cuda_sched_trace.skel.h"
#include "cuda_launch_trace_event.h"

// Default CUDA library paths
static const char *cuda_driver_libs[] = {
    "/usr/local/cuda/lib64/libcuda.so",
    "/usr/lib/x86_64-linux-gnu/libcuda.so",
    "/usr/lib64/libcuda.so",
    NULL
};

static const char *cuda_runtime_libs[] = {
    "/usr/local/cuda/lib64/libcudart.so",
    "/usr/lib/x86_64-linux-gnu/libcudart.so",
    "/usr/lib64/libcudart.so",
    NULL
};

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
    if (level >= LIBBPF_WARN)
        return vfprintf(stderr, format, args);
    return 0;
}

static volatile sig_atomic_t exiting = 0;
static __u64 start_time_ns = 0;

static void print_stats(struct cuda_sched_trace_bpf *skel)
{
    int stats_fd = bpf_map__fd(skel->maps.stats);
    __u32 key;
    __u64 val;
    __u64 stats[10] = {0};

    for (key = 0; key < 10; key++) {
        if (bpf_map_lookup_elem(stats_fd, &key, &val) == 0) {
            stats[key] = val;
        }
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "===============================================================================\n");
    fprintf(stderr, "CUDA + SCHEDULER + IRQ TRACE SUMMARY\n");
    fprintf(stderr, "===============================================================================\n");
    fprintf(stderr, "cuLaunchKernel            %8llu\n", stats[0]);
    fprintf(stderr, "cudaLaunchKernel          %8llu\n", stats[1]);
    fprintf(stderr, "cudaSync Enter            %8llu\n", stats[2]);
    fprintf(stderr, "cudaSync Exit             %8llu\n", stats[3]);
    fprintf(stderr, "Sched Switches Tracked    %8llu\n", stats[4]);
    fprintf(stderr, "Hard IRQs Tracked         %8llu\n", stats[6]);
    fprintf(stderr, "Soft IRQs Tracked         %8llu\n", stats[7]);
    if (stats[5] > 0) {
        fprintf(stderr, "Dropped Events            %8llu\n", stats[5]);
    }
    fprintf(stderr, "===============================================================================\n");
}

static void sig_handler(int sig)
{
    exiting = 1;
}

static const char *hook_type_str(__u32 hook_type)
{
    switch (hook_type) {
    case HOOK_CULAUNCHKERNEL:
        return "cuLaunchKernel";
    case HOOK_CUDALAUNCHKERNEL:
        return "cudaLaunchKernel";
    case HOOK_SYNC_ENTER:
        return "syncEnter";
    case HOOK_SYNC_EXIT:
        return "syncExit";
    case HOOK_SCHED_SWITCH:
        return "schedSwitch";
    case HOOK_HARDIRQ_ENTRY:
        return "hardirqEntry";
    case HOOK_HARDIRQ_EXIT:
        return "hardirqExit";
    case HOOK_SOFTIRQ_ENTRY:
        return "softirqEntry";
    case HOOK_SOFTIRQ_EXIT:
        return "softirqExit";
    default:
        return "unknown";
    }
}

// Softirq names (from kernel include/trace/events/irq.h)
static const char *softirq_name(__u32 vec_nr)
{
    static const char *names[] = {
        "HI", "TIMER", "NET_TX", "NET_RX", "BLOCK",
        "IRQ_POLL", "TASKLET", "SCHED", "HRTIMER", "RCU"
    };
    if (vec_nr < sizeof(names) / sizeof(names[0]))
        return names[vec_nr];
    return "unknown";
}

static int handle_event(void *ctx, void *data, size_t data_sz)
{
    __u64 elapsed_ns;

    // Check event type by size
    if (data_sz == sizeof(struct cuda_launch_event)) {
        const struct cuda_launch_event *e = data;

        if (start_time_ns == 0)
            start_time_ns = e->timestamp_ns;

        elapsed_ns = e->timestamp_ns - start_time_ns;

        // Unified CSV format for all events
        // timestamp_ns,event_type,pid,tid,comm,cpu,
        // grid_x,grid_y,grid_z,block_x,block_y,block_z,shared_mem,stream,
        // last_offcpu_ns,last_oncpu_ns

        printf("%llu,%s,%u,%u,%s,%u,",
               (unsigned long long)elapsed_ns,
               hook_type_str(e->hook_type),
               e->pid,
               e->tid,
               e->comm,
               e->cpu_id);

        // CUDA launch specific fields
        if (e->hook_type == HOOK_CULAUNCHKERNEL || e->hook_type == HOOK_CUDALAUNCHKERNEL) {
            printf("%u,%u,%u,%u,%u,%u,%u,0x%llx,",
                   e->grid_dim_x, e->grid_dim_y, e->grid_dim_z,
                   e->block_dim_x, e->block_dim_y, e->block_dim_z,
                   e->shared_mem_bytes,
                   (unsigned long long)e->stream);
        } else {
            // For sched_switch events
            printf(",,,,,,,,");
        }

        // Scheduler fields (0 means N/A, >0 means actual timestamp)
        printf("%llu,%llu\n",
               (unsigned long long)e->last_offcpu_ns,
               (unsigned long long)e->last_oncpu_ns);

    } else if (data_sz == sizeof(struct sync_event)) {
        const struct sync_event *e = data;

        if (start_time_ns == 0)
            start_time_ns = e->timestamp_ns;

        elapsed_ns = e->timestamp_ns - start_time_ns;

        // CSV for sync events
        printf("%llu,%s,%u,%u,%s,%u,",
               (unsigned long long)elapsed_ns,
               hook_type_str(e->hook_type),
               e->pid,
               e->tid,
               e->comm,
               e->cpu_id);

        // Empty CUDA fields
        printf(",,,,,,,,");

        // Empty sched fields for sync events
        printf(",\n");

    } else if (data_sz == sizeof(struct irq_event)) {
        const struct irq_event *e = data;

        if (start_time_ns == 0)
            start_time_ns = e->timestamp_ns;

        elapsed_ns = e->timestamp_ns - start_time_ns;

        // CSV for IRQ events
        // timestamp_ns,event_type,pid,tid,comm,cpu,
        // grid_x,grid_y,grid_z,block_x,block_y,block_z,shared_mem,stream,
        // last_offcpu_ns,last_oncpu_ns,irq_num,irq_name,duration_ns

        printf("%llu,%s,%u,%u,%s,%u,",
               (unsigned long long)elapsed_ns,
               hook_type_str(e->hook_type),
               e->pid,
               e->tid,
               e->comm,
               e->cpu_id);

        // Empty CUDA fields
        printf(",,,,,,,,");

        // Empty sched fields
        printf(",,");

        // IRQ specific fields
        if (e->hook_type == HOOK_SOFTIRQ_ENTRY || e->hook_type == HOOK_SOFTIRQ_EXIT) {
            printf("%u,%s,%llu\n",
                   e->irq,
                   softirq_name(e->irq),
                   (unsigned long long)e->duration_ns);
        } else {
            // Hardirq
            printf("%u,%s,%llu\n",
                   e->irq,
                   e->irq_name[0] ? e->irq_name : "unknown",
                   (unsigned long long)e->duration_ns);
        }
    }

    return 0;
}

static const char *find_library(const char **lib_paths)
{
    for (int i = 0; lib_paths[i] != NULL; i++) {
        if (access(lib_paths[i], F_OK) == 0) {
            return lib_paths[i];
        }
    }
    return NULL;
}

static int attach_uprobe(struct bpf_program *prog, const char *lib_path,
                         const char *func_name, struct bpf_link **link)
{
    LIBBPF_OPTS(bpf_uprobe_opts, uprobe_opts,
        .func_name = func_name,
        .retprobe = false,
    );

    *link = bpf_program__attach_uprobe_opts(prog, -1, lib_path, 0, &uprobe_opts);
    if (!*link) {
        fprintf(stderr, "Warning: Failed to attach uprobe to %s: %s\n",
                func_name, strerror(errno));
        return -1;
    }

    fprintf(stderr, "Attached uprobe to %s in %s\n", func_name, lib_path);
    return 0;
}

static int attach_uretprobe(struct bpf_program *prog, const char *lib_path,
                            const char *func_name, struct bpf_link **link)
{
    LIBBPF_OPTS(bpf_uprobe_opts, uprobe_opts,
        .func_name = func_name,
        .retprobe = true,
    );

    *link = bpf_program__attach_uprobe_opts(prog, -1, lib_path, 0, &uprobe_opts);
    if (!*link) {
        fprintf(stderr, "Warning: Failed to attach uretprobe to %s: %s\n",
                func_name, strerror(errno));
        return -1;
    }

    fprintf(stderr, "Attached uretprobe to %s in %s\n", func_name, lib_path);
    return 0;
}

int main(int argc, char **argv)
{
    struct cuda_sched_trace_bpf *skel = NULL;
    struct ring_buffer *rb = NULL;
    struct bpf_link *link_sched = NULL;
    struct bpf_link *link_culaunch = NULL;
    struct bpf_link *link_cudalaunch = NULL;
    struct bpf_link *link_sync_enter = NULL;
    struct bpf_link *link_sync_exit = NULL;
    struct bpf_link *link_hardirq_entry = NULL;
    struct bpf_link *link_hardirq_exit = NULL;
    struct bpf_link *link_softirq_entry = NULL;
    struct bpf_link *link_softirq_exit = NULL;
    const char *cuda_driver_lib = NULL;
    const char *cuda_runtime_lib = NULL;
    int err;

    // Parse command-line arguments
    if (argc > 1) {
        cuda_driver_lib = argv[1];
    } else {
        cuda_driver_lib = find_library(cuda_driver_libs);
    }

    if (argc > 2) {
        cuda_runtime_lib = argv[2];
    } else {
        cuda_runtime_lib = find_library(cuda_runtime_libs);
    }

    // Set up libbpf
    libbpf_set_print(libbpf_print_fn);

    // Open BPF application
    skel = cuda_sched_trace_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    // Load & verify BPF programs
    err = cuda_sched_trace_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load and verify BPF skeleton: %d\n", err);
        goto cleanup;
    }

    // Attach tracepoint for sched_switch manually
    link_sched = bpf_program__attach(skel->progs.sched_switch);
    if (!link_sched) {
        err = -errno;
        fprintf(stderr, "Failed to attach sched_switch tracepoint: %s\n", strerror(errno));
        goto cleanup;
    }
    fprintf(stderr, "Attached tracepoint: sched_switch\n");

    // Attach IRQ tracepoints
    link_hardirq_entry = bpf_program__attach(skel->progs.irq_handler_entry);
    if (!link_hardirq_entry) {
        fprintf(stderr, "Warning: Failed to attach irq_handler_entry: %s\n", strerror(errno));
    } else {
        fprintf(stderr, "Attached tracepoint: irq_handler_entry\n");
    }

    link_hardirq_exit = bpf_program__attach(skel->progs.irq_handler_exit);
    if (!link_hardirq_exit) {
        fprintf(stderr, "Warning: Failed to attach irq_handler_exit: %s\n", strerror(errno));
    } else {
        fprintf(stderr, "Attached tracepoint: irq_handler_exit\n");
    }

    link_softirq_entry = bpf_program__attach(skel->progs.softirq_entry);
    if (!link_softirq_entry) {
        fprintf(stderr, "Warning: Failed to attach softirq_entry: %s\n", strerror(errno));
    } else {
        fprintf(stderr, "Attached tracepoint: softirq_entry\n");
    }

    link_softirq_exit = bpf_program__attach(skel->progs.softirq_exit);
    if (!link_softirq_exit) {
        fprintf(stderr, "Warning: Failed to attach softirq_exit: %s\n", strerror(errno));
    } else {
        fprintf(stderr, "Attached tracepoint: softirq_exit\n");
    }

    // Attach uprobes manually
    if (cuda_driver_lib) {
        fprintf(stderr, "Using CUDA Driver library: %s\n", cuda_driver_lib);
        attach_uprobe(skel->progs.trace_cuLaunchKernel, cuda_driver_lib,
                     "cuLaunchKernel", &link_culaunch);
    }

    if (cuda_runtime_lib) {
        fprintf(stderr, "Using CUDA Runtime library: %s\n", cuda_runtime_lib);
        attach_uprobe(skel->progs.trace_cudaLaunchKernel, cuda_runtime_lib,
                     "cudaLaunchKernel", &link_cudalaunch);
        attach_uprobe(skel->progs.trace_cudaDeviceSynchronize_enter, cuda_runtime_lib,
                     "cudaDeviceSynchronize", &link_sync_enter);
        attach_uretprobe(skel->progs.trace_cudaDeviceSynchronize_exit, cuda_runtime_lib,
                        "cudaDeviceSynchronize", &link_sync_exit);
    }

    // Set up ring buffer polling
    rb = ring_buffer__new(bpf_map__fd(skel->maps.events), handle_event, NULL, NULL);
    if (!rb) {
        err = -1;
        fprintf(stderr, "Failed to create ring buffer\n");
        goto cleanup;
    }

    // Set up signal handler
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    // Print CSV header to stdout
    printf("timestamp_ns,event_type,pid,tid,comm,cpu,grid_x,grid_y,grid_z,block_x,block_y,block_z,shared_mem,stream,last_offcpu_ns,last_oncpu_ns,irq_num,irq_name,duration_ns\n");

    fprintf(stderr, "Tracing CUDA + scheduler + IRQ events... Press Ctrl-C to stop.\n");
    fprintf(stderr, "CSV output will be written to stdout\n");

    // Process events
    while (!exiting) {
        err = ring_buffer__poll(rb, 100);
        if (err == -EINTR) {
            err = 0;
            break;
        }
        if (err < 0) {
            fprintf(stderr, "Error polling ring buffer: %d\n", err);
            break;
        }
    }

    print_stats(skel);

cleanup:
    bpf_link__destroy(link_sched);
    bpf_link__destroy(link_culaunch);
    bpf_link__destroy(link_cudalaunch);
    bpf_link__destroy(link_sync_enter);
    bpf_link__destroy(link_sync_exit);
    bpf_link__destroy(link_hardirq_entry);
    bpf_link__destroy(link_hardirq_exit);
    bpf_link__destroy(link_softirq_entry);
    bpf_link__destroy(link_softirq_exit);
    ring_buffer__free(rb);
    cuda_sched_trace_bpf__destroy(skel);
    return err < 0 ? -err : 0;
}
