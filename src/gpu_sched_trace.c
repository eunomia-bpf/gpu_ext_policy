// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2025 */
/*
 * GPU Scheduler Trace Tool - Userspace component
 *
 * Loads the BPF program and displays GPU scheduling events in real-time.
 * Shows TSG creation, scheduling, work submission, and destruction events.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include "gpu_sched_trace.skel.h"
#include "gpu_sched_trace_event.h"

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
    return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;

static void sig_handler(int sig)
{
    exiting = true;
}

static const char *hook_type_str(int type)
{
    switch (type) {
    case HOOK_TASK_INIT:    return "TASK_INIT";
    case HOOK_BIND:         return "BIND";
    case HOOK_TOKEN_REQUEST: return "TOKEN_REQ";
    case HOOK_TASK_DESTROY: return "TASK_DESTROY";
    default:                return "UNKNOWN";
    }
}

static const char *engine_type_str(int type)
{
    switch (type) {
    case ENGINE_TYPE_GRAPHICS: return "GRAPHICS";
    case ENGINE_TYPE_COPY:     return "COPY";
    case ENGINE_TYPE_NVDEC:    return "NVDEC";
    case ENGINE_TYPE_NVENC:    return "NVENC";
    case ENGINE_TYPE_NVJPEG:   return "NVJPEG";
    default:                   return "UNKNOWN";
    }
}

static const char *interleave_str(int level)
{
    switch (level) {
    case INTERLEAVE_LEVEL_LOW:    return "LOW";
    case INTERLEAVE_LEVEL_MEDIUM: return "MEDIUM";
    case INTERLEAVE_LEVEL_HIGH:   return "HIGH";
    default:                      return "UNKNOWN";
    }
}

static int handle_event(void *ctx, void *data, size_t data_sz)
{
    struct gpu_sched_event *e = data;
    struct timespec ts;
    struct tm *tm;
    char ts_buf[32];

    // Convert timestamp to human-readable format
    clock_gettime(CLOCK_REALTIME, &ts);
    tm = localtime(&ts.tv_sec);
    strftime(ts_buf, sizeof(ts_buf), "%H:%M:%S", tm);

    switch (e->hook_type) {
    case HOOK_TASK_INIT:
        printf("%s.%06llu [CPU%02u] %-12s PID=%-6u %-16s TSG=%-4llu engine=%-8s timeslice=%llu us interleave=%s runlist=%u\n",
               ts_buf, (unsigned long long)(e->timestamp_ns % 1000000000 / 1000),
               e->cpu, hook_type_str(e->hook_type),
               e->tgid, e->comm,
               (unsigned long long)e->tsg_id,
               engine_type_str(e->engine_type),
               (unsigned long long)e->timeslice_us,
               interleave_str(e->interleave_level),
               e->runlist_id);
        break;

    case HOOK_BIND:
        printf("%s.%06llu [CPU%02u] %-12s PID=%-6u %-16s TSG=%-4llu channels=%u timeslice=%llu us interleave=%s allowed=%s\n",
               ts_buf, (unsigned long long)(e->timestamp_ns % 1000000000 / 1000),
               e->cpu, hook_type_str(e->hook_type),
               e->tgid, e->comm,
               (unsigned long long)e->tsg_id,
               e->channel_count,
               (unsigned long long)e->timeslice_us,
               interleave_str(e->interleave_level),
               e->allow ? "yes" : "no");
        break;

    case HOOK_TOKEN_REQUEST:
        // Token request - triggered by GET_WORK_SUBMIT_TOKEN ioctl (typically for sync)
        printf("%s.%06llu [CPU%02u] %-12s PID=%-6u %-16s TSG=%-4llu channel=%u token=%u\n",
               ts_buf, (unsigned long long)(e->timestamp_ns % 1000000000 / 1000),
               e->cpu, hook_type_str(e->hook_type),
               e->tgid, e->comm,
               (unsigned long long)e->tsg_id,
               e->channel_id,
               e->token);
        break;

    case HOOK_TASK_DESTROY:
        printf("%s.%06llu [CPU%02u] %-12s PID=%-6u %-16s TSG=%-4llu\n",
               ts_buf, (unsigned long long)(e->timestamp_ns % 1000000000 / 1000),
               e->cpu, hook_type_str(e->hook_type),
               e->tgid, e->comm,
               (unsigned long long)e->tsg_id);
        break;

    default:
        printf("%s.%06llu [CPU%02u] %-12s PID=%-6u %-16s (unknown event)\n",
               ts_buf, (unsigned long long)(e->timestamp_ns % 1000000000 / 1000),
               e->cpu, hook_type_str(e->hook_type),
               e->tgid, e->comm);
        break;
    }

    return 0;
}

static void print_stats(struct gpu_sched_trace_bpf *skel)
{
    int fd = bpf_map__fd(skel->maps.stats);
    __u64 task_init = 0, bind = 0, token_request = 0, task_destroy = 0, dropped = 0, read_failed = 0;
    __u32 key;

    key = 0; bpf_map_lookup_elem(fd, &key, &task_init);
    key = 1; bpf_map_lookup_elem(fd, &key, &bind);
    key = 2; bpf_map_lookup_elem(fd, &key, &token_request);
    key = 3; bpf_map_lookup_elem(fd, &key, &task_destroy);
    key = 4; bpf_map_lookup_elem(fd, &key, &dropped);
    key = 5; bpf_map_lookup_elem(fd, &key, &read_failed);

    printf("\n=== Statistics ===\n");
    printf("task_init:     %llu\n", (unsigned long long)task_init);
    printf("bind:          %llu\n", (unsigned long long)bind);
    printf("token_request: %llu\n", (unsigned long long)token_request);
    printf("task_destroy:  %llu\n", (unsigned long long)task_destroy);
    printf("dropped:       %llu\n", (unsigned long long)dropped);
    printf("read_failed:   %llu\n", (unsigned long long)read_failed);
}

static void usage(const char *prog)
{
    fprintf(stderr, "Usage: %s [-h]\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  -h    Show this help\n");
    fprintf(stderr, "\nTraces GPU scheduling events from nvidia.ko:\n");
    fprintf(stderr, "  TASK_INIT    - TSG (channel group) creation\n");
    fprintf(stderr, "  BIND         - TSG bind to hardware runlist (one-time, admission control)\n");
    fprintf(stderr, "  TOKEN_REQ    - Work submit token request (for sync, via ioctl)\n");
    fprintf(stderr, "  TASK_DESTROY - TSG destruction\n");
    fprintf(stderr, "\nNote: TOKEN_REQ is triggered by GET_WORK_SUBMIT_TOKEN ioctl,\n");
    fprintf(stderr, "      NOT on every kernel launch (which bypasses the kernel).\n");
}

int main(int argc, char **argv)
{
    struct gpu_sched_trace_bpf *skel;
    struct ring_buffer *rb = NULL;
    int err;
    int opt;

    while ((opt = getopt(argc, argv, "h")) != -1) {
        switch (opt) {
        case 'h':
            usage(argv[0]);
            return 0;
        default:
            usage(argv[0]);
            return 1;
        }
    }

    /* Set up libbpf errors and debug info callback */
    libbpf_set_print(libbpf_print_fn);

    /* Set up signal handlers */
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    /* Open and load BPF program */
    skel = gpu_sched_trace_bpf__open_and_load();
    if (!skel) {
        fprintf(stderr, "Failed to open and load BPF skeleton\n");
        return 1;
    }

    /* Attach BPF programs */
    err = gpu_sched_trace_bpf__attach(skel);
    if (err) {
        fprintf(stderr, "Failed to attach BPF skeleton: %d\n", err);
        fprintf(stderr, "Make sure nvidia.ko with GPU sched hooks is loaded\n");
        goto cleanup;
    }

    /* Set up ring buffer */
    rb = ring_buffer__new(bpf_map__fd(skel->maps.events), handle_event, NULL, NULL);
    if (!rb) {
        fprintf(stderr, "Failed to create ring buffer\n");
        err = -1;
        goto cleanup;
    }

    printf("GPU Scheduler Trace started. Press Ctrl+C to stop.\n");
    printf("Tracing: nv_gpu_sched_{task_init,bind,token_request,task_destroy}\n");
    printf("---\n");

    /* Poll ring buffer */
    while (!exiting) {
        err = ring_buffer__poll(rb, 100 /* timeout, ms */);
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
    ring_buffer__free(rb);
    gpu_sched_trace_bpf__destroy(skel);
    return err < 0 ? 1 : 0;
}
