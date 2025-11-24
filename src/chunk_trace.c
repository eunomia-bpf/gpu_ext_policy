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
#include "chunk_trace.skel.h"

static volatile sig_atomic_t exiting = 0;

// Hook type names
static const char *hook_names[] = {
    [0] = "UNKNOWN",
    [1] = "ACTIVATE",
    [2] = "POPULATE",
    [3] = "DEPOPULATE",
    [4] = "EVICTION_PREPARE",
};

// Event structure (must match BPF side)
struct hook_event {
    __u64 timestamp_ns;
    __u32 hook_type;
    __u32 cpu;
    __u64 chunk_addr;
    __u64 list_addr;
};

// Statistics
static __u64 stats[5] = {0};
static __u64 start_time_ns = 0;

static void sig_handler(int sig)
{
    exiting = 1;
}

static __u64 get_time_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (__u64)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static int handle_event(void *ctx, void *data, size_t data_sz)
{
    const struct hook_event *e = data;
    __u64 elapsed_ms;

    if (start_time_ns == 0)
        start_time_ns = e->timestamp_ns;

    elapsed_ms = (e->timestamp_ns - start_time_ns) / 1000000;

    const char *hook_name = (e->hook_type < 5) ? hook_names[e->hook_type] : "UNKNOWN";

    if (e->hook_type == 4) {
        // EVICTION_PREPARE: chunk_addr is used_list, list_addr is unused_list
        printf("%-10llu %-30s %-18s used=0x%llx unused=0x%llx\n",
               elapsed_ms,
               hook_name,
               "---",
               e->chunk_addr,
               e->list_addr);
    } else {
        // Regular hooks
        printf("%-10llu %-30s 0x%-16llx 0x%llx\n",
               elapsed_ms,
               hook_name,
               e->chunk_addr,
               e->list_addr);
    }

    return 0;
}

static void print_stats(struct chunk_trace_bpf *skel)
{
    int stats_fd = bpf_map__fd(skel->maps.stats);
    __u32 key;
    __u64 val;

    printf("\n");
    printf("================================================================================\n");
    printf("BPF HOOK SUMMARY\n");
    printf("================================================================================\n");
    printf("\n");
    printf("Hook                         Calls\n");
    printf("--------------------------------------------------------------------------------\n");

    // Read all stats
    for (key = 0; key < 5; key++) {
        if (bpf_map_lookup_elem(stats_fd, &key, &val) == 0) {
            stats[key] = val;
        }
    }

    printf("ACTIVATE                  %8llu\n", stats[0]);
    printf("POPULATE                  %8llu\n", stats[1]);
    printf("DEPOPULATE                %8llu\n", stats[2]);
    printf("EVICTION_PREPARE          %8llu\n", stats[3]);
    printf("--------------------------------------------------------------------------------\n");
    printf("TOTAL                     %8llu\n",
           stats[0] + stats[1] + stats[2] + stats[3]);

    if (stats[4] > 0) {
        printf("\n⚠️  Dropped events:          %8llu\n", stats[4]);
    }

    printf("================================================================================\n");
}

int main(int argc, char **argv)
{
    struct chunk_trace_bpf *skel;
    struct ring_buffer *rb = NULL;
    int err;

    // Set up libbpf errors and debug info callback
    libbpf_set_print(NULL);

    // Open BPF application
    skel = chunk_trace_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    // Load & verify BPF programs
    err = chunk_trace_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load and verify BPF skeleton: %d\n", err);
        goto cleanup;
    }

    // Attach tracepoints
    err = chunk_trace_bpf__attach(skel);
    if (err) {
        fprintf(stderr, "Failed to attach BPF skeleton: %d\n", err);
        goto cleanup;
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

    printf("Tracing BPF hooks... Hit Ctrl-C to end.\n");
    printf("%-10s %-30s %-18s %-15s\n", "TIME(ms)", "HOOK", "CHUNK_ADDR", "LIST_ADDR");

    // Process events
    while (!exiting) {
        err = ring_buffer__poll(rb, 100 /* timeout, ms */);
        // Ctrl-C will cause -EINTR
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
    chunk_trace_bpf__destroy(skel);
    return -err;
}
