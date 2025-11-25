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
#include "prefetch_trace.skel.h"
#include "prefetch_trace_event.h"

static volatile sig_atomic_t exiting = 0;

static __u64 start_time_ns = 0;

static void print_stats(struct prefetch_trace_bpf *skel)
{
    int stats_fd = bpf_map__fd(skel->maps.stats);
    __u32 key;
    __u64 val;
    __u64 stats[3] = {0};

    // Read all stats
    for (key = 0; key < 3; key++) {
        if (bpf_map_lookup_elem(stats_fd, &key, &val) == 0) {
            stats[key] = val;
        }
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "================================================================================\n");
    fprintf(stderr, "PREFETCH TRACE SUMMARY\n");
    fprintf(stderr, "================================================================================\n");
    fprintf(stderr, "BEFORE_COMPUTE            %8llu\n", stats[0]);
    fprintf(stderr, "ON_TREE_ITER              %8llu\n", stats[1]);
    fprintf(stderr, "--------------------------------------------------------------------------------\n");
    fprintf(stderr, "TOTAL                     %8llu\n", stats[0] + stats[1]);
    if (stats[2] > 0) {
        fprintf(stderr, "DROPPED                   %8llu\n", stats[2]);
    }
    fprintf(stderr, "================================================================================\n");
}

static void sig_handler(int sig)
{
    exiting = 1;
}

static int handle_event(void *ctx, void *data, size_t data_sz)
{
    const struct prefetch_event *e = data;
    __u64 elapsed_ms;

    if (start_time_ns == 0)
        start_time_ns = e->timestamp_ns;

    elapsed_ms = (e->timestamp_ns - start_time_ns) / 1000000;

    // CSV output format:
    // time_ms,cpu,page_index,max_first,max_outer,tree_offset,leaf_count,level_count,pages_accessed
    printf("%llu,%u,%u,%u,%u,%u,%u,%u,%u\n",
           elapsed_ms,
           e->cpu,
           e->page_index,
           e->max_region_first,
           e->max_region_outer,
           e->tree_offset,
           e->tree_leaf_count,
           e->tree_level_count,
           e->pages_accessed);

    return 0;
}

int main(int argc, char **argv)
{
    struct prefetch_trace_bpf *skel;
    struct ring_buffer *rb = NULL;
    int err;

    // Set up libbpf errors and debug info callback
    libbpf_set_print(NULL);

    // Open BPF application
    skel = prefetch_trace_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    // Load & verify BPF programs
    err = prefetch_trace_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load and verify BPF skeleton: %d\n", err);
        goto cleanup;
    }

    // Attach tracepoints
    err = prefetch_trace_bpf__attach(skel);
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

    // Print CSV header
    printf("time_ms,cpu,page_index,max_first,max_outer,tree_offset,leaf_count,level_count,pages_accessed\n");

    fprintf(stderr, "Tracing prefetch hooks... Press Ctrl-C to stop.\n");

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
    prefetch_trace_bpf__destroy(skel);
    return -err;
}
