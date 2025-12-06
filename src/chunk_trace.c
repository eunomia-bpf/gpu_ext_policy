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
#include "chunk_trace_event.h"

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
    return vfprintf(stderr, format, args);
}

static volatile sig_atomic_t exiting = 0;

// Hook type names
static const char *hook_names[] = {
    [0] = "UNKNOWN",
    [1] = "ACTIVATE",
    [2] = "POPULATE",
    [3] = "EVICTION_PREPARE",
};

// Statistics
static __u64 stats[4] = {0};
static __u64 va_block_count = 0;  // Count events with VA block info
static __u64 va_block_null = 0;   // Count events without VA block info
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

    const char *hook_name = (e->hook_type < 4) ? hook_names[e->hook_type] : "UNKNOWN";

    // CSV output format:
    // time_ms,hook_type,pid,owner_pid,va_space,cpu,chunk_addr,list_addr,va_block,va_start,va_end,va_page_index

    if (e->hook_type == 4) {
        // EVICTION_PREPARE: chunk_addr is used_list, list_addr is unused_list
        printf("%llu,%s,%u,,,%u,0x%llx,0x%llx,,,,%u\n",
               elapsed_ms,
               hook_name,
               e->pid,
               e->cpu,
               e->chunk_addr,  // used_list
               e->list_addr,   // unused_list
               e->va_page_index);
    } else {
        // Regular hooks
        if (e->va_block != 0) {
            printf("%llu,%s,%u,%u,0x%llx,%u,0x%llx,0x%llx,0x%llx,0x%llx,0x%llx,%u\n",
                   elapsed_ms,
                   hook_name,
                   e->pid,
                   e->owner_pid,
                   e->va_space,
                   e->cpu,
                   e->chunk_addr,
                   e->list_addr,
                   e->va_block,
                   e->va_start,
                   e->va_end,
                   e->va_page_index);
            va_block_count++;
        } else {
            printf("%llu,%s,%u,,,%u,0x%llx,0x%llx,,,,%u\n",
                   elapsed_ms,
                   hook_name,
                   e->pid,
                   e->cpu,
                   e->chunk_addr,
                   e->list_addr,
                   e->va_page_index);
            va_block_null++;
        }
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
    for (key = 0; key < 4; key++) {
        if (bpf_map_lookup_elem(stats_fd, &key, &val) == 0) {
            stats[key] = val;
        }
    }

    printf("ACTIVATE                  %8llu\n", stats[0]);
    printf("POPULATE                  %8llu\n", stats[1]);
    printf("EVICTION_PREPARE          %8llu\n", stats[2]);
    printf("--------------------------------------------------------------------------------\n");
    printf("TOTAL                     %8llu\n",
           stats[0] + stats[1] + stats[2]);

    if (stats[3] > 0) {
        printf("\n⚠️  Dropped events:          %8llu\n", stats[3]);
    }

    printf("================================================================================\n");
    printf("VA BLOCK TRACKING\n");
    printf("================================================================================\n");
    printf("With VA block:            %8llu\n", va_block_count);
    printf("Without VA block (NULL):  %8llu\n", va_block_null);

    if (va_block_count + va_block_null > 0) {
        printf("Coverage:                 %7.1f%%\n",
               va_block_count * 100.0 / (va_block_count + va_block_null));
    }

    if (va_block_count == 0 && va_block_null > 0) {
        printf("\n⚠️  WARNING: No VA blocks detected!\n");
        printf("   This likely means the struct offset is incorrect.\n");
        printf("   Expected offset to va_block: ~40-48 bytes from chunk base\n");
    } else if (va_block_count > 0) {
        printf("\n✓ VA blocks successfully tracked!\n");
    }

    printf("================================================================================\n");
}

int main(int argc, char **argv)
{
    struct chunk_trace_bpf *skel;
    struct ring_buffer *rb = NULL;
    int err;

    // Set up libbpf errors and debug info callback
    libbpf_set_print(libbpf_print_fn);

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

    // Print CSV header
    printf("time_ms,hook_type,pid,owner_pid,va_space,cpu,chunk_addr,list_addr,va_block,va_start,va_end,va_page_index\n");

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

    // print_stats(skel);  // Disabled to keep CSV output clean

cleanup:
    ring_buffer__free(rb);
    chunk_trace_bpf__destroy(skel);
    return -err;
}
