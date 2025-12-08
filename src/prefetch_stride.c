/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Stride-based Prefetch Policy - Userspace Loader
 *
 * Detects stride access patterns and prefetches accordingly.
 */
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "prefetch_stride.skel.h"
#include "cleanup_struct_ops.h"

/* Configuration keys - must match BPF program */
#define CONFIG_CONFIDENCE_THRESHOLD  0
#define CONFIG_PREFETCH_PAGES        1
#define CONFIG_MAX_STRIDE            2

/* Stats structure - must match BPF program */
struct stride_stats {
    __u64 total_faults;
    __u64 prefetch_issued;
    __u64 stride_detected;
    __u64 no_prefetch;
};

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
    return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;

void handle_signal(int sig) {
    exiting = true;
}

static void print_stats(struct prefetch_stride_bpf *skel) {
    int stats_fd = bpf_map__fd(skel->maps.global_stats);
    struct stride_stats stats;
    __u32 key = 0;

    printf("\n=== Stride Prefetch Statistics ===\n");

    if (bpf_map_lookup_elem(stats_fd, &key, &stats) == 0) {
        printf("  Total page faults:    %llu\n", stats.total_faults);
        printf("  Prefetches issued:    %llu", stats.prefetch_issued);
        if (stats.total_faults > 0)
            printf(" (%.1f%%)", 100.0 * stats.prefetch_issued / stats.total_faults);
        printf("\n");
        printf("  No prefetch:          %llu", stats.no_prefetch);
        if (stats.total_faults > 0)
            printf(" (%.1f%%)", 100.0 * stats.no_prefetch / stats.total_faults);
        printf("\n");
    } else {
        printf("  No stats available\n");
    }
}

static void usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -t N    Confidence threshold (default: 2)\n");
    printf("          Number of consecutive matching strides before prefetching\n");
    printf("  -n N    Prefetch pages (default: 2)\n");
    printf("          Number of pages to prefetch at predicted location\n");
    printf("  -m N    Max stride (default: 128)\n");
    printf("          Maximum allowed stride in pages (larger strides ignored)\n");
    printf("  -h      Show this help\n");
    printf("\nStride-based prefetch policy:\n");
    printf("  - Detects regular stride access patterns (e.g., 0,4,8,12,...)\n");
    printf("  - After detecting stable stride, prefetches next predicted page\n");
    printf("  - Confidence decays by 1 when stride changes (robust to outliers)\n");
    printf("\nExample:\n");
    printf("  %s -t 2 -n 4 -m 64\n", prog);
    printf("  Prefetch 4 pages after 2 consecutive matching strides (max stride 64)\n");
}

int main(int argc, char **argv) {
    struct prefetch_stride_bpf *skel;
    struct bpf_link *link;
    int err;
    __u64 confidence_threshold = 2;
    __u64 prefetch_pages = 2;
    __u64 max_stride = 128;
    int opt;

    while ((opt = getopt(argc, argv, "t:n:m:h")) != -1) {
        switch (opt) {
            case 't':
                confidence_threshold = atoll(optarg);
                break;
            case 'n':
                prefetch_pages = atoll(optarg);
                if (prefetch_pages == 0) prefetch_pages = 1;
                break;
            case 'm':
                max_stride = atoll(optarg);
                if (max_stride == 0) max_stride = 1;
                break;
            case 'h':
            default:
                usage(argv[0]);
                return opt == 'h' ? 0 : 1;
        }
    }

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    libbpf_set_print(libbpf_print_fn);

    cleanup_old_struct_ops();

    skel = prefetch_stride_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    err = prefetch_stride_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
        goto cleanup;
    }

    /* Set configuration */
    int config_fd = bpf_map__fd(skel->maps.config);
    __u32 key;

    key = CONFIG_CONFIDENCE_THRESHOLD;
    bpf_map_update_elem(config_fd, &key, &confidence_threshold, BPF_ANY);

    key = CONFIG_PREFETCH_PAGES;
    bpf_map_update_elem(config_fd, &key, &prefetch_pages, BPF_ANY);

    key = CONFIG_MAX_STRIDE;
    bpf_map_update_elem(config_fd, &key, &max_stride, BPF_ANY);

    /* Attach kprobe for va_block tracking */
    skel->links.prefetch_get_hint_va_block = bpf_program__attach(skel->progs.prefetch_get_hint_va_block);
    if (!skel->links.prefetch_get_hint_va_block) {
        err = -errno;
        fprintf(stderr, "Failed to attach kprobe: %s (%d)\n", strerror(-err), err);
        goto cleanup;
    }

    /* Attach struct_ops */
    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_stride);
    if (!link) {
        err = -errno;
        fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
        goto cleanup;
    }

    printf("Successfully loaded stride-based prefetch policy!\n");
    printf("\nConfiguration:\n");
    printf("  Confidence threshold: %llu\n", confidence_threshold);
    printf("  Prefetch pages:       %llu\n", prefetch_pages);
    printf("  Max stride:           %llu pages\n", max_stride);
    printf("\nPress Ctrl-C to exit...\n");

    while (!exiting) {
        sleep(5);
        print_stats(skel);
    }

    printf("\nDetaching struct_ops...\n");
    print_stats(skel);
    bpf_link__destroy(link);

cleanup:
    prefetch_stride_bpf__destroy(skel);
    return err < 0 ? -err : 0;
}
