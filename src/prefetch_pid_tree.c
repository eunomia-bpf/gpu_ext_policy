/* SPDX-License-Identifier: GPL-2.0 */
/*
 * PID-based Prefetch Policy - Userspace Loader
 *
 * Allocates prefetch bandwidth based on process priority.
 * High priority PID gets lower threshold (more prefetch bandwidth).
 * Low priority PID gets higher threshold (less prefetch bandwidth).
 */
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "prefetch_pid_tree.skel.h"
#include "cleanup_struct_ops.h"
#include "eviction_common.h"

static __u64 g_priority_pid = 0;
static __u64 g_low_priority_pid = 0;

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
    return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;

void handle_signal(int sig) {
    exiting = true;
}

static void print_stats(struct prefetch_pid_tree_bpf *skel) {
    int pid_stats_fd = bpf_map__fd(skel->maps.pid_stats);
    struct pid_chunk_stats ps;
    __u32 pid;
    __u64 total_activate = 0;
    __u64 total_allow = 0;
    __u64 total_deny = 0;

    printf("\n=== Per-PID Statistics ===\n");

    if (g_priority_pid > 0) {
        pid = (__u32)g_priority_pid;
        if (bpf_map_lookup_elem(pid_stats_fd, &pid, &ps) == 0) {
            __u64 total = ps.policy_allow + ps.policy_deny;
            printf("  High priority PID %u:\n", pid);
            printf("    Current active chunks: %llu\n", ps.current_count);
            printf("    Total activated: %llu\n", ps.total_activate);
            printf("    Total used calls: %llu\n", ps.total_used);
            printf("    Policy allow (prefetched): %llu", ps.policy_allow);
            if (total > 0)
                printf(" (%.1f%%)", 100.0 * ps.policy_allow / total);
            printf("\n");
            printf("    Policy deny (skipped): %llu", ps.policy_deny);
            if (total > 0)
                printf(" (%.1f%%)", 100.0 * ps.policy_deny / total);
            printf("\n");

            total_activate += ps.total_activate;
            total_allow += ps.policy_allow;
            total_deny += ps.policy_deny;
        } else {
            printf("  High priority PID %u: no data\n", pid);
        }
    }

    if (g_low_priority_pid > 0) {
        pid = (__u32)g_low_priority_pid;
        if (bpf_map_lookup_elem(pid_stats_fd, &pid, &ps) == 0) {
            __u64 total = ps.policy_allow + ps.policy_deny;
            printf("  Low priority PID %u:\n", pid);
            printf("    Current active chunks: %llu\n", ps.current_count);
            printf("    Total activated: %llu\n", ps.total_activate);
            printf("    Total used calls: %llu\n", ps.total_used);
            printf("    Policy allow (prefetched): %llu", ps.policy_allow);
            if (total > 0)
                printf(" (%.1f%%)", 100.0 * ps.policy_allow / total);
            printf("\n");
            printf("    Policy deny (skipped): %llu", ps.policy_deny);
            if (total > 0)
                printf(" (%.1f%%)", 100.0 * ps.policy_deny / total);
            printf("\n");

            total_activate += ps.total_activate;
            total_allow += ps.policy_allow;
            total_deny += ps.policy_deny;
        } else {
            printf("  Low priority PID %u: no data\n", pid);
        }
    }

    printf("\n=== Summary ===\n");
    printf("  Total activated: %llu\n", total_activate);
    __u64 grand_total = total_allow + total_deny;
    printf("  Policy allow (prefetched): %llu", total_allow);
    if (grand_total > 0)
        printf(" (%.1f%%)", 100.0 * total_allow / grand_total);
    printf("\n");
    printf("  Policy deny (skipped): %llu", total_deny);
    if (grand_total > 0)
        printf(" (%.1f%%)", 100.0 * total_deny / grand_total);
    printf("\n");
}

static void usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\nPID-based Prefetch Bandwidth Allocation Policy\n");
    printf("\nOptions:\n");
    printf("  -p PID     Set high priority PID\n");
    printf("  -P N       Set high priority threshold (0-100, default: 20)\n");
    printf("  -l PID     Set low priority PID\n");
    printf("  -L N       Set low priority threshold (0-100, default: 80)\n");
    printf("  -d N       Set default threshold for other PIDs (default: 50)\n");
    printf("  -h         Show this help\n");
    printf("\nHow it works:\n");
    printf("  Lower threshold = easier to prefetch = MORE bandwidth\n");
    printf("  Higher threshold = harder to prefetch = LESS bandwidth\n");
    printf("\nExample:\n");
    printf("  %s -p 1234 -P 20 -l 5678 -L 80\n", prog);
}

int main(int argc, char **argv) {
    struct prefetch_pid_tree_bpf *skel;
    struct bpf_link *link;
    int err;
    __u64 priority_pid = 0;
    __u64 priority_param = 20;     /* Default: 20% threshold (easy to prefetch) */
    __u64 low_priority_pid = 0;
    __u64 low_priority_param = 80; /* Default: 80% threshold (hard to prefetch) */
    __u64 default_param = 50;      /* Default: 50% threshold */
    int opt;

    while ((opt = getopt(argc, argv, "p:P:l:L:d:h")) != -1) {
        switch (opt) {
            case 'p':
                priority_pid = atoi(optarg);
                g_priority_pid = priority_pid;
                break;
            case 'P':
                priority_param = atoll(optarg);
                if (priority_param > 100) {
                    fprintf(stderr, "Error: threshold must be 0-100\n");
                    return 1;
                }
                break;
            case 'l':
                low_priority_pid = atoi(optarg);
                g_low_priority_pid = low_priority_pid;
                break;
            case 'L':
                low_priority_param = atoll(optarg);
                if (low_priority_param > 100) {
                    fprintf(stderr, "Error: threshold must be 0-100\n");
                    return 1;
                }
                break;
            case 'd':
                default_param = atoll(optarg);
                if (default_param > 100) {
                    fprintf(stderr, "Error: threshold must be 0-100\n");
                    return 1;
                }
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

    skel = prefetch_pid_tree_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    err = prefetch_pid_tree_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
        goto cleanup;
    }

    /* Set configuration */
    int config_fd = bpf_map__fd(skel->maps.config);
    __u32 key;

    key = CONFIG_PRIORITY_PID;
    bpf_map_update_elem(config_fd, &key, &priority_pid, BPF_ANY);

    key = CONFIG_PRIORITY_PARAM;
    bpf_map_update_elem(config_fd, &key, &priority_param, BPF_ANY);

    key = CONFIG_LOW_PRIORITY_PID;
    bpf_map_update_elem(config_fd, &key, &low_priority_pid, BPF_ANY);

    key = CONFIG_LOW_PRIORITY_PARAM;
    bpf_map_update_elem(config_fd, &key, &low_priority_param, BPF_ANY);

    key = CONFIG_DEFAULT_PARAM;
    bpf_map_update_elem(config_fd, &key, &default_param, BPF_ANY);

    /* Attach kprobe manually (not through __attach which also attaches struct_ops) */
    skel->links.prefetch_get_hint_va_block = bpf_program__attach(skel->progs.prefetch_get_hint_va_block);
    if (!skel->links.prefetch_get_hint_va_block) {
        err = -errno;
        fprintf(stderr, "Failed to attach kprobe: %s (%d)\n", strerror(-err), err);
        goto cleanup;
    }

    /* Then attach struct_ops separately */
    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_prefetch_pid_tree);
    if (!link) {
        err = -errno;
        fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
        goto cleanup;
    }

    printf("Successfully loaded PID-based prefetch policy!\n");
    printf("\nConfiguration (lower threshold = more prefetch bandwidth):\n");
    printf("  High priority PID: %llu (threshold: %llu%%)\n", priority_pid, priority_param);
    printf("  Low priority PID:  %llu (threshold: %llu%%)\n", low_priority_pid, low_priority_param);
    printf("  Default threshold: %llu%%\n", default_param);
    printf("\nPress Ctrl-C to exit...\n");

    while (!exiting) {
        sleep(5);
        print_stats(skel);
    }

    printf("\nDetaching struct_ops...\n");
    print_stats(skel);
    bpf_link__destroy(link);

cleanup:
    prefetch_pid_tree_bpf__destroy(skel);
    return err < 0 ? -err : 0;
}
