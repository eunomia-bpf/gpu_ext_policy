/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Combined PID-based Prefetch + Probabilistic LRU Eviction Policy - Userspace Loader
 *
 * Prefetch: High priority PID gets lower threshold (more bandwidth)
 * Eviction: priority/10 determines move_tail frequency (lower = more protection)
 */
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "prefetch_eviction_pid.skel.h"
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

static void print_stats(struct prefetch_eviction_pid_bpf *skel) {
    int pid_stats_fd = bpf_map__fd(skel->maps.pid_stats);
    struct pid_chunk_stats ps;
    __u32 pid;
    __u64 total_current = 0;
    __u64 total_activate = 0;
    __u64 total_used = 0;
    __u64 total_allow = 0;
    __u64 total_deny = 0;

    printf("\n=== Per-PID Statistics ===\n");

    if (g_priority_pid > 0) {
        pid = (__u32)g_priority_pid;
        if (bpf_map_lookup_elem(pid_stats_fd, &pid, &ps) == 0) {
            __u64 used_total = ps.policy_allow + ps.policy_deny;
            printf("  High priority PID %u:\n", pid);
            printf("    Current active chunks: %llu\n", ps.current_count);
            printf("    Total activated: %llu\n", ps.total_activate);
            printf("    Total used calls: %llu\n", ps.total_used);
            printf("    Policy allow (moved): %llu", ps.policy_allow);
            if (used_total > 0)
                printf(" (%.1f%%)", 100.0 * ps.policy_allow / used_total);
            printf("\n");
            printf("    Policy deny (not moved): %llu", ps.policy_deny);
            if (used_total > 0)
                printf(" (%.1f%%)", 100.0 * ps.policy_deny / used_total);
            printf("\n");

            total_current += ps.current_count;
            total_activate += ps.total_activate;
            total_used += ps.total_used;
            total_allow += ps.policy_allow;
            total_deny += ps.policy_deny;
        } else {
            printf("  High priority PID %u: no data\n", pid);
        }
    }

    if (g_low_priority_pid > 0) {
        pid = (__u32)g_low_priority_pid;
        if (bpf_map_lookup_elem(pid_stats_fd, &pid, &ps) == 0) {
            __u64 used_total = ps.policy_allow + ps.policy_deny;
            printf("  Low priority PID %u:\n", pid);
            printf("    Current active chunks: %llu\n", ps.current_count);
            printf("    Total activated: %llu\n", ps.total_activate);
            printf("    Total used calls: %llu\n", ps.total_used);
            printf("    Policy allow (moved): %llu", ps.policy_allow);
            if (used_total > 0)
                printf(" (%.1f%%)", 100.0 * ps.policy_allow / used_total);
            printf("\n");
            printf("    Policy deny (not moved): %llu", ps.policy_deny);
            if (used_total > 0)
                printf(" (%.1f%%)", 100.0 * ps.policy_deny / used_total);
            printf("\n");

            total_current += ps.current_count;
            total_activate += ps.total_activate;
            total_used += ps.total_used;
            total_allow += ps.policy_allow;
            total_deny += ps.policy_deny;
        } else {
            printf("  Low priority PID %u: no data\n", pid);
        }
    }

    printf("\n=== Summary ===\n");
    printf("  Total current chunks: %llu\n", total_current);
    printf("  Total activated: %llu\n", total_activate);
    printf("  Total used calls: %llu\n", total_used);
    __u64 grand_total = total_allow + total_deny;
    printf("  Policy allow (moved): %llu", total_allow);
    if (grand_total > 0)
        printf(" (%.1f%%)", 100.0 * total_allow / grand_total);
    printf("\n");
    printf("  Policy deny (not moved): %llu", total_deny);
    if (grand_total > 0)
        printf(" (%.1f%%)", 100.0 * total_deny / grand_total);
    printf("\n");
}

static void usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -p PID     Set high priority PID\n");
    printf("  -P N       Set high priority param (0-100, default 10)\n");
    printf("  -l PID     Set low priority PID\n");
    printf("  -L N       Set low priority param (0-100, default 80)\n");
    printf("  -d N       Set default param for other PIDs (default 50)\n");
    printf("  -h         Show this help\n");
    printf("\nCombined prefetch + eviction policy:\n");
    printf("  Prefetch: param is threshold (lower = more prefetch bandwidth)\n");
    printf("  Eviction: param/10 = decay (how many accesses before move_tail)\n");
    printf("            param=0  -> every access moves to tail (max protection)\n");
    printf("            param=50 -> every 5 accesses moves to tail\n");
    printf("            param=100-> every 10 accesses moves to tail (min protection)\n");
    printf("\nExample:\n");
    printf("  %s -p 1234 -P 10 -l 5678 -L 80\n", prog);
    printf("  High priority (PID 1234): more prefetch + more eviction protection\n");
    printf("  Low priority (PID 5678): less prefetch + less eviction protection\n");
}

int main(int argc, char **argv) {
    struct prefetch_eviction_pid_bpf *skel;
    struct bpf_link *link;
    int err;
    __u64 priority_pid = 0;
    __u64 priority_param = 10;     /* Default: 10 (prefetch thresh 10%, eviction decay 1) */
    __u64 low_priority_pid = 0;
    __u64 low_priority_param = 80; /* Default: 80 (prefetch thresh 80%, eviction decay 8) */
    __u64 default_param = 50;      /* Default: 50 (prefetch thresh 50%, eviction decay 5) */
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
                    fprintf(stderr, "Error: param must be 0-100\n");
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
                    fprintf(stderr, "Error: param must be 0-100\n");
                    return 1;
                }
                break;
            case 'd':
                default_param = atoll(optarg);
                if (default_param > 100) {
                    fprintf(stderr, "Error: param must be 0-100\n");
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

    skel = prefetch_eviction_pid_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    err = prefetch_eviction_pid_bpf__load(skel);
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

    /* Attach kprobe manually (for prefetch PID caching) */
    skel->links.prefetch_get_hint_va_block = bpf_program__attach(skel->progs.prefetch_get_hint_va_block);
    if (!skel->links.prefetch_get_hint_va_block) {
        err = -errno;
        fprintf(stderr, "Failed to attach kprobe: %s (%d)\n", strerror(-err), err);
        goto cleanup;
    }

    /* Then attach struct_ops separately */
    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_prefetch_eviction_pid);
    if (!link) {
        err = -errno;
        fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
        goto cleanup;
    }

    printf("Successfully loaded combined prefetch + eviction policy!\n");
    printf("\nConfiguration (param: prefetch threshold + eviction decay):\n");
    printf("  High priority PID: %llu (param: %llu, decay: %llu)\n",
           priority_pid, priority_param,
           priority_param / 10 > 0 ? priority_param / 10 : 1);
    printf("  Low priority PID:  %llu (param: %llu, decay: %llu)\n",
           low_priority_pid, low_priority_param,
           low_priority_param / 10 > 0 ? low_priority_param / 10 : 1);
    printf("  Default param:     %llu (decay: %llu)\n",
           default_param,
           default_param / 10 > 0 ? default_param / 10 : 1);
    printf("\nPress Ctrl-C to exit...\n");

    while (!exiting) {
        sleep(5);
        print_stats(skel);
    }

    printf("\nDetaching struct_ops...\n");
    print_stats(skel);
    bpf_link__destroy(link);

cleanup:
    prefetch_eviction_pid_bpf__destroy(skel);
    return err < 0 ? -err : 0;
}
