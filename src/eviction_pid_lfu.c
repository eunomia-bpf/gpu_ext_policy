#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "eviction_pid_lfu.skel.h"
#include "cleanup_struct_ops.h"

#define CONFIG_PRIORITY_PID 0
#define CONFIG_PRIORITY_QUOTA 1
#define CONFIG_LOW_PRIORITY_PID 2
#define CONFIG_LOW_PRIORITY_QUOTA 3
#define CONFIG_DEFAULT_QUOTA 4

/* Per-PID statistics structure - must match BPF side */
struct pid_chunk_stats {
    __u64 current_count;    /* Current active chunk count */
    __u64 total_activate;   /* Total chunks activated */
    __u64 total_used;       /* Total chunk_used calls */
    __u64 in_quota;         /* Times within quota (moved) */
    __u64 over_quota;       /* Times over quota (not moved) */
};

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

static void print_stats(struct eviction_pid_lfu_bpf *skel) {
    int pid_stats_fd = bpf_map__fd(skel->maps.pid_chunk_count);
    struct pid_chunk_stats ps;
    __u32 pid;
    __u64 total_current = 0;
    __u64 total_activate = 0;
    __u64 total_used = 0;
    __u64 total_in_quota = 0;
    __u64 total_over_quota = 0;

    printf("\n=== Per-PID Statistics ===\n");

    if (g_priority_pid > 0) {
        pid = (__u32)g_priority_pid;
        if (bpf_map_lookup_elem(pid_stats_fd, &pid, &ps) == 0) {
            __u64 used_total = ps.in_quota + ps.over_quota;
            printf("  High priority PID %u:\n", pid);
            printf("    Current active chunks: %llu\n", ps.current_count);
            printf("    Total activated: %llu\n", ps.total_activate);
            printf("    Total used calls: %llu\n", ps.total_used);
            printf("    In quota (moved): %llu", ps.in_quota);
            if (used_total > 0)
                printf(" (%.1f%%)", 100.0 * ps.in_quota / used_total);
            printf("\n");
            printf("    Over quota (not moved): %llu", ps.over_quota);
            if (used_total > 0)
                printf(" (%.1f%%)", 100.0 * ps.over_quota / used_total);
            printf("\n");

            total_current += ps.current_count;
            total_activate += ps.total_activate;
            total_used += ps.total_used;
            total_in_quota += ps.in_quota;
            total_over_quota += ps.over_quota;
        } else {
            printf("  High priority PID %u: no data\n", pid);
        }
    }

    if (g_low_priority_pid > 0) {
        pid = (__u32)g_low_priority_pid;
        if (bpf_map_lookup_elem(pid_stats_fd, &pid, &ps) == 0) {
            __u64 used_total = ps.in_quota + ps.over_quota;
            printf("  Low priority PID %u:\n", pid);
            printf("    Current active chunks: %llu\n", ps.current_count);
            printf("    Total activated: %llu\n", ps.total_activate);
            printf("    Total used calls: %llu\n", ps.total_used);
            printf("    In quota (moved): %llu", ps.in_quota);
            if (used_total > 0)
                printf(" (%.1f%%)", 100.0 * ps.in_quota / used_total);
            printf("\n");
            printf("    Over quota (not moved): %llu", ps.over_quota);
            if (used_total > 0)
                printf(" (%.1f%%)", 100.0 * ps.over_quota / used_total);
            printf("\n");

            total_current += ps.current_count;
            total_activate += ps.total_activate;
            total_used += ps.total_used;
            total_in_quota += ps.in_quota;
            total_over_quota += ps.over_quota;
        } else {
            printf("  Low priority PID %u: no data\n", pid);
        }
    }

    printf("\n=== Summary ===\n");
    printf("  Total current chunks: %llu\n", total_current);
    printf("  Total activated: %llu\n", total_activate);
    printf("  Total used calls: %llu\n", total_used);
    __u64 grand_total = total_in_quota + total_over_quota;
    printf("  In quota (moved): %llu", total_in_quota);
    if (grand_total > 0)
        printf(" (%.1f%%)", 100.0 * total_in_quota / grand_total);
    printf("\n");
    printf("  Over quota (not moved): %llu", total_over_quota);
    if (grand_total > 0)
        printf(" (%.1f%%)", 100.0 * total_over_quota / grand_total);
    printf("\n");
}

static void usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -p PID     Set high priority PID\n");
    printf("  -P PERCENT Set high priority quota percentage (0-100, 0=unlimited)\n");
    printf("  -l PID     Set low priority PID\n");
    printf("  -L PERCENT Set low priority quota percentage (0-100)\n");
    printf("  -d PERCENT Set default quota percentage for other PIDs (0=unlimited)\n");
    printf("  -h         Show this help\n");
    printf("\nQuota-based eviction (percentage of total chunks):\n");
    printf("  - Within quota: chunks are moved to tail (LRU, protected)\n");
    printf("  - Over quota: chunks are not moved (easier to evict)\n");
    printf("\nExample:\n");
    printf("  %s -p 1234 -P 80 -l 5678 -L 20\n", prog);
    printf("  High priority (PID 1234): 80%% quota (protected)\n");
    printf("  Low priority (PID 5678): 20%% quota (excess easier to evict)\n");
}

int main(int argc, char **argv) {
    struct eviction_pid_lfu_bpf *skel;
    struct bpf_link *link;
    int err;
    __u64 priority_pid = 0;
    __u64 priority_quota = 0;      /* 0 = unlimited */
    __u64 low_priority_pid = 0;
    __u64 low_priority_quota = 100; /* Default: 100 chunks */
    __u64 default_quota = 0;       /* 0 = unlimited */
    int opt;

    while ((opt = getopt(argc, argv, "p:P:l:L:d:h")) != -1) {
        switch (opt) {
            case 'p':
                priority_pid = atoi(optarg);
                g_priority_pid = priority_pid;
                break;
            case 'P':
                priority_quota = atoll(optarg);
                break;
            case 'l':
                low_priority_pid = atoi(optarg);
                g_low_priority_pid = low_priority_pid;
                break;
            case 'L':
                low_priority_quota = atoll(optarg);
                break;
            case 'd':
                default_quota = atoll(optarg);
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

    skel = eviction_pid_lfu_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    err = eviction_pid_lfu_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
        goto cleanup;
    }

    /* Set configuration */
    int config_fd = bpf_map__fd(skel->maps.config);
    __u32 key;

    key = CONFIG_PRIORITY_PID;
    bpf_map_update_elem(config_fd, &key, &priority_pid, BPF_ANY);

    key = CONFIG_PRIORITY_QUOTA;
    bpf_map_update_elem(config_fd, &key, &priority_quota, BPF_ANY);

    key = CONFIG_LOW_PRIORITY_PID;
    bpf_map_update_elem(config_fd, &key, &low_priority_pid, BPF_ANY);

    key = CONFIG_LOW_PRIORITY_QUOTA;
    bpf_map_update_elem(config_fd, &key, &low_priority_quota, BPF_ANY);

    key = CONFIG_DEFAULT_QUOTA;
    bpf_map_update_elem(config_fd, &key, &default_quota, BPF_ANY);

    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_pid_lfu);
    if (!link) {
        err = -errno;
        fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
        goto cleanup;
    }

    printf("Successfully loaded quota-based eviction policy!\n");
    printf("\nConfiguration (quota as percentage of total chunks):\n");
    printf("  High priority PID: %llu (quota: %s)\n",
           priority_pid, priority_quota == 0 ? "unlimited" : "set");
    if (priority_quota > 0)
        printf("    Quota: %llu%%\n", priority_quota);
    printf("  Low priority PID:  %llu (quota: %llu%%)\n",
           low_priority_pid, low_priority_quota);
    printf("  Default quota:     %s\n",
           default_quota == 0 ? "unlimited" : "set");
    if (default_quota > 0)
        printf("    Quota: %llu%%\n", default_quota);
    printf("\nPress Ctrl-C to exit...\n");

    while (!exiting) {
        sleep(5);
        print_stats(skel);
    }

    printf("\nDetaching struct_ops...\n");
    print_stats(skel);
    bpf_link__destroy(link);

cleanup:
    eviction_pid_lfu_bpf__destroy(skel);
    return err < 0 ? -err : 0;
}
