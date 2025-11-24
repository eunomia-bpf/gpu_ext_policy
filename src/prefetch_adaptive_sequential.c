#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <time.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "prefetch_adaptive_sequential.skel.h"
#include "cleanup_struct_ops.h"
#include "nvml_monitor.h"

static volatile bool exiting = false;
static nvmlDevice_t nvml_device = NULL;

void handle_signal(int sig) {
    exiting = true;
}

/* Get GPU PCIe throughput in MB/s using NVML */
static unsigned long long get_pcie_throughput_mbps(void) {
    if (!nvml_device) {
        return 0;
    }

    unsigned long long throughput_kbps = nvml_get_pcie_throughput_kbps(nvml_device);
    return throughput_kbps / 1024;
}

/* Prefetch percentage proportional to PCIe traffic.
 * Goal: More traffic -> more prefetch (keep GPU fed with data)
 *
 * 20 GB/s = 20480 MB/s max.
 * - High traffic (20 GB/s)  -> 100% prefetch (aggressive, all pages)
 * - Low traffic (0 MB/s)    -> 0% prefetch (conservative, no pages)
 *
 * This matches tree_iter's behavior:
 * - tree_iter: high traffic -> low threshold -> more prefetch
 * - sequential: high traffic -> high percentage -> more pages
 */
static unsigned int calculate_prefetch_percentage(unsigned long long throughput_mbps) {
    const unsigned long long max_mbps = 20480ULL; /* 20 GB/s */
    const unsigned int min_pct = 30;              /* low traffic: prefetch less */
    const unsigned int max_pct = 100;            /* high traffic: prefetch all */

    if (throughput_mbps >= max_mbps)
        return max_pct;

    /* ratio: 0.0 when idle, 1.0 when fully loaded */
    double ratio = (double)throughput_mbps / (double)max_mbps; /* 0..1 */
    unsigned int pct = (unsigned int)(min_pct + (max_pct - min_pct) * ratio + 0.5);
    if (pct < min_pct) pct = min_pct;
    if (pct > max_pct) pct = max_pct;
    return pct;
}

static void print_usage(const char *prog) {
    printf("Usage: %s [OPTIONS]\n", prog);
    printf("\nOptions:\n");
    printf("  -p PCT    Set fixed prefetch percentage (0-100), disables adaptive mode\n");
    printf("  -h        Show this help\n");
    printf("\nWithout -p, the program uses adaptive mode based on PCIe throughput.\n");
}

int main(int argc, char **argv) {
    struct prefetch_adaptive_sequential_bpf *skel;
    struct bpf_link *link;
    int err;
    int pct_map_fd;
    unsigned int key = 0;
    int fixed_pct = -1;  /* -1 means adaptive mode */
    int opt;

    while ((opt = getopt(argc, argv, "p:h")) != -1) {
        switch (opt) {
        case 'p':
            fixed_pct = atoi(optarg);
            if (fixed_pct < 0 || fixed_pct > 100) {
                fprintf(stderr, "Error: percentage must be 0-100\n");
                return 1;
            }
            break;
        case 'h':
            print_usage(argv[0]);
            return 0;
        default:
            print_usage(argv[0]);
            return 1;
        }
    }

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    /* Initialize NVML for adaptive mode */
    if (fixed_pct < 0) {
        nvml_device = nvml_init_device();
        if (!nvml_device) {
            fprintf(stderr, "Warning: Failed to initialize NVML, using fixed 100%% mode\n");
            fixed_pct = 100;
        }
    }

    /* Check and report old struct_ops instances */
    cleanup_old_struct_ops();

    /* Open BPF application */
    skel = prefetch_adaptive_sequential_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    /* Load BPF programs */
    err = prefetch_adaptive_sequential_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
        goto cleanup;
    }

    /* Get prefetch percentage map FD */
    pct_map_fd = bpf_map__fd(skel->maps.prefetch_pct_map);
    if (pct_map_fd < 0) {
        fprintf(stderr, "Failed to get prefetch_pct_map FD\n");
        err = pct_map_fd;
        goto cleanup;
    }

    /* Set initial percentage */
    unsigned int initial_pct = (fixed_pct >= 0) ? (unsigned int)fixed_pct : 100;
    err = bpf_map_update_elem(pct_map_fd, &key, &initial_pct, BPF_ANY);
    if (err) {
        fprintf(stderr, "Failed to set initial percentage: %d\n", err);
        goto cleanup;
    }

    /* Register struct_ops */
    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_adaptive_sequential);
    if (!link) {
        err = -errno;
        fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
        goto cleanup;
    }

    printf("Successfully loaded and attached BPF adaptive_sequential policy!\n");
    if (fixed_pct >= 0) {
        printf("Mode: Fixed prefetch percentage = %d%%\n", fixed_pct);
    } else {
        printf("Mode: Adaptive (based on PCIe throughput)\n");
        printf("Monitoring PCIe traffic and updating percentage every second...\n");
    }
    printf("Monitor dmesg for BPF debug output.\n");
    printf("\nPress Ctrl-C to exit and detach the policy...\n\n");

    /* Main loop */
    while (!exiting) {
        if (fixed_pct < 0) {
            /* Adaptive mode: update percentage based on PCIe throughput */
            unsigned long long throughput = get_pcie_throughput_mbps();
            unsigned int pct = calculate_prefetch_percentage(throughput);

            err = bpf_map_update_elem(pct_map_fd, &key, &pct, BPF_ANY);
            if (err) {
                fprintf(stderr, "Failed to update prefetch percentage: %d\n", err);
            } else {
                printf("[%ld] PCIe: %llu MB/s -> Prefetch: %u%%\n",
                       time(NULL), throughput, pct);
            }
        }
        sleep(1);
    }

    printf("\nDetaching struct_ops...\n");
    bpf_link__destroy(link);

cleanup:
    prefetch_adaptive_sequential_bpf__destroy(skel);

    if (nvml_device) {
        nvml_cleanup();
    }

    return err < 0 ? -err : 0;
}
