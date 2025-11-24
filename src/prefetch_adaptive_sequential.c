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

/* Configuration */
static struct {
    int fixed_pct;           /* -1 means adaptive mode */
    unsigned int min_pct;    /* min percentage for adaptive mode */
    unsigned int max_pct;    /* max percentage for adaptive mode */
    unsigned long long max_mbps;  /* max PCIe throughput for scaling */
    int invert;              /* invert the adaptive logic */
} config = {
    .fixed_pct = -1,
    .min_pct = 30,
    .max_pct = 100,
    .max_mbps = 20480ULL,    /* 20 GB/s */
    .invert = 0,
};

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

/* Calculate prefetch percentage based on PCIe throughput
 *
 * Normal mode (invert=0):
 *   High traffic -> high percentage -> more prefetch
 *   Low traffic  -> low percentage  -> less prefetch
 *
 * Inverted mode (invert=1):
 *   High traffic -> low percentage  -> less prefetch (bandwidth constrained)
 *   Low traffic  -> high percentage -> more prefetch (bandwidth available)
 */
static unsigned int calculate_prefetch_percentage(unsigned long long throughput_mbps) {
    if (throughput_mbps >= config.max_mbps) {
        return config.invert ? config.min_pct : config.max_pct;
    }

    double ratio = (double)throughput_mbps / (double)config.max_mbps; /* 0..1 */

    if (config.invert) {
        ratio = 1.0 - ratio;  /* invert: high traffic -> low ratio */
    }

    unsigned int pct = (unsigned int)(config.min_pct +
                                      (config.max_pct - config.min_pct) * ratio + 0.5);
    if (pct < config.min_pct) pct = config.min_pct;
    if (pct > config.max_pct) pct = config.max_pct;
    return pct;
}

static void print_usage(const char *prog) {
    printf("Usage: %s [OPTIONS]\n", prog);
    printf("\nPrefetch Adaptive Sequential Policy\n");
    printf("Controls what percentage of max_prefetch_region to prefetch.\n");
    printf("\nOptions:\n");
    printf("  -p PCT        Set fixed prefetch percentage (0-100), disables adaptive mode\n");
    printf("  -m MIN        Set minimum percentage for adaptive mode (default: %u)\n", config.min_pct);
    printf("  -M MAX        Set maximum percentage for adaptive mode (default: %u)\n", config.max_pct);
    printf("  -b MBPS       Set max PCIe bandwidth for scaling in MB/s (default: %llu)\n", config.max_mbps);
    printf("  -i            Invert adaptive logic (high traffic -> less prefetch)\n");
    printf("  -h            Show this help\n");
    printf("\nExamples:\n");
    printf("  %s -p 100              # Fixed 100%% prefetch (like always_max)\n", prog);
    printf("  %s -p 0                # Fixed 0%% prefetch (like none)\n", prog);
    printf("  %s -p 50               # Fixed 50%% prefetch\n", prog);
    printf("  %s                     # Adaptive mode (default)\n", prog);
    printf("  %s -m 20 -M 80         # Adaptive with custom range 20-80%%\n", prog);
    printf("  %s -i                  # Inverted: less prefetch when busy\n", prog);
    printf("\nWithout -p, uses adaptive mode based on PCIe throughput.\n");
}

int main(int argc, char **argv) {
    struct prefetch_adaptive_sequential_bpf *skel;
    struct bpf_link *link;
    int err;
    int pct_map_fd;
    unsigned int key = 0;
    int opt;

    while ((opt = getopt(argc, argv, "p:m:M:b:ih")) != -1) {
        switch (opt) {
        case 'p':
            config.fixed_pct = atoi(optarg);
            if (config.fixed_pct < 0 || config.fixed_pct > 100) {
                fprintf(stderr, "Error: percentage must be 0-100\n");
                return 1;
            }
            break;
        case 'm':
            config.min_pct = (unsigned int)atoi(optarg);
            if (config.min_pct > 100) {
                fprintf(stderr, "Error: min percentage must be 0-100\n");
                return 1;
            }
            break;
        case 'M':
            config.max_pct = (unsigned int)atoi(optarg);
            if (config.max_pct > 100) {
                fprintf(stderr, "Error: max percentage must be 0-100\n");
                return 1;
            }
            break;
        case 'b':
            config.max_mbps = (unsigned long long)atoll(optarg);
            break;
        case 'i':
            config.invert = 1;
            break;
        case 'h':
            print_usage(argv[0]);
            return 0;
        default:
            print_usage(argv[0]);
            return 1;
        }
    }

    /* Validate min/max */
    if (config.min_pct > config.max_pct) {
        fprintf(stderr, "Error: min percentage (%u) cannot be greater than max (%u)\n",
                config.min_pct, config.max_pct);
        return 1;
    }

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    /* Initialize NVML for adaptive mode */
    if (config.fixed_pct < 0) {
        nvml_device = nvml_init_device();
        if (!nvml_device) {
            fprintf(stderr, "Warning: Failed to initialize NVML, using fixed 100%% mode\n");
            config.fixed_pct = 100;
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
    unsigned int initial_pct = (config.fixed_pct >= 0) ? (unsigned int)config.fixed_pct : config.max_pct;
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
    if (config.fixed_pct >= 0) {
        printf("Mode: Fixed prefetch percentage = %d%%\n", config.fixed_pct);
    } else {
        printf("Mode: Adaptive (based on PCIe throughput)\n");
        printf("  Range: %u%% - %u%%\n", config.min_pct, config.max_pct);
        printf("  Max bandwidth: %llu MB/s\n", config.max_mbps);
        printf("  Invert: %s\n", config.invert ? "yes" : "no");
        printf("Monitoring PCIe traffic and updating percentage every second...\n");
    }
    printf("Monitor dmesg for BPF debug output.\n");
    printf("\nPress Ctrl-C to exit and detach the policy...\n\n");

    /* Main loop */
    while (!exiting) {
        if (config.fixed_pct < 0) {
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
