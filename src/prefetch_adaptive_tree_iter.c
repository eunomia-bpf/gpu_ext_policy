#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <time.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "prefetch_adaptive_tree_iter.skel.h"
#include "cleanup_struct_ops.h"
#include "nvml_monitor.h"

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
    return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;
static nvmlDevice_t nvml_device = NULL;

/* Configuration */
static struct {
    int fixed_thresh;        /* -1 means adaptive mode */
    unsigned int min_thresh; /* min threshold for adaptive mode */
    unsigned int max_thresh; /* max threshold for adaptive mode */
    unsigned long long max_mbps;  /* max PCIe throughput for scaling */
    int invert;              /* invert the adaptive logic */
} config = {
    .fixed_thresh = -1,
    .min_thresh = 0,         /* high traffic: easier to prefetch */
    .max_thresh = 100,       /* low traffic: harder to prefetch */
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

/* Calculate threshold based on PCIe throughput
 *
 * BPF logic: counter * 100 > subregion_pages * threshold
 *   - Low threshold  -> easier to pass -> more prefetch
 *   - High threshold -> harder to pass -> less prefetch
 *
 * Normal mode (invert=0):
 *   High traffic -> low threshold  -> more prefetch (keep GPU fed)
 *   Low traffic  -> high threshold -> less prefetch (save bandwidth)
 *
 * Inverted mode (invert=1):
 *   High traffic -> high threshold -> less prefetch (bandwidth constrained)
 *   Low traffic  -> low threshold  -> more prefetch (bandwidth available)
 */
static unsigned int calculate_threshold(unsigned long long throughput_mbps) {
    if (throughput_mbps >= config.max_mbps) {
        return config.invert ? config.max_thresh : config.min_thresh;
    }

    double ratio = (double)throughput_mbps / (double)config.max_mbps; /* 0..1 */

    unsigned int thresh;
    if (config.invert) {
        /* Inverted: high traffic -> high threshold */
        thresh = (unsigned int)(config.min_thresh +
                               (config.max_thresh - config.min_thresh) * ratio + 0.5);
    } else {
        /* Normal: high traffic -> low threshold (inverse relationship) */
        double inv = 1.0 - ratio;
        thresh = (unsigned int)(config.min_thresh +
                               (config.max_thresh - config.min_thresh) * inv + 0.5);
    }

    if (thresh < config.min_thresh) thresh = config.min_thresh;
    if (thresh > config.max_thresh) thresh = config.max_thresh;
    return thresh;
}

static void print_usage(const char *prog) {
    printf("Usage: %s [OPTIONS]\n", prog);
    printf("\nPrefetch Adaptive Tree Iter Policy\n");
    printf("Controls threshold for prefetching based on access density.\n");
    printf("BPF logic: prefetch if (counter * 100 > subregion_pages * threshold)\n");
    printf("  - Low threshold  -> easier to prefetch\n");
    printf("  - High threshold -> harder to prefetch\n");
    printf("\nOptions:\n");
    printf("  -t THRESH     Set fixed threshold (0-100), disables adaptive mode\n");
    printf("  -m MIN        Set minimum threshold for adaptive mode (default: %u)\n", config.min_thresh);
    printf("  -M MAX        Set maximum threshold for adaptive mode (default: %u)\n", config.max_thresh);
    printf("  -b MBPS       Set max PCIe bandwidth for scaling in MB/s (default: %llu)\n", config.max_mbps);
    printf("  -i            Invert adaptive logic (high traffic -> higher threshold)\n");
    printf("  -h            Show this help\n");
    printf("\nExamples:\n");
    printf("  %s -t 0               # Fixed threshold 0%% (always prefetch)\n", prog);
    printf("  %s -t 100             # Fixed threshold 100%% (almost never prefetch)\n", prog);
    printf("  %s -t 50              # Fixed threshold 50%%\n", prog);
    printf("  %s                    # Adaptive mode (default)\n", prog);
    printf("  %s -m 10 -M 90        # Adaptive with custom range 10-90%%\n", prog);
    printf("  %s -i                 # Inverted: higher threshold when busy\n", prog);
    printf("\nWithout -t, uses adaptive mode based on PCIe throughput.\n");
}

int main(int argc, char **argv) {
    struct prefetch_adaptive_tree_iter_bpf *skel;
    struct bpf_link *link;
    int err;
    int threshold_map_fd;
    unsigned int key = 0;
    int opt;

    while ((opt = getopt(argc, argv, "t:m:M:b:ih")) != -1) {
        switch (opt) {
        case 't':
            config.fixed_thresh = atoi(optarg);
            if (config.fixed_thresh < 0 || config.fixed_thresh > 100) {
                fprintf(stderr, "Error: threshold must be 0-100\n");
                return 1;
            }
            break;
        case 'm':
            config.min_thresh = (unsigned int)atoi(optarg);
            if (config.min_thresh > 100) {
                fprintf(stderr, "Error: min threshold must be 0-100\n");
                return 1;
            }
            break;
        case 'M':
            config.max_thresh = (unsigned int)atoi(optarg);
            if (config.max_thresh > 100) {
                fprintf(stderr, "Error: max threshold must be 0-100\n");
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
    if (config.min_thresh > config.max_thresh) {
        fprintf(stderr, "Error: min threshold (%u) cannot be greater than max (%u)\n",
                config.min_thresh, config.max_thresh);
        return 1;
    }

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    /* Set up libbpf debug output */
    libbpf_set_print(libbpf_print_fn);

    /* Initialize NVML for adaptive mode */
    if (config.fixed_thresh < 0) {
        nvml_device = nvml_init_device();
        if (!nvml_device) {
            fprintf(stderr, "Warning: Failed to initialize NVML, using fixed threshold 50%%\n");
            config.fixed_thresh = 50;
        }
    }

    /* Check and report old struct_ops instances */
    cleanup_old_struct_ops();

    /* Open BPF application */
    skel = prefetch_adaptive_tree_iter_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    /* Load BPF programs */
    err = prefetch_adaptive_tree_iter_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
        goto cleanup;
    }

    /* Get threshold map FD */
    threshold_map_fd = bpf_map__fd(skel->maps.threshold_map);
    if (threshold_map_fd < 0) {
        fprintf(stderr, "Failed to get threshold_map FD\n");
        err = threshold_map_fd;
        goto cleanup;
    }

    /* Set initial threshold */
    unsigned int initial_thresh = (config.fixed_thresh >= 0) ?
                                  (unsigned int)config.fixed_thresh : config.max_thresh;
    err = bpf_map_update_elem(threshold_map_fd, &key, &initial_thresh, BPF_ANY);
    if (err) {
        fprintf(stderr, "Failed to set initial threshold: %d\n", err);
        goto cleanup;
    }

    /* Register struct_ops */
    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_adaptive_tree_iter);
    if (!link) {
        err = -errno;
        fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
        goto cleanup;
    }

    printf("Successfully loaded and attached BPF adaptive_tree_iter policy!\n");
    if (config.fixed_thresh >= 0) {
        printf("Mode: Fixed threshold = %d%%\n", config.fixed_thresh);
    } else {
        printf("Mode: Adaptive (based on PCIe throughput)\n");
        printf("  Range: %u%% - %u%%\n", config.min_thresh, config.max_thresh);
        printf("  Max bandwidth: %llu MB/s\n", config.max_mbps);
        printf("  Invert: %s\n", config.invert ? "yes" : "no");
        printf("Monitoring PCIe traffic and updating threshold every second...\n");
    }
    printf("Monitor tracepipe for BPF debug output.\n");
    printf("\nPress Ctrl-C to exit and detach the policy...\n\n");

    /* Main loop */
    while (!exiting) {
        if (config.fixed_thresh < 0) {
            /* Adaptive mode: update threshold based on PCIe throughput */
            unsigned long long throughput = get_pcie_throughput_mbps();
            unsigned int threshold = calculate_threshold(throughput);

            err = bpf_map_update_elem(threshold_map_fd, &key, &threshold, BPF_ANY);
            if (err) {
                fprintf(stderr, "Failed to update threshold map: %d\n", err);
            } else {
                printf("[%ld] PCIe: %llu MB/s -> Threshold: %u%%\n",
                       time(NULL), throughput, threshold);
            }
        }
        sleep(1);
    }

    printf("\nDetaching struct_ops...\n");
    bpf_link__destroy(link);

cleanup:
    prefetch_adaptive_tree_iter_bpf__destroy(skel);

    if (nvml_device) {
        nvml_cleanup();
    }

    return err < 0 ? -err : 0;
}
