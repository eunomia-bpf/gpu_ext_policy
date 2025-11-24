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

    /* Get throughput in KB/s, convert to MB/s */
    unsigned long long throughput_kbps = nvml_get_pcie_throughput_kbps(nvml_device);
    return throughput_kbps / 1024;  /* Convert KB/s to MB/s */
}

/* Threshold inversely proportional to PCIe traffic.
 * BPF logic: counter * 100 > subregion_pages * threshold
 *   - Low threshold  -> easier to pass -> more prefetch
 *   - High threshold -> harder to pass -> less prefetch
 *
 * Goal: More traffic -> more prefetch (keep GPU fed with data)
 *
 * 20 GB/s = 20480 MB/s max.
 * - High traffic (20 GB/s)  -> threshold = 0%   -> prefetch more
 * - Low traffic (0 MB/s)    -> threshold = 100% -> prefetch less
 */
static unsigned int calculate_threshold(unsigned long long throughput_mbps) {
    const unsigned long long max_mbps = 20480ULL; /* 20 GB/s */
    const unsigned int max_thresh = 100;          /* low traffic: less prefetch */
    const unsigned int min_thresh = 0;            /* high traffic: more prefetch */

    if (throughput_mbps >= max_mbps)
        return min_thresh;

    double ratio = (double)throughput_mbps / (double)max_mbps; /* 0..1 */
    double inv = 1.0 - ratio;
    unsigned int thresh = (unsigned int)(min_thresh + (max_thresh - min_thresh) * inv + 0.5);
    if (thresh < min_thresh) thresh = min_thresh;
    if (thresh > max_thresh) thresh = max_thresh;
    return thresh;
}

int main(int argc, char **argv) {
    struct prefetch_adaptive_tree_iter_bpf *skel;
    struct bpf_link *link;
    int err;
    int threshold_map_fd;
    unsigned int key = 0;

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    /* Initialize NVML */
    nvml_device = nvml_init_device();
    if (!nvml_device) {
        fprintf(stderr, "Warning: Failed to initialize NVML, using fallback mode\n");
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

    /* Register struct_ops */
    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_adaptive_tree_iter);
    if (!link) {
        err = -errno;
        fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
        goto cleanup;
    }

    printf("Successfully loaded and attached BPF adaptive_tree_iter policy!\n");
    printf("Monitoring PCIe traffic and updating threshold every second...\n");
    printf("Monitor dmesg for BPF debug output.\n");
    printf("\nPress Ctrl-C to exit and detach the policy...\n\n");

    /* Main loop: Monitor PCIe traffic and update threshold */
    while (!exiting) {
        unsigned long long throughput = get_pcie_throughput_mbps();
        unsigned int threshold = calculate_threshold(throughput);

        /* Update threshold in BPF map */
        err = bpf_map_update_elem(threshold_map_fd, &key, &threshold, BPF_ANY);
        if (err) {
            fprintf(stderr, "Failed to update threshold map: %d\n", err);
        } else {
            /* Print current stats */
            printf("[%ld] PCIe Throughput: %llu MB/s -> Threshold: %u%%\n",
                   time(NULL), throughput, threshold);
        }

        sleep(1);
    }

    printf("\nDetaching struct_ops...\n");
    bpf_link__destroy(link);

cleanup:
    prefetch_adaptive_tree_iter_bpf__destroy(skel);

    /* Cleanup NVML */
    if (nvml_device) {
        nvml_cleanup();
    }

    return err < 0 ? -err : 0;
}
