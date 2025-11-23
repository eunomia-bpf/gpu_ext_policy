#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <time.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "prefetch_adaptive_simple.skel.h"
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

/* Calculate threshold based on PCIe throughput
 * Logic:
 *  - Low traffic (<100 MB/s): Aggressive prefetch (30%)
 *  - Medium traffic (100-300 MB/s): Default prefetch (51%)
 *  - High traffic (>300 MB/s): Conservative prefetch (75%)
 */
static unsigned int calculate_threshold(unsigned long long throughput_mbps) {
    if (throughput_mbps > 300)
        return 75;  // High traffic -> conservative
    else if (throughput_mbps > 100)
        return 51;  // Medium traffic -> default
    else
        return 30;  // Low traffic -> aggressive
}

int main(int argc, char **argv) {
    struct prefetch_adaptive_simple_bpf *skel;
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
    skel = prefetch_adaptive_simple_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    /* Load BPF programs */
    err = prefetch_adaptive_simple_bpf__load(skel);
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
    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_adaptive_simple);
    if (!link) {
        err = -errno;
        fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
        goto cleanup;
    }

    printf("Successfully loaded and attached BPF adaptive_simple policy!\n");
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
    prefetch_adaptive_simple_bpf__destroy(skel);

    /* Cleanup NVML */
    if (nvml_device) {
        nvml_cleanup();
    }

    return err < 0 ? -err : 0;
}
