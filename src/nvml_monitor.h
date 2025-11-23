#ifndef _NVML_MONITOR_H
#define _NVML_MONITOR_H

#include <nvml.h>
#include <stdio.h>

/* Initialize NVML and get first device handle */
static inline nvmlDevice_t nvml_init_device(void) {
    nvmlDevice_t device;
    nvmlReturn_t result;

    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "Failed to initialize NVML: %s\n", nvmlErrorString(result));
        return NULL;
    }

    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "Failed to get device handle: %s\n", nvmlErrorString(result));
        nvmlShutdown();
        return NULL;
    }

    return device;
}

/* Get PCIe throughput in KB/s (both RX and TX combined) */
static inline unsigned long long nvml_get_pcie_throughput_kbps(nvmlDevice_t device) {
    unsigned int rx_throughput = 0, tx_throughput = 0;
    nvmlReturn_t result;

    /* Get RX (receive) throughput */
    result = nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_RX_BYTES, &rx_throughput);
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "Failed to get RX throughput: %s\n", nvmlErrorString(result));
        return 0;
    }

    /* Get TX (transmit) throughput */
    result = nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_TX_BYTES, &tx_throughput);
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "Failed to get TX throughput: %s\n", nvmlErrorString(result));
        return 0;
    }

    /* Return total throughput (RX + TX) in KB/s */
    return (unsigned long long)(rx_throughput + tx_throughput);
}

/* Cleanup NVML */
static inline void nvml_cleanup(void) {
    nvmlShutdown();
}

#endif /* _NVML_MONITOR_H */
