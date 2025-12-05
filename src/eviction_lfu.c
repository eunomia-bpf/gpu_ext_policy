#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "eviction_lfu.skel.h"
#include "cleanup_struct_ops.h"

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
    return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;

void handle_signal(int sig) {
    exiting = true;
}

int main(int argc, char **argv) {
    struct eviction_lfu_bpf *skel;
    struct bpf_link *link;
    int err;

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    /* Set up libbpf debug output */
    libbpf_set_print(libbpf_print_fn);

    /* Check and report old struct_ops instances */
    cleanup_old_struct_ops();

    /* Open BPF application */
    skel = eviction_lfu_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    /* Load BPF programs */
    err = eviction_lfu_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
        goto cleanup;
    }

    /* Register struct_ops */
    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_lfu_clean);
    if (!link) {
        err = -errno;
        fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
        goto cleanup;
    }

    printf("Successfully loaded and attached BPF LFU eviction policy!\n");
    printf("The LFU (Least Frequently Used) eviction policy is now active.\n");
    printf("Chunks will be evicted based on access frequency.\n");
    printf("\nPolicy behavior:\n");
    printf("  - Low frequency chunks (< 3 accesses): near HEAD (evicted first)\n");
    printf("  - High frequency chunks (>= 10 accesses): moved to TAIL (protected)\n");
    printf("\nMonitor dmesg for BPF debug output:\n");
    printf("  sudo dmesg -w | grep 'BPF LFU'\n");
    printf("\nPress Ctrl-C to exit and detach the policy...\n");

    /* Wait for signal */
    while (!exiting) {
        sleep(1);
    }

    printf("\nDetaching struct_ops...\n");
    bpf_link__destroy(link);

cleanup:
    eviction_lfu_bpf__destroy(skel);
    return err < 0 ? -err : 0;
}
