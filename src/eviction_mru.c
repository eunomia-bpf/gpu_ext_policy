#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "eviction_mru.skel.h"
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
    struct eviction_mru_bpf *skel;
    struct bpf_link *link;
    int err;

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    /* Set up libbpf debug output */
    libbpf_set_print(libbpf_print_fn);

    /* Check and report old struct_ops instances */
    cleanup_old_struct_ops();

    /* Open BPF application */
    skel = eviction_mru_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    /* Load BPF programs */
    err = eviction_mru_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
        goto cleanup;
    }

    /* Register struct_ops */
    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_mru);
    if (!link) {
        err = -errno;
        fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
        goto cleanup;
    }

    printf("Successfully loaded and attached BPF MRU eviction policy!\n");
    printf("MRU: Most recently used chunks are evicted first.\n");
    printf("Best for: sequential/streaming workloads (e.g., MoE prefill)\n");
    printf("\nPress Ctrl-C to exit and detach the policy...\n");

    while (!exiting) {
        sleep(1);
    }

    printf("\nDetaching struct_ops...\n");
    bpf_link__destroy(link);

cleanup:
    eviction_mru_bpf__destroy(skel);
    return err < 0 ? -err : 0;
}
