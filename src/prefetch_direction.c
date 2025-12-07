#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "prefetch_direction.skel.h"
#include "cleanup_struct_ops.h"

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
    return vfprintf(stderr, format, args);
}

#define PREFETCH_FORWARD  0
#define PREFETCH_BACKWARD 1

static volatile bool exiting = false;

void handle_signal(int sig) {
    exiting = true;
}

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s [OPTIONS]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -d <direction>   Prefetch direction: 'forward' (default) or 'backward'\n");
    fprintf(stderr, "  -n <num_pages>   Number of pages to prefetch (default: 0 = all available)\n");
    fprintf(stderr, "  -h               Show this help message\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Direction modes:\n");
    fprintf(stderr, "  forward:  Prefetch pages AFTER the faulting page (higher addresses)\n");
    fprintf(stderr, "            For sequential access patterns (low -> high)\n");
    fprintf(stderr, "  backward: Prefetch pages BEFORE the faulting page (lower addresses)\n");
    fprintf(stderr, "            For reverse access patterns (high -> low)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Number of pages:\n");
    fprintf(stderr, "  0:   Prefetch all available pages in the direction (up to max_prefetch_region)\n");
    fprintf(stderr, "  N>0: Prefetch exactly N pages (or fewer if not available)\n");
}

int main(int argc, char **argv) {
    struct prefetch_direction_bpf *skel;
    struct bpf_link *link;
    int err;
    int opt;
    unsigned int direction = PREFETCH_FORWARD;  /* Default: forward */
    unsigned int num_pages = 0;                 /* Default: 0 = all available */

    /* Parse command line arguments */
    while ((opt = getopt(argc, argv, "d:n:h")) != -1) {
        switch (opt) {
        case 'd':
            if (strcmp(optarg, "forward") == 0 || strcmp(optarg, "f") == 0) {
                direction = PREFETCH_FORWARD;
            } else if (strcmp(optarg, "backward") == 0 || strcmp(optarg, "b") == 0) {
                direction = PREFETCH_BACKWARD;
            } else {
                fprintf(stderr, "Invalid direction: %s\n", optarg);
                print_usage(argv[0]);
                return 1;
            }
            break;
        case 'n':
            num_pages = (unsigned int)atoi(optarg);
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

    /* Set up libbpf debug output */
    libbpf_set_print(libbpf_print_fn);

    /* Check and report old struct_ops instances */
    cleanup_old_struct_ops();

    /* Open BPF application */
    skel = prefetch_direction_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    /* Load BPF programs */
    err = prefetch_direction_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
        goto cleanup;
    }

    /* Set initial direction in BPF map */
    int dir_map_fd = bpf_map__fd(skel->maps.prefetch_direction_map);
    if (dir_map_fd < 0) {
        fprintf(stderr, "Failed to get direction map fd: %d\n", dir_map_fd);
        err = dir_map_fd;
        goto cleanup;
    }

    unsigned int key = 0;
    err = bpf_map_update_elem(dir_map_fd, &key, &direction, BPF_ANY);
    if (err) {
        fprintf(stderr, "Failed to set direction in map: %d\n", err);
        goto cleanup;
    }

    /* Set num_pages in BPF map */
    int num_map_fd = bpf_map__fd(skel->maps.prefetch_num_pages_map);
    if (num_map_fd < 0) {
        fprintf(stderr, "Failed to get num_pages map fd: %d\n", num_map_fd);
        err = num_map_fd;
        goto cleanup;
    }

    err = bpf_map_update_elem(num_map_fd, &key, &num_pages, BPF_ANY);
    if (err) {
        fprintf(stderr, "Failed to set num_pages in map: %d\n", err);
        goto cleanup;
    }

    /* Register struct_ops */
    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_direction);
    if (!link) {
        err = -errno;
        fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
        goto cleanup;
    }

    printf("Successfully loaded and attached BPF directional prefetch policy!\n");
    printf("Direction: %s\n", direction == PREFETCH_FORWARD ? "FORWARD (prefetch higher addresses)" : "BACKWARD (prefetch lower addresses)");
    printf("Num pages: %u %s\n", num_pages, num_pages == 0 ? "(all available)" : "");
    printf("Monitor tracepipe for BPF debug output.\n");
    printf("\nPress Ctrl-C to exit and detach the policy...\n");

    /* Wait for signal */
    while (!exiting) {
        sleep(1);
    }

    printf("\nDetaching struct_ops...\n");
    bpf_link__destroy(link);

cleanup:
    prefetch_direction_bpf__destroy(skel);
    return err < 0 ? -err : 0;
}
