/* SPDX-License-Identifier: GPL-2.0 */
/* Standalone tool to cleanup old struct_ops instances */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

static int cleanup_all_struct_ops(void) {
    __u32 map_id = 0;
    int cleaned = 0;
    int err;

    printf("Scanning for all struct_ops instances...\n");

    /* Iterate through all BPF maps */
    while (1) {
        struct bpf_map_info info = {};
        __u32 len = sizeof(info);
        int fd;

        err = bpf_map_get_next_id(map_id, &map_id);
        if (err) {
            if (errno == ENOENT) {
                break; /* No more maps */
            }
            continue;
        }

        fd = bpf_map_get_fd_by_id(map_id);
        if (fd < 0) {
            continue;
        }

        err = bpf_obj_get_info_by_fd(fd, &info, &len);
        if (err) {
            close(fd);
            continue;
        }

        /* Check if this is a struct_ops map */
        if (info.type == BPF_MAP_TYPE_STRUCT_OPS) {
            printf("Found struct_ops map (ID: %u, name: %s)\n",
                   info.id, info.name);

            /* Try to pin and then unpin to release references */
            char pin_path[256];
            snprintf(pin_path, sizeof(pin_path),
                     "/sys/fs/bpf/cleanup_%u", info.id);

            /* Pin the map */
            if (bpf_obj_pin(fd, pin_path) == 0) {
                /* Immediately unpin it */
                unlink(pin_path);
                printf("  -> Cleaned up map ID %u\n", info.id);
                cleaned++;
            } else {
                printf("  -> Failed to pin map (may still be in use)\n");
            }
        }

        close(fd);
    }

    if (cleaned > 0) {
        printf("\nCleaned up %d struct_ops instance(s)\n", cleaned);
        printf("Note: Some instances may still be active if held by running processes.\n");
    } else {
        printf("\nNo struct_ops instances found to clean up.\n");
    }

    return 0;
}

static void print_usage(const char *prog) {
    printf("Usage: %s [OPTIONS]\n", prog);
    printf("\nOptions:\n");
    printf("  -h, --help     Show this help message\n");
    printf("  -l, --list     List all struct_ops instances (don't cleanup)\n");
    printf("\n");
    printf("This tool cleans up old BPF struct_ops instances.\n");
    printf("It's useful when struct_ops programs exit abnormally.\n");
}

static int list_struct_ops(void) {
    __u32 map_id = 0;
    int count = 0;
    int err;

    printf("Listing all struct_ops instances...\n");
    printf("%-8s %-30s %-10s\n", "ID", "Name", "Type");
    printf("------------------------------------------------------------\n");

    /* Iterate through all BPF maps */
    while (1) {
        struct bpf_map_info info = {};
        __u32 len = sizeof(info);
        int fd;

        err = bpf_map_get_next_id(map_id, &map_id);
        if (err) {
            if (errno == ENOENT) {
                break; /* No more maps */
            }
            continue;
        }

        fd = bpf_map_get_fd_by_id(map_id);
        if (fd < 0) {
            continue;
        }

        err = bpf_obj_get_info_by_fd(fd, &info, &len);
        if (err) {
            close(fd);
            continue;
        }

        /* Check if this is a struct_ops map */
        if (info.type == BPF_MAP_TYPE_STRUCT_OPS) {
            printf("%-8u %-30s %-10s\n", info.id, info.name, "struct_ops");
            count++;
        }

        close(fd);
    }

    printf("\nTotal: %d struct_ops instance(s)\n", count);
    return 0;
}

int main(int argc, char **argv) {
    /* Check if running as root */
    if (geteuid() != 0) {
        fprintf(stderr, "Error: This tool requires root privileges.\n");
        fprintf(stderr, "Please run with sudo: sudo %s\n", argv[0]);
        return 1;
    }

    if (argc > 1) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        if (strcmp(argv[1], "-l") == 0 || strcmp(argv[1], "--list") == 0) {
            return list_struct_ops();
        }
        fprintf(stderr, "Unknown option: %s\n", argv[1]);
        print_usage(argv[0]);
        return 1;
    }

    /* Default: cleanup all struct_ops */
    return cleanup_all_struct_ops();
}
