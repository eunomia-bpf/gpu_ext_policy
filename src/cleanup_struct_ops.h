/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _CLEANUP_STRUCT_OPS_H
#define _CLEANUP_STRUCT_OPS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

/* Cleanup old struct_ops instances
 * Returns: 0 on success (no old instances found)
 *          -EEXIST if old instances were found
 */
static inline int cleanup_old_struct_ops(void) {
    __u32 map_id = 0;
    int cleaned = 0;
    int err;

    printf("Checking for old struct_ops instances...\n");

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

        /* Check if this is our struct_ops map */
        if (info.type == BPF_MAP_TYPE_STRUCT_OPS &&
            (strcmp(info.name, "uvm_ops_always_max") == 0 ||
             strcmp(info.name, "uvm_ops") == 0)) {
            printf("Found old struct_ops map (ID: %u, name: %s)\n",
                   info.id, info.name);
            printf("Attempting to clean up...\n");

            /* Try to pin and then unpin to release references */
            char pin_path[256];
            snprintf(pin_path, sizeof(pin_path),
                     "/sys/fs/bpf/old_testmod_%u", info.id);

            /* Pin the map */
            if (bpf_obj_pin(fd, pin_path) == 0) {
                /* Immediately unpin it */
                unlink(pin_path);
                printf("Cleaned up pinned reference\n");
            }

            cleaned++;
        }

        close(fd);
    }

    if (cleaned > 0) {
        printf("Found %d old struct_ops instance(s)\n", cleaned);
        printf("Note: Old instances may still be active if held by running processes.\n");
        printf("Please kill any running struct_ops processes first.\n");
        return -EEXIST;
    }

    printf("No old struct_ops instances found.\n");
    return 0;
}

#endif /* _CLEANUP_STRUCT_OPS_H */
