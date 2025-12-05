#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <dirent.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "struct_ops.skel.h"

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
    return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;

void handle_signal(int sig) {
    exiting = true;
}

/* Cleanup old struct_ops instances if any */
static int cleanup_old_struct_ops(void) {
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
            strcmp(info.name, "uvm_ops") == 0) {
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

static int trigger_struct_ops(const char *message) {
    int fd, ret;
    
    fd = open("/proc/bpf_testmod_trigger", O_WRONLY);
    if (fd < 0) {
        perror("open /proc/bpf_testmod_trigger");
        return -1;
    }
    
    ret = write(fd, message, strlen(message));
    if (ret < 0) {
        perror("write");
        close(fd);
        return -1;
    }
    
    close(fd);
    return 0;
}

int main(int argc, char **argv) {
    struct struct_ops_bpf *skel;
    struct bpf_link *link;
    int err;

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    /* Set up libbpf debug output */
    libbpf_set_print(libbpf_print_fn);

    /* Check and report old struct_ops instances */
    cleanup_old_struct_ops();

    /* Open BPF application */
    skel = struct_ops_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    /* Load BPF programs */
    err = struct_ops_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
        goto cleanup;
    }

    /* Register struct_ops */
    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops);
    if (!link) {
        fprintf(stderr, "Failed to attach struct_ops\n");
        err = -1;
        goto cleanup;
    }

    printf("Successfully loaded and attached BPF struct_ops!\n");
    printf("Triggering struct_ops callbacks...\n");
    
    /* Trigger the struct_ops by writing to proc file */
    if (trigger_struct_ops("Hello from userspace!") < 0) {
        printf("Failed to trigger struct_ops - is the kernel module loaded?\n");
        printf("Load it with: sudo insmod module/hello.ko\n");
    } else {
        printf("Triggered struct_ops successfully! Check dmesg for output.\n");
    }
    
    printf("\nPress Ctrl-C to exit...\n");

    /* Main loop - trigger periodically */
    while (!exiting) {
        sleep(2);
        if (!exiting && trigger_struct_ops("Periodic trigger") == 0) {
            printf("Triggered struct_ops again...\n");
        }
    }

    printf("\nDetaching struct_ops...\n");
    bpf_link__destroy(link);

cleanup:
    struct_ops_bpf__destroy(skel);
    return err < 0 ? -err : 0;
}
