// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2025 */
/*
 * GPU Scheduler struct_ops loader
 *
 * Loads the GPU scheduler struct_ops BPF program and allows
 * setting per-process timeslice policies.
 *
 * Usage:
 *   ./gpu_sched_struct_ops                    # Run with no policies
 *   ./gpu_sched_struct_ops -p python:2000     # Set python timeslice to 2000us
 *   ./gpu_sched_struct_ops -p cuda:500 -p python:2000  # Multiple policies
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include "gpu_sched_set_timeslices.skel.h"

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
    return vfprintf(stderr, format, args);
}

#define TASK_COMM_LEN 16
#define MAX_POLICIES 64

static volatile bool exiting = false;

struct policy_entry {
    char comm[TASK_COMM_LEN];
    __u64 timeslice_us;
};

static void sig_handler(int sig)
{
    exiting = true;
}

static void print_stats(struct gpu_sched_set_timeslices_bpf *skel)
{
    int fd = bpf_map__fd(skel->maps.stats);
    __u64 task_init = 0, bind = 0, task_destroy = 0, timeslice_mod = 0;
    __u32 key;

    key = 0; bpf_map_lookup_elem(fd, &key, &task_init);
    key = 1; bpf_map_lookup_elem(fd, &key, &bind);
    key = 2; bpf_map_lookup_elem(fd, &key, &task_destroy);
    key = 3; bpf_map_lookup_elem(fd, &key, &timeslice_mod);

    printf("\n=== Statistics ===\n");
    printf("task_init:      %llu\n", (unsigned long long)task_init);
    printf("bind:           %llu\n", (unsigned long long)bind);
    printf("task_destroy:   %llu\n", (unsigned long long)task_destroy);
    printf("timeslice_mod:  %llu\n", (unsigned long long)timeslice_mod);
}

static void usage(const char *prog)
{
    fprintf(stderr, "Usage: %s [-h] [-p process:timeslice_us] ...\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  -h                Show this help\n");
    fprintf(stderr, "  -p PROC:TIME      Set timeslice for process (can be repeated)\n");
    fprintf(stderr, "\nExamples:\n");
    fprintf(stderr, "  %s -p python:2000              # python gets 2000us timeslice\n", prog);
    fprintf(stderr, "  %s -p uvmbench:500 -p cuda:1000  # multiple policies\n", prog);
    fprintf(stderr, "\nThe program runs until Ctrl+C. Check dmesg for bpf_printk output.\n");
}

static int parse_policy(const char *arg, struct policy_entry *entry)
{
    char *colon = strchr(arg, ':');
    if (!colon) {
        fprintf(stderr, "Invalid policy format: %s (expected PROC:TIME)\n", arg);
        return -1;
    }

    size_t name_len = colon - arg;
    if (name_len >= TASK_COMM_LEN) {
        fprintf(stderr, "Process name too long: %.*s\n", (int)name_len, arg);
        return -1;
    }

    memset(entry->comm, 0, TASK_COMM_LEN);
    memcpy(entry->comm, arg, name_len);
    entry->timeslice_us = strtoull(colon + 1, NULL, 10);

    if (entry->timeslice_us == 0) {
        fprintf(stderr, "Invalid timeslice: %s\n", colon + 1);
        return -1;
    }

    return 0;
}

int main(int argc, char **argv)
{
    struct gpu_sched_set_timeslices_bpf *skel;
    struct bpf_link *link = NULL;
    struct policy_entry policies[MAX_POLICIES];
    int num_policies = 0;
    int err;
    int opt;

    while ((opt = getopt(argc, argv, "hp:")) != -1) {
        switch (opt) {
        case 'h':
            usage(argv[0]);
            return 0;
        case 'p':
            if (num_policies >= MAX_POLICIES) {
                fprintf(stderr, "Too many policies (max %d)\n", MAX_POLICIES);
                return 1;
            }
            if (parse_policy(optarg, &policies[num_policies]) < 0) {
                return 1;
            }
            num_policies++;
            break;
        default:
            usage(argv[0]);
            return 1;
        }
    }

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    /* Set up libbpf debug output */
    libbpf_set_print(libbpf_print_fn);

    /* Open and load BPF program */
    skel = gpu_sched_set_timeslices_bpf__open_and_load();
    if (!skel) {
        fprintf(stderr, "Failed to open and load BPF skeleton\n");
        return 1;
    }

    /* Add policies to the map */
    if (num_policies > 0) {
        int map_fd = bpf_map__fd(skel->maps.process_timeslice);
        for (int i = 0; i < num_policies; i++) {
            err = bpf_map_update_elem(map_fd, policies[i].comm,
                                      &policies[i].timeslice_us, BPF_ANY);
            if (err) {
                fprintf(stderr, "Failed to add policy for %s: %s\n",
                        policies[i].comm, strerror(errno));
            } else {
                printf("Policy: %s -> %llu us\n",
                       policies[i].comm, (unsigned long long)policies[i].timeslice_us);
            }
        }
    }

    /* Attach struct_ops */
    link = bpf_map__attach_struct_ops(skel->maps.gpu_sched_ops);
    if (!link) {
        fprintf(stderr, "Failed to attach struct_ops: %s\n", strerror(errno));
        fprintf(stderr, "Make sure nvidia.ko with GPU sched struct_ops is loaded\n");
        err = -1;
        goto cleanup;
    }

    printf("GPU Scheduler struct_ops attached. Press Ctrl+C to stop.\n");
    if (num_policies == 0) {
        printf("No policies configured. Use -p PROC:TIME to set timeslices.\n");
    }
    printf("Check: sudo cat /sys/kernel/debug/tracing/trace_pipe\n");
    printf("---\n");

    /* Wait for exit signal */
    while (!exiting) {
        sleep(1);
    }

    print_stats(skel);
    err = 0;

cleanup:
    bpf_link__destroy(link);
    gpu_sched_set_timeslices_bpf__destroy(skel);
    return err < 0 ? 1 : 0;
}
