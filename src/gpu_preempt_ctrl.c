// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2025 */
/*
 * gpu_preempt_ctrl.c - Userspace GPU preempt control tool
 *
 * This tool:
 * 1. Loads the eBPF program to trace GPU TSG creation events
 * 2. Captures hClient/hTsg handles from tracepoints
 * 3. Provides interactive commands to preempt/control TSGs
 *
 * Similar to GPreempt patch but uses tracepoints instead of kernel modifications.
 *
 * Usage:
 *   sudo ./gpu_preempt_ctrl [-v] [-p PID]
 *
 * Commands (interactive mode):
 *   list                    - List all tracked TSGs
 *   preempt <hClient> <hTsg> [timeout_ms]  - Preempt a TSG
 *   preempt-pid <pid> [timeout_ms]         - Preempt all TSGs of a process
 *   timeslice <hClient> <hTsg> <us>        - Set timeslice
 *   interleave <hClient> <hTsg> <level>    - Set interleave level (0=LOW, 1=MED, 2=HIGH)
 *   quit                    - Exit
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/ioctl.h>
#include <pthread.h>
#include <stdint.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include "gpu_preempt_ctrl.skel.h"
#include "gpu_preempt_ctrl_event.h"

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
    return vfprintf(stderr, format, args);
}

/* NVIDIA ioctl constants */
#define NV_IOCTL_MAGIC      'F'
#define NV_IOCTL_BASE       200
#define NV_ESC_IOCTL_XFER_CMD   (NV_IOCTL_BASE + 11)

static volatile bool exiting = false;
static int verbose = 0;
static int g_nvidia_fd = -1;

/* nv_ioctl_xfer_t for indirect ioctl */
typedef struct {
    uint32_t cmd;
    uint32_t size;
    void    *ptr __attribute__((aligned(8)));
} nv_ioctl_xfer_t;

/* NVOS54_PARAMETERS for RM control ioctl */
typedef struct {
    uint32_t hClient;
    uint32_t hObject;
    uint32_t cmd;
    uint32_t flags;
    void    *params __attribute__((aligned(8)));
    uint32_t paramsSize;
    uint32_t status;
} NVOS54_PARAMETERS;

/* NVA06C_CTRL_PREEMPT_PARAMS */
typedef struct {
    uint8_t  bWait;
    uint8_t  bManualTimeout;
    uint32_t timeoutUs;
} NVA06C_CTRL_PREEMPT_PARAMS;

/* NVA06C_CTRL_TIMESLICE_PARAMS */
typedef struct {
    uint64_t timesliceUs;
} NVA06C_CTRL_TIMESLICE_PARAMS;

/* NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS */
typedef struct {
    uint32_t tsgInterleaveLevel;
} NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS;

/* TSG tracking - simple in-memory list */
#define MAX_TRACKED_TSGS 256
static struct tsg_info tracked_tsgs[MAX_TRACKED_TSGS];
static int num_tracked_tsgs = 0;
static pthread_mutex_t tsg_lock = PTHREAD_MUTEX_INITIALIZER;

static void sig_handler(int sig)
{
    exiting = true;
}

static const char *event_type_str(int type)
{
    switch (type) {
    case EVENT_TSG_CREATE:   return "TSG_CREATE";
    case EVENT_TSG_SCHEDULE: return "TSG_SCHEDULE";
    case EVENT_TSG_DESTROY:  return "TSG_DESTROY";
    default:                 return "UNKNOWN";
    }
}

static const char *engine_type_str(int type)
{
    switch (type) {
    case ENGINE_TYPE_GRAPHICS: return "GRAPHICS";
    case ENGINE_TYPE_COPY:     return "COPY";
    case ENGINE_TYPE_NVDEC:    return "NVDEC";
    case ENGINE_TYPE_NVENC:    return "NVENC";
    case ENGINE_TYPE_NVJPEG:   return "NVJPEG";
    default:                   return "UNKNOWN";
    }
}

static const char *interleave_str(int level)
{
    switch (level) {
    case INTERLEAVE_LEVEL_LOW:    return "LOW";
    case INTERLEAVE_LEVEL_MEDIUM: return "MEDIUM";
    case INTERLEAVE_LEVEL_HIGH:   return "HIGH";
    default:                      return "UNKNOWN";
    }
}

/* Open NVIDIA device for ioctl */
static int open_nvidia_device(void)
{
    int fd = open("/dev/nvidiactl", O_RDWR);
    if (fd < 0) {
        fd = open("/dev/nvidia0", O_RDWR);
    }
    return fd;
}

/* Send RM control command via ioctl */
static int rm_control(int fd, uint32_t hClient, uint32_t hObject,
                      uint32_t cmd, void *params, uint32_t paramsSize)
{
    NVOS54_PARAMETERS ctrl;
    nv_ioctl_xfer_t xfer;
    int ret;

    memset(&ctrl, 0, sizeof(ctrl));
    ctrl.hClient = hClient;
    ctrl.hObject = hObject;
    ctrl.cmd = cmd;
    ctrl.flags = 0;
    ctrl.params = params;
    ctrl.paramsSize = paramsSize;
    ctrl.status = 0;

    /* Use xfer command to pass large structure */
    memset(&xfer, 0, sizeof(xfer));
    xfer.cmd = NV_ESC_RM_CONTROL;
    xfer.size = sizeof(ctrl);
    xfer.ptr = &ctrl;

    /* Use _IOWR to encode ioctl command with proper size */
    ret = ioctl(fd, _IOWR(NV_IOCTL_MAGIC, NV_ESC_IOCTL_XFER_CMD, nv_ioctl_xfer_t), &xfer);

    if (ret < 0) {
        return -errno;
    }
    return ctrl.status;
}

/* Preempt a TSG */
static int do_preempt(uint32_t hClient, uint32_t hTsg, uint32_t timeout_ms)
{
    NVA06C_CTRL_PREEMPT_PARAMS params;
    uint64_t start, end;
    struct timespec ts;
    int ret;

    if (g_nvidia_fd < 0) {
        fprintf(stderr, "NVIDIA device not open\n");
        return -1;
    }

    memset(&params, 0, sizeof(params));
    params.bWait = 1;
    params.bManualTimeout = 1;
    params.timeoutUs = timeout_ms * 1000;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    start = ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;

    ret = rm_control(g_nvidia_fd, hClient, hTsg, NVA06C_CTRL_CMD_PREEMPT,
                     &params, sizeof(params));

    clock_gettime(CLOCK_MONOTONIC, &ts);
    end = ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;

    printf("  PREEMPT hClient=0x%x hTsg=0x%x result=%d duration=%lu us\n",
           hClient, hTsg, ret, (unsigned long)(end - start));

    return ret;
}

/* Set TSG timeslice */
static int do_set_timeslice(uint32_t hClient, uint32_t hTsg, uint64_t timeslice_us)
{
    NVA06C_CTRL_TIMESLICE_PARAMS params;
    int ret;

    if (g_nvidia_fd < 0) {
        fprintf(stderr, "NVIDIA device not open\n");
        return -1;
    }

    memset(&params, 0, sizeof(params));
    params.timesliceUs = timeslice_us;

    ret = rm_control(g_nvidia_fd, hClient, hTsg, NVA06C_CTRL_CMD_SET_TIMESLICE,
                     &params, sizeof(params));

    printf("  SET_TIMESLICE hClient=0x%x hTsg=0x%x timeslice=%lu us result=%d\n",
           hClient, hTsg, (unsigned long)timeslice_us, ret);

    return ret;
}

/* Set TSG interleave level (priority) */
static int do_set_interleave(uint32_t hClient, uint32_t hTsg, uint32_t level)
{
    NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS params;
    int ret;

    if (g_nvidia_fd < 0) {
        fprintf(stderr, "NVIDIA device not open\n");
        return -1;
    }

    memset(&params, 0, sizeof(params));
    params.tsgInterleaveLevel = level;

    ret = rm_control(g_nvidia_fd, hClient, hTsg, NVA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL,
                     &params, sizeof(params));

    printf("  SET_INTERLEAVE hClient=0x%x hTsg=0x%x level=%s result=%d\n",
           hClient, hTsg, interleave_str(level), ret);

    return ret;
}

/* Add TSG to tracking list */
static void track_tsg(struct tsg_info *info)
{
    pthread_mutex_lock(&tsg_lock);

    /* Check if already tracked */
    for (int i = 0; i < num_tracked_tsgs; i++) {
        if (tracked_tsgs[i].hTsg == info->hTsg &&
            tracked_tsgs[i].hClient == info->hClient) {
            /* Update existing */
            tracked_tsgs[i] = *info;
            pthread_mutex_unlock(&tsg_lock);
            return;
        }
    }

    /* Add new */
    if (num_tracked_tsgs < MAX_TRACKED_TSGS) {
        tracked_tsgs[num_tracked_tsgs++] = *info;
    }

    pthread_mutex_unlock(&tsg_lock);
}

/* Remove TSG from tracking list */
static void untrack_tsg(uint32_t hClient, uint32_t hTsg)
{
    pthread_mutex_lock(&tsg_lock);

    for (int i = 0; i < num_tracked_tsgs; i++) {
        if (tracked_tsgs[i].hTsg == hTsg &&
            tracked_tsgs[i].hClient == hClient) {
            /* Remove by swapping with last */
            tracked_tsgs[i] = tracked_tsgs[num_tracked_tsgs - 1];
            num_tracked_tsgs--;
            break;
        }
    }

    pthread_mutex_unlock(&tsg_lock);
}

/* List all tracked TSGs */
static void list_tsgs(void)
{
    pthread_mutex_lock(&tsg_lock);

    printf("\n=== Tracked TSGs (%d) ===\n", num_tracked_tsgs);
    printf("%-10s %-10s %-10s %-8s %-8s %-10s %-8s %s\n",
           "hClient", "hTsg", "tsg_id", "engine", "runlist", "timeslice", "level", "process");
    printf("--------------------------------------------------------------------------------\n");

    for (int i = 0; i < num_tracked_tsgs; i++) {
        struct tsg_info *t = &tracked_tsgs[i];
        printf("0x%08x 0x%08x %-10llu %-8s %-8u %-10llu %-8s %s (pid=%u)\n",
               t->hClient, t->hTsg, (unsigned long long)t->tsg_id,
               engine_type_str(t->engine_type), t->runlist_id,
               (unsigned long long)t->timeslice_us,
               interleave_str(t->interleave_level),
               t->comm, t->pid);
    }
    printf("\n");

    pthread_mutex_unlock(&tsg_lock);
}

/* Find and preempt all TSGs of a process */
static void preempt_pid(uint32_t pid, uint32_t timeout_ms)
{
    int count = 0;

    pthread_mutex_lock(&tsg_lock);

    printf("Preempting all TSGs for PID %u...\n", pid);

    for (int i = 0; i < num_tracked_tsgs; i++) {
        struct tsg_info *t = &tracked_tsgs[i];
        if (t->pid == pid) {
            pthread_mutex_unlock(&tsg_lock);
            do_preempt(t->hClient, t->hTsg, timeout_ms);
            pthread_mutex_lock(&tsg_lock);
            count++;
        }
    }

    pthread_mutex_unlock(&tsg_lock);

    printf("Preempted %d TSGs\n", count);
}

/* Handle event from BPF */
static int handle_event(void *ctx, void *data, size_t data_sz)
{
    struct gpu_ctrl_event *e = data;
    struct timespec ts;
    struct tm *tm;
    char ts_buf[32];

    clock_gettime(CLOCK_REALTIME, &ts);
    tm = localtime(&ts.tv_sec);
    strftime(ts_buf, sizeof(ts_buf), "%H:%M:%S", tm);

    switch (e->event_type) {
    case EVENT_TSG_CREATE:
        if (verbose) {
            printf("%s.%06llu [CPU%02u] %-12s PID=%-6u %-16s "
                   "hClient=0x%x hTsg=0x%x tsg=%llu engine=%s timeslice=%llu runlist=%u\n",
                   ts_buf, (unsigned long long)(e->timestamp_ns % 1000000000 / 1000),
                   e->cpu, event_type_str(e->event_type),
                   e->tgid, e->comm,
                   e->hClient, e->hTsg,
                   (unsigned long long)e->tsg_id,
                   engine_type_str(e->engine_type),
                   (unsigned long long)e->timeslice_us,
                   e->runlist_id);
        }

        /* Track this TSG */
        {
            struct tsg_info info = {
                .hClient = e->hClient,
                .hTsg = e->hTsg,
                .tsg_id = e->tsg_id,
                .engine_type = e->engine_type,
                .runlist_id = e->runlist_id,
                .timeslice_us = e->timeslice_us,
                .interleave_level = e->interleave_level,
                .pid = e->tgid,
                .create_time_ns = e->timestamp_ns,
            };
            memcpy(info.comm, e->comm, sizeof(info.comm));
            track_tsg(&info);
        }
        break;

    case EVENT_TSG_SCHEDULE:
        if (verbose) {
            printf("%s.%06llu [CPU%02u] %-12s PID=%-6u %-16s "
                   "hClient=0x%x hTsg=0x%x tsg=%llu channels=%u timeslice=%llu\n",
                   ts_buf, (unsigned long long)(e->timestamp_ns % 1000000000 / 1000),
                   e->cpu, event_type_str(e->event_type),
                   e->tgid, e->comm,
                   e->hClient, e->hTsg,
                   (unsigned long long)e->tsg_id,
                   e->channel_count,
                   (unsigned long long)e->timeslice_us);
        }
        break;

    case EVENT_TSG_DESTROY:
        if (verbose) {
            printf("%s.%06llu [CPU%02u] %-12s PID=%-6u %-16s "
                   "hClient=0x%x hTsg=0x%x tsg=%llu\n",
                   ts_buf, (unsigned long long)(e->timestamp_ns % 1000000000 / 1000),
                   e->cpu, event_type_str(e->event_type),
                   e->tgid, e->comm,
                   e->hClient, e->hTsg,
                   (unsigned long long)e->tsg_id);
        }

        /* Untrack this TSG */
        untrack_tsg(e->hClient, e->hTsg);
        break;
    }

    return 0;
}

static void print_stats(struct gpu_preempt_ctrl_bpf *skel)
{
    int fd = bpf_map__fd(skel->maps.stats);
    __u64 tsg_create = 0, tsg_schedule = 0, tsg_destroy = 0, dropped = 0;
    __u32 key;

    key = STAT_TSG_CREATE;  bpf_map_lookup_elem(fd, &key, &tsg_create);
    key = STAT_TSG_SCHEDULE; bpf_map_lookup_elem(fd, &key, &tsg_schedule);
    key = STAT_TSG_DESTROY;  bpf_map_lookup_elem(fd, &key, &tsg_destroy);
    key = STAT_DROPPED;      bpf_map_lookup_elem(fd, &key, &dropped);

    printf("\n=== Statistics ===\n");
    printf("tsg_create:   %llu\n", (unsigned long long)tsg_create);
    printf("tsg_schedule: %llu\n", (unsigned long long)tsg_schedule);
    printf("tsg_destroy:  %llu\n", (unsigned long long)tsg_destroy);
    printf("dropped:      %llu\n", (unsigned long long)dropped);
}

/* Interactive command handler */
static void handle_command(char *line)
{
    char cmd[64];
    uint32_t hClient, hTsg, val;
    uint64_t val64;

    if (sscanf(line, "%63s", cmd) != 1)
        return;

    if (strcmp(cmd, "list") == 0) {
        list_tsgs();
    }
    else if (strcmp(cmd, "preempt") == 0) {
        val = 100;  /* Default timeout 100ms */
        if (sscanf(line, "%*s %x %x %u", &hClient, &hTsg, &val) >= 2) {
            do_preempt(hClient, hTsg, val);
        } else {
            printf("Usage: preempt <hClient> <hTsg> [timeout_ms]\n");
        }
    }
    else if (strcmp(cmd, "preempt-pid") == 0) {
        val = 100;
        if (sscanf(line, "%*s %u %u", &hClient, &val) >= 1) {
            preempt_pid(hClient, val);
        } else {
            printf("Usage: preempt-pid <pid> [timeout_ms]\n");
        }
    }
    else if (strcmp(cmd, "timeslice") == 0) {
        if (sscanf(line, "%*s %x %x %lu", &hClient, &hTsg, &val64) == 3) {
            do_set_timeslice(hClient, hTsg, val64);
        } else {
            printf("Usage: timeslice <hClient> <hTsg> <microseconds>\n");
        }
    }
    else if (strcmp(cmd, "interleave") == 0) {
        if (sscanf(line, "%*s %x %x %u", &hClient, &hTsg, &val) == 3) {
            do_set_interleave(hClient, hTsg, val);
        } else {
            printf("Usage: interleave <hClient> <hTsg> <level> (0=LOW, 1=MED, 2=HIGH)\n");
        }
    }
    else if (strcmp(cmd, "help") == 0) {
        printf("Commands:\n");
        printf("  list                              - List all tracked TSGs\n");
        printf("  preempt <hClient> <hTsg> [ms]     - Preempt TSG (default 100ms timeout)\n");
        printf("  preempt-pid <pid> [ms]            - Preempt all TSGs of a process\n");
        printf("  timeslice <hClient> <hTsg> <us>   - Set timeslice\n");
        printf("  interleave <hClient> <hTsg> <lvl> - Set interleave (0=LOW,1=MED,2=HIGH)\n");
        printf("  quit                              - Exit\n");
    }
    else if (strcmp(cmd, "quit") == 0 || strcmp(cmd, "exit") == 0) {
        exiting = true;
    }
    else {
        printf("Unknown command: %s (type 'help' for commands)\n", cmd);
    }
}

/* Thread for reading stdin commands */
static void *command_thread(void *arg)
{
    char line[256];

    while (!exiting) {
        printf("gpu> ");
        fflush(stdout);

        if (fgets(line, sizeof(line), stdin) == NULL) {
            break;
        }

        /* Remove newline */
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n')
            line[len-1] = '\0';

        if (line[0] != '\0') {
            handle_command(line);
        }
    }

    return NULL;
}

static void usage(const char *prog)
{
    fprintf(stderr, "Usage: %s [-h] [-v]\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  -h    Show this help\n");
    fprintf(stderr, "  -v    Verbose mode (show all events)\n");
    fprintf(stderr, "\nTraces GPU TSG events via tracepoints and provides preempt control.\n");
    fprintf(stderr, "Requires NVIDIA driver with tracepoints enabled.\n");
    fprintf(stderr, "\nInteractive commands (type 'help' at prompt):\n");
    fprintf(stderr, "  list                    - List tracked TSGs\n");
    fprintf(stderr, "  preempt <hClient> <hTsg> - Preempt a TSG\n");
    fprintf(stderr, "  preempt-pid <pid>       - Preempt all TSGs of a process\n");
}

int main(int argc, char **argv)
{
    struct gpu_preempt_ctrl_bpf *skel;
    struct ring_buffer *rb = NULL;
    pthread_t cmd_thread;
    int err;
    int opt;

    while ((opt = getopt(argc, argv, "hv")) != -1) {
        switch (opt) {
        case 'h':
            usage(argv[0]);
            return 0;
        case 'v':
            verbose = 1;
            break;
        default:
            usage(argv[0]);
            return 1;
        }
    }

    /* Set up libbpf errors and debug info callback */
    libbpf_set_print(libbpf_print_fn);

    /* Set up signal handlers */
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    /* Open NVIDIA device */
    g_nvidia_fd = open_nvidia_device();
    if (g_nvidia_fd < 0) {
        fprintf(stderr, "Warning: Failed to open NVIDIA device: %s\n", strerror(errno));
        fprintf(stderr, "Preempt/control commands will not work.\n");
    } else {
        printf("Opened NVIDIA device (fd=%d)\n", g_nvidia_fd);
    }

    /* Open and load BPF program */
    skel = gpu_preempt_ctrl_bpf__open_and_load();
    if (!skel) {
        fprintf(stderr, "Failed to open and load BPF skeleton\n");
        fprintf(stderr, "Make sure NVIDIA driver with tracepoints is loaded.\n");
        fprintf(stderr, "Check: ls /sys/kernel/tracing/events/nvidia/\n");
        err = 1;
        goto cleanup;
    }

    /* Attach BPF programs */
    err = gpu_preempt_ctrl_bpf__attach(skel);
    if (err) {
        fprintf(stderr, "Failed to attach BPF skeleton: %d\n", err);
        fprintf(stderr, "Tracepoints may not be available.\n");
        fprintf(stderr, "Check: cat /sys/kernel/tracing/available_events | grep nvidia\n");
        goto cleanup;
    }

    /* Set up ring buffer */
    rb = ring_buffer__new(bpf_map__fd(skel->maps.events), handle_event, NULL, NULL);
    if (!rb) {
        fprintf(stderr, "Failed to create ring buffer\n");
        err = -1;
        goto cleanup;
    }

    printf("GPU Preempt Control started. Press Ctrl+C to stop.\n");
    printf("Tracing: nvidia_gpu_tsg_{create,schedule,destroy} tracepoints\n");
    if (!verbose) {
        printf("Use -v for verbose event output.\n");
    }
    printf("Type 'help' for commands.\n");
    printf("---\n");

    /* Start command thread */
    if (pthread_create(&cmd_thread, NULL, command_thread, NULL) != 0) {
        perror("Failed to create command thread");
        err = -1;
        goto cleanup;
    }

    /* Poll ring buffer */
    while (!exiting) {
        err = ring_buffer__poll(rb, 100 /* timeout, ms */);
        if (err == -EINTR) {
            err = 0;
            break;
        }
        if (err < 0) {
            fprintf(stderr, "Error polling ring buffer: %d\n", err);
            break;
        }
    }

    /* Signal command thread to exit */
    pthread_cancel(cmd_thread);
    pthread_join(cmd_thread, NULL);

    print_stats(skel);

cleanup:
    ring_buffer__free(rb);
    gpu_preempt_ctrl_bpf__destroy(skel);
    if (g_nvidia_fd >= 0)
        close(g_nvidia_fd);
    return err < 0 ? 1 : 0;
}
