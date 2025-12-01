/*
 * test_preempt_ioctl.c - Test preempt ioctl with real handles from eBPF
 *
 * This test:
 * 1. Starts a CUDA program in background
 * 2. Uses eBPF to capture hClient/hTsg handles
 * 3. Sends actual preempt ioctl
 * 4. Measures and reports results
 *
 * Build:
 *   gcc -o test_preempt_ioctl test_preempt_ioctl.c -lpthread
 *
 * Run:
 *   sudo ./test_preempt_ioctl ./test_preempt_ctrl
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <stdint.h>
#include <errno.h>
#include <time.h>
#include <signal.h>
#include <linux/ioctl.h>

#define NV_ESC_RM_CONTROL   0x2A
#define NV_IOCTL_MAGIC      'F'
#define NV_IOCTL_BASE       200
#define NV_ESC_IOCTL_XFER_CMD   (NV_IOCTL_BASE + 11)

#define NVA06C_CTRL_CMD_PREEMPT             0xa06c0105
#define NVA06C_CTRL_CMD_SET_TIMESLICE       0xa06c0103
#define NVA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL 0xa06c0107

/* nv_ioctl_xfer_t for indirect ioctl */
typedef struct {
    uint32_t cmd;
    uint32_t size;
    void    *ptr __attribute__((aligned(8)));
} nv_ioctl_xfer_t;

typedef struct {
    uint32_t hClient;
    uint32_t hObject;
    uint32_t cmd;
    uint32_t flags;
    void    *params __attribute__((aligned(8)));
    uint32_t paramsSize;
    uint32_t status;
} NVOS54_PARAMETERS;

typedef struct {
    uint8_t  bWait;
    uint8_t  bManualTimeout;
    uint32_t timeoutUs;
} NVA06C_CTRL_PREEMPT_PARAMS;

typedef struct {
    uint64_t timesliceUs;
} NVA06C_CTRL_TIMESLICE_PARAMS;

typedef struct {
    uint32_t tsgInterleaveLevel;
} NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS;

static uint64_t get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

static int open_nvidia_device(void) {
    int fd = open("/dev/nvidiactl", O_RDWR);
    if (fd < 0) {
        fd = open("/dev/nvidia0", O_RDWR);
    }
    return fd;
}

static int rm_control(int fd, uint32_t hClient, uint32_t hObject,
                      uint32_t cmd, void *params, uint32_t paramsSize) {
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

    /* Use _IOWR to encode the ioctl command with proper size */
    ret = ioctl(fd, _IOWR(NV_IOCTL_MAGIC, NV_ESC_IOCTL_XFER_CMD, nv_ioctl_xfer_t), &xfer);

    if (ret < 0) {
        fprintf(stderr, "    [DEBUG] ioctl failed: ret=%d errno=%d (%s)\n", ret, errno, strerror(errno));
        return -errno;
    }
    fprintf(stderr, "    [DEBUG] ioctl succeeded: ret=%d ctrl.status=0x%x\n", ret, ctrl.status);
    return ctrl.status;
}

static int do_preempt(int fd, uint32_t hClient, uint32_t hTsg, uint32_t timeout_ms) {
    NVA06C_CTRL_PREEMPT_PARAMS params;
    uint64_t start, end;
    int ret;

    memset(&params, 0, sizeof(params));
    params.bWait = 1;
    params.bManualTimeout = 1;
    params.timeoutUs = timeout_ms * 1000;

    start = get_time_us();
    ret = rm_control(fd, hClient, hTsg, NVA06C_CTRL_CMD_PREEMPT, &params, sizeof(params));
    end = get_time_us();

    printf("  PREEMPT hClient=0x%08x hTsg=0x%08x result=%d (0x%x) duration=%lu us\n",
           hClient, hTsg, ret, ret, (unsigned long)(end - start));

    return ret;
}

static int do_set_timeslice(int fd, uint32_t hClient, uint32_t hTsg, uint64_t timeslice_us) {
    NVA06C_CTRL_TIMESLICE_PARAMS params;
    int ret;

    memset(&params, 0, sizeof(params));
    params.timesliceUs = timeslice_us;

    ret = rm_control(fd, hClient, hTsg, NVA06C_CTRL_CMD_SET_TIMESLICE, &params, sizeof(params));
    printf("  SET_TIMESLICE hClient=0x%08x hTsg=0x%08x timeslice=%lu us result=%d\n",
           hClient, hTsg, (unsigned long)timeslice_us, ret);

    return ret;
}

static int do_set_interleave(int fd, uint32_t hClient, uint32_t hTsg, uint32_t level) {
    NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS params;
    const char *level_names[] = {"LOW", "MEDIUM", "HIGH"};
    int ret;

    memset(&params, 0, sizeof(params));
    params.tsgInterleaveLevel = level;

    ret = rm_control(fd, hClient, hTsg, NVA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL, &params, sizeof(params));
    printf("  SET_INTERLEAVE hClient=0x%08x hTsg=0x%08x level=%s result=%d\n",
           hClient, hTsg, level < 3 ? level_names[level] : "UNKNOWN", ret);

    return ret;
}

/* Parse handle from trace output line like:
 * "nvidia_gpu_tsg_schedule: hClient=0xc1d000cb hTsg=0x5c000038 ..."
 */
static int parse_trace_line(const char *line, uint32_t *hClient, uint32_t *hTsg) {
    const char *p;

    p = strstr(line, "hClient=");
    if (!p) return -1;
    *hClient = strtoul(p + 8, NULL, 16);

    p = strstr(line, "hTsg=");
    if (!p) return -1;
    *hTsg = strtoul(p + 5, NULL, 16);

    return 0;
}

int main(int argc, char *argv[]) {
    int nvidia_fd;
    pid_t cuda_pid;
    char trace_cmd[512];
    FILE *trace_fp;
    char line[1024];
    uint32_t hClient = 0, hTsg = 0;
    int found_handle = 0;
    const char *cuda_prog = argc > 1 ? argv[1] : "./test_preempt_ctrl";

    printf("=== GPU Preempt ioctl Test ===\n\n");

    /* Open NVIDIA device */
    nvidia_fd = open_nvidia_device();
    if (nvidia_fd < 0) {
        fprintf(stderr, "Failed to open NVIDIA device: %s\n", strerror(errno));
        return 1;
    }
    printf("Opened NVIDIA device (fd=%d)\n", nvidia_fd);

    /* Enable tracepoint */
    printf("\nEnabling tracepoints...\n");
    system("echo 1 | sudo tee /sys/kernel/tracing/events/nvidia/nvidia_gpu_tsg_schedule/enable > /dev/null 2>&1");
    system("echo 1 | sudo tee /sys/kernel/tracing/events/nvidia/nvidia_gpu_tsg_create/enable > /dev/null 2>&1");
    system("echo | sudo tee /sys/kernel/tracing/trace > /dev/null 2>&1");  /* Clear trace buffer */

    /* Start CUDA program */
    printf("Starting CUDA program: %s\n", cuda_prog);
    cuda_pid = fork();
    if (cuda_pid == 0) {
        /* Child - run CUDA program */
        execl(cuda_prog, cuda_prog, "0", "10000000000", NULL);
        perror("execl failed");
        exit(1);
    } else if (cuda_pid < 0) {
        perror("fork failed");
        close(nvidia_fd);
        return 1;
    }

    printf("CUDA program started with PID %d\n", cuda_pid);

    /* Wait for TSG creation */
    printf("Waiting for TSG creation...\n");
    sleep(3);

    /* Read trace to get handles */
    printf("\nReading trace for handles...\n");
    trace_fp = popen("cat /sys/kernel/tracing/trace | grep nvidia_gpu_tsg_schedule | tail -5", "r");
    if (trace_fp) {
        while (fgets(line, sizeof(line), trace_fp)) {
            printf("  Trace: %s", line);
            if (parse_trace_line(line, &hClient, &hTsg) == 0) {
                found_handle = 1;
                printf("  -> Found: hClient=0x%08x hTsg=0x%08x\n", hClient, hTsg);
            }
        }
        pclose(trace_fp);
    }

    if (!found_handle) {
        printf("\nNo handles found in trace. Checking tsg_create events...\n");
        trace_fp = popen("cat /sys/kernel/tracing/trace | grep nvidia_gpu_tsg_create | tail -5", "r");
        if (trace_fp) {
            while (fgets(line, sizeof(line), trace_fp)) {
                printf("  Trace: %s", line);
                if (parse_trace_line(line, &hClient, &hTsg) == 0) {
                    found_handle = 1;
                    printf("  -> Found: hClient=0x%08x hTsg=0x%08x\n", hClient, hTsg);
                }
            }
            pclose(trace_fp);
        }
    }

    if (!found_handle) {
        printf("\nERROR: No handles found!\n");
        printf("Make sure the NVIDIA driver has tracepoints enabled.\n");
        kill(cuda_pid, SIGTERM);
        waitpid(cuda_pid, NULL, 0);
        close(nvidia_fd);
        return 1;
    }

    /* Test preempt */
    printf("\n=== Testing PREEMPT ===\n");
    do_preempt(nvidia_fd, hClient, hTsg, 100);

    /* Test timeslice change */
    printf("\n=== Testing SET_TIMESLICE ===\n");
    do_set_timeslice(nvidia_fd, hClient, hTsg, 2000);  /* 2ms */

    /* Test interleave level change */
    printf("\n=== Testing SET_INTERLEAVE_LEVEL ===\n");
    do_set_interleave(nvidia_fd, hClient, hTsg, 2);  /* HIGH */

    /* Another preempt test */
    printf("\n=== Testing PREEMPT again ===\n");
    do_preempt(nvidia_fd, hClient, hTsg, 100);

    /* Cleanup */
    printf("\nStopping CUDA program...\n");
    kill(cuda_pid, SIGTERM);
    waitpid(cuda_pid, NULL, 0);

    /* Disable tracepoints */
    system("echo 0 | sudo tee /sys/kernel/tracing/events/nvidia/nvidia_gpu_tsg_schedule/enable > /dev/null 2>&1");
    system("echo 0 | sudo tee /sys/kernel/tracing/events/nvidia/nvidia_gpu_tsg_create/enable > /dev/null 2>&1");

    close(nvidia_fd);
    printf("\nDone.\n");
    return 0;
}
