/*
 * test_preempt_ctrl.cu - Simple CUDA test program for gpu_preempt_ctrl
 *
 * Launches a long-running GPU kernel that can be preempted by the
 * gpu_preempt_ctrl tool.
 *
 * Build:
 *   nvcc -o test_preempt_ctrl test_preempt_ctrl.cu
 *
 * Run:
 *   ./test_preempt_ctrl
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>

static volatile int running = 1;

void signal_handler(int sig) {
    printf("\nReceived signal %d, stopping...\n", sig);
    running = 0;
}

// Long-running kernel - busy loop
__global__ void busy_kernel(unsigned long long iterations, int *output) {
    unsigned long long count = 0;
    for (unsigned long long i = 0; i < iterations; i++) {
        count += i;
    }
    // Prevent optimization
    if (threadIdx.x == 0) {
        *output = (int)(count & 0xFFFFFFFF);
    }
}

// Simple kernel for quick test
__global__ void simple_kernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2 + 1;
    }
}

int main(int argc, char *argv[]) {
    int *d_data;
    int *d_output;
    int h_output;
    int iterations = 10;
    unsigned long long loop_count = 10000000000ULL; // Very long loop

    printf("=== GPU Preempt Control Test ===\n");
    printf("PID: %d\n\n", getpid());

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Parse arguments
    if (argc > 1) {
        iterations = atoi(argv[1]);
    }
    if (argc > 2) {
        loop_count = strtoull(argv[2], NULL, 10);
    }

    printf("Iterations: %d\n", iterations);
    printf("Loop count per kernel: %llu\n\n", loop_count);

    // Get device info
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    printf("Using GPU: %s\n", prop.name);

    // Allocate device memory
    cudaMalloc(&d_data, 1024 * sizeof(int));
    cudaMalloc(&d_output, sizeof(int));

    printf("\nStarting GPU kernels...\n");
    printf("Run 'sudo ./gpu_preempt_ctrl -v' in another terminal to monitor.\n");
    printf("Use 'preempt-pid %d' command to preempt this process.\n\n", getpid());

    int iter = 0;
    while (running && (iterations <= 0 || iter < iterations)) {
        printf("[%d] Launching busy_kernel (loop_count=%llu)...\n", iter, loop_count);

        busy_kernel<<<1, 1>>>(loop_count, d_output);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("[%d] Kernel sync result: %s\n", iter, cudaGetErrorString(err));
        } else {
            cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
            printf("[%d] Kernel completed, output=%d\n", iter, h_output);
        }

        iter++;

        if (running && iterations > 0 && iter < iterations) {
            sleep(1);
        }
    }

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_output);

    printf("\nDone. Total iterations: %d\n", iter);
    return 0;
}
