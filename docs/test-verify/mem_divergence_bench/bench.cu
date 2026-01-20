/**
 * GPU Memory Access & Thread Divergence Formal Benchmark
 *
 * Evaluates:
 * 1. Memory coalescing efficiency (stride access patterns)
 * 2. Thread divergence overhead (branch factor)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Configuration
#define DATA_SIZE (1 << 20)  // 1M elements
#define ITERATIONS 100
#define BLOCK_SIZE 256

// ==================== Memory Coalescing Kernel ====================
// Formal parameter: stride - controls memory access pattern
// stride=1: coalesced (optimal), stride=32: non-coalesced (worst)
__global__ void mem_coalescing_kernel(float* data, int stride, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (tid * stride) % n;

    // Read-modify-write to prevent optimization
    float val = data[idx];
    val = val * 1.01f + 0.01f;
    data[idx] = val;
}

// ==================== Thread Divergence Kernel ====================
// Formal parameter: div_factor - controls branch divergence
// div_factor=1: no divergence, div_factor=32: max divergence (32 paths)
__device__ float compute_path(int path, int work_amount) {
    float result = 0.0f;
    for (int i = 0; i < work_amount; i++) {
        switch (path) {
            case 0:  result += sinf((float)i * 0.01f); break;
            case 1:  result += cosf((float)i * 0.01f); break;
            case 2:  result += tanf((float)i * 0.001f); break;
            case 3:  result += expf((float)i * 0.0001f); break;
            case 4:  result += logf((float)(i + 1)); break;
            case 5:  result += sqrtf((float)(i + 1)); break;
            case 6:  result += sinf((float)i * 0.02f); break;
            case 7:  result += cosf((float)i * 0.02f); break;
            case 8:  result += sinf((float)i * 0.03f); break;
            case 9:  result += cosf((float)i * 0.03f); break;
            case 10: result += sinf((float)i * 0.04f); break;
            case 11: result += cosf((float)i * 0.04f); break;
            case 12: result += sinf((float)i * 0.05f); break;
            case 13: result += cosf((float)i * 0.05f); break;
            case 14: result += sinf((float)i * 0.06f); break;
            case 15: result += cosf((float)i * 0.06f); break;
            case 16: result += sinf((float)i * 0.07f); break;
            case 17: result += cosf((float)i * 0.07f); break;
            case 18: result += sinf((float)i * 0.08f); break;
            case 19: result += cosf((float)i * 0.08f); break;
            case 20: result += sinf((float)i * 0.09f); break;
            case 21: result += cosf((float)i * 0.09f); break;
            case 22: result += sinf((float)i * 0.10f); break;
            case 23: result += cosf((float)i * 0.10f); break;
            case 24: result += sinf((float)i * 0.11f); break;
            case 25: result += cosf((float)i * 0.11f); break;
            case 26: result += sinf((float)i * 0.12f); break;
            case 27: result += cosf((float)i * 0.12f); break;
            case 28: result += sinf((float)i * 0.13f); break;
            case 29: result += cosf((float)i * 0.13f); break;
            case 30: result += sinf((float)i * 0.14f); break;
            default: result += cosf((float)i * 0.15f); break;
        }
    }
    return result;
}

__global__ void thread_divergence_kernel(float* data, int div_factor, int work_amount) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane_id = threadIdx.x % 32;  // Position within warp
    int path = lane_id % div_factor;  // Which path this thread takes

    data[tid] = compute_path(path, work_amount);
}

// ==================== Timing Utilities ====================
float run_memory_benchmark(float* d_data, int stride, int n) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Warmup
    mem_coalescing_kernel<<<blocks, BLOCK_SIZE>>>(d_data, stride, n);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        mem_coalescing_kernel<<<blocks, BLOCK_SIZE>>>(d_data, stride, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / ITERATIONS;  // Average time per iteration
}

float run_divergence_benchmark(float* d_data, int div_factor, int work_amount) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (DATA_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Warmup
    thread_divergence_kernel<<<blocks, BLOCK_SIZE>>>(d_data, div_factor, work_amount);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        thread_divergence_kernel<<<blocks, BLOCK_SIZE>>>(d_data, div_factor, work_amount);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / ITERATIONS;
}

// ==================== Main ====================
int main(int argc, char** argv) {
    // Get GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float theoretical_bandwidth = prop.memoryClockRate * 1000.0f * (prop.memoryBusWidth / 8) * 2 / 1e9;  // GB/s

    printf("GPU: %s\n", prop.name);
    printf("Theoretical Memory Bandwidth: %.1f GB/s\n", theoretical_bandwidth);
    printf("Data size: %d elements (%.2f MB)\n\n", DATA_SIZE, DATA_SIZE * sizeof(float) / 1e6);

    // Allocate memory
    float *d_data;
    cudaMalloc(&d_data, DATA_SIZE * sizeof(float));
    cudaMemset(d_data, 0, DATA_SIZE * sizeof(float));

    // Open CSV file
    FILE* csv = fopen("results.csv", "w");
    fprintf(csv, "test_type,parameter,time_ms,bandwidth_gbps,efficiency\n");

    // ==================== Memory Coalescing Benchmark ====================
    printf("=== Memory Coalescing Benchmark ===\n");
    printf("%-10s %10s %15s %12s\n", "stride", "time(ms)", "bandwidth(GB/s)", "efficiency");
    printf("--------------------------------------------------\n");

    int strides[] = {1, 2, 4, 8, 16, 32};
    int num_strides = sizeof(strides) / sizeof(strides[0]);

    for (int i = 0; i < num_strides; i++) {
        int stride = strides[i];
        float time_ms = run_memory_benchmark(d_data, stride, DATA_SIZE);

        // Calculate effective bandwidth (bytes transferred per second)
        // Each thread does 1 read + 1 write = 8 bytes
        float bytes_transferred = (float)DATA_SIZE * 2 * sizeof(float) * ITERATIONS;
        float bandwidth = bytes_transferred / (time_ms * ITERATIONS * 1e6);  // GB/s
        float efficiency = bandwidth / theoretical_bandwidth;

        printf("%-10d %10.3f %15.1f %11.1f%%\n", stride, time_ms, bandwidth, efficiency * 100);
        fprintf(csv, "memory,%d,%.4f,%.2f,%.4f\n", stride, time_ms, bandwidth, efficiency);
    }

    // ==================== Thread Divergence Benchmark ====================
    printf("\n=== Thread Divergence Benchmark ===\n");
    printf("%-10s %10s %12s\n", "div_factor", "time(ms)", "slowdown");
    printf("-------------------------------------\n");

    int div_factors[] = {1, 2, 4, 8, 16, 32};
    int num_divs = sizeof(div_factors) / sizeof(div_factors[0]);
    int work_amount = 100;  // Iterations per path
    float div_baseline = 0;

    for (int i = 0; i < num_divs; i++) {
        int div_factor = div_factors[i];
        float time_ms = run_divergence_benchmark(d_data, div_factor, work_amount);

        if (i == 0) div_baseline = time_ms;
        float slowdown = time_ms / div_baseline;

        printf("%-10d %10.3f %11.2fx\n", div_factor, time_ms, slowdown);
        fprintf(csv, "divergence,%d,%.4f,NA,%.4f\n", div_factor, time_ms, 1.0f / slowdown);
    }

    fclose(csv);
    printf("\nResults saved to: results.csv\n");

    // Cleanup
    cudaFree(d_data);

    return 0;
}
