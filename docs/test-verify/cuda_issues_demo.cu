/**
 * CUDA Programming Issues Demo
 *
 * This program demonstrates common CUDA programming pitfalls:
 * 1. Thread Divergence - performance degradation from divergent branches
 * 2. Deadlocks - __syncthreads() in divergent code, spinlocks
 * 3. Memory Bottlenecks - non-coalesced access, bank conflicts
 *
 * Compile: nvcc -O2 -arch=sm_70 -o cuda_issues_demo cuda_issues_demo.cu
 * Run: ./cuda_issues_demo
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

//=============================================================================
// PART 1: THREAD DIVERGENCE
//=============================================================================

// BAD: Highly divergent - each thread may take different path
__global__ void divergent_kernel(float *data, float *output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    float val = data[tid];

    // Divergence: different threads take different branches based on data
    if (val < 0.25f) {
        output[tid] = val * 2.0f;
    } else if (val < 0.5f) {
        output[tid] = val * 3.0f;
    } else if (val < 0.75f) {
        output[tid] = val * 4.0f;
    } else {
        output[tid] = val * 5.0f;
    }
}

// GOOD: No divergence - all threads execute same path
__global__ void uniform_kernel(float *data, float *output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    float val = data[tid];

    // No divergence: predicated execution or same path for all
    // Using arithmetic instead of branches
    float multiplier = 2.0f + 3.0f * val;  // Smooth function, no branches
    output[tid] = val * multiplier;
}

// EXTREME divergence: thread-ID based branches
__global__ void extreme_divergent_kernel(float *data, float *output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    float val = data[tid];
    float result = 0.0f;

    // Each thread in warp takes different path based on lane ID
    int lane = threadIdx.x % 32;
    switch (lane % 8) {
        case 0: result = val * 1.0f; break;
        case 1: result = val * 2.0f; break;
        case 2: result = val * 3.0f; break;
        case 3: result = val * 4.0f; break;
        case 4: result = val * 5.0f; break;
        case 5: result = val * 6.0f; break;
        case 6: result = val * 7.0f; break;
        case 7: result = val * 8.0f; break;
    }

    output[tid] = result;
}

void test_divergence() {
    printf("=============================================================\n");
    printf("PART 1: THREAD DIVERGENCE\n");
    printf("=============================================================\n\n");

    const int N = 1024 * 1024 * 16;
    const int SIZE = N * sizeof(float);
    const int THREADS = 256;
    const int BLOCKS = (N + THREADS - 1) / THREADS;
    const int ITERATIONS = 100;

    float *h_data = (float*)malloc(SIZE);
    float *d_data, *d_output;

    // Initialize with random data to cause divergence
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)rand() / RAND_MAX;
    }

    cudaMalloc(&d_data, SIZE);
    cudaMalloc(&d_output, SIZE);
    cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    uniform_kernel<<<BLOCKS, THREADS>>>(d_data, d_output, N);
    cudaDeviceSynchronize();

    // Test uniform kernel
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        uniform_kernel<<<BLOCKS, THREADS>>>(d_data, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float uniform_time;
    cudaEventElapsedTime(&uniform_time, start, stop);

    // Test divergent kernel
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        divergent_kernel<<<BLOCKS, THREADS>>>(d_data, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float divergent_time;
    cudaEventElapsedTime(&divergent_time, start, stop);

    // Test extreme divergent kernel
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        extreme_divergent_kernel<<<BLOCKS, THREADS>>>(d_data, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float extreme_time;
    cudaEventElapsedTime(&extreme_time, start, stop);

    printf("Data-dependent branch divergence test:\n");
    printf("  Uniform kernel (no branches):     %8.2f ms\n", uniform_time);
    printf("  Divergent kernel (4 branches):    %8.2f ms (%.2fx slower)\n",
           divergent_time, divergent_time / uniform_time);
    printf("  Extreme divergent (8 branches):   %8.2f ms (%.2fx slower)\n\n",
           extreme_time, extreme_time / uniform_time);

    printf("Explanation:\n");
    printf("  - Uniform: All threads execute same arithmetic operations\n");
    printf("  - Divergent: Threads take different paths based on data values\n");
    printf("  - Extreme: Each lane in warp takes different switch case\n");
    printf("  - GPU must serialize divergent paths, reducing throughput\n\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_output);
    free(h_data);
}

//=============================================================================
// PART 2: DEADLOCK DEMONSTRATIONS
//=============================================================================

// DEADLOCK EXAMPLE 1: __syncthreads() in divergent code
// WARNING: This WILL hang the GPU if uncommented!
__global__ void deadlock_syncthreads_kernel(int *data) {
    int tid = threadIdx.x;

    // Only some threads enter this branch
    if (tid < 16) {
        data[tid] = tid * 2;

        // DEADLOCK! Threads 16-31 never reach this barrier
        // __syncthreads();  // UNCOMMENT TO CAUSE DEADLOCK

        data[tid] += 1;
    }
}

// DEADLOCK EXAMPLE 2: Spinlock within warp
// WARNING: This WILL hang the GPU if uncommented!
__device__ int spinlock = 0;

__global__ void deadlock_spinlock_kernel(int *data) {
    int tid = threadIdx.x;

    // All threads try to acquire the same lock
    // Thread 0 gets it, but can't release until warp converges
    // Other threads spin forever waiting for lock

    // DEADLOCK pattern - DO NOT USE:
    // while (atomicCAS(&spinlock, 0, 1) != 0) { }  // Acquire
    // data[tid] = tid;  // Critical section
    // atomicExch(&spinlock, 0);  // Release - never reached!

    // Safe version: use atomics without spinlock
    atomicAdd(&data[0], 1);
}

// DEADLOCK EXAMPLE 3: Producer-consumer within warp
__device__ volatile int producer_flag = 0;

__global__ void deadlock_producer_consumer_kernel(int *data) {
    int tid = threadIdx.x;

    if (tid == 0) {
        // Producer
        data[0] = 42;
        __threadfence();
        // producer_flag = 1;  // Signal - but warp can't converge!
    } else {
        // Consumer - spins waiting for producer
        // while (producer_flag == 0) { }  // DEADLOCK!
        // int val = data[0];
    }

    // Safe version: all threads do same work
    data[tid] = tid;
}

void test_deadlock_explanation() {
    printf("=============================================================\n");
    printf("PART 2: DEADLOCK PATTERNS (Explanation Only - Not Executed)\n");
    printf("=============================================================\n\n");

    printf("Deadlock Pattern 1: __syncthreads() in Divergent Code\n");
    printf("─────────────────────────────────────────────────────────────\n");
    printf("  if (threadIdx.x < 16) {\n");
    printf("      // Only threads 0-15 enter\n");
    printf("      __syncthreads();  // DEADLOCK: threads 16-31 never arrive\n");
    printf("  }\n\n");
    printf("  Why: __syncthreads() requires ALL threads in block to participate.\n");
    printf("       If some threads skip it, the kernel hangs forever.\n\n");

    printf("Deadlock Pattern 2: Spinlock Within Warp\n");
    printf("─────────────────────────────────────────────────────────────\n");
    printf("  while (atomicCAS(&lock, 0, 1) != 0) { }  // Acquire\n");
    printf("  // critical section\n");
    printf("  atomicExch(&lock, 0);  // Release\n\n");
    printf("  Why: In SIMT, thread 0 acquires lock but can't proceed to release\n");
    printf("       until warp converges. Threads 1-31 spin forever waiting.\n");
    printf("       This is a circular dependency → DEADLOCK.\n\n");

    printf("Deadlock Pattern 3: Producer-Consumer Within Warp\n");
    printf("─────────────────────────────────────────────────────────────\n");
    printf("  if (tid == 0) {\n");
    printf("      data = compute();\n");
    printf("      flag = 1;  // Signal consumers\n");
    printf("  } else {\n");
    printf("      while (flag == 0);  // Wait for producer - DEADLOCK!\n");
    printf("  }\n\n");
    printf("  Why: Producer can't set flag until warp converges, but consumers\n");
    printf("       are spinning waiting for flag. Neither can proceed.\n\n");

    printf("NOTE: These kernels are not executed to avoid hanging the GPU.\n");
    printf("      In production, these patterns cause unrecoverable hangs.\n\n");
}

//=============================================================================
// PART 3: MEMORY BOTTLENECKS
//=============================================================================

// COALESCED: Consecutive threads access consecutive memory
__global__ void coalesced_access_kernel(float *data, float *output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        // GOOD: Thread i accesses element i
        output[tid] = data[tid] * 2.0f;
    }
}

// NON-COALESCED: Strided access pattern
__global__ void strided_access_kernel(float *data, float *output, int n, int stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = (tid * stride) % n;
    if (tid < n) {
        // BAD: Thread i accesses element i*stride
        // Each thread hits different cache line!
        output[tid] = data[idx] * 2.0f;
    }
}

// RANDOM ACCESS: Completely unpredictable pattern
__global__ void random_access_kernel(float *data, float *output, int *indices, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        // WORST: Random access pattern
        int idx = indices[tid];
        output[tid] = data[idx] * 2.0f;
    }
}

// BANK CONFLICT: Shared memory access with conflicts
__global__ void bank_conflict_kernel(float *data, float *output, int n) {
    __shared__ float smem[32][33];  // Padded to avoid conflicts

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_tid = tid + bid * blockDim.x;

    if (global_tid < n) {
        // Load to shared memory
        smem[tid][0] = data[global_tid];
        __syncthreads();

        // BAD: All threads access column 0 → 32-way bank conflict
        // float val = smem[tid][0];  // Conflict!

        // GOOD: Each thread accesses different column
        float val = smem[tid][tid % 32];  // No conflict

        output[global_tid] = val * 2.0f;
    }
}

// EXCESSIVE MEMORY TRAFFIC: Reading same data multiple times
__global__ void excessive_traffic_kernel(float *data, float *coeffs, float *output,
                                          int n, int num_coeffs) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float sum = 0.0f;

        // BAD: Reading data[tid] from global memory every iteration
        for (int i = 0; i < num_coeffs; i++) {
            sum += data[tid] * coeffs[i];  // data[tid] read 100 times!
        }

        output[tid] = sum;
    }
}

// OPTIMIZED: Cache in register
__global__ void cached_access_kernel(float *data, float *coeffs, float *output,
                                      int n, int num_coeffs) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float sum = 0.0f;
        float local_data = data[tid];  // Cache in register - read once!

        for (int i = 0; i < num_coeffs; i++) {
            sum += local_data * coeffs[i];  // Register access
        }

        output[tid] = sum;
    }
}

void test_memory_bottlenecks() {
    printf("=============================================================\n");
    printf("PART 3: MEMORY BOTTLENECKS\n");
    printf("=============================================================\n\n");

    const int N = 1024 * 1024 * 4;
    const int SIZE = N * sizeof(float);
    const int THREADS = 256;
    const int BLOCKS = (N + THREADS - 1) / THREADS;
    const int ITERATIONS = 100;
    const int NUM_COEFFS = 100;

    float *h_data = (float*)malloc(SIZE);
    float *h_coeffs = (float*)malloc(NUM_COEFFS * sizeof(float));
    int *h_indices = (int*)malloc(N * sizeof(int));
    float *d_data, *d_output, *d_coeffs;
    int *d_indices;

    // Initialize
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
        h_indices[i] = rand() % N;  // Random indices
    }
    for (int i = 0; i < NUM_COEFFS; i++) {
        h_coeffs[i] = 1.0f / (i + 1);
    }

    cudaMalloc(&d_data, SIZE);
    cudaMalloc(&d_output, SIZE);
    cudaMalloc(&d_indices, N * sizeof(int));
    cudaMalloc(&d_coeffs, NUM_COEFFS * sizeof(float));
    cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coeffs, h_coeffs, NUM_COEFFS * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    coalesced_access_kernel<<<BLOCKS, THREADS>>>(d_data, d_output, N);
    cudaDeviceSynchronize();

    printf("Test A: Memory Access Patterns\n");
    printf("─────────────────────────────────────────────────────────────\n");

    // Test coalesced access
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        coalesced_access_kernel<<<BLOCKS, THREADS>>>(d_data, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float coalesced_time;
    cudaEventElapsedTime(&coalesced_time, start, stop);
    float coalesced_bw = (2.0f * SIZE * ITERATIONS) / (coalesced_time / 1000.0f) / 1e9;

    // Test strided access (stride = 32, worst case)
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        strided_access_kernel<<<BLOCKS, THREADS>>>(d_data, d_output, N, 32);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float strided_time;
    cudaEventElapsedTime(&strided_time, start, stop);
    float strided_bw = (2.0f * SIZE * ITERATIONS) / (strided_time / 1000.0f) / 1e9;

    // Test random access
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        random_access_kernel<<<BLOCKS, THREADS>>>(d_data, d_output, d_indices, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float random_time;
    cudaEventElapsedTime(&random_time, start, stop);
    float random_bw = (2.0f * SIZE * ITERATIONS) / (random_time / 1000.0f) / 1e9;

    printf("  Coalesced access:    %8.2f ms  (%.1f GB/s effective)\n",
           coalesced_time, coalesced_bw);
    printf("  Strided (stride=32): %8.2f ms  (%.1f GB/s effective) - %.1fx slower\n",
           strided_time, strided_bw, strided_time / coalesced_time);
    printf("  Random access:       %8.2f ms  (%.1f GB/s effective) - %.1fx slower\n\n",
           random_time, random_bw, random_time / coalesced_time);

    printf("Test B: Register Caching vs Repeated Global Access\n");
    printf("─────────────────────────────────────────────────────────────\n");

    // Test excessive memory traffic
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        excessive_traffic_kernel<<<BLOCKS, THREADS>>>(d_data, d_coeffs, d_output, N, NUM_COEFFS);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float excessive_time;
    cudaEventElapsedTime(&excessive_time, start, stop);

    // Test cached access
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        cached_access_kernel<<<BLOCKS, THREADS>>>(d_data, d_coeffs, d_output, N, NUM_COEFFS);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cached_time;
    cudaEventElapsedTime(&cached_time, start, stop);

    printf("  Repeated global reads (%dx): %8.2f ms\n", NUM_COEFFS, excessive_time);
    printf("  Register cached:             %8.2f ms - %.1fx faster\n\n",
           cached_time, excessive_time / cached_time);

    printf("Explanation:\n");
    printf("  - Coalesced: 32 threads access 32 consecutive floats = 1 transaction\n");
    printf("  - Strided: 32 threads access 32 different cache lines = 32 transactions\n");
    printf("  - Random: Unpredictable pattern, no coalescing possible\n");
    printf("  - Caching: Store frequently used data in registers (not global memory)\n\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_output);
    cudaFree(d_indices);
    cudaFree(d_coeffs);
    free(h_data);
    free(h_coeffs);
    free(h_indices);
}

//=============================================================================
// PART 4: COMBINED PATHOLOGICAL CASE
//=============================================================================

// WORST CASE: Combines all issues - divergence + non-coalesced + contention
__global__ void pathological_kernel(float *data, float *output, int *shared_counter, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    float val = data[tid];

    // Issue 1: Thread divergence based on data
    float result;
    if (val < 0.25f) {
        // Issue 2: Non-coalesced access (strided read)
        int idx = (tid * 17) % n;
        result = data[idx] * 2.0f;
    } else if (val < 0.5f) {
        int idx = (tid * 31) % n;
        result = data[idx] * 3.0f;
    } else if (val < 0.75f) {
        int idx = (tid * 47) % n;
        result = data[idx] * 4.0f;
    } else {
        int idx = (tid * 63) % n;
        result = data[idx] * 5.0f;
    }

    // Issue 3: Atomic contention on single counter
    atomicAdd(shared_counter, 1);

    output[tid] = result;
}

// GOOD VERSION: No divergence, coalesced access, per-thread counters
__global__ void optimized_kernel(float *data, float *output, int *per_thread_counts, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    float val = data[tid];

    // No divergence: arithmetic instead of branches
    float multiplier = 2.0f + 3.0f * val;

    // Coalesced access
    float result = data[tid] * multiplier;

    // Per-thread counter (no contention)
    per_thread_counts[tid] = 1;

    output[tid] = result;
}

void test_combined_issues() {
    printf("=============================================================\n");
    printf("PART 4: COMBINED PATHOLOGICAL CASE\n");
    printf("=============================================================\n\n");

    const int N = 1024 * 1024;
    const int SIZE = N * sizeof(float);
    const int THREADS = 256;
    const int BLOCKS = (N + THREADS - 1) / THREADS;
    const int ITERATIONS = 50;

    float *h_data = (float*)malloc(SIZE);
    float *d_data, *d_output;
    int *d_counter, *d_per_thread;

    for (int i = 0; i < N; i++) {
        h_data[i] = (float)rand() / RAND_MAX;
    }

    cudaMalloc(&d_data, SIZE);
    cudaMalloc(&d_output, SIZE);
    cudaMalloc(&d_counter, sizeof(int));
    cudaMalloc(&d_per_thread, N * sizeof(int));
    cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);

    int zero = 0;
    cudaMemcpy(d_counter, &zero, sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    optimized_kernel<<<BLOCKS, THREADS>>>(d_data, d_output, d_per_thread, N);
    cudaDeviceSynchronize();

    // Test optimized kernel
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        optimized_kernel<<<BLOCKS, THREADS>>>(d_data, d_output, d_per_thread, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float optimized_time;
    cudaEventElapsedTime(&optimized_time, start, stop);

    // Test pathological kernel
    cudaMemcpy(d_counter, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        pathological_kernel<<<BLOCKS, THREADS>>>(d_data, d_output, d_counter, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float pathological_time;
    cudaEventElapsedTime(&pathological_time, start, stop);

    printf("Pathological kernel combines:\n");
    printf("  1. Data-dependent branch divergence (4 paths)\n");
    printf("  2. Non-coalesced memory access (different strides per path)\n");
    printf("  3. Atomic contention (all threads update same counter)\n\n");

    printf("Results (%d iterations):\n", ITERATIONS);
    printf("─────────────────────────────────────────────────────────────\n");
    printf("  Optimized kernel:    %8.2f ms\n", optimized_time);
    printf("  Pathological kernel: %8.2f ms\n", pathological_time);
    printf("  Slowdown:            %.1fx\n\n", pathological_time / optimized_time);

    printf("This demonstrates how multiple issues compound:\n");
    printf("  - Each issue alone might cause 2-4x slowdown\n");
    printf("  - Combined, they can cause 10-100x slowdown\n");
    printf("  - In multi-tenant GPU environments, this is a DoS vector\n\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_output);
    cudaFree(d_counter);
    cudaFree(d_per_thread);
    free(h_data);
}

//=============================================================================
// MAIN
//=============================================================================

int main(int argc, char **argv) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║     CUDA Programming Issues Demo                              ║\n");
    printf("║     Thread Divergence, Deadlocks, and Memory Bottlenecks      ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    // Check CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Max Threads/Block: %d\n", prop.maxThreadsPerBlock);
    printf("Memory: %.1f GB\n\n", prop.totalGlobalMem / 1e9);

    // Run tests
    test_divergence();
    test_deadlock_explanation();
    test_memory_bottlenecks();
    test_combined_issues();

    printf("=============================================================\n");
    printf("SUMMARY\n");
    printf("=============================================================\n\n");
    printf("These issues are critical for GPU eBPF verification because:\n\n");
    printf("1. eBPF hooks run on EVERY GPU thread\n");
    printf("   → Divergent eBPF code causes warp-wide stalls\n\n");
    printf("2. eBPF map operations are memory accesses\n");
    printf("   → Non-uniform map access causes contention\n\n");
    printf("3. eBPF helpers may require synchronization\n");
    printf("   → Divergent helper calls serialize execution\n\n");
    printf("4. Traditional eBPF verification checks:\n");
    printf("   ✓ Memory safety\n");
    printf("   ✓ Bounded loops\n");
    printf("   ✓ Valid helper usage\n\n");
    printf("5. GPU-aware verification must ALSO check:\n");
    printf("   ✗ Warp-uniform control flow\n");
    printf("   ✗ Warp-uniform side effects\n");
    printf("   ✗ Bounded atomic contention\n");
    printf("   ✗ Coalesced memory patterns\n\n");

    return 0;
}
