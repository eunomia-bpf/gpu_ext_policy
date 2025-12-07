/**
 * gemm.cuh - GEMM kernel from PolyBench/GPU
 *
 * Source: PolyBench/GPU 1.0
 * Original file: CUDA/GEMM/gemm.cu
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Web: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * This is an adapted standalone version for UVM benchmark integration.
 * The kernel computes: C = alpha * A * B + beta * C
 */

#ifndef GEMM_CUH
#define GEMM_CUH

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cmath>

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

// ============================================================================
// Original PolyBench GEMM Kernel (adapted for standalone use)
// ============================================================================
// C = alpha * A * B + beta * C
// A: NI x NK, B: NK x NJ, C: NI x NJ

__global__ void gemm_kernel(int ni, int nj, int nk,
                            DATA_TYPE alpha, DATA_TYPE beta,
                            DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < ni) && (j < nj))
	{
		c[i * nj + j] *= beta;
		int k;
		for(k = 0; k < nk; k++)
		{
			c[i * nj + j] += alpha * a[i * nk + k] * b[k * nj + j];
		}
	}
}

// ============================================================================
// Wrapper function for UVM benchmark integration
// ============================================================================

inline void run_gemm(size_t total_working_set, const std::string& mode,
                     size_t stride_bytes, int iterations,
                     std::vector<float>& runtimes, KernelResult& result) {
    (void)stride_bytes;

    // 权重复用设计：一个大 buffer + 偏移访问
    // 模拟神经网络推理：每层权重 dim x hidden，多个 token 遍历所有层
    const int dim = 1024;
    const int hidden = 3072;
    size_t layer_size = (size_t)dim * hidden * sizeof(DATA_TYPE);  // ~12MB per layer

    // 根据 total_working_set 计算层数
    int num_layers = total_working_set / layer_size;
    if (num_layers < 1) num_layers = 1;

    size_t weights_size = (size_t)num_layers * layer_size;

    // 分配权重 buffer（主要内存）
    DATA_TYPE* weights;
    if (mode == "device") {
        CUDA_CHECK(cudaMalloc(&weights, weights_size));
        CUDA_CHECK(cudaMemset(weights, 0, weights_size));
    } else {
        CUDA_CHECK(cudaMallocManaged(&weights, weights_size));
        // CPU 初始化（确保页面在 CPU）
        size_t total_elements = (size_t)num_layers * dim * hidden;
        size_t report_interval = total_elements / 20;  // 5% increments
        if (report_interval == 0) report_interval = 1;
        int last_percent = -1;

        fprintf(stderr, "Initializing weights (%zu MB)...\n", weights_size / (1024*1024));
        for (size_t i = 0; i < total_elements; i++) {
            weights[i] = (DATA_TYPE)(i % 1000) / 1000.0f;
            if (i % report_interval == 0) {
                int percent = (int)(i * 100 / total_elements);
                if (percent != last_percent && percent % 5 == 0) {
                    fprintf(stderr, "  %d%% complete\r", percent);
                    fflush(stderr);
                    last_percent = percent;
                }
            }
        }
        fprintf(stderr, "  100%% complete\n");
    }

    // 分配 activation buffer（很小，不是内存压力来源）
    DATA_TYPE *x, *out;
    size_t x_size = dim * sizeof(DATA_TYPE);
    size_t out_size = hidden * sizeof(DATA_TYPE);

    if (mode == "device") {
        CUDA_CHECK(cudaMalloc(&x, x_size));
        CUDA_CHECK(cudaMalloc(&out, out_size));
        CUDA_CHECK(cudaMemset(x, 0, x_size));
        CUDA_CHECK(cudaMemset(out, 0, out_size));
    } else {
        CUDA_CHECK(cudaMallocManaged(&x, x_size));
        CUDA_CHECK(cudaMallocManaged(&out, out_size));
        for (int i = 0; i < dim; i++) {
            x[i] = (DATA_TYPE)i / dim;
        }
        for (int i = 0; i < hidden; i++) {
            out[i] = 0.0f;
        }
    }

    // Apply UVM hints
    if (mode != "device" && mode != "uvm") {
        int dev;
        CUDA_CHECK(cudaGetDevice(&dev));
        apply_uvm_hints(weights, weights_size, mode, dev);
        apply_uvm_hints(x, x_size, mode, dev);
        apply_uvm_hints(out, out_size, mode, dev);
        if (mode == "uvm_prefetch") {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // Launch configuration for GEMV: W[dim x hidden] @ x[dim] -> out[hidden]
    // 实际上是 (1 x dim) @ (dim x hidden) = (1 x hidden)
    dim3 block(256);
    dim3 grid((hidden + block.x - 1) / block.x);

    // 多个 token 遍历所有层
    int num_tokens = 10;

    auto launch = [&]() {
        for (int t = 0; t < num_tokens; t++) {
            for (int l = 0; l < num_layers; l++) {
                DATA_TYPE* W = weights + (size_t)l * dim * hidden;
                // GEMM: (1 x dim) @ (dim x hidden) = (1 x hidden)
                // ni=1, nj=hidden, nk=dim
                gemm_kernel<<<grid, block>>>(1, hidden, dim, 1.0f, 0.0f, x, W, out);
                // 不 sync，继续下一层
            }
        }
    };

    time_kernel(launch, /*warmup=*/2, iterations, runtimes, result);

    // bytes_accessed: 每个 token 访问所有层权重 + activation
    // 权重: num_layers * dim * hidden * sizeof(float)
    // activation: dim + hidden (很小，忽略不计)
    result.bytes_accessed = (size_t)num_tokens * num_layers * dim * hidden * sizeof(DATA_TYPE);

    // Cleanup
    CUDA_CHECK(cudaFree(weights));
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(out));
}

#endif // GEMM_CUH
