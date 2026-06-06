#include <stdint.h>
#include <cuda_fp16.h>

__global__ void add_kernel(__half* A, const __half* B, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        A[i] = __hadd(A[i], B[i]);
    }
}

extern "C" void run_kernel(__half* A, const __half* B, int64_t numel) {
    int threads = 256;
    int blocks = int((numel + threads - 1) / threads);
    if (blocks > 65535) {
        blocks = 65535;
    }
    add_kernel<<<blocks, threads>>>(A, B, numel);
}

