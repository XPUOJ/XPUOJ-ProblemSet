#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {

constexpr int A_SCALE_BLOCK = 128;
constexpr int WARPS_PER_BLOCK = 8;
constexpr unsigned FULL_MASK = 0xffffffffu;

__device__ __forceinline__ int signed_low4(uint8_t packed) {
    int v = int(packed & 0x0F);
    return v >= 8 ? v - 16 : v;
}

__device__ __forceinline__ int signed_high4(uint8_t packed) {
    int v = int((packed >> 4) & 0x0F);
    return v >= 8 ? v - 16 : v;
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_MASK, v, offset);
    }
    return v;
}

__global__ void warp_dot_kernel(
    const __nv_fp8_e4m3* __restrict__ A,
    const float* __restrict__ A_scale,
    const uint8_t* __restrict__ B_packed,
    const float* __restrict__ B_scale,
    const int32_t* __restrict__ m_indices,
    __nv_bfloat16* __restrict__ D,
    int64_t M_total,
    int64_t K,
    int64_t N,
    int64_t num_groups,
    int64_t group_k
) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int64_t out_idx = int64_t(blockIdx.x) * WARPS_PER_BLOCK + warp_in_block;
    const int64_t total = M_total * N;
    if (out_idx >= total) {
        return;
    }

    const int64_t row = out_idx / N;
    const int64_t col = out_idx - row * N;
    const int group = m_indices[row];

    if (group < 0 || group >= num_groups) {
        if (lane == 0) {
            D[out_idx] = __float2bfloat16(0.0f);
        }
        return;
    }

    const int64_t a_k_blocks = (K + A_SCALE_BLOCK - 1) / A_SCALE_BLOCK;
    const int64_t b_k_blocks = (K + group_k - 1) / group_k;
    const int64_t k_packed = (K + 1) / 2;

    float acc = 0.0f;
    for (int64_t kb = 0; kb < K; kb += group_k) {
        const int64_t kend = min(kb + group_k, K);
        const float bs = B_scale[(int64_t(group) * N + col) * b_k_blocks + kb / group_k];
        const float as = A_scale[row * a_k_blocks + kb / A_SCALE_BLOCK];

        for (int64_t k = kb + lane * 2; k < kend; k += 64) {
            const uint8_t packed = B_packed[(int64_t(group) * N + col) * k_packed + (k >> 1)];
            const float a0 = static_cast<float>(A[row * K + k]) * as;
            acc += a0 * (float(signed_low4(packed)) * bs);
            if (k + 1 < kend) {
                const float a1 = static_cast<float>(A[row * K + k + 1]) * as;
                acc += a1 * (float(signed_high4(packed)) * bs);
            }
        }
    }

    acc = warp_sum(acc);
    if (lane == 0) {
        D[out_idx] = __float2bfloat16(acc);
    }
}

}  // namespace

extern "C" void run_kernel(
    const __nv_fp8_e4m3* A,
    const float* A_scale,
    const uint8_t* B_packed,
    const float* B_scale,
    const int32_t* m_indices,
    __nv_bfloat16* D,
    int64_t M_total,
    int64_t K,
    int64_t N,
    int64_t num_groups,
    int64_t group_k
) {
    dim3 block(32 * WARPS_PER_BLOCK);
    dim3 grid(static_cast<unsigned int>((M_total * N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK));
    warp_dot_kernel<<<grid, block>>>(
        A, A_scale, B_packed, B_scale, m_indices, D, M_total, K, N, num_groups, group_k
    );
}
