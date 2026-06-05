#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {

constexpr int A_SCALE_BLOCK = 128;

__device__ __forceinline__ float fp8_to_float(__nv_fp8_e4m3 x) {
    return static_cast<float>(x);
}

__device__ __forceinline__ int unpack_signed_int4(uint8_t packed, int k_mod_2) {
    int v = k_mod_2 == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
    return v >= 8 ? v - 16 : v;
}

__global__ void baseline_kernel(
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
    const int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M_total || col >= N) {
        return;
    }

    const int32_t group = m_indices[row];
    if (group < 0 || group >= num_groups) {
        D[row * N + col] = __float2bfloat16(0.0f);
        return;
    }

    const int64_t a_k_blocks = (K + A_SCALE_BLOCK - 1) / A_SCALE_BLOCK;
    const int64_t b_k_blocks = (K + group_k - 1) / group_k;
    const int64_t k_packed = (K + 1) / 2;

    float acc = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        const float a = fp8_to_float(A[row * K + k]) *
                        A_scale[row * a_k_blocks + k / A_SCALE_BLOCK];

        const uint8_t packed = B_packed[(int64_t(group) * N + col) * k_packed + k / 2];
        const int b_q = unpack_signed_int4(packed, int(k & 1));
        const float b = float(b_q) *
                        B_scale[(int64_t(group) * N + col) * b_k_blocks + k / group_k];

        acc += a * b;
    }

    D[row * N + col] = __float2bfloat16(acc);
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
    dim3 block(16, 16);
    dim3 grid(
        static_cast<unsigned int>((N + block.x - 1) / block.x),
        static_cast<unsigned int>((M_total + block.y - 1) / block.y)
    );
    baseline_kernel<<<grid, block>>>(
        A, A_scale, B_packed, B_scale, m_indices, D, M_total, K, N, num_groups, group_k
    );
}

