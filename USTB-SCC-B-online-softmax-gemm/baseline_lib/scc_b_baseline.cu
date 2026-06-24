#include <stdint.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

static constexpr int TILE_M = 32;
static constexpr int TILE_N = 32;
static constexpr int MAX_D = 128;

__global__ void fused_online_softmax_gemm_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ mask,
    int64_t B,
    int64_t H,
    int64_t S,
    int64_t D,
    float alpha,
    __nv_bfloat16* __restrict__ O
) {
    int tid = threadIdx.x;
    int block_bh = blockIdx.x;
    int block_m = blockIdx.y;

    int b = block_bh / static_cast<int>(H);
    int h = block_bh % static_cast<int>(H);

    int q_row_start = block_m * TILE_M + tid;
    bool valid_row = (q_row_start < S);

    const __nv_bfloat16* Q_base = Q + b * (H * S * D) + h * (S * D);
    const __nv_bfloat16* K_base = K + b * (H * S * D) + h * (S * D);
    const __nv_bfloat16* V_base = V + b * (H * S * D) + h * (S * D);
    __nv_bfloat16* O_base = O + b * (H * S * D) + h * (S * D);

    int64_t mask_stride_b = 0;
    const __nv_bfloat16* mask_base = nullptr;
    bool has_mask = false;
    if (mask != nullptr) {
        has_mask = true;
        mask_stride_b = S * S;
        mask_base = mask + b * mask_stride_b;
    }

    float q_row[MAX_D];
#pragma unroll
    for (int d = 0; d < D; d++) {
        if (valid_row) {
            q_row[d] = __bfloat162float(Q_base[q_row_start * D + d]) * alpha;
        } else {
            q_row[d] = 0.0f;
        }
    }

    float m = -1e30f;
    float l = 0.0f;
    float o_acc[MAX_D];
#pragma unroll
    for (int d = 0; d < D; d++) {
        o_acc[d] = 0.0f;
    }

    __shared__ float K_tile[TILE_N][MAX_D];
    __shared__ float V_tile[TILE_N][MAX_D];

    for (int kv_start = 0; kv_start < S; kv_start += TILE_N) {
        for (int base_d = 0; base_d < D; base_d += TILE_M) {
            int d = base_d + tid;

            if (d < D) {
                for (int r = 0; r < TILE_N; r++) {
                    int kv_idx = kv_start + r;
                    float kval = 0.0f;
                    float vval = 0.0f;
                    if (kv_idx < S) {
                        kval = __bfloat162float(K_base[kv_idx * D + d]);
                        vval = __bfloat162float(V_base[kv_idx * D + d]);
                    }
                    K_tile[r][d] = kval;
                    V_tile[r][d] = vval;
                }
            }
        }
        __syncthreads();

        float s_tile[TILE_N];
#pragma unroll
        for (int j = 0; j < TILE_N; j++) {
            s_tile[j] = 0.0f;
        }

        if (valid_row) {
#pragma unroll
            for (int j = 0; j < TILE_N; j++) {
                float dot = 0.0f;
#pragma unroll
                for (int d = 0; d < D; d++) {
                    dot += q_row[d] * K_tile[j][d];
                }
                s_tile[j] = dot;
            }
        }

        if (has_mask) {
#pragma unroll
            for (int j = 0; j < TILE_N; j++) {
                int kv_col = kv_start + j;
                if (kv_col < S) {
                    float mval = __bfloat162float(mask_base[q_row_start * S + kv_col]);
                    s_tile[j] += mval;
                } else {
                    s_tile[j] = -1e30f;
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < TILE_N; j++) {
                int kv_col = kv_start + j;
                if (kv_col >= S) {
                    s_tile[j] = -1e30f;
                }
            }
        }

        if (valid_row) {
            float tile_max = -1e30f;
#pragma unroll
            for (int j = 0; j < TILE_N; j++) {
                if (s_tile[j] > tile_max) {
                    tile_max = s_tile[j];
                }
            }

            bool tile_valid = (tile_max > -1e29f);
            if (tile_valid) {
                float m_old = m;
                float m_new = (m_old > tile_max) ? m_old : tile_max;

                float scale = expf(m_old - m_new);
                l *= scale;
#pragma unroll
                for (int d = 0; d < D; d++) {
                    o_acc[d] *= scale;
                }

                float p_sum = 0.0f;
                float p_tile[TILE_N];
#pragma unroll
                for (int j = 0; j < TILE_N; j++) {
                    int kv_col = kv_start + j;
                    if (kv_col < S) {
                        float p = expf(s_tile[j] - m_new);
                        if (s_tile[j] < -1e29f) {
                            p = 0.0f;
                        }
                        p_tile[j] = p;
                        p_sum += p;
                    } else {
                        p_tile[j] = 0.0f;
                    }
                }
                l += p_sum;

#pragma unroll
                for (int j = 0; j < TILE_N; j++) {
                    float p = p_tile[j];
                    if (p > 0.0f) {
#pragma unroll
                        for (int d = 0; d < D; d++) {
                            o_acc[d] += p * V_tile[j][d];
                        }
                    }
                }

                m = m_new;
            }
        }

        __syncthreads();
    }

    if (valid_row) {
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
#pragma unroll
        for (int d = 0; d < D; d++) {
            O_base[q_row_start * D + d] = __float2bfloat16(o_acc[d] * inv_l);
        }
    }
}

extern "C" void run_kernel(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    const __nv_bfloat16* mask,
    int64_t B,
    int64_t H,
    int64_t S,
    int64_t D,
    float alpha,
    __nv_bfloat16* O
) {
    dim3 grid(B * H, (S + TILE_M - 1) / TILE_M);
    dim3 block(TILE_M);

    fused_online_softmax_gemm_kernel<<<grid, block>>>(
        Q, K, V, mask, B, H, S, D, alpha, O
    );
}
