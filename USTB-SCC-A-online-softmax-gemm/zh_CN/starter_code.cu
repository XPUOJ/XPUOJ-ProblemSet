/**
 * start_code.cu — Online Softmax + GEMM 融合 CUDA Baseline
 * ==========================================================
 * 实现: O = softmax(α · Q @ K^T + mask) @ V
 *
 * 算法: Online Softmax, 共享内存 tiled GEMM (FMA, 非 Tensor Core)
 *
 * Grid:  (B * H, ceil(S / TILE_M))
 * Block: TILE_M 线程, 每个线程负责 Q 的一行 + O 的一行
 *
 * Tile 参数:
 *   TILE_M = 32  — Q 的行 tile (每个 block 处理 32 行)
 *   TILE_N = 32  — K/V 的内循环 tile
 *
 * 正确性要求:
 *   torch.allclose(rtol=1e-2, atol=1e-2), bf16→float32 后比较
 *   中间矩阵 S 和 P 不得完整写入 HBM
 */

#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------
static constexpr int TILE_M = 32;   // Q 块行数 (= blockDim.x)
static constexpr int TILE_N = 32;   // K/V 块行数 (内循环步长)
static constexpr int MAX_D  = 128;  // 最大支持的 head 维度

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__global__ void fused_online_softmax_gemm_kernel(
    const __nv_bfloat16* __restrict__ Q,   // [B, H, S, D]
    const __nv_bfloat16* __restrict__ K,   // [B, H, S, D]
    const __nv_bfloat16* __restrict__ V,   // [B, H, S, D]
    const __nv_bfloat16* __restrict__ mask,// [B, 1, S, S] or nullptr
    int64_t B,
    int64_t H,
    int64_t S,
    int64_t D,
    float   alpha,
    __nv_bfloat16* __restrict__ O          // [B, H, S, D]
) {
    // ── 线程/block 索引 ──
    int tid      = threadIdx.x;                    // 0 .. TILE_M-1
    int block_bh = blockIdx.x;                     // 0 .. B*H - 1
    int block_m  = blockIdx.y;                     // 0 .. ceil(S / TILE_M) - 1

    int b = block_bh / (int)H;
    int h = block_bh % (int)H;

    // 当前 block 负责的 Q 行范围
    int q_row_start = block_m * TILE_M + tid;      // 当前线程负责的 Q 行 (全局索引)
    bool valid_row  = (q_row_start < S);

    // ── 基地址 ──
    const __nv_bfloat16* Q_base = Q + b * (H * S * D) + h * (S * D);
    const __nv_bfloat16* K_base = K + b * (H * S * D) + h * (S * D);
    const __nv_bfloat16* V_base = V + b * (H * S * D) + h * (S * D);
    __nv_bfloat16*       O_base = O + b * (H * S * D) + h * (S * D);

    int64_t mask_stride_b = 0;
    const __nv_bfloat16* mask_base = nullptr;
    bool has_mask = false;
    if (mask != nullptr) {
        has_mask = true;
        mask_stride_b = S * S;                      // mask shape: [B, 1, S, S]
        mask_base = mask + b * mask_stride_b;       // mask[b, 0, :, :]
    }

    // ── 寄存器: 加载当前 Q 行 ──
    float q_row[MAX_D];   // Q 的一行, float32
    #pragma unroll
    for (int d = 0; d < D; d++) {
        if (valid_row) {
            q_row[d] = __bfloat162float(Q_base[q_row_start * D + d]) * alpha;
        } else {
            q_row[d] = 0.0f;
        }
    }

    // ── Online softmax 状态 (float32) ──
    float m      = -1e30f;
    float l      = 0.0f;
    float o_acc[MAX_D];
    #pragma unroll
    for (int d = 0; d < D; d++) {
        o_acc[d] = 0.0f;
    }

    // ── 共享内存: K tile 和 V tile ──
    __shared__ float K_tile[TILE_N][MAX_D];
    __shared__ float V_tile[TILE_N][MAX_D];

    // ── 主循环: 遍历 K/V 的 TILE_N 块 ──
    for (int kv_start = 0; kv_start < S; kv_start += TILE_N) {

        // --- 协作加载 K tile 和 V tile 到共享内存 ---
        // 每个线程加载 D 维中若干列 (stride = TILE_M)
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

        // --- 当前线程: 计算 S_row_tile = Q_row @ K_tile^T ---
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

        // --- 应用 mask ---
        if (has_mask) {
            #pragma unroll
            for (int j = 0; j < TILE_N; j++) {
                int kv_col = kv_start + j;
                if (kv_col < S) {
                    float mval = __bfloat162float(mask_base[q_row_start * S + kv_col]);
                    s_tile[j] += mval;
                } else {
                    s_tile[j] = -1e30f;  // 越界 → -inf
                }
            }
        } else {
            // 无 mask: 越界位置置 -inf
            #pragma unroll
            for (int j = 0; j < TILE_N; j++) {
                int kv_col = kv_start + j;
                if (kv_col >= S) {
                    s_tile[j] = -1e30f;
                }
            }
        }

        // --- Online Softmax 更新 ---
        if (valid_row) {
            // Step 1: 找当前 tile 的最大值
            float tile_max = -1e30f;
            #pragma unroll
            for (int j = 0; j < TILE_N; j++) {
                if (s_tile[j] > tile_max) tile_max = s_tile[j];
            }

            // 如果整个 tile 都是 -inf (mask/越界), 跳过
            bool tile_valid = (tile_max > -1e29f);

            if (tile_valid) {
                // Step 2: m_new = max(m_old, tile_max)
                float m_old = m;
                float m_new = (m_old > tile_max) ? m_old : tile_max;

                // Step 3: 旧结果贬值
                float scale = expf(m_old - m_new);
                l *= scale;
                #pragma unroll
                for (int d = 0; d < D; d++) {
                    o_acc[d] *= scale;
                }

                // Step 4: 加入当前 tile 贡献
                float p_sum = 0.0f;
                float p_tile[TILE_N];
                #pragma unroll
                for (int j = 0; j < TILE_N; j++) {
                    int kv_col = kv_start + j;
                    if (kv_col < S) {
                        float p = expf(s_tile[j] - m_new);
                        // 对应 mask 值为 -inf 时, exp(-inf) = 0
                        if (s_tile[j] < -1e29f) p = 0.0f;
                        p_tile[j] = p;
                        p_sum += p;
                    } else {
                        p_tile[j] = 0.0f;
                    }
                }
                l += p_sum;

                // Step 5: O += P_tile @ V_tile
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

    // ── 最终归一化: O_row = O_acc / l ──
    if (valid_row) {
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        #pragma unroll
        for (int d = 0; d < D; d++) {
            O_base[q_row_start * D + d] = __float2bfloat16(o_acc[d] * inv_l);
        }
    }
}

// ---------------------------------------------------------------------------
// 对外接口
// ---------------------------------------------------------------------------
extern "C" void run_kernel(
    const __nv_bfloat16* Q,     // [B, H, S, D]
    const __nv_bfloat16* K,     // [B, H, S, D]
    const __nv_bfloat16* V,     // [B, H, S, D]
    const __nv_bfloat16* mask,  // [B, 1, S, S] or nullptr
    int64_t B,
    int64_t H,
    int64_t S,
    int64_t D,
    float   alpha,
    __nv_bfloat16* O            // [B, H, S, D]
) {
    dim3 grid(B * H, (S + TILE_M - 1) / TILE_M);
    dim3 block(TILE_M);

    fused_online_softmax_gemm_kernel<<<grid, block>>>(
        Q, K, V, mask, B, H, S, D, alpha, O
    );
}
