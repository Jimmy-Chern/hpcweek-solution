#include <vector>
#include <cstdlib>
#include <cstring> // For memset
#include <ctime>
#include <omp.h>
#include <algorithm>
#include <arm_neon.h> 

/**
 * @brief 稀疏矩阵-稠密矩阵乘法 (SPMM) 优化版: Vout = A * Vin
 * * 优化点: 
 * 1. OpenMP 行并行 (M 维度) - schedule(dynamic)。
 * 2. I 循环 (非零元) 4x 展开: 保持寄存器平衡和良好的 V_in 局部性。
 * 3. J 维度 (内循环) 保持 2x float32x4_t (8 float) NEON FMA。
 * 4. [新增] I 循环中添加显式预取，以隐藏下一组 V_in 的不规则访存延迟。
 */
void spmm_cpu_opt( 
    const int* __restrict__ ptr, 
    const int* __restrict__ idx, 
    const float* __restrict__ val, 
    const float* __restrict__ vin, 
    float* __restrict__ vout, 
    const int num_v, 
    const int INFEATURE, 
    const int k)
{
    const int stride = 8;
    const int vectorized_limit = INFEATURE / stride * stride; 

    #pragma omp parallel for schedule(dynamic)
    for (int m = 0; m < num_v; ++m) 
    { 
        int begin = ptr[m], end = ptr[m + 1];
        float* __restrict__ vout_row = vout + m * INFEATURE;

        memset(vout_row, 0, INFEATURE * sizeof(float));

        int i = begin;
        
        // [优化] 找到 4x 展开的安全边界
        int end_unrolled = begin + (end - begin) / 4 * 4;

        for (i = begin; i < end_unrolled; i += 4) {
            
            // =========================================================
            // [新增优化] 显式数据预取下一组 I 迭代所需的 V_in 行
            // =========================================================
            if (i + 4 < end) {
                // 预取下一组 4 个 V_in 行 (i+4, i+5, i+6, i+7) 的起始地址
                // 预取到最高级别缓存 (3)
                __builtin_prefetch(vin + idx[i + 4] * INFEATURE, 0, 3);
                
                // 预取 i+5/i+6，以防它们距离太远
                if (i + 5 < end) __builtin_prefetch(vin + idx[i + 5] * INFEATURE, 0, 3);
                if (i + 6 < end) __builtin_prefetch(vin + idx[i + 6] * INFEATURE, 0, 3);
                // 预取指令的开销很小，但能有效隐藏内存延迟
            }
            // =========================================================

            // --- 迭代 i ---
            const float val_mi_0 = val[i];
            const int n_0 = idx[i];
            const float* __restrict__ vin_row_0 = vin + n_0 * INFEATURE;
            const float32x4_t val_vec_0 = vmovq_n_f32(val_mi_0);

            // --- 迭代 i+1 ---
            const float val_mi_1 = val[i+1];
            const int n_1 = idx[i+1];
            const float* __restrict__ vin_row_1 = vin + n_1 * INFEATURE;
            const float32x4_t val_vec_1 = vmovq_n_f32(val_mi_1);

            // --- 迭代 i+2 ---
            const float val_mi_2 = val[i+2];
            const int n_2 = idx[i+2];
            const float* __restrict__ vin_row_2 = vin + n_2 * INFEATURE;
            const float32x4_t val_vec_2 = vmovq_n_f32(val_mi_2);
            
            // --- 迭代 i+3 ---
            const float val_mi_3 = val[i+3];
            const int n_3 = idx[i+3];
            const float* __restrict__ vin_row_3 = vin + n_3 * INFEATURE;
            const float32x4_t val_vec_3 = vmovq_n_f32(val_mi_3);

            int j = 0;
            
            // --- NEON 主体循环 (J 维度, 步长 8) ---
            for (j = 0; j < vectorized_limit; j += stride) {
                // 1. 加载 2 组 vout 寄存器
                float32x4_t vout_vec0 = vld1q_f32(vout_row + j + 0);
                float32x4_t vout_vec1 = vld1q_f32(vout_row + j + 4);

                // --- 处理迭代 i ---
                float32x4_t vin_vec0_0 = vld1q_f32(vin_row_0 + j + 0);
                float32x4_t vin_vec1_0 = vld1q_f32(vin_row_0 + j + 4);
                vout_vec0 = vfmaq_f32(vout_vec0, vin_vec0_0, val_vec_0);
                vout_vec1 = vfmaq_f32(vout_vec1, vin_vec1_0, val_vec_0);

                // --- 处理迭代 i+1 ---
                float32x4_t vin_vec0_1 = vld1q_f32(vin_row_1 + j + 0);
                float32x4_t vin_vec1_1 = vld1q_f32(vin_row_1 + j + 4);
                vout_vec0 = vfmaq_f32(vout_vec0, vin_vec0_1, val_vec_1);
                vout_vec1 = vfmaq_f32(vout_vec1, vin_vec1_1, val_vec_1);
                
                // --- 处理迭代 i+2 ---
                float32x4_t vin_vec0_2 = vld1q_f32(vin_row_2 + j + 0);
                float32x4_t vin_vec1_2 = vld1q_f32(vin_row_2 + j + 4);
                vout_vec0 = vfmaq_f32(vout_vec0, vin_vec0_2, val_vec_2);
                vout_vec1 = vfmaq_f32(vout_vec1, vin_vec1_2, val_vec_2);

                // --- 处理迭代 i+3 ---
                float32x4_t vin_vec0_3 = vld1q_f32(vin_row_3 + j + 0);
                float32x4_t vin_vec1_3 = vld1q_f32(vin_row_3 + j + 4);
                vout_vec0 = vfmaq_f32(vout_vec0, vin_vec0_3, val_vec_3);
                vout_vec1 = vfmaq_f32(vout_vec1, vin_vec1_3, val_vec_3);

                // 3. 写回 2 组结果
                vst1q_f32(vout_row + j + 0, vout_vec0);
                vst1q_f32(vout_row + j + 4, vout_vec1);
            }
            
            // --- 尾部处理 (Epilogue) for J loop ---
            for (; j < INFEATURE; ++j) {
                // 合并四次更新
                vout_row[j] += vin_row_0[j] * val_mi_0 + 
                               vin_row_1[j] * val_mi_1 +
                               vin_row_2[j] * val_mi_2 +
                               vin_row_3[j] * val_mi_3;
            }
        } // end of unrolled 4x i loop

        // 3. I 循环尾部处理 (处理剩余 0, 1, 2, 或 3 个非零元)
        // ... (保持不变，使用 2x 和 1x 展开处理尾部)
        
        // 处理剩余的 2 或 3 个
        if (i + 1 < end) { 
            // --- 迭代 i ---
            const float val_mi_0 = val[i];
            const int n_0 = idx[i];
            const float* __restrict__ vin_row_0 = vin + n_0 * INFEATURE;
            const float32x4_t val_vec_0 = vmovq_n_f32(val_mi_0);

            // --- 迭代 i+1 ---
            const float val_mi_1 = val[i+1];
            const int n_1 = idx[i+1];
            const float* __restrict__ vin_row_1 = vin + n_1 * INFEATURE;
            const float32x4_t val_vec_1 = vmovq_n_f32(val_mi_1);

            int j = 0;
            for (j = 0; j < vectorized_limit; j += stride) {
                float32x4_t vout_vec0 = vld1q_f32(vout_row + j + 0);
                float32x4_t vout_vec1 = vld1q_f32(vout_row + j + 4);
                
                float32x4_t vin_vec0_0 = vld1q_f32(vin_row_0 + j + 0);
                float32x4_t vin_vec1_0 = vld1q_f32(vin_row_0 + j + 4);
                vout_vec0 = vfmaq_f32(vout_vec0, vin_vec0_0, val_vec_0);
                vout_vec1 = vfmaq_f32(vout_vec1, vin_vec1_0, val_vec_0);
                
                float32x4_t vin_vec0_1 = vld1q_f32(vin_row_1 + j + 0);
                float32x4_t vin_vec1_1 = vld1q_f32(vin_row_1 + j + 4);
                vout_vec0 = vfmaq_f32(vout_vec0, vin_vec0_1, val_vec_1);
                vout_vec1 = vfmaq_f32(vout_vec1, vin_vec1_1, val_vec_1);
                
                vst1q_f32(vout_row + j + 0, vout_vec0);
                vst1q_f32(vout_row + j + 4, vout_vec1);
            }
            for (; j < INFEATURE; ++j) {
                vout_row[j] += vin_row_0[j] * val_mi_0 + vin_row_1[j] * val_mi_1;
            }
            i += 2;
        }

        // 处理最后剩下的 1 个
        if (i < end) { 
            float val_mi = val[i];
            int n = idx[i];
            const float* __restrict__ vin_row = vin + n * INFEATURE;
            float32x4_t val_vec = vmovq_n_f32(val_mi); 

            int j = 0;
            for (j = 0; j < vectorized_limit; j += stride) {
                float32x4_t vout_vec0 = vld1q_f32(vout_row + j + 0);
                float32x4_t vout_vec1 = vld1q_f32(vout_row + j + 4);
                
                float32x4_t vin_vec0 = vld1q_f32(vin_row + j + 0);
                float32x4_t vin_vec1 = vld1q_f32(vin_row + j + 4);
                
                vout_vec0 = vfmaq_f32(vout_vec0, vin_vec0, val_vec);
                vout_vec1 = vfmaq_f32(vout_vec1, vin_vec1, val_vec);
                
                vst1q_f32(vout_row + j + 0, vout_vec0);
                vst1q_f32(vout_row + j + 4, vout_vec1);
            }
            for (; j < INFEATURE; ++j) {
                vout_row[j] += vin_row[j] * val_mi;
            }
        }
    }
}
