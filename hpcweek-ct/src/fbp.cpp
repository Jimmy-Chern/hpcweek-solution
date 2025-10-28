#include <cmath>        // 包含数学函数 (如 cos, sin, std::floor)
#include <algorithm>    // 包含算法函数 (如 max, min, copy)
#include <vector>       // 包含 std::vector 容器
#include <omp.h>        // 包含 OpenMP 库
#include <arm_neon.h>   // 包含 ARM NEON SIMD 指令集
#include <iostream>     // 包含输入/输出流

#include "fbp.h"        // 包含自定义头文件

// 使用 constexpr double PI 以避免多次计算
constexpr double PI = 3.14159265358979323846;

// ------------------------------------------------------------
// STEP 1: Ramp Filter Functions
// ------------------------------------------------------------

// 生成空间域的 Ramp 滤波器核（与原代码相同）
static std::vector<float> ramp_kernel(int len, float d = 1.0f) {
    if (len % 2 == 0) len += 1; // 确保长度为奇数以保证对称性
    int K = len / 2;

    std::vector<float> h(len, 0.0f);
    h[K] = 1.0f / (4.0f * d * d);

    for (int n = 1; n <= K; ++n) {
        if (n % 2 == 1) {
            float val = -1.0f / (float(PI) * float(PI) * n * n * d * d);
            h[K + n] = val;
            h[K - n] = val;
        }
    }
    return h;
}

/**
 * @brief 滤波投影（串行版本，优化了内存）
 * 优化 #2：移除了 OpenMP pragma 以避免嵌套并行。
 * 优化 #4：将 'tmp_row' 的分配移到角度循环之外。
 * 优化 #5：使用 NEON vld2q_f32 (跨步加载) 仅处理稀疏核的非零值 (k=0 和 k 为奇数)。
 */
static void filter_projections_serial(float* sino, int n_angles, int n_det, const std::vector<float>& kernel) {
    int K = int(kernel.size() / 2);
    int kernel_len = kernel.size();
    
    // 优化 #4：在循环外分配一次临时缓冲区
    std::vector<float> tmp_row(n_det);

    // 这是一个串行循环（由外层 OMP 循环的单个线程执行）
    for (int a = 0; a < n_angles; ++a) {
        float* row = &sino[a * n_det];
        
        // --- 1D Convolution with NEON (Strided Load) ---
        for (int x = 0; x < n_det; ++x) {
            float acc = 0.0f; // 标量累加器
            
            int k_start = std::max(-K, -x);
            int k_end = std::min(K, n_det - 1 - x);
            
            // NEON 向量化累加器 (用于奇数 k)
            float32x4_t v_acc_conv = vdupq_n_f32(0.0f);
            int k;

            // 1. Handle k=0 (center point) 单独处理中心点
            if (k_start <= 0 && 0 <= k_end) {
                acc += row[x] * kernel[K];
            }

            // 2. Handle positive odd k (k = 1, 3, 5, ...)
            // 找到第一个 k >= k_start 且 k >= 1 的奇数
            k = (k_start <= 0) ? 1 : (k_start % 2 != 0 ? k_start : k_start + 1);
            
            // 向量化循环：一次处理 4 个奇数 (k, k+2, k+4, k+6)
            for (; k <= k_end - 7; k += 8) { 
                // k 是奇数。vld2q_f32 从 &row[x+k] 开始加载
                // val[0] = {row[x+k], row[x+k+2], row[x+k+4], row[x+k+6]}
                // val[1] = {row[x+k+1], row[x+k+3], row[x+k+5], row[x+k+7]} (偶数偏移, 忽略)
                float32x4x2_t v_row_pair = vld2q_f32(&row[x + k]);
                float32x4x2_t v_kern_pair = vld2q_f32(&kernel[K + k]);

                // 仅累加 val[0] (奇数偏移) 的乘积
                v_acc_conv = vmlaq_f32(v_acc_conv, v_row_pair.val[0], v_kern_pair.val[0]);
            }
            // 标量收尾：处理剩余的正奇数 k
            for (; k <= k_end; k += 2) {
                acc += row[x + k] * kernel[K + k];
            }

            // 3. Handle negative odd k (k = -1, -3, -5, ...)
            // 找到第一个 k <= k_end 且 k <= -1 的奇数
            k = (k_end >= 0) ? -1 : (k_end % 2 != 0 ? k_end : k_end - 1);
            
            // 向量化循环：一次处理 4 个奇数 (k, k-2, k-4, k-6)
            for (; k >= k_start + 7; k -= 8) {
                // k 是奇数。vld2q_f32 从 &row[x+k-6] 开始加载
                // val[0] = {row[x+k-6], row[x+k-4], row[x+k-2], row[x+k]}
                float32x4x2_t v_row_pair = vld2q_f32(&row[x + k - 6]);
                float32x4x2_t v_kern_pair = vld2q_f32(&kernel[K + k - 6]);
                
                // 仅累加 val[0] (奇数偏移) 的乘积
                v_acc_conv = vmlaq_f32(v_acc_conv, v_row_pair.val[0], v_kern_pair.val[0]);
            }
            // 标量收尾：处理剩余的负奇数 k
            for (; k >= k_start; k -= 2) {
                 acc += row[x + k] * kernel[K + k];
            }
            
            // 4. 水平求和
            acc += vaddvq_f32(v_acc_conv); // 加上所有向量化奇数 k 的累加结果
            tmp_row[x] = acc; // 存入临时缓冲区
        }

        // 将滤波结果写回输入数组
        std::copy(tmp_row.begin(), tmp_row.end(), row);
    }
}

// ------------------------------------------------------------
// STEP 2: Backprojection Function (反投影)
// (此部分为完整代码，修复了因“...”导致的编译错误)
// ------------------------------------------------------------

void fbp_reconstruct_3d(
    float* sino_buffer,
    float* recon_buffer,
    int n_slices,
    int n_angles,
    int n_det,
    const std::vector<float>& angles_deg
) {
    size_t slice_size = size_t(n_angles) * n_det; // 单一切片 sinogram 大小
    size_t recon_size = size_t(n_det) * n_det; // 单一切片重建图像大小

    // ============================================================
    // STEP 0: Precomputation (预计算)
    // ============================================================
    auto kernel = ramp_kernel(n_det | 1); // 计算 Ramp 核

    // 预计算所有角度的三角函数值
    std::vector<float> cos_theta(n_angles);
    std::vector<float> sin_theta(n_angles);
    const float deg2rad = float(PI) / 180.0f;

    // 此处预计算也可以并行化
    #pragma omp parallel for 
    for (int ai = 0; ai < n_angles; ++ai) {
        float th = angles_deg[ai] * deg2rad;
        cos_theta[ai] = std::cos(th); 
        sin_theta[ai] = std::sin(th);
    }

    // 图像几何参数
    const float cx = (n_det - 1) * 0.5f; // 图像中心 X
    const float cy = (n_det - 1) * 0.5f; // 图像中心 Y
    const float t_half = (n_det - 1) * 0.5f; // 探测器偏移量
    const float scale = float(PI) / float(n_angles); // 归一化因子

    // ============================================================
    // STEP 1: Filter all projections (Ramp 滤波)
    // ============================================================

    // 优化 #2：仅并行化最外层的切片循环
    #pragma omp parallel for schedule(dynamic)
    for (int slice_id = 0; slice_id < n_slices; ++slice_id) {
        float* sino_slice = sino_buffer + slice_id * slice_size;
        // 调用优化的串行滤波版本 (现在使用 vld2q_f32)
        filter_projections_serial(sino_slice, n_angles, n_det, kernel); 
    }

    // ============================================================
    // STEP 2: Backprojection (反投影)
    // ============================================================
    
    // 使用 collapse(2) 将两层循环合并，增加并行粒度
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int slice_id = 0; slice_id < n_slices; ++slice_id) {
        for (int y = 0; y < n_det; ++y) {
            
            // --- 修复：恢复了这些指针的定义 ---
            const float* sino_slice = sino_buffer + slice_id * slice_size;
            float* recon_slice = recon_buffer + slice_id * recon_size;

            float yr = y - cy; // Y 坐标相对于图像中心
            float* recon_row = recon_slice + y * n_det;

            // 遍历 X 轴
            for (int x = 0; x < n_det; ++x) {
                float xr = x - cx; // X 坐标相对于图像中心
                float acc = 0.0f; // 最终的标量累加器

                // 优化 #3：初始化 NEON 累加器
                float32x4_t v_acc = vdupq_n_f32(0.0f); 

                // --- Angle Loop (NEON 优化) ---
                int ai = 0;
                for (; ai <= n_angles - 4; ai += 4) {
                    // --- 修复：恢复了这些向量的定义 ---
                    // 加载 4 个 cos 和 sin 值
                    float32x4_t v_c = vld1q_f32(&cos_theta[ai]);
                    float32x4_t v_s = vld1q_f32(&sin_theta[ai]);
                    
                    // 将 xr, yr, t_half 广播到 4 元素向量
                    float32x4_t v_xr = vdupq_n_f32(xr);
                    float32x4_t v_yr = vdupq_n_f32(yr);
                    float32x4_t v_t_half = vdupq_n_f32(t_half);
                    
                    // 1. (NEON) 计算 4 个投影距离 t = xr*c + yr*s
                    float32x4_t v_t = vmlaq_f32(vmulq_f32(v_xr, v_c), v_yr, v_s); 
                    
                    // 2. (NEON) 转换为探测器坐标 u = t + t_half
                    float32x4_t v_u = vaddq_f32(v_t, v_t_half);
                    
                    // 优化 #3：标量插值，但准备用于 NEON 累加
                    float interp_vals[4]; // 临时数组存储 4 个插值结果

                    // 标量回退循环 (k=0 到 3)
                    for (int k = 0; k < 4; ++k) {
                        // --- 修复：恢复了这些变量的定义 ---
                        float u = v_u[k]; // 获取第 k 个 u 值
                        
                        // 恢复：根据您的反馈，std::floor 导致 MSE 错误，换回 (int)
                        int u0 = (int)u;
                        float du = u - (float)u0;
                        int u1 = u0 + 1;
                        
                        // 获取对应的 sinogram 行
                        const float* sino_row = sino_slice + (ai + k) * n_det;
                        
                        // 边界检查和线性插值
                        float v0 = 0.0f;
                        float v1 = 0.0f;
                        if (u0 >= 0 && u0 < n_det) v0 = sino_row[u0];
                        if (u1 >= 0 && u1 < n_det) v1 = sino_row[u1];
                        
                        // 计算插值结果
                        interp_vals[k] = (1.0f - du) * v0 + du * v1;
                    }

                    // (接上文) 优化 #3：将 4 个标量插值结果加载回 NEON 并进行向量累加
                    float32x4_t v_interp = vld1q_f32(interp_vals);
                    v_acc = vaddq_f32(v_acc, v_interp); // v_acc += v_interp
                }
                
                // 优化 #3：水平求和 NEON 累加器的结果
                acc = vaddvq_f32(v_acc);
                
                // --- 标量余数循环 ---
                // (处理末尾 0-3 个剩余的角度)
                for (; ai < n_angles; ++ai) {
                    // --- 修复：恢复了这些变量的定义 ---
                    float c = cos_theta[ai];
                    float s = sin_theta[ai];
                    float t = xr * c + yr * s;
                    float u = t + t_half;

                    // 恢复：根据您的反馈，std::floor 导致 MSE 错误，换回 (int)
                    int u0 = (int)u;
                    float du = u - (float)u0;
                    int u1 = u0 + 1;

                    const float* sino_row = sino_slice + ai * n_det;
                    float v0 = 0.0f;
                    float v1 = 0.0f;
                    if (u0 >= 0 && u0 < n_det) v0 = sino_row[u0];
                    if (u1 >= 0 && u1 < n_det) v1 = sino_row[u1];
                    
                    // 累加到主标量累加器
                    acc += (1.0f - du) * v0 + du * v1;
                }

                // 应用归一化因子并存储最终像素值
                recon_row[x] = acc * scale;
            }
        }
    }
}


