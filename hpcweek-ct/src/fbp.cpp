#include <cmath>
#include <algorithm>
#include <vector>
#include <omp.h>
#include <arm_neon.h>
#include <iostream>
#include <complex> // <-- 必需：用于 FFT

#include "fbp.h"

// 使用 constexpr double PI 以避免多次计算
constexpr double PI = 3.14159265358979323846;

// [!!! FIX !!!] 切换到 double 精度
using Complex = std::complex<double>;

// ------------------------------------------------------------
// STEP 1: Ramp Filter Functions (FFT 版本)
// ------------------------------------------------------------

// ramp_kernel 函数 (与原代码相同, 仍生成 float)
static std::vector<float> ramp_kernel(int len, float d = 1.0f) {
    if (len % 2 == 0) len += 1; 
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

// 辅助函数：计算大于等于 n 的下一个 2 的幂次
static int next_power_of_2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

/**
 * @brief (新) 迭代式、In-place Radix-2 FFT (DIT)
 * [!!! FIX !!!] 现在操作 std::complex<double>
 */
static void fft(std::vector<Complex>& data, int M, bool inverse) {
    if (M <= 1) return;

    // 1. Bit-Reversal Permutation
    int j = 0;
    for (int i = 1; i < M; ++i) {
        int bit = M >> 1;
        while (j >= bit) {
            j -= bit;
            bit >>= 1;
        }
        j += bit;
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }

    // 2. 蝶形运算
    for (int len = 2; len <= M; len <<= 1) {
        // [!!! FIX !!!] 全部使用 double 精度
        double ang = 2.0 * PI / (double)len * (inverse ? -1.0 : 1.0);
        Complex wlen(std::cos(ang), std::sin(ang)); // wlen 现在是 (double, double)

        for (int i = 0; i < M; i += len) {
            Complex w(1.0, 0.0); 
            for (int k = 0; k < len / 2; ++k) {
                Complex u = data[i + k];
                Complex v = data[i + k + len / 2] * w;
                
                data[i + k] = u + v;
                data[i + k + len / 2] = u - v;
                
                w *= wlen; 
            }
        }
    }

    // 3. (仅 iFFT) 缩放
    if (inverse) {
        double scale = 1.0 / (double)M;
        for (int i = 0; i < M; ++i) {
            data[i] *= scale;
        }
    }
}


/**
 * @brief (新) 滤波投影（FFT 版本）
 * [!!! FIX !!!] 执行 double <-> float 转换
 */
static void filter_projections_fft(float* sino, int n_angles, int n_det, int M,
                                   const std::vector<Complex>& kernel_freq) // kernel_freq 已是 double
{
    // [!!! FIX !!!] buffer 现在是 double
    std::vector<Complex> buffer(M);

    for (int a = 0; a < n_angles; ++a) {
        float* row = &sino[a * n_det];

        // 1. 复制数据 (float -> double) 并补零
        for (int i = 0; i < n_det; ++i) {
            buffer[i] = {(double)row[i], 0.0}; // 转换
        }
        std::fill(buffer.begin() + n_det, buffer.end(), Complex(0.0, 0.0));

        // 2. 正向 FFT (double)
        fft(buffer, M, false);

        // 3. 频率域乘法 (double)
        for (int i = 0; i < M; ++i) {
            buffer[i] *= kernel_freq[i];
        }

        // 4. 逆向 FFT (double)
        fft(buffer, M, true);

        // 5. 将结果 (double -> float) 复制回
        for (int i = 0; i < n_det; ++i) {
            row[i] = (float)buffer[i].real(); // 转换
        }
    }
}


// ------------------------------------------------------------
// STEP 2: Backprojection Function (反投影)
// (此部分为 "想法二" 的优化版本, 保持 float 和 NEON)
// ------------------------------------------------------------

void fbp_reconstruct_3d(
    float* sino_buffer,
    float* recon_buffer,
    int n_slices,
    int n_angles,
    int n_det,
    const std::vector<float>& angles_deg
) {
    size_t slice_size = size_t(n_angles) * n_det;
    size_t recon_size = size_t(n_det) * n_det;

    // ============================================================
    // STEP 0: Precomputation (预计算)
    // [!!! FIX !!!] 预计算 double 精度的 FFT 核
    // ============================================================

    // 1. 生成空间域核 (float)
    auto kernel_spatial = ramp_kernel(n_det | 1);
    int N_spatial = n_det;
    int K_spatial = kernel_spatial.size(); 
    int K = K_spatial / 2;

    // 2. 计算 FFT 所需的补零大小 (M)
    int M = next_power_of_2(N_spatial + K_spatial - 1); 

    // 3. [!!! FIX !!!] 创建 double 精度的补零核
    std::vector<Complex> kernel_freq(M, {0.0, 0.0});

    // 4. 复制核并执行 "FFT Shift" (float -> double)
    // (使用我们推导出的正确逻辑)
    
    // h_pad[0] = h_conv[0]
    kernel_freq[0] = {(double)kernel_spatial[K], 0.0};
    
    // h_pad[n] = h_conv[n] = kernel_spatial[K-n]
    for (int n = 1; n <= K; ++n) {
        kernel_freq[n] = {(double)kernel_spatial[K - n], 0.0};
    }
    
    // h_pad[M-n] = h_conv[-n] = kernel_spatial[K+n]
    for (int n = 1; n <= K; ++n) { 
        kernel_freq[M - n] = {(double)kernel_spatial[K + n], 0.0};
    }
    
    // 5. 预计算核的 FFT (double)
    fft(kernel_freq, M, false);


    // --- (以下反投影部分保持 float 和 NEON 不变) ---

    // 预计算所有角度的三角函数值
    std::vector<float> cos_theta(n_angles);
    std::vector<float> sin_theta(n_angles);
    const float deg2rad = float(PI) / 180.0f;

    #pragma omp parallel for schedule(static)
    for (int ai = 0; ai < n_angles; ++ai) {
        float th = angles_deg[ai] * deg2rad;
        cos_theta[ai] = std::cos(th);
        sin_theta[ai] = std::sin(th);
    }

    // 图像几何参数
    const float cx = (n_det - 1) * 0.5f;
    const float cy = (n_det - 1) * 0.5f;
    const float t_half = (n_det - 1) * 0.5f;
    const float scale = float(PI) / float(n_angles);

    // ============================================================
    // STEP 1: Filter all projections (Ramp 滤波)
    // (调用 double-precision FFT 版本)
    // ============================================================

    #pragma omp parallel for schedule(static)
    for (int slice_id = 0; slice_id < n_slices; ++slice_id) {
        float* sino_slice = sino_buffer + slice_id * slice_size;
        // 调用新的 double-precision FFT 滤波函数
        filter_projections_fft(sino_slice, n_angles, n_det, M, kernel_freq);
    }

    // ============================================================
    // STEP 2: Backprojection (反投影)
    // (使用 "想法二" 的缓存优化 + 精度修正版本)
    // ============================================================

    #pragma omp parallel for collapse(2) schedule(static)
    for (int slice_id = 0; slice_id < n_slices; ++slice_id) {
        for (int y = 0; y < n_det; ++y) {

            const float* sino_slice = sino_buffer + slice_id * slice_size;
            float* recon_slice = recon_buffer + slice_id * recon_size;
            float* recon_row = recon_slice + y * n_det;
            
            float yr = y - cy; 

            // 1. 将整行累加器清零 (float)
            int x_zero = 0;
            float32x4_t v_zero = vdupq_n_f32(0.0f);
            for (; x_zero <= n_det - 4; x_zero += 4) {
                vst1q_f32(&recon_row[x_zero], v_zero);
            }
            for (; x_zero < n_det; ++x_zero) {
                recon_row[x_zero] = 0.0f;
            }

            // 2. 将角度循环 (ai) 移到外层
            for (int ai = 0; ai < n_angles; ++ai) {
                
                const float* sino_row = sino_slice + ai * n_det;
                const float c = cos_theta[ai];
                const float s = sin_theta[ai];
                const float yr_s = yr * s; 

                // 3. 在内层遍历 x 轴 (NEON)
                float32x4_t v_c = vdupq_n_f32(c);
                float32x4_t v_yr_s = vdupq_n_f32(yr_s);
                float32x4_t v_t_half = vdupq_n_f32(t_half);
                float32x4_t v_cx = vdupq_n_f32(cx);
                float32x4_t v_idx_step = vdupq_n_f32(4.0f);
                float32x4_t v_x_idx = {0.0f, 1.0f, 2.0f, 3.0f}; 

                int x = 0;
                for (; x <= n_det - 4; x += 4) {
                    float32x4_t v_xr = vsubq_f32(v_x_idx, v_cx);
                    float32x4_t v_t = vmlaq_f32(v_yr_s, v_xr, v_c);
                    float32x4_t v_u = vaddq_f32(v_t, v_t_half);

                    // 标量 Gather (L1 缓存命中)
                    float interp_vals[4]; 
                    for (int k = 0; k < 4; ++k) {
                        float u = v_u[k];
                        int u0 = (int)u; 
                        float du = u - (float)u0;
                        int u1 = u0 + 1;

                        float v0 = 0.0f;
                        float v1 = 0.0f;
                        if (u0 >= 0 && u0 < n_det) v0 = sino_row[u0];
                        if (u1 >= 0 && u1 < n_det) v1 = sino_row[u1];
                        interp_vals[k] = (1.0f - du) * v0 + du * v1;
                    }
                    
                    // NEON 累加
                    float32x4_t v_interp = vld1q_f32(interp_vals);
                    float32x4_t v_acc_old = vld1q_f32(&recon_row[x]); 
                    v_acc_old = vaddq_f32(v_acc_old, v_interp);
                    vst1q_f32(&recon_row[x], v_acc_old); 

                    v_x_idx = vaddq_f32(v_x_idx, v_idx_step); 
                }

                // 标量收尾
                for (; x < n_det; ++x) {
                    float xr = x - cx;
                    float t = xr * c + yr_s;
                    float u = t + t_half;
                    
                    int u0 = (int)u;
                    float du = u - (float)u0;
                    int u1 = u0 + 1;

                    float v0 = 0.0f;
                    float v1 = 0.0f;
                    if (u0 >= 0 && u0 < n_det) v0 = sino_row[u0];
                    if (u1 >= 0 && u1 < n_det) v1 = sino_row[u1];
                    recon_row[x] += (1.0f - du) * v0 + du * v1;
                }
            } // end for(ai)

            // 4. 应用 scale 因子
            float32x4_t v_scale = vdupq_n_f32(scale);
            int x_scale = 0;
            for (; x_scale <= n_det - 4; x_scale += 4) {
                 float32x4_t v_val = vld1q_f32(&recon_row[x_scale]);
                 v_val = vmulq_f32(v_val, v_scale);
                 vst1q_f32(&recon_row[x_scale], v_val);
            }
            for (; x_scale < n_det; ++x_scale) {
                recon_row[x_scale] *= scale;
            }
        } // end for(y)
    } // end for(slice_id)
}
