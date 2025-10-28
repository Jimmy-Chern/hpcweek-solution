#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>     // Include string.h for memcpy
#include <stddef.h>     // Include stddef.h for size_t
#include <stdint.h>     // Include stdint.h for int32_t etc.
#include <riscv_vector.h> // **RVV Intrinsics Header**

#include "qmatmul.h"

// For compatibility, redefine macros if they are not in qmatmul.h
#ifndef UNUSED
#define UNUSED(x) (void)(x)
#endif

#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif

// ------------------------------------------------------------------------------------------------
// Function 1: ggml_compute_forward_mul_mat_one_chunk (恢复到原始代码，但移除不必要的K分块逻辑)
// ------------------------------------------------------------------------------------------------
void ggml_compute_forward_mul_mat_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dest,
    const enum ggml_type compute_type,
    const int64_t num_rows_per_vec_dot,
    const int64_t A_row_start,
    const int64_t A_row_end,
    const int64_t B_col_start,
    const int64_t B_col_end
) {
    // calculate tensor C = A * B
    const struct ggml_tensor * C = dest;
    const struct ggml_tensor * A = C->src[0];
    const struct ggml_tensor * B = C->src[1];

    // Get tensor shape (number of elements per dimension)
    const int A_shape[4] = {A->ne[0], A->ne[1], A->ne[2], A->ne[3]};
    const int B_shape[4] = {B->ne[0], B->ne[1], B->ne[2], B->ne[3]};
    const int C_shape[4] = {C->ne[0], C->ne[1], C->ne[2], C->ne[3]};
    // Get tensor strides in bytes per dimension
    const size_t A_bstride[4] = {A->nb[0], A->nb[1], A->nb[2], A->nb[3]};
    const size_t B_bstride[4] = {B->nb[0], B->nb[1], B->nb[2], B->nb[3]};
    const size_t C_bstride[4] = {C->nb[0], C->nb[1], C->nb[2], C->nb[3]};

    // Some unused parameters
    UNUSED(num_rows_per_vec_dot);
    UNUSED(C_shape);

    ggml_vec_dot_t const vec_dot = type_traits_cpu[compute_type].vec_dot;
    enum ggml_type const vec_dot_type = type_traits_cpu[compute_type].vec_dot_type;

    // compute broadcast factors (handle potential zero dims)
    const int broadcast_factor_dim2 = (A_shape[2] == 0 || B_shape[2] == 0 || A_shape[2] == 0) ? 1 : B_shape[2] / A_shape[2];
    const int broadcast_factor_dim3 = (A_shape[3] == 0 || B_shape[3] == 0 || A_shape[3] == 0) ? 1 : B_shape[3] / A_shape[3];


    // If no work, return
    if (A_row_start >= A_row_end || B_col_start >= B_col_end){
        return;
    }

    // Get pointer to B data (potentially converted)
    const void * B_data = (B->type == vec_dot_type) ? B->data : params->wdata;
    // Calculate stride between rows/columns of B in the potentially converted layout
    const size_t B_data_bstride = (B_shape[0] == 0) ? 0 : ggml_row_size(vec_dot_type, B_shape[0]);


    // ** 使用 BLOCK_SIZE = 64 (原始代码中的分块大小) **
    const int BLOCK_SIZE = 64;

    // ** Temp Buffer Size = 64 **
    float temp[BLOCK_SIZE]; 

    // Main calculation loops (M/N 维度分块)
    for (int j = B_col_start; j < B_col_end; j += BLOCK_SIZE){ // Iterate C column blocks (N-dim)
        for (int i = A_row_start; i < A_row_end; i += BLOCK_SIZE){ // Iterate C row blocks (M-dim)
            
            // Iterate C columns within the block
            for (int jj = j; jj < j + BLOCK_SIZE && jj < B_col_end; jj ++){

                // Calculate indices
                int A_indices[4] = {0}, B_indices[4] = {0}, C_indices[4] = {0};

                // Deconstruct jj index based on B's shape
                if (B_shape[1] > 0 && B_shape[2] > 0) {
                   B_indices[3] = jj / (B_shape[2] * B_shape[1]);
                   B_indices[2] = (jj % (B_shape[2] * B_shape[1])) / B_shape[1];
                   B_indices[1] = jj % B_shape[1];
                } else if (B_shape[1] > 0) {
                   B_indices[3] = 0; B_indices[2] = 0;
                   B_indices[1] = jj % B_shape[1];
                } else {
                   B_indices[3] = 0; B_indices[2] = 0; B_indices[1] = 0;
                }

                // Broadcast indices for A
                A_indices[3] = (broadcast_factor_dim3 <= 0) ? 0 : B_indices[3] / broadcast_factor_dim3;
                A_indices[2] = (broadcast_factor_dim2 <= 0) ? 0 : B_indices[2] / broadcast_factor_dim2;

                // Indices for C
                C_indices[1] = B_indices[1];
                C_indices[2] = B_indices[2];
                C_indices[3] = B_indices[3];

                // Calculate pointers
                const char * A_base = (const char *)A->data
                                     + A_indices[2] * A_bstride[2]
                                     + A_indices[3] * A_bstride[3];

                const char * B_col = (const char *)B_data
                                   + B_indices[1] * B_data_bstride
                                   + B_indices[2] * B_bstride[2]
                                   + B_indices[3] * B_bstride[3];

                float * C_ptr_base = (float *)((char *)C->data
                                     + C_indices[1] * C_bstride[1]
                                     + C_indices[2] * C_bstride[2]
                                     + C_indices[3] * C_bstride[3]);

                // Perform dot products for the block
                const int num_rows_to_compute = (int)(MIN(i + BLOCK_SIZE, A_row_end) - i);

                // 逐行调用 vec_dot
                for (int ii = i; ii < i + BLOCK_SIZE && ii < A_row_end; ii ++){
                    const char * A_row_ii = A_base + ii * A_bstride[1];
                    // Call the appropriate vec_dot function (我们的 K=8 展开版本)
                    vec_dot(A_shape[0], temp + (ii - i), 0, A_row_ii, 0, B_col, 0, 1);
                }

                // Copy results from temp buffer to C tensor
                float * C_col_i = C_ptr_base + i; 
                if (num_rows_to_compute > 0) {
                       memcpy(C_col_i, temp, (size_t)num_rows_to_compute * sizeof(float));
                }
            }
        }
    }
}


// ------------------------------------------------------------------------------------------------
// 函数 2: Core Dot Product Optimization (K 维度 8 路展开)
// ------------------------------------------------------------------------------------------------
void rvllm_vec_dot_q4_0_q8_0(int n, float * restrict result, size_t byte_stride_result, const void * restrict vec_x, size_t byte_stride_vec_x, const void * restrict vec_y, size_t byte_stride_vec_y, int num_rows_per_vec_dot) {

    const int BLOCK_SIZE_QK = 32; // 32
    const int num_blocks = n / BLOCK_SIZE_QK;

    UNUSED(byte_stride_result);
    UNUSED(byte_stride_vec_x);
    UNUSED(byte_stride_vec_y);
    UNUSED(num_rows_per_vec_dot);

    const block_q4_0 * restrict x = (const block_q4_0 * restrict)__builtin_assume_aligned(vec_x, 32);
    const block_q8_0 * restrict y = (const block_q8_0 * restrict)__builtin_assume_aligned(vec_y, 32);

    float res = 0.0f;

    // Set max vector lengths needed (LMUL=2 for i16, LMUL=4 for i32)
    const size_t vl_max_16_m2 = __riscv_vsetvlmax_e16m2(); // Max vl for i16m2 accumulator
    const size_t vl_max_32_m4 = __riscv_vsetvlmax_e32m4(); // Max vl for i32m4 widening/reduction
    const size_t block_len_bytes_x = BLOCK_SIZE_QK / 2; // = 16 bytes for x

    int block = 0;

    // ** K 维度 8 路展开 **
    for (; block + 7 < num_blocks; block += 8) {

        // 软件预取 (Prefetching)
        __builtin_prefetch(&x[block + 8], 0, 0);
        __builtin_prefetch(&y[block + 8], 0, 0);

        // --- 初始化 8 个 i16m2 累加器 ---
        vint16m2_t sum16_vec0 = __riscv_vmv_v_x_i16m2(0, vl_max_16_m2);
        vint16m2_t sum16_vec1 = __riscv_vmv_v_x_i16m2(0, vl_max_16_m2);
        vint16m2_t sum16_vec2 = __riscv_vmv_v_x_i16m2(0, vl_max_16_m2);
        vint16m2_t sum16_vec3 = __riscv_vmv_v_x_i16m2(0, vl_max_16_m2);
        vint16m2_t sum16_vec4 = __riscv_vmv_v_x_i16m2(0, vl_max_16_m2);
        vint16m2_t sum16_vec5 = __riscv_vmv_v_x_i16m2(0, vl_max_16_m2);
        vint16m2_t sum16_vec6 = __riscv_vmv_v_x_i16m2(0, vl_max_16_m2);
        vint16m2_t sum16_vec7 = __riscv_vmv_v_x_i16m2(0, vl_max_16_m2);

        // --- Inner loop: Load, Dequantize, and Multiply-Accumulate ---
        {
            // Pointers for 8 blocks
            const uint8_t * x_qs_ptr0 = (const uint8_t *)x[block + 0].qs;
            const int8_t * y_qs_ptr0 = (const int8_t *)y[block + 0].qs;
            const uint8_t * x_qs_ptr1 = (const uint8_t *)x[block + 1].qs;
            const int8_t * y_qs_ptr1 = (const int8_t *)y[block + 1].qs;
            const uint8_t * x_qs_ptr2 = (const uint8_t *)x[block + 2].qs;
            const int8_t * y_qs_ptr2 = (const int8_t *)y[block + 2].qs;
            const uint8_t * x_qs_ptr3 = (const uint8_t *)x[block + 3].qs;
            const int8_t * y_qs_ptr3 = (const int8_t *)y[block + 3].qs;
            const uint8_t * x_qs_ptr4 = (const uint8_t *)x[block + 4].qs;
            const int8_t * y_qs_ptr4 = (const int8_t *)y[block + 4].qs;
            const uint8_t * x_qs_ptr5 = (const uint8_t *)x[block + 5].qs;
            const int8_t * y_qs_ptr5 = (const int8_t *)y[block + 5].qs;
            const uint8_t * x_qs_ptr6 = (const uint8_t *)x[block + 6].qs;
            const int8_t * y_qs_ptr6 = (const int8_t *)y[block + 6].qs;
            const uint8_t * x_qs_ptr7 = (const uint8_t *)x[block + 7].qs;
            const int8_t * y_qs_ptr7 = (const int8_t *)y[block + 7].qs;

            // RVV 循环 (处理 K_s=32 维度)
            for (size_t i_bytes = 0; i_bytes < block_len_bytes_x;) {
                size_t vl = __riscv_vsetvl_e8m1(block_len_bytes_x - i_bytes);

                size_t y_byte_offset = i_bytes * 2;

                // --- 8-way Parallel Load Y ---
                vint8m1_t y_lo_vec0 = __riscv_vle8_v_i8m1(y_qs_ptr0 + y_byte_offset, vl);
                vint8m1_t y_hi_vec0 = __riscv_vle8_v_i8m1(y_qs_ptr0 + y_byte_offset + vl, vl);
                vint8m1_t y_lo_vec1 = __riscv_vle8_v_i8m1(y_qs_ptr1 + y_byte_offset, vl);
                vint8m1_t y_hi_vec1 = __riscv_vle8_v_i8m1(y_qs_ptr1 + y_byte_offset + vl, vl);
                vint8m1_t y_lo_vec2 = __riscv_vle8_v_i8m1(y_qs_ptr2 + y_byte_offset, vl);
                vint8m1_t y_hi_vec2 = __riscv_vle8_v_i8m1(y_qs_ptr2 + y_byte_offset + vl, vl);
                vint8m1_t y_lo_vec3 = __riscv_vle8_v_i8m1(y_qs_ptr3 + y_byte_offset, vl);
                vint8m1_t y_hi_vec3 = __riscv_vle8_v_i8m1(y_qs_ptr3 + y_byte_offset + vl, vl);
                vint8m1_t y_lo_vec4 = __riscv_vle8_v_i8m1(y_qs_ptr4 + y_byte_offset, vl);
                vint8m1_t y_hi_vec4 = __riscv_vle8_v_i8m1(y_qs_ptr4 + y_byte_offset + vl, vl);
                vint8m1_t y_lo_vec5 = __riscv_vle8_v_i8m1(y_qs_ptr5 + y_byte_offset, vl);
                vint8m1_t y_hi_vec5 = __riscv_vle8_v_i8m1(y_qs_ptr5 + y_byte_offset + vl, vl);
                vint8m1_t y_lo_vec6 = __riscv_vle8_v_i8m1(y_qs_ptr6 + y_byte_offset, vl);
                vint8m1_t y_hi_vec6 = __riscv_vle8_v_i8m1(y_qs_ptr6 + y_byte_offset + vl, vl);
                vint8m1_t y_lo_vec7 = __riscv_vle8_v_i8m1(y_qs_ptr7 + y_byte_offset, vl);
                vint8m1_t y_hi_vec7 = __riscv_vle8_v_i8m1(y_qs_ptr7 + y_byte_offset + vl, vl);


                // --- 8-way Parallel Load X ---
                vuint8m1_t x_byte_vec_u0 = __riscv_vle8_v_u8m1(x_qs_ptr0 + i_bytes, vl);
                vuint8m1_t x_byte_vec_u1 = __riscv_vle8_v_u8m1(x_qs_ptr1 + i_bytes, vl);
                vuint8m1_t x_byte_vec_u2 = __riscv_vle8_v_u8m1(x_qs_ptr2 + i_bytes, vl);
                vuint8m1_t x_byte_vec_u3 = __riscv_vle8_v_u8m1(x_qs_ptr3 + i_bytes, vl);
                vuint8m1_t x_byte_vec_u4 = __riscv_vle8_v_u8m1(x_qs_ptr4 + i_bytes, vl);
                vuint8m1_t x_byte_vec_u5 = __riscv_vle8_v_u8m1(x_qs_ptr5 + i_bytes, vl);
                vuint8m1_t x_byte_vec_u6 = __riscv_vle8_v_u8m1(x_qs_ptr6 + i_bytes, vl);
                vuint8m1_t x_byte_vec_u7 = __riscv_vle8_v_u8m1(x_qs_ptr7 + i_bytes, vl);


                // --- 8-way Parallel Dequantize X (Block 0 示例) ---
                vint8m1_t x_qs_lo0 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(x_byte_vec_u0, 0x0F, vl)), 8, vl);
                vint8m1_t x_qs_hi0 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(x_byte_vec_u0, 4, vl)), 8, vl);
                vint8m1_t x_qs_lo1 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(x_byte_vec_u1, 0x0F, vl)), 8, vl);
                vint8m1_t x_qs_hi1 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(x_byte_vec_u1, 4, vl)), 8, vl);
                vint8m1_t x_qs_lo2 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(x_byte_vec_u2, 0x0F, vl)), 8, vl);
                vint8m1_t x_qs_hi2 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(x_byte_vec_u2, 4, vl)), 8, vl);
                vint8m1_t x_qs_lo3 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(x_byte_vec_u3, 0x0F, vl)), 8, vl);
                vint8m1_t x_qs_hi3 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(x_byte_vec_u3, 4, vl)), 8, vl);
                vint8m1_t x_qs_lo4 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(x_byte_vec_u4, 0x0F, vl)), 8, vl);
                vint8m1_t x_qs_hi4 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(x_byte_vec_u4, 4, vl)), 8, vl);
                vint8m1_t x_qs_lo5 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(x_byte_vec_u5, 0x0F, vl)), 8, vl);
                vint8m1_t x_qs_hi5 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(x_byte_vec_u5, 4, vl)), 8, vl);
                vint8m1_t x_qs_lo6 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(x_byte_vec_u6, 0x0F, vl)), 8, vl);
                vint8m1_t x_qs_hi6 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(x_byte_vec_u6, 4, vl)), 8, vl);
                vint8m1_t x_qs_lo7 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(x_byte_vec_u7, 0x0F, vl)), 8, vl);
                vint8m1_t x_qs_hi7 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(x_byte_vec_u7, 4, vl)), 8, vl);


                // ** 8-way Parallel vwmacc into i16m2 accumulator **
                sum16_vec0 = __riscv_vwmacc_vv_i16m2_tu(sum16_vec0, x_qs_lo0, y_lo_vec0, vl);
                sum16_vec0 = __riscv_vwmacc_vv_i16m2_tu(sum16_vec0, x_qs_hi0, y_hi_vec0, vl);
                sum16_vec1 = __riscv_vwmacc_vv_i16m2_tu(sum16_vec1, x_qs_lo1, y_lo_vec1, vl);
                sum16_vec1 = __riscv_vwmacc_vv_i16m2_tu(sum16_vec1, x_qs_hi1, y_hi_vec1, vl);
                sum16_vec2 = __riscv_vwmacc_vv_i16m2_tu(sum16_vec2, x_qs_lo2, y_lo_vec2, vl);
                sum16_vec2 = __riscv_vwmacc_vv_i16m2_tu(sum16_vec2, x_qs_hi2, y_hi_vec2, vl);
                sum16_vec3 = __riscv_vwmacc_vv_i16m2_tu(sum16_vec3, x_qs_lo3, y_lo_vec3, vl);
                sum16_vec3 = __riscv_vwmacc_vv_i16m2_tu(sum16_vec3, x_qs_hi3, y_hi_vec3, vl);
                sum16_vec4 = __riscv_vwmacc_vv_i16m2_tu(sum16_vec4, x_qs_lo4, y_lo_vec4, vl);
                sum16_vec4 = __riscv_vwmacc_vv_i16m2_tu(sum16_vec4, x_qs_hi4, y_hi_vec4, vl);
                sum16_vec5 = __riscv_vwmacc_vv_i16m2_tu(sum16_vec5, x_qs_lo5, y_lo_vec5, vl);
                sum16_vec5 = __riscv_vwmacc_vv_i16m2_tu(sum16_vec5, x_qs_hi5, y_hi_vec5, vl);
                sum16_vec6 = __riscv_vwmacc_vv_i16m2_tu(sum16_vec6, x_qs_lo6, y_lo_vec6, vl);
                sum16_vec6 = __riscv_vwmacc_vv_i16m2_tu(sum16_vec6, x_qs_hi6, y_hi_vec6, vl);
                sum16_vec7 = __riscv_vwmacc_vv_i16m2_tu(sum16_vec7, x_qs_lo7, y_lo_vec7, vl);
                sum16_vec7 = __riscv_vwmacc_vv_i16m2_tu(sum16_vec7, x_qs_hi7, y_hi_vec7, vl);


                i_bytes += vl;
            } // End of inner loop (i_bytes)
        } // End of inner scope

        // --- Widen and Reduction ---

        // 1. Widen 8 intermediate i16m2 sums to i32m4 (分批处理以减轻寄存器压力)
        size_t current_vl_for_widen = __riscv_vsetvl_e16m2(vl_max_16_m2);
        vint32m4_t sum32_vec0 = __riscv_vsext_vf2_i32m4(sum16_vec0, current_vl_for_widen);
        vint32m4_t sum32_vec1 = __riscv_vsext_vf2_i32m4(sum16_vec1, current_vl_for_widen);
        vint32m4_t sum32_vec2 = __riscv_vsext_vf2_i32m4(sum16_vec2, current_vl_for_widen);
        vint32m4_t sum32_vec3 = __riscv_vsext_vf2_i32m4(sum16_vec3, current_vl_for_widen);
        vint32m4_t sum32_vec4 = __riscv_vsext_vf2_i32m4(sum16_vec4, current_vl_for_widen);
        vint32m4_t sum32_vec5 = __riscv_vsext_vf2_i32m4(sum16_vec5, current_vl_for_widen);
        vint32m4_t sum32_vec6 = __riscv_vsext_vf2_i32m4(sum16_vec6, current_vl_for_widen);
        vint32m4_t sum32_vec7 = __riscv_vsext_vf2_i32m4(sum16_vec7, current_vl_for_widen);

        // 2. Reduction
        const vint32m1_t reduction_init_m1 = __riscv_vmv_v_x_i32m1(0, 1);
        size_t current_vl_for_reduce = __riscv_vsetvl_e32m4(vl_max_32_m4);

        vint32m1_t temp_acc0 = __riscv_vredsum_vs_i32m4_i32m1(sum32_vec0, reduction_init_m1, current_vl_for_reduce);
        vint32m1_t temp_acc1 = __riscv_vredsum_vs_i32m4_i32m1(sum32_vec1, reduction_init_m1, current_vl_for_reduce);
        vint32m1_t temp_acc2 = __riscv_vredsum_vs_i32m4_i32m1(sum32_vec2, reduction_init_m1, current_vl_for_reduce);
        vint32m1_t temp_acc3 = __riscv_vredsum_vs_i32m4_i32m1(sum32_vec3, reduction_init_m1, current_vl_for_reduce);
        vint32m1_t temp_acc4 = __riscv_vredsum_vs_i32m4_i32m1(sum32_vec4, reduction_init_m1, current_vl_for_reduce);
        vint32m1_t temp_acc5 = __riscv_vredsum_vs_i32m4_i32m1(sum32_vec5, reduction_init_m1, current_vl_for_reduce);
        vint32m1_t temp_acc6 = __riscv_vredsum_vs_i32m4_i32m1(sum32_vec6, reduction_init_m1, current_vl_for_reduce);
        vint32m1_t temp_acc7 = __riscv_vredsum_vs_i32m4_i32m1(sum32_vec7, reduction_init_m1, current_vl_for_reduce);


        // 3. Extract scalar sum
        int32_t block_sum0 = __riscv_vmv_x_s_i32m1_i32(temp_acc0);
        int32_t block_sum1 = __riscv_vmv_x_s_i32m1_i32(temp_acc1);
        int32_t block_sum2 = __riscv_vmv_x_s_i32m1_i32(temp_acc2);
        int32_t block_sum3 = __riscv_vmv_x_s_i32m1_i32(temp_acc3);
        int32_t block_sum4 = __riscv_vmv_x_s_i32m1_i32(temp_acc4);
        int32_t block_sum5 = __riscv_vmv_x_s_i32m1_i32(temp_acc5);
        int32_t block_sum6 = __riscv_vmv_x_s_i32m1_i32(temp_acc6);
        int32_t block_sum7 = __riscv_vmv_x_s_i32m1_i32(temp_acc7);

        // 4. Dequantization
        res += (float)block_sum0 * _GGML_CPU_FP16_TO_FP32(x[block + 0].d) * _GGML_CPU_FP16_TO_FP32(y[block + 0].d);
        res += (float)block_sum1 * _GGML_CPU_FP16_TO_FP32(x[block + 1].d) * _GGML_CPU_FP16_TO_FP32(y[block + 1].d);
        res += (float)block_sum2 * _GGML_CPU_FP16_TO_FP32(x[block + 2].d) * _GGML_CPU_FP16_TO_FP32(y[block + 2].d);
        res += (float)block_sum3 * _GGML_CPU_FP16_TO_FP32(x[block + 3].d) * _GGML_CPU_FP16_TO_FP32(y[block + 3].d);
        res += (float)block_sum4 * _GGML_CPU_FP16_TO_FP32(x[block + 4].d) * _GGML_CPU_FP16_TO_FP32(y[block + 4].d);
        res += (float)block_sum5 * _GGML_CPU_FP16_TO_FP32(x[block + 5].d) * _GGML_CPU_FP16_TO_FP32(y[block + 5].d);
        res += (float)block_sum6 * _GGML_CPU_FP16_TO_FP32(x[block + 6].d) * _GGML_CPU_FP16_TO_FP32(y[block + 6].d);
        res += (float)block_sum7 * _GGML_CPU_FP16_TO_FP32(x[block + 7].d) * _GGML_CPU_FP16_TO_FP32(y[block + 7].d);


    } // End of K=8 unrolled loop

    // --- K 维度清理循环 (Cleanup Loop) ---
    for (; block < num_blocks; block++) {
        vint16m2_t sum16_vec = __riscv_vmv_v_x_i16m2(0, vl_max_16_m2);
        const uint8_t * x_qs_uint_ptr = (const uint8_t *)x[block].qs;
        const int8_t * y_qs_ptr = (const int8_t *)y[block].qs;

        for (size_t i_bytes = 0; i_bytes < block_len_bytes_x;) {
            size_t vl = __riscv_vsetvl_e8m1(block_len_bytes_x - i_bytes);
            vuint8m1_t x_byte_vec_u = __riscv_vle8_v_u8m1(x_qs_uint_ptr + i_bytes, vl);
            size_t y_byte_offset = i_bytes * 2;
            vint8m1_t y_lo_vec = __riscv_vle8_v_i8m1(y_qs_ptr + y_byte_offset, vl);
            vint8m1_t y_hi_vec = __riscv_vle8_v_i8m1(y_qs_ptr + y_byte_offset + vl, vl);
            vuint8m1_t x_lo_bits_u = __riscv_vand_vx_u8m1(x_byte_vec_u, 0x0F, vl);
            vint8m1_t x_lo_bits = __riscv_vreinterpret_v_u8m1_i8m1(x_lo_bits_u);
            vint8m1_t x_qs_lo = __riscv_vsub_vx_i8m1(x_lo_bits, 8, vl);
            vuint8m1_t x_hi_bits_u = __riscv_vsrl_vx_u8m1(x_byte_vec_u, 4, vl);
            vint8m1_t x_hi_bits = __riscv_vreinterpret_v_u8m1_i8m1(x_hi_bits_u);
            vint8m1_t x_qs_hi = __riscv_vsub_vx_i8m1(x_hi_bits, 8, vl);

            // Use vwmacc into i16m2
            sum16_vec = __riscv_vwmacc_vv_i16m2_tu(sum16_vec, x_qs_lo, y_lo_vec, vl);
            sum16_vec = __riscv_vwmacc_vv_i16m2_tu(sum16_vec, x_qs_hi, y_hi_vec, vl);

            i_bytes += vl;
        }

        // Widen the i16m2 sum to i32m4
        size_t current_vl_for_widen = __riscv_vsetvl_e16m2(vl_max_16_m2);
        vint32m4_t sum32_vec = __riscv_vsext_vf2_i32m4(sum16_vec, current_vl_for_widen);

        // Reduction
        const vint32m1_t initial_m1 = __riscv_vmv_v_x_i32m1(0, 1);
        size_t current_vl_for_reduce = __riscv_vsetvl_e32m4(vl_max_32_m4);
        vint32m1_t temp_acc = __riscv_vredsum_vs_i32m4_i32m1(sum32_vec, initial_m1, current_vl_for_reduce);
        int32_t block_sum = __riscv_vmv_x_s_i32m1_i32(temp_acc);

        // Dequantization
        res += (float)block_sum * _GGML_CPU_FP16_TO_FP32(x[block].d) * _GGML_CPU_FP16_TO_FP32(y[block].d);
    } // End of cleanup loop

    // Final result assignment
    *result = res;

} 


// ------------------------------------------------------------------------------------------------
// 函数 1: ggml_compute_forward_mul_mat_one_chunk (M/N=32 缓存分块，K 维度不分块)
// ------------------------------------------------------------------------------------------------
void ggml_compute_forward_mul_mat_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dest,
    const enum ggml_type compute_type,
    const int64_t num_rows_per_vec_dot,
    const int64_t A_row_start,
    const int64_t A_row_end,
    const int64_t B_col_start,
    const int64_t B_col_end
) {
    // calculate tensor C = A * B
    const struct ggml_tensor * C = dest;
    const struct ggml_tensor * A = C->src[0];
    const struct ggml_tensor * B = C->src[1];

    // Get tensor shape (number of elements per dimension)
    const int A_shape[4] = {A->ne[0], A->ne[1], A->ne[2], A->ne[3]};
    const int B_shape[4] = {B->ne[0], B->ne[1], B->ne[2], B->ne[3]};
    // Get tensor strides in bytes per dimension
    const size_t A_bstride[4] = {A->nb[0], A->nb[1], A->nb[2], A->nb[3]};
    const size_t B_bstride[4] = {B->nb[0], B->nb[1], B->nb[2], B->nb[3]};
    const size_t C_bstride[4] = {C->nb[0], C->nb[1], C->nb[2], C->nb[3]};

    // Some unused parameters
    UNUSED(num_rows_per_vec_dot);

    ggml_vec_dot_t const vec_dot = type_traits_cpu[compute_type].vec_dot;
    enum ggml_type const vec_dot_type = type_traits_cpu[compute_type].vec_dot_type;

    // compute broadcast factors (handle potential zero dims)
    const int broadcast_factor_dim2 = (A_shape[2] == 0 || B_shape[2] == 0 || A_shape[2] == 0) ? 1 : B_shape[2] / A_shape[2];
    const int broadcast_factor_dim3 = (A_shape[3] == 0 || B_shape[3] == 0 || A_shape[3] == 0) ? 1 : B_shape[3] / A_shape[3];


    // If no work, return
    if (A_row_start >= A_row_end || B_col_start >= B_col_end){
        return;
    }

    // Get pointer to B data (potentially converted)
    const void * B_data = (B->type == vec_dot_type) ? B->data : params->wdata;
    const size_t B_data_bstride = (B_shape[0] == 0) ? 0 : ggml_row_size(vec_dot_type, B_shape[0]);


    // ** M/N 分块大小 = 32 **
    const int BLOCK_SIZE_MN = 32;
    
    // ** K 维度：不分块 **
    const int QK_SIZE = 32;
    
    assert(A_shape[0] % QK_SIZE == 0);

    // Temp Buffer Size = 32 (用于存储 C 的一行结果)
    float temp[BLOCK_SIZE_MN]; 

    // Main calculation loops (M/N 维度分块)
    for (int j = B_col_start; j < B_col_end; j += BLOCK_SIZE_MN){ // N 维度分块 (32)
        for (int i = A_row_start; i < A_row_end; i += BLOCK_SIZE_MN){ // M 维度分块 (32)
            
            // 目标: 计算 C[i...i+32, j...j+32] 块

            // 1. N 维度列循环 (jj)
            for (int jj = j; jj < j + BLOCK_SIZE_MN && jj < B_col_end; jj ++){
                
                const int num_rows_to_compute = (int)(MIN(i + BLOCK_SIZE_MN, A_row_end) - i);
                if (num_rows_to_compute <= 0) continue;

                // --- (索引计算部分不变) ---
                int A_indices[4] = {0}, B_indices[4] = {0}, C_indices[4] = {0};

                if (B_shape[1] > 0 && B_shape[2] > 0) {
                   B_indices[3] = jj / (B_shape[2] * B_shape[1]);
                   B_indices[2] = (jj % (B_shape[2] * B_shape[1])) / B_shape[1];
                   B_indices[1] = jj % B_shape[1];
                } else if (B_shape[1] > 0) {
                   B_indices[3] = 0; B_indices[2] = 0;
                   B_indices[1] = jj % B_shape[1];
                } else {
                   B_indices[3] = 0; B_indices[2] = 0; B_indices[1] = 0;
                }
                A_indices[3] = (broadcast_factor_dim3 <= 0) ? 0 : B_indices[3] / broadcast_factor_dim3;
                A_indices[2] = (broadcast_factor_dim2 <= 0) ? 0 : B_indices[2] / broadcast_factor_dim2;
                C_indices[1] = B_indices[1];
                C_indices[2] = B_indices[2];
                C_indices[3] = B_indices[3];

                // --- (指针计算部分不变) ---
                const char * A_base = (const char *)A->data
                                     + A_indices[2] * A_bstride[2]
                                     + A_indices[3] * A_bstride[3];

                const char * B_col = (const char *)B_data
                                   + B_indices[1] * B_data_bstride
                                   + B_indices[2] * B_bstride[2]
                                   + B_indices[3] * B_bstride[3];

                float * C_ptr_base = (float *)((char *)C->data
                                     + C_indices[1] * C_bstride[1]
                                     + C_indices[2] * C_bstride[2]
                                     + C_indices[3] * C_bstride[3]);

                // 3. 逐行调用 vec_dot (M 展开)
                for (int ii = i; ii < i + BLOCK_SIZE_MN && ii < A_row_end; ii ++){
                    
                    const char * A_row_ii = A_base + ii * A_bstride[1];
                    
                    // Call the appropriate vec_dot function (K 维度不分块)
                    vec_dot(A_shape[0],         // n (完整的 K 维度)
                            temp + (ii - i),    // result (写入 temp)
                            0,                  // byte_stride_result
                            A_row_ii,           // vec_x 
                            0,                  // byte_stride_vec_x (单行)
                            B_col,              // vec_y
                            0,                  // byte_stride_vec_y
                            1);                 // num_rows_per_vec_dot = 1
                }
                
                // Copy results from temp buffer to C tensor
                float * C_col_i = C_ptr_base + i; 
                if (num_rows_to_compute > 0) {
                       memcpy(C_col_i, temp, (size_t)num_rows_to_compute * sizeof(float));
                }
            } // 结束 jj (N 维度) 循环
        } // 结束 i (M 维度) 循环
    } // 结束 j (N 维度) 循环
}
