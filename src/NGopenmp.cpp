#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm> // 用于 std::min/max
#include <tuple>     // 用于返回 tuple
#include <omp.h>     // 用于 OpenMP 并行化
#include <limits>    // 用于 std::numeric_limits
#include <chrono>    // 【启用】: 包含计时器
#include <iostream>  // 【启用】: 包含打印
#include <cstring>   // For std::memcpy
#include <atomic>    // For atomic bool in stability check

namespace py = pybind11;

// --- 返回类型 ---
// GridResult: 最终返回给 Python 的类型
using GridResult = std::pair<std::vector<std::vector<int>>, std::pair<int, int>>;
// InternalResult: 内部函数返回边界信息 {min_y, max_y, min_x, max_x}
// 如果灭绝，返回 {-1, -1, -1, -1}
using InternalResult = std::tuple<int, int, int, int>;

// --- 参数 ---
constexpr int RESIZE_SLACK = 10; // 预留余量 (行/列)

// ------------------------------------------------------------------------------------------------
// 内部计算函数 (操作扁平数组)
// ------------------------------------------------------------------------------------------------
InternalResult Next_Generation_Cpp_Internal(
    const std::vector<int>& current_flat, // 输入缓冲区 (带 padding H+2 x W+2)
    std::vector<int>& next_flat,          // 输出缓冲区 (带 padding H+2 x W+2)
    int H, int W                          // 当前逻辑尺寸 (不含 padding)
) {
    int padded_H = H + 2;
    int padded_W = W + 2;

    // --- [内部 Profiling] ---
    double time_neighbors_cpu = 0;
    double time_state_cpu = 0;
    double time_rules_cpu = 0;
    double time_write_cpu = 0;
    auto wall_compute_start = std::chrono::high_resolution_clock::now();
    // -------------------------

    // 1. 确保输出缓冲区容量足够并清零
    size_t required_size = (size_t)padded_H * padded_W;
    next_flat.assign(required_size, 0);

    // 2. 计算下一代 (基于扁平数组)
    int min_y = padded_H, max_y = -1, min_x = padded_W, max_x = -1;

    #pragma omp parallel for collapse(2) \
        reduction(min:min_y, min_x) \
        reduction(max:max_y, max_x) \
        reduction(+:time_neighbors_cpu, time_state_cpu, time_rules_cpu, time_write_cpu) \
        schedule(static)
    for (int y = 0; y < padded_H; ++y) {
        for (int x = 0; x < padded_W; ++x) {

            auto internal_timer_start = std::chrono::high_resolution_clock::now();

            // [Neighbors] - 从 current_flat (H+2 x W+2) 读取
            int live_neighbors = 0;
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    if (i == 0 && j == 0) continue;
                    int neighbor_y = y + i;
                    int neighbor_x = x + j;
                    // 边界检查 (因为 current_flat 只有一层 padding)
                    if (neighbor_y >= 0 && neighbor_y < padded_H && neighbor_x >= 0 && neighbor_x < padded_W) {
                         live_neighbors += current_flat[neighbor_y * padded_W + neighbor_x];
                    }
                }
            }
             auto internal_timer_neighbors = std::chrono::high_resolution_clock::now();

            // [Get State] - 从 current_flat 读
            int current_state = current_flat[y * padded_W + x];
            auto internal_timer_state = std::chrono::high_resolution_clock::now();

            // [Apply Rules]
            bool is_alive_next = (live_neighbors == 3) || (live_neighbors == 2 && current_state == 1);
            auto internal_timer_rules = std::chrono::high_resolution_clock::now();

            // [Write Result] - 写入 next_flat
            if (is_alive_next) {
                next_flat[y * padded_W + x] = 1;
                 if (y < min_y) min_y = y;
                 if (y > max_y) max_y = y;
                 if (x < min_x) min_x = x;
                 if (x > max_x) max_x = x;
            }
             auto internal_timer_write = std::chrono::high_resolution_clock::now();

            time_neighbors_cpu += std::chrono::duration_cast<std::chrono::microseconds>(internal_timer_neighbors - internal_timer_start).count();
            time_state_cpu     += std::chrono::duration_cast<std::chrono::microseconds>(internal_timer_state - internal_timer_neighbors).count();
            time_rules_cpu     += std::chrono::duration_cast<std::chrono::microseconds>(internal_timer_rules - internal_timer_state).count();
            time_write_cpu     += std::chrono::duration_cast<std::chrono::microseconds>(internal_timer_write - internal_timer_rules).count();
        }
    } // --- OMP 并行区域结束 ---
    auto wall_compute_end = std::chrono::high_resolution_clock::now();

    // 【Profiling】: 打印内部计算 Profiling
    auto& out = std::cerr;
    out << "    --- Internal Compute Profiling (us) ---" << std::endl;
    out << "    [WALL] Internal Compute Loop: " << std::chrono::duration_cast<std::chrono::microseconds>(wall_compute_end - wall_compute_start).count() << std::endl;
    out << "    --- CPU Time (us) ---" << std::endl;
    out << "    [CPU] Neighbors:              " << time_neighbors_cpu << std::endl;
    out << "    [CPU] Get State:              " << time_state_cpu << std::endl;
    out << "    [CPU] Apply Rules:            " << time_rules_cpu << std::endl;
    out << "    [CPU] Write Result:           " << time_write_cpu << std::endl;
    out << "    ---------------------------------------" << std::endl;


    if (max_y == -1) {
        return {-1, -1, -1, -1}; // 灭绝信号
    }

    return {min_y, max_y, min_x, max_x};
}


// ------------------------------------------------------------------------------------------------
// 主迭代函数 (重构 + Profiling)
// ------------------------------------------------------------------------------------------------
std::vector<std::vector<int>> expand_cpp(
    const std::vector<std::vector<int>>& initial_grid,
    int generations) {

    // 【Profiling】
    auto wall_total_start = std::chrono::high_resolution_clock::now();
    double total_compute_wall_time = 0; // 累加每一代的计算 Wall time
    double total_stability_wall_time = 0; // 累加每一代的稳定性检查 Wall time

    if (initial_grid.empty() || initial_grid[0].empty()) {
        return {};
    }

    int H = initial_grid.size();
    int W = initial_grid.0].size();

    // --- [1. Initial Alloc & Conversion] ---
    auto wall_initial_alloc_start = std::chrono::high_resolution_clock::now();

    int initial_padded_H = H + 2;
    int initial_padded_W = W + 2;
    size_t initial_size = (size_t)initial_padded_H * initial_padded_W;

    std::vector<int> buffer_A(initial_size, 0);
    std::vector<int> buffer_B(initial_size, 0);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (initial_grid[y][x] == 1) {
                buffer_A[(y + 1) * initial_padded_W + (x + 1)] = 1;
            }
        }
    }
    auto wall_initial_alloc_end = std::chrono::high_resolution_clock::now();


    std::vector<int>* current_buf = &buffer_A;
    std::vector<int>* next_buf = &buffer_B;

    InternalResult prev_bounds = {-2, -2, -2, -2};
    InternalResult current_bounds = {-1, -1, -1, -1};
    int current_H = H;
    int current_W = W;

    int actual_generations = 0; // 记录实际运行的代数

    for (int gen = 0; gen < generations; ++gen) {
        actual_generations = gen + 1; // 更新实际运行代数

        // --- [2. Internal Compute Step] ---
        auto wall_gen_start = std::chrono::high_resolution_clock::now();

        int current_padded_H = current_H + 2;
        int current_padded_W = current_W + 2;
        size_t required_size = (size_t)current_padded_H * current_padded_W;

        // 检查并扩展缓冲区容量
        bool resized = false;
        if (current_buf->capacity() < required_size) {
             size_t new_cap = required_size + (size_t)(RESIZE_SLACK + 2) * current_padded_W;
             current_buf->reserve(new_cap);
             resized = true;
             std::cerr << "[expand_cpp] Resized current_buf capacity at gen " << gen << " for H=" << current_H << std::endl;
        }
         if (next_buf->capacity() < required_size) {
             size_t new_cap = required_size + (size_t)(RESIZE_SLACK + 2) * current_padded_W;
             next_buf->reserve(new_cap);
             resized = true; // 即使只resize一个，也标记为resized
             std::cerr << "[expand_cpp] Resized next_buf capacity at gen " << gen << " for H=" << current_H << std::endl;
         }
         // 如果 reserve 导致容量变化，可能需要调整 size 以匹配 required_size
         // 但 assign 会处理 size，所以这里不需要显式 resize(size)
         if(resized && current_buf->size() < required_size) {
             // 如果reserve后size不足，用0填充到需要的大小
             current_buf->resize(required_size, 0);
             std::cerr << "[expand_cpp] Adjusted current_buf size after reserve at gen " << gen << std::endl;
         }


        std::cerr << "[expand_cpp] Gen " << gen << " Start:" << std::endl;
        current_bounds = Next_Generation_Cpp_Internal(*current_buf, *next_buf, current_H, current_W);
        auto wall_gen_compute_end = std::chrono::high_resolution_clock::now();
        total_compute_wall_time += std::chrono::duration_cast<std::chrono::microseconds>(wall_gen_compute_end - wall_gen_start).count(); // 累加计算时间

        // 检查灭绝
        if (std::get<0>(current_bounds) == -1) {
             std::cerr << "[expand_cpp] Extinction detected at gen " << gen << std::endl;
             current_H = 0; // 标记最终 H=0
             current_W = 0;
             actual_generations = gen; // 灭绝发生在 gen 这一代开始时
             break; // 退出循环
        }

        // --- [3. Stability Check] ---
        auto wall_stability_start = std::chrono::high_resolution_clock::now();
        int min_y, max_y, min_x, max_x;
        std::tie(min_y, max_y, min_x, max_x) = current_bounds;
        int new_H = max_y - min_y + 1;
        int new_W = max_x - min_x + 1;

        bool stable = false;
        if (current_bounds == prev_bounds) {
            std::atomic<bool> content_differs(false);
            int prev_padded_W = current_W + 2; // 上一代的 padded_W
            #pragma omp parallel for schedule(static)
            for (int y = min_y; y <= max_y; ++y) {
                 if (content_differs.load(std::memory_order_relaxed)) continue;

                 int current_start_idx = y * prev_padded_W + min_x; // prev_grid 在 current_buf
                 int next_start_idx = y * prev_padded_W + min_x;    // new_grid 在 next_buf

                 // 确保索引在缓冲区范围内 (理论上应该在)
                 if (current_start_idx + new_W <= current_buf->size() &&
                     next_start_idx + new_W <= next_buf->size())
                 {
                     if (std::memcmp(&(*current_buf)[current_start_idx], &(*next_buf)[next_start_idx], new_W * sizeof(int)) != 0) {
                         content_differs.store(true, std::memory_order_relaxed);
                     }
                 } else {
                     // 索引越界，说明逻辑有问题，强制标记为不稳定
                     content_differs.store(true, std::memory_order_relaxed);
                     #pragma omp critical
                     std::cerr << "[ERROR] Stability check index out of bounds at gen " << gen << "!" << std::endl;
                 }
            }
             if (!content_differs.load()) {
                 stable = true;
                 std::cerr << "[expand_cpp] Stable state detected at gen " << gen << std::endl;
             } else {
                  std::cerr << "[expand_cpp] Bounds match but content differs at gen " << gen << std::endl;
             }
        }
        auto wall_stability_end = std::chrono::high_resolution_clock::now();
        total_stability_wall_time += std::chrono::duration_cast<std::chrono::microseconds>(wall_stability_end - wall_stability_start).count(); // 累加检查时间


        // 更新 H, W 为下一代做准备
        current_H = new_H;
        current_W = new_W;
        prev_bounds = current_bounds;

        // --- [4. Swap Buffers] ---
        std::swap(current_buf, next_buf);
        auto wall_gen_end = std::chrono::high_resolution_clock::now(); // 包含 swap 的时间

        std::cerr << "  [WALL] Gen " << gen << " Total:         "
                  << std::chrono::duration_cast<std::chrono::microseconds>(wall_gen_end - wall_gen_start).count() << " us" << std::endl;
        std::cerr << "  [WALL]   Compute Step:  " // Wall time of NG_Internal call
                  << std::chrono::duration_cast<std::chrono::microseconds>(wall_gen_compute_end - wall_gen_start).count() << " us" << std::endl;
        std::cerr << "  [WALL]   Stability Chk: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(wall_stability_end - wall_stability_start).count() << " us" << std::endl;
        std::cerr << "  [WALL]   Swap Overhead: " // Swap 很快，主要是计时误差
                  << std::chrono::duration_cast<std::chrono::microseconds>(wall_gen_end - wall_stability_end).count() << " us" << std::endl;

        if (stable) break;

    } // end generation loop

    // --- [5. Final Conversion] ---
    auto wall_final_convert_start = std::chrono::high_resolution_clock::now();

    int final_H = current_H;
    int final_W = current_W;
    std::vector<std::vector<int>> final_grid_2d; // 初始为空

    if (final_H > 0 && final_W > 0) {
        final_grid_2d.resize(final_H, std::vector<int>(final_W));
        int final_min_y, final_max_y, final_min_x, final_max_x;
        // 如果灭绝，bounds 是 {-1,-1,-1,-1}；如果稳定，是稳定时的 bounds
        if (std::get<0>(current_bounds) != -1) {
            std::tie(final_min_y, final_max_y, final_min_x, final_max_x) = current_bounds;
            int final_padded_W = final_W + 2; // 需要用最终的 W 计算 padded_W

            // 从 current_buf 的 (final_min_y + y, final_min_x) 处读取
            // (串行转换)
            for (int y = 0; y < final_H; ++y) {
                int start_index = (final_min_y + y) * final_padded_W + final_min_x;
                // 边界检查，防止因稳定检查逻辑错误导致越界
                if(start_index + final_W <= current_buf->size()){
                    for (int x = 0; x < final_W; ++x) {
                        final_grid_2d[y][x] = (*current_buf)[start_index + x];
                    }
                } else {
                     #pragma omp critical
                     std::cerr << "[ERROR] Final conversion index out of bounds!" << std::endl;
                     // 返回一个空网格或部分网格
                     final_grid_2d.clear(); // 清空以表示错误
                     break;
                }
            }
        }
    } // else: H 或 W <= 0 (灭绝), final_grid_2d 保持为空


    auto wall_final_convert_end = std::chrono::high_resolution_clock::now();
    auto wall_total_end = std::chrono::high_resolution_clock::now();

    // --- [6. 打印最终报告] ---
     auto& out = std::cerr;
     out << "--- expand_cpp Profiling Summary (us) ---" << std::endl;
     out << "[WALL] Initial Alloc & Convert: " << std::chrono::duration_cast<std::chrono::microseconds>(wall_initial_alloc_end - wall_initial_alloc_start).count() << std::endl;
     out << "[WALL] Total Compute Wall Time: " << total_compute_wall_time << " (across " << actual_generations << " gens)" << std::endl;
     out << "[WALL] Total Stability Wall Time:" << total_stability_wall_time << " (across " << actual_generations << " gens)" << std::endl;
     out << "[WALL] Final Convert:           " << std::chrono::duration_cast<std::chrono::microseconds>(wall_final_convert_end - wall_final_convert_start).count() << std::endl;
     out << "[WALL] Total expand_cpp Wall Time:" << std::chrono::duration_cast<std::chrono::microseconds>(wall_total_end - wall_total_start).count() << std::endl;
     out << "---------------------------------------" << std::endl;

    return final_grid_2d;
}

// Pybind11 模块绑定
PYBIND11_MODULE(NG, m) {
    m.def("Expand_Cpp", &expand_cpp,
          "Simulate multiple generations using flat buffers internally (Meticulous Profiling).",
          py::arg("initial_grid"), py::arg("generations"));

    // 我们不再导出 Next_Generation_Cpp，因为它现在是内部逻辑
}
