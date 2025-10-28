#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm> // 用于 std::min/max
#include <tuple>     // 用于返回 tuple
#include <omp.h>     // 用于 OpenMP 并行化
#include <limits>    // 用于 std::numeric_limits
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
/**
 * @brief 使用 OpenMP 并行计算下一代，基于扁平的一维缓冲区。
 * * @param current_flat 输入缓冲区 (带 1 圈 padding)
 * @param next_flat 输出缓冲区 (带 1 圈 padding)
 * @param H 当前逻辑高度 (不含 padding)
 * @param W 当前逻辑宽度 (不含 padding)
 * @return InternalResult 包含 {min_y, max_y, min_x, max_x} 或灭绝信号 {-1, -1, -1, -1}。
 */
InternalResult Next_Generation_Cpp_Internal(
    const std::vector<int>& current_flat, // 输入缓冲区 (带 padding H+2 x W+2)
    std::vector<int>& next_flat,          // 输出缓冲区 (带 padding H+2 x W+2)
    int H, int W                          // 当前逻辑尺寸 (不含 padding)
) {
    int padded_H = H + 2;
    int padded_W = W + 2;

    // 1. 确保输出缓冲区容量足够并清零
    size_t required_size = (size_t)padded_H * padded_W;
    next_flat.assign(required_size, 0);

    // 2. 计算下一代 (基于扁平数组)
    int min_y = padded_H, max_y = -1, min_x = padded_W, max_x = -1;

    // OpenMP 并行计算，使用 reduction 收集新的活动边界
    #pragma omp parallel for collapse(2) \
        reduction(min:min_y, min_x) \
        reduction(max:max_y, max_x) \
        schedule(static)
    for (int y = 0; y < padded_H; ++y) {
        for (int x = 0; x < padded_W; ++x) {

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
            
            // [Get State] - 从 current_flat 读
            int current_state = current_flat[y * padded_W + x];

            // [Apply Rules]
            bool is_alive_next = (live_neighbors == 3) || (live_neighbors == 2 && current_state == 1);

            // [Write Result] - 写入 next_flat 并更新边界
            if (is_alive_next) {
                next_flat[y * padded_W + x] = 1;
                // Reduction 变量会在 OMP 循环结束后合并
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
            }
        }
    } // --- OMP 并行区域结束 ---

    if (max_y == -1) {
        return {-1, -1, -1, -1}; // 灭绝信号
    }

    return {min_y, max_y, min_x, max_x};
}


// ------------------------------------------------------------------------------------------------
// 主迭代函数 (无 Profiling)
// ------------------------------------------------------------------------------------------------
/**
 * @brief 模拟多代康威生命游戏，使用扁平缓冲区和 OpenMP。
 * * @param initial_grid 初始 2D 网格。
 * @param generations 最大迭代代数。
 * @return std::vector<std::vector<int>> 最终的 2D 网格。
 */
std::vector<std::vector<int>> expand_cpp(
    const std::vector<std::vector<int>>& initial_grid,
    int generations) {

    if (initial_grid.empty() || initial_grid[0].empty()) {
        return {};
    }

    int H = initial_grid.size();
    int W = initial_grid[0].size();

    // --- [1. Initial Alloc & Conversion] ---
    int initial_padded_H = H + 2;
    int initial_padded_W = W + 2;
    size_t initial_size = (size_t)initial_padded_H * initial_padded_W;

    // 双缓冲区
    std::vector<int> buffer_A(initial_size, 0);
    std::vector<int> buffer_B(initial_size, 0);

    // 将初始 2D 网格拷贝到带 padding 的 A 缓冲区
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (initial_grid[y][x] == 1) {
                // (y+1, x+1) 是带 padding 的坐标
                buffer_A[(y + 1) * initial_padded_W + (x + 1)] = 1;
            }
        }
    }

    std::vector<int>* current_buf = &buffer_A;
    std::vector<int>* next_buf = &buffer_B;

    InternalResult prev_bounds = {-2, -2, -2, -2}; // 用于初始化，确保第一次不匹配
    InternalResult current_bounds = {-1, -1, -1, -1}; // 占位符
    int current_H = H;
    int current_W = W;

    for (int gen = 0; gen < generations; ++gen) {

        // --- [2. Internal Compute Step] ---
        int current_padded_H = current_H + 2;
        int current_padded_W = current_W + 2;
        size_t required_size = (size_t)current_padded_H * current_padded_W;

        // 检查并扩展缓冲区容量
        if (current_buf->capacity() < required_size) {
             size_t new_cap = required_size + (size_t)RESIZE_SLACK * current_padded_W;
             current_buf->reserve(new_cap);
        }
        if (next_buf->capacity() < required_size) {
             size_t new_cap = required_size + (size_t)RESIZE_SLACK * current_padded_W;
             next_buf->reserve(new_cap);
        }
        // 如果 current_buf 的 size 小于 required_size，需要调整 size 并补 0 (如果需要)
        // 注意：Next_Generation_Cpp_Internal 会在开始时调用 assign(required_size, 0)，
        // 确保 next_buf 的 size 正确，但我们需要确保 current_buf 的 size 至少能覆盖读取范围。
        // 由于 current_buf 是上一代计算的结果，其 size 应该已经正确或更大，
        // 除非 current_H 或 current_W 增长超出了上一次 reserve 的容量。
        // 为了安全起见，我们将 current_buf resize 到当前逻辑所需的最大容量 (current_padded_H * current_padded_W) 
        // 实际使用 current_H 和 current_W 来定义当前活动区域。

        current_bounds = Next_Generation_Cpp_Internal(*current_buf, *next_buf, current_H, current_W);
        
        // 检查灭绝
        if (std::get<0>(current_bounds) == -1) {
             current_H = 0; // 标记最终 H=0
             current_W = 0;
             break; // 退出循环
        }

        // --- [3. Stability Check] ---
        int min_y, max_y, min_x, max_x;
        // 修正：确保所有四个变量都被正确解包
        std::tie(min_y, max_y, min_x, max_x) = current_bounds; 
        int new_H = max_y - min_y + 1;
        int new_W = max_x - min_x + 1;

        bool stable = false;
        if (current_bounds == prev_bounds) {
            // 边界相同，检查内容是否相同
            std::atomic<bool> content_differs(false);
            int prev_padded_W = current_W + 2; // 上一代的 padded_W

            #pragma omp parallel for schedule(static)
            for (int y = min_y; y <= max_y; ++y) {
                 if (content_differs.load(std::memory_order_relaxed)) continue;

                 int current_start_idx = y * prev_padded_W + min_x;
                 int next_start_idx = y * prev_padded_W + min_x;
                 
                 // 使用 memcmp 比较两个缓冲区中活动区域的对应行
                 if (std::memcmp(&(*current_buf)[current_start_idx], &(*next_buf)[next_start_idx], new_W * sizeof(int)) != 0) {
                      content_differs.store(true, std::memory_order_relaxed);
                 }
            }
            if (!content_differs.load()) {
                 stable = true;
            }
        }

        // 更新 H, W 为下一代做准备
        current_H = new_H;
        current_W = new_W;
        prev_bounds = current_bounds;

        // --- [4. Swap Buffers] ---
        std::swap(current_buf, next_buf);

        if (stable) break;

    } // end generation loop

    // --- [5. Final Conversion] ---
    int final_H = current_H;
    int final_W = current_W;
    std::vector<std::vector<int>> final_grid_2d;

    if (final_H > 0 && final_W > 0) {
        final_grid_2d.resize(final_H, std::vector<int>(final_W));
        int final_min_y, final_max_y, final_min_x, final_max_x;
        
        if (std::get<0>(current_bounds) != -1) {
            std::tie(final_min_y, final_max_y, final_min_x, final_max_x) = current_bounds;
            int final_padded_W = current_W + 2; // 使用最终的 W 计算 padded_W

            // 从 current_buf 的活动区域拷贝到 2D 网格 (串行转换)
            for (int y = 0; y < final_H; ++y) {
                int start_index = (final_min_y + y) * final_padded_W + final_min_x;
                // 使用 memcpy 进行整行拷贝以提高效率
                std::memcpy(final_grid_2d[y].data(), &(*current_buf)[start_index], final_W * sizeof(int));
            }
        }
    }

    return final_grid_2d;
}

// Pybind11 模块绑定
PYBIND11_MODULE(NG, m) {
    m.def("Expand_Cpp", &expand_cpp,
          "Simulate multiple generations using flat buffers and OpenMP.",
          py::arg("initial_grid"), py::arg("generations"));
}

