#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm> // 用于 std::min/max
#include <tuple>     // 用于返回 tuple
#include <omp.h>     // 用于 OpenMP 并行化
#include <limits>    // 用于 std::numeric_limits
#include <utility>   // For std::move
#include <cstring>   // For std::memcmp
#include <atomic>    // For atomic bool in stability check
#include <cstdint>   // For uint8_t
#include <arm_neon.h> // NEON intrinsic 头文件

namespace py = pybind11;

// --- 类型定义 ---
// 核心缓冲区类型改为 uint8_t
using FlatBuffer = std::vector<uint8_t>;
// 外部接口返回类型不变
using GridResult = std::pair<std::vector<std::vector<int>>, std::pair<int, int>>;
// 内部边界返回类型不变
using InternalResult = std::tuple<int, int, int, int>;

// --- 参数 ---
constexpr int RESIZE_SLACK = 10; // 预留余量
constexpr int NEON_SIZE = 16;    // 每次处理 16 个细胞

// ------------------------------------------------------------------------------------------------
// 内部计算函数 (操作扁平数组)
// ------------------------------------------------------------------------------------------------
InternalResult Next_Generation_Internal(
    const FlatBuffer& current_flat, // 输入缓冲区 (uint8_t, 带 padding H+2 x W+2)
    FlatBuffer& next_flat,          // 输出缓冲区 (uint8_t, 带 padding H+2 x W+2)
    int H, int W,                   // 当前逻辑尺寸 (H_out, W_out)
    int in_min_y, int in_min_x,     // 输入数据在缓冲区中的偏移量
    int in_padded_H, int in_padded_W // 输入缓冲区的实际尺寸
) {
    int padded_H = H + 2;
    int padded_W = W + 2;
    size_t required_size = (size_t)padded_H * padded_W;

    // 清零并设置大小 (必须清零，因为 next_flat 存储了下一代的结果)
    next_flat.assign(required_size, 0);

    // 2. 计算下一代 (基于扁平数组)
    // 初始化边界为最大/最小值，作为 OpenMP 约简的目标
    int min_y = padded_H, max_y = -1, min_x = padded_W, max_x = -1;

    // NEON 向量化处理区域: y 从 1 到 H (跳过顶底边界)
    int y_start = 1;
    int y_end = padded_H - 1;
    
    // x 的 NEON 区域: 从 1 到 W (跳过左右边界)，并且是 16 的倍数
    int x_start_neon = 1;
    
    // **NEON 安全边界修正**
    // 约束条件: x_loop < in_padded_W - in_min_x - 15 (为安全取 -16)
    int x_loop_limit = in_padded_W - in_min_x - 16;

    int vector_cells_available = (x_loop_limit > 1) ? (x_loop_limit - 1) : 0;
    
    int vector_cells_aligned = (vector_cells_available / NEON_SIZE) * NEON_SIZE;
    
    // x_vec_end 是 NEON 循环的终止点 (第一个不执行 NEON 的 x 值)
    int x_vec_end = 1 + vector_cells_aligned; 
    
    // 注意：已移除 temp_output 和 alignas，以提高性能。


    // --- OMP 并行区域：基于行 (y) ---
    #pragma omp parallel reduction(min:min_y, min_x) reduction(max:max_y, max_x)
    {
        // 声明线程私有的边界变量，用于在每次写入活细胞时更新边界
        int thread_min_y = padded_H, thread_max_y = -1;
        int thread_min_x = padded_W, thread_max_x = -1;
        
        // 外部循环：并行处理行
        #pragma omp for schedule(static)
        for (int y = 0; y < padded_H; ++y) {
            
            // 当前行在输入缓冲区 current_flat 中的基准 y 坐标 (对应 next_flat[y][x] 的中心细胞)
            int current_read_y_base = y + in_min_y - 1; 
            
            // 内部循环：NEON + 标量混合处理列
            int x = 0;

            // -----------------------------------------------------------------------
            // 1. 标量处理：左边界 (x=0) 和 1-cell 缓冲区 (x < x_start_neon)
            // -----------------------------------------------------------------------
            for (; x < x_start_neon; ++x) {
                
                // [Neighbors]
                int live_neighbors = 0;
                for (int i = -1; i <= 1; ++i) {
                    for (int j = -1; j <= 1; ++j) {
                        if (i == 0 && j == 0) continue;

                        int read_neighbor_y = y + i + in_min_y - 1;
                        int read_neighbor_x = x + j + in_min_x - 1;

                        if (read_neighbor_y >= 0 && read_neighbor_y < in_padded_H &&
                            read_neighbor_x >= 0 && read_neighbor_x < in_padded_W) {
                            live_neighbors += current_flat[read_neighbor_y * in_padded_W + read_neighbor_x];
                        }
                    }
                }

                // [Get State]
                int current_state = 0;
                int current_read_x = x + in_min_x - 1;
                if (current_read_y_base >= 0 && current_read_y_base < in_padded_H && current_read_x >= 0 && current_read_x < in_padded_W) {
                    current_state = current_flat[current_read_y_base * in_padded_W + current_read_x];
                }

                // [Apply Rules & Write Result]
                bool is_alive_next = (live_neighbors == 3) || (live_neighbors == 2 && current_state == 1);

                if (is_alive_next) {
                    next_flat[y * padded_W + x] = 1;
                    if (y < thread_min_y) thread_min_y = y;
                    if (y > thread_max_y) thread_max_y = y;
                    if (x < thread_min_x) thread_min_x = x;
                    if (x > thread_max_x) thread_max_x = x;
                }
            }
            
            // -----------------------------------------------------------------------
            // 2. NEON 向量化处理：核心区域 (y=1 到 H，x=1 到 W-16)
            // -----------------------------------------------------------------------
            // 只有 y 在 (1, padded_H - 2) 范围内才进行 NEON 处理，以避免不必要的边界检查
            if (y >= y_start && y < y_end) {
                
                const int input_W = in_padded_W; // 输入缓冲区的宽度
                
                for (; x < x_vec_end; x += NEON_SIZE) {
                    
                    // 指向输入缓冲区 current_flat 中 3 个相关行的起始点 (x-1 的位置)
                    const uint8_t* row_base = current_flat.data();
                    
                    // 相对输入缓冲区起始位置 (in_min_x - 1 + x - 1)
                    int input_x_offset_start = in_min_x - 1 + x - 1;

                    // 指向 T_L (Top Left)
                    const uint8_t* row_y_minus_1 = row_base + (current_read_y_base - 1) * input_W + input_x_offset_start;
                    const uint8_t* row_y = row_base + current_read_y_base * input_W + input_x_offset_start;
                    const uint8_t* row_y_plus_1 = row_base + (current_read_y_base + 1) * input_W + input_x_offset_start;

                    // 1. 加载 9 个向量所需的 9 个数据块 
                    // T_L 对应 x-1, T_C 对应 x, T_R 对应 x+1
                    
                    // Row Y-1 (Top)
                    uint8x16_t T_L = vld1q_u8(row_y_minus_1);
                    uint8x16_t T_C = vld1q_u8(row_y_minus_1 + 1);
                    uint8x16_t T_R = vld1q_u8(row_y_minus_1 + 2); 
                    
                    // Row Y (Middle)
                    uint8x16_t M_L = vld1q_u8(row_y);
                    uint8x16_t M_C = vld1q_u8(row_y + 1); // 也是 Current State
                    uint8x16_t M_R = vld1q_u8(row_y + 2); 
                    
                    // Row Y+1 (Bottom)
                    uint8x16_t B_L = vld1q_u8(row_y_plus_1);
                    uint8x16_t B_C = vld1q_u8(row_y_plus_1 + 1);
                    uint8x16_t B_R = vld1q_u8(row_y_plus_1 + 2); 
                    
                    // 2. 邻居求和 (8 个邻居)
                    uint8x16_t neighbor_sum = vaddq_u8(T_L, T_C);
                    neighbor_sum = vaddq_u8(neighbor_sum, T_R);
                    
                    neighbor_sum = vaddq_u8(neighbor_sum, M_L);
                    neighbor_sum = vaddq_u8(neighbor_sum, M_R); // 排除 M_C (中心细胞)
                    
                    neighbor_sum = vaddq_u8(neighbor_sum, B_L);
                    neighbor_sum = vaddq_u8(neighbor_sum, B_C);
                    neighbor_sum = vaddq_u8(neighbor_sum, B_R);

                    // 3. 应用规则
                    uint8x16_t current_state = M_C; 
                    
                    // Rule 1: Birth - Sum is exactly 3
                    uint8x16_t mask_birth = vceqq_u8(neighbor_sum, vdupq_n_u8(3));
                    
                    // Rule 2: Survival - Sum is 2 AND Current state is 1
                    uint8x16_t mask_sum_2 = vceqq_u8(neighbor_sum, vdupq_n_u8(2));
                    uint8x16_t mask_current_alive = vceqq_u8(current_state, vdupq_n_u8(1));
                    uint8x16_t mask_survival = vandq_u8(mask_sum_2, mask_current_alive);
                    
                    // Combined result: Birth OR Survival (0xFF/0x00 mask)
                    uint8x16_t result_mask = vorrq_u8(mask_birth, mask_survival);
                    
                    // 4. 转换结果 (0xFF -> 1, 0x00 -> 0) 并存储
                    uint8x16_t next_state = vshrq_n_u8(result_mask, 7);
                    
                    uint8_t* output_ptr = next_flat.data() + y * padded_W + x;
                    vst1q_u8(output_ptr, next_state);

                    // 5. **边界更新修正 (恢复到块级更新以提高速度)**
                    uint8x8_t high = vget_high_u8(next_state);
                    uint8x8_t low = vget_low_u8(next_state);
                    
                    // 检查是否存在非零值（即是否存在活细胞）
                    if (vmaxv_u8(vmax_u8(high, low)) == 1) { 
                        
                        // Y 边界是准确的
                        if (y < thread_min_y) thread_min_y = y;
                        if (y > thread_max_y) thread_max_y = y;

                        // X 边界：使用 std::min/max 确保边界框能正确收缩或扩展，而不是简单赋值
                        thread_min_x = std::min(thread_min_x, x);
                        thread_max_x = std::max(thread_max_x, x + NEON_SIZE - 1);
                    }
                }
            }


            // -----------------------------------------------------------------------
            // 3. 标量处理：剩余尾部和右边界 (包括 NEON 无法对齐的单元和右侧填充列)
            // -----------------------------------------------------------------------
            for (; x < padded_W; ++x) { // 确保覆盖到最右侧的填充列 (x = padded_W - 1)
                
                // 当前单元在输入缓冲区的 X 坐标 (如果超出边界，则为 0)
                int current_read_x = x + in_min_x - 1;

                // [Neighbors]
                int live_neighbors = 0;
                for (int i = -1; i <= 1; ++i) {
                    for (int j = -1; j <= 1; ++j) {
                        if (i == 0 && j == 0) continue;

                        int read_neighbor_y = y + i + in_min_y - 1;
                        int read_neighbor_x = x + j + in_min_x - 1;

                        // 显式边界检查，防止越界读
                        if (read_neighbor_y >= 0 && read_neighbor_y < in_padded_H &&
                            read_neighbor_x >= 0 && read_neighbor_x < in_padded_W) {
                            live_neighbors += current_flat[read_neighbor_y * in_padded_W + read_neighbor_x];
                        }
                    }
                }

                // [Get State]
                int current_state = 0;
                // 显式边界检查，防止越界读
                if (current_read_y_base >= 0 && current_read_y_base < in_padded_H && current_read_x >= 0 && current_read_x < in_padded_W) {
                    current_state = current_flat[current_read_y_base * in_padded_W + current_read_x];
                }

                // [Apply Rules & Write Result]
                bool is_alive_next = (live_neighbors == 3) || (live_neighbors == 2 && current_state == 1);

                if (is_alive_next) {
                    next_flat[y * padded_W + x] = 1;
                    if (y < thread_min_y) thread_min_y = y;
                    if (y > thread_max_y) thread_max_y = y;
                    if (x < thread_min_x) thread_min_x = x;
                    if (x > thread_max_x) thread_max_x = x;
                }
            } // End x loop
        } // End OpenMP y loop
        
        // 线程局部边界合并到 OpenMP 约简变量
        #pragma omp critical
        {
            min_y = std::min(min_y, thread_min_y);
            max_y = std::max(max_y, thread_max_y);
            min_x = std::min(min_x, thread_min_x);
            max_x = std::max(max_x, thread_max_x);
        }
    } // --- OMP 并行区域结束 ---

    if (max_y == -1) {
        return {-1, -1, -1, -1}; // 灭绝信号
    }

    // 返回新的边界 (相对于 H+2 x W+2 网格)
    return {min_y, max_y, min_x, max_x};
}


// ------------------------------------------------------------------------------------------------
// 主迭代函数 (保持不变)
// ------------------------------------------------------------------------------------------------
std::vector<std::vector<int>> expand_cpp(
    const std::vector<std::vector<int>>& initial_grid,
    int generations) {

    if (initial_grid.empty() || initial_grid[0].empty()) {
        return {};
    }

    int H = initial_grid.size();
    int W = initial_grid[0].size();

    // --- [1. 初始分配 & 转换 (仅一次)] ---
    int initial_padded_H = H + 2;
    int initial_padded_W = W + 2;
    size_t initial_size = (size_t)initial_padded_H * initial_padded_W;

    // 缓冲区改为 uint8_t
    FlatBuffer buffer_A(initial_size, 0);
    FlatBuffer buffer_B(initial_size, 0);

    // 初始数据从 int 转换为 uint8_t
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (initial_grid[y][x] == 1) {
                buffer_A[(y + 1) * initial_padded_W + (x + 1)] = 1;
            }
        }
    }

    FlatBuffer* current_buf = &buffer_A;
    FlatBuffer* next_buf = &buffer_B;

    InternalResult prev_bounds = {1, H, 1, W}; // 初始数据在 (1,1)
    InternalResult current_bounds = {-1, -1, -1, -1};
    int current_H = H;
    int current_W = W;

    int in_min_y = 1;
    int in_min_x = 1;
    int in_padded_H = initial_padded_H;
    int in_padded_W = initial_padded_W;

    for (int gen = 0; gen < generations; ++gen) {

        // --- [2. 检查缓冲区容量并按需扩展] ---
        int current_padded_H = current_H + 2;
        int current_padded_W = current_W + 2;
        size_t required_size = (size_t)current_padded_H * current_padded_W;

        if (buffer_A.capacity() < required_size) {
            // 注意：因为是 uint8_t，不需要乘以 sizeof(int)
            size_t new_cap = required_size + (size_t)(RESIZE_SLACK + 2) * current_padded_W;
            buffer_A.reserve(new_cap);
        }
        if (buffer_B.capacity() < required_size) {
            size_t new_cap = required_size + (size_t)(RESIZE_SLACK + 2) * current_padded_W;
            buffer_B.reserve(new_cap);
        }

        size_t last_required_size = (size_t)in_padded_H * in_padded_W;
        if (current_buf->size() < last_required_size) {
            current_buf->resize(last_required_size, 0); // 用 0 填充
        }

        // --- [3. 调用内部计算函数] ---
        current_bounds = Next_Generation_Internal(*current_buf, *next_buf,
            current_H, current_W,
            in_min_y, in_min_x,
            in_padded_H, in_padded_W
        );

        // --- [4. 检查灭绝] ---
        if (std::get<0>(current_bounds) == -1) {
            current_H = 0;
            current_W = 0;
            prev_bounds = {-1,-1,-1,-1}; // 标记为灭绝
            break;
        }

        // --- [5. 稳定性检查] ---
        int min_y, max_y, min_x, max_x;
        std::tie(min_y, max_y, min_x, max_x) = current_bounds;
        int new_H = max_y - min_y + 1;
        int new_W = max_x - min_x + 1;

        bool stable = false;

        int prev_H = (std::get<0>(prev_bounds) == -1) ? 0 : (std::get<1>(prev_bounds) - std::get<0>(prev_bounds) + 1);
        int prev_W = (std::get<0>(prev_bounds) == -1) ? 0 : (std::get<3>(prev_bounds) - std::get<2>(prev_bounds) + 1);

        if (new_H == prev_H && new_W == prev_W) {
            // 尺寸相同，现在比较内容
            std::atomic<bool> content_differs(false);

            int current_read_padded_W = in_padded_W;
            int next_read_padded_W = current_padded_W;

            int prev_min_y = std::get<0>(prev_bounds);
            int prev_min_x = std::get<2>(prev_bounds);

            #pragma omp parallel for schedule(static)
            for (int y = 0; y < new_H; ++y) {
                if (content_differs.load(std::memory_order_relaxed)) continue;

                // current_buf (上一代) 数据在 (prev_min_y, prev_min_x)
                size_t current_start_idx = (size_t)(y + prev_min_y) * current_read_padded_W + prev_min_x;

                // next_buf (当前代) 数据在 (min_y, min_x)
                size_t next_start_idx = (size_t)(y + min_y) * next_read_padded_W + min_x;

                if (current_start_idx + new_W > current_buf->size() ||
                    next_start_idx + new_W > next_buf->size())
                {
                    content_differs.store(true, std::memory_order_relaxed);
                    continue;
                }

                // 逐行比较: 使用 sizeof(uint8_t)
                if (std::memcmp(&(*current_buf)[current_start_idx],
                                 &(*next_buf)[next_start_idx],
                                 (size_t)new_W * sizeof(uint8_t)) != 0)
                {
                    content_differs.store(true, std::memory_order_relaxed);
                }
            }
            if (!content_differs.load()) {
                stable = true;
            }
        }

        // --- [6. 更新状态] ---
        int H_for_next_read = current_padded_H;
        int W_for_next_read = current_padded_W;

        prev_bounds = current_bounds;
        current_H = new_H;
        current_W = new_W;

        in_min_y = min_y;
        in_min_x = min_x;
        in_padded_H = H_for_next_read;
        in_padded_W = W_for_next_read;


        // --- [7. Swap Buffers] ---
        std::swap(current_buf, next_buf);

        if (stable) break; // 如果稳定，退出循环

    } // end generation loop

    // --- [8. 最终转换 (仅一次): uint8_t -> int] ---
    int final_H = current_H;
    int final_W = current_W;
    std::vector<std::vector<int>> final_grid_2d;

    if (final_H > 0 && final_W > 0 && std::get<0>(prev_bounds) != -1) {
        final_grid_2d.resize(final_H, std::vector<int>(final_W));

        int final_min_y, final_min_x;
        std::tie(final_min_y, std::ignore, final_min_x, std::ignore) = prev_bounds;
        int final_padded_W = in_padded_W;

        // 【关键修正】由于类型不同 (uint8_t -> int)，不能使用 memcpy。
        // 使用 OpenMP 并行循环进行安全转换和拷贝。
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < final_H; ++y) {
            size_t start_index = (size_t)(final_min_y + y) * final_padded_W + final_min_x;
            if(start_index + final_W <= current_buf->size()){
                for (int x = 0; x < final_W; ++x) {
                    // 显式转换：uint8_t -> int
                    final_grid_2d[y][x] = (int)(*current_buf)[start_index + x];
                }
            } else {
                   // 理论上不应发生，但用于安全检查
                   // 由于在并行区域内无法安全修改 final_grid_2d 的大小或状态，
                   // 我们在这里只打印错误或依赖主线程的最终大小检查。
            }
        }
    }

    return final_grid_2d;
}

// ------------------------------------------------------------------------------------------------
// Next_Generation_Wrapper (可选 Python 接口, 保持不变)
// ------------------------------------------------------------------------------------------------
GridResult Next_Generation_Wrapper(const std::vector<std::vector<int>>& current_grid) {
    if (current_grid.empty() || current_grid[0].empty()) {
        return {{}, {0, 0}};
    }
    int H = current_grid.size();
    int W = current_grid[0].size();
    int padded_H = H + 2;
    int padded_W = W + 2;
    size_t required_size = (size_t)padded_H * padded_W;

    // 缓冲区改为 uint8_t
    FlatBuffer current_flat(required_size, 0);
    FlatBuffer next_flat(required_size, 0);

    // int -> uint8_t 转换
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            // int 赋值给 uint8_t
            if (current_grid[y][x] == 1) {
                current_flat[(y + 1) * padded_W + (x + 1)] = 1;
            }
        }
    }

    // 调用 Internal 函数
    InternalResult bounds = Next_Generation_Internal(
        current_flat, next_flat, H, W,
        1, 1, padded_H, padded_W
    );

    if (std::get<0>(bounds) == -1) {
        return {{}, {0, 0}};
    }

    int min_y, max_y, min_x, max_x;
    std::tie(min_y, max_y, min_x, max_x) = bounds;
    int new_H = max_y - min_y + 1;
    int new_W = max_x - min_x + 1;
    std::vector<std::vector<int>> new_grid(new_H, std::vector<int>(new_W));

    int next_flat_padded_W = W + 2;

    // uint8_t -> int 转换 (使用循环)
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < new_H; ++y) {
        int start_index = (min_y + y) * next_flat_padded_W + min_x;
        for (int x = 0; x < new_W; ++x) {
            // 显式转换：uint8_t -> int
            new_grid[y][x] = (int)next_flat[start_index + x];
        }
    }

    int dy = min_y - 1;
    int dx = min_x - 1;

    return {std::move(new_grid), {dy, dx}};
}


// Pybind11 模块绑定
PYBIND11_MODULE(NG, m) {
    m.def("Expand_Cpp", &expand_cpp,
          "Simulate multiple generations using flat uint8_t buffers internally (Optimized Naive).",
          py::arg("initial_grid"), py::arg("generations"));

    m.def("Next_Generation_Cpp", &Next_Generation_Wrapper,
          "Calculates the next single generation (Wrapper for internal uint8_t flat buffer logic).",
          py::arg("current_grid"));
}
