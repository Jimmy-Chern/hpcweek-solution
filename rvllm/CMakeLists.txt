cmake_minimum_required(VERSION 3.16)
project(HPCWeek_rvLLM)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(GGML_DIR /hpcweek/rvllm/ggml)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GGML_DIR}/include
    ${GGML_DIR}/src
    ${GGML_DIR}/src/ggml-cpu
)

add_library(qmatmul SHARED
    src/qmatmul.c src/mul_mat_compute.c src/type_traits.c
)

target_compile_options(
    qmatmul PRIVATE
    -std=gnu11
    -fopenmp
    -Ofast           # 保留 Ofast
    # -flto          # 禁用 LTO (因为它会触发编译器 Bug)
    -fno-finite-math-only 
    -march=rv64gcv
    -mabi=lp64d
    -Wno-unused-function
    -Wall
    -Wextra
    -fPIC
    -ftree-vectorize

)

# --- 修正后的链接选项 ---
target_link_options(
    qmatmul PRIVATE
    
    # 链接 OpenMP 库
    -lgomp
    
    # 确保运行时能找到 ggml 库
    -Wl,-rpath,${GGML_DIR}/lib
    
    # LTO 需要的 march 标志 (即使 LTO 被注释掉，保留它也无害)
    -march=rv64gcv
    
    # --- **这就是修复方案** ---
    # 告诉 GCC 动态链接 libgcc (libgcc_s.so)
    # 而不是静态链接 (libgcc.a), 以避免符号冲突。
    -shared-libgcc
)
