cmake_minimum_required(VERSION 3.10)
project(CSRConverter CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(EXPORT_COMPILE_COMMANDS ON)

# 包含头文件目录
include_directories(include)



# 查找OpenMP包
find_package(OpenMP REQUIRED)
# 包含头文件目录
include_directories(include)

# 源文件
file(GLOB SOURCES "src/*.cpp" "*.cpp")

# 创建可执行文件
add_executable(spmm ${SOURCES})

# 创建可执行文件
# add_executable(spmm ${SOURCES})

target_compile_options(spmm PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>: -O3 -ffast-math -march=native -mtune=native -Wall -g -Wextra>
)
target_link_libraries(spmm PRIVATE OpenMP::OpenMP_CXX)

