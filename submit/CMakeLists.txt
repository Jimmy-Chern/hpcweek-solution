cmake_minimum_required(VERSION 3.10)
project(mc_tick)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 确保 C++17 被严格使用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- 添加 OpenMP 查找 ---

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
message(STATUS "Found OpenMP, enabling parallel build.")
else()
message(WARNING "OpenMP not found. Build will be single-threaded.")
endif()

# --- OpenMP 结束 ---

include_directories(${CMAKE_SOURCE_DIR}/world ${CMAKE_SOURCE_DIR}/tick ${CMAKE_SOURCE_DIR}/util ${CMAKE_SOURCE_DIR}/player)

if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-unused-private-field") # -fsanitize=address
set(CMAKE_CXX_FLAGS_DEBUG "-g -O3 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

message(STATUS "${CMAKE_BUILD_TYPE}")

# (改进：使用 CMAKE_SOURCE_DIR 保证路径正确)

FILE(GLOB WORLS_SRC
"${CMAKE_SOURCE_DIR}/world/*.cpp"
)

FILE(GLOB TICK_SRC
"${CMAKE_SOURCE_DIR}/tick/*.cpp"
)

FILE(GLOB UTIL_SRC
"${CMAKE_SOURCE_DIR}/util/*.cpp"
)

FILE(GLOB TEST_SRC
"${CMAKE_SOURCE_DIR}/test/*.cpp"
)

add_library(mc_tick_lib SHARED
${WORLS_SRC}
${TICK_SRC}
${UTIL_SRC}
)

# --- 将 OpenMP 链接到你的库 ---

if(OpenMP_FOUND)
target_link_libraries(mc_tick_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

# --- 链接结束 ---

foreach(test_file ${TEST_SRC})
get_filename_component(test_name ${test_file} NAME_WE)
add_executable(${test_name} ${test_file} ${WORLS_SRC} ${TICK_SRC} ${UTIL_SRC})

# --- 也要链接到测试程序 ---
if(OpenMP_FOUND)
    target_link_libraries(${test_name} PUBLIC OpenMP::OpenMP_CXX)
endif()
# --- 链接结束 ---

endforeach()
