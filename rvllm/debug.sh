#!/bin/bash

# ==============================================================================
# Slurm 配置 (SBATCH directives)
# ==============================================================================
#
#SBATCH --job-name=rvllm-run     # 任务名称
#SBATCH --partition=riscv        # ** 必须在 riscv 分区上运行 **
#SBATCH --nodes=1                # ** 必须使用 1 个节点 **
#SBATCH --ntasks=1               # 在这个节点上只运行 1 个任务
#SBATCH --cpus-per-task=8        # ** 必须申请 8 个 CPU 核心 **
#SBATCH --time=00:05:00          # ** 任务超时时间 5 分钟 **
#
#SBATCH --output=slurm-%j.out    # 将标准输出重定向到 slurm-[JobId].out
#SBATCH --error=slurm-%j.err     # 将标准错误重定向到 slurm-[JobId].err
#
# ==============================================================================

# --- 打印作业信息 ---
echo "==============================================================="
echo "作业 ID (Job ID): $SLURM_JOB_ID"
echo "运行节点 (Node): $SLURM_JOB_NODELIST"
echo "开始时间 (Start Time): $(date)"
echo "==============================================================="


# --- 步骤 1: 编译 ---

echo ">>> 步骤 1: 正在编译 libqmatmul.so..."

# 清理并创建 build 目录
rm -rf build
mkdir -p build

# 运行 CMake 配置
cmake -S . -B build

# 并行编译
cmake --build build -v -j 8

# 检查编译是否成功
if [ ! -f "./build/lib/libqmatmul.so" ]; then
    echo "!!! 编译失败! 未找到 ./build/lib/libqmatmul.so"
    echo ">>> 请检查 slurm-[JobId].err 文件中的详细编译日志。"
    exit 1
fi

echo ">>> 编译成功!"


# --- 步骤 2: 运行 ---

echo ">>> 步骤 2: 运行 llama.cpp 性能测试..."

# 设置 OMP_NUM_THREADS 和 LD_LIBRARY_PATH (保持不变)
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="/hpcweek/rvllm/llama.cpp/bin:$LD_LIBRARY_PATH"

# 定义路径
LLAMA_EXECUTABLE="/hpcweek/rvllm/llama.cpp/bin/llama-cli"

# --- **关键修改** ---
# 使用你找到的真实文件名！
MODEL_PATH="/hpcweek/rvllm/model/deepseek-r1-distill-qwen-1.5b-q4_0.gguf"
# --- -------------- ---


# 检查可执行文件是否存在
if [ ! -f "$LLAMA_EXECUTABLE" ]; then
    echo "!!! 运行失败! 未找到 $LLAMA_EXECUTABLE"
    exit 1
fi

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "!!! 运行失败! 模型文件未找到: $MODEL_PATH"
    echo ">>> 请确保 MODEL_PATH 变量中的文件名正确无误！"
    exit 1
fi

# 运行 (移除了 --perf 和 --log-disable)
LD_PRELOAD=./build/lib/libqmatmul.so \
$LLAMA_EXECUTABLE \
    -m $MODEL_PATH \
    -p "Once upon a time, in a land far, far away, there" \
    -n 265 \
    -t 8

echo "==============================================================="
echo "作业完成 (Job Finished): $(date)"
echo "==============================================================="
