#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=conway_job     # 任务名称
#SBATCH --partition=armv82      # 提交到 armv82 分区 (根据 sinfo, kp[02-07] 在此分区)
#SBATCH --nodes=1               # 申请 1 个节点
#SBATCH --ntasks=1              # 在该节点上运行 1 个任务
#SBATCH --cpus-per-task=8       # 为这个任务申请 8 个 CPU 核心 (Slurm 会自动设置 OMP_NUM_THREADS=8)
#SBATCH --time=00:05:00         # 任务时间上限 (10 分钟，对于 100 次迭代应该足够)
#SBATCH --output=slurm-%j.out   # 标准输出重定向到 slurm-[Job_ID].out
#SBATCH --error=slurm-%j.err    # 标准错误重定向到 slurm-[Job_ID].err

# 确保脚本在任何命令失败时立即退出
set -e

echo "=========================================================="
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_NODELIST"
echo "Running on partition: $SLURM_JOB_PARTITION"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Working directory: $(pwd)"
echo "=========================================================="

# --- 1. 设置环境 ---
echo "Setting up environment..."
# 切换到 hpcweek-conway 根目录
# $HOME 变量会由 Slurm 自动解析为您的主目录
cd /hpcweek/home/s3240106190/hpcweek-conway

# 激活 Python 虚拟环境 (myenv)
if [ -f "myenv/bin/activate" ]; then
    source myenv/bin/activate
    echo "Activated Python environment: $(which python)"
else
    echo "Error: Virtual environment 'myenv' not found in $(pwd)!"
    exit 1
fi

# --- 2. 编译和安装 ---
echo "Cleaning and building C++ extension..."

# (推荐) 先清理旧的构建产物
python setup.py clean --all

# 运行您在 instr2 中提供的所有构建命令
python setup.py bdist_wheel
python setup.py build
python setup.py build_ext
python setup.py install
python setup.py install_lib

echo "Build and install complete."

# --- 3. 运行主程序 ---
echo "Running main program..."
cd src

# 运行您的程序
python main.py -F ../data/sample2.in -I 100

echo "=========================================================="
echo "Job finished successfully."
echo "=========================================================="
