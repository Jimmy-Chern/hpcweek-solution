#!/bin/bash

# --- Slurm 任务配置 ---
#SBATCH --job-name=CT_FBP_Opt     # 任务名称：CT滤波反投影优化
#SBATCH --partition=armv82        # 必需：指定运行分区为 armv82 (对应鲲鹏 920)
#SBATCH --nodes=1                 # 申请 1 个节点
#SBATCH --ntasks=1                # 本脚本总共只执行 1 个任务
#SBATCH --cpus-per-task=32        # 核心设置：申请 32 个 CPU 核心给该任务
#SBATCH --time=00:1:00           # 任务运行的时间上限：1 分钟足够
#SBATCH --output=/hpcweek/home/s3240106190/hpcweek-ct/slurm-%j.out     # 任务结果输出文件 (使用 JobID 命名)
#SBATCH --error=/hpcweek/home/s3240106190/hpcweek-ct/slurm-%j.err      # 任务报错信息输出文件

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
rm -rf build 
mkdir build
cmake -B build
cmake --build build -- -j $SLURM_CPUS_PER_TASK


mkdir -p recon_out

echo "--- Starting CT Reconstruction (Running with $OMP_NUM_THREADS threads) ---"

START_TIME=$(date +%s)
./build/ct_recon ./input/sinogram.hdf5
END_TIME=$(date +%s)

ELAPSED_TIME=$((END_TIME - START_TIME))
echo "--- CT Reconstruction Finished ---"
echo "Elapsed Time: $ELAPSED_TIME seconds"
echo "Reconstructed images saved in recon_out/"
echo "--------------------------------------"

