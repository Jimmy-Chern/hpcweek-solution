# Personal Solution Write-up for the 1st ZJU HPC Week Competition

This repository contains my personal solution write-up for the 1st High-Performance Computing (HPC) Week competition held at Zhejiang University (ZJU). My solution achieved the 8th rank on the online judge (OJ).

## Key Optimization Techniques:

The primary performance gains came from the following areas:

1.  **Minecraft Lighting Simulation:** Utilized **OpenMP** and **atomic locks** to accelerate the lighting rendering calculations.
2.  **Conway's Game of Life:**
    * Switched the memory storage strategy from loading a new state vector each iteration to a **double-buffering** approach, **reusing two blocks of contiguous memory** alternately.
    * Implemented **low-precision calculation** using `int8` data type.
    * Applied **OpenMP** for further parallel acceleration.
3.  **Sparse Matrix-Matrix Multiplication (SPMM) with CSR Format:**
    * Accelerated memory access by employing an algorithm where each **single element of matrix A multiplies an entire row of matrix B** (often referred to as a row-wise block approach), leveraging spatial locality.
4.  **Other General Optimizations (Omitted for brevity):**
    * General optimizations include leveraging **SIMD/NEON** for vectorization (e.g., in CT reconstruction) and applying techniques like **SVE** to accelerate dot-product computations in $\text{Q}_4 / \text{Q}_8$ quantization inference.
## Core Project Solutions / Code Structure

This repository contains optimization solutions for the ZJU HPC Week competition problems. The core optimized files for each task are listed below, along with the cluster execution commands used by the author.

| Task | Core Optimized Files | Cluster Execution Method (Author's Reference) | Notes |
| :--- | :--- | :--- | :--- |
| **1. Conway's Game of Life** | `hpcweek-conway/src/NG.cpp` <br> `hpcweek-conway/src/conway.py` | `cd /hpcweek/home/s3240106190/hpcweek1/hpcweek-conway` <br> `&& bash instr2` | Uses `instr2` to launch the job on the SLURM cluster. |
| **2. Parallel Compression** | `parrallelCompression/compress.c` | *(No execution script provided)* | Core file for the parallel compression implementation. |
| **3. CT Reconstruction** | `hpcweek/home/s3240106190/hpcweek1/hpcweek-ct/src/fbp.cpp` | *(Execution method embedded in cluster scripts)* | Uses FBP (Filtered Back Projection) algorithm. |
| **4. SPMM (Sparse Matrix-Matrix Multiplication)** | `hpcweek1/HPCWeek_SPMM/src/spmm_opt.cpp` | *(Execution method embedded in cluster scripts)* | Optimized for SPMM, likely utilizing sparse matrix formats and parallelism. |
| **5. RVLLM (Quantization)** | `hpcweek/home/s3240106190/hpcweek1/rvllm/src/qmatnul.c` | `cd ~/hpcweek1/rvllm` <br> `&& bash run` | Optimization for quantized matrix multiplication in the LLM context. |
| **6. Minecraft Lighting** | `hpcweek/home/s3240106190/hpcweek1/mctickshub/mcticks2/world` | `cd ~/hpcweek1/mcticks` <br> `&& bash clean_and_judge_mcticks2` | Optimization related to lighting tick mechanics in the Minecraft simulation. |

### Note on Execution

The execution commands listed above are specific references to the author's setup on the cluster (using SLURM job submission scripts like `instr2` or `run`).

**If you do not have the same environment:** Please refer to the source code and the provided shell scripts for the underlying compilation and running logic, which can be adapted for your local environment.

## Challenges and Failed Optimization Attempts

During the optimization process, several approaches were attempted but resulted in either performance regressions or were not fully debugged due to time constraints:

1.  **Conway's Game of Life (NEON Vectorization):**
    * An attempt was made to use NEON instructions to process the state of 16 cells simultaneously. This effort resulted in **negative optimization (performance regression)**, indicating that the overhead introduced by vectorization likely outweighed the computational benefits in this specific context.

2.  **Minecraft Lighting Simulation (Incremental Algorithm):**
    * The classic incremental lighting algorithm was implemented. However, due to time constraints, the implementation could not be fully debugged and validated. The semi-finished work related to this attempt is located in `mcticks/world`.

3.  **RVLLM (Tiled Matrix Multiplication for Quantization):**
    * An alternative approach using tiled (blocked) matrix multiplication was explored within the `ggml_compute_forward_mul_mat_one_chunk` function.
    * **Experimentation showed that performance immediately degraded upon columnar blocking of matrix A.**
    * This observation suggests that **memory access locality** (maintaining contiguous memory access) was a more critical factor for performance than optimizing the computation-to-memory-access ratio (C/M ratio). (The current dot-product algorithm has a C/M ratio of $1/2$, while the tiled approach *theoretically* aimed for a higher ratio like $M \times N / (M + N)$, but suffered due to poor spatial locality caused by frequent non-contiguous row accesses).
