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
