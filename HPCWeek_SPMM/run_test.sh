#!/bin/bash
set -e
./build/spmm -n 20480 -f ./data/orani678.mtx -t 10
./build/spmm -n 4096 -f ./data/psmigr_1.mtx -t 10
./build/spmm -n 2048 -f ./data/psmigr_2_block_0_0.mtx -t 10
./build/spmm -n 10240 -f ./data/psmigr_3_block_0_0.mtx -t 10
./build/spmm -n 4096 -f ./data/heart1.mtx -t 10
./build/spmm -m 10240 -n 10240 -k 128 -s 0.95 -t 10
./build/spmm -m 8196 -n 8196 -k 1024 -s 0.95 -t 10
