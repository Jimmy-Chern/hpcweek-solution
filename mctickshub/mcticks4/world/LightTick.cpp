#include "LightTick.h"
#include "Block.h"
#include "Chunk.h"        // 引入 Chunk 定义
#include "ChunkCoord.h"   // 引入 ChunkCoord 定义
#include <queue>
#include <vector>
#include <tuple>
#include <mutex>
#include <omp.h>
#include <unordered_map>  // 引入 std::unordered_map 定义
#include <cstdio>
#include <utility>        // 引入 std::pair

// 假设 BlockInfo, globalBlockRegistry 等在 Block.h 中定义

// 移除全局互斥锁参数，现在使用线程局部的 add_chunks
void floodFillLight_parallel(std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> &chunks,
    int chunkX, int chunkZ, // 注意：原代码的 chunkY 应该是 chunkZ
    int blockX, int blockY, int blockZ,
    unsigned char lightLevel,
    // 改变：现在接收的是当前线程的局部 add_chunks 引用
    std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> &local_add_chunks) {

    // 修复：原代码中的 chunkY 应为 chunkZ
    std::queue<std::tuple<int, int, int, int, int, unsigned char, bool>> toVisit;
    toVisit.push({chunkX, chunkZ, blockX, blockY, blockZ, lightLevel, true});
    bool source = true;

    while (!toVisit.empty()){
        auto [cX, cZ, bX, bY, bZ, level, fromAbove] = toVisit.front();
        toVisit.pop();

        if (level <= 0) continue;
        if (bX < 0) { bX = 15; cX -= 1; }
        if (bX >= 16) { bX = 0; cX += 1; }
        if (bY < -64 || bY >= 320) continue;
        if (bZ < 0) { bZ = 15; cZ -= 1; }
        if (bZ >= 16) { bZ = 0; cZ += 1; }

        ChunkCoord coord{cX, cZ};

        Chunk *chunk = nullptr;

        auto it_chunks = chunks.find(coord);
        if (it_chunks == chunks.end()){

            // 改变：不再使用全局锁，直接在线程局部的 map 中查找和插入
            auto it_add = local_add_chunks.find(coord);
            if (it_add == local_add_chunks.end()) {
                // 使用 piecewise_construct 进行原地构造，避免复制
                it_add = local_add_chunks.emplace(std::piecewise_construct,
                                                 std::forward_as_tuple(coord),
                                                 std::forward_as_tuple()).first;
            }
            chunk = &it_add->second;

        } else {
            chunk = &it_chunks->second;
        }

        int block_type = chunk->getBlockID(bX, bY, bZ);
        const BlockInfo* info = globalBlockRegistry.getBlockInfo(block_type);

        if (info && info->visualProps.lightOpacity == 15 && !fromAbove) continue;

        // 注意：setLightLevelAtomic_SetMax 内部使用了原子操作，无需额外的锁
        bool setOccurred = chunk->setLightLevelAtomic_SetMax(bX, bY, bZ, level);

        if (!setOccurred && !source) continue;

        if (info && info->visualProps.lightOpacity == 15 && !source) continue;

        source = false;
        toVisit.push({cX, cZ, bX, bY, bZ - 1, level - 1, false});
        toVisit.push({cX, cZ, bX, bY, bZ + 1, level - 1, false});
        toVisit.push({cX, cZ, bX, bY - 1, bZ, level - 1, true});
        toVisit.push({cX, cZ, bX + 1, bY, bZ, level - 1, false});
        toVisit.push({cX, cZ, bX - 1, bY, bZ, level - 1, false});
        toVisit.push({cX, cZ, bX, bY + 1, bZ, level - 1, false});
    }
}

void lightTick(std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> &chunks) {
    // ---------------------- 计时器开始 ----------------------
    double start_time = omp_get_wtime();
    // ------------------------------------------------------

    // 改变：使用一个 vector 存储线程局部的 add_chunks
    int num_threads = omp_get_max_threads();
    std::vector<std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash>> add_chunks_storage(num_threads);

    // --- 修复：将 map 转换为 vector ---
    std::vector<std::pair<ChunkCoord, Chunk*>> chunk_vec;
    chunk_vec.reserve(chunks.size());
    for (auto& pair : chunks) {
        chunk_vec.emplace_back(pair.first, &pair.second);
    }

    // --- 1. 并行清空光照 ---
    #pragma omp parallel for
    for (size_t i = 0; i < chunk_vec.size(); ++i) {
        chunk_vec[i].second->clearLightData();
    }

    // --- 2. 并行收集所有光源 (优化: 避免 critical) ---
    // 改变：使用 vector of vector 来收集局部光源，最后串行合并（比 critical 更快）
    std::vector<std::vector<std::tuple<int, int, int, int, int, unsigned char>>> all_localSources(num_threads);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        std::vector<std::tuple<int, int, int, int, int, unsigned char>>& localSources = all_localSources[thread_id];

        #pragma omp for schedule(guided) nowait
        for (size_t i = 0; i < chunk_vec.size(); ++i) {
            int chunk_x = chunk_vec[i].first.x;
            int chunk_z = chunk_vec[i].first.z;
            Chunk &chunk = *chunk_vec[i].second;

            for (int x = 0; x < 16; ++x){
                for (int y = -64; y < 320; ++y){
                    for (int z = 0; z < 16; ++z){
                        int block_type = chunk.getBlockID(x, y, z);
                        // 注意：这里假设 globalBlockRegistry.getBlockInfo(block_type) 是线程安全的
                        unsigned char emission = globalBlockRegistry.getBlockInfo(block_type)->visualProps.lightEmission;
                        if (emission > 0) {
                            localSources.emplace_back(chunk_x, chunk_z, x, y, z, emission);
                        }
                    }
                }
            }
        }
    } // 显式屏障在 parallel 块结束时自动发生

    // 串行合并局部光源（比使用 #pragma omp critical 快得多）
    std::vector<std::tuple<int, int, int, int, int, unsigned char>> lightSources;
    for (auto& localSources : all_localSources) {
        lightSources.insert(lightSources.end(), 
                            std::make_move_iterator(localSources.begin()), 
                            std::make_move_iterator(localSources.end()));
    }

    // --- 3. 并行执行 Flood Fill ---
    // 保持 guided 调度，以平衡不同光源的光照传播工作量
    #pragma omp parallel for schedule(dynamic, 10)
    for (size_t k = 0; k < lightSources.size(); ++k) {
        // 获取当前线程 ID，用于访问线程局部的 map
        int thread_id = omp_get_thread_num();
        std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> &local_add_chunks = add_chunks_storage[thread_id];

        auto [i, j, x, y, z, emission] = lightSources[k];
        // 改变：传递局部 map
        floodFillLight_parallel(chunks, i, j, x, y, z, emission, local_add_chunks);
    }

    // --- 4. 串行合并新区块 ---
    for (int i = 0; i < num_threads; ++i) {
        // 使用 merge 合并，比遍历插入高效
        chunks.merge(add_chunks_storage[i]);
    }

    // ---------------------- 计时器结束 ----------------------
    double end_time = omp_get_wtime();
    double elapsed_seconds = end_time - start_time;

    // 转换为毫秒 (ms)
    double elapsed_milliseconds = elapsed_seconds * 1000.0;

    // 使用 printf 输出结果，精确到毫秒 (ms)
    printf("[PROFILING] lightTick finished in %.3f ms (%.6f s).\n", elapsed_milliseconds, elapsed_seconds);
    // ------------------------------------------------------
}
