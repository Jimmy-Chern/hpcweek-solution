//___________________________________________________

#include "LightTick.h"
#include "Block.h"
#include "Chunk.h"
#include "ChunkCoord.h"
#include <queue>
#include <vector>      // 1. 引入
#include <tuple>       // 2. 引入 (为了 piecewise_construct)
#include <mutex>       // 3. 引入 (虽然全局锁被移除，但仍保留，因为可能会用于其他地方)
#include <omp.h>       // 4. 引入 (需要修改 CMakeLists.txt 来启用)
#include <unordered_map>

// 移除全局互斥锁参数，现在使用线程局部的 add_chunks
void floodFillLight_parallel(std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> &chunks,
    int chunkX, int chunkY,
    int blockX, int blockY, int blockZ,
    unsigned char lightLevel,
    // 改变：现在接收的是当前线程的局部 add_chunks 引用
    std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> &local_add_chunks) {

    std::queue<std::tuple<int, int, int, int, int, unsigned char, bool>> toVisit;
    toVisit.push({chunkX, chunkY, blockX, blockY, blockZ, lightLevel, true});
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
                // (修复) 使用 piecewise_construct 进行原地构造，避免复制
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

    // --- 2. 并行收集所有光源 ---
    std::vector<std::tuple<int, int, int, int, int, unsigned char>> lightSources;

    #pragma omp parallel
    {
        std::vector<std::tuple<int, int, int, int, int, unsigned char>> localSources;

        #pragma omp for schedule(guided) nowait
        for (size_t i = 0; i < chunk_vec.size(); ++i) {
            int chunk_x = chunk_vec[i].first.x;
            int chunk_z = chunk_vec[i].first.z;
            Chunk &chunk = *chunk_vec[i].second;

            for (int x = 0; x < 16; ++x){
                for (int y = -64; y < 320; ++y){
                    for (int z = 0; z < 16; ++z){
                        int block_type = chunk.getBlockID(x, y, z);
                        unsigned char emission = globalBlockRegistry.getBlockInfo(block_type)->visualProps.lightEmission;
                        if (emission > 0) {
                            localSources.emplace_back(chunk_x, chunk_z, x, y, z, emission);
                        }
                    }
                }
            }
        }

        #pragma omp critical
        lightSources.insert(lightSources.end(), localSources.begin(), localSources.end());
    }

    // --- 3. 并行执行 Flood Fill ---
    // 改变：不再需要全局互斥锁
    // std::mutex add_chunks_mutex;

    #pragma omp parallel for schedule(guided)
    for (size_t k = 0; k < lightSources.size(); ++k) {
        // 获取当前线程 ID，用于访问线程局部的 map
        int thread_id = omp_get_thread_num();
        std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> &local_add_chunks = add_chunks_storage[thread_id];

        auto [i, j, x, y, z, emission] = lightSources[k];
        // 改变：传递局部 map
        floodFillLight_parallel(chunks, i, j, x, y, z, emission, local_add_chunks);
    }

    // --- 4. 串行合并新区块 ---
    // 改变：串行地将所有线程局部 map 合并到全局 chunks 中
    for (int i = 0; i < num_threads; ++i) {
        chunks.merge(add_chunks_storage[i]);
        // add_chunks_storage[i].clear() 在 merge 后已为空
    }
}


