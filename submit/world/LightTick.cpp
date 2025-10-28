#include "LightTick.h"
#include "Block.h"
#include "Chunk.h"
#include "ChunkCoord.h"
#include <queue>
#include <vector>     // 1. 引入
#include <tuple>      // 2. 引入 (为了 piecewise_construct)
#include <mutex>      // 3. 引入
#include <omp.h>      // 4. 引入 (需要修改 CMakeLists.txt 来启用)

void floodFillLight_parallel(std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> &chunks, 
    int chunkX, int chunkY,
    int blockX, int blockY, int blockZ, 
    unsigned char lightLevel, 
    std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> &add_chunks,
    std::mutex &add_chunks_mutex) { 
        
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
            std::lock_guard<std::mutex> lock(add_chunks_mutex);
            auto it_add = add_chunks.find(coord);
            if (it_add == add_chunks.end()) {
                // (修复) 使用 piecewise_construct 进行原地构造，避免复制
                it_add = add_chunks.emplace(std::piecewise_construct,
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
    std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> add_chunks;
    
    // --- 修复：将 map 转换为 vector ---
    // 这个 vector 将用于两个并行的 for 循环
    std::vector<std::pair<ChunkCoord, Chunk*>> chunk_vec;
    chunk_vec.reserve(chunks.size());
    for (auto& pair : chunks) {
        chunk_vec.emplace_back(pair.first, &pair.second);
    }
    
    // --- 1. 并行清空光照 ---
    // (现在迭代 vector)
    #pragma omp parallel for
    for (size_t i = 0; i < chunk_vec.size(); ++i) { 
        chunk_vec[i].second->clearLightData();
    }

    // --- 2. 并行收集所有光源 ---
    std::vector<std::tuple<int, int, int, int, int, unsigned char>> lightSources;
    
    #pragma omp parallel
    {
        std::vector<std::tuple<int, int, int, int, int, unsigned char>> localSources;
        
        // --- 修复：并行迭代 vector, 而不是 map ---
        #pragma omp for
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
    std::mutex add_chunks_mutex;
    
    #pragma omp parallel for
    for (size_t k = 0; k < lightSources.size(); ++k) { 
        auto [i, j, x, y, z, emission] = lightSources[k];
        floodFillLight_parallel(chunks, i, j, x, y, z, emission, add_chunks, add_chunks_mutex);
    }

    // --- 4. 串行合并新区块 ---
    // 修复：使用 merge() 替代 insert() 来移动元素，避免复制
    chunks.merge(add_chunks);
    add_chunks.clear();
}


