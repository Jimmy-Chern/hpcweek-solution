#include "LightTick.h"
#include "Block.h"
#include "Chunk.h"
#include "ChunkCoord.h"
#include "BlockBehaviour.h"
#include "Tuple.h"
#include <queue>
#include <vector>
#include <tuple>
#include <algorithm> // for std::max
#include <cmath>
#include <unordered_map>
#include <omp.h>
#include <utility>
#include <cstddef>
#include <stdexcept>
#include <atomic>
#include <cstdio> // 用于调试

// --- 全局变量 ---
extern BlockRegistry globalBlockRegistry;
extern BlockStateRegistry globalBlockStateRegistry;
std::queue<std::tuple<int, int, int, int, int, unsigned char>> g_lightAddQueue;
std::queue<std::tuple<int, int, int, int, int, unsigned char>> g_lightRemoveQueue;
std::atomic_bool g_isWorldInitialized = false;

//
// =======================================================================
// PART 1: 旧的、并行的、用于 Tick 0 初始化的代码 (最大限度还原原始逻辑)
// =======================================================================
//

// (Helper for parallel init - 完全还原原始逻辑)
void floodFillLight_parallel(std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> &chunks,
    int chunkX, int chunkZ, int blockX, int blockY, int blockZ, unsigned char lightLevel,
    std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> &local_add_chunks) {

    std::queue<std::tuple<int, int, int, int, int, unsigned char, bool>> toVisit;
    toVisit.push({chunkX, chunkZ, blockX, blockY, blockZ, lightLevel, true});
    bool source = true;

    while (!toVisit.empty()){
        auto [cX, cZ, bX, bY, bZ, level, fromAbove] = toVisit.front();
        toVisit.pop();
        if (level <= 0) continue;
        if (bX < 0) { bX = 15; cX -= 1; } if (bX >= 16) { bX = 0; cX += 1; }
        if (bY < -64 || bY >= 320) continue;
        if (bZ < 0) { bZ = 15; cZ -= 1; } if (bZ >= 16) { bZ = 0; cZ += 1; }
        ChunkCoord coord{cX, cZ};
        Chunk *chunk = nullptr;
        auto it_chunks = chunks.find(coord);
        if (it_chunks == chunks.end()){
            auto it_add = local_add_chunks.find(coord);
            if (it_add == local_add_chunks.end()) {
                it_add = local_add_chunks.emplace(std::piecewise_construct, std::forward_as_tuple(coord), std::forward_as_tuple()).first;
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
        unsigned char nextLevel_Queue = level - 1;
        if (nextLevel_Queue > 0) {
            toVisit.push({cX, cZ, bX, bY, bZ - 1, nextLevel_Queue, false});
            toVisit.push({cX, cZ, bX, bY, bZ + 1, nextLevel_Queue, false});
            toVisit.push({cX, cZ, bX, bY - 1, bZ, nextLevel_Queue, true});
            toVisit.push({cX, cZ, bX + 1, bY, bZ, nextLevel_Queue, false});
            toVisit.push({cX, cZ, bX - 1, bY, bZ, nextLevel_Queue, false});
            toVisit.push({cX, cZ, bX, bY + 1, bZ, nextLevel_Queue, false});
        }
    }
}


// Tick 0 并行初始化 (*** 还原 OMP Critical 合并方式 ***)
void lightTick_Parallel_Initialize(std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> &chunks) {
    int num_threads = omp_get_max_threads();
    std::vector<std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash>> add_chunks_storage(num_threads);
    std::vector<std::pair<ChunkCoord, Chunk*>> chunk_vec;
    chunk_vec.reserve(chunks.size());
    for (auto& pair : chunks) { chunk_vec.emplace_back(pair.first, &pair.second); }

    #pragma omp parallel for
    for (size_t i = 0; i < chunk_vec.size(); ++i) { chunk_vec[i].second->clearLightData(); }

    // *** 修改点 1：定义一个共享的 lightSources (像原始版本一样) ***
    std::vector<std::tuple<int, int, int, int, int, unsigned char>> lightSources;
    // (不再需要 vector of vector: all_localSources)

    #pragma omp parallel
    {
        // *** 修改点 2：在并行区域内定义局部的 localSources ***
        std::vector<std::tuple<int, int, int, int, int, unsigned char>> localSources;

        #pragma omp for schedule(guided) nowait
        for (size_t i = 0; i < chunk_vec.size(); ++i) {
            int chunk_x = chunk_vec[i].first.x; int chunk_z = chunk_vec[i].first.z;
            Chunk &chunk = *chunk_vec[i].second;
            for (int x = 0; x < 16; ++x) for (int y = -64; y < 320; ++y) for (int z = 0; z < 16; ++z) {
                int block_type = chunk.getBlockID(x, y, z);
                // *** 还原原始版本不安全的 getBlockInfo 调用 ***
                // (如果这导致崩溃，说明原始版本本身就有 Bug)
                // const BlockInfo* info = globalBlockRegistry.getBlockInfo(block_type);
                // unsigned char emission = 0;
                // if (info) { emission = info->visualProps.lightEmission; }
                unsigned char emission = globalBlockRegistry.getBlockInfo(block_type)->visualProps.lightEmission; // 可能崩溃！
                if (emission > 0) {
                    localSources.emplace_back(chunk_x, chunk_z, x, y, z, emission);
                }
            }
        }

        // *** 修改点 3：使用 OMP Critical 合并 localSources 到共享的 lightSources ***
        #pragma omp critical
        lightSources.insert(lightSources.end(), localSources.begin(), localSources.end());

    } // OMP barrier (隐式)

    // (不再需要串行合并步骤)

    #pragma omp parallel for schedule(guided)
    for (size_t k = 0; k < lightSources.size(); ++k) {
        int thread_id = omp_get_thread_num();
        auto& local_add_chunks = add_chunks_storage[thread_id];
        auto [i, j, x, y, z, emission] = lightSources[k];
        floodFillLight_parallel(chunks, i, j, x, y, z, emission, local_add_chunks);
    }

    for (int i = 0; i < num_threads; ++i) { chunks.merge(add_chunks_storage[i]); }
}

// ... (PART 2 和 PART 3 保持不变) ...

//
// =======================================================================
// PART 2: 新的、增量的、用于 Tick 1+ 的代码 (保持不变)
// =======================================================================
//
const BlockInfo* getBlockInfo(std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> &chunks, int cX, int cZ, int bX, int bY, int bZ) { /* ... same ... */
    if (bX < 0) { bX = 15; cX -= 1; } if (bX >= 16) { bX = 0; cX += 1; } if (bZ < 0) { bZ = 15; cZ -= 1; } if (bZ >= 16) { bZ = 0; cZ += 1; } if (bY < -64 || bY >= 320) return nullptr; ChunkCoord coord{cX, cZ}; auto it = chunks.find(coord); if (it == chunks.end()) return nullptr; int block_type = it->second.getBlockID(bX, bY, bZ); return globalBlockRegistry.getBlockInfo(block_type);
}
unsigned char getLightLevel(std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> &chunks, int cX, int cZ, int bX, int bY, int bZ) { /* ... same ... */
    if (bX < 0) { bX = 15; cX -= 1; } if (bX >= 16) { bX = 0; cX += 1; } if (bZ < 0) { bZ = 15; cZ -= 1; } if (bZ >= 16) { bZ = 0; cZ += 1; } if (bY < -64 || bY >= 320) return 0; ChunkCoord coord{cX, cZ}; auto it = chunks.find(coord); if (it == chunks.end()) return 0; return it->second.getLightLevel(bX, bY, bZ);
}
void setLightLevel(std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> &chunks, int cX, int cZ, int bX, int bY, int bZ, unsigned char level) { /* ... same ... */
    if (bX < 0) { bX = 15; cX -= 1; } if (bX >= 16) { bX = 0; cX += 1; } if (bZ < 0) { bZ = 15; cZ -= 1; } if (bZ >= 16) { bZ = 0; cZ += 1; } if (bY < -64 || bY >= 320) return; ChunkCoord coord{cX, cZ}; auto it = chunks.find(coord); if (it != chunks.end()) { it->second.setLightLevel(bX, bY, bZ, level); }
}
void propagateAdd(std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> &chunks) { /* ... same ... */
    std::queue<std::tuple<int, int, int, int, int, unsigned char>> toVisit = std::move(g_lightAddQueue); int processed_count = 0;
    while (!toVisit.empty()){ processed_count++; auto [cX, cZ, bX, bY, bZ, level] = toVisit.front(); toVisit.pop(); if (level <= 0) continue; int neighbors[6][3] = { { 1, 0, 0}, {-1, 0, 0}, { 0, 1, 0}, { 0,-1, 0}, { 0, 0, 1}, { 0, 0,-1} }; for (int i = 0; i < 6; ++i) { int nX = bX + neighbors[i][0], nY = bY + neighbors[i][1], nZ = bZ + neighbors[i][2]; int nCX = cX, nCZ = cZ; if (nX < 0) { nX = 15; nCX -= 1; } if (nX >= 16) { nX = 0; nCX += 1; } if (nZ < 0) { nZ = 15; nCZ -= 1; } if (nZ >= 16) { nZ = 0; nCZ += 1; } if (nY < -64 || nY >= 320) continue; const BlockInfo* info = getBlockInfo(chunks, nCX, nCZ, nX, nY, nZ); unsigned char lightOpacity = info ? info->visualProps.lightOpacity : 0; unsigned char decay = std::max((unsigned char)1, lightOpacity); unsigned char newLevel = (unsigned char)std::max(0, (int)level - (int)decay); if (newLevel <= 0) continue; unsigned char currentLevel = getLightLevel(chunks, nCX, nCZ, nX, nY, nZ); if (newLevel > currentLevel) { setLightLevel(chunks, nCX, nCZ, nX, nY, nZ, newLevel); toVisit.push({nCX, nCZ, nX, nY, nZ, newLevel}); } } }
}
void propagateRemove(std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> &chunks) { /* ... same ... */
    std::queue<std::tuple<int, int, int, int, int, unsigned char>> toVisit = std::move(g_lightRemoveQueue); int processed_count = 0;
    while (!toVisit.empty()) { processed_count++; auto [cX, cZ, bX, bY, bZ, oldLevel] = toVisit.front(); toVisit.pop(); if (bY < -64 || bY >= 320) continue; unsigned char currentLevel = getLightLevel(chunks, cX, cZ, bX, bY, bZ); if (currentLevel != oldLevel) { continue; } unsigned char maxNeighborLevel = 0; int neighbors[6][3] = { { 1, 0, 0}, {-1, 0, 0}, { 0, 1, 0}, { 0,-1, 0}, { 0, 0, 1}, { 0, 0,-1} }; const BlockInfo* selfInfo = getBlockInfo(chunks, cX, cZ, bX, bY, bZ); if (selfInfo) { maxNeighborLevel = std::max(maxNeighborLevel, (unsigned char)selfInfo->visualProps.lightEmission); } for (int i = 0; i < 6; ++i) { int nX = bX + neighbors[i][0], nY = bY + neighbors[i][1], nZ = bZ + neighbors[i][2]; int nCX = cX, nCZ = cZ; if (nX < 0) { nX = 15; nCX -= 1; } if (nX >= 16) { nX = 0; nCX += 1; } if (nZ < 0) { nZ = 15; nCZ -= 1; } if (nZ >= 16) { nZ = 0; nCZ += 1; } if (nY < -64 || nY >= 320) continue; unsigned char neighborLight = getLightLevel(chunks, nCX, nCZ, nX, nY, nZ); unsigned char lightOpacity = selfInfo ? selfInfo->visualProps.lightOpacity : 0; unsigned char decay = std::max((unsigned char)1, lightOpacity); unsigned char attenuatedLight = (unsigned char)std::max(0, (int)neighborLight - (int)decay); maxNeighborLevel = std::max(maxNeighborLevel, attenuatedLight); } if (maxNeighborLevel < currentLevel) { setLightLevel(chunks, cX, cZ, bX, bY, bZ, maxNeighborLevel); for (int i = 0; i < 6; ++i) { int nX = bX + neighbors[i][0], nY = bY + neighbors[i][1], nZ = bZ + neighbors[i][2]; int nCX = cX, nCZ = cZ; if (nX < 0) { nX = 15; nCX -= 1; } if (nX >= 16) { nX = 0; nCX += 1; } if (nZ < 0) { nZ = 15; nCZ -= 1; } if (nZ >= 16) { nZ = 0; nCZ += 1; } if (nY < -64 || nY >= 320) continue; unsigned char neighborLight = getLightLevel(chunks, nCX, nCZ, nX, nY, nZ); if (neighborLight > 0) { toVisit.push({nCX, nCZ, nX, nY, nZ, neighborLight}); } } } else if (maxNeighborLevel > currentLevel) { g_lightAddQueue.push({cX, cZ, bX, bY, bZ, maxNeighborLevel}); } }
}
void lightTick_Incremental_Update(std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> &chunks) { /* ... same ... */
    propagateRemove(chunks); propagateAdd(chunks);
}
//
// =======================================================================
// PART 3: 最终的 "路由器" 函数和外部接口 (保持不变)
// =======================================================================
//
void lightTick(std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> &chunks) { /* ... same ... */
    if (!g_isWorldInitialized.exchange(true)) {
        std::queue<std::tuple<int, int, int, int, int, unsigned char>> emptyAdd; std::swap(g_lightAddQueue, emptyAdd);
        std::queue<std::tuple<int, int, int, int, int, unsigned char>> emptyRemove; std::swap(g_lightRemoveQueue, emptyRemove);
        lightTick_Parallel_Initialize(chunks);
    } else {
        lightTick_Incremental_Update(chunks);
    }
}
void addLightChange(const BlockPos& pos, int oldLightEmission, int newLightEmission) { /* ... same ... */
    int cX = pos.x >> 4; int cZ = pos.z >> 4; int bX = pos.x & 0xF; int bY = pos.y; int bZ = pos.z & 0xF;
    if (oldLightEmission > 0 && oldLightEmission >= newLightEmission) {
        g_lightRemoveQueue.push({cX, cZ, bX, bY, bZ, (unsigned char)oldLightEmission});
    }
    if (newLightEmission > 0 && newLightEmission > oldLightEmission) {
        g_lightAddQueue.push({cX, cZ, bX, bY, bZ, (unsigned char)newLightEmission});
    }
}
