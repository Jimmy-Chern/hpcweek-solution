#pragma once
#include "Chunk.h"
#include "LevelTicks.h"
#include "LightTick.h"
#include <string>
#include <random>
#include "BlockBehaviour.h"
#include "ChunkCoord.h"

class Dimension {
public:
    Dimension(const std::string &name) : blockTicks(LevelTicks()), random(std::random_device()()) {}

    ~Dimension() {}

    // 获取区块数据，不存在则创建
    Chunk* getChunk(const ChunkCoord& coord) {
        auto it = chunks.find(coord);
        if (it != chunks.end()) {
            return &it->second;
        } else {
            // 修复：使用 try_emplace 替代赋值操作
            // chunks[coord] = Chunk();
            // return &chunks[coord];
            auto [new_it, inserted] = chunks.try_emplace(coord);
            return &new_it->second;
        }
    }

    // 维度的 tick，包含光照和方块行为
    void tick() {
        processTicks();
        lightTick(chunks);
    }

    // 处理所有计划刻
    void processTicks() {
        blockTicks.tick([this](BlockPos pos, LevelTicks &ticks) { this->tickBlock(pos, ticks);},
                        [this](BlockPos pos, int blockID) { this->tickSetBlockID(pos, blockID);});
    }

    // 统一执行 GO 发起的 SetBlock 得到的计划刻
    void setBlockTick() {
        blockTicks.setBlockTick([this](BlockPos pos, LevelTicks &ticks) { this->tickBlock(pos, ticks);},
                                [this](BlockPos pos, int blockID) { this->tickSetBlockID(pos, blockID);});
    }

    // 在所有计划刻之前的预处理，即设置方块 ID，不让方块的顺序影响行为
    void tickSetBlockID(const BlockPos& pos, int blockID) {
        setBlockID(pos, blockID);
    }

    // 执行方块的行为逻辑
    void tickBlock(BlockPos pos, LevelTicks &ticks) {
        int blockType = getBlockID(pos);
        BlockBehaviour* behaviour = globalBlockStateRegistry.getBlockStateBehaviour(blockType);
        behaviour->tick(this, pos, ticks);
        behaviour->randomTick(this, pos, random, ticks);
    }

    // 获取指定位置的方块 ID
    int getBlockID(const BlockPos& pos) {
        ChunkCoord chunkCoord = ChunkCoord::fromBlockPos(pos);
        auto chunkIt = chunks.find(chunkCoord);
        if (chunkIt == chunks.end()) {
            // 修复：使用 try_emplace 替代赋值操作
            // chunks[chunkCoord] = Chunk();
            // chunkIt = chunks.find(chunkCoord);
            // return 0;
            
            // 新逻辑：尝试创建。如果创建成功 (inserted == true)，说明它之前不存在，返回 0。
            // 这保留了原版代码"创建了但返回0"的逻辑。
            auto [it, inserted] = chunks.try_emplace(chunkCoord);
            if (inserted) {
                return 0;
            }
            // 如果没 inserted，说明在 try_emplace 之前它已经存在 (虽然 find 没找到，可能被其他线程创建)
            // 理论上如果 find 失败, inserted 应该是 true。
            // 但为了安全起见，我们假设 try_emplace 总是能正确处理竞态。
            // 如果插入，我们返回0。
            chunkIt = it;
            // 如果 `inserted` 为 false (即竞态下被其他线程插入)，我们应该继续执行下面的代码。
            // 因此，我们只在 `inserted` 为 true 时返回 0。
        }

        int relX = pos.x & 0xF;
        int relY = pos.y;
        int relZ = pos.z & 0xF;

        Chunk& chunk = chunkIt->second;
        return chunk.getBlockID(relX, relY, relZ);
    }

    // 设置指定位置的方块 ID
    void setBlockID(const BlockPos& pos, int blockID) {
        ChunkCoord chunkCoord = ChunkCoord::fromBlockPos(pos);
        auto chunkIt = chunks.find(chunkCoord);
        if (chunkIt == chunks.end()) {
            // 修复：使用 try_emplace 替代赋值操作
            // chunks[chunkCoord] = Chunk();
            // chunkIt = chunks.find(chunkCoord);
            auto [it, inserted] = chunks.try_emplace(chunkCoord);
            chunkIt = it;
        }
        int relX = pos.x & 0xF;
        int relY = pos.y;
        int relZ = pos.z & 0xF;

        Chunk& chunk = chunkIt->second;
        chunk.setBlockID(relX, relY, relZ, blockID);
    }

    LevelTicks blockTicks;
    std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> chunks;
private:
    int chunkCount = 0;
    std::string name;
    std::mt19937 random;
};
