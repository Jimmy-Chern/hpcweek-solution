#pragma once
#include "Chunk.h"
#include "LevelTicks.h"
#include "LightTick.h"
#include <string>
#include <random>
#include "BlockBehaviour.h"
#include "ChunkCoord.h"
#include <mutex> // 1. 引入 mutex
#include <unordered_map>

class Dimension {
public:
    Dimension(const std::string &name) : blockTicks(LevelTicks()), random(std::random_device()()), name(name) {}

    ~Dimension() {}

    // 获取区块数据，不存在则创建
    Chunk* getChunk(const ChunkCoord& coord) {
        // 2. 加锁保护：保护 chunks 容器结构
        std::lock_guard<std::mutex> lock(chunks_mutex_);
        auto it = chunks.find(coord);
        if (it != chunks.end()) {
            return &it->second;
        } else {
            auto [new_it, inserted] = chunks.try_emplace(coord);
            return &new_it->second;
        }
    }

    // 维度的 tick，包含光照和方块行为
    void tick() {
        processTicks();
        // 注意：lightTick(chunks) 应该处理线程安全问题，
        // 这里只是保持原有代码结构。如果 chunks 在 lightTick 中被修改，也需要锁。
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
        // 2. 加锁保护：保护 chunks 容器结构
        // **注意：getChunk/setBlockID 已经在内部加锁，这里为了完整性也保持加锁。**
        std::lock_guard<std::mutex> lock(chunks_mutex_);
        auto chunkIt = chunks.find(chunkCoord);
        if (chunkIt == chunks.end()) {
            auto [it, inserted] = chunks.try_emplace(chunkCoord);
            if (inserted) {
                return 0;
            }
            chunkIt = it;
        }

        int relX = pos.x & 0xF;
        int relY = pos.y;
        int relZ = pos.z & 0xF;

        Chunk& chunk = chunkIt->second;
        // 注意：Chunk::getBlockID 是读操作，我们依赖 Chunk.h 中 getBlockID/setBlockID 
        // 内部或外部的锁来实现线程安全写入。这里只保护了对 `chunks` 容器的访问。
        return chunk.getBlockID(relX, relY, relZ);
    }

    // 设置指定位置的方块 ID
    void setBlockID(const BlockPos& pos, int blockID) {
        ChunkCoord chunkCoord = ChunkCoord::fromBlockPos(pos);
        // 2. 加锁保护：保护 chunks 容器结构和写入
        std::lock_guard<std::mutex> lock(chunks_mutex_);
        
        auto chunkIt = chunks.find(chunkCoord);
        if (chunkIt == chunks.end()) {
            auto [it, inserted] = chunks.try_emplace(chunkCoord);
            chunkIt = it;
        }
        int relX = pos.x & 0xF;
        int relY = pos.y;
        int relZ = pos.z & 0xF;

        Chunk& chunk = chunkIt->second;
        // setBlockID 是写操作，必须保护。
        // chunks_mutex_ 锁保护了对 chunk.setBlockID 的并发调用。
        chunk.setBlockID(relX, relY, relZ, blockID);
    }

    LevelTicks blockTicks;
    std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> chunks;
private:
    int chunkCount = 0;
    std::string name;
    std::mt19937 random;
    // 3. 新增成员：保护 chunks 容器和 setBlockID 写入
    std::mutex chunks_mutex_; 
};