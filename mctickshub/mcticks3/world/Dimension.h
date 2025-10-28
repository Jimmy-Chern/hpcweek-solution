#pragma once
#include "Chunk.h"
#include "LevelTicks.h"
#include "LightTick.h"
#include <string>
#include <random>
#include "BlockBehaviour.h"
#include "ChunkCoord.h"
#include <shared_mutex> // 1. 引入 shared_mutex
#include <unordered_map>

// 假设 BlockPos, globalBlockStateRegistry, lightTick 等在此处或引用的头文件中定义

class Dimension {
public:
    // 构造函数：为避免 -Wreorder 警告，这里调整了 name 和 random 的初始化顺序
    Dimension(const std::string &name) 
        : blockTicks(LevelTicks()), random(std::random_device()()), name(name) {}

    ~Dimension() {}

    // 获取区块数据，不存在则创建
    Chunk* getChunk(const ChunkCoord& coord) {
        // 2. 加锁保护：因为可能对 chunks 容器执行 try_emplace (写操作)，所以使用 unique_lock (独占锁)
        std::unique_lock<std::shared_mutex> lock(chunks_mutex_);
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
        // 这里只是保持原有代码结构。
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
        // 2. 加锁保护：使用 shared_lock (共享锁) 允许多个线程同时读取 chunks
        std::shared_lock<std::shared_mutex> lock(chunks_mutex_);
        
        auto chunkIt = chunks.find(chunkCoord);
        if (chunkIt == chunks.end()) {
            // 注意：这里是读取操作，我们不允许在共享锁下修改容器。
            // 默认返回 0，除非您愿意引入复杂的锁升级逻辑或假设区块已加载。
            return 0; 
        }

        int relX = pos.x & 0xF;
        int relY = pos.y;
        int relZ = pos.z & 0xF;

        Chunk& chunk = chunkIt->second;
        // Chunk::getBlockID 是读操作，现在 chunks 容器的查找已安全。
        return chunk.getBlockID(relX, relY, relZ);
    }

    // 设置指定位置的方块 ID
    void setBlockID(const BlockPos& pos, int blockID) {
        ChunkCoord chunkCoord = ChunkCoord::fromBlockPos(pos);
        // 2. 加锁保护：因为可能对 chunks 容器执行 try_emplace (写操作) 或修改 Chunk，所以使用 unique_lock (独占锁)
        std::unique_lock<std::shared_mutex> lock(chunks_mutex_);
        
        auto chunkIt = chunks.find(chunkCoord);
        if (chunkIt == chunks.end()) {
            auto [it, inserted] = chunks.try_emplace(chunkCoord);
            chunkIt = it;
        }
        int relX = pos.x & 0xF;
        int relY = pos.y;
        int relZ = pos.z & 0xF;

        Chunk& chunk = chunkIt->second;
        // setBlockID 是写操作，必须在独占锁内执行。
        chunk.setBlockID(relX, relY, relZ, blockID);
    }

    LevelTicks blockTicks;
    std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> chunks;
private:
    int chunkCount = 0;
    // 为避免 -Wreorder 警告，将 random 声明放在 name 之前 (可选)
    std::mt19937 random;
    std::string name;
    // 3. 替换为共享互斥锁：保护 chunks 容器和 setBlockID 写入
    std::shared_mutex chunks_mutex_; 
};
