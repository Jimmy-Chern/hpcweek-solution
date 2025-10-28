#pragma once
#include "Tuple.h"
#include <random>
#include <unordered_map>
#include "LevelTicks.h"
class Dimension;

// 方块行为接口，赛题目前实现了水和岩浆的行为
class BlockBehaviour {
public:
    BlockBehaviour() {}
    
    virtual ~BlockBehaviour() = default;
    
    virtual void tick(Dimension* dimension, const BlockPos& pos, LevelTicks &ticks) {

    }
    
    virtual void randomTick(Dimension* dimension, const BlockPos& pos, std::mt19937& random, LevelTicks &ticks) {

    }
};


// 根据方块类型映射对应的方块行为。
class BlockStateRegistry {
public:

    void registerBlockState(int blockType, BlockBehaviour* behaviour) {
        registry_[blockType] = behaviour;
    }
    
    BlockBehaviour* getBlockStateBehaviour(int blockType) {
        auto it = registry_.find(blockType);
        if (it != registry_.end()) {
            return it->second;
        }
        return defaultBehaviour;
    }
private:
    std::unordered_map<int, BlockBehaviour*> registry_;
    BlockBehaviour* defaultBehaviour = new BlockBehaviour(); // 默认返回空气方块状态
};

extern BlockStateRegistry globalBlockStateRegistry;