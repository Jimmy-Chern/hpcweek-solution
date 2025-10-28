#pragma once
#include <string>
#include <unordered_map>
#include "BlockInfo.h"
#include <iostream>

class BlockRegistry {
public:
    // 注册方块信息，stage 表示有多少个状态，比如水有 16 个状态
    void registerBlock(int id, const BlockInfo& info, int stage = 1) {
        for (int i = 0; i < stage; ++i) {
            BlockInfo newInfo = info;
            newInfo.stage = i % 8;
            registry[id + i] = newInfo;
        }
    }

    // 获取方块信息
    const BlockInfo* getBlockInfo(int id) const {
        auto it = registry.find(id);
        if (it != registry.end()) {
            return &it->second;
        }
        return &defaultBlockInfo;
    }
private:
    std::unordered_map<int, BlockInfo> registry;
    // 默认方块信息
    BlockInfo defaultBlockInfo = BlockInfo(
        "Default Block",
        BlockInfo::VisualProperties(0, 15),
        BlockInfo::InteractionProperties(true, false)
    );

};

extern BlockRegistry globalBlockRegistry;



