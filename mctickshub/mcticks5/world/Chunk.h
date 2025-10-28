#pragma once
#include <cstring>
#include <cstdint>
#include <atomic> // 1. 引入 atomic

extern int MAX_CHUNKS;

// Chunk 的 NBT 结构定义
extern "C"{
typedef struct {
  int16_t blockcount;
  int32_t blocks_state[4096];
  int32_t biomes[64];

  uint8_t  sky_light[2048];
  // 2. 改为 atomic 数组
  std::atomic<uint8_t> block_light[2048];
} Section;
typedef struct {
    int32_t last_update;
    Section sections[24];
} BrChunk; 
}

class Chunk {
public:

    // 修复：使用构造函数初始化列表 brchunk{} 来进行零初始化
    // 这会避免调用被删除的赋值操作符
    Chunk() : brchunk{} { 
        // 之前错误的 memset 和 brchunk = {} 都已移除

        // set default sky light to max
        for (int i = 0; i < 24; ++i) {
            std::memset(brchunk.sections[i].sky_light, 0xFF, sizeof(brchunk.sections[i].sky_light));
            
            // brchunk{} 已经将所有 block_light 初始化为 0
        }
    }

    ~Chunk() {
    }

    inline int getBlockID(int x, int y, int z) {
        if (y < -64 || y >= 320) {
            return 0;
        }
        int sectionY = getSectionY(y);
        int localY = y + 64 - sectionY * 16;
        int localIndex = (localY * 16 * 16) + (z * 16) + x;
        return brchunk.sections[sectionY].blocks_state[localIndex];
    }

    inline void setBlockID(int x, int y, int z, int state_id) {
        if (y < -64 || y >= 320) {
            return ;
        }
        int sectionY = getSectionY(y);
        int localY = y + 64 - sectionY * 16;
        int localIndex = (localY * 16 * 16) + (z * 16) + x;
        brchunk.sections[sectionY].blocks_state[localIndex] = state_id;
    }

    inline unsigned char getLightLevel(int x, int y, int z) const {
        if (y < -64 || y >= 320) {
            return 0;
        }
        int sectionY = getSectionY(y);
        int localY = y + 64 - sectionY * 16;
        int localIndex = (localY * 16 * 16) + (z * 16) + x;
        // 4. 使用 load() 来原子读取
        unsigned char currentByte = brchunk.sections[sectionY].block_light[localIndex / 2].load(std::memory_order_relaxed) ;
        return (localIndex % 2 == 0) ? (currentByte & 0x0F) : (currentByte >> 4);
    }

// *** 为 LightTick.cpp 添加缺失的非原子 setLightLevel ***
    inline void setLightLevel(int x, int y, int z, unsigned char level) {
        if (y < -64 || y >= 320) {
            return ;
        }
        int sectionY = getSectionY(y);
        int localY = y + 64 - sectionY * 16;
        int localIndex = (localY * 16 * 16) + (z * 16) + x;
        int byteIndex = localIndex / 2;

        // 获取对应的原子字节
        std::atomic<uint8_t> &atomicByte = brchunk.sections[sectionY].block_light[byteIndex];
        
        // 因为增量算法是串行的，我们可以安全地执行 "读-改-写"
        // 我们使用 relaxed 内存顺序，因为我们不需要线程间的同步
        unsigned char currentByte = atomicByte.load(std::memory_order_relaxed);

        if (localIndex % 2 == 0) { // 低 4 位
            unsigned char newByte = (currentByte & 0xF0) | (level & 0x0F);
            atomicByte.store(newByte, std::memory_order_relaxed);
        } else { // 高 4 位
            unsigned char newByte = (currentByte & 0x0F) | ((level & 0x0F) << 4);
            atomicByte.store(newByte, std::memory_order_relaxed);
        }
    }
    // 6. 添加新的原子 "set-if-max" 函数
    inline bool setLightLevelAtomic_SetMax(int x, int y, int z, unsigned char level) {
        if (y < -64 || y >= 320) return false;

        int sectionY = getSectionY(y);
        int localY = y + 64 - sectionY * 16;
        int localIndex = (localY * 16 * 16) + (z * 16) + x;
        int byteIndex = localIndex / 2;
        
        std::atomic<uint8_t> &currentByte = brchunk.sections[sectionY].block_light[byteIndex];
        unsigned char oldByte = currentByte.load(std::memory_order_relaxed);
        unsigned char newByte;

        if (localIndex % 2 == 0) { // 低 4 位
            unsigned char oldLevel = oldByte & 0x0F;
            if (oldLevel >= level) return false;
            
            do {
                if ((oldByte & 0x0F) >= level) return false;
                newByte = (oldByte & 0xF0) | (level & 0x0F);
            } while (!currentByte.compare_exchange_weak(oldByte, newByte, std::memory_order_release, std::memory_order_relaxed));
        } else { // 高 4 位
            unsigned char oldLevel = oldByte >> 4;
            if (oldLevel >= level) return false;
            
            do {
                if ((oldByte >> 4) >= level) return false;
                newByte = (oldByte & 0x0F) | ((level & 0x0F) << 4);
            } while (!currentByte.compare_exchange_weak(oldByte, newByte, std::memory_order_release, std::memory_order_relaxed));
        }
        return true;
    }


    void clearLightData() {
        for (int i = 0; i < 24; ++i) {
             // 7. 使用循环和 store 来清空
            for (int j = 0; j < 2048; ++j) {
                brchunk.sections[i].block_light[j].store(0, std::memory_order_relaxed);
            }
        }
    }
    
    BrChunk brchunk;
private:
    
    int getSectionY(int y) const {
        return (y + 64) / 16;
    }

    int getIndex(int x, int y, int z) const {
        int localY = y + 64 - getSectionY(y) * 16;
        return (localY * 16 * 16) + (z * 16) + x;
    }

    friend class Bridging;

};

