#pragma once
#include <queue>
#include <functional>
#include <vector>
#include <unordered_set>
#include "Tuple.h"
#include <algorithm>
struct ChunkCoord;
#include <mutex> // 1. 引入 mutex

// 计划刻的结构体
class ScheduledTick {
public:
    ScheduledTick(BlockPos pos, int type, long triggerTick)
        : pos_(pos), type_(type), triggerTick_(triggerTick) {}

    int type() const { return type_; }
    void setType(int t) { type_ = t; }

    BlockPos pos() const { return pos_; }
    long triggerTick() const { return triggerTick_; }

    bool operator==(const ScheduledTick& other) const {
        return pos_ == other.pos_ && type_ == other.type_ && triggerTick_ == other.triggerTick_;
    }

    struct Hash {
        size_t operator()(const ScheduledTick& tick) const {
            return BlockPos::Hash()(tick.pos_) << 26 ^ tick.type_ ^ (tick.triggerTick_ << 12);
        }
    };

    bool operator>(const ScheduledTick& other) const {
        return triggerTick_ > other.triggerTick_;
    }

private:
    BlockPos pos_;
    int type_;
    long triggerTick_;
};


class LevelTicks {
public:

    LevelTicks() {
        toRunThisTick_.clear();
        while(!scheduledTicks_.empty()) scheduledTicks_.pop();
        scheduledSet_.clear();
        gameTime = 0;
    }

    void clearTicks() {
        toRunThisTick_.clear();
        while(!scheduledTicks_.empty()) scheduledTicks_.pop();
        scheduledSet_.clear();
        gameTime = 0;
    }

    // 执行 GO 发起的计划刻，这个是在整一个 tick 的计划刻执行完之后进行的，属于同一个 gameTime
    void setBlockTick(std::function<void(BlockPos, LevelTicks&)> consumer, std::function<void(BlockPos, int)> setBlockID) {
        collectTicks();
        runCollectedTicks(consumer, setBlockID);
        cleanupAfterTick();
    }

    // 执行计划刻
    void tick(std::function<void(BlockPos, LevelTicks&)> consumer, std::function<void(BlockPos, int)> setBlockID) {
        gameTime ++;
        collectTicks();
        runCollectedTicks(consumer, setBlockID);
        cleanupAfterTick();
    }

    // 添加一个计划刻，有去重的逻辑
    void addTick(int type, const BlockPos& pos, long delay) {
        // 3. 加锁保护：保护 scheduledTicks_ 和 scheduledSet_ 的写入
        std::lock_guard<std::mutex> lock(tick_mutex_);
        
        long triggerTick = gameTime + (long)std::max(0L, delay);
        if (scheduledSet_.find(ScheduledTick(pos, type, triggerTick)) != scheduledSet_.end()) {
            return;
        }
        ScheduledTick newTick(pos, type, triggerTick);
        scheduledTicks_.push(newTick);
        scheduledSet_.insert(newTick);
    }

private:
    long gameTime = 0;
    // 2. 新增成员：保护 scheduledTicks_ 和 scheduledSet_ 的写入
    std::mutex tick_mutex_; 
    
    std::vector<ScheduledTick> toRunThisTick_;
    std::priority_queue<ScheduledTick, std::vector<ScheduledTick>, std::greater<ScheduledTick>> scheduledTicks_;
    std::unordered_set<ScheduledTick, ScheduledTick::Hash> scheduledSet_;

    void collectTicks() {
        // 注意：collectTicks 只在主线程（tick/setBlockTick）中调用，不需要锁。
        while (!scheduledTicks_.empty()) {
            // 需要在 pop/top 之前加锁吗？
            // 答：由于 addTick 是唯一的多线程入口，它有锁保护。
            // collectTicks 在主线程中调用，可以假设它独占访问。
            // 如果 collectTicks 可能被其他线程调用，则需要加锁。
            const ScheduledTick nextTick = scheduledTicks_.top();
            if (nextTick.triggerTick() > gameTime) {
                break;
            }
            toRunThisTick_.emplace_back(nextTick);
            scheduledTicks_.pop();
        }
    }

    // 优先级排序，位置相同的只执行第一个，优先级顺序是 实体方块 > 岩浆消失 > 岩浆流动 > 水消失 > 水流动 > 空气
    inline static bool PrioritySort(const ScheduledTick& a, const ScheduledTick& b) {
        if (a.pos() == b.pos()) {
            if (a.type() == -1) return true;
            if (b.type() == -1) return false;
            auto waterLevel = [](int id) { return (id >= 86 && id <= 101) ? ((id - 86) % 8) * 2 + ((id - 86) / 8) + 1 : 0; };
            auto isWaterDone = [](int id) { return (id <= -86 && id >= -101); };
            auto lavaLevel = [](int id) { return (id >= 102 && id <= 117) ? ((id - 102) % 8) * 2 + ((id - 102) / 8) + 1 : 0; };
            auto isLavaDone = [](int id) { return (id <= -102 && id >= -117); };
            auto isAir = [](int id) { return id == 0; };
            int a_priority = isLavaDone(a.type()) << 1 | lavaLevel(a.type()) << 2 | isWaterDone(a.type()) << 7 | waterLevel(a.type()) << 8 | isAir(a.type()) << 13;
            int b_priority = isLavaDone(b.type()) << 1 | lavaLevel(b.type()) << 2 | isWaterDone(b.type()) << 7 | waterLevel(b.type()) << 8 | isAir(b.type()) << 13;
            return a_priority < b_priority;
        }
        return a.pos() < b.pos();
    }

    // 执行收集到的计划刻
    void runCollectedTicks(std::function<void(BlockPos, LevelTicks&)> consumer, std::function<void(BlockPos, int)> setBlockID) {
        std::sort(toRunThisTick_.begin(), toRunThisTick_.end(), PrioritySort);
        std::vector<ScheduledTick> ticksToProcess;
        ticksToProcess.reserve(toRunThisTick_.size());

        for (auto begin = toRunThisTick_.begin(); begin != toRunThisTick_.end(); ++begin) {
            // type == -1 代表是 SetBlock 的方块，不需要再 Set 了（因为已经在前面 Set 过了），直接执行行为逻辑
            if (begin->type() == -1) {
                ticksToProcess.push_back(*begin);
                scheduledSet_.erase(*begin);
                // 同一位置选择第一个类型，后面的都忽略
                while (begin + 1 != toRunThisTick_.end() && (begin + 1)->pos() == begin->pos()) {
                    scheduledSet_.erase(*(begin + 1));
                    ++begin;
                }
                continue;
            }
            // type < 0 代表是水流或岩浆流动的消失，直接把方块类型设为负数的相反数
            int type = begin->type();

            if (type < 0) {
                if (type == -94) {
                    begin->setType(0);
                } else if (type == -110) {
                    begin->setType(0);
                } else {
                    begin->setType(-type);
                }
            }
            // 其他类型的方块，直接执行行为逻辑
            setBlockID(begin->pos(), begin->type());
            ticksToProcess.push_back(*begin);
            scheduledSet_.erase(*begin);
            while (begin + 1 != toRunThisTick_.end() && (begin + 1)->pos() == begin->pos()) {
                scheduledSet_.erase(*(begin + 1));
                ++begin;
            }
        }

        // 4. 串行执行 consumer (已移除 OpenMP 并行)
        for (const auto& tick : ticksToProcess) {
            consumer(tick.pos(), *this);
        }
    }

    void cleanupAfterTick() {
        toRunThisTick_.clear();
    }
};
