#pragma once

#include "Chunk.h"
#include "ChunkCoord.h"
#include <unordered_map>
#include "Tuple.h"

void lightTick(std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> &chunks);
void addLightChange(const BlockPos& pos, int oldLightEmission, int newLightEmission);
