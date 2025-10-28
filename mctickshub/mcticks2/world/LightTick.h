#pragma once

#include "Chunk.h"
#include "ChunkCoord.h"
#include <unordered_map>

void lightTick(std::unordered_map<ChunkCoord, Chunk, ChunkCoord::Hash> &chunks);