// Game capture: a DLSS game's motion vectors as one vector per 16x16 block
// of the captured frame, for the encoder's motion search (see
// src/platform/windows/game_capture/motion_hints.cpp).
//
// luma_cs: the frame's luma, kept for checking the next frame's vectors.
// blocks_cs: per block, the medoid of a 4x4 grid of the game's vectors (a
// block on an object's edge takes one surface's motion, not a blend), kept
// only if it predicts the block from the previous frame better than zero
// motion does (the game's vectors know nothing of what is drawn after DLSS,
// such as the HUD). Output: quarter pixels, pointing from the block to where
// it was in the previous frame; x = kNone for no vector.

cbuffer params : register(b0) {
  uint2 frame_size;  // pixels
  uint2 blocks;  // 16x16 blocks (columns, rows)
  float2 texel_per_pixel;  // vector-texture texels per frame pixel
  float2 pixel_per_unit;  // stored vector -> frame pixels (region/frame ratio and DLSS's scale)
  uint linear_light;  // the colour is linear (scRGB, or read through an _SRGB view): compress it
  uint verify;  // prev_luma holds the previous frame (blocks_cs sends nothing otherwise)
  uint diagnose;  // also test the flipped, halved and doubled vector (counters 5-7, 9)
  uint vector_row;  // this candidate set's first row in the vector texture
  uint stats_base;  // this candidate's first counter
  uint field_base;  // ... and first field entry
  uint score_cap;  // the most one block adds to the score (so the frame's total fits 32 bits)
  uint pad;
};

Texture2D<float4> color : register(t0);
Texture2D<float2> vectors : register(t1);
Texture2D<float> prev_luma : register(t2);
Texture2D<float> cur_luma_in : register(t3);
RWTexture2D<float> cur_luma : register(u0);
RWStructuredBuffer<int2> field : register(u1);
// Counters per candidate (uints, from stats_base): 0 blocks, 1 without a
// vector (none / out of frame), 2 DLSS said exactly no motion, 3 the vector
// beat zero, 4 zero beat the vector, 8 a vector under half a pixel (sent,
// not checked); on diagnosing frames, of the blocks with a vector checked:
// 5 the flipped vector fits better than it, 6 half of it does, 7 twice it
// does, 9 such blocks. 10: the score, the sum over blocks of the best
// prediction error (the vector's or zero's), which picks the candidate.
RWByteAddressBuffer stats : register(u2);

void count(uint i) {
  stats.InterlockedAdd((stats_base + i) * 4, 1);
}

// (x8, and at most score_cap a block, set from the block count so the
// total stays below 2^32; a non-finite error counts as the most)
void score(float sad) {
  const float scaled = sad * 8.0;
  stats.InterlockedAdd((stats_base + 10) * 4, scaled >= 0.0 && scaled < (float) score_cap ? (uint) scaled : score_cap);
}

static const int kNone = (int) 0x80000000;

// Finite, tested on the exponent bits (no float optimisation can assume it away)
bool finite2(float2 v) {
  return (asuint(v.x) & 0x7f800000u) != 0x7f800000u && (asuint(v.y) & 0x7f800000u) != 0x7f800000u;
}

float luma_of(float3 c) {
  float l = dot(max(c, 0), float3(0.2126, 0.7152, 0.0722));
  return linear_light ? sqrt(l) : l;
}

[numthreads(8, 8, 1)] void luma_cs(uint3 id: SV_DispatchThreadID) {
  if (any(id.xy >= frame_size)) {
    return;
  }
  cur_luma[id.xy] = luma_of(color.Load(int3(id.xy, 0)).rgb);
}

[numthreads(8, 8, 1)] void blocks_cs(uint3 id: SV_DispatchThreadID) {
  if (any(id.xy >= blocks)) {
    return;
  }
  const uint index = field_base + id.y * blocks.x + id.x;
  const int2 origin = int2(id.xy) * 16;
  const int2 last = int2(frame_size) - 1;
  if (!verify) {
    field[index] = int2(kNone, 0);
    return;
  }
  count(0);

  float2 s[16];
  uint n = 0;
  for (uint j = 0; j < 4; ++j) {
    for (uint i = 0; i < 4; ++i) {
      const float2 p = float2(origin) + float2(i * 4 + 2.5, j * 4 + 2.5);
      if (p.x < (float) frame_size.x && p.y < (float) frame_size.y) {
        const int2 t = int2(p * texel_per_pixel) + int2(0, vector_row);
        const float2 m = vectors.Load(int3(t, 0)) * pixel_per_unit;
        if (finite2(m)) {
          s[n++] = m;  // (a sample that is not a number would poison every cost)
        }
      }
    }
  }

  float best = 3.0e38;
  float2 v = float2(0, 0);
  for (uint a = 0; a < n; ++a) {
    float cost = 0;
    for (uint b = 0; b < n; ++b) {
      cost += abs(s[a].x - s[b].x) + abs(s[a].y - s[b].y);
    }
    if (cost < best) {
      best = cost;
      v = s[a];
    }
  }
  // No samples, or a vector that leads out of the frame: no hint (scored as zero)
  const float2 reference = float2(origin) + 8.0 + v;
  const bool have = n > 0 && reference.x >= 0.0 && reference.y >= 0.0 && reference.x < (float) frame_size.x && reference.y < (float) frame_size.y;
  const int2 d = have ? int2(round(v)) : int2(0, 0);
  const bool moving = any(d != 0);

  // Zero motion's error always (the score compares candidates over the same
  // blocks); the vector's when it moves by a pixel or more; on diagnosing
  // frames also its flip, half and double (a wrong sign or scale shows up
  // there), each rounded on its own
  float sad_0 = 0, sad_v = 0, sad_flip = 0, sad_half = 0, sad_double = 0;
  const bool diag = moving && diagnose;
  const int2 half_d = int2(round(v * 0.5));
  const int2 double_d = int2(round(v * 2.0));
  for (int y = 0; y < 16; ++y) {
    for (int x = 0; x < 16; ++x) {
      const int2 q = origin + int2(x, y);
      if (q.x > last.x || q.y > last.y) {
        continue;
      }
      const float c = cur_luma_in.Load(int3(q, 0));
      sad_0 += abs(c - prev_luma.Load(int3(q, 0)));
      if (moving) {
        sad_v += abs(c - prev_luma.Load(int3(clamp(q + d, int2(0, 0), last), 0)));
      }
      if (diag) {
        sad_flip += abs(c - prev_luma.Load(int3(clamp(q - d, int2(0, 0), last), 0)));
        sad_half += abs(c - prev_luma.Load(int3(clamp(q + half_d, int2(0, 0), last), 0)));
        sad_double += abs(c - prev_luma.Load(int3(clamp(q + double_d, int2(0, 0), last), 0)));
      }
    }
  }

  if (!have) {
    field[index] = int2(kNone, 0);
    count(1);
    score(sad_0);
    return;
  }
  if (diag) {
    count(9);
    if (sad_flip < sad_v) {
      count(5);
    }
    if (sad_half < sad_v) {
      count(6);
    }
    if (sad_double < sad_v) {
      count(7);
    }
  }
  if (moving) {
    if (sad_0 <= sad_v) {
      v = float2(0, 0);  // (ties to zero: the cheaper vector)
      count(4);
      score(sad_0);
    } else {
      count(3);
      score(sad_v);
    }
  } else {
    count(any(v != 0) ? 8 : 2);
    score(sad_0);
  }
  field[index] = int2(round(v * 4));
}
