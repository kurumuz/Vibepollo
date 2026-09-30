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
  uint diagnose;  // also test the flipped, halved and doubled vector (counters 5-7)
  uint pad;
};

Texture2D<float4> color : register(t0);
Texture2D<float2> vectors : register(t1);
Texture2D<float> prev_luma : register(t2);
Texture2D<float> cur_luma_in : register(t3);
RWTexture2D<float> cur_luma : register(u0);
RWStructuredBuffer<int2> field : register(u1);
// Counters (uints): 0 blocks, 1 without a vector (none / out of frame / not
// checkable), 2 DLSS said exactly no motion, 3 the vector beat zero, 4 zero
// beat the vector, 8 a vector under half a pixel (sent, not checked); of the
// blocks with a vector checked, on diagnosing frames: 5 the flipped vector
// fits better than it, 6 half of it does, 7 twice it does, 9 such blocks
RWByteAddressBuffer stats : register(u2);

void count(uint i) {
  stats.InterlockedAdd(i * 4, 1);
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
  const uint index = id.y * blocks.x + id.x;
  const int2 origin = int2(id.xy) * 16;
  count(0);

  float2 s[16];
  uint n = 0;
  for (uint j = 0; j < 4; ++j) {
    for (uint i = 0; i < 4; ++i) {
      const float2 p = float2(origin) + float2(i * 4 + 2.5, j * 4 + 2.5);
      if (p.x < (float) frame_size.x && p.y < (float) frame_size.y) {
        const int2 t = int2(p * texel_per_pixel);
        const float2 m = vectors.Load(int3(t, 0)) * pixel_per_unit;
        if (finite2(m)) {
          s[n++] = m;  // (a sample that is not a number would poison every cost)
        }
      }
    }
  }
  if (n == 0 || !verify) {
    field[index] = int2(kNone, 0);
    count(1);
    return;
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

  // A vector that leads out of the frame is no hint
  const float2 reference = float2(origin) + 8.0 + v;
  if (reference.x < 0.0 || reference.y < 0.0 || reference.x >= (float) frame_size.x || reference.y >= (float) frame_size.y) {
    field[index] = int2(kNone, 0);
    count(1);
    return;
  }

  const int2 d = int2(round(v));
  if (any(d != 0)) {
    // The vector against zero motion; on diagnosing frames also its flip,
    // half and double (a wrong sign or scale shows up there), each rounded
    // on its own
    float sad_v = 0, sad_0 = 0, sad_flip = 0, sad_half = 0, sad_double = 0;
    const int2 last = int2(frame_size) - 1;
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
        sad_v += abs(c - prev_luma.Load(int3(clamp(q + d, int2(0, 0), last), 0)));
        if (diagnose) {
          sad_flip += abs(c - prev_luma.Load(int3(clamp(q - d, int2(0, 0), last), 0)));
          sad_half += abs(c - prev_luma.Load(int3(clamp(q + half_d, int2(0, 0), last), 0)));
          sad_double += abs(c - prev_luma.Load(int3(clamp(q + double_d, int2(0, 0), last), 0)));
        }
      }
    }
    if (diagnose) {
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
    if (sad_0 <= sad_v) {
      v = float2(0, 0);  // (ties to zero: the cheaper vector)
      count(4);
    } else {
      count(3);
    }
  } else if (any(v != 0)) {
    count(8);
  } else {
    count(2);
  }
  field[index] = int2(round(v * 4));
}
