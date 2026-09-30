// MVC2 motion field encoder (the format and the host side: src/platform/windows/game_capture/
// mvc2_encoder.h / .cpp). Encodes one game frame's motion field (one quarter-pixel
// vector per 16x16 block of the encoded picture) into a stream the client decodes per vector.
// A frame palette of up to 32 motion models (translation, affine, homography) is searched
// on the GPU; each 8x8-vector tile then picks the cheapest of a local model, one palette
// model, two (a selector mask) or four (labels), plus residuals within +-E.
//
// Dispatches (see mvc2_encoder.cpp for the order):
//   clear_cs       the hash table, list counters; per frame also the state and the fitted set
//   tile_fit_cs    per tile: masks, local plans, 4 hypotheses (affine, homography, 2 translations)
//   prior_cs       the previous frame's palette as hypotheses (none: every frame from scratch)
//   dedupe_cs, list_cs    equal serialized models merged; live items listed (atomic appends)
//   sums_cs (on_sample), sbits_cs, sdedupe_cs, slist_cs, sgreedy_cs, keep_cs
//                  the shortlist: every seed refit on a 1/16 sample, sample bitsets, merged by
//                  signature, a greedy on the sample picks the seeds kept
//   args_cs, sums_cs, solve_cs     full-field least-squares refits (split into parts)
//   finalize_cs, bits_cs           slot lists; per slot, the vectors it predicts exactly
//   count_cs, select_cs (rounds)   greedy palette selection, several models per round
//   refresh_cs     (halfway) translations from the vectors not fitted yet
//   tile_encode_cs, scan_cs, pack_cs    per tile the cheapest plan, header, payloads
// The search evaluates models in float; everything written uses the exact (int64, emulated)
// predictions the decoder computes, so the error bound holds whatever the search picked.

cbuffer params : register(b0) {
  uint cols;
  uint rows;
  uint tiles_x;
  uint tiles_y;
  uint E;
  uint tiles;
  uint B;
  uint nvec;
  uint words;  // per bitset
  uint nhyp;
  uint cover0;  // new vectors a model must fit, by type
  uint cover1;
  uint cover2;
  uint phase;  // bits_cs: 0 all slots (and init), 1 the refits, 2 the refreshed translations
  uint mode;  // which items dedupe_cs / list_cs work on (see there)
  uint tsize;  // hash table entries (a power of two)
  uint short_n;  // seeds to keep by their sample score (0: keep all)
  uint on_sample;  // sums_cs: least squares over the sample only (the shortlist's cheap refits)
  uint nparts;  // sums_cs: groups per hypothesis (each sums a share of the vectors; solve_cs adds them)
  uint fast;  // 1: only the previous frame's models as hypotheses (no fresh search)
};

struct Tile {
  uint4 a;  // mask0, mask1, n | has_mask << 8 | affine ok << 9, constant plan word
  uint4 b;  // constant widths, affine plan words (2), affine widths
  uint4 c;  // tile header, mode word, selector / labels 0, 1
  uint4 d;  // labels 2, 3
};
struct Hyp {  // (a palette model as well)
  uint type;  // 0 translation, 1 affine, 2 homography, 3 none
  uint dead;
  uint2 pad;
  int4 p0;  // serialized parameters (mvc2.h order)
  int4 p1;
  float4 lo;  // the same model for the search: translation x, y; affine a0-a3; homography g0-g3
  float4 hi;  // affine a4, a5; homography g4-g7
};

StructuredBuffer<int2> field : register(t0);  // x == -32768: no vector
RWStructuredBuffer<Tile> tinfo : register(u0);
RWStructuredBuffer<Hyp> hyps : register(u1);  // 3 slots per hypothesis
RWStructuredBuffer<uint> bits : register(u2);  // a bitset of `words` words per slot, then the fitted set
RWStructuredBuffer<uint> counts : register(u3);  // per slot: its hash table entry
RWStructuredBuffer<Hyp> pal : register(u4);
RWByteAddressBuffer state : register(u5);  // (112: the previous frame's models, 116: vectors not fitted before the refresh)
                                           // K, done, rounds, K before the refresh, best key, live hypotheses, live slots, slots needing bitsets;
                                           // bytes 32-79: indirect arguments (count, select, sums/pick, bits)
RWStructuredBuffer<uint> offs : register(u6);  // per tile: first payload word; [tiles]: total; [tiles + 1]: header bytes
RWByteAddressBuffer stream : register(u7);
RWStructuredBuffer<float> partials : register(u9);  // sums_cs: 56 sums per listed hypothesis and part
RWStructuredBuffer<Hyp> prior : register(u10);  // the previous frame's palette (count: state 112)
RWStructuredBuffer<uint> lists : register(u8);  // live hypotheses [0, nhyp); live slots [nhyp, 4 nhyp); slots needing bitsets [4 nhyp, 7 nhyp);
                                                // hash table: owners [7 nhyp, + tsize), smallest slot per entry [+ tsize, + 2 tsize);
                                                // shortlist: the kept seeds (a bit each) [7 nhyp + 2 tsize, + 64)

static const int kInvalid = -32768;
static const uint kMaxModels = 32;

int quantize(int r) {
  const uint q = ((uint) abs(r) + E) / (2 * E + 1);
  return r >= 0 ? (int) q : -(int) q;
}

uint zigzag(int q) {
  return (asuint(q) << 1) ^ asuint(q >> 31);
}

uint bits_for(uint z) {
  return z == 0 ? 0 : firstbithigh(z) + 1;
}

int local_model(int c, int gu, int gw, int i, int j) {
  return c + ((gu * (2 * i - 7) + gw * (2 * j - 7)) >> 4);
}

int sx8(uint b) {
  return (int) (b << 24) >> 24;
}

uint words_of(uint type) {
  return type == 0 ? 1 : type == 1 ? 6 : 8;
}

// ---- int64 (two's complement, x low, y high)

uint2 add64(uint2 a, uint2 b) {
  const uint lo = a.x + b.x;
  return uint2(lo, a.y + b.y + (lo < a.x ? 1u : 0u));
}

uint2 neg64(uint2 a) {
  return add64(uint2(~a.x, ~a.y), uint2(1, 0));
}

uint2 sub64(uint2 a, uint2 b) {
  return add64(a, neg64(b));
}

bool isneg64(uint2 a) {
  return (a.y >> 31) != 0;
}

uint2 shl64(uint2 a, uint s) {  // (0 < s < 32)
  return uint2(a.x << s, (a.y << s) | (a.x >> (32 - s)));
}

uint2 i64(int a) {
  return uint2(asuint(a), a < 0 ? 0xffffffffu : 0u);
}

uint2 umul32(uint a, uint b) {
  const uint al = a & 0xffff, ah = a >> 16, bl = b & 0xffff, bh = b >> 16;
  const uint ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
  const uint mid = lh + hl;
  const uint midc = mid < lh ? 0x10000u : 0u;
  const uint lo = ll + (mid << 16);
  return uint2(lo, hh + (mid >> 16) + midc + (lo < ll ? 1u : 0u));
}

uint2 smul32(int a, uint b) {
  const uint2 r = umul32((uint) abs(a), b);
  return a < 0 ? neg64(r) : r;
}

uint2 mul64u(uint2 a, uint b) {  // (|a| b < 2^63)
  const bool neg = isneg64(a);
  const uint2 m = neg ? neg64(a) : a;
  uint2 r = umul32(m.x, b);
  r.y += m.y * b;
  return neg ? neg64(r) : r;
}

float f64(uint2 a) {
  return (float) (int) a.y * 4294967296.0 + (float) a.x;
}

// A homography model's rounded quarter-pixel component: floor((8 n + D) / (2 D)) (mvc2.h)
int hom_exact(int a, int b, int t, int p6, int p7, uint u, uint w, uint x) {
  const uint2 q = add64(smul32(p6, u), smul32(p7, w));
  const uint2 D = add64(q, uint2(0, 256));  // (+ 2^40)
  const uint2 n = sub64(add64(shl64(add64(smul32(a, u), smul32(b, w)), 10), shl64(i64(t), 24)), mul64u(q, x));
  const uint2 num = add64(shl64(n, 3), D), den = shl64(D, 1);
  int qi = (int) floor(f64(num) / f64(den));
  uint2 prod = mul64u(den, (uint) abs(qi));
  uint2 rem = sub64(num, qi < 0 ? neg64(prod) : prod);
  [unroll] for (int it = 0; it < 2; ++it) {  // (branchless: fxc)
    const uint2 up = add64(rem, den), down = sub64(rem, den);
    const bool low = isneg64(rem), high = !isneg64(down);
    qi += low ? -1 : high ? 1 : 0;
    rem = low ? up : high ? down : rem;
  }
  return qi;
}

int2 eval_exact(Hyp m, uint c, uint r) {
  if (m.type == 0) {
    return m.p0.xy;
  }
  if (m.type == 1) {
    const int ci = (int) c, ri = (int) r;
    return int2((m.p0.x + m.p0.y * ci + m.p0.z * ri + 2048) >> 12, (m.p0.w + m.p1.x * ci + m.p1.y * ri + 2048) >> 12);
  }
  const uint u = c * B + B / 2, w = r * B + B / 2;
  return int2(hom_exact(m.p0.x, m.p0.y, m.p0.z, m.p1.z, m.p1.w, u, w, u), hom_exact(m.p0.w, m.p1.x, m.p1.y, m.p1.z, m.p1.w, u, w, w));
}

// ---- float models (the search)

float2 eval_f(uint type, float4 lo, float4 hi, uint c, uint r, out bool ok) {
  ok = type <= 2;
  if (type == 0) {
    return lo.xy;
  }
  if (type == 1) {
    return float2(lo.x + lo.y * c + lo.z * r, lo.w + hi.x * c + hi.y * r);
  }
  const float u = (float) (c * B + B / 2), w = (float) (r * B + B / 2);
  const float q = hi.z * u + hi.w * w, D = 1 + q;
  ok = ok && D > 0.25;
  return 4 * float2(lo.x * u + lo.y * w + lo.z - u * q, lo.w * u + hi.x * w + hi.y - w * q) / max(D, 0.25);
}

bool exact_f(uint type, float4 lo, float4 hi, uint c, uint r, int2 v) {
  bool ok;
  const float2 p = eval_f(type, lo, hi, c, r, ok);
  return ok && all(abs(float2(v) - round(p)) <= (float) E);
}

bool near_f(uint type, float4 lo, float4 hi, uint c, uint r, int2 v) {
  bool ok;
  const float2 p = eval_f(type, lo, hi, c, r, ok);
  return ok && all(abs(float2(v) - p) <= (float) (2 * E + 1));
}

// Normalized direct linear homography fit: upper triangle of the 8x8 normal matrix (36), right side (8)
void acc_dlt(inout float s[44], float u, float w, float u2, float w2) {
  const float r1[8] = {u, w, 1, 0, 0, 0, -u * u2, -w * u2};
  const float r2[8] = {0, 0, 0, u, w, 1, -u * w2, -w * w2};
  uint idx = 0;
  [unroll] for (uint i = 0; i < 8; ++i) {
    [unroll] for (uint j = i; j < 8; ++j) {
      s[idx] += r1[i] * r1[j] + r2[i] * r2[j];
      idx++;
    }
  }
  [unroll] for (uint k = 0; k < 8; ++k) {
    s[36 + k] += r1[k] * u2 + r2[k] * w2;
  }
}

void acc_dlt56(inout float s[56], float u, float w, float u2, float w2) {
  const float r1[8] = {u, w, 1, 0, 0, 0, -u * u2, -w * u2};
  const float r2[8] = {0, 0, 0, u, w, 1, -u * w2, -w * w2};
  uint idx = 12;
  [unroll] for (uint i = 0; i < 8; ++i) {
    [unroll] for (uint j = i; j < 8; ++j) {
      s[idx] += r1[i] * r1[j] + r2[i] * r2[j];
      idx++;
    }
  }
  [unroll] for (uint k = 0; k < 8; ++k) {
    s[48 + k] += r1[k] * u2 + r2[k] * w2;
  }
}

// The fit in pixel coordinates (normalization: (x - mu, y - mw) * sc) as search parameters; false if unusable
bool solve_hom(float s[44], float mu, float mw, float sc, out float4 lo, out float4 hi) {
  float A[8][9];
  uint idx = 0;
  [unroll] for (uint i = 0; i < 8; ++i) {
    [unroll] for (uint j = i; j < 8; ++j) {
      A[i][j] = s[idx];
      A[j][i] = s[idx];
      idx++;
    }
  }
  [unroll] for (uint k = 0; k < 8; ++k) {
    A[k][8] = s[36 + k];
  }
  bool ok = true;
  // (fully unrolled, pivoting by swapping in any larger row: constant indices only)
  [unroll] for (uint col = 0; col < 8; ++col) {
    [unroll] for (uint r = col + 1; r < 8; ++r) {
      const bool swap = abs(A[r][col]) > abs(A[col][col]);
      [unroll] for (uint j2 = col; j2 < 9; ++j2) {
        const float x = A[col][j2], y = A[r][j2];
        A[col][j2] = swap ? y : x;
        A[r][j2] = swap ? x : y;
      }
    }
    ok = ok && abs(A[col][col]) > 1e-20;
    [unroll] for (uint r2 = col + 1; r2 < 8; ++r2) {
      const float f = A[r2][col] / A[col][col];
      [unroll] for (uint j3 = col; j3 < 9; ++j3) {
        A[r2][j3] -= f * A[col][j3];
      }
    }
  }
  float h[8];
  [unroll] for (int i2 = 7; i2 >= 0; --i2) {
    float acc = A[i2][8];
    [unroll] for (int j4 = i2 + 1; j4 < 8; ++j4) {
      acc -= A[i2][j4] * h[j4];
    }
    h[i2] = acc / A[i2][i2];
  }
  const float3x3 Hn = float3x3(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1);
  const float3x3 T = float3x3(sc, 0, -sc * mu, 0, sc, -sc * mw, 0, 0, 1);
  const float3x3 Ti = float3x3(1 / sc, 0, mu, 0, 1 / sc, mw, 0, 0, 1);
  float3x3 H = mul(Ti, mul(Hn, T));
  ok = ok && abs(H[2][2]) > 1e-20;
  H /= H[2][2];
  lo = float4(H[0][0] - 1, H[0][1], H[0][2], H[1][0]);
  hi = float4(H[1][1] - 1, H[1][2], H[2][0], H[2][1]);
  ok = ok && all(abs(lo) < 1e30) && all(abs(hi) < 1e30);
  // The denominator well positive over the field
  const float u0 = (float) (B / 2), u1 = (float) ((cols - 1) * B + B / 2), w0 = u0, w1 = (float) ((rows - 1) * B + B / 2);
  ok = ok && 1 + hi.z * u0 + hi.w * w0 > 0.3 && 1 + hi.z * u1 + hi.w * w0 > 0.3 && 1 + hi.z * u0 + hi.w * w1 > 0.3 && 1 + hi.z * u1 + hi.w * w1 > 0.3;
  return ok;
}

// Hypotheses 0 .. 4 tiles - 1 come from the tiles (4 each; h & 3 >= 2 the translations), the
// next 32 are the previous frame's palette
uint tile_hyps() {
  return 4 * tiles;
}

bool refresh_seed(uint h) {
  return h < tile_hyps() && (h & 3) >= 2;
}

// The serialized form of a search model (mvc2.h), with the search model rebuilt from it; false if not representable
bool quantize_model(inout Hyp h) {
  uint type = h.type;
  const float4 lo = h.lo, hi = h.hi;
  int4 p0 = int4(0, 0, 0, 0), p1 = int4(0, 0, 0, 0);
  bool ok = type <= 2;
  if (type == 0) {
    ok = ok && all(abs(lo.xy) < 8192);
    p0.xy = ok ? int2(round(lo.xy)) : int2(0, 0);
  } else if (type == 1) {
    const float4 a = lo * 4096, b = hi * 4096;
    ok = all(abs(a) < 2.0e9) && all(abs(b.xy) < 2.0e9);
    p0 = ok ? int4(round(a)) : int4(0, 0, 0, 0);
    p1 = ok ? int4(int2(round(b.xy)), 0, 0) : int4(0, 0, 0, 0);
    if (p0.y == 0 && p0.z == 0 && p1.x == 0 && p1.y == 0) {
      type = 0;
      p0 = int4((p0.x + 2048) >> 12, (p0.w + 2048) >> 12, 0, 0);
    } else {
      const float bx = abs((float) p0.x) + abs((float) p0.y) * cols + abs((float) p0.z) * rows + 2048;
      const float by = abs((float) p0.w) + abs((float) p1.x) * cols + abs((float) p1.y) * rows + 2048;
      ok = ok && max(bx, by) < 2.0e9;
    }
  } else if (type == 2) {
    const float4 a = lo * float4(1073741824.0, 1073741824.0, 65536.0, 1073741824.0), b = hi * float4(1073741824.0, 65536.0, 1099511627776.0, 1099511627776.0);
    ok = all(abs(a) < 2.1e9) && all(abs(b) < 2.1e9);
    p0 = ok ? int4(round(a)) : int4(0, 0, 0, 0);
    p1 = ok ? int4(round(b)) : int4(0, 0, 0, 0);
    const float u1 = (float) ((cols - 1) * B + B / 2), w1 = (float) ((rows - 1) * B + B / 2), u0 = (float) (B / 2);
    const float lim = 1.05 * 274877906944.0 - 1099511627776.0;  // (2^40 + p6 u + p7 w >= 1.05 2^38 at the corners)
    ok = ok && (float) p1.z * u0 + (float) p1.w * u0 > lim && (float) p1.z * u1 + (float) p1.w * u0 > lim && (float) p1.z * u0 + (float) p1.w * w1 > lim && (float) p1.z * u1 + (float) p1.w * w1 > lim;
  }
  h.type = ok ? type : 3;
  h.dead = ok ? 0 : 1;
  h.p0 = p0;
  h.p1 = p1;
  if (type == 0) {
    h.lo = float4((float2) p0.xy, 0, 0);
    h.hi = float4(0, 0, 0, 0);
  } else if (type == 1) {
    h.lo = float4(p0) / 4096;
    h.hi = float4((float2) p1.xy / 4096, 0, 0);
  } else if (type == 2) {
    h.lo = float4(p0) / float4(1073741824.0, 1073741824.0, 65536.0, 1073741824.0);
    h.hi = float4(p1) / float4(1073741824.0, 65536.0, 1099511627776.0, 1099511627776.0);
  }
  return ok;
}

Hyp none_hyp() {
  Hyp h;
  h.type = 3;
  h.dead = 1;
  h.pad = uint2(0, 0);
  h.p0 = int4(0, 0, 0, 0);
  h.p1 = int4(0, 0, 0, 0);
  h.lo = float4(0, 0, 0, 0);
  h.hi = float4(0, 0, 0, 0);
  return h;
}

// Hypothesis h's model into slot 3h (serialized), its refit slots emptied
void put_hyp(uint h, Hyp m) {
  quantize_model(m);
  hyps[3 * h] = m;
  hyps[3 * h + 1] = none_hyp();
  hyps[3 * h + 2] = none_hyp();
}

// ---- prior_cs: 32 threads; the previous frame's palette as hypotheses 4 tiles ..

[numthreads(32, 1, 1)] void prior_cs(uint k: SV_GroupIndex) {
  const uint h = tile_hyps() + k;
  Hyp m = none_hyp();
  if (k < state.Load(112)) {
    m = prior[k];
    m.dead = 0;
  }
  hyps[3 * h] = m;
  hyps[3 * h + 1] = none_hyp();
  hyps[3 * h + 2] = none_hyp();
}

// ---- tile_fit_cs

groupshared float g_sum[12][64];
groupshared float g_dlt[44][64];
groupshared uint g_mask[2];
groupshared uint g_cov[2];
groupshared uint g_zm[4];
groupshared int g_lm[10];  // constant cx, cy; affine cx, cy, gxu, gxw, gyu, gyw; affine usable; n
groupshared float g_ld[6];  // affine least squares (tile coordinates) d0, d1, d2 for x, y
groupshared int2 g_v[64];
groupshared uint g_cnt[64];

[numthreads(64, 1, 1)] void tile_fit_cs(uint3 gid: SV_GroupID, uint k: SV_GroupIndex) {
  const uint t = gid.y * tiles_x + gid.x;
  const int i = (int) (k % 8), j = (int) (k / 8);
  const uint ox = gid.x * 8, oy = gid.y * 8, c = ox + i, r = oy + j;
  const bool inside = c < cols && r < rows;
  const int2 v = inside ? field[r * cols + c] : int2(kInvalid, 0);
  const bool valid = inside && v.x != kInvalid;
  if (k < 2) {
    g_mask[k] = 0;
    g_cov[k] = 0;
  }
  if (k < 4) {
    g_zm[k] = 0;
  }
  g_v[k] = valid ? v : int2(kInvalid, 0);
  GroupMemoryBarrierWithGroupSync();
  if (inside) {
    InterlockedOr(g_cov[k >> 5], 1u << (k & 31));
  }
  if (valid) {
    InterlockedOr(g_mask[k >> 5], 1u << (k & 31));
  }
  // Local least squares (MVC1 plan) sums
  const float u = 2 * i - 7, w = 2 * j - 7, X = v.x, Y = v.y, on = valid ? 1.0 : 0.0;
  g_sum[0][k] = on;
  g_sum[1][k] = on * u;
  g_sum[2][k] = on * w;
  g_sum[3][k] = on * u * u;
  g_sum[4][k] = on * w * w;
  g_sum[5][k] = on * u * w;
  g_sum[6][k] = on * X;
  g_sum[7][k] = on * u * X;
  g_sum[8][k] = on * w * X;
  g_sum[9][k] = on * Y;
  g_sum[10][k] = on * u * Y;
  g_sum[11][k] = on * w * Y;
  // Homography sums, normalized to the tile
  {
    float s[44];
    [unroll] for (uint q = 0; q < 44; ++q) {
      s[q] = 0;
    }
    const float mu = (ox + 4.0) * B, mw = (oy + 4.0) * B, sc = 1.0 / (8.0 * B);
    if (valid) {
      const float pu = (c * B + B / 2 - mu) * sc, pw = (r * B + B / 2 - mw) * sc;
      acc_dlt(s, pu, pw, pu + X * 0.25 * sc, pw + Y * 0.25 * sc);
    }
    [unroll] for (uint q2 = 0; q2 < 44; ++q2) {
      g_dlt[q2][k] = s[q2];
    }
  }
  // Translations: how often this vector occurs in the tile and its eight neighbours (a small
  // object's vectors often straddle tiles), at its first occurrence in the tile
  {
    bool first = true;
    for (uint q = 0; q < k; ++q) {
      first = first && !(g_v[q].x == v.x && g_v[q].y == v.y);
    }
    uint cnt = 0;
    if (valid && first) {
      const uint c0 = ox >= 8 ? ox - 8 : 0, r0 = oy >= 8 ? oy - 8 : 0, c1 = min(ox + 16, cols), r1 = min(oy + 16, rows);
      for (uint rr = r0; rr < r1; ++rr) {
        for (uint cc = c0; cc < c1; ++cc) {
          const int2 o = field[rr * cols + cc];
          cnt += o.x == v.x && o.y == v.y ? 1 : 0;
        }
      }
    }
    g_cnt[k] = cnt;
  }
  GroupMemoryBarrierWithGroupSync();
  for (uint s2 = 32; s2 > 0; s2 >>= 1) {
    if (k < s2) {
      [unroll] for (int q = 0; q < 12; ++q) {
        g_sum[q][k] += g_sum[q][k + s2];
      }
      [unroll] for (int q3 = 0; q3 < 44; ++q3) {
        g_dlt[q3][k] += g_dlt[q3][k + s2];
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }
  if (k == 0) {
    const float n = g_sum[0][0];
    g_lm[9] = (int) n;
    g_lm[0] = n > 0 ? (int) round(g_sum[6][0] / n) : 0;
    g_lm[1] = n > 0 ? (int) round(g_sum[9][0] / n) : 0;
    const float a00 = n, a01 = g_sum[1][0], a02 = g_sum[2][0], a11 = g_sum[3][0], a12 = g_sum[5][0], a22 = g_sum[4][0];
    const float det = a00 * (a11 * a22 - a12 * a12) - a01 * (a01 * a22 - a12 * a02) + a02 * (a01 * a12 - a11 * a02);
    g_lm[8] = abs(det) > 1e-6 ? 1 : 0;
    [unroll] for (int comp = 0; comp < 2; ++comp) {
      const float b0 = g_sum[6 + comp * 3][0], b1 = g_sum[7 + comp * 3][0], b2 = g_sum[8 + comp * 3][0];
      const float d0 = b0 * (a11 * a22 - a12 * a12) - a01 * (b1 * a22 - a12 * b2) + a02 * (b1 * a12 - a11 * b2);
      const float d1 = a00 * (b1 * a22 - a12 * b2) - b0 * (a01 * a22 - a12 * a02) + a02 * (a01 * b2 - b1 * a02);
      const float d2 = a00 * (a11 * b2 - b1 * a12) - a01 * (a01 * b2 - b1 * a02) + b0 * (a01 * a12 - a11 * a02);
      const float safe = g_lm[8] ? det : 1.0;
      g_ld[comp * 3] = d0 / safe;
      g_ld[comp * 3 + 1] = d1 / safe;
      g_ld[comp * 3 + 2] = d2 / safe;
      g_lm[2 + comp] = (int) round(d0 / safe);
      g_lm[4 + comp * 2] = (int) clamp(round(d1 / safe * 16), -128, 127);
      g_lm[5 + comp * 2] = (int) clamp(round(d2 / safe * 16), -128, 127);
    }
  }
  GroupMemoryBarrierWithGroupSync();
  if (valid) {
    InterlockedMax(g_zm[0], zigzag(quantize(v.x - g_lm[0])));
    InterlockedMax(g_zm[1], zigzag(quantize(v.y - g_lm[1])));
    InterlockedMax(g_zm[2], zigzag(quantize(v.x - local_model(g_lm[2], g_lm[4], g_lm[5], i, j))));
    InterlockedMax(g_zm[3], zigzag(quantize(v.y - local_model(g_lm[3], g_lm[6], g_lm[7], i, j))));
  }
  GroupMemoryBarrierWithGroupSync();
  if (k == 0) {
    const uint n = (uint) g_lm[9];
    const bool has_mask = g_mask[0] != g_cov[0] || g_mask[1] != g_cov[1];
    Tile tl;
    tl.a = uint4(g_mask[0], g_mask[1], n | (has_mask ? 1u << 8 : 0u) | (g_lm[8] ? 1u << 9 : 0u), (asuint(g_lm[0]) & 0xffff) | asuint(g_lm[1]) << 16);
    tl.b = uint4(bits_for(g_zm[0]) | bits_for(g_zm[1]) << 5, (asuint(g_lm[2]) & 0xffff) | asuint(g_lm[3]) << 16,
                 (asuint(g_lm[4]) & 0xff) | (asuint(g_lm[5]) & 0xff) << 8 | (asuint(g_lm[6]) & 0xff) << 16 | asuint(g_lm[7]) << 24, bits_for(g_zm[2]) | bits_for(g_zm[3]) << 5);
    tl.c = uint4(0, 0, 0, 0);
    tl.d = uint4(0, 0, 0, 0);
    tinfo[t] = tl;
    // Hypotheses: affine in frame coordinates (u = 2 (c - ox) - 7), homography, two translations
    Hyp h = none_hyp();
    h.type = n >= 8 && g_lm[8] ? 1 : 3;
    const float ax = g_ld[0] - g_ld[1] * (2.0 * ox + 7) - g_ld[2] * (2.0 * oy + 7), ay = g_ld[3] - g_ld[4] * (2.0 * ox + 7) - g_ld[5] * (2.0 * oy + 7);
    h.lo = float4(ax, 2 * g_ld[1], 2 * g_ld[2], ay);
    h.hi = float4(2 * g_ld[4], 2 * g_ld[5], 0, 0);
    put_hyp(t * 4, h);
    float s[44];
    [unroll] for (uint q = 0; q < 44; ++q) {
      s[q] = g_dlt[q][0];
    }
    float4 lo, hi;
    const bool ok = n >= 16 && solve_hom(s, (ox + 4.0) * B, (oy + 4.0) * B, 1.0 / (8.0 * B), lo, hi);
    h.type = ok ? 2 : 3;
    h.lo = lo;
    h.hi = hi;
    put_hyp(t * 4 + 1, h);
    uint b1 = 64, b2 = 64, c1 = 0, c2 = 0;
    for (uint q2 = 0; q2 < 64; ++q2) {
      const uint cq = g_cnt[q2];
      if (cq > c1) {
        b2 = b1;
        c2 = c1;
        b1 = q2;
        c1 = cq;
      } else if (cq > c2) {
        b2 = q2;
        c2 = cq;
      }
    }
    h.hi = float4(0, 0, 0, 0);
    h.type = c1 >= cover0 ? 0 : 3;
    h.lo = float4(b1 < 64 ? (float2) g_v[b1] : float2(0, 0), 0, 0);
    put_hyp(t * 4 + 2, h);
    h.type = c2 >= cover0 ? 0 : 3;
    h.lo = float4(b2 < 64 ? (float2) g_v[b2] : float2(0, 0), 0, 0);
    put_hyp(t * 4 + 3, h);
  }
}

// ---- refresh_cs: a group per tile; translation hypotheses from the vectors no model fits yet

[numthreads(64, 1, 1)] void refresh_cs(uint3 gid: SV_GroupID, uint k: SV_GroupIndex) {
  const uint t = gid.y * tiles_x + gid.x;
  const uint ox = gid.x * 8, oy = gid.y * 8, c = ox + k % 8, r = oy + k / 8;
  const uint cov = 3 * nhyp * words;
  const bool inside = c < cols && r < rows;
  const uint kk = r * cols + c;
  const int2 v = inside ? field[kk] : int2(kInvalid, 0);
  const bool open = inside && v.x != kInvalid && ((bits[cov + (kk >> 5)] >> (kk & 31)) & 1) == 0;
  g_v[k] = open ? v : int2(kInvalid, 0);
  GroupMemoryBarrierWithGroupSync();
  bool first = true;
  for (uint q = 0; q < k; ++q) {
    first = first && !(g_v[q].x == v.x && g_v[q].y == v.y);
  }
  uint cnt = 0;
  if (open && first) {
    const uint c0 = ox >= 8 ? ox - 8 : 0, r0 = oy >= 8 ? oy - 8 : 0, c1 = min(ox + 16, cols), r1 = min(oy + 16, rows);
    for (uint rr = r0; rr < r1; ++rr) {
      for (uint cc = c0; cc < c1; ++cc) {
        const uint ko = rr * cols + cc;
        const int2 o = field[ko];
        cnt += o.x == v.x && o.y == v.y && ((bits[cov + (ko >> 5)] >> (ko & 31)) & 1) == 0 ? 1 : 0;
      }
    }
  }
  g_cnt[k] = cnt;
  GroupMemoryBarrierWithGroupSync();
  if (k == 0) {
    uint b1 = 64, b2 = 64, n1 = 0, n2 = 0;
    for (uint q2 = 0; q2 < 64; ++q2) {
      const uint cq = g_cnt[q2];
      if (cq > n1) {
        b2 = b1;
        n2 = n1;
        b1 = q2;
        n1 = cq;
      } else if (cq > n2) {
        b2 = q2;
        n2 = cq;
      }
    }
    Hyp h = none_hyp();
    h.type = n1 >= cover0 ? 0 : 3;
    h.lo = float4(b1 < 64 ? (float2) g_v[b1] : float2(0, 0), 0, 0);
    put_hyp(t * 4 + 2, h);
    h.type = n2 >= cover0 ? 0 : 3;
    h.lo = float4(b2 < 64 ? (float2) g_v[b2] : float2(0, 0), 0, 0);
    put_hyp(t * 4 + 3, h);
  }
}

// ---- Deduplication and lists. Items by mode: 0 the hypotheses' seed slots 3h; 1 every slot;
// 2 the refreshed translation seeds (3h, h & 3 >= 2). A slot whose serialized model equals a
// smaller slot's is dropped. Lists are filled by atomic appends: their order does not matter.

uint item_slot(uint i, out bool ok) {
  if (mode == 1) {
    ok = i < 3 * nhyp;
    return i;
  }
  const uint h = mode == 2 ? (i / 2) * 4 + 2 + (i & 1) : i;
  ok = h < nhyp;
  return 3 * h;
}

uint key_hash(Hyp m) {
  uint x = m.type * 0x9e3779b1u;
  const uint w[8] = {asuint(m.p0.x), asuint(m.p0.y), asuint(m.p0.z), asuint(m.p0.w), asuint(m.p1.x), asuint(m.p1.y), asuint(m.p1.z), asuint(m.p1.w)};
  [unroll] for (uint q = 0; q < 8; ++q) {
    x = (x ^ w[q]) * 0x85ebca77u;
    x ^= x >> 13;
  }
  return x;
}

// The kept one of equal models: the previous frame's first, then the smallest slot
uint dedupe_key(uint sl) {
  return (sl / 3 >= tile_hyps() ? 0u : 0x80000000u) | sl;
}

bool same_model(Hyp a, Hyp b) {
  return a.type == b.type && all(a.p0 == b.p0) && all(a.p1 == b.p1);
}

// The shortlist: seeds scored on a sample (vectors (4i + 2, 4j + 2)); regions of 4x4 tiles
uint scols() {
  return (cols + 1) / 4;
}

uint nsamples() {
  return scols() * ((rows + 1) / 4);
}

uint nregions() {
  return ((tiles_x + 3) / 4) * ((tiles_y + 3) / 4);
}

uint region_of(uint h) {
  const uint t = h / 4;
  return ((t % tiles_x) / 4) + ((t / tiles_x) / 4) * ((tiles_x + 3) / 4);
}

// clear_cs (words or 2 tsize threads, whichever more): the hash table; phase 0 also the state and the fitted set
[numthreads(64, 1, 1)] void clear_cs(uint3 id: SV_DispatchThreadID) {
  const uint i = id.x, tb = 7 * nhyp;
  if (i < tsize) {
    lists[tb + i] = 0;
    lists[tb + tsize + i] = 0xffffffffu;
  }
  if (phase == 0 && mode == 0) {
    if (i < words) {
      uint word = 0;
      for (uint b = 0; b < 32; ++b) {
        const uint kk = i * 32 + b;
        word |= (kk >= nvec || field[kk].x == kInvalid) ? 1u << b : 0u;
      }
      bits[3 * nhyp * words + i] = word;
    }
    if (i == 0) {
      state.Store4(0, uint4(0, 0, 0, 0));
      state.Store4(16, uint4(0, 0, 0, 0));
      state.Store(16, 0xffffffffu);  // (no refresh yet)
      state.Store(116, 0);
    }
  }
  if (phase == 2 && i < words) {
    uint dummy;
    state.InterlockedAdd(116, countbits(~bits[3 * nhyp * words + i]), dummy);  // (vectors the models before the refresh leave unfitted)
  }
  if (i == 0) {
    // (list counters: the hypothesis list's is the sums dispatch width, the slot list's the count dispatch width)
    state.Store(28, 0);
    state.Store3(32, uint3(0, 1, 1));
    if (mode == 0) {
      state.Store3(56, uint3(0, 1, 1));
    }
  }
}

// dedupe_cs (a thread per item): insert into the hash table, record the smallest slot per model
[numthreads(64, 1, 1)] void dedupe_cs(uint3 id: SV_DispatchThreadID) {
  bool ok;
  const uint sl = item_slot(id.x, ok);
  if (!ok) {
    return;
  }
  const Hyp m = hyps[sl];
  counts[sl] = 0xffffffffu;
  if (m.dead) {
    return;
  }
  const uint tb = 7 * nhyp, mask = tsize - 1;
  uint e = key_hash(m) & mask;
  [allow_uav_condition] for (uint probe = 0; probe < tsize; ++probe) {
    uint owner;
    InterlockedCompareExchange(lists[tb + e], 0, sl + 1, owner);
    if (owner == 0 || same_model(hyps[owner - 1], m)) {
      counts[sl] = e;
      InterlockedMin(lists[tb + tsize + e], dedupe_key(sl));
      break;
    }
    e = (e + 1) & mask;
  }
}

// list_cs (a thread per item): duplicates dropped; the live items appended. Mode 0: the hypothesis
// list; modes 1, 2: phase 0 every live slot to both slot lists, phase 2 (mode 2 over every slot) the
// refreshed ones to the bitset list and every live slot to the counting list
[numthreads(64, 1, 1)] void list_cs(uint3 id: SV_DispatchThreadID) {
  const uint i = id.x;
  const uint tb = 7 * nhyp;
  if (mode == 4) {
    const uint h = tile_hyps() + i;
    if (i < 32 && !hyps[3 * h].dead) {
      uint at;
      state.InterlockedAdd(56, 1, at);
      lists[at] = h;
    }
    return;
  }
  if (mode == 5) {
    if (i < 96) {
      const uint sl = 3 * tile_hyps() + i;
      if (!hyps[sl].dead) {
        uint at;
        state.InterlockedAdd(28, 1, at);
        lists[4 * nhyp + at] = sl;
      }
    }
    return;
  }
  if (mode == 3) {
    if (i < nhyp && !hyps[3 * i].dead) {
      uint at;
      state.InterlockedAdd(56, 1, at);
      lists[at] = i;
    }
    return;
  }
  if (mode == 0) {
    if (i >= nhyp) {
      return;
    }
    const uint sl = 3 * i, e = counts[sl];
    if (fast && i < tile_hyps()) {
      hyps[sl].dead = 1;  // (only the previous frame's models)
      return;
    }
    if (hyps[sl].dead) {
      return;
    }
    if (!fast && e != 0xffffffffu && lists[tb + tsize + e] != dedupe_key(sl)) {
      hyps[sl].dead = 1;
      return;
    }
    uint at;
    state.InterlockedAdd(56, 1, at);
    lists[at] = i;
    return;
  }
  if (i >= 3 * nhyp) {
    return;
  }
  const uint sl = i;
  const bool fresh = mode == 1 || (sl % 3 == 0 && refresh_seed(sl / 3));  // (deduplicated just now)
  if (hyps[sl].dead) {
    return;
  }
  if (fresh) {
    const uint e = counts[sl];
    if (e != 0xffffffffu && lists[tb + tsize + e] != dedupe_key(sl)) {
      hyps[sl].dead = 1;
      return;
    }
  }
  uint at;
  state.InterlockedAdd(32, 1, at);
  lists[nhyp + at] = sl;
  if (fresh) {
    state.InterlockedAdd(28, 1, at);
    lists[4 * nhyp + at] = sl;
  }
}

// finalize_cs (one thread): the bitset dispatch size; a batch of rounds armed
[numthreads(1, 1, 1)] void finalize_cs() {
  const uint nb = state.Load(28), K = state.Load(0);
  const bool room = K < kMaxModels;
  state.Store(24, state.Load(32));
  state.Store3(68, uint3((words + 63) / 64, (nb + 31) / 32, 1));
  if (phase == 2) {
    state.Store(12, K);  // (models before the refresh)
    state.Store(16, state.Load(8));  // (rounds before the refresh)
  }
  state.Store(4, room ? 0 : 1);
  if (!room) {
    state.Store(32, 0);
  }
  state.Store3(44, uint3(room ? 1 : 0, 1, 1));
}

// ---- The shortlist: every seed refit on the sample (sums_cs on_sample), then a greedy selection
// on the sample (vectors (4i + 2, 4j + 2)) picks the seeds whose slots cover it: only those are
// refit on the whole field and searched at full resolution. Sample bitsets live in the first
// sample-words of each slot's full bitset (overwritten later).

uint swords() {
  return (nsamples() + 31) / 32;
}

uint shash(uint w, uint k) {
  return w * (((0x9e3779b1u + 2 * k * 0x85ebca6bu)) | 1u);
}

uint shash2(uint w, uint k) {
  const uint x = (w ^ (k * 0x27d4eb2fu)) * 0x165667b1u;
  return x ^ (x >> 15);
}

// sbits_cs: a group per listed seed; its three slots' sample bitsets, a hash of each
groupshared uint g_sh[3];
groupshared uint g_sh2[3];

[numthreads(64, 1, 1)] void sbits_cs(uint3 gid: SV_GroupID, uint k: SV_GroupIndex) {
  const uint h = lists[gid.x], sc = scols(), ns = nsamples(), sw = swords();
  if (k < 3) {
    g_sh[k] = 0;
    g_sh2[k] = 0;
  }
  GroupMemoryBarrierWithGroupSync();
  [unroll] for (uint v = 0; v < 3; ++v) {
    const Hyp m = hyps[3 * h + v];
    uint hs = 0, hs2 = 0;
    for (uint w = k; w < sw; w += 64) {
      uint word = 0;
      if (!m.dead) {
        for (uint b = 0; b < 32; ++b) {
          const uint q = w * 32 + b;
          if (q < ns) {
            const uint c = (q % sc) * 4 + 2, r = (q / sc) * 4 + 2;
            const int2 x = field[r * cols + c];
            word |= x.x != kInvalid && exact_f(m.type, m.lo, m.hi, c, r, x) ? 1u << b : 0u;
          }
        }
      }
      bits[(3 * h + v) * words + w] = word;
      hs += shash(word, w);
      hs2 += shash2(word, w);
    }
    InterlockedAdd(g_sh[v], hs);
    InterlockedAdd(g_sh2[v], hs2);
  }
  GroupMemoryBarrierWithGroupSync();
  if (k < 3) {
    // The slot's signature: its predictions at 4x4 anchors spread over the field, in whole pixels (two
    // hashes); and its sample coverage. (Past the sample words of its bitset, overwritten later.)
    const Hyp m = hyps[3 * h + k];
    uint ha = m.dead ? 0x51ed270bu : 0x2545f491u, hb = m.dead ? 0x68e31da4u : 0x9e3779b9u;
    for (uint a = 0; a < 16; ++a) {
      const uint c = ((cols - 1) * (2 * (a % 4) + 1)) / 8, r = ((rows - 1) * (2 * (a / 4) + 1)) / 8;
      bool ok;
      const float2 p = eval_f(m.type, m.lo, m.hi, c, r, ok);
      const int2 q = ok ? int2(floor(p / 4 + 0.5)) : int2(-99999, -99999);
      ha = (ha ^ asuint(q.x)) * 0x85ebca77u;
      ha = (ha ^ asuint(q.y)) * 0xc2b2ae3du;
      hb = (hb ^ (asuint(q.x) * 0x27d4eb2fu)) * 0x165667b1u;
      hb = (hb ^ (asuint(q.y) * 0x9e3779b1u)) * 0x85ebca6bu;
      ha ^= ha >> 16;
      hb ^= hb >> 13;
    }
    counts[3 * h + k] = ha;
    bits[(3 * h + k) * words + words - 1] = hb;
    uint cov = 0;
    for (uint w = 0; w < sw; ++w) {
      cov += countbits(bits[(3 * h + k) * words + w]);
    }
    bits[(3 * h + k) * words + words - 2] = cov;
  }
}

// sdedupe_cs (a thread per listed seed's slot): slots with equal signatures merged, the one covering
// most of the sample (then the smallest slot) kept
[numthreads(64, 1, 1)] void sdedupe_cs(uint3 id: SV_DispatchThreadID) {
  const uint i = id.x / 3, v = id.x % 3;
  if (i >= state.Load(56)) {
    return;
  }
  const uint sl = 3 * lists[i] + v;
  const Hyp m = hyps[sl];
  if (m.dead) {
    return;
  }
  const uint tb = 7 * nhyp, mask = tsize - 1, sw = swords();
  uint e = counts[sl] & mask;
  [allow_uav_condition] for (uint probe = 0; probe < tsize; ++probe) {
    uint owner;
    InterlockedCompareExchange(lists[tb + e], 0, sl + 1, owner);
    // (equal: both signature hashes; a collision would only merge two candidates)
    const bool same = owner == 0 || (counts[owner - 1] == counts[sl] && bits[(owner - 1) * words + words - 1] == bits[sl * words + words - 1]);
    if (same) {
      counts[sl] = e;
      InterlockedMin(lists[tb + tsize + e], 0xffffffffu - (min(bits[sl * words + words - 2], 65535u) << 16 | (0xffff - sl)));
      break;
    }
    e = (e + 1) & mask;
  }
}

// slist_cs (a thread per listed seed's slot): the kept ones listed (the bitset list, as candidates)
[numthreads(64, 1, 1)] void slist_cs(uint3 id: SV_DispatchThreadID) {
  const uint i = id.x / 3, v = id.x % 3;
  if (i >= state.Load(56)) {
    return;
  }
  const uint sl = 3 * lists[i] + v;
  if (hyps[sl].dead || 0xffff - ((0xffffffffu - lists[7 * nhyp + tsize + counts[sl]]) & 0xffff) != sl) {
    return;
  }
  uint at;
  state.InterlockedAdd(28, 1, at);
  lists[4 * nhyp + at] = sl;
}

#ifndef SHORT_STEPS
#define SHORT_STEPS 48
#endif
#ifndef SG_THREADS
#define SG_THREADS 1024
#endif
#ifndef SHORT_MIN_GAIN
#define SHORT_MIN_GAIN 1
#endif
// sgreedy_cs: one group; greedy on the sample over the listed candidates (gain >= 2 samples; the
// chosen slots' seeds kept). Lazy: each thread keeps its candidates' gains, which only fall; a step
// recounts only those whose stale gain could still win (certified as in select_cs).
static const uint kShortSteps = SHORT_STEPS;
static const uint kPer = 4096 / SG_THREADS;  // candidates per thread (up to 4096)
static const uint kSG = SG_THREADS;
static const uint kMinGain = SHORT_MIN_GAIN;
groupshared uint g_sfit[64];
groupshared uint g_keep[128];
groupshared uint g_sbest;
groupshared uint g_stop;

uint sgain(uint sl, uint sw) {
  uint g = 0;
  for (uint w = 0; w < sw; ++w) {
    g += countbits(bits[sl * words + w] & ~g_sfit[w]);
  }
  return g;
}

[numthreads(kSG, 1, 1)] void sgreedy_cs(uint k: SV_GroupIndex) {
  const uint sc = scols(), ns = nsamples(), sw = swords(), n = min(state.Load(28), kSG * kPer), lb = 4 * nhyp;
  if (k < 64) {
    uint word = 0;
    for (uint b = 0; b < 32; ++b) {
      const uint q = k * 32 + b;
      bool fitted = true;
      if (q < ns) {
        fitted = field[((q / sc) * 4 + 2) * cols + (q % sc) * 4 + 2].x == kInvalid;
      }
      word |= fitted ? 1u << b : 0u;
    }
    g_sfit[k] = word;
  }
  if (k < 128) {
    g_keep[k] = 0;
  }
  if (k == 0) {
    g_stop = 0;
  }
  GroupMemoryBarrierWithGroupSync();
  uint slot[kPer], gain[kPer];
  [unroll] for (uint j = 0; j < kPer; ++j) {
    const uint c = k + j * kSG;
    slot[j] = c < n ? lists[lb + c] : 0;
    gain[j] = c < n ? bits[slot[j] * words + words - 2] : 0;  // (the sample coverage from sbits_cs: nothing fitted yet but invalid vectors)
  }
  [loop] for (uint step = 0; step < kShortSteps; ++step) {
    if (g_stop != 0) {
      break;  // (uniform: groupshared, read after a barrier)
    }
    // The largest stale gain T; recount those within T / 2 of it; if the best of them still beats T / 2
    // (every other gain is below that), it is the step's pick; else recount down to it and take the best
    if (k == 0) {
      g_sbest = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    uint local = 0;
    [unroll] for (uint j1 = 0; j1 < kPer; ++j1) {
      local = max(local, gain[j1]);
    }
    InterlockedMax(g_sbest, local);
    GroupMemoryBarrierWithGroupSync();
    const uint T = g_sbest, stop = g_stop;
    const uint lim = max(T / 2, kMinGain);
    GroupMemoryBarrierWithGroupSync();
    if (k == 0) {
      g_sbest = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    uint best = 0;
    if (!stop && T >= kMinGain) {
      [unroll] for (uint j2 = 0; j2 < kPer; ++j2) {
        if (gain[j2] >= lim) {
          gain[j2] = sgain(slot[j2], sw);
          best = gain[j2] >= kMinGain ? max(best, min(gain[j2], 65535u) << 16 | (0xffff - slot[j2])) : best;
        }
      }
    }
    InterlockedMax(g_sbest, best);
    GroupMemoryBarrierWithGroupSync();
    uint key = g_sbest;
    if ((key >> 16) < lim) {
      // (not certified: the others with stale gains in [best, lim) recounted too; no best: all from 2)
      const uint lim2 = key != 0 ? key >> 16 : kMinGain;
      GroupMemoryBarrierWithGroupSync();
      uint best2 = 0;
      [unroll] for (uint j3 = 0; j3 < kPer; ++j3) {
        if (gain[j3] >= lim2 && gain[j3] < lim) {
          gain[j3] = sgain(slot[j3], sw);
          best2 = gain[j3] >= kMinGain ? max(best2, min(gain[j3], 65535u) << 16 | (0xffff - slot[j3])) : best2;
        }
      }
      InterlockedMax(g_sbest, best2);
      GroupMemoryBarrierWithGroupSync();
      key = g_sbest;
    } else {
      GroupMemoryBarrierWithGroupSync();
      GroupMemoryBarrierWithGroupSync();
    }
    if (key != 0 && !stop) {
      const uint sl = 0xffff - (key & 0xffff);
      if (k < sw) {
        g_sfit[k] |= bits[sl * words + k];
      }
      if (k == 0) {
        const uint h = sl / 3;
        if (((g_keep[h >> 5] >> (h & 31)) & 1) == 0) {
          partials[h] = (float) sl;  // (the seed's first pick: the full refit starts from it; as a float value: denormals may flush)
        }
        g_keep[h >> 5] |= 1u << (h & 31);
      }
      [unroll] for (uint j4 = 0; j4 < kPer; ++j4) {
        gain[j4] = slot[j4] == sl ? 0 : gain[j4];
      }
    } else if (k == 0) {
      g_stop = 1;
    }
    GroupMemoryBarrierWithGroupSync();
  }
  if (k < 128) {
    lists[7 * nhyp + 2 * tsize + k] = g_keep[k];
  }
  if (k == 0) {
    state.Store(88, state.Load(56));  // (the list length, for keep_cs)
    state.Store(56, 0);
    state.Store(108, n);  // (candidates, for the report)
  }
}

// keep_cs (a thread per listed seed): the seeds not kept dropped, with their sample refits
[numthreads(64, 1, 1)] void keep_cs(uint3 id: SV_DispatchThreadID) {
  if (id.x >= state.Load(88)) {
    return;
  }
  const uint h = lists[id.x];
  const bool picked = ((lists[7 * nhyp + 2 * tsize + (h >> 5)] >> (h & 31)) & 1) != 0;
  if (!picked && h >= tile_hyps()) {
    return;  // (the previous frame's models are always kept)
  }
  if (!picked) {
    hyps[3 * h].dead = 1;
    hyps[3 * h + 1].dead = 1;
    hyps[3 * h + 2].dead = 1;
  } else {
    const uint sl = (uint) partials[h];
    if (sl != 3 * h) {
      hyps[3 * h] = hyps[sl];  // (a sample refit won: refit from it)
    }
  }
}

// ---- bits_cs: (words / 64, slots needing bitsets / 32) groups of 64 threads; each group loads
// 64 words' vectors (2048) once and evaluates 32 listed slots on them

static const uint kBitsSlots = 32;
groupshared uint g_bv[64 * 33];  // packed vectors x | y << 16, vector b of word k at k * 33 + b (padded: no bank conflicts)
groupshared uint g_bcr[64 * 33];  // c | r << 16 alike

[numthreads(64, 1, 1)] void bits_cs(uint3 gid: SV_GroupID, uint k: SV_GroupIndex) {
  const uint jw = gid.x * 64 + k, nb = state.Load(28);
  for (uint q = k; q < 64 * 32; q += 64) {
    const uint kk = gid.x * 64 * 32 + q;
    const int2 v = kk < nvec ? field[kk] : int2(kInvalid, 0);
    g_bv[q + q / 32] = (asuint(v.x) & 0xffff) | asuint(v.y) << 16;
    g_bcr[q + q / 32] = (kk % cols) | (kk / cols) << 16;
  }
  GroupMemoryBarrierWithGroupSync();
  if (jw >= words) {
    return;
  }
  for (uint i = 0; i < kBitsSlots; ++i) {
    const uint li = gid.y * kBitsSlots + i;
    if (li >= nb) {
      break;
    }
    const uint sl = lists[4 * nhyp + li];
    const Hyp hy = hyps[sl];
    uint word = 0;
    for (uint b2 = 0; b2 < 32; ++b2) {
      const uint pv = g_bv[k * 33 + b2], cr = g_bcr[k * 33 + b2];
      const int2 v = int2((int) (pv << 16) >> 16, (int) pv >> 16);
      word |= v.x != kInvalid && exact_f(hy.type, hy.lo, hy.hi, cr & 0xffff, cr >> 16, v) ? 1u << b2 : 0u;
    }
    bits[sl * words + jw] = word;
  }
}

static const uint kSums = 256;
groupshared float g_red[14][kSums];
groupshared float g_s[56];

void solve_refits(uint h, Hyp m) {
  const float fc = cols * 0.5, fr = rows * 0.5, mu = cols * B * 0.5, mw = rows * B * 0.5, sc = 1.0 / 1024;
  {
    // Affine (grid coordinates, centred)
    Hyp a = none_hyp();
    const float n = g_s[0], a01 = g_s[1], a02 = g_s[2], a11 = g_s[3], a22 = g_s[4], a12 = g_s[5];
    const float det = n * (a11 * a22 - a12 * a12) - a01 * (a01 * a22 - a12 * a02) + a02 * (a01 * a12 - a11 * a02);
    if (m.dead) {
      a.type = 3;
    } else if (n >= 3 && abs(det) > 1e-3 * n * n * n) {
      float res[6];
      [unroll] for (int comp = 0; comp < 2; ++comp) {
        const float b0 = g_s[6 + comp * 3], b1 = g_s[7 + comp * 3], b2 = g_s[8 + comp * 3];
        const float k0 = (b0 * (a11 * a22 - a12 * a12) - a01 * (b1 * a22 - a12 * b2) + a02 * (b1 * a12 - a11 * b2)) / det;
        const float p = (n * (b1 * a22 - a12 * b2) - b0 * (a01 * a22 - a12 * a02) + a02 * (a01 * b2 - b1 * a02)) / det;
        const float q = (n * (a11 * b2 - b1 * a12) - a01 * (a01 * b2 - b1 * a02) + b0 * (a01 * a12 - a11 * a02)) / det;
        res[comp * 3] = k0 - p * fc - q * fr;
        res[comp * 3 + 1] = p;
        res[comp * 3 + 2] = q;
      }
      a.type = 1;
      a.lo = float4(res[0], res[1], res[2], res[3]);
      a.hi = float4(res[4], res[5], 0, 0);
    } else if (n >= 1) {
      a.type = 0;
      a.lo = float4(g_s[6] / n, g_s[9] / n, 0, 0);
    }
    quantize_model(a);
    hyps[3 * h + 1] = a;
    float d[44];
    [unroll] for (uint q6 = 0; q6 < 44; ++q6) {
      d[q6] = g_s[12 + q6];
    }
    Hyp hm = none_hyp();
    float4 lo, hi;
    hm.type = !m.dead && n >= 16 && solve_hom(d, mu, mw, sc, lo, hi) ? 2 : 3;
    hm.lo = lo;
    hm.hi = hi;
    quantize_model(hm);
    hyps[3 * h + 2] = hm;
  }
}


// ---- sums_cs: (listed hypotheses, parts) groups


[numthreads(kSums, 1, 1)] void sums_cs(uint3 gid: SV_GroupID, uint k: SV_GroupIndex) {
  const uint h = lists[gid.x];
  const Hyp m = hyps[3 * h];
  const float fc = cols * 0.5, fr = rows * 0.5, mu = cols * B * 0.5, mw = rows * B * 0.5, sc = 1.0 / 1024;
  float s[56];
  [unroll] for (uint q = 0; q < 56; ++q) {
    s[q] = 0;
  }
  const uint sc_ = scols(), n_items = on_sample ? nsamples() : nvec;
  const uint share = (n_items + nparts - 1) / nparts, k0 = gid.y * share + k, k1 = min(n_items, (gid.y + 1) * share);
  if (!m.dead) {
    uint c = k0 % cols, r = k0 / cols;
    const uint dc = kSums % cols, dr = kSums / cols;
    for (uint kk = k0; kk < k1; kk += kSums) {
      if (on_sample) {
        c = (kk % sc_) * 4 + 2;
        r = (kk / sc_) * 4 + 2;
      }
      const int2 v = field[r * cols + c];
      if (v.x != kInvalid && near_f(m.type, m.lo, m.hi, c, r, v)) {
        const float cc = c - fc, rr = r - fr, X = v.x, Y = v.y;
        s[0] += 1;
        s[1] += cc;
        s[2] += rr;
        s[3] += cc * cc;
        s[4] += rr * rr;
        s[5] += cc * rr;
        s[6] += X;
        s[7] += cc * X;
        s[8] += rr * X;
        s[9] += Y;
        s[10] += cc * Y;
        s[11] += rr * Y;
        const float pu = (c * B + B / 2 - mu) * sc, pw = (r * B + B / 2 - mw) * sc;
        acc_dlt56(s, pu, pw, pu + X * 0.25 * sc, pw + Y * 0.25 * sc);
      }
      c += dc;
      r += dr;
      if (c >= cols) {
        c -= cols;
        r += 1;
      }
    }
  }
  [unroll] for (uint chunk = 0; chunk < 4; ++chunk) {
    [unroll] for (uint q4 = 0; q4 < 14; ++q4) {
      g_red[q4][k] = s[chunk * 14 + q4];
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint st = kSums / 2; st > 0; st >>= 1) {
      if (k < st) {
        [unroll] for (uint q5 = 0; q5 < 14; ++q5) {
          g_red[q5][k] += g_red[q5][k + st];
        }
      }
      GroupMemoryBarrierWithGroupSync();
    }
    if (k < 14) {
      g_s[chunk * 14 + k] = g_red[k][0];
    }
    GroupMemoryBarrierWithGroupSync();
  }
  if (nparts == 1) {
    if (k == 0) {
      solve_refits(h, m);
    }
  } else if (k < 56) {
    partials[(gid.x * nparts + gid.y) * 56 + k] = g_s[k];
  }
}

// solve_cs: a group per listed hypothesis; its parts' sums added, the refits solved
[numthreads(64, 1, 1)] void solve_cs(uint3 gid: SV_GroupID, uint k: SV_GroupIndex) {
  const uint h = lists[gid.x];
  if (k < 56) {
    float x = 0;
    for (uint p = 0; p < nparts; ++p) {
      x += partials[(gid.x * nparts + p) * 56 + k];
    }
    g_s[k] = x;
  }
  GroupMemoryBarrierWithGroupSync();
  if (k == 0) {
    solve_refits(h, hyps[3 * h]);
  }
}

// args_cs: one thread; the sums dispatch (listed hypotheses x parts)
[numthreads(1, 1, 1)] void args_cs() {
  state.Store3(92, uint3(state.Load(56), nparts, 1));
}

// ---- count_cs: a group per live slot

groupshared uint g_count;

[numthreads(256, 1, 1)] void count_cs(uint3 gid: SV_GroupID, uint k: SV_GroupIndex) {
  const uint sl = lists[nhyp + gid.x], cov = 3 * nhyp * words;
  if (k == 0) {
    g_count = 0;
  }
  GroupMemoryBarrierWithGroupSync();
  const uint dead = hyps[sl].dead, type = hyps[sl].type;
  uint n = 0;
  if (!dead) {
    for (uint jw = k; jw < words; jw += 256) {
      n += countbits(bits[sl * words + jw] & ~bits[cov + jw]);
    }
  }
  InterlockedAdd(g_count, n);
  GroupMemoryBarrierWithGroupSync();
  if (k == 0) {
    const uint need = type == 0 ? cover0 : type == 1 ? cover1 : cover2;
    counts[gid.x] = !dead && g_count >= need ? min(g_count, 65535u) << 16 | (0xffff - sl) : 0;  // (its key; 0: not eligible)
  }
}

// ---- select_cs: one group; lazy greedy over the round's top kTop candidates. A candidate's count
// only falls as vectors get fitted, so once one's current key beats the (kTop + 1)-th stale key,
// nothing outside the top can beat it: accepting several models per round gives exactly the
// sequence of one per round.

static const uint kTop = 8;
groupshared uint g_top[kTop + 1];  // stale keys, descending; [kTop]: the bound
groupshared uint g_tmax;
groupshared uint g_fresh[kTop];
groupshared uint g_in;  // bit c: candidate c still in play
groupshared uint g_take;
groupshared uint g_K;

[numthreads(256, 1, 1)] void select_cs(uint k: SV_GroupIndex) {
  const uint ns = state.Load(32), cov = 3 * nhyp * words;
  uint prev = 0xffffffffu;
  [loop] for (uint q = 0; q <= kTop; ++q) {
    if (k == 0) {
      g_tmax = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    uint best = 0;
    for (uint li = k; li < ns; li += 256) {
      const uint key = counts[li];
      best = key < prev && key > best ? key : best;  // (keys are unique: they end in the slot)
    }
    InterlockedMax(g_tmax, best);
    GroupMemoryBarrierWithGroupSync();
    if (k == 0) {
      g_top[q] = g_tmax;
    }
    GroupMemoryBarrierWithGroupSync();
    prev = g_top[q];
  }
  if (k == 0) {
    uint in_play = 0;
    for (uint c = 0; c < kTop; ++c) {
      in_play |= g_top[c] != 0 ? 1u << c : 0u;
    }
    g_in = in_play;
    g_K = state.Load(0);
  }
  GroupMemoryBarrierWithGroupSync();
  const uint bound = g_top[kTop];
  for (uint step = 0; step < kTop; ++step) {
    if (k < kTop) {
      g_fresh[k] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    const uint in_play = g_in;
    if (in_play != 0) {
      uint f[kTop];
      [unroll] for (uint c = 0; c < kTop; ++c) {
        f[c] = 0;
      }
      for (uint jw = k; jw < words; jw += 256) {
        const uint open = ~bits[cov + jw];
        [unroll] for (uint c2 = 0; c2 < kTop; ++c2) {
          if ((in_play >> c2) & 1) {
            f[c2] += countbits(bits[(0xffff - (g_top[c2] & 0xffff)) * words + jw] & open);
          }
        }
      }
      [unroll] for (uint c3 = 0; c3 < kTop; ++c3) {
        if (f[c3] != 0) {
          InterlockedAdd(g_fresh[c3], f[c3]);
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
    if (k == 0) {
      uint take = 0xffffffffu, best_key = 0, still = in_play;
      for (uint c = 0; c < kTop; ++c) {
        if ((still >> c) & 1) {
          const uint sl = 0xffff - (g_top[c] & 0xffff), type = hyps[sl].type;
          const uint need = type == 0 ? cover0 : type == 1 ? cover1 : cover2;
          if (g_fresh[c] < need) {
            still &= ~(1u << c);  // (never eligible again)
            continue;
          }
          const uint key = min(g_fresh[c], 65535u) << 16 | (g_top[c] & 0xffff);
          if (key > best_key) {
            best_key = key;
            take = c;
          }
        }
      }
      if (take != 0xffffffffu && best_key > bound && g_K < kMaxModels) {
        const uint sl = 0xffff - (g_top[take] & 0xffff);
        pal[g_K] = hyps[sl];
        hyps[sl].dead = 1;
        g_K += 1;
        still &= ~(1u << take);
      } else {
        take = 0xffffffffu;
        still = 0;  // (the rest waits for the next round's counts)
      }
      g_in = still;
      g_take = take;
    }
    GroupMemoryBarrierWithGroupSync();
    const uint take = g_take;
    if (take != 0xffffffffu) {
      const uint sl = 0xffff - (g_top[take] & 0xffff);
      for (uint jw2 = k; jw2 < words; jw2 += 256) {
        bits[cov + jw2] |= bits[sl * words + jw2];
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }
  if (k == 0) {
    state.Store(0, g_K);
    state.Store(8, state.Load(8) + 1);
    if (g_top[0] == 0 || g_K >= kMaxModels) {
      state.Store(4, 1);
      state.Store3(32, uint3(0, 1, 1));
      state.Store3(44, uint3(0, 1, 1));
    }
  }
}

// ---- tile_encode_cs: a group per tile

groupshared uint g_z[kMaxModels][64];  // residual widths bits(zx) | bits(zy) << 8 per model and vector
groupshared uint g_w1[kMaxModels][2];
groupshared uint g_wins[kMaxModels];
groupshared uint g_cand[6];
groupshared uint g_ncand;
groupshared uint g_pw[15][2];
groupshared uint g_psel[15][2];
groupshared uint g_qw[2];
groupshared uint g_qlab[4];

// The 15 pairs (a < b) of six candidates, in order (0,1) (0,2) ... (4,5); computed, not a static
// const table: indexing one here makes fxc fail ("invalid read of more specific predicate")
uint pair_a(uint p) {
  return p < 5 ? 0 : p < 9 ? 1 : p < 12 ? 2 : p < 14 ? 3 : 4;
}

uint pair_b(uint p) {
  const uint a = pair_a(p);
  const uint first = a == 0 ? 0 : a == 1 ? 5 : a == 2 ? 9 : a == 3 ? 12 : 14;
  return a + 1 + (p - first);
}

uint cost_of(uint z) {
  return (z & 0xff) + (z >> 8);
}

[numthreads(64, 1, 1)] void tile_encode_cs(uint3 gid: SV_GroupID, uint k: SV_GroupIndex) {
  const uint t = gid.y * tiles_x + gid.x;
  const Tile tl = tinfo[t];
  const uint K = state.Load(0);
  const uint c = gid.x * 8 + k % 8, r = gid.y * 8 + k / 8;
  const bool valid = ((k < 32 ? tl.a.x >> k : tl.a.y >> (k - 32)) & 1) != 0;
  const int2 v = valid ? field[r * cols + c] : int2(0, 0);
  if (k < kMaxModels) {
    g_w1[k][0] = 0;
    g_w1[k][1] = 0;
    g_wins[k] = 0;
  }
  if (k < 15) {
    g_pw[k][0] = 0;
    g_pw[k][1] = 0;
    g_psel[k][0] = 0;
    g_psel[k][1] = 0;
  }
  if (k < 4) {
    g_qlab[k] = 0;
  }
  if (k < 2) {
    g_qw[k] = 0;
  }
  GroupMemoryBarrierWithGroupSync();
  if (valid) {
    uint best_m = 0, best_c = 0xffffffff;
    for (uint m = 0; m < K; ++m) {
      const int2 p = eval_exact(pal[m], c, r);
      const uint zx = zigzag(quantize(v.x - p.x)), zy = zigzag(quantize(v.y - p.y));
      g_z[m][k] = bits_for(zx) | bits_for(zy) << 8;
      InterlockedMax(g_w1[m][0], zx);
      InterlockedMax(g_w1[m][1], zy);
      const uint cst = bits_for(zx) + bits_for(zy);
      if (cst < best_c) {
        best_c = cst;
        best_m = m;
      }
    }
    if (K > 0) {
      InterlockedAdd(g_wins[best_m], 1);
    }
  }
  GroupMemoryBarrierWithGroupSync();
  if (k == 0) {
    // The (up to) six models best for most vectors here
    uint taken = 0;
    const uint nc = min(K, 6u);
    for (uint q = 0; q < nc; ++q) {
      uint bm = 0, bw = 0;
      bool found = false;
      for (uint m = 0; m < K; ++m) {
        if (((taken >> m) & 1) == 0 && (!found || g_wins[m] > bw)) {
          bm = m;
          bw = g_wins[m];
          found = true;
        }
      }
      taken |= 1u << bm;
      g_cand[q] = bm;
    }
    g_ncand = nc;
  }
  GroupMemoryBarrierWithGroupSync();
  const uint nc = g_ncand;
  if (valid) {
    for (uint p = 0; p < 15; ++p) {
      if (pair_b(p) < nc) {
        const uint za = g_z[g_cand[pair_a(p)]][k], zb = g_z[g_cand[pair_b(p)]][k];
        const bool useb = cost_of(zb) < cost_of(za);
        const uint z = useb ? zb : za;
        InterlockedMax(g_pw[p][0], z & 0xff);
        InterlockedMax(g_pw[p][1], z >> 8);
        if (useb) {
          InterlockedOr(g_psel[p][k >> 5], 1u << (k & 31));
        }
      }
    }
    if (K >= 3) {
      const uint nm = min(K, 4u);
      uint bk = 0, bz = g_z[g_cand[0]][k];
      for (uint q = 1; q < nm; ++q) {
        const uint z = g_z[g_cand[q]][k];
        if (cost_of(z) < cost_of(bz)) {
          bk = q;
          bz = z;
        }
      }
      InterlockedMax(g_qw[0], bz & 0xff);
      InterlockedMax(g_qw[1], bz >> 8);
      InterlockedOr(g_qlab[k / 16], bk << (2 * (k % 16)));
    }
  }
  GroupMemoryBarrierWithGroupSync();
  if (k == 0) {
    const uint n = tl.a.z & 0xff;
    const bool has_mask = ((tl.a.z >> 8) & 1) != 0, aff_ok = ((tl.a.z >> 9) & 1) != 0;
    const uint mw = has_mask ? 2 : 0;
    uint mode = 0, size = 0xffffffff, word = 0, idx = 0;
    uint4 sel = uint4(0, 0, 0, 0);
    if (n > 0) {
      // (sizes as mvc2.h: residual words only when a width is nonzero, except modes 4-5)
      uint wx = tl.b.x & 31, wy = (tl.b.x >> 5) & 31;
      uint sz = mw + 1 + (wx + wy != 0 ? 1 + (n * (wx + wy) + 31) / 32 : 0);
      mode = 1;
      size = sz;
      word = tl.b.x;
      if (aff_ok) {
        wx = tl.b.w & 31;
        wy = (tl.b.w >> 5) & 31;
        sz = mw + 2 + (wx + wy != 0 ? 1 + (n * (wx + wy) + 31) / 32 : 0);
        if (sz < size) {
          mode = 2;
          size = sz;
          word = tl.b.w;
        }
      }
      for (uint m = 0; m < K; ++m) {
        wx = bits_for(g_w1[m][0]);
        wy = bits_for(g_w1[m][1]);
        sz = mw + (wx + wy != 0 ? 1 + (n * (wx + wy) + 31) / 32 : 0);
        if (sz < size && wx <= 16 && wy <= 16) {  // (palette plans: at most 16 bits a component, as local ones)
          mode = 3;
          size = sz;
          word = wx | wy << 5;
          idx = m;
        }
      }
      for (uint p = 0; p < 15; ++p) {
        if (pair_b(p) < nc) {
          wx = g_pw[p][0];
          wy = g_pw[p][1];
          sz = mw + 3 + (n * (wx + wy) + 31) / 32;
          if (sz < size && wx <= 16 && wy <= 16) {
            mode = 4;
            size = sz;
            word = g_cand[pair_a(p)] | g_cand[pair_b(p)] << 5 | wx << 10 | wy << 15;
            sel = uint4(g_psel[p][0], g_psel[p][1], 0, 0);
          }
        }
      }
      if (K >= 3) {
        wx = g_qw[0];
        wy = g_qw[1];
        sz = mw + 5 + (n * (wx + wy) + 31) / 32;
        if (sz < size && wx <= 16 && wy <= 16) {
          const uint nm = min(K, 4u);
          mode = 5;
          size = sz;
          word = g_cand[0] | g_cand[min(1u, nm - 1)] << 5 | g_cand[min(2u, nm - 1)] << 10 | g_cand[min(3u, nm - 1)] << 15 | wx << 20 | wy << 25;
          sel = uint4(g_qlab[0], g_qlab[1], g_qlab[2], g_qlab[3]);
        }
      }
    } else {
      size = 0;
    }
    const uint header = size | mode << 7 | (has_mask ? 1u << 10 : 0u) | (mode == 3 ? idx << 11 : 0u);
    tinfo[t].c = uint4(header, word, sel.x, sel.y);
    tinfo[t].d = uint4(sel.z, sel.w, 0, 0);
  }
}

// ---- scan_cs: one group

groupshared uint g_scan[1024];
groupshared uint g_carry;
groupshared uint g_pwords;
groupshared uint g_hb;

[numthreads(1024, 1, 1)] void scan_cs(uint k: SV_GroupIndex) {
  const uint K = state.Load(0);
  if (k == 0) {
    g_carry = 0;
    uint pw = 0, aff = 0, hom = 0;
    for (uint m = 0; m < K; ++m) {
      const Hyp md = pal[m];
      if (md.type == 0) {
        stream.Store(28 + pw * 4, (asuint(md.p0.x) & 0xffff) | asuint(md.p0.y) << 16);
      } else {
        const int ps[8] = {md.p0.x, md.p0.y, md.p0.z, md.p0.w, md.p1.x, md.p1.y, md.p1.z, md.p1.w};
        for (uint q = 0; q < words_of(md.type); ++q) {
          stream.Store(28 + (pw + q) * 4, asuint(ps[q]));
        }
      }
      pw += words_of(md.type);
      aff |= md.type == 1 ? 1u << m : 0u;
      hom |= md.type == 2 ? 1u << m : 0u;
    }
    g_pwords = pw;
    g_hb = 28 + pw * 4 + ((tiles * 2 + 3) & ~3u);
    stream.Store(16, K | pw << 8 | B << 24);
    stream.Store(20, aff);
    stream.Store(24, hom);
  }
  GroupMemoryBarrierWithGroupSync();
  for (uint base = 0; base < tiles; base += 1024) {
    const uint t = base + k;
    const uint size = t < tiles ? (tinfo[t].c.x & 0x7f) : 0;
    g_scan[k] = size;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 1; s < 1024; s <<= 1) {
      const uint add = k >= s ? g_scan[k - s] : 0;
      GroupMemoryBarrierWithGroupSync();
      g_scan[k] += add;
      GroupMemoryBarrierWithGroupSync();
    }
    if (t < tiles) {
      offs[t] = g_carry + g_scan[k] - size;
    }
    GroupMemoryBarrierWithGroupSync();
    if (k == 0) {
      g_carry += g_scan[1023];
    }
    GroupMemoryBarrierWithGroupSync();
  }
  const uint thb = 28 + g_pwords * 4;
  for (uint p = k; p * 2 < tiles; p += 1024) {
    const uint lo = tinfo[p * 2].c.x & 0xffff;
    const uint hi = p * 2 + 1 < tiles ? tinfo[p * 2 + 1].c.x & 0xffff : 0;
    stream.Store(thb + p * 4, lo | hi << 16);
  }
  if (k < K) {
    prior[k] = pal[k];  // (the next frame's first hypotheses)
  }
  if (k == 0) {
    state.Store(112, K);
    offs[tiles] = g_carry;
    offs[tiles + 1] = g_hb;
    stream.Store(0, 0x3243564du);
    stream.Store(4, cols | rows << 16);
    stream.Store(8, tiles_x | tiles_y << 16);
    stream.Store(12, g_carry | E << 24);
  }
}

// ---- pack_cs: a group per tile

groupshared uint g_words[64];

void put_bits(uint value, uint nb, uint bit) {  // (nb <= 16)
  if (nb > 0) {
    InterlockedOr(g_words[bit >> 5], value << (bit & 31));
    if ((bit & 31) + nb > 32) {
      InterlockedOr(g_words[(bit >> 5) + 1], value >> (32 - (bit & 31)));
    }
  }
}

[numthreads(64, 1, 1)] void pack_cs(uint3 gid: SV_GroupID, uint k: SV_GroupIndex) {
  const uint t = gid.y * tiles_x + gid.x;
  const Tile tl = tinfo[t];
  const uint header = tl.c.x, word = tl.c.y, size = header & 0x7f, mode = (header >> 7) & 7;
  const bool has_mask = ((header >> 10) & 1) != 0;
  const uint wx = mode >= 4 ? (word >> (mode == 4 ? 10 : 20)) & 31 : word & 31;
  const uint wy = mode >= 4 ? (word >> (mode == 4 ? 15 : 25)) & 31 : (word >> 5) & 31;
  const uint mw = has_mask ? 2 : 0;
  const bool res = mode >= 4 || wx + wy > 0;
  const uint hw = mw + (mode == 1 ? 1 : mode == 2 ? 2 : mode == 4 ? 3 : mode == 5 ? 5 : 0) + (mode <= 3 && res ? 1 : 0);
  const int i = (int) (k % 8), j = (int) (k / 8);
  const uint c = gid.x * 8 + i, r = gid.y * 8 + j;
  const uint m0 = tl.a.x, m1 = tl.a.y;
  const bool valid = mode != 0 && ((k < 32 ? m0 >> k : m1 >> (k - 32)) & 1) != 0;
  g_words[k] = 0;
  GroupMemoryBarrierWithGroupSync();
  if (valid) {
    const int2 v = field[r * cols + c];
    int2 p;
    if (mode == 1) {
      p = int2((int) (tl.a.w << 16) >> 16, (int) tl.a.w >> 16);
    } else if (mode == 2) {
      const uint g = tl.b.z;
      p = int2(local_model((int) (tl.b.y << 16) >> 16, sx8(g), sx8(g >> 8), i, j), local_model((int) tl.b.y >> 16, sx8(g >> 16), sx8(g >> 24), i, j));
    } else {
      uint m;
      if (mode == 3) {
        m = header >> 11;
      } else if (mode == 4) {
        m = ((k < 32 ? tl.c.z >> k : tl.c.w >> (k - 32)) & 1) ? (word >> 5) & 31 : word & 31;
      } else {
        const uint labw = k < 16 ? tl.c.z : k < 32 ? tl.c.w : k < 48 ? tl.d.x : tl.d.y;
        m = (word >> (5 * ((labw >> (2 * (k % 16))) & 3))) & 31;
      }
      p = eval_exact(pal[m], c, r);
    }
    const uint below = k < 32 ? countbits(m0 & ((1u << k) - 1)) : countbits(m0) + countbits(m1 & ((1u << (k - 32)) - 1));
    const uint zx = zigzag(quantize(v.x - p.x)), zy = zigzag(quantize(v.y - p.y));
    const uint bit = below * (wx + wy);
    put_bits(zx, wx, bit);
    put_bits(zy, wy, bit + wx);
  }
  GroupMemoryBarrierWithGroupSync();
  if (mode != 0) {
    const uint n = countbits(m0) + countbits(m1);
    const uint body = res ? (n * (wx + wy) + 31) / 32 : 0;
    const uint base = offs[tiles + 1] + offs[t] * 4;
    if (k < body) {
      stream.Store(base + (hw + k) * 4, g_words[k]);
    }
    if (k == 0) {
      uint wi = 0;
      if (has_mask) {
        stream.Store(base + wi++ * 4, m0);
        stream.Store(base + wi++ * 4, m1);
      }
      if (mode == 1) {
        stream.Store(base + wi++ * 4, tl.a.w);
      } else if (mode == 2) {
        stream.Store(base + wi++ * 4, tl.b.y);
        stream.Store(base + wi++ * 4, tl.b.z);
      }
      if (mode <= 3 && res) {
        stream.Store(base + wi++ * 4, word & 0x3ff);
      }
      if (mode == 4) {
        stream.Store(base + wi++ * 4, word);
        stream.Store(base + wi++ * 4, tl.c.z);
        stream.Store(base + wi++ * 4, tl.c.w);
      }
      if (mode == 5) {
        stream.Store(base + wi++ * 4, word);
        stream.Store(base + wi++ * 4, tl.c.z);
        stream.Store(base + wi++ * 4, tl.c.w);
        stream.Store(base + wi++ * 4, tl.d.x);
        stream.Store(base + wi++ * 4, tl.d.y);
      }

    }
  }
}
