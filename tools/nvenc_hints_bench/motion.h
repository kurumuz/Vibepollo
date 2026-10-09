#pragma once
// Synthetic game-like scenes with exact per-pixel motion, and NVENC hints
// derived from that motion field the way the real pipeline would derive them
// from DLSS motion vectors.
//
// A scene is a stack of layers. Each world layer is seen through its own
// camera (rotation and zoom shared, translation scaled by depth for
// parallax); sprites move in screen space; the HUD is static. The motion of
// a pixel is "where the surface visible there was in the previous frame",
// in screen pixels: previous position minus current position, which is the
// encoder's convention (the vector points from the current block to its
// reference).
#ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
  #define NOMINMAX
#endif
#include <windows.h>
#include <nvEncodeAPI.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace motion {
static_assert(sizeof(NVENC_EXTERNAL_ME_HINT) == 4, "NVENC integer hint ABI");
static_assert(sizeof(NVENC_EXTERNAL_ME_SB_HINT) == 6, "NVENC AV1 hint ABI: three 16-bit words");
static_assert(sizeof(NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE) == 16, "NVENC counts ABI");

constexpr double kPi = 3.14159265358979323846;

inline uint32_t hash(uint32_t v) { v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; return v ^ (v >> 16); }
inline uint32_t hash2(int64_t i, int64_t j, uint32_t seed) {
    return hash(uint32_t(i) * 0x9e3779b9u ^ hash(uint32_t(j) + seed * 0x632be5abu));
}
inline double unit(uint32_t h) { return double(h & 0xffff) / 65535.0; }
/// Bilinear value noise on a `cell`-pixel lattice, 0..1 (continuous, so sub-pixel motion is smooth)
inline double vnoise(double x, double y, double cell, uint32_t seed) {
    double gx = x / cell, gy = y / cell, fx0 = std::floor(gx), fy0 = std::floor(gy);
    int64_t ix = int64_t(fx0), iy = int64_t(fy0);
    double fx = gx - fx0, fy = gy - fy0;
    double a = unit(hash2(ix, iy, seed)), b = unit(hash2(ix + 1, iy, seed));
    double c = unit(hash2(ix, iy + 1, seed)), d = unit(hash2(ix + 1, iy + 1, seed));
    double top = a + (b - a) * fx, bottom = c + (d - c) * fx;
    return top + (bottom - top) * fy;
}
inline double fract(double x) { return x - std::floor(x); }
inline uint8_t clamp8(double v) { return uint8_t(std::clamp(v, 0.0, 255.0)); }

// Textures, in the layer's world coordinates
/// Coarse-to-fine structure, like terrain or sky
inline uint8_t tex_multiscale(double x, double y) {
    return clamp8(16 + 96 * vnoise(x, y, 256, 1) + 56 * vnoise(x, y, 64, 2) + 32 * vnoise(x, y, 16, 3)
                  + 24 * vnoise(x, y, 4, 4) + 12 * vnoise(x, y, 1, 5));
}
/// The original control: blocky fine noise only (16/4/1 px), no coarse structure
inline double noise_raw(double x, double y) {
    int64_t ix = int64_t(std::floor(x)), iy = int64_t(std::floor(y));
    auto div = [](int64_t v, int64_t n) { return v >= 0 ? v / n : -((-v + n - 1) / n); };
    uint32_t a = hash2(div(ix, 16), div(iy, 16), 11), b = hash2(div(ix, 4), div(iy, 4), 12), c = hash2(ix, iy, 13);
    return 16 + (a % 112) + (b % 80) + (c % 28);
}
inline uint8_t tex_noise(double x, double y) { return uint8_t(noise_raw(x, y)); }
/// A controlled pair, identical fine detail (0.6 x the control's), with and
/// without coarse structure: the only difference is what a coarse search can see
inline uint8_t tex_fine(double x, double y) { return clamp8(0.6 * noise_raw(x, y) + 51); }
inline uint8_t tex_fine_coarse(double x, double y) {
    return clamp8(0.6 * noise_raw(x, y) + 102 * (0.6 * vnoise(x, y, 256, 1) + 0.4 * vnoise(x, y, 64, 2)));
}
/// Leaves: sharp-edged blobs and fine detail, nothing coarser than ~8 px
inline uint8_t tex_foliage(double x, double y) {
    double leaf = vnoise(x, y, 8, 21) > 0.55 ? 60 : 0;
    double vein = vnoise(x, y, 3, 22) > 0.6 ? -35 : 0;
    return clamp8(40 + leaf + vein + 50 * vnoise(x, y, 2, 23) + 45 * vnoise(x, y, 1, 24));
}
/// A grate: 10 px vertical bars, 14 px slats: repetitive, so a search can lock onto the wrong period
inline uint8_t tex_grate(double x, double y) {
    double v = fract(x / 10) < 0.5 ? 190 : 55;
    if (fract(y / 14) < 0.3) v -= 35;
    return clamp8(v + 20 * vnoise(x, y, 2, 31));
}
/// Top-down track: asphalt (fine grain), dashed lane lines every 360 px
/// (60 px dashes, 160 px period: repetitive along the motion), grass patches
inline uint8_t tex_track(double x, double y) {
    if (vnoise(x, y, 900, 61) > 0.64) return tex_foliage(x, y);
    double v = 70 + 30 * vnoise(x, y, 3, 62) + 18 * vnoise(x, y, 1, 63) + 14 * vnoise(x, y, 120, 64);
    if (std::abs(fract(y / 360) - 0.5) * 360 < 7 && fract(x / 160) < 0.375) v = 215 + 15 * vnoise(x, y, 2, 65);
    return clamp8(v);
}
/// Clouds seen from above: soft, bright, low contrast
inline uint8_t tex_cloud(double x, double y) { return clamp8(185 + 45 * vnoise(x, y, 90, 71) + 15 * vnoise(x, y, 12, 72)); }
/// Chase-camera ground, in road units (X across, Z along; 1 = the camera's
/// height): lanes 1.2 wide, dashed centre lines (1.5 on, 1.5 off), solid
/// edges, grass and roadside posts beyond. Texels are fine near the camera.
inline uint8_t tex_road(double X, double Z) {
    double ax = std::abs(X), tx = X * 250, tz = Z * 250;
    if (ax > 3.6) {
        if (fract(Z / 4) < 0.04 && ax < 3.9) return 235;  // posts
        return tex_foliage(tx, tz);
    }
    double v = 75 + 28 * vnoise(tx, tz, 3, 81) + 16 * vnoise(tx, tz, 1, 82) + 12 * vnoise(tx, tz, 60, 83);
    if (std::abs(ax - 3.4) < 0.08) v = 225;
    for (double lane : {0.0, 1.2, -1.2})
        if (std::abs(X - lane) < 0.05 && fract(Z / 3) < 0.5) v = 225;
    return clamp8(v);
}
/// Sky: a gradient with clouds, in screen pixels shifted by the yaw
inline uint8_t tex_sky(double x, double y) { return clamp8(150 + 40 * vnoise(x, y, 220, 91) + 15 * vnoise(x, y, 30, 92) - y * 0.03); }
inline uint8_t tex_hud(double x, double y) { return uint8_t(48 + (((int(x) / 8) ^ (int(y) / 8)) & 1) * 160); }

using Texture = uint8_t (*)(double, double);
inline Texture texture_from(const std::string& s) {
    if (s == "multiscale") return tex_multiscale;
    if (s == "noise") return tex_noise;
    if (s == "fine") return tex_fine;
    if (s == "finecoarse") return tex_fine_coarse;
    if (s == "foliage") return tex_foliage;
    throw std::runtime_error("unknown texture " + s);
}

enum class Kind { pan, orbit, parallax, objects, foliage, game, topdown, chase };
inline Kind kind_from(const std::string& s) {
    if (s == "pan") return Kind::pan;
    if (s == "orbit") return Kind::orbit;
    if (s == "parallax") return Kind::parallax;
    if (s == "objects") return Kind::objects;
    if (s == "foliage") return Kind::foliage;
    if (s == "game") return Kind::game;
    if (s == "topdown") return Kind::topdown;
    if (s == "chase") return Kind::chase;
    throw std::runtime_error("unknown scene " + s);
}

enum Layer : int { kFar = 0, kMid = 1, kNear = 2, kHud = 3, kSprite0 = 8 };
/// Scoring regions: where the pixel's surface was visible in the previous
/// frame (overlap), newly visible (exposed), the HUD, and sprites
enum Region : uint8_t { kOverlap = 0, kExposed = 1, kHudRegion = 2, kSpriteRegion = 3, kRegions = 4 };
inline const char* region_name(int r) { return r == kOverlap ? "overlap" : r == kExposed ? "exposed" : r == kHudRegion ? "hud" : "sprites"; }

/// verified: the game-like vectors (on), each checked against the frames
/// being encoded: the block keeps its vector only if it predicts the block
/// from the previous frame better than zero motion does
enum class Hints { off, on, oracle, verified, wrong, empty, none, zero };  // none: every block invalid; zero: every block (0,0)

/// The luma of the frame being encoded and of its reference (for verified hints)
struct Frames { const uint8_t* cur; const uint8_t* prev; int w, h; };
/// Sum of absolute differences between the size x size block at (x0, y0) of
/// the current frame and the previous frame displaced by (mx, my), clipped
/// to the frame, reference samples clamped to the edge (as encoders pad)
inline uint64_t sad(const Frames& f, int x0, int y0, int size, int mx, int my) {
    uint64_t sum = 0;
    int x1 = std::min(x0 + size, f.w), y1 = std::min(y0 + size, f.h);
    for (int y = y0; y < y1; ++y) {
        int ry = std::clamp(y + my, 0, f.h - 1);
        const uint8_t* c = f.cur + size_t(y) * f.w;
        const uint8_t* r = f.prev + size_t(ry) * f.w;
        for (int x = x0; x < x1; ++x) sum += uint64_t(std::abs(int(c[x]) - int(r[std::clamp(x + mx, 0, f.w - 1)])));
    }
    return sum;
}
/// verified: keep (mx, my) only if it beats zero motion (ties go to zero, the cheaper vector)
inline void verify(const Frames* f, int x0, int y0, int size, int& mx, int& my) {
    if (!f || (!mx && !my)) return;
    if (sad(*f, x0, y0, size, 0, 0) <= sad(*f, x0, y0, size, mx, my)) mx = my = 0;
}

struct Vec { double x, y; };

class Scene {
public:
    static constexpr int kSprite = 48;
    static constexpr int kHudX0 = 64, kHudX1 = 704, kHudY0 = 64, kHudY1 = 320;

    /// `speed`: the scene's motion scale in screen px/frame (see cam());
    /// `far_texture`: the far layer's texture (the foliage scene forces foliage)
    Scene(Kind kind, double speed, double dy, int w, int h, Texture far_texture)
        : kind_(kind), s_(speed), dy_(dy), w_(w), h_(h), c_{w / 2.0, h / 2.0} {
        far_tex_ = kind == Kind::foliage ? tex_foliage : far_texture;
        mid_ = kind == Kind::parallax || kind == Kind::game || kind == Kind::chase;
        near_ = kind == Kind::parallax || kind == Kind::game || kind == Kind::foliage || kind == Kind::topdown;
        int sprites = kind == Kind::objects ? 16 : kind == Kind::game ? 8 : kind == Kind::topdown ? 7 : kind == Kind::chase ? 4 : 0;
        for (int i = 0; i < sprites; ++i) {
            double angle = 2 * kPi * unit(hash(uint32_t(i) * 7 + 1));
            double v = s_ * (0.5 + unit(hash(uint32_t(i) * 7 + 2)));
            if (kind == Kind::topdown || kind == Kind::chase) {
                // Cars: near the player's speed, so they drift on screen
                // (topdown: the first is the player, held at the centre)
                v = i == 0 && kind == Kind::topdown ? 0 : s_ * (kind == Kind::chase ? 0.03 : 0.08) * (1 + 2 * unit(hash(uint32_t(i) * 7 + 2)));
            }
            Vec p0 {unit(hash(uint32_t(i) * 7 + 3)) * (w_ - kSprite), unit(hash(uint32_t(i) * 7 + 4)) * (h_ - kSprite)};
            if (kind == Kind::topdown && i == 0) p0 = {c_.x - kSprite / 2.0, c_.y - kSprite / 2.0};
            if (kind == Kind::chase) p0.y = horizon() + 40 + unit(hash(uint32_t(i) * 7 + 4)) * 0.25 * (h_ - horizon());
            sprites_.push_back({p0, {v * std::cos(angle), v * std::sin(angle)}, uint32_t(40 + i)});
        }
    }

    int layer_at(int t, Vec p, bool hud = true) const {
        if (hud && in_hud(p)) return kHud;
        for (int i = int(sprites_.size()) - 1; i >= 0; --i) {
            Vec o = sprite_pos(i, t);
            if (p.x >= o.x && p.x < o.x + kSprite && p.y >= o.y && p.y < o.y + kSprite) return kSprite0 + i;
        }
        if (kind_ == Kind::chase) return p.y < horizon() ? kMid : kFar;
        if (kind_ == Kind::topdown) return near_ && cloud(to_world(t, kNear, p)) ? kNear : kFar;
        if (near_ && pillar(to_world(t, kNear, p))) return kNear;
        if (mid_ && bush(to_world(t, kMid, p))) return kMid;
        return kFar;
    }
    /// Where the surface on `layer` at screen position p in frame t was in frame t-1
    Vec previous(int t, Vec p, int layer) const {
        if (layer == kHud) return p;
        if (layer >= kSprite0) {
            Vec a = sprite_pos(layer - kSprite0, t), b = sprite_pos(layer - kSprite0, t - 1);
            return {p.x + b.x - a.x, p.y + b.y - a.y};
        }
        return to_screen(t - 1, layer, to_world(t, layer, p));
    }
    bool inside(Vec p) const { return p.x >= 0 && p.y >= 0 && p.x < w_ && p.y < h_; }
    /// The pixel's value in frame t and its scoring region
    uint8_t shade(int t, Vec p, uint8_t* region = nullptr) const {
        int layer = layer_at(t, p);
        if (region) {
            if (layer == kHud) *region = kHudRegion;
            else if (layer >= kSprite0) *region = kSpriteRegion;
            else {
                Vec q = previous(t, p, layer);
                *region = inside(q) && layer_at(t - 1, q) == layer ? kOverlap : kExposed;
            }
        }
        if (layer == kHud) return tex_hud(p.x, p.y);
        if (layer >= kSprite0) {
            const auto& s = sprites_[size_t(layer - kSprite0)];
            Vec o = sprite_pos(layer - kSprite0, t);
            double u = p.x - o.x, v = p.y - o.y;
            double base = ((int(u / 8) + int(v / 8)) & 1) ? 225 : 35;
            return clamp8(base + 25 * vnoise(u, v, 2, s.seed) - 12);
        }
        Vec w = to_world(t, layer, p);
        if (kind_ == Kind::topdown) return layer == kNear ? tex_cloud(w.x, w.y) : tex_track(w.x, w.y);
        if (kind_ == Kind::chase) {
            if (layer == kMid) return tex_sky(w.x, w.y);
            // Fog toward the horizon (games fade the distance; it also keeps
            // the far road from aliasing)
            double z = chase_depth(p.y), fog = std::exp(-z / 18);
            return clamp8(fog * tex_road(w.x, w.y) + (1 - fog) * 160);
        }
        if (layer == kNear) return tex_grate(w.x, w.y);
        if (layer == kMid) return tex_foliage(w.x, w.y);
        return far_tex_(w.x, w.y);
    }
    Vec to_world_for_test(int t, int layer, Vec p) const { return to_world(t, layer, p); }
    int width() const { return w_; }
    int height() const { return h_; }

private:
    struct Sprite { Vec p0, v; uint32_t seed; };
    struct Cam { double theta, zoom; Vec t; };

    bool in_hud(Vec p) const { return p.x >= kHudX0 && p.x < kHudX1 && p.y >= kHudY0 && p.y < kHudY1; }
    static bool cloud(Vec w) { return vnoise(w.x, w.y, 700, 73) > 0.6; }
    double horizon() const { return 0.38 * h_; }
    /// Chase: focal length in px, and the road distance seen on row y
    double focal() const { return 0.55 * h_; }
    double chase_depth(double y) const { return focal() / std::max(y - horizon(), 1e-3); }
    /// Chase: forward speed in camera heights per frame (the bottom row moves s px/frame)
    double chase_speed() const { double b = h_ - horizon(); return s_ * focal() / (b * b); }
    /// Chase: yaw (gentle curves), radians
    double chase_yaw(int t) const { return 0.12 * std::sin(2 * kPi * t / 240.0); }
    static bool pillar(Vec w) { return fract(w.x / 700) * 700 < 180; }
    bool bush(Vec w) const {
        double edge = 0.5 * h_ + 140 * (vnoise(w.x, 0, 300, 51) - 0.5) + 36 * (vnoise(w.x, 0, 40, 52) - 0.5);
        return w.y > edge;
    }
    static double bounce(double x, double range) {
        double m = std::fmod(x, 2 * range);
        if (m < 0) m += 2 * range;
        return m <= range ? m : 2 * range - m;
    }
    Vec sprite_pos(int i, int t) const {
        const auto& s = sprites_[size_t(i)];
        return {bounce(s.p0.x + s.v.x * t, w_ - kSprite), bounce(s.p0.y + s.v.y * t, h_ - kSprite)};
    }
    double depth(int layer) const {
        if (kind_ == Kind::topdown) return layer == kNear ? 1.7 : 1;
        if (kind_ != Kind::parallax && kind_ != Kind::game) return 1;
        return layer == kFar ? 0.1 : layer == kMid ? 0.35 : 1;
    }
    /// pan/foliage: translation s px/frame (and dy). orbit: roll with ~1.1*s
    /// px/frame at the corners, zoom oscillating with peak rate s/6000 per
    /// frame. parallax: far/mid/near layers at 0.1/0.35/1 * s. objects: slow
    /// pan (s/8) under sprites at 0.5..1.5 * s in all directions. game:
    /// parallax plus a gentle oscillating roll, sprites and HUD.
    Cam cam(int t, int layer) const {
        const double period = 120;
        Cam c{0, 1, {0, 0}};
        switch (kind_) {
            case Kind::pan: case Kind::foliage: c.t = {s_ * t, dy_ * t}; break;
            case Kind::orbit:
                c.theta = s_ / 2000 * t;
                c.zoom = std::exp(s_ / 6000 * period / (2 * kPi) * std::sin(2 * kPi * t / period));
                break;
            case Kind::parallax: c.t = {s_ * t * depth(layer), 0}; break;
            case Kind::objects: c.t = {s_ / 8 * t, 0}; break;
            case Kind::game:
                c.t = {s_ * t * depth(layer), 0};
                c.theta = s_ / 8000 * period / (2 * kPi) * std::sin(2 * kPi * t / period);
                break;
            case Kind::topdown: {
                // Following the player along a weaving road: forward s px/frame,
                // sideways up to 0.3 s; the clouds are nearer the camera (1.7x)
                double side = 0.3 * s_ * 240 / (2 * kPi) * std::sin(2 * kPi * t / 240.0);
                c.t = {s_ * t * depth(layer), side * depth(layer)};
                break;
            }
            case Kind::chase: break;  // (see to_world)
        }
        return c;
    }
    Vec to_world(int t, int layer, Vec p) const {
        if (kind_ == Kind::chase) {
            // Sky: at infinity, only the yaw moves it (screen px + yaw * focal)
            if (layer == kMid) return {p.x + chase_yaw(t) * focal(), p.y};
            // Ground: the camera ray to the road plane, rotated by the yaw,
            // offset by the distance driven (road units)
            double z = chase_depth(p.y), x = (p.x - c_.x) * z / focal();
            double yaw = chase_yaw(t), cs = std::cos(yaw), sn = std::sin(yaw);
            return {x * cs + z * sn, -x * sn + z * cs + chase_speed() * t};
        }
        Cam c = cam(t, layer);
        double qx = p.x - c_.x, qy = p.y - c_.y, cs = std::cos(c.theta), sn = std::sin(c.theta);
        return {c_.x + (qx * cs - qy * sn) / c.zoom + c.t.x, c_.y + (qx * sn + qy * cs) / c.zoom + c.t.y};
    }
    Vec to_screen(int t, int layer, Vec w) const {
        if (kind_ == Kind::chase) {
            if (layer == kMid) return {w.x - chase_yaw(t) * focal(), w.y};
            double yaw = chase_yaw(t), cs = std::cos(yaw), sn = std::sin(yaw);
            double wx = w.x, wz = w.y - chase_speed() * t;
            double x = wx * cs - wz * sn, z = wx * sn + wz * cs;
            if (z <= 1e-3) return {-1e9, -1e9};  // (behind the camera: off screen)
            return {c_.x + x * focal() / z, horizon() + focal() / z};
        }
        Cam c = cam(t, layer);
        double qx = (w.x - c_.x - c.t.x) * c.zoom, qy = (w.y - c_.y - c.t.y) * c.zoom;
        double cs = std::cos(c.theta), sn = std::sin(c.theta);
        return {c_.x + qx * cs + qy * sn, c_.y - qx * sn + qy * cs};
    }

    Kind kind_;
    double s_, dy_;
    int w_, h_;
    Vec c_;
    Texture far_tex_;
    bool mid_ = false, near_ = false;
    std::vector<Sprite> sprites_;
};

/// Rows [0, n) split over all processors
inline void parallel_rows(int n, const std::function<void(int, int)>& fn) {
    static const int threads = std::max(1, int(GetActiveProcessorCount(ALL_PROCESSOR_GROUPS)));
    int count = std::min(threads, n);
    struct Job { const std::function<void(int, int)>* fn; int y0, y1; };
    std::vector<Job> jobs(static_cast<size_t>(count));
    std::vector<HANDLE> handles;
    for (int i = 0; i < count; ++i) jobs[size_t(i)] = {&fn, n * i / count, n * (i + 1) / count};
    for (int i = 1; i < count; ++i) {
        HANDLE h = CreateThread(nullptr, 0, [](void* p) -> DWORD { auto* j = static_cast<Job*>(p); (*j->fn)(j->y0, j->y1); return 0; },
                                &jobs[size_t(i)], 0, nullptr);
        if (!h) {
            for (HANDLE started : handles) { WaitForSingleObject(started, INFINITE); CloseHandle(started); }
            throw std::runtime_error("CreateThread");
        }
        handles.push_back(h);
    }
    fn(jobs[0].y0, jobs[0].y1);
    for (HANDLE h : handles) { WaitForSingleObject(h, INFINITE); CloseHandle(h); }
}

/// Frame t as NV12 (luma from the scene, flat chroma), and each pixel's region
inline void frame(std::vector<uint8_t>& dst, std::vector<uint8_t>& regions, const Scene& scene, int t) {
    int w = scene.width(), h = scene.height();
    dst.resize(size_t(w) * h * 3 / 2);
    regions.resize(size_t(w) * h);
    parallel_rows(h, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < w; ++x) {
                size_t i = size_t(y) * w + x;
                dst[i] = scene.shade(t, {x + 0.5, y + 0.5}, &regions[i]);
            }
    });
    std::fill(dst.begin() + size_t(w) * h, dst.end(), uint8_t(128));
}

struct Block { double x, y; bool valid; };
/// One vector for a size x size block at (x0, y0), from a k x k grid of
/// samples of the per-pixel field: the medoid of the valid samples (robust
/// to mixed surfaces at edges; a GPU pass could do the same on DLSS vectors).
/// Hints::on knows only what a game's motion vectors know: the motion of the
/// surface under the pixel, ignoring the HUD, and the frame edge. Hints::oracle
/// also knows disocclusion and that the HUD is static.
inline Block block(const Scene& scene, int t, int x0, int y0, int size, int k, Hints mode) {
    if (mode == Hints::none) return {0, 0, false};
    if (mode == Hints::zero) return {0, 0, true};
    Vec samples[64];
    int n = 0, total = 0;
    bool oracle = mode == Hints::oracle;  // verified samples like on
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < k; ++i) {
            Vec p{x0 + (i + 0.5) * size / k, y0 + (j + 0.5) * size / k};
            if (!scene.inside(p)) continue;
            ++total;
            int layer = scene.layer_at(t, p, oracle);
            Vec q = scene.previous(t, p, layer);
            if (!scene.inside(q)) continue;
            if (oracle && scene.layer_at(t - 1, q, true) != layer) continue;
            samples[n++] = {q.x - p.x, q.y - p.y};
        }
    if (n == 0 || 2 * n < total) return {0, 0, false};
    int best = 0;
    double best_cost = 1e300;
    for (int a = 0; a < n; ++a) {
        double cost = 0;
        for (int b = 0; b < n; ++b) cost += std::abs(samples[a].x - samples[b].x) + std::abs(samples[a].y - samples[b].y);
        if (cost < best_cost) { best_cost = cost; best = a; }
    }
    double sign = mode == Hints::wrong ? -1 : 1;
    return {sign * samples[best].x, sign * samples[best].y, true};
}

/// H.264/HEVC hints for frame t (motion to t-1): one L0 candidate per 16x16
/// macroblock, or (block8) four 8x8 candidates per macroblock (raster within
/// it), macroblocks in raster order (ctu_order: 2x2 macroblock groups, the
/// HEVC 32x32 CTU experiment)
inline void mb_hints(std::vector<NVENC_EXTERNAL_ME_HINT>& out, const Scene& scene, int t, Hints mode, bool block8, bool ctu_order, const Frames* frames = nullptr) {
    const int cols = (scene.width() + 15) / 16, rows = (scene.height() + 15) / 16;
    const int per = block8 ? 4 : 1;
    std::vector<int> order;
    order.reserve(size_t(cols) * rows);
    if (!ctu_order) {
        for (int y = 0; y < rows; ++y) for (int x = 0; x < cols; ++x) order.push_back(y * cols + x);
    } else {
        for (int y = 0; y < rows; y += 2) for (int x = 0; x < cols; x += 2)
            for (int yy = 0; yy < 2; ++yy) for (int xx = 0; xx < 2; ++xx)
                if (y + yy < rows && x + xx < cols) order.push_back((y + yy) * cols + x + xx);
    }
    out.assign(order.size() * per, NVENC_EXTERNAL_ME_HINT{});
    parallel_rows(int(order.size()), [&](int i0, int i1) {
        for (int i = i0; i < i1; ++i) {
            int mb = order[size_t(i)], bx = (mb % cols) * 16, by = (mb / cols) * 16;
            for (int part = 0; part < per; ++part) {
                int px = block8 ? bx + (part & 1) * 8 : bx, py = block8 ? by + (part >> 1) * 8 : by, size = block8 ? 8 : 16;
                Block b = block(scene, t, px, py, size, block8 ? 2 : 4, mode);
                int mx = int(std::lround(b.x)), my = int(std::lround(b.y));
                bool valid = b.valid && mx >= -2048 && mx <= 2047 && my >= -512 && my <= 511;
                if (valid && mode == Hints::verified) verify(frames, px, py, size, mx, my);
                auto& h = out[size_t(i) * per + part];
                h.mvx = valid ? mx : 0;
                h.mvy = valid ? my : 0;
                h.refidx = valid ? 0 : -1;  // signed 5-bit field: bit pattern 31 = invalid
                h.dir = 0;
                h.partType = block8 ? -1 : 0;  // signed 2-bit field: bit pattern 3 = 8x8
                h.lastofPart = -1;  // signed 1-bit field: bit set; every 8x8 is its own partition
                h.lastOfMB = part == per - 1 ? -1 : 0;
            }
        }
    });
}
/// AV1: superblocks in raster order, each split into cu x cu coding units
/// (64, 32, 16 or 8), one quarter-pel L0 candidate per CU, CUs in quadtree
/// (Z) order within the superblock; CUs starting outside the frame are left
/// out, so edge superblocks carry fewer
inline void sb_hints(std::vector<NVENC_EXTERNAL_ME_SB_HINT>& out, const Scene& scene, int t, Hints mode, int cu = 64, const Frames* frames = nullptr) {
    const int cols = (scene.width() + 63) / 64, rows = (scene.height() + 63) / 64;
    const int per_side = 64 / cu, per_sb = per_side * per_side;
    const int size_code = cu == 8 ? 0 : cu == 16 ? 1 : cu == 32 ? 2 : 3;
    // Z order: de-interleave the CU index bits into (x, y)
    auto z = [](int i, int& x, int& y) {
        x = y = 0;
        for (int b = 0; b < 3; ++b) { x |= ((i >> (2 * b)) & 1) << b; y |= ((i >> (2 * b + 1)) & 1) << b; }
    };
    std::vector<std::vector<NVENC_EXTERNAL_ME_SB_HINT>> per_row(static_cast<size_t>(rows));
    parallel_rows(rows, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            auto& row = per_row[size_t(y)];
            row.clear();
            for (int x = 0; x < cols; ++x) {
                size_t first = row.size();
                for (int i = 0; i < per_sb; ++i) {
                    int cx, cy;
                    z(i, cx, cy);
                    int px = x * 64 + cx * cu, py = y * 64 + cy * cu;
                    if (px >= scene.width() || py >= scene.height()) continue;
                    Block b = block(scene, t, px, py, cu, cu >= 32 ? 8 : cu == 16 ? 4 : 2, mode);
                    int mx = int(std::lround(b.x * 4)), my = int(std::lround(b.y * 4));
                    bool valid = b.valid && std::abs(mx) <= 4092 && std::abs(my) <= 2044;
                    if (valid && mode == Hints::verified) {
                        // Checked at full-pel (the rounded vector); zero wins -> send zero
                        int fx = int(std::lround(b.x)), fy = int(std::lround(b.y));
                        verify(frames, px, py, cu, fx, fy);
                        if (!fx && !fy) mx = my = 0;
                    }
                    NVENC_EXTERNAL_ME_SB_HINT a{};
                    a.refidx = valid ? 0 : -1;  // bit pattern 31
                    a.direction = 0; a.bi = 0; a.partition_type = 0;
                    // Signed 3-bit fields hold 0..7 as bit patterns; signed 2-bit cu_size likewise
                    int x8 = cx * cu / 8, y8 = cy * cu / 8;
                    a.x8 = x8 >= 4 ? x8 - 8 : x8;
                    a.y8 = y8 >= 4 ? y8 - 8 : y8;
                    a.cu_size = size_code >= 2 ? size_code - 4 : size_code;
                    a.last_of_cu = -1;
                    a.last_of_sb = 0;
                    a.mvx = valid ? mx : 0;
                    a.mvy = valid ? my : 0;
                    row.push_back(a);
                }
                if (row.size() > first) row.back().last_of_sb = -1;
            }
        }
    });
    out.clear();
    for (auto& row : per_row) out.insert(out.end(), row.begin(), row.end());
}

/// Depth of a layer for the depth-edge feature (world units; only ratios
/// matter through log depth). The HUD is not in a game's depth buffer.
inline double layer_depth(int layer) {
    return layer == kFar ? 100 : layer == kMid ? 30 : layer == kNear ? 10 : 5;
}
/// Zero-sum rounding: subtract the mean, round with error diffusion, clamp
inline void zero_sum(std::vector<int8_t>& out, std::vector<double> v, double limit) {
    double mean = 0;
    for (double x : v) mean += x;
    mean /= double(v.size());
    double carry = 0;
    out.resize(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        double want = std::clamp(v[i] - mean, -limit, limit) + carry;
        int q = int(std::lround(want));
        carry = want - q;
        out[i] = int8_t(q);
    }
}
/// A per-superblock (64x64, raster) QP delta map for frame t, in the codec's
/// QP units (`scale` per HEVC QP step: 1 for HEVC, 4 for AV1 q-index), from
/// what a game's depth and motion vectors know (surface motion ignoring the
/// HUD, frame edges). kind:
///  edges:   the "importance" proposal: 0.45 motion discontinuity + 0.25
///           motion variance + 0.30 log-depth edges, 3x3 blurred, percentile
///           classes top 5% -4, 10% -3, 15% -2, 20% -1, remaining 50% +2
///  persist: +4 x fraction of the block whose content leaves the frame next
///           frame (extrapolated from its motion), -4 x fraction newly visible
///           at the frame edge; mean removed
/// Both are zero-sum so rate control keeps its budget.
inline void qp_map(std::vector<int8_t>& out, const Scene& scene, int t, const std::string& kind, int scale) {
    const int cols = (scene.width() + 63) / 64, rows = (scene.height() + 63) / 64, n = cols * rows;
    std::vector<double> md(static_cast<size_t>(n)), mv(static_cast<size_t>(n)), de(static_cast<size_t>(n)), exits(static_cast<size_t>(n)), fresh(static_cast<size_t>(n));
    parallel_rows(rows, [&](int y0, int y1) {
        for (int by = y0; by < y1; ++by)
            for (int bx = 0; bx < cols; ++bx) {
                Vec m[8][8];
                double d[8][8];
                bool in[8][8];
                double sx = 0, sy = 0, sxx = 0, syy = 0;
                int count = 0, exiting = 0, entering = 0;
                for (int j = 0; j < 8; ++j)
                    for (int i = 0; i < 8; ++i) {
                        Vec p{bx * 64 + 4 + 8.0 * i, by * 64 + 4 + 8.0 * j};
                        in[j][i] = scene.inside(p);
                        if (!in[j][i]) continue;
                        int layer = scene.layer_at(t, p, false);
                        Vec q = scene.previous(t, p, layer);
                        m[j][i] = {q.x - p.x, q.y - p.y};
                        d[j][i] = std::log(layer_depth(layer));
                        sx += m[j][i].x; sy += m[j][i].y; sxx += m[j][i].x * m[j][i].x; syy += m[j][i].y * m[j][i].y;
                        ++count;
                        // Velocity is -mv: next frame the content is at p - mv
                        if (!scene.inside({p.x - m[j][i].x, p.y - m[j][i].y})) ++exiting;
                        if (!scene.inside(q)) ++entering;
                    }
                double dmv = 0, ddep = 0;
                int pairs = 0;
                for (int j = 0; j < 8; ++j)
                    for (int i = 0; i < 8; ++i) {
                        if (!in[j][i]) continue;
                        if (i + 1 < 8 && in[j][i + 1]) {
                            dmv += std::abs(m[j][i].x - m[j][i + 1].x) + std::abs(m[j][i].y - m[j][i + 1].y);
                            ddep += std::abs(d[j][i] - d[j][i + 1]);
                            ++pairs;
                        }
                        if (j + 1 < 8 && in[j + 1][i]) {
                            dmv += std::abs(m[j][i].x - m[j + 1][i].x) + std::abs(m[j][i].y - m[j + 1][i].y);
                            ddep += std::abs(d[j][i] - d[j + 1][i]);
                            ++pairs;
                        }
                    }
                size_t k = size_t(by) * cols + bx;
                md[k] = pairs ? dmv / pairs : 0;
                de[k] = pairs ? ddep / pairs : 0;
                mv[k] = count ? (sxx / count - (sx / count) * (sx / count)) + (syy / count - (sy / count) * (sy / count)) : 0;
                exits[k] = count ? double(exiting) / count : 0;
                fresh[k] = count ? double(entering) / count : 0;
            }
    });
    std::vector<double> want(static_cast<size_t>(n));
    if (kind == "persist") {
        for (int k = 0; k < n; ++k) want[size_t(k)] = (4 * exits[size_t(k)] - 4 * fresh[size_t(k)]) * scale;
        zero_sum(out, want, 8.0 * scale);
        return;
    }
    // edges: normalise each feature by its frame maximum, combine, blur 3x3
    auto norm = [](std::vector<double>& v) {
        double hi = *std::max_element(v.begin(), v.end());
        if (hi > 0) for (double& x : v) x /= hi;
    };
    norm(md); norm(mv); norm(de);
    std::vector<double> score(static_cast<size_t>(n)), blurred(static_cast<size_t>(n));
    for (int k = 0; k < n; ++k) score[size_t(k)] = 0.45 * md[size_t(k)] + 0.25 * mv[size_t(k)] + 0.30 * de[size_t(k)];
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x) {
            double sum = 0;
            int c = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    int xx = x + dx, yy = y + dy;
                    if (xx < 0 || yy < 0 || xx >= cols || yy >= rows) continue;
                    sum += score[size_t(yy) * cols + xx];
                    ++c;
                }
            blurred[size_t(y) * cols + x] = sum / c;
        }
    // Percentile classes by rank (ties broken by position)
    std::vector<int> order(static_cast<size_t>(n));
    for (int k = 0; k < n; ++k) order[size_t(k)] = k;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return blurred[size_t(a)] > blurred[size_t(b)]; });
    for (int r = 0; r < n; ++r) {
        double f = double(r) / n;
        int level = f < 0.05 ? -4 : f < 0.15 ? -3 : f < 0.30 ? -2 : f < 0.50 ? -1 : 2;
        want[size_t(order[size_t(r)])] = level * scale;
    }
    zero_sum(out, want, 8.0 * scale);
}

inline void selftest() {
    auto fail = [](const char* what) { throw std::runtime_error(std::string("selftest: ") + what); };
    // Pan: every pixel moves by -s; the reference is s to the right: mv = +s
    Scene pan(Kind::pan, 128, 0, 3840, 2160, tex_multiscale);
    std::vector<NVENC_EXTERNAL_ME_HINT> mb;
    mb_hints(mb, pan, 5, Hints::on, false, false);
    if (mb.size() != 240 * 135) fail("mb count");
    auto h = mb[60 * 240 + 120];
    if (h.mvx != 128 || h.mvy || h.refidx) fail("pan direction");
    uint32_t packed = 0;
    std::memcpy(&packed, &h, 4);
    if ((packed & 4095) != 128 || ((packed >> 22) & 31) != 0 || (packed >> 30) != 3) fail("integer bit packing");
    // Content is where the vector says: frame t at p equals frame t-1 at p + mv
    for (Vec p : {Vec{1000.5, 700.5}, Vec{2000.5, 1500.5}})
        if (pan.shade(5, p) != pan.shade(4, {p.x + 128, p.y})) fail("pan content/vector mismatch");
    Scene noise(Kind::pan, 128, 0, 3840, 2160, tex_noise);
    if (noise.shade(5, {1000.5, 700.5}) != noise.shade(4, {1128.5, 700.5})) fail("noise content/vector mismatch");
    // Right edge: the reference lies outside the frame
    if (mb[60 * 240 + 239].refidx != -1) fail("out of frame not invalid");
    // HUD: a game's vectors carry the world motion under the HUD; the oracle knows it is static
    if (mb[5 * 240 + 5].mvx != 128) fail("game-like HUD");
    mb_hints(mb, pan, 5, Hints::oracle, false, false);
    if (mb[5 * 240 + 5].mvx != 0 || mb[5 * 240 + 5].refidx) fail("oracle HUD");
    // 8x8: four per macroblock, only the last flagged
    mb_hints(mb, pan, 5, Hints::on, true, false);
    if (mb.size() != 4 * 240 * 135 || mb[3].lastOfMB != -1 || mb[2].lastOfMB || mb[0].partType != -1) fail("8x8 layout");
    // CTU order: same entries, permuted
    mb_hints(mb, pan, 5, Hints::on, false, true);
    if (mb.size() != 240 * 135) fail("ctu order count");
    // Regions: the right edge is exposed, the middle overlaps, the HUD is the HUD
    uint8_t r = 0;
    pan.shade(5, {3830.5, 1000.5}, &r); if (r != kExposed) fail("exposed region");
    pan.shade(5, {1000.5, 1000.5}, &r); if (r != kOverlap) fail("overlap region");
    pan.shade(5, {100.5, 100.5}, &r); if (r != kHudRegion) fail("hud region");
    // New scenes: content is where the motion says (frame t at p equals
    // frame t-1 at previous(p)), away from sprites and the HUD
    for (Kind k : {Kind::topdown, Kind::chase}) {
        Scene sc(k, 128, 0, 3840, 2160, tex_multiscale);
        for (Vec p : {Vec{1000.5, 1400.5}, Vec{2900.5, 1900.5}, Vec{1700.5, 1100.5}, Vec{2000.5, 400.5}}) {
            int layer = sc.layer_at(9, p);
            if (layer >= kSprite0 || layer == kHud) continue;
            Vec q = sc.previous(9, p, layer);
            if (!sc.inside(q) || sc.layer_at(8, q) != layer) continue;
            Vec w0 = sc.to_world_for_test(9, layer, p), w1 = sc.to_world_for_test(8, layer, q);
            if (std::abs(w0.x - w1.x) > 1e-6 || std::abs(w0.y - w1.y) > 1e-6) fail(k == Kind::chase ? "chase motion" : "topdown motion");
        }
    }
    {
        // Chase: the bottom row moves ~s px/frame (down), the horizon barely
        Scene sc(Kind::chase, 128, 0, 3840, 2160, tex_multiscale);
        Vec q = sc.previous(60, {1920.5, 2159.5}, kFar);  // (t = 60: the yaw is turning around, its rate ~0)
        if (std::abs((2159.5 - q.y) - 128) > 20) fail("chase bottom speed");
    }
    // Orbit: the centre barely moves, the corners move ~1.1*s
    Scene orbit(Kind::orbit, 200, 0, 3840, 2160, tex_multiscale);
    Vec c{1920, 1080}, corner{3700, 2000};
    Vec qc = orbit.previous(10, c, kFar), qk = orbit.previous(10, corner, kFar);
    if (std::hypot(qc.x - c.x, qc.y - c.y) > 0.5) fail("orbit centre");
    double speed = std::hypot(qk.x - corner.x, qk.y - corner.y);
    if (speed < 150 || speed > 350) fail("orbit corner speed");
    // The same surface point at t and at its previous position at t-1
    Vec p{3000.5, 500.5};
    Vec q = orbit.previous(10, p, kFar);
    if (std::abs(int(orbit.shade(10, p)) - int(orbit.shade(9, q))) > 2) fail("orbit content/vector mismatch");
    // Parallax: near moves 10x far
    Scene parallax(Kind::parallax, 100, 0, 3840, 2160, tex_multiscale);
    Vec farq = parallax.previous(3, {1000, 300}, kFar), nearq = parallax.previous(3, {1000, 300}, kNear);
    if (std::abs(farq.x - 1010) > 1e-6 || std::abs(nearq.x - 1100) > 1e-6) fail("parallax rates");
    // Sprites: a pixel on a sprite moves with the sprite
    Scene objects(Kind::objects, 300, 0, 3840, 2160, tex_multiscale);
    int hits = 0;
    for (int y = 0; y < 2160 && !hits; y += 8)
        for (int x = 0; x < 3840; x += 8) {
            Vec s{x + 0.5, y + 0.5};
            int layer = objects.layer_at(20, s, false);
            if (layer < kSprite0) continue;
            Vec rs = objects.previous(20, s, layer);
            if (objects.layer_at(19, rs, false) == layer && std::abs(int(objects.shade(20, s)) - int(objects.shade(19, rs))) > 1) fail("sprite content/vector mismatch");
            ++hits;
            break;
        }
    if (!hits) fail("no sprite found");
    // Verified: rendered frames 4 and 5; the HUD's world vector loses to zero, the scene's vector wins
    std::vector<uint8_t> f4, f5, r4, r5;
    frame(f4, r4, pan, 4);
    frame(f5, r5, pan, 5);
    Frames frames{f5.data(), f4.data(), 3840, 2160};
    mb_hints(mb, pan, 5, Hints::verified, false, false, &frames);
    if (mb[5 * 240 + 5].mvx != 0 || mb[5 * 240 + 5].refidx) fail("verified HUD");
    if (mb[60 * 240 + 120].mvx != 128) fail("verified scene");
    // QP maps are zero-sum; persist: the exiting left edge gets +, the entering right edge -
    std::vector<int8_t> qm;
    qp_map(qm, pan, 5, "persist", 4);
    long total = 0;
    for (int8_t v : qm) total += v;
    if (qm.size() != 60 * 34 || std::abs(total) > 1 || qm[20 * 60 + 0] <= 0 || qm[20 * 60 + 59] >= 0 || qm[20 * 60 + 30] != 0) fail("persist map");
    qp_map(qm, parallax, 3, "edges", 4);
    total = 0;
    for (int8_t v : qm) total += v;
    if (std::abs(total) > 1) fail("edges map not zero-sum");
    // AV1 packing
    std::vector<NVENC_EXTERNAL_ME_SB_HINT> sb;
    sb_hints(sb, pan, 5, Hints::on);
    if (sb.size() != 60 * 34) fail("sb count");
    auto a = sb[20 * 60 + 30];
    uint16_t words[3]{};
    std::memcpy(words, &a, 6);
    if (a.mvx != 512 || a.mvy || (words[0] & 0x6000) != 0x6000 || (words[1] >> 14) != 3) fail("AV1 bit packing");
    // 16x16 CUs: 16 per superblock (12 in the 48-px bottom row), Z order, positions in 8 px units
    sb_hints(sb, pan, 5, Hints::on, 16);
    if (sb.size() != 60 * 33 * 16 + 60 * 12) fail("16x16 CU count");
    auto u = [](int v, int bits) { return v & ((1 << bits) - 1); };
    const auto& c1 = sb[1];  // second CU of the first SB: x8 = 2, y8 = 0
    const auto& c3 = sb[3];  // fourth: x8 = 2, y8 = 2
    const auto& c15 = sb[15];  // last: x8 = 6, y8 = 6, last of SB
    if (u(c1.x8, 3) != 2 || c1.y8 || u(c3.x8, 3) != 2 || u(c3.y8, 3) != 2 || u(c15.x8, 3) != 6 || u(c15.y8, 3) != 6
        || c1.cu_size != 1 || c1.last_of_sb || !c15.last_of_sb) fail("16x16 CU layout");
}
}  // namespace motion
