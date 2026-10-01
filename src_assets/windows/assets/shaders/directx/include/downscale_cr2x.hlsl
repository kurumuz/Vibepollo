// The source read of a conversion: one bilinear tap, or -- when the source is
// exactly twice the output in both directions (a game rendered at 2x the
// stream, downscaled here) -- a Catmull-Rom 2x downscale. One tap at 2x is a
// 2x2 box, which aliases fine detail into shimmer that the encoder then spends
// bits on. The kernel is CR(d/2) over source texels at d = +-0.5, +-1.5,
// +-2.5, +-3.5 from the output pixel's centre (normalised weights 0.43359375,
// 0.11328125, -0.03515625, -0.01171875); same-sign neighbours share one
// bilinear read (inner pair at 0.70714286, weight 0.546875; outer at 2.75,
// weight -0.046875), so the separable 8x8 footprint is 16 reads, and the four
// outer corners (0.047^2 each, under 1% together) are dropped and the rest
// renormalised: 12 reads. (moonlight-android's PostProcessPresenter uses the
// same kernel for a 4K stream on a 1080p panel.)
//
// Expects `image` and `def_sampler` (linear, clamped) to be declared.

cbuffer downscale_cbuffer : register(b2) {
    float2 downscale_texel;  // one source texel, in texture coordinates
    uint downscale_cr2x;     // 0 (or unbound): a single tap
    uint downscale_pad;
};

static const float kCrP0 = 2.75;
static const float kCrP1 = 0.70714286;
static const float kCrW0 = -0.046875;
static const float kCrW1 = 0.546875;

float3 cr2x_tap(float2 uv, float x, float y) {
    return image.Sample(def_sampler, uv + float2(x, y) * downscale_texel).rgb;
}

float3 cr2x_row(float2 uv, float y) {
    return kCrW0 * cr2x_tap(uv, -kCrP0, y) + kCrW1 * cr2x_tap(uv, -kCrP1, y) + kCrW1 * cr2x_tap(uv, kCrP1, y) + kCrW0 * cr2x_tap(uv, kCrP0, y);
}

float3 cr2x_row_inner(float2 uv, float y) {
    return kCrW1 * cr2x_tap(uv, -kCrP1, y) + kCrW1 * cr2x_tap(uv, kCrP1, y);
}

float3 sample_source(float2 uv) {
    if (downscale_cr2x != 0) {
        const float3 c = (kCrW0 * cr2x_row_inner(uv, -kCrP0) + kCrW1 * cr2x_row(uv, -kCrP1) + kCrW1 * cr2x_row(uv, kCrP1) +
                          kCrW0 * cr2x_row_inner(uv, kCrP0)) /
                         (1.0 - 4.0 * kCrW0 * kCrW0);
        // (the negative lobes may undershoot black; linear-light sources may exceed 1)
        return max(c, 0.0);
    }
    return image.Sample(def_sampler, uv, 0).rgb;
}
