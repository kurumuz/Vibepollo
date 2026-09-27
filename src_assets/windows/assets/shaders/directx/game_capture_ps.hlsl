// Converts an in-game capture frame into the capture texture's encoding, the
// one the desktop compositor would have produced for the same game
// (src/platform/windows/game_capture/game_source.cpp, converter_t).
#include "include/base_vs_types.hlsl"

Texture2D image : register(t0);
SamplerState def_sampler : register(s0);

cbuffer params : register(b0) {
    int mode;               // see converter_t modes
    float sdr_white_scale;  // compositor SDR white as a scRGB multiplier (1.0 = 80 nits)
    float2 pad;
};

float3 srgb_to_linear(float3 c)
{
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

float3 linear_to_srgb(float3 c)
{
    return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

// SMPTE ST 2084 EOTF: PQ code value -> nits
float3 pq_to_nits(float3 n)
{
    static const float m1 = 2610.0 / 4096.0 / 4;
    static const float m2 = 2523.0 / 4096.0 * 128;
    static const float c1 = 3424.0 / 4096.0;
    static const float c2 = 2413.0 / 4096.0 * 32;
    static const float c3 = 2392.0 / 4096.0 * 32;

    float3 np = pow(saturate(n), 1.0 / m2);
    return pow(max(np - c1, 0.0) / (c2 - c3 * np), 1.0 / m1) * 10000.0;
}

float3 rec2020_to_rec709(float3 c)
{
    static const float3x3 m =
    {
         1.660491, -0.587641, -0.072850,
        -0.124550,  1.132900, -0.008349,
        -0.018151, -0.100579,  1.118730
    };
    return mul(m, c);
}

float4 main_ps(vertex_t input) : SV_Target
{
    float3 rgb = image.Sample(def_sampler, input.tex_coord).rgb;

    if (mode == 1) {
        rgb = rgb * sdr_white_scale;
    } else if (mode == 2) {
        rgb = srgb_to_linear(saturate(rgb)) * sdr_white_scale;
    } else if (mode == 3) {
        rgb = rec2020_to_rec709(pq_to_nits(rgb)) / 80.0;
    } else if (mode == 4) {
        rgb = linear_to_srgb(saturate(rgb));
    }

    return float4(rgb, 1.0);
}
