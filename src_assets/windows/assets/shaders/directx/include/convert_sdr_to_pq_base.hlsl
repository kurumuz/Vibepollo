#include "include/common.hlsl"

cbuffer sdr_to_pq_params_cbuffer : register(b1) {
    float sdr_white_nits;
    // 0 selects the piecewise sRGB EOTF; any positive value is a pure power
    // law. SDR content is ENCODED per the sRGB spec but consumer displays
    // DECODE with a plain ~2.2 power curve, so content is mastered against
    // 2.2 -- linearizing with the piecewise function instead lifts near-black
    // (the classic washed-shadows SDR-in-HDR complaint).
    float sdr_gamma;
    float2 sdr_to_pq_params_padding;
    // Source-primaries -> BT.2020 rows, computed host-side. At gamut wideness
    // 0 this is exactly the Rec.709 -> BT.2020 matrix (colorimetric sRGB); at
    // 100 the source values are interpreted as Display P3, reproducing the
    // wide-gamut stretch of an SDR monitor's native mode.
    float4 gamut_r0;
    float4 gamut_r1;
    float4 gamut_r2;
};

float3 RemoveSRGBCurve(float3 x)
{
    return x <= 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4);
}

float3 CONVERT_FUNCTION(float3 input)
{
    float3 srgb = saturate(input);
    float3 rgb = sdr_gamma > 0.0 ? pow(srgb, sdr_gamma) : RemoveSRGBCurve(srgb);
    rgb = float3(dot(gamut_r0.xyz, rgb), dot(gamut_r1.xyz, rgb), dot(gamut_r2.xyz, rgb));
    rgb *= sdr_white_nits;
    return NitsToPQ(rgb);
}
