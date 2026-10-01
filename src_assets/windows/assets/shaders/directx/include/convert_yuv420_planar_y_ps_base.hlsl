Texture2D image : register(t0);
SamplerState def_sampler : register(s0);

cbuffer color_matrix_cbuffer : register(b0) {
    float4 color_vec_y;
    float4 color_vec_u;
    float4 color_vec_v;
    float2 range_y;
    float2 range_uv;
};

#include "include/base_vs_types.hlsl"
#include "include/downscale_cr2x.hlsl"

float main_ps(vertex_t input) : SV_Target
{
    float3 rgb = CONVERT_FUNCTION(sample_source(input.tex_coord));

    float y = dot(color_vec_y.xyz, rgb) + color_vec_y.w;

    return y * range_y.x + range_y.y;
}
