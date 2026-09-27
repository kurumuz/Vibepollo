// Fullscreen triangle for converting an in-game capture frame
// (src/platform/windows/game_capture/game_source.cpp)
#include "include/base_vs.hlsl"

vertex_t main_vs(uint vertex_id : SV_VertexID)
{
    return generate_fullscreen_triangle_vertex(vertex_id, float2(0, 0), 0);
}
