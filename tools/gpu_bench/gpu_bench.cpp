// GPU cost of Vibepollo's streaming work, piece by piece, against a game.
//
// host-alone <shaders> : each host workload per game frame, timed alone on an
//                        idle GPU (the host's own shaders, its formats and
//                        dispatch/draw sequences): (a) the game frame drawn
//                        into the capture image (sRGB -> scRGB), (b) the
//                        motion pass, (c) the P010 conversion for the encoder
// hook-alone           : the hook's D3D12 work in the game's frame, alone:
//                        the motion-vector copy after a DLSS evaluation, and
//                        at Present the back-buffer copy and the vector sets
// game [--hook] [secs] : a synthetic GPU-bound game (D3D12 compute, ~12 ms a
//                        frame, its "DLSS" output then its "back buffer"),
//                        optionally with the hook's work recorded into its
//                        frames as the hook records it; prints its frame GPU
//                        times
// host-load <shaders> <hz> <secs> : the host workloads (a)+(b)+(c) at <hz>, at
//                        the host's GPU priority (realtime process class,
//                        raised GPU thread priority), to run beside `game`
#include <d3d11.h>
#include <d3d11_1.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define CHECK(x) do { HRESULT hr_ = (x); if (FAILED(hr_)) { std::printf("FAILED 0x%08lx: %s (line %d)\n", (unsigned long) hr_, #x, __LINE__); std::exit(1); } } while (0)

static const uint32_t W = 2880, H = 1800;  // the game's frames
static const uint32_t MW = 1440, MH = 900;  // its DLSS vectors (render resolution)
static const int kSets = 3, kStatsPerSet = 12;
// The synthetic game's passes as many short dispatches (row strips), as a
// real frame's hundreds of draws and dispatches: 1 = one long dispatch each
static int g_chunks = 1;

static std::string stat(std::vector<double> v) {
  if (v.empty()) return "n/a";
  std::sort(v.begin(), v.end());
  char buf[128];
  std::snprintf(buf, sizeof(buf), "p50 %7.3f  p95 %7.3f  p99 %7.3f  max %7.3f ms", v[v.size() / 2], v[v.size() * 95 / 100], v[v.size() * 99 / 100], v.back());
  return buf;
}

static IDXGIAdapter1 *nvidia_adapter() {
  IDXGIFactory1 *factory = nullptr;
  CHECK(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **) &factory));
  IDXGIAdapter1 *adapter = nullptr;
  for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
    DXGI_ADAPTER_DESC1 d;
    adapter->GetDesc1(&d);
    if (d.VendorId == 0x10DE) {
      return adapter;
    }
    adapter->Release();
  }
  std::printf("no NVIDIA adapter\n");
  std::exit(1);
}

static ID3DBlob *compile_file(const std::string &file, const char *entry, const char *target) {
  std::wstring w(file.begin(), file.end());
  ID3DBlob *blob = nullptr, *msg = nullptr;
  HRESULT hr = D3DCompileFromFile(w.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, entry, target, D3DCOMPILE_ENABLE_STRICTNESS, 0, &blob, &msg);
  if (msg && FAILED(hr)) std::printf("%s %s: %.*s\n", file.c_str(), entry, (int) msg->GetBufferSize(), (const char *) msg->GetBufferPointer());
  CHECK(hr);
  return blob;
}

static ID3DBlob *compile_text(const char *src, const char *entry, const char *target) {
  ID3DBlob *blob = nullptr, *msg = nullptr;
  HRESULT hr = D3DCompile(src, std::strlen(src), "inline", nullptr, nullptr, entry, target, 0, 0, &blob, &msg);
  if (msg && FAILED(hr)) std::printf("%s: %.*s\n", entry, (int) msg->GetBufferSize(), (const char *) msg->GetBufferPointer());
  CHECK(hr);
  return blob;
}

// ---- host workloads (D3D11) --------------------------------------------------

struct motion_params_t {
  uint32_t frame_size[2], blocks[2];
  float texel_per_pixel[2], pixel_per_unit[2];
  uint32_t linear_light, verify, diagnose, vector_row, stats_base, field_base, score_cap, pad;
  uint32_t cells[2];
  float static_luma, ui_contrast;
};

struct host_t {
  ID3D11Device *dev = nullptr;
  ID3D11DeviceContext *ctx = nullptr;
  // (a) game frame -> capture image
  ID3D11Texture2D *game_frame;
  ID3D11ShaderResourceView *game_srv;
  ID3D11Texture2D *capture;  // R16G16B16A16_FLOAT
  ID3D11RenderTargetView *capture_rtv;
  ID3D11ShaderResourceView *capture_srv;
  ID3D11VertexShader *gc_vs;
  ID3D11PixelShader *gc_ps;
  ID3D11Buffer *gc_params;
  ID3D11SamplerState *point, *linear;
  // (b) motion pass
  ID3D11ComputeShader *luma_cs, *blocks_cs, *mask_cs;
  ID3D11Texture2D *vectors;
  ID3D11ShaderResourceView *vectors_srv, *color8_srv;
  ID3D11Texture2D *luma[2];
  ID3D11ShaderResourceView *luma_srv[2];
  ID3D11UnorderedAccessView *luma_uav[2];
  ID3D11Buffer *field, *field_staging, *stats, *stats_staging, *mask, *mask_staging, *mcb;
  ID3D11UnorderedAccessView *field_uav, *stats_uav, *mask_uav;
  uint32_t per_field, cw, ch, mask_words;
  int cur = 0;
  uint64_t frame = 0;
  // (c) P010 conversion
  ID3D11Texture2D *p010;
  ID3D11RenderTargetView *y_rtv, *uv_rtv;
  ID3D11VertexShader *y_vs, *uv_vs;
  ID3D11PixelShader *y_ps, *uv_ps;
  ID3D11Buffer *color_matrix, *rotate, *subsample;
};

static void raw_buffer(ID3D11Device *dev, uint32_t words, ID3D11Buffer **buf, ID3D11UnorderedAccessView **uav, ID3D11Buffer **staging) {
  D3D11_BUFFER_DESC md {};
  md.ByteWidth = words * 4;
  md.Usage = D3D11_USAGE_DEFAULT;
  md.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
  md.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
  CHECK(dev->CreateBuffer(&md, nullptr, buf));
  D3D11_UNORDERED_ACCESS_VIEW_DESC ud {};
  ud.Format = DXGI_FORMAT_R32_TYPELESS;
  ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
  ud.Buffer.NumElements = words;
  ud.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
  CHECK(dev->CreateUnorderedAccessView(*buf, &ud, uav));
  D3D11_BUFFER_DESC sd = md;
  sd.Usage = D3D11_USAGE_STAGING;
  sd.BindFlags = 0;
  sd.MiscFlags = 0;
  sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
  CHECK(dev->CreateBuffer(&sd, nullptr, staging));
}

static ID3D11Buffer *cbuffer(ID3D11Device *dev, UINT size, const void *data) {
  D3D11_BUFFER_DESC bd {};
  bd.ByteWidth = (size + 15) & ~15u;
  bd.Usage = D3D11_USAGE_DEFAULT;
  bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
  std::vector<uint8_t> init(bd.ByteWidth, 0);
  std::memcpy(init.data(), data, size);
  D3D11_SUBRESOURCE_DATA sd {init.data(), 0, 0};
  ID3D11Buffer *b;
  CHECK(dev->CreateBuffer(&bd, &sd, &b));
  return b;
}

// priority: the D3DKMT_SCHEDULINGPRIORITYCLASS (-1: leave it), thread_priority
// the IDXGIDevice::SetGPUThreadPriority value (INT32_MIN: leave it)
static void host_init(host_t &h, const std::string &dir, int priority, int thread_priority = INT32_MIN) {
  UINT flags = 0;
  D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1;
  CHECK(D3D11CreateDevice(nvidia_adapter(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, flags, &fl, 1, D3D11_SDK_VERSION, &h.dev, nullptr, &h.ctx));
  {
    typedef LONG(APIENTRY * set_prio_t)(HANDLE, int);
    auto set = (set_prio_t) GetProcAddress(GetModuleHandleW(L"gdi32.dll"), "D3DKMTSetProcessSchedulingPriorityClass");
    LONG r = priority >= 0 && set ? set(GetCurrentProcess(), priority) : 0;
    HRESULT t = S_OK;
    INT got = 0;
    IDXGIDevice *dxgi = nullptr;
    h.dev->QueryInterface(__uuidof(IDXGIDevice), (void **) &dxgi);
    if (thread_priority != INT32_MIN) {
      t = dxgi->SetGPUThreadPriority(thread_priority);
    }
    dxgi->GetGPUThreadPriority(&got);
    std::printf("host priority: process class %d %s, GPU thread priority 0x%x %s (now 0x%x)\n", priority, r == 0 ? "ok" : "FAILED",
                thread_priority == INT32_MIN ? 0 : (unsigned) thread_priority, SUCCEEDED(t) ? "ok" : "FAILED", (unsigned) got);
  }
  std::mt19937 rng(7);
  // The game's frame: 8-bit sRGB-encoded SDR, as Witcher 3's
  {
    std::vector<uint32_t> px(W * H);
    for (auto &p : px) p = rng() | 0xff000000u;
    D3D11_TEXTURE2D_DESC td {};
    td.Width = W; td.Height = H; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM; td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA sd {px.data(), W * 4, 0};
    CHECK(h.dev->CreateTexture2D(&td, &sd, &h.game_frame));
    CHECK(h.dev->CreateShaderResourceView(h.game_frame, nullptr, &h.game_srv));
    h.color8_srv = h.game_srv;
    td.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    CHECK(h.dev->CreateTexture2D(&td, nullptr, &h.capture));
    CHECK(h.dev->CreateRenderTargetView(h.capture, nullptr, &h.capture_rtv));
    CHECK(h.dev->CreateShaderResourceView(h.capture, nullptr, &h.capture_srv));
  }
  {
    auto vs = compile_file(dir + "/game_capture_vs.hlsl", "main_vs", "vs_5_0");
    auto ps = compile_file(dir + "/game_capture_ps.hlsl", "main_ps", "ps_5_0");
    CHECK(h.dev->CreateVertexShader(vs->GetBufferPointer(), vs->GetBufferSize(), nullptr, &h.gc_vs));
    CHECK(h.dev->CreatePixelShader(ps->GetBufferPointer(), ps->GetBufferSize(), nullptr, &h.gc_ps));
    struct { int mode; float white; float pad[2]; } p {2 /* sRGB -> scRGB */, 2.5f, {0, 0}};
    h.gc_params = cbuffer(h.dev, sizeof(p), &p);
    D3D11_SAMPLER_DESC sd {};
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.MaxLOD = D3D11_FLOAT32_MAX;
    CHECK(h.dev->CreateSamplerState(&sd, &h.point));
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    CHECK(h.dev->CreateSamplerState(&sd, &h.linear));
  }
  // (b) the motion pass, as motion_pass_t::init
  {
    const char *e[3] = {"luma_cs", "blocks_cs", "mask_cs"};
    ID3D11ComputeShader **cs[3] = {&h.luma_cs, &h.blocks_cs, &h.mask_cs};
    for (int i = 0; i < 3; ++i) {
      auto b = compile_file(dir + "/game_capture_motion_cs.hlsl", e[i], "cs_5_0");
      CHECK(h.dev->CreateComputeShader(b->GetBufferPointer(), b->GetBufferSize(), nullptr, cs[i]));
    }
    // Vectors: the sets stacked, at render resolution; ~21% of blocks move
    // 2 px (checked), the rest a fraction of a pixel
    std::vector<uint32_t> mv(MW * MH * kSets);
    auto half = [](float f) -> uint16_t {
      uint32_t x; std::memcpy(&x, &f, 4);
      uint32_t s = (x >> 16) & 0x8000; int e2 = ((x >> 23) & 0xff) - 112; uint32_t m = x & 0x7fffff;
      if (e2 <= 0) return (uint16_t) s;
      return (uint16_t) (s | (e2 << 10) | (m >> 13));
    };
    for (uint32_t i = 0; i < MW * MH * kSets; ++i) {
      const bool large = (((i / MW) / 8) * 131 + ((i % MW) / 8) * 7) % 100 < 21;
      mv[i] = half(large ? -1.0f : -0.1f);  // (render-resolution texels: x2 on the frame)
    }
    D3D11_TEXTURE2D_DESC td {};
    td.Width = MW; td.Height = MH * kSets; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R16G16_FLOAT; td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA sd {mv.data(), MW * 4, 0};
    CHECK(h.dev->CreateTexture2D(&td, &sd, &h.vectors));
    CHECK(h.dev->CreateShaderResourceView(h.vectors, nullptr, &h.vectors_srv));
    for (int i = 0; i < 2; ++i) {
      D3D11_TEXTURE2D_DESC lt {};
      lt.Width = W; lt.Height = H; lt.MipLevels = 1; lt.ArraySize = 1;
      lt.Format = DXGI_FORMAT_R16_FLOAT; lt.SampleDesc.Count = 1;
      lt.Usage = D3D11_USAGE_DEFAULT; lt.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      CHECK(h.dev->CreateTexture2D(&lt, nullptr, &h.luma[i]));
      CHECK(h.dev->CreateShaderResourceView(h.luma[i], nullptr, &h.luma_srv[i]));
      CHECK(h.dev->CreateUnorderedAccessView(h.luma[i], nullptr, &h.luma_uav[i]));
    }
    const uint32_t bw = (W + 15) / 16, bh = (H + 15) / 16;
    h.per_field = bw * bh;
    D3D11_BUFFER_DESC fd {};
    fd.ByteWidth = h.per_field * kSets * 8; fd.Usage = D3D11_USAGE_DEFAULT;
    fd.BindFlags = D3D11_BIND_UNORDERED_ACCESS; fd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; fd.StructureByteStride = 8;
    CHECK(h.dev->CreateBuffer(&fd, nullptr, &h.field));
    CHECK(h.dev->CreateUnorderedAccessView(h.field, nullptr, &h.field_uav));
    D3D11_BUFFER_DESC sb {};
    sb.ByteWidth = h.per_field * 8; sb.Usage = D3D11_USAGE_STAGING; sb.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    CHECK(h.dev->CreateBuffer(&sb, nullptr, &h.field_staging));
    h.cw = (W + 3) / 4; h.ch = (H + 3) / 4; h.mask_words = (h.cw * h.ch + 31) / 32;
    raw_buffer(h.dev, h.mask_words, &h.mask, &h.mask_uav, &h.mask_staging);
    raw_buffer(h.dev, kStatsPerSet * kSets, &h.stats, &h.stats_uav, &h.stats_staging);
    motion_params_t p {};
    h.mcb = cbuffer(h.dev, sizeof(p), &p);
  }
  // (c) P010 conversion: Y and UV planes, the 10-bit SDR shaders (linear:
  // the capture image is scRGB)
  {
    D3D11_TEXTURE2D_DESC td {};
    td.Width = W; td.Height = H; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_P010; td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_RENDER_TARGET;
    CHECK(h.dev->CreateTexture2D(&td, nullptr, &h.p010));
    D3D11_RENDER_TARGET_VIEW_DESC rd {};
    rd.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
    rd.Format = DXGI_FORMAT_R16_UNORM;
    CHECK(h.dev->CreateRenderTargetView(h.p010, &rd, &h.y_rtv));
    rd.Format = DXGI_FORMAT_R16G16_UNORM;
    CHECK(h.dev->CreateRenderTargetView(h.p010, &rd, &h.uv_rtv));
    auto yvs = compile_file(dir + "/convert_yuv420_planar_y_vs.hlsl", "main_vs", "vs_5_0");
    auto yps = compile_file(dir + "/convert_yuv420_planar_y_ps_linear.hlsl", "main_ps", "ps_5_0");
    auto uvvs = compile_file(dir + "/convert_yuv420_packed_uv_type0_vs.hlsl", "main_vs", "vs_5_0");
    auto uvps = compile_file(dir + "/convert_yuv420_packed_uv_type0_ps_linear.hlsl", "main_ps", "ps_5_0");
    CHECK(h.dev->CreateVertexShader(yvs->GetBufferPointer(), yvs->GetBufferSize(), nullptr, &h.y_vs));
    CHECK(h.dev->CreatePixelShader(yps->GetBufferPointer(), yps->GetBufferSize(), nullptr, &h.y_ps));
    CHECK(h.dev->CreateVertexShader(uvvs->GetBufferPointer(), uvvs->GetBufferSize(), nullptr, &h.uv_vs));
    CHECK(h.dev->CreatePixelShader(uvps->GetBufferPointer(), uvps->GetBufferSize(), nullptr, &h.uv_ps));
    float cm[16] = {0.2126f, 0.7152f, 0.0722f, 0.0625f, -0.1146f, -0.3854f, 0.5f, 0.5f, 0.5f, -0.4542f, -0.0458f, 0.5f, 0.8588f, 0.0627f, 0.8784f, 0.0627f};
    h.color_matrix = cbuffer(h.dev, sizeof(cm), cm);
    int rot[4] = {0, 0, 0, 0};
    h.rotate = cbuffer(h.dev, sizeof(rot), rot);
    float off[4] = {-0.5f / W, 0, 0, 0};
    h.subsample = cbuffer(h.dev, sizeof(off), off);
  }
}

static void host_draw_frame(host_t &h) {  // (a), as converter_t::convert
  D3D11_VIEWPORT vp {0, 0, (float) W, (float) H, 0, 1};
  h.ctx->OMSetRenderTargets(1, &h.capture_rtv, nullptr);
  h.ctx->RSSetViewports(1, &vp);
  h.ctx->IASetInputLayout(nullptr);
  h.ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  h.ctx->VSSetShader(h.gc_vs, nullptr, 0);
  h.ctx->PSSetShader(h.gc_ps, nullptr, 0);
  h.ctx->PSSetConstantBuffers(0, 1, &h.gc_params);
  h.ctx->PSSetSamplers(0, 1, &h.point);
  h.ctx->PSSetShaderResources(0, 1, &h.game_srv);
  h.ctx->Draw(3, 0);
  ID3D11ShaderResourceView *ns = nullptr;
  h.ctx->PSSetShaderResources(0, 1, &ns);
  ID3D11RenderTargetView *nr = nullptr;
  h.ctx->OMSetRenderTargets(1, &nr, nullptr);
}

static void host_motion(host_t &h, bool readback_wait) {  // (b), as motion_pass_t::run
  motion_params_t p {};
  p.frame_size[0] = W; p.frame_size[1] = H;
  p.blocks[0] = (W + 15) / 16; p.blocks[1] = (H + 15) / 16;
  p.linear_light = 0;
  p.verify = h.frame > 0 ? 1 : 0;
  p.cells[0] = h.cw; p.cells[1] = h.ch;
  p.static_luma = 1.0f / 255; p.ui_contrast = 8.0f / 255;
  p.score_cap = std::min<uint32_t>(25000, 0xffffffffu / h.per_field);
  p.diagnose = (h.frame % 4) == 0 ? 1 : 0;
  h.ctx->UpdateSubresource(h.mcb, 0, nullptr, &p, 0, 0);
  h.ctx->CSSetConstantBuffers(0, 1, &h.mcb);
  h.ctx->CSSetShaderResources(0, 1, &h.color8_srv);
  h.ctx->CSSetUnorderedAccessViews(0, 1, &h.luma_uav[h.cur], nullptr);
  h.ctx->CSSetShader(h.luma_cs, nullptr, 0);
  h.ctx->Dispatch((W + 7) / 8, (H + 7) / 8, 1);
  ID3D11UnorderedAccessView *nu[4] = {};
  h.ctx->CSSetUnorderedAccessViews(0, 1, nu, nullptr);
  const int cur = h.cur, prev = 1 - cur;
  h.cur = prev;
  if (!p.verify) {
    ++h.frame;
    return;
  }
  ID3D11ShaderResourceView *in[4] = {nullptr, h.vectors_srv, h.luma_srv[prev], h.luma_srv[cur]};
  h.ctx->CSSetShaderResources(0, 4, in);
  const UINT zero[4] = {};
  h.ctx->ClearUnorderedAccessViewUint(h.stats_uav, zero);
  ID3D11UnorderedAccessView *out[3] = {nullptr, h.field_uav, h.stats_uav};
  h.ctx->CSSetUnorderedAccessViews(0, 3, out, nullptr);
  h.ctx->CSSetShader(h.blocks_cs, nullptr, 0);
  for (int s = 0; s < kSets; ++s) {
    p.texel_per_pixel[0] = (float) MW / W; p.texel_per_pixel[1] = (float) MH / H;
    p.pixel_per_unit[0] = (float) W / MW; p.pixel_per_unit[1] = (float) H / MH;
    p.vector_row = s * MH; p.stats_base = s * kStatsPerSet; p.field_base = s * h.per_field;
    h.ctx->UpdateSubresource(h.mcb, 0, nullptr, &p, 0, 0);
    h.ctx->Dispatch((p.blocks[0] + 7) / 8, (p.blocks[1] + 7) / 8, 1);
  }
  p.field_base = 0;
  h.ctx->UpdateSubresource(h.mcb, 0, nullptr, &p, 0, 0);
  h.ctx->ClearUnorderedAccessViewUint(h.mask_uav, zero);
  h.ctx->CSSetUnorderedAccessViews(3, 1, &h.mask_uav, nullptr);
  h.ctx->CSSetShader(h.mask_cs, nullptr, 0);
  h.ctx->Dispatch((h.cw + 7) / 8, (h.ch + 7) / 8, 1);
  h.ctx->CopyResource(h.mask_staging, h.mask);
  h.ctx->CopyResource(h.stats_staging, h.stats);
  D3D11_BOX box {0, 0, 0, h.per_field * 8, 1, 1};
  h.ctx->CopySubresourceRegion(h.field_staging, 0, 0, 0, 0, h.field, 0, &box);
  ID3D11ShaderResourceView *ns[4] = {};
  h.ctx->CSSetShaderResources(0, 4, ns);
  h.ctx->CSSetUnorderedAccessViews(0, 4, nu, nullptr);
  h.ctx->CSSetShader(nullptr, nullptr, 0);
  if (readback_wait) {  // (the host reads the field back right away)
    D3D11_MAPPED_SUBRESOURCE m;
    while (h.ctx->Map(h.stats_staging, 0, D3D11_MAP_READ, 0, &m) == DXGI_ERROR_WAS_STILL_DRAWING) {}
    h.ctx->Unmap(h.stats_staging, 0);
  }
  ++h.frame;
}

static void host_convert(host_t &h) {  // (c), as d3d_base_encode_device::convert
  h.ctx->PSSetShaderResources(0, 1, &h.capture_srv);
  h.ctx->PSSetSamplers(0, 1, &h.linear);
  h.ctx->PSSetConstantBuffers(0, 1, &h.color_matrix);
  ID3D11Buffer *none = nullptr;
  h.ctx->PSSetConstantBuffers(1, 1, &none);
  h.ctx->PSSetConstantBuffers(2, 1, &none);
  h.ctx->VSSetConstantBuffers(1, 1, &h.rotate);
  h.ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  h.ctx->IASetInputLayout(nullptr);
  D3D11_VIEWPORT y {0, 0, (float) W, (float) H, 0, 1}, uv {0, 0, (float) W / 2, (float) H / 2, 0, 1};
  h.ctx->OMSetRenderTargets(1, &h.y_rtv, nullptr);
  h.ctx->VSSetShader(h.y_vs, nullptr, 0);
  h.ctx->PSSetShader(h.y_ps, nullptr, 0);
  h.ctx->RSSetViewports(1, &y);
  h.ctx->Draw(3, 0);
  h.ctx->OMSetRenderTargets(1, &h.uv_rtv, nullptr);
  h.ctx->VSSetShader(h.uv_vs, nullptr, 0);
  h.ctx->VSSetConstantBuffers(0, 1, &h.subsample);
  h.ctx->PSSetShader(h.uv_ps, nullptr, 0);
  h.ctx->RSSetViewports(1, &uv);
  h.ctx->Draw(3, 0);
  ID3D11ShaderResourceView *ns = nullptr;
  h.ctx->PSSetShaderResources(0, 1, &ns);
  ID3D11RenderTargetView *nr = nullptr;
  h.ctx->OMSetRenderTargets(1, &nr, nullptr);
}

static int host_alone(const std::string &dir) {
  host_t h;
  host_init(h, dir, -1);
  enum { T0, TA, TB, TC, TN };
  const int kFrames = 400;
  std::vector<double> a, b, c, total;
  for (int f = 0; f < kFrames + 2; ++f) {
    ID3D11Query *dj, *ts[TN];
    D3D11_QUERY_DESC qd {D3D11_QUERY_TIMESTAMP_DISJOINT, 0};
    CHECK(h.dev->CreateQuery(&qd, &dj));
    qd.Query = D3D11_QUERY_TIMESTAMP;
    for (auto &t : ts) CHECK(h.dev->CreateQuery(&qd, &t));
    h.ctx->Begin(dj);
    h.ctx->End(ts[T0]);
    host_draw_frame(h);
    h.ctx->End(ts[TA]);
    host_motion(h, false);
    h.ctx->End(ts[TB]);
    host_convert(h);
    h.ctx->End(ts[TC]);
    h.ctx->End(dj);
    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT d;
    while (h.ctx->GetData(dj, &d, sizeof(d), 0) != S_OK) {}
    UINT64 t[TN];
    for (int i = 0; i < TN; ++i) while (h.ctx->GetData(ts[i], &t[i], 8, 0) != S_OK) {}
    if (f >= 2 && !d.Disjoint) {
      const double ms = 1000.0 / d.Frequency;
      a.push_back((t[TA] - t[T0]) * ms);
      b.push_back((t[TB] - t[TA]) * ms);
      c.push_back((t[TC] - t[TB]) * ms);
      total.push_back((t[TC] - t[T0]) * ms);
    }
    dj->Release();
    for (auto q : ts) q->Release();
  }
  std::printf("host workloads per game frame, alone (2880x1800), %zu frames:\n", total.size());
  std::printf("  (a) game frame -> capture image   %s\n", stat(a).c_str());
  std::printf("  (b) motion pass (3 sets + mask)   %s\n", stat(b).c_str());
  std::printf("  (c) P010 conversion (Y + UV)      %s\n", stat(c).c_str());
  std::printf("  total                             %s\n", stat(total).c_str());
  return 0;
}

// parts: any of "a", "b", "c" (default all); "nowait": no readback wait;
// "normal": no priority raise
static int host_load(const std::string &dir, double hz, double secs, const std::string &opts) {
  const bool all = opts.find_first_of("abc") == std::string::npos;
  const bool do_a = all || opts.find('a') != std::string::npos;
  const bool do_b = all || opts.find('b') != std::string::npos;
  const bool do_c = all || opts.find('c') != std::string::npos;
  const bool wait = opts.find("nowait") == std::string::npos;
  const bool split = opts.find("split") != std::string::npos;  // a flush after each part (as separate submissions)
  host_t h;
  // cls=normal|above|high|realtime (default realtime), tp=none|7|abs30|abs20 (default abs30): the host's own is realtime + abs30
  auto opt = [&](const char *key, const char *dflt) {
    auto i = opts.find(key);
    if (i == std::string::npos) return std::string(dflt);
    auto j = opts.find(' ', i);
    return opts.substr(i + std::strlen(key), j == std::string::npos ? std::string::npos : j - i - std::strlen(key));
  };
  const std::string cls = opt("cls=", "realtime"), tp = opt("tp=", "abs30");
  const int cls_v = cls == "normal" ? 2 : cls == "above" ? 3 : cls == "high" ? 4 : 5;  // (D3DKMT: IDLE 0, BELOW_NORMAL 1, NORMAL 2, ABOVE_NORMAL 3, HIGH 4, REALTIME 5)
  const int tp_v = tp == "none" ? INT32_MIN : tp == "7" ? 7 : tp == "abs20" ? 0x40000014 : 0x4000001E;
  host_init(h, dir, cls_v, tp_v);
  const auto period = std::chrono::duration<double>(1.0 / hz);
  auto next = std::chrono::steady_clock::now();
  const auto end = next + std::chrono::duration<double>(secs);
  uint64_t n = 0;
  std::vector<double> latency;  // submit -> the host's work for the frame done
  ID3D11Query *done;
  D3D11_QUERY_DESC qd {D3D11_QUERY_EVENT, 0};
  CHECK(h.dev->CreateQuery(&qd, &done));
  while (std::chrono::steady_clock::now() < end) {
    const auto t0 = std::chrono::steady_clock::now();
    if (do_a) { host_draw_frame(h); if (split) h.ctx->Flush(); }
    if (do_b) { host_motion(h, wait); if (split) h.ctx->Flush(); }
    if (do_c) host_convert(h);
    h.ctx->End(done);
    h.ctx->Flush();
    BOOL ok = FALSE;
    while (h.ctx->GetData(done, &ok, sizeof(ok), 0) != S_OK) { YieldProcessor(); }
    latency.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    ++n;
    next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
    std::this_thread::sleep_until(next);
  }
  std::printf("host-load: %llu frames of host work at %.0f Hz; submit -> done %s\n", (unsigned long long) n, hz, stat(latency).c_str());
  return 0;
}

// ---- the game and the hook (D3D12) -------------------------------------------

static const char *kGameCs = R"(
cbuffer c : register(b0) { uint iterations; uint width; uint height; uint y0; };
RWTexture2D<float4> target : register(u0);
[numthreads(8, 8, 1)] void main(uint3 tid : SV_DispatchThreadID) {
  uint3 id = tid + uint3(0, y0, 0);
  if (id.x >= width || id.y >= height) return;
  float2 p = float2(id.xy) / float2(width, height);
  float4 acc = float4(p, 0.5, 1);
  [loop] for (uint i = 0; i < iterations; ++i) {
    acc = frac(acc * 1.0173 + sin(acc.wzyx * 3.1 + i * 0.01));
  }
  target[id.xy] = acc;
}
)";

struct d3d12_t {
  ID3D12Device *dev;
  ID3D12CommandQueue *queue;
  ID3D12CommandAllocator *allocs[2];
  ID3D12GraphicsCommandList *lists[2];
  ID3D12CommandAllocator *alloc;  // (the slot being recorded)
  ID3D12GraphicsCommandList *list;
  ID3D12Fence *fence;
  UINT64 fence_value = 0;
  UINT64 slot_value[2] = {0, 0};
  HANDLE event;
  ID3D12QueryHeap *qheap;
  ID3D12Resource *qreads[2];
  ID3D12Resource *qread;
  UINT64 ts_freq;
  ID3D12RootSignature *root;
  ID3D12PipelineState *pso;
  ID3D12DescriptorHeap *heap;
  UINT inc;
  ID3D12Resource *back, *mv;  // the game's "back buffer" and "DLSS output"
  ID3D12Resource *shared_frame, *ring, *shared_motion;  // the hook's
};

static D3D12_RESOURCE_BARRIER transition(ID3D12Resource *r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
  D3D12_RESOURCE_BARRIER t {};
  t.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  t.Transition.pResource = r;
  t.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  t.Transition.StateBefore = a;
  t.Transition.StateAfter = b;
  return t;
}

static ID3D12Resource *texture12(ID3D12Device *dev, UINT w, UINT h, DXGI_FORMAT f, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state, bool shared) {
  D3D12_HEAP_PROPERTIES hp {D3D12_HEAP_TYPE_DEFAULT};
  D3D12_RESOURCE_DESC d {};
  d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  d.Width = w; d.Height = h; d.DepthOrArraySize = 1; d.MipLevels = 1;
  d.Format = f; d.SampleDesc.Count = 1; d.Flags = flags;
  ID3D12Resource *r;
  CHECK(dev->CreateCommittedResource(&hp, shared ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE, &d, state, nullptr, __uuidof(ID3D12Resource), (void **) &r));
  return r;
}

static void d3d12_init(d3d12_t &g) {
  CHECK(D3D12CreateDevice(nvidia_adapter(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), (void **) &g.dev));
  D3D12_COMMAND_QUEUE_DESC qd {D3D12_COMMAND_LIST_TYPE_DIRECT};
  CHECK(g.dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void **) &g.queue));
  CHECK(g.queue->GetTimestampFrequency(&g.ts_freq));
  for (int i = 0; i < 2; ++i) {
    CHECK(g.dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), (void **) &g.allocs[i]));
    CHECK(g.dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g.allocs[i], nullptr, __uuidof(ID3D12GraphicsCommandList), (void **) &g.lists[i]));
    CHECK(g.lists[i]->Close());
  }
  CHECK(g.dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void **) &g.fence));
  g.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  D3D12_QUERY_HEAP_DESC qh {D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 16, 0};  // (8 a slot)
  CHECK(g.dev->CreateQueryHeap(&qh, __uuidof(ID3D12QueryHeap), (void **) &g.qheap));
  D3D12_HEAP_PROPERTIES rb {D3D12_HEAP_TYPE_READBACK};
  D3D12_RESOURCE_DESC bd {};
  bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = 16 * 8; bd.Height = 1; bd.DepthOrArraySize = 1; bd.MipLevels = 1;
  bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  for (int i = 0; i < 2; ++i) {
    CHECK(g.dev->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, __uuidof(ID3D12Resource), (void **) &g.qreads[i]));
  }
  // root: 4 constants (b0), a UAV table (u0)
  D3D12_DESCRIPTOR_RANGE range {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 0};
  D3D12_ROOT_PARAMETER params[2] {};
  params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  params[0].Constants.Num32BitValues = 4;
  params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  params[1].DescriptorTable.NumDescriptorRanges = 1;
  params[1].DescriptorTable.pDescriptorRanges = &range;
  D3D12_ROOT_SIGNATURE_DESC rs {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
  ID3DBlob *sig, *err;
  CHECK(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err));
  CHECK(g.dev->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), __uuidof(ID3D12RootSignature), (void **) &g.root));
  auto cs = compile_text(kGameCs, "main", "cs_5_0");
  D3D12_COMPUTE_PIPELINE_STATE_DESC pd {};
  pd.pRootSignature = g.root;
  pd.CS = {cs->GetBufferPointer(), cs->GetBufferSize()};
  CHECK(g.dev->CreateComputePipelineState(&pd, __uuidof(ID3D12PipelineState), (void **) &g.pso));
  // The game's resources: its back buffer (8-bit, as Witcher 3's) and its
  // DLSS vectors; the hook's: the shared frame texture, the vector ring
  // entry, the shared motion texture (the sets stacked)
  g.back = texture12(g.dev, W, H, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, false);
  g.mv = texture12(g.dev, MW, MH, DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, false);
  g.shared_frame = texture12(g.dev, W, H, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS, D3D12_RESOURCE_STATE_COMMON, true);
  g.ring = texture12(g.dev, MW, MH, DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_SOURCE, false);
  g.shared_motion = texture12(g.dev, MW, MH * kSets, DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS, D3D12_RESOURCE_STATE_COMMON, true);
  D3D12_DESCRIPTOR_HEAP_DESC hd {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
  CHECK(g.dev->CreateDescriptorHeap(&hd, __uuidof(ID3D12DescriptorHeap), (void **) &g.heap));
  g.inc = g.dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  auto cpu = g.heap->GetCPUDescriptorHandleForHeapStart();
  g.dev->CreateUnorderedAccessView(g.mv, nullptr, nullptr, cpu);
  cpu.ptr += g.inc;
  g.dev->CreateUnorderedAccessView(g.back, nullptr, nullptr, cpu);
}

static void d3d12_wait_value(d3d12_t &g, UINT64 v) {
  if (v && g.fence->GetCompletedValue() < v) {
    CHECK(g.fence->SetEventOnCompletion(v, g.event));
    WaitForSingleObject(g.event, INFINITE);
  }
}

// The timestamps a finished slot recorded (ms): [0] frame GPU time, [1] the
// hook's eval copy, [2] the hook's Present copies, [3] render (to Present)
static std::vector<double> read_slot(d3d12_t &g, int slot) {
  UINT64 *t;
  D3D12_RANGE r {0, 5 * 8};
  CHECK(g.qreads[slot]->Map(0, &r, (void **) &t));
  const double ms = 1000.0 / (double) g.ts_freq;
  std::vector<double> out = {(t[4] - t[0]) * ms, (t[2] - t[1]) * ms, (t[4] - t[3]) * ms, (t[3] - t[0]) * ms};
  D3D12_RANGE none {0, 0};
  g.qreads[slot]->Unmap(0, &none);
  return out;
}

// One frame of the synthetic game, with the hook's work as it records it:
// "DLSS" writes the vectors (then the hook copies them into its ring), the
// main pass writes the back buffer, Present (the hook copies the frame and
// the vector sets into its shared textures). Returns timestamps (ms):
// [0] the frame's GPU time, [1] the hook's eval copy, [2] the hook's Present copies
static void submit_frame(d3d12_t &g, int slot, UINT iter_dlss, UINT iter_main, bool hook) {
  g.alloc = g.allocs[slot];
  g.list = g.lists[slot];
  g.qread = g.qreads[slot];
  CHECK(g.alloc->Reset());
  CHECK(g.list->Reset(g.alloc, g.pso));
  g.list->SetComputeRootSignature(g.root);
  g.list->SetDescriptorHeaps(1, &g.heap);
  auto gpu = g.heap->GetGPUDescriptorHandleForHeapStart();
  int q = 0;
  const int q0 = slot * 8;
  q = q0;
  g.list->EndQuery(g.qheap, D3D12_QUERY_TYPE_TIMESTAMP, q++);  // 0: start
  g.list->SetComputeRootDescriptorTable(1, gpu);
  {
    const int chunks = std::max(1, g_chunks / 4);  // ("DLSS": a quarter of the passes)
    const UINT rows = ((MH + chunks - 1) / chunks + 7) & ~7u;
    for (UINT y = 0; y < MH; y += rows) {
      UINT c1[4] = {iter_dlss, MW, MH, y};
      g.list->SetComputeRoot32BitConstants(0, 4, c1, 0);
      g.list->Dispatch((MW + 7) / 8, (std::min(rows, MH - y) + 7) / 8, 1);
    }
  }
  D3D12_RESOURCE_BARRIER b = transition(g.mv, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  g.list->ResourceBarrier(1, &b);
  g.list->EndQuery(g.qheap, D3D12_QUERY_TYPE_TIMESTAMP, q++);  // 1: after DLSS
  if (hook) {  // the hook's copy after the evaluation (motion_after_eval12)
    D3D12_RESOURCE_BARRIER before[2] = {transition(g.ring, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
                                        transition(g.mv, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE)};
    D3D12_RESOURCE_BARRIER after[2] = {transition(g.ring, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE),
                                       transition(g.mv, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)};
    D3D12_TEXTURE_COPY_LOCATION dst {g.ring, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    D3D12_TEXTURE_COPY_LOCATION src {g.mv, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    D3D12_BOX box {0, 0, 0, MW, MH, 1};
    g.list->ResourceBarrier(2, before);
    g.list->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
    g.list->ResourceBarrier(2, after);
  }
  g.list->EndQuery(g.qheap, D3D12_QUERY_TYPE_TIMESTAMP, q++);  // 2: after the eval copy
  b = transition(g.mv, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  g.list->ResourceBarrier(1, &b);
  b = transition(g.back, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  g.list->ResourceBarrier(1, &b);
  gpu.ptr += g.inc;
  g.list->SetComputeRootDescriptorTable(1, gpu);
  {
    const UINT rows = ((H + g_chunks - 1) / g_chunks + 7) & ~7u;
    for (UINT y = 0; y < H; y += rows) {
      UINT c2[4] = {iter_main, W, H, y};
      g.list->SetComputeRoot32BitConstants(0, 4, c2, 0);
      g.list->Dispatch((W + 7) / 8, (std::min(rows, H - y) + 7) / 8, 1);
    }
  }
  b = transition(g.back, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);  // (PRESENT)
  g.list->ResourceBarrier(1, &b);
  g.list->EndQuery(g.qheap, D3D12_QUERY_TYPE_TIMESTAMP, q++);  // 3: frame rendered
  if (hook) {  // the hook at Present (record12): the frame, then the vector sets
    D3D12_RESOURCE_BARRIER to_copy = transition(g.back, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_RESOURCE_BARRIER to_present = transition(g.back, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    g.list->ResourceBarrier(1, &to_copy);
    g.list->CopyResource(g.shared_frame, g.back);
    g.list->ResourceBarrier(1, &to_present);
    D3D12_TEXTURE_COPY_LOCATION dst {g.shared_motion, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    D3D12_TEXTURE_COPY_LOCATION src {g.ring, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    D3D12_BOX box {0, 0, 0, MW, MH, 1};
    for (int i = 0; i < kSets; ++i) {
      g.list->CopyTextureRegion(&dst, 0, i * MH, 0, &src, &box);
    }
  }
  g.list->EndQuery(g.qheap, D3D12_QUERY_TYPE_TIMESTAMP, q++);  // 4: after Present's copies
  g.list->ResolveQueryData(g.qheap, D3D12_QUERY_TYPE_TIMESTAMP, q0, q - q0, g.qread, 0);
  CHECK(g.list->Close());
  ID3D12CommandList *lists[] = {g.list};
  g.queue->ExecuteCommandLists(1, lists);
  CHECK(g.queue->Signal(g.fence, ++g.fence_value));
  g.slot_value[slot] = g.fence_value;
}

// One frame alone, waited for (calibration, the hook alone)
static std::vector<double> game_frame(d3d12_t &g, UINT iter_dlss, UINT iter_main, bool hook) {
  submit_frame(g, 0, iter_dlss, iter_main, hook);
  d3d12_wait_value(g, g.slot_value[0]);
  return read_slot(g, 0);
}

static void calibrate(d3d12_t &g, UINT &iter_dlss, UINT &iter_main) {
  // ~3 ms of "DLSS" and ~9 ms of main pass: a ~12 ms GPU frame, as Witcher 3's
  iter_dlss = 64;
  iter_main = 64;
  for (int k = 0; k < 6; ++k) {
    auto t1 = game_frame(g, iter_dlss, 1, false);
    const double dlss_ms = t1[3];
    auto t2 = game_frame(g, 1, iter_main, false);
    const double main_ms = t2[3];
    iter_dlss = std::max<UINT>(1, (UINT) (iter_dlss * 3.0 / std::max(0.05, dlss_ms)));
    iter_main = std::max<UINT>(1, (UINT) (iter_main * 9.0 / std::max(0.05, main_ms)));
  }
}

static int game(bool hook, double secs, UINT fixed_dlss, UINT fixed_main) {
  d3d12_t g;
  d3d12_init(g);
  UINT iter_dlss = fixed_dlss, iter_main = fixed_main;
  if (!iter_dlss || !iter_main) {
    calibrate(g, iter_dlss, iter_main);
  }
  std::vector<double> frame, eval_copy, present_copy, render;
  const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(secs);
  // GPU-bound: the next frame is always queued behind the one running (two
  // in flight), so the GPU never idles between them and anything else on it
  // has to cut into the game's frames
  uint64_t n = 0;
  while (std::chrono::steady_clock::now() < end) {
    const int slot = n % 2;
    if (g.slot_value[slot]) {
      d3d12_wait_value(g, g.slot_value[slot]);
      auto t = read_slot(g, slot);
      frame.push_back(t[0]);
      eval_copy.push_back(t[1]);
      present_copy.push_back(t[2]);
      render.push_back(t[3]);
    }
    submit_frame(g, slot, iter_dlss, iter_main, hook);
    ++n;
  }
  d3d12_wait_value(g, g.fence_value);
  std::printf("game%s: %zu frames (iterations dlss %u, main %u, %d main dispatches)\n", hook ? " + hook" : "", frame.size(), iter_dlss, iter_main, g_chunks);
  std::printf("  frame GPU time          %s\n", stat(frame).c_str());
  {
    std::vector<double> v = frame;
    std::sort(v.begin(), v.end());
    const double med = v[v.size() / 2];
    int over[4] = {};
    const double lim[4] = {0.5, 1, 2, 5};
    for (double x : frame) for (int i = 0; i < 4; ++i) if (x - med > lim[i]) ++over[i];
    std::printf("  frames over the median by >0.5/1/2/5 ms: %d / %d / %d / %d of %zu\n", over[0], over[1], over[2], over[3], frame.size());
  }
  std::printf("  render (to Present)     %s\n", stat(render).c_str());
  if (hook) {
    std::printf("  hook: eval copy         %s\n", stat(eval_copy).c_str());
    std::printf("  hook: Present copies    %s\n", stat(present_copy).c_str());
  }
  return 0;
}

static int hook_alone() {
  d3d12_t g;
  d3d12_init(g);
  std::vector<double> eval_copy, present_copy;
  for (int i = 0; i < 400; ++i) {
    auto t = game_frame(g, 1, 1, true);  // (a trivial game: the hook's work alone)
    eval_copy.push_back(t[1]);
    present_copy.push_back(t[2]);
  }
  std::printf("hook work per game frame, alone (frame 2880x1800 RGBA8, vectors 1440x900 RG16F x3):\n");
  std::printf("  eval copy (after DLSS)            %s\n", stat(eval_copy).c_str());
  std::printf("  Present: frame + vector sets      %s\n", stat(present_copy).c_str());
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::printf("usage: gpu_bench host-alone <shaders> | hook-alone | game [--hook] [secs] | host-load <shaders> <hz> <secs>\n");
    return 1;
  }
  const std::string mode = argv[1];
  if (mode == "caps") {
    IDXGIAdapter2 *a2 = nullptr;
    nvidia_adapter()->QueryInterface(__uuidof(IDXGIAdapter2), (void **) &a2);
    DXGI_ADAPTER_DESC2 d;
    a2->GetDesc2(&d);
    const char *gfx[] = {"DMA buffer", "primitive", "triangle", "pixel", "instruction"};
    const char *cmp[] = {"DMA buffer", "dispatch", "thread group", "thread", "instruction"};
    std::printf("%ls: graphics preemption %s, compute preemption %s\n", d.Description, gfx[d.GraphicsPreemptionGranularity], cmp[d.ComputePreemptionGranularity]);
    return 0;
  }
  if (mode == "host-alone" && argc >= 3) return host_alone(argv[2]);
  if (mode == "hook-alone") return hook_alone();
  if (mode == "game") {
    if (const char *c = std::getenv("GAME_CHUNKS")) g_chunks = std::max(1, std::atoi(c));
    int a = 2;
    bool hook = argc > a && std::string(argv[a]) == "--hook";
    if (hook) ++a;
    double secs = argc > a ? std::atof(argv[a++]) : 15;
    UINT dl = argc > a ? std::atoi(argv[a++]) : 0;
    UINT mn = argc > a ? std::atoi(argv[a++]) : 0;
    return game(hook, secs, dl, mn);
  }
  if (mode == "host-load" && argc >= 5) {
    std::string opts;
    for (int i = 5; i < argc; ++i) opts += std::string(argv[i]) + " ";
    return host_load(argv[2], std::atof(argv[3]), std::atof(argv[4]), opts);
  }
  std::printf("bad arguments\n");
  return 1;
}
