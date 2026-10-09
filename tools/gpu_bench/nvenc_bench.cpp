// NVENC load as the host runs it (AV1 P4, ultra-low-latency tuning, 10-bit
// P010 2880x1800, two-pass at quarter resolution, split-frame encode forced,
// CBR), at a given rate, to run beside gpu_bench's synthetic game.
// Usage: nvenc_bench <hz> <secs> [1pass] [nosplit] [prio=high|realtime|normal]
#include <d3d11.h>
#include <dxgi.h>
#include <ffnvcodec/nvEncodeAPI.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define CHECK(x) do { HRESULT hr_ = (x); if (FAILED(hr_)) { std::printf("FAILED 0x%08lx: %s (line %d)\n", (unsigned long) hr_, #x, __LINE__); std::exit(1); } } while (0)
#define NVCHECK(x) do { NVENCSTATUS s_ = (x); if (s_ != NV_ENC_SUCCESS) { std::printf("NVENC FAILED %d: %s (line %d)\n", (int) s_, #x, __LINE__); std::exit(1); } } while (0)

static const uint32_t W = 2880, H = 1800;

static std::string stat(std::vector<double> v) {
  if (v.empty()) return "n/a";
  std::sort(v.begin(), v.end());
  char buf[128];
  std::snprintf(buf, sizeof(buf), "p50 %7.3f  p95 %7.3f  p99 %7.3f  max %7.3f ms", v[v.size() / 2], v[v.size() * 95 / 100], v[v.size() * 99 / 100], v.back());
  return buf;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::printf("usage: nvenc_bench <hz> <secs> [1pass] [nosplit] [prio=high|realtime|normal]\n");
    return 1;
  }
  const double hz = std::atof(argv[1]), secs = std::atof(argv[2]);
  std::string opts;
  for (int i = 3; i < argc; ++i) opts += std::string(argv[i]) + " ";
  const bool two_pass = opts.find("1pass") == std::string::npos;
  const bool split = opts.find("nosplit") == std::string::npos;
  const int prio = opts.find("prio=realtime") != std::string::npos ? 5 : opts.find("prio=normal") != std::string::npos ? 2 : 4;

  IDXGIFactory1 *factory;
  CHECK(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **) &factory));
  IDXGIAdapter1 *adapter = nullptr, *pick = nullptr;
  for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
    DXGI_ADAPTER_DESC1 d;
    adapter->GetDesc1(&d);
    if (d.VendorId == 0x10DE && !pick) pick = adapter; else adapter->Release();
  }
  ID3D11Device *dev;
  ID3D11DeviceContext *ctx;
  D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1;
  CHECK(D3D11CreateDevice(pick, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, &fl, 1, D3D11_SDK_VERSION, &dev, nullptr, &ctx));
  {
    // The host's GPU scheduling (now: HIGH class, relative +7)
    typedef LONG(APIENTRY * set_prio_t)(HANDLE, int);
    auto set = (set_prio_t) GetProcAddress(GetModuleHandleW(L"gdi32.dll"), "D3DKMTSetProcessSchedulingPriorityClass");
    LONG r = set ? set(GetCurrentProcess(), prio) : -1;
    IDXGIDevice *dxgi;
    dev->QueryInterface(__uuidof(IDXGIDevice), (void **) &dxgi);
    HRESULT t = dxgi->SetGPUThreadPriority(7);
    std::printf("nvenc_bench: class %d %s, relative +7 %s; %s, split %s\n", prio, r == 0 ? "ok" : "FAILED", SUCCEEDED(t) ? "ok" : "FAILED",
                two_pass ? "two-pass quarter" : "one pass", split ? "forced" : "off");
  }

  // Two P010 frames of noise, alternated (static content would barely encode)
  std::mt19937 rng(3);
  ID3D11Texture2D *frames[2];
  for (int f = 0; f < 2; ++f) {
    std::vector<uint16_t> data(W * H * 3 / 2);
    for (auto &v : data) v = (uint16_t) ((rng() & 1023) << 6);
    D3D11_TEXTURE2D_DESC td {};
    td.Width = W; td.Height = H; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_P010; td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_RENDER_TARGET;
    D3D11_SUBRESOURCE_DATA sd {data.data(), W * 2, 0};
    CHECK(dev->CreateTexture2D(&td, &sd, &frames[f]));
  }

  HMODULE lib = LoadLibraryW(L"nvEncodeAPI64.dll");
  if (!lib) { std::printf("no nvEncodeAPI64.dll\n"); return 1; }
  auto create = (NVENCSTATUS(NVENCAPI *)(NV_ENCODE_API_FUNCTION_LIST *)) GetProcAddress(lib, "NvEncodeAPICreateInstance");
  NV_ENCODE_API_FUNCTION_LIST nv {NV_ENCODE_API_FUNCTION_LIST_VER};
  NVCHECK(create(&nv));
  NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS op {NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER};
  op.device = dev;
  op.deviceType = NV_ENC_DEVICE_TYPE_DIRECTX;
  op.apiVersion = NVENCAPI_VERSION;
  void *enc = nullptr;
  NVCHECK(nv.nvEncOpenEncodeSessionEx(&op, &enc));

  NV_ENC_PRESET_CONFIG preset {};
  preset.version = NV_ENC_PRESET_CONFIG_VER;
  preset.presetCfg.version = NV_ENC_CONFIG_VER;
  NVCHECK(nv.nvEncGetEncodePresetConfigEx(enc, NV_ENC_CODEC_AV1_GUID, NV_ENC_PRESET_P4_GUID, NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY, &preset));
  NV_ENC_CONFIG cfg = preset.presetCfg;
  cfg.gopLength = NVENC_INFINITE_GOPLENGTH;
  cfg.frameIntervalP = 1;
  cfg.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
  cfg.rcParams.averageBitRate = 150'000'000;
  cfg.rcParams.vbvBufferSize = cfg.rcParams.averageBitRate / 70 * 5 / 2;  // (vbv +150%)
  cfg.rcParams.enableLookahead = 0;
  cfg.rcParams.multiPass = two_pass ? NV_ENC_TWO_PASS_QUARTER_RESOLUTION : NV_ENC_MULTI_PASS_DISABLED;
  cfg.rcParams.enableAQ = 0;
  cfg.encodeCodecConfig.av1Config.inputBitDepth = NV_ENC_BIT_DEPTH_10;
  cfg.encodeCodecConfig.av1Config.outputBitDepth = NV_ENC_BIT_DEPTH_10;
  cfg.encodeCodecConfig.av1Config.idrPeriod = NVENC_INFINITE_GOPLENGTH;
  NV_ENC_INITIALIZE_PARAMS init {NV_ENC_INITIALIZE_PARAMS_VER};
  init.encodeGUID = NV_ENC_CODEC_AV1_GUID;
  init.presetGUID = NV_ENC_PRESET_P4_GUID;
  init.tuningInfo = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
  init.encodeWidth = init.darWidth = W;
  init.encodeHeight = init.darHeight = H;
  init.frameRateNum = 70;
  init.frameRateDen = 1;
  init.enablePTD = 1;
  init.encodeConfig = &cfg;
  init.splitEncodeMode = split ? NV_ENC_SPLIT_AUTO_FORCED_MODE : NV_ENC_SPLIT_DISABLE_MODE;
  NVCHECK(nv.nvEncInitializeEncoder(enc, &init));

  NV_ENC_REGISTERED_PTR reg[2];
  for (int f = 0; f < 2; ++f) {
    NV_ENC_REGISTER_RESOURCE rr {NV_ENC_REGISTER_RESOURCE_VER};
    rr.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX;
    rr.resourceToRegister = frames[f];
    rr.width = W;
    rr.height = H;
    rr.bufferFormat = NV_ENC_BUFFER_FORMAT_YUV420_10BIT;
    rr.bufferUsage = NV_ENC_INPUT_IMAGE;
    NVCHECK(nv.nvEncRegisterResource(enc, &rr));
    reg[f] = rr.registeredResource;
  }
  NV_ENC_CREATE_BITSTREAM_BUFFER bb {NV_ENC_CREATE_BITSTREAM_BUFFER_VER};
  NVCHECK(nv.nvEncCreateBitstreamBuffer(enc, &bb));

  const auto period = std::chrono::duration<double>(1.0 / hz);
  auto next = std::chrono::steady_clock::now();
  const auto end = next + std::chrono::duration<double>(secs);
  std::vector<double> latency;
  uint64_t n = 0, bytes = 0;
  while (std::chrono::steady_clock::now() < end) {
    const auto t0 = std::chrono::steady_clock::now();
    NV_ENC_MAP_INPUT_RESOURCE mr {NV_ENC_MAP_INPUT_RESOURCE_VER};
    mr.registeredResource = reg[n & 1];
    NVCHECK(nv.nvEncMapInputResource(enc, &mr));
    NV_ENC_PIC_PARAMS pp {NV_ENC_PIC_PARAMS_VER};
    pp.inputWidth = W;
    pp.inputHeight = H;
    pp.inputBuffer = mr.mappedResource;
    pp.bufferFmt = mr.mappedBufferFmt;
    pp.outputBitstream = bb.bitstreamBuffer;
    pp.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
    pp.inputTimeStamp = n;
    NVCHECK(nv.nvEncEncodePicture(enc, &pp));
    NV_ENC_LOCK_BITSTREAM lb {NV_ENC_LOCK_BITSTREAM_VER};
    lb.outputBitstream = bb.bitstreamBuffer;
    NVCHECK(nv.nvEncLockBitstream(enc, &lb));
    bytes += lb.bitstreamSizeInBytes;
    NVCHECK(nv.nvEncUnlockBitstream(enc, bb.bitstreamBuffer));
    NVCHECK(nv.nvEncUnmapInputResource(enc, mr.mappedResource));
    latency.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    ++n;
    next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
    std::this_thread::sleep_until(next);
  }
  std::printf("nvenc_bench: %llu frames at %.0f Hz, %.1f Mbit/s; encode (submit -> bitstream) %s\n", (unsigned long long) n, hz,
              bytes * 8.0 / 1e6 / secs, stat(latency).c_str());
  return 0;
}
