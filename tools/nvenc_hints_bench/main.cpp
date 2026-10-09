#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <nvEncodeAPI.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "motion.h"

using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
template<class T> struct Com {
    T* p=nullptr; ~Com(){if(p)p->Release();} T** put(){return &p;} T* operator->()const{return p;}
    Com()=default; Com(const Com&)=delete; Com& operator=(const Com&)=delete;
};
struct File { FILE* p=nullptr; ~File(){if(p)fclose(p);} };
static void hr(HRESULT r,const char* op) { if(FAILED(r)) { std::ostringstream o; o<<op<<" HRESULT=0x"<<std::hex<<uint32_t(r); throw std::runtime_error(o.str()); } }
struct Options {
    int w=3840,h=2160,frames=240,warmup=30,fps=60,preset=1,qp=28,dy=0,repeats=1;
    uint32_t bitrate=30000000;
    std::string codec="all",mode="both",recon="auto",split="off",out,hints="pair",multipass="off",texture="multiscale",scene="pan";
    int hintBlock=16,sbCu=64;
    std::string qpmap="off";
    std::vector<int> speeds{0,32,128,512,1023};
    bool dumpSource=false,ctuOrder=false,selftest=false;
};
static void usage() {
    puts("nvenc_hints.exe [--codec all|h264|hevc|av1] [--mode both|qp|cbr]\n"
         "  [--speeds 0,32,128,512,1023] [--dy 0] [--frames 240] [--warmup 30]\n"
         "  [--qp 28] [--bitrate 30000000] [--fps 60] [--preset 1..7]\n"
         "  [--hints pair|all|off|on|oracle|verified|empty|wrong] [--repeats 1] [--split off|auto|forced]\n"
         "  [--scene pan|orbit|parallax|objects|foliage|game|topdown|chase] [--hint-block 16|8] [--sb-cu 64|32|16|8] [--qpmap off|edges|persist]\n"
         "  [--texture multiscale|noise|fine|finecoarse|foliage] [--multipass off|quarter|full] [--recon auto|off|required] [--output-dir PATH] [--dump-source]\n"
         "  [--width 3840] [--height 2160] [--hevc-order raster16|ctu32] [--selftest]\n"
         "Speed: the scene's motion scale in screen px/frame (pan: translation; orbit: corner speed of the roll;\n"
         "parallax/game: near layer; objects: sprites). Hints: on = what game motion vectors know (surface motion,\n"
         "frame edge); oracle = also disocclusion and a static HUD; pair = off,on; all = off,on,oracle.\n"
         "No claimed hardware search radius; sweep establishes it empirically. --selftest needs no GPU.");
}
static Options parse(int argc,char** argv) {
    Options o;
    for(int i=1;i<argc;++i) {
        std::string k=argv[i];
        if(k=="--help"){usage();std::exit(0);} if(k=="--selftest"){o.selftest=true;continue;}
        if(k=="--dump-source"){o.dumpSource=true;continue;}
        if(++i==argc)throw std::runtime_error("missing value for "+k);
        std::string v=argv[i]; size_t used=0;
        auto number=[&](){int n=std::stoi(v,&used);if(used!=v.size())throw std::runtime_error("invalid integer "+v);return n;};
        if(k=="--codec")o.codec=v; else if(k=="--mode")o.mode=v; else if(k=="--recon")o.recon=v;
        else if(k=="--split")o.split=v; else if(k=="--multipass")o.multipass=v; else if(k=="--texture")o.texture=v; else if(k=="--scene")o.scene=v; else if(k=="--output-dir")o.out=v; else if(k=="--hints")o.hints=v;
        else if(k=="--hevc-order") {if(v!="raster16"&&v!="ctu32")throw std::runtime_error("invalid HEVC order");o.ctuOrder=v=="ctu32";}
        else if(k=="--frames")o.frames=number(); else if(k=="--warmup")o.warmup=number();
        else if(k=="--fps")o.fps=number(); else if(k=="--preset")o.preset=number(); else if(k=="--qp")o.qp=number();
        else if(k=="--dy")o.dy=number(); else if(k=="--hint-block")o.hintBlock=number(); else if(k=="--sb-cu")o.sbCu=number(); else if(k=="--qpmap")o.qpmap=v; else if(k=="--repeats")o.repeats=number();
        else if(k=="--width")o.w=number(); else if(k=="--height")o.h=number();
        else if(k=="--bitrate"){int n=number();if(n<=0)throw std::runtime_error("invalid bitrate");o.bitrate=uint32_t(n);}
        else if(k=="--speeds") {o.speeds.clear();std::istringstream s(v);std::string n;while(std::getline(s,n,',')){size_t p;int speed=std::stoi(n,&p);if(p!=n.size()||std::abs(speed)>4096)throw std::runtime_error("invalid speed");o.speeds.push_back(speed);}}
        else throw std::runtime_error("unknown option "+k);
    }
    if(o.w<768||o.h<384||o.w>8192||o.h>8192||(o.w&1)||(o.h&1)||o.frames<2||o.frames>100000||o.warmup<0||o.warmup>=o.frames
       ||o.fps<1||o.fps>1000||o.preset<1||o.preset>7||o.qp<0||o.qp>51||std::abs(o.dy)>4096||o.repeats<1||o.repeats>100||o.speeds.empty())throw std::runtime_error("option out of range");
    auto oneOf=[](const std::string& x,std::initializer_list<const char*> vals){for(auto* v:vals)if(x==v)return true;return false;};
    if(!oneOf(o.codec,{"all","h264","hevc","av1"})||!oneOf(o.mode,{"both","qp","cbr"})||!oneOf(o.recon,{"auto","off","required"})
       ||!oneOf(o.split,{"off","auto","forced"})||!oneOf(o.multipass,{"off","quarter","full"})||!oneOf(o.texture,{"multiscale","noise","fine","finecoarse","foliage"})||!oneOf(o.scene,{"pan","orbit","parallax","objects","foliage","game","topdown","chase"})
       ||(o.hintBlock!=16&&o.hintBlock!=8)||(o.sbCu!=64&&o.sbCu!=32&&o.sbCu!=16&&o.sbCu!=8)||!oneOf(o.qpmap,{"off","edges","persist"})||!oneOf(o.hints,{"pair","all","off","on","oracle","verified","empty","wrong","none","zero"}))throw std::runtime_error("invalid choice");
    if(o.dumpSource&&o.out.empty())throw std::runtime_error("--dump-source needs --output-dir");
    return o;
}
struct Runtime {
    HMODULE dll=nullptr; NV_ENCODE_API_FUNCTION_LIST api{};
    Com<IDXGIFactory1> factory; Com<IDXGIAdapter1> adapter;
    Com<ID3D11Device> device; Com<ID3D11DeviceContext> context;
    Com<ID3D11Query> ready;
    ~Runtime(){if(dll)FreeLibrary(dll);}
    void init(){
        hr(CreateDXGIFactory1(__uuidof(IDXGIFactory1),reinterpret_cast<void**>(factory.put())),"CreateDXGIFactory1");
        for(UINT i=0;;++i){IDXGIAdapter1* a=nullptr;if(factory->EnumAdapters1(i,&a)==DXGI_ERROR_NOT_FOUND)break;DXGI_ADAPTER_DESC1 d{};a->GetDesc1(&d);
            if(d.VendorId==0x10de && !(d.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)){adapter.p=a;std::wcerr<<L"NVIDIA adapter: "<<d.Description<<L" LUID="<<d.AdapterLuid.HighPart<<L":"<<d.AdapterLuid.LowPart<<L"\n";break;}a->Release();}
        if(!adapter.p)throw std::runtime_error("No NVIDIA DXGI adapter");
        D3D_FEATURE_LEVEL level=D3D_FEATURE_LEVEL_11_0;
        hr(D3D11CreateDevice(adapter.p,D3D_DRIVER_TYPE_UNKNOWN,nullptr,0,&level,1,D3D11_SDK_VERSION,device.put(),nullptr,context.put()),"D3D11CreateDevice");
        D3D11_QUERY_DESC q{D3D11_QUERY_EVENT,0};hr(device->CreateQuery(&q,ready.put()),"CreateQuery");
        wchar_t system[MAX_PATH]{};GetSystemDirectoryW(system,MAX_PATH);std::wstring path=std::wstring(system)+L"\\nvEncodeAPI64.dll";
        dll=LoadLibraryW(path.c_str());if(!dll)throw std::runtime_error("LoadLibraryW(system nvEncodeAPI64.dll), Win32="+std::to_string(GetLastError()));
        using Max=NVENCSTATUS(NVENCAPI*)(uint32_t*);using Create=NVENCSTATUS(NVENCAPI*)(NV_ENCODE_API_FUNCTION_LIST*);
        auto maxVersion=reinterpret_cast<Max>(reinterpret_cast<void*>(GetProcAddress(dll,"NvEncodeAPIGetMaxSupportedVersion")));
        auto create=reinterpret_cast<Create>(reinterpret_cast<void*>(GetProcAddress(dll,"NvEncodeAPICreateInstance")));
        if(!maxVersion||!create)throw std::runtime_error("NVENC loader entry points missing");
        uint32_t version=0;if(maxVersion(&version)!=NV_ENC_SUCCESS)throw std::runtime_error("NvEncodeAPIGetMaxSupportedVersion failed");
        std::cerr<<"NVENC max API="<<(version>>4)<<"."<<(version&15)<<" build API="<<NVENCAPI_MAJOR_VERSION<<"."<<NVENCAPI_MINOR_VERSION<<"\n";
        if(version<((NVENCAPI_MAJOR_VERSION<<4)|NVENCAPI_MINOR_VERSION))throw std::runtime_error("Driver older than header API: install an API 13.0 capable driver");
        api.version=NV_ENCODE_API_FUNCTION_LIST_VER;if(create(&api)!=NV_ENC_SUCCESS)throw std::runtime_error("NvEncodeAPICreateInstance failed");
    }
    void finishUpload(){context->End(ready.p);context->Flush();auto t=Clock::now();for(;;){HRESULT r=context->GetData(ready.p,nullptr,0,D3D11_ASYNC_GETDATA_DONOTFLUSH);if(r==S_OK)break;hr(r,"upload GetData");if(ms(t)>10000)throw std::runtime_error("GPU upload timeout");Sleep(0);}}
};
static GUID codecGuid(const std::string& c){return c=="h264"?NV_ENC_CODEC_H264_GUID:c=="hevc"?NV_ENC_CODEC_HEVC_GUID:NV_ENC_CODEC_AV1_GUID;}
static GUID presetGuid(int p){const GUID g[]={NV_ENC_PRESET_P1_GUID,NV_ENC_PRESET_P2_GUID,NV_ENC_PRESET_P3_GUID,NV_ENC_PRESET_P4_GUID,NV_ENC_PRESET_P5_GUID,NV_ENC_PRESET_P6_GUID,NV_ENC_PRESET_P7_GUID};return g[p-1];}
struct Session {
    Runtime& r;void* enc=nullptr;bool initialized=false,recon=false,locked=false;
    NV_ENC_REGISTERED_PTR inputReg=nullptr,reconReg=nullptr;
    NV_ENC_INPUT_PTR inputMap=nullptr,reconMap=nullptr;NV_ENC_OUTPUT_PTR bitstream=nullptr;
    Com<ID3D11Texture2D> input,reconstructed,staging;
    Session(Runtime& runtime):r(runtime){}
    ~Session(){
        if(!enc)return;
        if(locked)r.api.nvEncUnlockBitstream(enc,bitstream);
        if(inputMap)r.api.nvEncUnmapInputResource(enc,inputMap);
        if(reconMap)r.api.nvEncUnmapInputResource(enc,reconMap);
        if(inputReg)r.api.nvEncUnregisterResource(enc,inputReg);
        if(reconReg)r.api.nvEncUnregisterResource(enc,reconReg);
        if(bitstream)r.api.nvEncDestroyBitstreamBuffer(enc,bitstream);
        r.api.nvEncDestroyEncoder(enc);
    }
    void check(NVENCSTATUS s,const char* op){if(s!=NV_ENC_SUCCESS){std::string msg=std::string(op)+" NVENCSTATUS="+std::to_string(s);if(enc&&r.api.nvEncGetLastErrorString){const char* e=r.api.nvEncGetLastErrorString(enc);if(e)msg+=" "+std::string(e);}throw std::runtime_error(msg);}}
    int cap(GUID codec,NV_ENC_CAPS c){NV_ENC_CAPS_PARAM p{};p.version=NV_ENC_CAPS_PARAM_VER;p.capsToQuery=c;int v=0;auto status=r.api.nvEncGetEncodeCaps(enc,codec,&p,&v);if(status!=NV_ENC_SUCCESS){std::cerr<<"cap "<<c<<" query failed="<<status<<"\n";return 0;}return v;}
    void init(const Options& o,const std::string& codec,const std::string& mode,const std::string& hint){
        NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS open{};open.version=NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;open.device=r.device.p;open.deviceType=NV_ENC_DEVICE_TYPE_DIRECTX;open.apiVersion=NVENCAPI_VERSION;
        check(r.api.nvEncOpenEncodeSessionEx(&open,&enc),"OpenEncodeSessionEx");
        GUID cg=codecGuid(codec), pg=presetGuid(o.preset);
        uint32_t n=0;check(r.api.nvEncGetEncodeGUIDCount(enc,&n),"GetEncodeGUIDCount");std::vector<GUID> supported(n);
        check(r.api.nvEncGetEncodeGUIDs(enc,supported.data(),n,&n),"GetEncodeGUIDs");bool found=false;for(auto& g:supported)if(IsEqualGUID(g,cg))found=true;if(!found)throw std::runtime_error(codec+" unsupported on this device");
        check(r.api.nvEncGetInputFormatCount(enc,cg,&n),"GetInputFormatCount");std::vector<NV_ENC_BUFFER_FORMAT> formats(n);
        check(r.api.nvEncGetInputFormats(enc,cg,formats.data(),n,&n),"GetInputFormats");found=false;for(auto f:formats)if(f==NV_ENC_BUFFER_FORMAT_NV12)found=true;if(!found)throw std::runtime_error("NV12 input unsupported");
        int supportsRecon=cap(cg,NV_ENC_CAPS_OUTPUT_RECON_SURFACE);
        recon=o.recon!="off"&&supportsRecon;
        if(o.recon=="required"&&!recon)throw std::runtime_error("Reconstructed output capability unavailable");
        std::cerr<<"caps codec="<<codec<<" recon="<<supportsRecon<<" engines="<<cap(cg,NV_ENC_CAPS_NUM_ENCODER_ENGINES)
                 <<" maxB="<<cap(cg,NV_ENC_CAPS_NUM_MAX_BFRAMES)<<" multipleRefs="<<cap(cg,NV_ENC_CAPS_SUPPORT_MULTIPLE_REF_FRAMES)
                 <<" intraRefresh="<<cap(cg,NV_ENC_CAPS_SUPPORT_INTRA_REFRESH)<<"\n";
        if(!recon)std::cerr<<"PSNR_Y=N/A: reconstructed output "<<(o.recon=="off"?"disabled":"unsupported")<<"; optional streams can be decoded externally.\n";
        NV_ENC_PRESET_CONFIG preset{};preset.version=NV_ENC_PRESET_CONFIG_VER;preset.presetCfg.version=NV_ENC_CONFIG_VER;
        check(r.api.nvEncGetEncodePresetConfigEx(enc,cg,pg,NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY,&preset),"GetEncodePresetConfigEx");
        auto cfg=preset.presetCfg;cfg.version=NV_ENC_CONFIG_VER;cfg.profileGUID=NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID;
        cfg.gopLength=NVENC_INFINITE_GOPLENGTH;cfg.frameIntervalP=1;cfg.frameFieldMode=NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;
        cfg.rcParams={};cfg.rcParams.version=NV_ENC_RC_PARAMS_VER;
        cfg.rcParams.rateControlMode=mode=="qp"?NV_ENC_PARAMS_RC_CONSTQP:NV_ENC_PARAMS_RC_CBR;
        cfg.rcParams.constQP.qpIntra=o.qp;cfg.rcParams.constQP.qpInterP=o.qp;cfg.rcParams.constQP.qpInterB=o.qp;
        if(mode=="cbr"){cfg.rcParams.averageBitRate=o.bitrate;cfg.rcParams.maxBitRate=o.bitrate;cfg.rcParams.vbvBufferSize=o.bitrate/o.fps;cfg.rcParams.vbvInitialDelay=cfg.rcParams.vbvBufferSize;}
        if(o.qpmap!="off"){if(codec!="av1")throw std::runtime_error("--qpmap is implemented for AV1 only");cfg.rcParams.qpMapMode=NV_ENC_QP_MAP_DELTA;}
        cfg.rcParams.zeroReorderDelay=1;cfg.rcParams.multiPass=o.multipass=="quarter"?NV_ENC_TWO_PASS_QUARTER_RESOLUTION:o.multipass=="full"?NV_ENC_TWO_PASS_FULL_RESOLUTION:NV_ENC_MULTI_PASS_DISABLED;
        if(codec=="h264"){auto& c=cfg.encodeCodecConfig.h264Config;c.idrPeriod=NVENC_INFINITE_GOPLENGTH;c.maxNumRefFrames=1;c.numRefL0=NV_ENC_NUM_REF_FRAMES_1;c.chromaFormatIDC=1;c.enableIntraRefresh=0;c.enableLTR=0;c.enableFillerDataInsertion=0;}
        if(codec=="hevc"){auto& c=cfg.encodeCodecConfig.hevcConfig;c.idrPeriod=NVENC_INFINITE_GOPLENGTH;c.maxNumRefFramesInDPB=1;c.numRefL0=NV_ENC_NUM_REF_FRAMES_1;c.chromaFormatIDC=1;c.enableIntraRefresh=0;c.enableLTR=0;c.enableFillerDataInsertion=0;}
        if(codec=="av1"){auto& c=cfg.encodeCodecConfig.av1Config;c.idrPeriod=NVENC_INFINITE_GOPLENGTH;c.maxNumRefFramesInDPB=1;c.numFwdRefs=NV_ENC_NUM_REF_FRAMES_1;c.chromaFormatIDC=1;c.enableIntraRefresh=0;c.enableLTR=0;c.enableBitstreamPadding=0;}
        NV_ENC_INITIALIZE_PARAMS p{};p.version=NV_ENC_INITIALIZE_PARAMS_VER;p.encodeGUID=cg;p.presetGUID=pg;
        p.encodeWidth=o.w;p.encodeHeight=o.h;p.darWidth=o.w;p.darHeight=o.h;p.frameRateNum=o.fps;p.frameRateDen=1;
        p.enablePTD=1;p.enableEncodeAsync=0;p.enableExternalMEHints=hint!="off";p.enableReconFrameOutput=recon;
        p.encodeConfig=&cfg;p.tuningInfo=NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
        p.splitEncodeMode=o.split=="off"?NV_ENC_SPLIT_DISABLE_MODE:o.split=="auto"?NV_ENC_SPLIT_AUTO_MODE:NV_ENC_SPLIT_AUTO_FORCED_MODE;
        if(p.enableExternalMEHints){if(codec=="av1")p.maxMEHintCountsPerBlock[0].numCandsPerSb=(64/o.sbCu)*(64/o.sbCu);else if(o.hintBlock==8)p.maxMEHintCountsPerBlock[0].numCandsPerBlk8x8=1;else p.maxMEHintCountsPerBlock[0].numCandsPerBlk16x16=1;}
        check(r.api.nvEncInitializeEncoder(enc,&p),"InitializeEncoder");initialized=true;
        D3D11_TEXTURE2D_DESC d{};d.Width=o.w;d.Height=o.h;d.MipLevels=1;d.ArraySize=1;d.Format=DXGI_FORMAT_NV12;d.SampleDesc.Count=1;d.Usage=D3D11_USAGE_DEFAULT;d.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        hr(r.device->CreateTexture2D(&d,nullptr,input.put()),"Create input NV12");inputReg=reg(input.p,o,NV_ENC_INPUT_IMAGE);
        if(recon){hr(r.device->CreateTexture2D(&d,nullptr,reconstructed.put()),"Create recon NV12");reconReg=reg(reconstructed.p,o,NV_ENC_OUTPUT_RECON);
            d.Usage=D3D11_USAGE_STAGING;d.BindFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;hr(r.device->CreateTexture2D(&d,nullptr,staging.put()),"Create recon staging");}
        NV_ENC_CREATE_BITSTREAM_BUFFER b{};b.version=NV_ENC_CREATE_BITSTREAM_BUFFER_VER;check(r.api.nvEncCreateBitstreamBuffer(enc,&b),"CreateBitstreamBuffer");bitstream=b.bitstreamBuffer;
    }
    NV_ENC_REGISTERED_PTR reg(ID3D11Texture2D* tex,const Options& o,NV_ENC_BUFFER_USAGE usage){
        NV_ENC_REGISTER_RESOURCE reg{};reg.version=NV_ENC_REGISTER_RESOURCE_VER;reg.resourceType=NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX;
        reg.width=o.w;reg.height=o.h;reg.resourceToRegister=tex;reg.bufferFormat=NV_ENC_BUFFER_FORMAT_NV12;reg.bufferUsage=usage;
        check(r.api.nvEncRegisterResource(enc,&reg),"RegisterResource");
        if(usage==NV_ENC_OUTPUT_RECON)std::cerr<<"recon driver_chroma_offset="<<reg.chromaOffset[0]<<" (PSNR reads luma only)\n";
        return reg.registeredResource;
    }
    NV_ENC_INPUT_PTR map(NV_ENC_REGISTERED_PTR registered){NV_ENC_MAP_INPUT_RESOURCE m{};m.version=NV_ENC_MAP_INPUT_RESOURCE_VER;m.registeredResource=registered;check(r.api.nvEncMapInputResource(enc,&m),"MapInputResource");return m.mappedResource;}
    /// Luma SSE of the reconstruction, total and per scoring region
    double sse(const std::vector<uint8_t>& source,const std::vector<uint8_t>& regions,const Options& o,double* regionSse,double* regionCount,double& ssim){
        r.context->CopyResource(staging.p,reconstructed.p);D3D11_MAPPED_SUBRESOURCE mapped{};
        hr(r.context->Map(staging.p,0,D3D11_MAP_READ,0,&mapped),"Map reconstructed staging");
        // Per-row partial sums, rows spread over all processors
        std::vector<double> rowSse(size_t(o.h)*motion::kRegions),rowCount(size_t(o.h)*motion::kRegions),rowSsim(size_t(o.h/8));
        // SSIM over non-overlapping 8x8 windows (luma)
        motion::parallel_rows(o.h/8,[&](int w0,int w1){for(int wy=w0;wy<w1;++wy){double acc=0;for(int wx=0;wx<o.w/8;++wx){double sa=0,sb2=0,saa=0,sbb=0,sab=0;
            for(int y=wy*8;y<wy*8+8;++y){auto* row=static_cast<const uint8_t*>(mapped.pData)+size_t(y)*mapped.RowPitch;for(int x=wx*8;x<wx*8+8;++x){double a=source[size_t(y)*o.w+x],b=row[x];sa+=a;sb2+=b;saa+=a*a;sbb+=b*b;sab+=a*b;}}
            double ma=sa/64,mb=sb2/64,va=saa/64-ma*ma,vb=sbb/64-mb*mb,cov=sab/64-ma*mb,c1=6.5025,c2=58.5225;
            acc+=((2*ma*mb+c1)*(2*cov+c2))/((ma*ma+mb*mb+c1)*(va+vb+c2));}rowSsim[size_t(wy)]=acc/(o.w/8);}});
        ssim=0;for(double v:rowSsim)ssim+=v;ssim/=double(rowSsim.size());
        motion::parallel_rows(o.h,[&](int y0,int y1){for(int y=y0;y<y1;++y){auto* row=static_cast<const uint8_t*>(mapped.pData)+size_t(y)*mapped.RowPitch;double* e=&rowSse[size_t(y)*motion::kRegions];double* c=&rowCount[size_t(y)*motion::kRegions];
            for(int x=0;x<o.w;++x){size_t i=size_t(y)*o.w+x;int d=int(row[x])-source[i];e[regions[i]]+=double(d)*d;c[regions[i]]+=1;}}});
        r.context->Unmap(staging.p,0);
        double sum=0;for(size_t i=0;i<rowSse.size();++i){sum+=rowSse[i];regionSse[i%motion::kRegions]+=rowSse[i];regionCount[i%motion::kRegions]+=rowCount[i];}
        return sum;
    }
};
static double psnr(double sse,double samples){return sse==0?INFINITY:10*std::log10(255.0*255.0*samples/sse);}
static void checkedWrite(FILE* f,const void* p,size_t n){if(f&&fwrite(p,1,n,f)!=n)throw std::runtime_error("output write failed");}
static void run(Runtime& r,const Options& o,const std::string& codec,const std::string& mode,const std::string& hints,int dx,int repeat){
    std::string id=codec+"_"+mode+"_"+o.scene+"_"+o.texture+"_mp"+o.multipass+"_sp"+o.split+"_P"+std::to_string(o.preset)+"_b"+std::to_string(codec=="av1"?o.sbCu:o.hintBlock)+(o.qpmap!="off"?"_qp"+o.qpmap:"")+"_"+hints+"_s"+std::to_string(dx)+"_dy"+std::to_string(o.dy)+"_r"+std::to_string(repeat);
    std::cerr<<"begin "<<id<<" P"<<o.preset<<" split="<<o.split<<" multipass="<<o.multipass<<" PTD=1 refs=1 ahead=0 AQ=0 vbv_frames=1\n";
    Session s(r);s.init(o,codec,mode,hints);
    File stream,raw;
    if(!o.out.empty()){
        std::string path=o.out+"/"+id+(codec=="av1"?".obu":codec=="h264"?".h264":".hevc");stream.p=fopen(path.c_str(),"wb");if(!stream.p)throw std::runtime_error("cannot open "+path);
        if(o.dumpSource){raw.p=fopen((o.out+"/"+id+".nv12").c_str(),"wb");if(!raw.p)throw std::runtime_error("cannot open source dump");}
    }
    motion::Scene scene(motion::kind_from(o.scene),dx,o.dy,o.w,o.h,motion::texture_from(o.texture));
    const motion::Hints mode_=hints=="on"?motion::Hints::on:hints=="oracle"?motion::Hints::oracle:hints=="verified"?motion::Hints::verified:hints=="wrong"?motion::Hints::wrong:hints=="empty"?motion::Hints::empty:hints=="none"?motion::Hints::none:hints=="zero"?motion::Hints::zero:motion::Hints::off;
    const bool sendHints=mode_==motion::Hints::on||mode_==motion::Hints::oracle||mode_==motion::Hints::verified||mode_==motion::Hints::wrong||mode_==motion::Hints::none||mode_==motion::Hints::zero;
    std::cerr<<"scene="<<o.scene<<" texture="<<o.texture<<" hint_block="<<o.hintBlock<<" sb_cu="<<o.sbCu<<" HEVC_order="<<(o.ctuOrder?"ctu32_experiment":"raster16")<<"\n";
    uint64_t totalBytes=0;double totalTime=0,totalSse=0,totalHintMs=0;std::vector<double> times;std::vector<uint8_t> source,regions,previous;
    double regionSse[motion::kRegions]{},regionCount[motion::kRegions]{};
    std::vector<NVENC_EXTERNAL_ME_HINT> mb;std::vector<NVENC_EXTERNAL_ME_SB_HINT> sb;std::vector<int8_t> qpm;
    std::vector<uint32_t> frameBytes;double totalSsim=0;
    for(int f=0;f<o.frames;++f){
        previous.swap(source);auto t=Clock::now();motion::frame(source,regions,scene,f);double gen=ms(t);
        motion::Frames frames{source.data(),previous.data(),o.w,o.h};
        double hintMs=0;
        if(f&&sendHints){t=Clock::now();if(codec=="av1")motion::sb_hints(sb,scene,f,mode_,o.sbCu,&frames);else motion::mb_hints(mb,scene,f,mode_,o.hintBlock==8,codec=="hevc"&&o.ctuOrder,&frames);hintMs=ms(t);}checkedWrite(raw.p,source.data(),source.size());
        t=Clock::now();r.context->UpdateSubresource(s.input.p,0,nullptr,source.data(),o.w,0);r.finishUpload();double upload=ms(t);
        s.inputMap=s.map(s.inputReg);if(s.recon)s.reconMap=s.map(s.reconReg);
        NV_ENC_PIC_PARAMS p{};p.version=NV_ENC_PIC_PARAMS_VER;p.inputWidth=o.w;p.inputHeight=o.h;p.inputBuffer=s.inputMap;
        p.bufferFmt=NV_ENC_BUFFER_FORMAT_NV12;p.outputBitstream=s.bitstream;p.pictureStruct=NV_ENC_PIC_STRUCT_FRAME;p.frameIdx=f;p.inputTimeStamp=f;p.inputDuration=1;
        if(f==0)p.encodePicFlags|=NV_ENC_PIC_FLAG_FORCEIDR;
        if(s.recon){p.outputReconBuffer=s.reconMap;p.encodePicFlags|=NV_ENC_PIC_FLAG_OUTPUT_RECON_FRAME;}
        if(o.qpmap!="off"){motion::qp_map(qpm,scene,f,o.qpmap,4);p.qpDeltaMap=qpm.data();p.qpDeltaMapSize=uint32_t(qpm.size());}
        if(f&&sendHints){
            p.meHintRefPicDist[0]=1;
            if(codec=="av1"){p.meHintCountsPerBlock[0].numCandsPerSb=(64/o.sbCu)*(64/o.sbCu);p.meExternalSbHints=sb.data();p.meSbHintsCount=uint32_t(sb.size());}
            else{if(o.hintBlock==8)p.meHintCountsPerBlock[0].numCandsPerBlk8x8=1;else p.meHintCountsPerBlock[0].numCandsPerBlk16x16=1;p.meExternalHints=mb.data();}
        }
        t=Clock::now();auto status=r.api.nvEncEncodePicture(s.enc,&p);double submit=ms(t);
        // No B frames/lookahead: NEED_MORE_INPUT would invalidate this one-buffer experiment. Do not deadlock on LockBitstream.
        s.check(status,"EncodePicture (one-buffer P-only experiment)");
        NV_ENC_LOCK_BITSTREAM lock{};lock.version=NV_ENC_LOCK_BITSTREAM_VER;lock.outputBitstream=s.bitstream;lock.doNotWait=0;
        s.check(r.api.nvEncLockBitstream(s.enc,&lock),"LockBitstream");s.locked=true;double encode=ms(t);
        if(lock.outputTimeStamp!=uint64_t(f))throw std::runtime_error("output timestamp mismatch; refusing mispaired metrics");
        uint32_t bytes=lock.bitstreamSizeInBytes,avgQp=lock.frameAvgQP,picType=lock.pictureType;
        checkedWrite(stream.p,lock.bitstreamBufferPtr,bytes);
        s.check(r.api.nvEncUnlockBitstream(s.enc,s.bitstream),"UnlockBitstream");s.locked=false;
        s.check(r.api.nvEncUnmapInputResource(s.enc,s.inputMap),"Unmap input");s.inputMap=nullptr;
        double err=0,quality=NAN,reconMs=0;
        if(s.recon){s.check(r.api.nvEncUnmapInputResource(s.enc,s.reconMap),"Unmap recon");s.reconMap=nullptr;t=Clock::now();double rs[motion::kRegions]{},rc[motion::kRegions]{};double ssim=0;err=s.sse(source,regions,o,rs,rc,ssim);if(f>=o.warmup)totalSsim+=ssim;
            if(f>=o.warmup)for(int k=0;k<motion::kRegions;++k){regionSse[k]+=rs[k];regionCount[k]+=rc[k];}
            quality=psnr(err,double(o.w)*o.h);reconMs=ms(t);}
        if(f>=o.warmup){totalBytes+=bytes;totalTime+=encode;totalSse+=err;totalHintMs+=hintMs;times.push_back(encode);frameBytes.push_back(bytes);}
        std::cout<<id<<","<<f<<","<<bytes<<","<<submit<<","<<encode<<","<<quality<<","<<avgQp<<","<<picType<<","<<gen<<","<<hintMs<<","<<upload<<","<<reconMs<<","<<(f>=o.warmup)<<"\n";
    }
    NV_ENC_PIC_PARAMS eos{};eos.version=NV_ENC_PIC_PARAMS_VER;eos.encodePicFlags=NV_ENC_PIC_FLAG_EOS;s.check(r.api.nvEncEncodePicture(s.enc,&eos),"EOS");
    std::sort(times.begin(),times.end());size_t n=times.size();
    std::cout<<"# summary "<<id<<" measured_frames="<<n<<" bytes_per_frame="<<double(totalBytes)/n<<" achieved_mbps="<<double(totalBytes)*8*o.fps/n/1e6
             <<" mean_encode_wait_ms="<<totalTime/n<<" p95_encode_wait_ms="<<times[std::min(n-1,size_t(std::ceil(n*.95)-1))]
             <<" psnr_y="<<(s.recon?psnr(totalSse,double(o.w)*o.h*n):NAN)<<" hint_ms="<<totalHintMs/n<<" ssim="<<totalSsim/n;
    {std::sort(frameBytes.begin(),frameBytes.end());double budget=double(o.bitrate)/o.fps/8;
     std::cout<<" p99_frame_vs_budget="<<frameBytes[std::min(n-1,size_t(std::ceil(n*.99)-1))]/budget<<" max_frame_vs_budget="<<frameBytes.back()/budget;}
    // Per region: PSNR and share of pixels (regions without pixels are omitted)
    for(int k=0;k<motion::kRegions;++k)if(regionCount[k]>0)
        std::cout<<" "<<motion::region_name(k)<<"="<<psnr(regionSse[k],regionCount[k])<<"@"<<std::setprecision(1)<<100*regionCount[k]/(double(o.w)*o.h*n)<<"%"<<std::setprecision(5);
    std::cout<<"\n";
}
int main(int argc,char** argv){
    try{
        auto o=parse(argc,argv);motion::selftest();if(o.selftest){puts("motion/ABI self-tests passed");return 0;}
        if(!o.out.empty()&&!CreateDirectoryA(o.out.c_str(),nullptr)&&GetLastError()!=ERROR_ALREADY_EXISTS)throw std::runtime_error("cannot create output directory (parent must exist)");
        Runtime r;r.init();std::cout<<std::fixed<<std::setprecision(5);
        std::cout<<"# width="<<o.w<<" height="<<o.h<<" fps="<<o.fps<<" qp="<<o.qp<<" bitrate="<<o.bitrate<<" preset=P"<<o.preset<<" split="<<o.split<<" recon="<<o.recon<<" scene="<<o.scene<<" texture="<<o.texture<<" hint_block="<<o.hintBlock<<"\n";
        std::cout<<"case,frame,bytes,submit_ms,encode_wait_ms,psnr_y,avg_qp,picture_type,generation_ms,hint_ms,upload_sync_ms,recon_readback_ms,measured\n";
        int failures=0,successes=0;
        for(const std::string codec:{"h264","hevc","av1"})if(o.codec=="all"||o.codec==codec)
            for(const std::string mode:{"qp","cbr"})if(o.mode=="both"||o.mode==mode)
                for(int speed:o.speeds)for(int rep=0;rep<o.repeats;++rep){
                    std::vector<std::string> hints=o.hints=="pair"?std::vector<std::string>{"off","on"}:o.hints=="all"?std::vector<std::string>{"off","on","oracle"}:std::vector<std::string>{o.hints};
                    if(rep&1)std::reverse(hints.begin(),hints.end());
                    for(auto& hint:hints)try{run(r,o,codec,mode,hint,speed,rep);++successes;}
                    catch(const std::exception& e){++failures;std::cerr<<"CASE FAILED codec="<<codec<<" mode="<<mode<<" hint="<<hint<<" speed="<<speed<<" rep="<<rep<<": "<<e.what()<<"\n";
                        std::cout<<"# FAILED codec="<<codec<<" mode="<<mode<<" hint="<<hint<<" speed="<<speed<<" rep="<<rep<<"\n";}
                }
        std::cerr<<"Completed="<<successes<<" failed="<<failures<<". Rejected hint cases are NOT no-benefit measurements.\n";return failures?1:0;
    }catch(const std::exception& e){std::cerr<<"fatal: "<<e.what()<<"\n";return 1;}
}