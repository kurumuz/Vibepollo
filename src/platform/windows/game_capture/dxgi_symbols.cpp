/**
 * @file src/platform/windows/game_capture/dxgi_symbols.cpp
 * @brief Resolves dxgi!CDXGISwapChain::PresentImpl for the game hook (see the
 *        header). Runs once in the background; the result is cached per
 *        dxgi.dll build under appdata, so the symbol download happens once
 *        per Windows update.
 */
#include "dxgi_symbols.h"

#include "src/logging.h"
#include "src/platform/common.h"

#include <windows.h>
#include <dbghelp.h>

#include <curl/curl.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace game_capture {
  namespace {
    // PresentImpl as the hook forwards it: HRESULT (this, const SPresentArgsCore *,
    // UINT dirty rects, const RECT *, UINT scroll rects, const DXGI_SCROLL_RECT *,
    // IDXGIResource *). A dxgi.dll whose symbol declares it otherwise is not
    // used: the hook's detour would forward the wrong arguments.
    constexpr const wchar_t *kPresentImplDecorated =
      L"?PresentImpl@CDXGISwapChain@@QEAAJPEBU?$SPresentArgsCore@UCDXGISwapChainWrapper@@@@IPEBUtagRECT@@IPEBUDXGI_SCROLL_RECT@@PEAUIDXGIResource@@@Z";
    constexpr const wchar_t *kPresentImplPrefix = L"?PresentImpl@CDXGISwapChain@@";
    constexpr const char *kSymbolServer = "https://msdl.microsoft.com/download/symbols/";

    std::mutex g_lock;
    std::atomic<bool> g_started {false};
    std::optional<dxgi_symbols_t> g_result;

    struct pe_info_t {
      std::uint32_t timestamp = 0;
      std::uint32_t image_size = 0;
      GUID guid {};
      std::uint32_t age = 0;
      std::string pdb_name;  // as recorded in the binary (dxgi.pdb)
    };

    // The file's PE header fields and its CodeView (RSDS) debug record
    bool read_pe(const std::filesystem::path &path, pe_info_t &out, std::string &why) {
      std::ifstream in(path, std::ios::binary);
      std::vector<char> file((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      if (file.size() < sizeof(IMAGE_DOS_HEADER)) {
        why = "cannot read " + path.string();
        return false;
      }
      const auto *dos = reinterpret_cast<const IMAGE_DOS_HEADER *>(file.data());
      if (dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew <= 0 || static_cast<std::size_t>(dos->e_lfanew) + sizeof(IMAGE_NT_HEADERS64) > file.size()) {
        why = "not a PE file";
        return false;
      }
      const auto *nt = reinterpret_cast<const IMAGE_NT_HEADERS64 *>(file.data() + dos->e_lfanew);
      if (nt->Signature != IMAGE_NT_SIGNATURE || nt->FileHeader.Machine != IMAGE_FILE_MACHINE_AMD64 || nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC) {
        why = "not an x64 PE image";
        return false;
      }
      out.timestamp = nt->FileHeader.TimeDateStamp;
      out.image_size = nt->OptionalHeader.SizeOfImage;

      const auto *sections = IMAGE_FIRST_SECTION(nt);
      auto to_offset = [&](std::uint32_t rva, std::uint32_t size) -> std::size_t {
        for (unsigned i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
          const auto &s = sections[i];
          if (rva >= s.VirtualAddress && rva + size <= s.VirtualAddress + std::max(s.Misc.VirtualSize, s.SizeOfRawData)) {
            return static_cast<std::size_t>(rva - s.VirtualAddress) + s.PointerToRawData;
          }
        }
        return 0;
      };
      const auto &dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DEBUG];
      const auto debug_offset = to_offset(dir.VirtualAddress, dir.Size);
      if (!dir.Size || !debug_offset || debug_offset + dir.Size > file.size()) {
        why = "no debug directory";
        return false;
      }
      for (std::uint32_t at = 0; at + sizeof(IMAGE_DEBUG_DIRECTORY) <= dir.Size; at += sizeof(IMAGE_DEBUG_DIRECTORY)) {
        const auto *entry = reinterpret_cast<const IMAGE_DEBUG_DIRECTORY *>(file.data() + debug_offset + at);
        if (entry->Type != IMAGE_DEBUG_TYPE_CODEVIEW || entry->SizeOfData < 24 || entry->PointerToRawData + entry->SizeOfData > file.size()) {
          continue;
        }
        const char *cv = file.data() + entry->PointerToRawData;
        if (std::memcmp(cv, "RSDS", 4) != 0) {
          continue;
        }
        std::memcpy(&out.guid, cv + 4, sizeof(GUID));
        std::memcpy(&out.age, cv + 20, sizeof(std::uint32_t));
        const std::size_t name_max = entry->SizeOfData - 24;
        out.pdb_name.assign(cv + 24, strnlen(cv + 24, name_max));
        // (only the file name: the record may hold a build machine's path)
        const auto slash = out.pdb_name.find_last_of("\\/");
        if (slash != std::string::npos) {
          out.pdb_name.erase(0, slash + 1);
        }
        if (out.pdb_name.empty()) {
          why = "empty PDB name in the debug record";
          return false;
        }
        return true;
      }
      why = "no CodeView debug record";
      return false;
    }

    // The symbol server's key for a PDB: the GUID's hex digits and the age
    std::string pdb_key(const pe_info_t &pe) {
      char buffer[64];
      std::snprintf(buffer, sizeof(buffer), "%08lX%04X%04X%02X%02X%02X%02X%02X%02X%02X%02X%X",
                    static_cast<unsigned long>(pe.guid.Data1), pe.guid.Data2, pe.guid.Data3,
                    pe.guid.Data4[0], pe.guid.Data4[1], pe.guid.Data4[2], pe.guid.Data4[3],
                    pe.guid.Data4[4], pe.guid.Data4[5], pe.guid.Data4[6], pe.guid.Data4[7], pe.age);
      return buffer;
    }

    bool download(const std::string &url, const std::filesystem::path &file, std::string &why) {
      std::error_code ec;
      std::filesystem::create_directories(file.parent_path(), ec);
      const auto part = file.wstring() + L"." + std::to_wstring(GetCurrentProcessId()) + L".part";
      FILE *fp = _wfopen(part.c_str(), L"wb");
      if (!fp) {
        why = "cannot create " + file.string();
        return false;
      }
      CURL *curl = curl_easy_init();
      if (!curl) {
        std::fclose(fp);
        why = "cannot create a CURL handle";
        return false;
      }
      curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
      curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, fwrite);
      curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
      curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);  // the symbol server redirects to its storage
      curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 8L);
      curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
      curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 20L);
      curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);
      curl_easy_setopt(curl, CURLOPT_USERAGENT, "Vibepollo");
      curl_easy_setopt(curl, CURLOPT_SSLVERSION, CURL_SSLVERSION_TLSv1_2);
      curl_easy_setopt(curl, CURLOPT_SSL_OPTIONS, CURLSSLOPT_NATIVE_CA);
#if LIBCURL_VERSION_NUM >= 0x075500
      curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS_STR, "https");
#else
      curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS, CURLPROTO_HTTPS);
#endif
      const CURLcode result = curl_easy_perform(curl);
      long status = 0;
      curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
      curl_easy_cleanup(curl);
      const bool closed = std::fclose(fp) == 0;
      if (result != CURLE_OK || !closed) {
        DeleteFileW(part.c_str());
        why = "download failed: " + std::string(curl_easy_strerror(result)) + " (HTTP " + std::to_string(status) + ")";
        return false;
      }
      if (!MoveFileExW(part.c_str(), file.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        why = "cannot move the downloaded file into place";
        return false;
      }
      return true;
    }

    bool looks_like_pdb(const std::filesystem::path &file) {
      std::ifstream in(file, std::ios::binary);
      char head[32] = {};
      in.read(head, sizeof(head));
      return in.gcount() == sizeof(head) && std::memcmp(head, "Microsoft C/C++ MSF 7.00\r\n\x1a" "DS\0\0\0", 32) == 0;
    }

    struct enum_ctx_t {
      DWORD64 base = 0;
      std::vector<std::pair<std::wstring, std::uint32_t>> matches;  // decorated name, RVA
    };

    BOOL CALLBACK on_symbol(PSYMBOL_INFOW info, ULONG, PVOID user) {
      auto *ctx = static_cast<enum_ctx_t *>(user);
      std::wstring name(info->Name, info->NameLen);
      // (NameLen has been seen to count a terminator or padding)
      while (!name.empty() && (name.back() == L'\0' || name.back() == L' ')) {
        name.pop_back();
      }
      if (name.rfind(kPresentImplPrefix, 0) == 0 && info->Address >= ctx->base) {
        ctx->matches.emplace_back(name, static_cast<std::uint32_t>(info->Address - ctx->base));
      }
      return TRUE;
    }

    // The RVA of PresentImpl from the PDB in `dir`, through DbgHelp, which
    // loads a PDB only if its GUID and age match the binary's
    bool lookup(const std::filesystem::path &dxgi, const std::filesystem::path &dir, std::uint32_t &rva, std::wstring &name, std::string &why) {
      // DbgHelp keys its sessions by this handle, which it documents need
      // not be a process handle when nothing is invaded: a value of our own
      // keeps this apart from any other DbgHelp user in the host
      HANDLE process = reinterpret_cast<HANDLE>(static_cast<std::uintptr_t>(0x56474350));  // "VGCP"
      const DWORD old_options = SymGetOptions();
      // Decorated names (no UNDNAME): the whole signature is compared
      SymSetOptions(SYMOPT_EXACT_SYMBOLS | SYMOPT_FAIL_CRITICAL_ERRORS | SYMOPT_IGNORE_NT_SYMPATH | SYMOPT_PUBLICS_ONLY);
      if (!SymInitializeW(process, dir.wstring().c_str(), FALSE)) {
        SymSetOptions(old_options);
        why = "SymInitialize failed (" + std::to_string(GetLastError()) + ")";
        return false;
      }
      // (the session ends and the options come back however this returns)
      struct session_t {
        HANDLE process;
        DWORD options;
        DWORD64 base = 0;

        ~session_t() {
          if (base) {
            SymUnloadModule64(process, base);
          }
          SymCleanup(process);
          SymSetOptions(options);
        }
      } session {process, old_options};
      bool ok = false;
      const DWORD64 base = session.base = SymLoadModuleExW(process, nullptr, dxgi.wstring().c_str(), nullptr, 0x10000000, 0, nullptr, 0);
      if (!base) {
        why = "SymLoadModuleEx failed (" + std::to_string(GetLastError()) + ")";
      } else {
        IMAGEHLP_MODULEW64 info {};
        info.SizeOfStruct = sizeof(info);
        enum_ctx_t ctx;
        ctx.base = base;
        if (!SymGetModuleInfoW64(process, base, &info) || info.SymType != SymPdb) {
          why = "DbgHelp did not load the PDB (symbol type " + std::to_string(info.SymType) + ")";
        } else if (!SymEnumSymbolsW(process, base, L"*PresentImpl*", on_symbol, &ctx)) {
          why = "SymEnumSymbols failed (" + std::to_string(GetLastError()) + ")";
        } else if (ctx.matches.size() != 1) {
          why = std::to_string(ctx.matches.size()) + " symbols named CDXGISwapChain::PresentImpl (need exactly one)";
        } else {
          name = ctx.matches[0].first;
          rva = ctx.matches[0].second;
          ok = true;
        }
      }
      return ok;
    }

    std::string narrow(const std::wstring &w) {
      std::string out;
      for (wchar_t c : w) {
        out += c < 128 ? static_cast<char>(c) : '?';
      }
      return out;
    }

    // The whole resolution: cached result, else PDB (downloaded once), else nothing
    void resolve() {
      wchar_t system[MAX_PATH];
      if (!GetSystemDirectoryW(system, MAX_PATH)) {
        BOOST_LOG(warning) << "Game capture: cannot find the system directory; D3D11 frames are copied before Present";
        return;
      }
      const std::filesystem::path dxgi = std::filesystem::path(system) / L"dxgi.dll";
      pe_info_t pe;
      std::string why;
      if (!read_pe(dxgi, pe, why)) {
        BOOST_LOG(warning) << "Game capture: dxgi.dll: " << why << "; D3D11 frames are copied before Present";
        return;
      }
      const auto key = pdb_key(pe);
      const auto dir = platf::appdata() / "dxgi-symbols" / key;
      const auto cache = dir / "presentimpl.txt";

      // A cached result for this exact build
      {
        std::ifstream in(cache);
        std::uint32_t ts = 0, size = 0, rva = 0;
        std::string decorated;
        if (in >> ts >> size >> rva >> decorated && ts == pe.timestamp && size == pe.image_size && rva != 0 && rva < pe.image_size && decorated == narrow(kPresentImplDecorated)) {
          std::lock_guard lg(g_lock);
          g_result = dxgi_symbols_t {pe.timestamp, pe.image_size, rva};
          BOOST_LOG(info) << "Game capture: dxgi.dll (timestamp 0x" << std::hex << pe.timestamp << std::dec << "): PresentImpl at RVA 0x" << std::hex << rva << std::dec << " (cached)";
          return;
        }
      }

      const auto pdb = dir / pe.pdb_name;
      const std::string url = std::string(kSymbolServer) + pe.pdb_name + "/" + key + "/" + pe.pdb_name;
      std::uint32_t rva = 0;
      std::wstring decorated;
      std::error_code ec;
      // A cached PDB that fails the lookup is fetched afresh once: a
      // truncated or wrong file must not disable this for good
      for (int attempt = 0; attempt < 2; ++attempt) {
        if (attempt == 1 || !std::filesystem::exists(pdb, ec) || !looks_like_pdb(pdb)) {
          BOOST_LOG(info) << "Game capture: downloading dxgi.dll's public symbols from " << url;
          if (!download(url, pdb, why)) {
            BOOST_LOG(warning) << "Game capture: " << why << "; D3D11 frames are copied before Present";
            return;
          }
          if (!looks_like_pdb(pdb)) {
            BOOST_LOG(warning) << "Game capture: the downloaded file is not a PDB; D3D11 frames are copied before Present";
            return;
          }
        }
        if (lookup(dxgi, dir, rva, decorated, why)) {
          break;
        }
        if (attempt == 1) {
          BOOST_LOG(warning) << "Game capture: dxgi.dll symbols: " << why << "; D3D11 frames are copied before Present";
          return;
        }
        BOOST_LOG(info) << "Game capture: dxgi.dll symbols: " << why << "; fetching the PDB again";
      }
      if (rva == 0 || rva >= pe.image_size) {
        BOOST_LOG(warning) << "Game capture: PresentImpl's RVA 0x" << std::hex << rva << std::dec << " is outside dxgi.dll; D3D11 frames are copied before Present";
        return;
      }
      if (decorated != kPresentImplDecorated) {
        // The hook's detour forwards a fixed argument list: a changed
        // declaration is not something to guess around
        BOOST_LOG(warning) << "Game capture: this dxgi.dll declares PresentImpl differently (" << narrow(decorated) << "); D3D11 frames are copied before Present until Vibepollo is updated";
        return;
      }
      {
        // (written whole, then moved into place: a reader never sees a part)
        const auto part = cache.wstring() + L"." + std::to_wstring(GetCurrentProcessId()) + L".part";
        std::ofstream out(std::filesystem::path(part), std::ios::trunc);
        out << pe.timestamp << ' ' << pe.image_size << ' ' << rva << ' ' << narrow(decorated) << '\n';
        out.close();
        if (!out || !MoveFileExW(part.c_str(), cache.wstring().c_str(), MOVEFILE_REPLACE_EXISTING)) {
          DeleteFileW(part.c_str());  // (not cached: resolved again next start)
        }
      }
      std::lock_guard lg(g_lock);
      g_result = dxgi_symbols_t {pe.timestamp, pe.image_size, rva};
      BOOST_LOG(info) << "Game capture: dxgi.dll (timestamp 0x" << std::hex << pe.timestamp << std::dec << "): PresentImpl at RVA 0x" << std::hex << rva << std::dec << " (from Microsoft's public symbols)";
    }
  }  // namespace

  void start_dxgi_symbol_resolution() {
    if (g_started.exchange(true)) {
      return;
    }
    try {
      std::thread([] {
        try {
          resolve();
        } catch (const std::exception &e) {
          BOOST_LOG(warning) << "Game capture: resolving dxgi.dll's PresentImpl failed: " << e.what() << "; D3D11 frames are copied before Present";
        } catch (...) {
          BOOST_LOG(warning) << "Game capture: resolving dxgi.dll's PresentImpl failed; D3D11 frames are copied before Present";
        }
      }).detach();
    } catch (const std::exception &e) {
      BOOST_LOG(warning) << "Game capture: cannot start resolving dxgi.dll's PresentImpl: " << e.what() << "; D3D11 frames are copied before Present";
    }
  }

  std::optional<dxgi_symbols_t> dxgi_symbols() {
    std::lock_guard lg(g_lock);
    return g_result;
  }

}  // namespace game_capture
