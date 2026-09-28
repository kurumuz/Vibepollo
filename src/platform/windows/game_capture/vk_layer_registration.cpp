/**
 * @file src/platform/windows/game_capture/vk_layer_registration.cpp
 * @brief See vk_layer_registration.h.
 */
#include "vk_layer_registration.h"

#include "src/logging.h"

#include <windows.h>

#include <atomic>
#include <filesystem>
#include <string>
#include <vector>

namespace game_capture {
  namespace {
    constexpr const wchar_t *kKey = L"SOFTWARE\\Khronos\\Vulkan\\ImplicitLayers";
    constexpr const wchar_t *kManifest = L"vibepollo_vk_layer.json";

    std::atomic<bool> g_done {false};

    std::string narrow(const std::wstring &w) {
      std::string out;
      for (wchar_t c : w) {
        out += c < 128 ? static_cast<char>(c) : '?';
      }
      return out;
    }

    // The value names under the key, in the order the registry lists them
    bool list(HKEY key, std::vector<std::wstring> &names) {
      names.clear();
      for (DWORD i = 0;; ++i) {
        wchar_t name[1024];
        DWORD length = static_cast<DWORD>(std::size(name));
        const LSTATUS status = RegEnumValueW(key, i, name, &length, nullptr, nullptr, nullptr, nullptr);
        if (status == ERROR_NO_MORE_ITEMS) {
          return true;
        }
        if (status != ERROR_SUCCESS) {
          return false;
        }
        names.emplace_back(name, length);
      }
    }
  }  // namespace

  void ensure_vk_layer_registered() {
    if (g_done.exchange(true)) {
      return;
    }
    wchar_t exe[MAX_PATH];
    if (!GetModuleFileNameW(nullptr, exe, MAX_PATH)) {
      return;
    }
    const auto manifest = std::filesystem::path(exe).parent_path() / L"tools" / kManifest;
    std::error_code ec;
    if (!std::filesystem::exists(manifest, ec)) {
      BOOST_LOG(info) << "Game capture: no Vulkan layer manifest at " << narrow(manifest.wstring()) << "; Vulkan games are not captured";
      return;
    }
    const std::wstring ours = manifest.wstring();

    HKEY key = nullptr;
    const LSTATUS opened = RegCreateKeyExW(HKEY_LOCAL_MACHINE, kKey, 0, nullptr, 0, KEY_READ | KEY_SET_VALUE | KEY_WOW64_64KEY, nullptr, &key, nullptr);
    if (opened != ERROR_SUCCESS) {
      BOOST_LOG(warning) << "Game capture: cannot open HKLM\\SOFTWARE\\Khronos\\Vulkan\\ImplicitLayers (" << opened << "); the Vulkan layer is not registered";
      return;
    }
    struct close_t {
      HKEY key;

      ~close_t() {
        RegCloseKey(key);
      }
    } close {key};

    std::vector<std::wstring> names;
    if (!list(key, names)) {
      BOOST_LOG(warning) << "Game capture: cannot list the Vulkan implicit layers; the Vulkan layer is not registered";
      return;
    }
    // Stale registrations of our layer (an older install path) go
    bool present = false;
    for (const auto &name : names) {
      const bool is_ours = _wcsicmp(name.c_str(), ours.c_str()) == 0;
      if (is_ours) {
        present = true;
      } else if (_wcsicmp(std::filesystem::path(name).filename().c_str(), kManifest) == 0) {
        RegDeleteValueW(key, name.c_str());
        BOOST_LOG(info) << "Game capture: removed a stale Vulkan layer registration (" << narrow(name) << ')';
      }
    }
    list(key, names);
    if (present && !names.empty() && _wcsicmp(names.back().c_str(), ours.c_str()) == 0) {
      BOOST_LOG(info) << "Game capture: the Vulkan layer is registered last of " << names.size() << " implicit layers";
      return;
    }
    // (Re)created, so it lists last
    if (present) {
      RegDeleteValueW(key, ours.c_str());
    }
    const DWORD zero = 0;
    const LSTATUS set = RegSetValueExW(key, ours.c_str(), 0, REG_DWORD, reinterpret_cast<const BYTE *>(&zero), sizeof(zero));
    if (set != ERROR_SUCCESS) {
      BOOST_LOG(warning) << "Game capture: cannot register the Vulkan layer (" << set << ")";
      return;
    }
    list(key, names);
    const bool last = !names.empty() && _wcsicmp(names.back().c_str(), ours.c_str()) == 0;
    BOOST_LOG(info) << "Game capture: Vulkan layer " << (present ? "moved to the end" : "registered") << " (" << narrow(ours) << ')'
                    << (last ? "" : "; but the registry does not list it last, so an overlay layer may draw after our copy");
  }

}  // namespace game_capture
