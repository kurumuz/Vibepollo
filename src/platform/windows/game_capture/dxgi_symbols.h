/**
 * @file src/platform/windows/game_capture/dxgi_symbols.h
 * @brief Locates dxgi!CDXGISwapChain::PresentImpl in the system dxgi.dll from
 *        Microsoft's public symbols, for the game hook's D3D11 copy point.
 *
 * DXGI's public Present and Present1 both call PresentImpl after whatever
 * hooks sit on them (Steam's overlay and the like) have run, and before any
 * of DXGI's presentation work; it is not exported, and its address changes
 * with every dxgi.dll build. So the host downloads that build's public PDB
 * from the Microsoft symbol server once (cached per build under appdata),
 * looks the function up with DbgHelp, and hands the RVA to the hook through
 * the shared block. The hook verifies it against the code before using it.
 */
#pragma once

#include <cstdint>
#include <optional>

namespace game_capture {

  struct dxgi_symbols_t {
    std::uint32_t timestamp;  ///< PE TimeDateStamp of the dxgi.dll resolved
    std::uint32_t image_size;  ///< its SizeOfImage
    std::uint32_t present_impl_rva;  ///< CDXGISwapChain::PresentImpl
  };

  /**
   * @brief Starts resolving in the background (once per process; later calls
   *        do nothing).
   */
  void start_dxgi_symbol_resolution();

  /**
   * @brief The result, once resolution succeeded; empty while it runs or
   *        after it failed (the reason is logged).
   */
  std::optional<dxgi_symbols_t> dxgi_symbols();

}  // namespace game_capture
