/**
 * @file tools/vk_layer/vk_layer_api.h
 * @brief The interface between Vibepollo's Vulkan layer (vibepollo_vk_layer.dll,
 *        loaded by the Vulkan loader into every Vulkan application) and the
 *        game hook (vibepollo_game_hook.dll, injected by the host into the
 *        one game it captures).
 *
 * The layer only does what must happen from the start of the application:
 * it adds VK_IMAGE_USAGE_TRANSFER_SRC_BIT to swapchains (a copy needs it,
 * games rarely ask for it, and it cannot be added afterwards), enables the
 * device extensions sharing a texture with D3D11 needs, and records each
 * swapchain's images, format, colour space and window. Nothing is copied
 * and nothing is shared by the layer itself: until a hook registers, its
 * vkQueuePresentKHR is a plain forward.
 *
 * The hook finds the layer with GetModuleHandleW(kVvkModuleName), calls
 * GetProcAddress(kVvkEntryPoint) and registers its callbacks. From then on
 * the layer calls them inside vkQueuePresentKHR (before and after forwarding
 * it) and before a swapchain or device is destroyed.
 */
#pragma once

#include <cstdint>

#include <windows.h>

#ifndef VK_USE_PLATFORM_WIN32_KHR
  #define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <vulkan/vulkan.h>

namespace vvk {

  constexpr std::uint32_t kApiVersion = 2;
  constexpr const wchar_t *kModuleName = L"vibepollo_vk_layer.dll";
  constexpr const char *kEntryPoint = "vibepollo_vk_api";

  /// What the hook needs of a device: the chain below the layer, so its own
  /// work reaches the driver without passing through any layer above.
  struct device_t {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    PFN_vkGetInstanceProcAddr next_gipa;  ///< below the layer
    PFN_vkGetDeviceProcAddr next_gdpa;  ///< below the layer
    std::uint32_t api_version;  ///< the application's (VkApplicationInfo), 0 = 1.0
    bool external_memory_win32;  ///< VK_KHR_external_memory_win32 is enabled on the device
  };

  /// A swapchain as created (the layer's record; valid until the
  /// swapchain_destroyed callback returns)
  struct swapchain_t {
    VkSwapchainKHR swapchain;
    HWND hwnd;  ///< of its VkSurfaceKHR (null: not a window surface)
    VkFormat format;
    VkColorSpaceKHR color_space;
    VkExtent2D extent;
    std::uint32_t image_count;
    const VkImage *images;
    bool transfer_src;  ///< its images allow TRANSFER_SRC (the layer added it, or the application asked)
    /// An ordinary swapchain this copy handles: TRANSFER_SRC, unprotected,
    /// one array layer, no shared-present mode, no device-group split
    bool capturable;
    VkPresentModeKHR present_mode;
  };

  /// A Present as the hook sees it, for one swapchain of the VkPresentInfoKHR
  struct present_t {
    VkQueue queue;
    std::uint32_t queue_family;
    VkQueueFlags queue_flags;  ///< of that family
    const device_t *device;
    const swapchain_t *swapchain;
    std::uint32_t image_index;  ///< into swapchain->images
    std::uint32_t swapchain_slot;  ///< its position in the VkPresentInfoKHR
    const VkPresentInfoKHR *info;  ///< the application's
  };

  /// Callbacks from the layer into the hook. All run on the calling
  /// application thread; the layer holds no lock of its own while they run.
  struct callbacks_t {
    /// Inside vkQueuePresentKHR, before forwarding. May replace the wait
    /// semaphores of the Present: returns true with `*wait_count` and
    /// `*waits` set (an array the hook owns, valid until post_present) to
    /// have the layer forward those instead.
    bool (*pre_present)(void *ctx, const present_t &present, std::uint32_t *wait_count, const VkSemaphore **waits);
    /// After forwarding, with its result (always called if pre_present was)
    void (*post_present)(void *ctx, const present_t &present, VkResult result);
    /// Before the swapchain is destroyed: the hook finishes with its images
    void (*swapchain_destroyed)(void *ctx, const device_t &device, VkSwapchainKHR swapchain);
    /// Before the device is destroyed: the hook releases everything it made on it
    void (*device_destroyed)(void *ctx, const device_t &device);
  };

  struct api_t {
    std::uint32_t version;  ///< kApiVersion
    /// Once per process; false if something registered already
    bool (*register_callbacks)(const callbacks_t *callbacks, void *ctx);
  };

  using entry_point_fn = const api_t *(*) (std::uint32_t version);

}  // namespace vvk
