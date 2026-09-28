/**
 * @file tools/vk_layer/vk_layer.cpp
 * @brief Vibepollo's Vulkan implicit layer (VK_LAYER_VIBEPOLLO_capture): the
 *        part of in-game capture that must be there from the application's
 *        start. See vk_layer_api.h for the division of work with the hook.
 *
 * It runs in every Vulkan process, so it does as little as it can and
 * forwards everything else untouched:
 *  - vkCreateDevice: enables VK_KHR_external_memory_win32 when the device
 *    supports it (a copy into a texture shared with D3D11 needs it);
 *  - vkCreateSwapchainKHR: adds TRANSFER_SRC to the image usage when the
 *    surface supports it (retrying without on failure), and records the
 *    swapchain: images, format, colour space, extent, window;
 *  - vkCreateImageView of such an image: gives the view the application's
 *    own usage (VkImageViewUsageCreateInfo), so a view of it still matches
 *    what an imageless framebuffer declared for it;
 *  - vkQueuePresentKHR: calls the hook when one registered, else forwards.
 *
 * Where it sits in the chain matters: below (nearer the driver than) an
 * overlay's layer (Steam's), the overlay has drawn before our Present runs.
 * The loader orders implicit layers as they are registered, so the host
 * keeps our registration last (game_capture/vk_layer_registration).
 */
#include "vk_layer_api.h"

#include <vulkan/vk_layer.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <new>
#include <unordered_map>
#include <vector>

namespace {

  // Dispatchable handles of one instance (instance, physical devices) or one
  // device (device, queues, command buffers) share the loader's dispatch
  // table pointer: it is the key of their records
  template<class T>
  void *key_of(T handle) {
    return *reinterpret_cast<void **>(handle);
  }

  struct lock_t {
    SRWLOCK lock = SRWLOCK_INIT;
  };

  lock_t g_lock;

  struct shared_t {
    explicit shared_t(lock_t &l):
        l(l) {
      AcquireSRWLockShared(&l.lock);
    }

    ~shared_t() {
      ReleaseSRWLockShared(&l.lock);
    }

    lock_t &l;
  };

  struct exclusive_t {
    explicit exclusive_t(lock_t &l):
        l(l) {
      AcquireSRWLockExclusive(&l.lock);
    }

    ~exclusive_t() {
      ReleaseSRWLockExclusive(&l.lock);
    }

    lock_t &l;
  };

  // ---- records ---------------------------------------------------------------

  struct instance_t {
    VkInstance instance = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr next_gipa = nullptr;
    std::uint32_t api_version = VK_API_VERSION_1_0;
    PFN_vkDestroyInstance DestroyInstance = nullptr;
    PFN_vkCreateWin32SurfaceKHR CreateWin32SurfaceKHR = nullptr;
    PFN_vkDestroySurfaceKHR DestroySurfaceKHR = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR GetPhysicalDeviceSurfaceCapabilitiesKHR = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties = nullptr;
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties = nullptr;
    std::unordered_map<VkSurfaceKHR, HWND> surfaces;
  };

  struct swapchain_rec_t {
    vvk::swapchain_t pub {};
    std::vector<VkImage> images;
    VkImageUsageFlags app_usage = 0;  ///< what the application asked for
  };

  struct device_rec_t {
    vvk::device_t pub {};
    instance_t *instance = nullptr;
    std::vector<VkQueueFamilyProperties> families;
    bool view_usage_info = false;  ///< VkImageViewUsageCreateInfo may be chained (1.1 or VK_KHR_maintenance2)
    PFN_vkDestroyDevice DestroyDevice = nullptr;
    PFN_vkGetDeviceQueue GetDeviceQueue = nullptr;
    PFN_vkGetDeviceQueue2 GetDeviceQueue2 = nullptr;
    PFN_vkCreateSwapchainKHR CreateSwapchainKHR = nullptr;
    PFN_vkDestroySwapchainKHR DestroySwapchainKHR = nullptr;
    PFN_vkGetSwapchainImagesKHR GetSwapchainImagesKHR = nullptr;
    PFN_vkQueuePresentKHR QueuePresentKHR = nullptr;
    PFN_vkCreateImageView CreateImageView = nullptr;
    std::unordered_map<VkQueue, std::uint32_t> queue_families;
    std::unordered_map<VkSwapchainKHR, std::unique_ptr<swapchain_rec_t>> swapchains;
    std::unordered_map<VkImage, swapchain_rec_t *> swapchain_images;  ///< images whose usage we widened
  };

  std::unordered_map<void *, std::unique_ptr<instance_t>> g_instances;
  std::unordered_map<void *, std::unique_ptr<device_rec_t>> g_devices;

  // The registered hook (set once, never cleared: the hook stays loaded)
  const vvk::callbacks_t *g_callbacks = nullptr;
  void *g_callbacks_ctx = nullptr;

  instance_t *find_instance(void *key) {
    shared_t lg(g_lock);
    const auto it = g_instances.find(key);
    return it == g_instances.end() ? nullptr : it->second.get();
  }

  device_rec_t *find_device(void *key) {
    shared_t lg(g_lock);
    const auto it = g_devices.find(key);
    return it == g_devices.end() ? nullptr : it->second.get();
  }

  bool has_extension(const VkDeviceCreateInfo *info, const char *name) {
    for (std::uint32_t i = 0; i < info->enabledExtensionCount; ++i) {
      if (std::strcmp(info->ppEnabledExtensionNames[i], name) == 0) {
        return true;
      }
    }
    return false;
  }

  // ---- instance --------------------------------------------------------------

  VKAPI_ATTR VkResult VKAPI_CALL layer_CreateInstance(const VkInstanceCreateInfo *info, const VkAllocationCallbacks *allocator, VkInstance *out) {
    auto *link = const_cast<VkLayerInstanceCreateInfo *>(static_cast<const VkLayerInstanceCreateInfo *>(info->pNext));
    while (link && !(link->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO && link->function == VK_LAYER_LINK_INFO)) {
      link = const_cast<VkLayerInstanceCreateInfo *>(static_cast<const VkLayerInstanceCreateInfo *>(link->pNext));
    }
    if (!link) {
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    const PFN_vkGetInstanceProcAddr next_gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;  // (the next layer's turn)
    const auto create = reinterpret_cast<PFN_vkCreateInstance>(next_gipa(VK_NULL_HANDLE, "vkCreateInstance"));
    if (!create) {
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    const VkResult result = create(info, allocator, out);
    if (result != VK_SUCCESS) {
      return result;
    }
    try {
      auto rec = std::make_unique<instance_t>();
      rec->instance = *out;
      rec->next_gipa = next_gipa;
      rec->api_version = info->pApplicationInfo && info->pApplicationInfo->apiVersion ? info->pApplicationInfo->apiVersion : VK_API_VERSION_1_0;
#define VVK_INSTANCE_FN(name) rec->name = reinterpret_cast<PFN_vk##name>(next_gipa(*out, "vk" #name))
      VVK_INSTANCE_FN(DestroyInstance);
      VVK_INSTANCE_FN(CreateWin32SurfaceKHR);
      VVK_INSTANCE_FN(DestroySurfaceKHR);
      VVK_INSTANCE_FN(GetPhysicalDeviceSurfaceCapabilitiesKHR);
      VVK_INSTANCE_FN(EnumerateDeviceExtensionProperties);
      VVK_INSTANCE_FN(GetPhysicalDeviceQueueFamilyProperties);
      VVK_INSTANCE_FN(GetPhysicalDeviceProperties);
#undef VVK_INSTANCE_FN
      exclusive_t lg(g_lock);
      g_instances[key_of(*out)] = std::move(rec);
    } catch (...) {
      // (no record: this instance is simply not captured)
    }
    return VK_SUCCESS;
  }

  VKAPI_ATTR void VKAPI_CALL layer_DestroyInstance(VkInstance instance, const VkAllocationCallbacks *allocator) {
    if (!instance) {
      return;
    }
    std::unique_ptr<instance_t> rec;
    {
      exclusive_t lg(g_lock);
      const auto it = g_instances.find(key_of(instance));
      if (it != g_instances.end()) {
        rec = std::move(it->second);
        g_instances.erase(it);
      }
    }
    if (rec && rec->DestroyInstance) {
      rec->DestroyInstance(instance, allocator);
    }
  }

  VKAPI_ATTR VkResult VKAPI_CALL layer_CreateWin32SurfaceKHR(VkInstance instance, const VkWin32SurfaceCreateInfoKHR *info, const VkAllocationCallbacks *allocator, VkSurfaceKHR *out) {
    instance_t *rec = find_instance(key_of(instance));
    if (!rec || !rec->CreateWin32SurfaceKHR) {
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    const VkResult result = rec->CreateWin32SurfaceKHR(instance, info, allocator, out);
    if (result == VK_SUCCESS) {
      try {
        exclusive_t lg(g_lock);
        rec->surfaces[*out] = info->hwnd;
      } catch (...) {
      }
    }
    return result;
  }

  VKAPI_ATTR void VKAPI_CALL layer_DestroySurfaceKHR(VkInstance instance, VkSurfaceKHR surface, const VkAllocationCallbacks *allocator) {
    instance_t *rec = find_instance(key_of(instance));
    if (!rec) {
      return;
    }
    {
      exclusive_t lg(g_lock);
      rec->surfaces.erase(surface);
    }
    if (rec->DestroySurfaceKHR) {
      rec->DestroySurfaceKHR(instance, surface, allocator);
    }
  }

  // ---- device ----------------------------------------------------------------

  VKAPI_ATTR VkResult VKAPI_CALL layer_CreateDevice(VkPhysicalDevice physical, const VkDeviceCreateInfo *info, const VkAllocationCallbacks *allocator, VkDevice *out) {
    auto *link = const_cast<VkLayerDeviceCreateInfo *>(static_cast<const VkLayerDeviceCreateInfo *>(info->pNext));
    while (link && !(link->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO && link->function == VK_LAYER_LINK_INFO)) {
      link = const_cast<VkLayerDeviceCreateInfo *>(static_cast<const VkLayerDeviceCreateInfo *>(link->pNext));
    }
    if (!link) {
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    instance_t *inst = find_instance(key_of(physical));
    const PFN_vkGetInstanceProcAddr next_gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    const PFN_vkGetDeviceProcAddr next_gdpa = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    const auto create = reinterpret_cast<PFN_vkCreateDevice>(next_gipa(inst ? inst->instance : VK_NULL_HANDLE, "vkCreateDevice"));
    if (!create) {
      return VK_ERROR_INITIALIZATION_FAILED;
    }

    // VK_KHR_external_memory_win32, if the device has it and the
    // application did not enable it (its dependency, external memory, is
    // core in 1.1; before that the application must have enabled it)
    std::vector<const char *> extensions;
    bool added = false;
    bool external_memory_win32 = has_extension(info, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    std::uint32_t physical_api = VK_API_VERSION_1_0;
    if (inst && inst->GetPhysicalDeviceProperties) {
      VkPhysicalDeviceProperties props {};
      inst->GetPhysicalDeviceProperties(physical, &props);
      physical_api = props.apiVersion;
    }
    const std::uint32_t api = inst ? std::min(inst->api_version, physical_api) : VK_API_VERSION_1_0;
    try {
      if (!external_memory_win32 && inst && inst->EnumerateDeviceExtensionProperties &&
          (api >= VK_API_VERSION_1_1 || has_extension(info, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME))) {
        std::uint32_t count = 0;
        inst->EnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> props(count);
        if (count && inst->EnumerateDeviceExtensionProperties(physical, nullptr, &count, props.data()) >= VK_SUCCESS) {
          for (const auto &p : props) {
            if (std::strcmp(p.extensionName, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME) == 0) {
              extensions.assign(info->ppEnabledExtensionNames, info->ppEnabledExtensionNames + info->enabledExtensionCount);
              extensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
              added = true;
              break;
            }
          }
        }
      }
    } catch (...) {
      added = false;
    }

    VkResult result = VK_ERROR_INITIALIZATION_FAILED;
    if (added) {
      VkDeviceCreateInfo widened = *info;
      widened.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
      widened.ppEnabledExtensionNames = extensions.data();
      result = create(physical, &widened, allocator, out);
      if (result == VK_SUCCESS) {
        external_memory_win32 = true;
      }
    }
    if (result != VK_SUCCESS) {
      result = create(physical, info, allocator, out);  // (as the application asked)
      if (result != VK_SUCCESS) {
        return result;
      }
    }

    try {
      auto rec = std::make_unique<device_rec_t>();
      rec->instance = inst;
      rec->pub.instance = inst ? inst->instance : VK_NULL_HANDLE;
      rec->pub.physical_device = physical;
      rec->pub.device = *out;
      rec->pub.next_gipa = next_gipa;
      rec->pub.next_gdpa = next_gdpa;
      rec->pub.api_version = api;
      rec->pub.external_memory_win32 = external_memory_win32;
      rec->view_usage_info = api >= VK_API_VERSION_1_1 || has_extension(info, VK_KHR_MAINTENANCE_2_EXTENSION_NAME);
      if (inst && inst->GetPhysicalDeviceQueueFamilyProperties) {
        std::uint32_t count = 0;
        inst->GetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
        rec->families.resize(count);
        inst->GetPhysicalDeviceQueueFamilyProperties(physical, &count, rec->families.data());
      }
#define VVK_DEVICE_FN(name) rec->name = reinterpret_cast<PFN_vk##name>(next_gdpa(*out, "vk" #name))
      VVK_DEVICE_FN(DestroyDevice);
      VVK_DEVICE_FN(GetDeviceQueue);
      VVK_DEVICE_FN(GetDeviceQueue2);
      VVK_DEVICE_FN(CreateSwapchainKHR);
      VVK_DEVICE_FN(DestroySwapchainKHR);
      VVK_DEVICE_FN(GetSwapchainImagesKHR);
      VVK_DEVICE_FN(QueuePresentKHR);
      VVK_DEVICE_FN(CreateImageView);
#undef VVK_DEVICE_FN
      exclusive_t lg(g_lock);
      g_devices[key_of(*out)] = std::move(rec);
    } catch (...) {
    }
    return VK_SUCCESS;
  }

  VKAPI_ATTR void VKAPI_CALL layer_DestroyDevice(VkDevice device, const VkAllocationCallbacks *allocator) {
    if (!device) {
      return;
    }
    device_rec_t *rec = find_device(key_of(device));
    if (rec && g_callbacks && g_callbacks->device_destroyed) {
      g_callbacks->device_destroyed(g_callbacks_ctx, rec->pub);
    }
    std::unique_ptr<device_rec_t> owned;
    {
      exclusive_t lg(g_lock);
      const auto it = g_devices.find(key_of(device));
      if (it != g_devices.end()) {
        owned = std::move(it->second);
        g_devices.erase(it);
      }
    }
    if (owned && owned->DestroyDevice) {
      owned->DestroyDevice(device, allocator);
    }
  }

  void note_queue(device_rec_t *rec, VkQueue queue, std::uint32_t family) {
    try {
      exclusive_t lg(g_lock);
      rec->queue_families[queue] = family;
    } catch (...) {
    }
  }

  VKAPI_ATTR void VKAPI_CALL layer_GetDeviceQueue(VkDevice device, std::uint32_t family, std::uint32_t index, VkQueue *out) {
    device_rec_t *rec = find_device(key_of(device));
    if (!rec || !rec->GetDeviceQueue) {
      *out = VK_NULL_HANDLE;
      return;
    }
    rec->GetDeviceQueue(device, family, index, out);
    if (*out) {
      note_queue(rec, *out, family);
    }
  }

  VKAPI_ATTR void VKAPI_CALL layer_GetDeviceQueue2(VkDevice device, const VkDeviceQueueInfo2 *info, VkQueue *out) {
    device_rec_t *rec = find_device(key_of(device));
    if (!rec || !rec->GetDeviceQueue2) {
      *out = VK_NULL_HANDLE;
      return;
    }
    rec->GetDeviceQueue2(device, info, out);
    if (*out) {
      note_queue(rec, *out, info->queueFamilyIndex);
    }
  }

  // ---- swapchain -------------------------------------------------------------

  VKAPI_ATTR VkResult VKAPI_CALL layer_CreateSwapchainKHR(VkDevice device, const VkSwapchainCreateInfoKHR *info, const VkAllocationCallbacks *allocator, VkSwapchainKHR *out) {
    device_rec_t *rec = find_device(key_of(device));
    if (!rec || !rec->CreateSwapchainKHR) {
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    bool transfer_src = (info->imageUsage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;
    bool widened = false;
    VkResult result = VK_ERROR_INITIALIZATION_FAILED;
    if (!transfer_src && rec->instance && rec->instance->GetPhysicalDeviceSurfaceCapabilitiesKHR) {
      VkSurfaceCapabilitiesKHR caps {};
      if (rec->instance->GetPhysicalDeviceSurfaceCapabilitiesKHR(rec->pub.physical_device, info->surface, &caps) == VK_SUCCESS &&
          (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
        VkSwapchainCreateInfoKHR wide = *info;
        wide.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        result = rec->CreateSwapchainKHR(device, &wide, allocator, out);
        widened = transfer_src = result == VK_SUCCESS;
      }
    }
    if (result != VK_SUCCESS) {
      result = rec->CreateSwapchainKHR(device, info, allocator, out);  // (as the application asked)
      if (result != VK_SUCCESS) {
        return result;
      }
    }

    try {
      auto sc = std::make_unique<swapchain_rec_t>();
      std::uint32_t count = 0;
      if (rec->GetSwapchainImagesKHR && rec->GetSwapchainImagesKHR(device, *out, &count, nullptr) >= VK_SUCCESS && count) {
        sc->images.resize(count);
        if (rec->GetSwapchainImagesKHR(device, *out, &count, sc->images.data()) < VK_SUCCESS) {
          sc->images.clear();
        }
        sc->images.resize(std::min<std::size_t>(count, sc->images.size()));
      }
      sc->app_usage = info->imageUsage;
      sc->pub.swapchain = *out;
      sc->pub.format = info->imageFormat;
      sc->pub.color_space = info->imageColorSpace;
      sc->pub.extent = info->imageExtent;
      sc->pub.image_count = static_cast<std::uint32_t>(sc->images.size());
      sc->pub.images = sc->images.data();
      sc->pub.transfer_src = transfer_src && !sc->images.empty();
      exclusive_t lg(g_lock);
      if (rec->instance) {
        const auto it = rec->instance->surfaces.find(info->surface);
        sc->pub.hwnd = it == rec->instance->surfaces.end() ? nullptr : it->second;
      }
      if (widened) {
        for (VkImage image : sc->images) {
          rec->swapchain_images[image] = sc.get();
        }
      }
      rec->swapchains[*out] = std::move(sc);
    } catch (...) {
    }
    return VK_SUCCESS;
  }

  VKAPI_ATTR void VKAPI_CALL layer_DestroySwapchainKHR(VkDevice device, VkSwapchainKHR swapchain, const VkAllocationCallbacks *allocator) {
    device_rec_t *rec = find_device(key_of(device));
    if (!rec) {
      return;
    }
    if (swapchain && g_callbacks && g_callbacks->swapchain_destroyed) {
      g_callbacks->swapchain_destroyed(g_callbacks_ctx, rec->pub, swapchain);
    }
    std::unique_ptr<swapchain_rec_t> owned;
    {
      exclusive_t lg(g_lock);
      const auto it = rec->swapchains.find(swapchain);
      if (it != rec->swapchains.end()) {
        owned = std::move(it->second);
        rec->swapchains.erase(it);
        for (VkImage image : owned->images) {
          rec->swapchain_images.erase(image);
        }
      }
    }
    if (rec->DestroySwapchainKHR) {
      rec->DestroySwapchainKHR(device, swapchain, allocator);
    }
  }

  // A view of a swapchain image whose usage we widened gets the
  // application's usage back: an imageless framebuffer declared that usage
  // for it, and must see it
  VKAPI_ATTR VkResult VKAPI_CALL layer_CreateImageView(VkDevice device, const VkImageViewCreateInfo *info, const VkAllocationCallbacks *allocator, VkImageView *out) {
    device_rec_t *rec = find_device(key_of(device));
    if (!rec || !rec->CreateImageView) {
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    VkImageUsageFlags app_usage = 0;
    bool narrow = false;
    if (rec->view_usage_info) {
      shared_t lg(g_lock);
      const auto it = rec->swapchain_images.find(info->image);
      if (it != rec->swapchain_images.end()) {
        app_usage = it->second->app_usage;
        narrow = true;
      }
    }
    if (narrow) {
      for (auto *p = static_cast<const VkBaseInStructure *>(info->pNext); p; p = p->pNext) {
        if (p->sType == VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO) {
          narrow = false;  // (the application restricts the view itself)
          break;
        }
      }
    }
    if (!narrow) {
      return rec->CreateImageView(device, info, allocator, out);
    }
    VkImageViewUsageCreateInfo usage {};
    usage.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO;
    usage.pNext = info->pNext;
    usage.usage = app_usage;
    VkImageViewCreateInfo narrowed = *info;
    narrowed.pNext = &usage;
    return rec->CreateImageView(device, &narrowed, allocator, out);
  }

  // ---- present ---------------------------------------------------------------

  VKAPI_ATTR VkResult VKAPI_CALL layer_QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR *info) {
    device_rec_t *rec = find_device(key_of(queue));
    if (!rec || !rec->QueuePresentKHR) {
      return VK_ERROR_DEVICE_LOST;
    }
    const vvk::callbacks_t *callbacks = g_callbacks;
    if (!callbacks || !callbacks->pre_present || !info || info->swapchainCount == 0) {
      return rec->QueuePresentKHR(queue, info);
    }

    // What the hook needs, looked up once (records outlive this call: the
    // application may not destroy what it is presenting)
    constexpr std::uint32_t kMax = 8;
    vvk::present_t presents[kMax] {};
    std::uint32_t n = 0;
    {
      shared_t lg(g_lock);
      const auto qf = rec->queue_families.find(queue);
      const std::uint32_t family = qf == rec->queue_families.end() ? UINT32_MAX : qf->second;
      const VkQueueFlags flags = family < rec->families.size() ? rec->families[family].queueFlags : 0;
      for (std::uint32_t i = 0; i < info->swapchainCount && n < kMax; ++i) {
        const auto it = rec->swapchains.find(info->pSwapchains[i]);
        if (it == rec->swapchains.end()) {
          continue;
        }
        auto &p = presents[n++];
        p.queue = queue;
        p.queue_family = family;
        p.queue_flags = flags;
        p.device = &rec->pub;
        p.swapchain = &it->second->pub;
        p.image_index = info->pImageIndices[i];
        p.swapchain_slot = i;
        p.info = info;
      }
    }

    VkPresentInfoKHR forwarded = *info;
    bool replaced = false;
    for (std::uint32_t i = 0; i < n; ++i) {
      std::uint32_t wait_count = 0;
      const VkSemaphore *waits = nullptr;
      if (callbacks->pre_present(g_callbacks_ctx, presents[i], &wait_count, &waits) && !replaced) {
        forwarded.waitSemaphoreCount = wait_count;
        forwarded.pWaitSemaphores = waits;
        replaced = true;
      }
    }
    const VkResult result = rec->QueuePresentKHR(queue, &forwarded);
    if (callbacks->post_present) {
      for (std::uint32_t i = 0; i < n; ++i) {
        callbacks->post_present(g_callbacks_ctx, presents[i], result);
      }
    }
    return result;
  }

  // ---- dispatch --------------------------------------------------------------

  PFN_vkVoidFunction device_intercept(const char *name) {
    struct entry_t {
      const char *name;
      PFN_vkVoidFunction fn;
    };

    static const entry_t table[] = {
      {"vkDestroyDevice", reinterpret_cast<PFN_vkVoidFunction>(&layer_DestroyDevice)},
      {"vkGetDeviceQueue", reinterpret_cast<PFN_vkVoidFunction>(&layer_GetDeviceQueue)},
      {"vkGetDeviceQueue2", reinterpret_cast<PFN_vkVoidFunction>(&layer_GetDeviceQueue2)},
      {"vkCreateSwapchainKHR", reinterpret_cast<PFN_vkVoidFunction>(&layer_CreateSwapchainKHR)},
      {"vkDestroySwapchainKHR", reinterpret_cast<PFN_vkVoidFunction>(&layer_DestroySwapchainKHR)},
      {"vkQueuePresentKHR", reinterpret_cast<PFN_vkVoidFunction>(&layer_QueuePresentKHR)},
      {"vkCreateImageView", reinterpret_cast<PFN_vkVoidFunction>(&layer_CreateImageView)},
    };
    for (const auto &e : table) {
      if (std::strcmp(name, e.name) == 0) {
        return e.fn;
      }
    }
    return nullptr;
  }

  VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL layer_GetDeviceProcAddr(VkDevice device, const char *name);

  PFN_vkVoidFunction instance_intercept(const char *name) {
    struct entry_t {
      const char *name;
      PFN_vkVoidFunction fn;
    };

    static const entry_t table[] = {
      {"vkCreateInstance", reinterpret_cast<PFN_vkVoidFunction>(&layer_CreateInstance)},
      {"vkDestroyInstance", reinterpret_cast<PFN_vkVoidFunction>(&layer_DestroyInstance)},
      {"vkCreateDevice", reinterpret_cast<PFN_vkVoidFunction>(&layer_CreateDevice)},
      {"vkCreateWin32SurfaceKHR", reinterpret_cast<PFN_vkVoidFunction>(&layer_CreateWin32SurfaceKHR)},
      {"vkDestroySurfaceKHR", reinterpret_cast<PFN_vkVoidFunction>(&layer_DestroySurfaceKHR)},
      {"vkGetDeviceProcAddr", reinterpret_cast<PFN_vkVoidFunction>(&layer_GetDeviceProcAddr)},
    };
    for (const auto &e : table) {
      if (std::strcmp(name, e.name) == 0) {
        return e.fn;
      }
    }
    return nullptr;
  }

  VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL layer_GetInstanceProcAddr(VkInstance instance, const char *name);

  VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL layer_GetDeviceProcAddr(VkDevice device, const char *name) {
    if (!name) {
      return nullptr;
    }
    if (std::strcmp(name, "vkGetDeviceProcAddr") == 0) {
      return reinterpret_cast<PFN_vkVoidFunction>(&layer_GetDeviceProcAddr);
    }
    device_rec_t *rec = device ? find_device(key_of(device)) : nullptr;
    if (!rec) {
      return nullptr;
    }
    if (PFN_vkVoidFunction fn = device_intercept(name)) {
      // (only what the device below has: returning ours for a function the
      // driver lacks would claim support it does not have)
      if (rec->pub.next_gdpa(device, name)) {
        return fn;
      }
      return nullptr;
    }
    return rec->pub.next_gdpa(device, name);
  }

  VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL layer_GetInstanceProcAddr(VkInstance instance, const char *name) {
    if (!name) {
      return nullptr;
    }
    if (std::strcmp(name, "vkGetInstanceProcAddr") == 0) {
      return reinterpret_cast<PFN_vkVoidFunction>(&layer_GetInstanceProcAddr);
    }
    if (PFN_vkVoidFunction fn = instance_intercept(name)) {
      if (std::strcmp(name, "vkCreateInstance") == 0 || !instance) {
        return fn;
      }
      instance_t *rec = find_instance(key_of(instance));
      return rec && rec->next_gipa(instance, name) ? fn : nullptr;
    }
    if (!instance) {
      return nullptr;
    }
    instance_t *rec = find_instance(key_of(instance));
    if (!rec) {
      return nullptr;
    }
    if (PFN_vkVoidFunction fn = device_intercept(name)) {
      return rec->next_gipa(instance, name) ? fn : nullptr;
    }
    return rec->next_gipa(instance, name);
  }

  // ---- hook registration -----------------------------------------------------

  bool register_callbacks(const vvk::callbacks_t *callbacks, void *ctx) {
    exclusive_t lg(g_lock);
    if (g_callbacks || !callbacks) {
      return false;
    }
    g_callbacks_ctx = ctx;
    g_callbacks = callbacks;
    return true;
  }

  const vvk::api_t g_api {vvk::kApiVersion, &register_callbacks};

}  // namespace

extern "C" {

  __declspec(dllexport) VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *version) {
    if (!version || version->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT) {
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (version->loaderLayerInterfaceVersion >= 2) {
      version->pfnGetInstanceProcAddr = &layer_GetInstanceProcAddr;
      version->pfnGetDeviceProcAddr = &layer_GetDeviceProcAddr;
      version->pfnGetPhysicalDeviceProcAddr = nullptr;
    }
    if (version->loaderLayerInterfaceVersion > 2) {
      version->loaderLayerInterfaceVersion = 2;
    }
    return VK_SUCCESS;
  }

  __declspec(dllexport) VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *name) {
    return layer_GetInstanceProcAddr(instance, name);
  }

  __declspec(dllexport) VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *name) {
    return layer_GetDeviceProcAddr(device, name);
  }

  __declspec(dllexport) const vvk::api_t *vibepollo_vk_api(std::uint32_t version) {
    return version == vvk::kApiVersion ? &g_api : nullptr;
  }
}
