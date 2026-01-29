//! Vulkan context for video operations.
//!
//! Manages Vulkan instance, device, and queues for video decode/encode.

use super::error::VulkanError;
use super::extensions;
use crate::gpu::{Codec, DecodeCapabilities, EncodeCapabilities, GpuPixelFormat};

use ash::vk;
use std::ffi::CStr;
use std::sync::Arc;

/// Vulkan context for video operations.
///
/// This struct manages the Vulkan instance, physical device, logical device,
/// and queues needed for video decode/encode operations.
pub struct VulkanContext {
    /// Vulkan entry point.
    #[allow(dead_code)]
    entry: ash::Entry,
    /// Vulkan instance.
    instance: ash::Instance,
    /// Physical device (GPU).
    physical_device: vk::PhysicalDevice,
    /// Logical device.
    device: Arc<ash::Device>,
    /// Graphics queue family index.
    #[allow(dead_code)]
    graphics_queue_family: u32,
    /// Video decode queue family index (if available).
    decode_queue_family: Option<u32>,
    /// Video encode queue family index (if available).
    encode_queue_family: Option<u32>,
    /// Graphics queue.
    #[allow(dead_code)]
    graphics_queue: vk::Queue,
    /// Video decode queue.
    decode_queue: Option<vk::Queue>,
    /// Video encode queue.
    encode_queue: Option<vk::Queue>,
    /// Device properties.
    device_properties: vk::PhysicalDeviceProperties,
    /// Supported decode codecs.
    decode_capabilities: Vec<DecodeCapabilities>,
    /// Supported encode codecs.
    encode_capabilities: Vec<EncodeCapabilities>,
}

impl VulkanContext {
    /// Create a new Vulkan context with video support.
    ///
    /// This will:
    /// 1. Load the Vulkan library
    /// 2. Create a Vulkan instance with required extensions
    /// 3. Find a GPU with video decode/encode support
    /// 4. Create a logical device with video queues
    ///
    /// Returns an error if Vulkan is not available or no compatible GPU is found.
    pub fn new() -> Result<Self, VulkanError> {
        // Load Vulkan library
        let entry = unsafe { ash::Entry::load()? };

        // Check instance version
        let instance_version = unsafe {
            entry
                .try_enumerate_instance_version()
                .map_err(|_| VulkanError::InitializationFailed)?
                .unwrap_or(vk::API_VERSION_1_0)
        };

        if vk::api_version_major(instance_version) < 1
            || (vk::api_version_major(instance_version) == 1
                && vk::api_version_minor(instance_version) < 3)
        {
            return Err(VulkanError::Other(format!(
                "Vulkan 1.3+ required, found {}.{}",
                vk::api_version_major(instance_version),
                vk::api_version_minor(instance_version)
            )));
        }

        // Create instance
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Parallax")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"Parallax")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let instance_extensions: Vec<*const i8> = vec![];

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions);

        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .map_err(|_| VulkanError::InitializationFailed)?
        };

        // Find physical device with video support
        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .map_err(|_| VulkanError::NoCompatibleDevice)?
        };

        if physical_devices.is_empty() {
            return Err(VulkanError::NoCompatibleDevice);
        }

        // Find best device with video support
        let (physical_device, device_properties, queue_families) =
            Self::select_physical_device(&instance, &physical_devices)?;

        // Find queue families
        let graphics_queue_family = queue_families
            .iter()
            .position(|qf| qf.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or(VulkanError::NoCompatibleDevice)? as u32;

        let decode_queue_family = queue_families
            .iter()
            .position(|qf| qf.queue_flags.contains(vk::QueueFlags::VIDEO_DECODE_KHR))
            .map(|i| i as u32);

        let encode_queue_family = queue_families
            .iter()
            .position(|qf| qf.queue_flags.contains(vk::QueueFlags::VIDEO_ENCODE_KHR))
            .map(|i| i as u32);

        // Build queue create infos
        let queue_priority = [1.0f32];
        let mut queue_create_infos = vec![
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(graphics_queue_family)
                .queue_priorities(&queue_priority),
        ];

        if let Some(decode_family) = decode_queue_family {
            if decode_family != graphics_queue_family {
                queue_create_infos.push(
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(decode_family)
                        .queue_priorities(&queue_priority),
                );
            }
        }

        if let Some(encode_family) = encode_queue_family {
            if encode_family != graphics_queue_family && Some(encode_family) != decode_queue_family
            {
                queue_create_infos.push(
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(encode_family)
                        .queue_priorities(&queue_priority),
                );
            }
        }

        // Get supported device extensions
        let available_extensions = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
                .unwrap_or_default()
        };

        let extension_names: Vec<&CStr> = available_extensions
            .iter()
            .map(|ext| unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) })
            .collect();

        // Build list of extensions to enable
        let mut device_extensions: Vec<*const i8> = vec![];

        // Add video extensions if available
        let video_extensions = [
            extensions::VIDEO_QUEUE,
            extensions::VIDEO_DECODE_QUEUE,
            extensions::VIDEO_ENCODE_QUEUE,
            extensions::VIDEO_DECODE_H264,
            extensions::VIDEO_DECODE_H265,
            extensions::VIDEO_DECODE_AV1,
            extensions::VIDEO_ENCODE_H264,
            extensions::VIDEO_ENCODE_H265,
            extensions::EXTERNAL_MEMORY,
            extensions::EXTERNAL_MEMORY_FD,
            extensions::EXTERNAL_MEMORY_DMABUF,
        ];

        for ext in &video_extensions {
            if extension_names.contains(ext) {
                device_extensions.push(ext.as_ptr());
            }
        }

        // Vulkan 1.3 features
        let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(true)
            .dynamic_rendering(true);

        let mut features2 =
            vk::PhysicalDeviceFeatures2::default().push_next(&mut vulkan_13_features);

        // Create logical device
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions)
            .push_next(&mut features2);

        let device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .map_err(|e| VulkanError::Other(format!("Failed to create device: {:?}", e)))?
        };

        let device = Arc::new(device);

        // Get queues
        let graphics_queue = unsafe { device.get_device_queue(graphics_queue_family, 0) };

        let decode_queue = decode_queue_family.map(|family| {
            if family == graphics_queue_family {
                graphics_queue
            } else {
                unsafe { device.get_device_queue(family, 0) }
            }
        });

        let encode_queue = encode_queue_family.map(|family| {
            if family == graphics_queue_family {
                graphics_queue
            } else if Some(family) == decode_queue_family {
                decode_queue.unwrap()
            } else {
                unsafe { device.get_device_queue(family, 0) }
            }
        });

        // Query video capabilities
        let decode_capabilities =
            Self::query_decode_capabilities(&instance, physical_device, &extension_names);
        let encode_capabilities =
            Self::query_encode_capabilities(&instance, physical_device, &extension_names);

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            graphics_queue_family,
            decode_queue_family,
            encode_queue_family,
            graphics_queue,
            decode_queue,
            encode_queue,
            device_properties,
            decode_capabilities,
            encode_capabilities,
        })
    }

    /// Select the best physical device with video support.
    fn select_physical_device(
        instance: &ash::Instance,
        devices: &[vk::PhysicalDevice],
    ) -> Result<
        (
            vk::PhysicalDevice,
            vk::PhysicalDeviceProperties,
            Vec<vk::QueueFamilyProperties>,
        ),
        VulkanError,
    > {
        // Prefer discrete GPUs, then integrated
        let mut best_device = None;
        let mut best_score = 0;

        for &device in devices {
            let properties = unsafe { instance.get_physical_device_properties(device) };
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(device) };

            // Check for graphics queue
            let has_graphics = queue_families
                .iter()
                .any(|qf| qf.queue_flags.contains(vk::QueueFlags::GRAPHICS));

            if !has_graphics {
                continue;
            }

            // Score based on device type
            let mut score = match properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 100,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 10,
                _ => 1,
            };

            // Bonus for video decode support
            if queue_families
                .iter()
                .any(|qf| qf.queue_flags.contains(vk::QueueFlags::VIDEO_DECODE_KHR))
            {
                score += 500;
            }

            // Bonus for video encode support
            if queue_families
                .iter()
                .any(|qf| qf.queue_flags.contains(vk::QueueFlags::VIDEO_ENCODE_KHR))
            {
                score += 500;
            }

            if score > best_score {
                best_score = score;
                best_device = Some((device, properties, queue_families));
            }
        }

        best_device.ok_or(VulkanError::NoCompatibleDevice)
    }

    /// Query decode capabilities for supported codecs.
    fn query_decode_capabilities(
        _instance: &ash::Instance,
        _physical_device: vk::PhysicalDevice,
        extension_names: &[&CStr],
    ) -> Vec<DecodeCapabilities> {
        let mut caps = Vec::new();

        // Check which decode extensions are available
        if extension_names.contains(&extensions::VIDEO_DECODE_H264) {
            caps.push(DecodeCapabilities {
                codec: Codec::H264,
                profiles: vec![66, 77, 100], // Baseline, Main, High
                max_level: 52,
                max_width: 4096,
                max_height: 2160,
                bit_depths: vec![8],
                output_formats: vec![GpuPixelFormat::Nv12],
            });
        }

        if extension_names.contains(&extensions::VIDEO_DECODE_H265) {
            caps.push(DecodeCapabilities {
                codec: Codec::H265,
                profiles: vec![1, 2], // Main, Main10
                max_level: 52,
                max_width: 8192,
                max_height: 4320,
                bit_depths: vec![8, 10],
                output_formats: vec![GpuPixelFormat::Nv12, GpuPixelFormat::P010],
            });
        }

        if extension_names.contains(&extensions::VIDEO_DECODE_AV1) {
            caps.push(DecodeCapabilities {
                codec: Codec::Av1,
                profiles: vec![0, 1], // Main, High
                max_level: 60,
                max_width: 8192,
                max_height: 4320,
                bit_depths: vec![8, 10],
                output_formats: vec![GpuPixelFormat::Nv12, GpuPixelFormat::P010],
            });
        }

        caps
    }

    /// Query encode capabilities for supported codecs.
    fn query_encode_capabilities(
        _instance: &ash::Instance,
        _physical_device: vk::PhysicalDevice,
        extension_names: &[&CStr],
    ) -> Vec<EncodeCapabilities> {
        use crate::gpu::RateControlMode;
        let mut caps = Vec::new();

        if extension_names.contains(&extensions::VIDEO_ENCODE_H264) {
            caps.push(EncodeCapabilities {
                codec: Codec::H264,
                profiles: vec![66, 77, 100], // Baseline, Main, High
                max_level: 52,
                max_width: 4096,
                max_height: 2160,
                bit_depths: vec![8],
                rate_control_modes: vec![RateControlMode::Cbr, RateControlMode::Vbr],
                max_bitrate: 100_000_000, // 100 Mbps
            });
        }

        if extension_names.contains(&extensions::VIDEO_ENCODE_H265) {
            caps.push(EncodeCapabilities {
                codec: Codec::H265,
                profiles: vec![1], // Main
                max_level: 52,
                max_width: 8192,
                max_height: 4320,
                bit_depths: vec![8, 10],
                rate_control_modes: vec![RateControlMode::Cbr, RateControlMode::Vbr],
                max_bitrate: 100_000_000,
            });
        }

        caps
    }

    /// Check if a codec is supported for decoding.
    pub fn supports_decode(&self, codec: Codec) -> bool {
        self.decode_capabilities.iter().any(|c| c.codec == codec)
    }

    /// Check if a codec is supported for encoding.
    pub fn supports_encode(&self, codec: Codec) -> bool {
        self.encode_capabilities.iter().any(|c| c.codec == codec)
    }

    /// Get decode capabilities for a specific codec.
    pub fn decode_capabilities(&self, codec: Codec) -> Option<&DecodeCapabilities> {
        self.decode_capabilities.iter().find(|c| c.codec == codec)
    }

    /// Get encode capabilities for a specific codec.
    pub fn encode_capabilities(&self, codec: Codec) -> Option<&EncodeCapabilities> {
        self.encode_capabilities.iter().find(|c| c.codec == codec)
    }

    /// Get all decode capabilities.
    pub fn all_decode_capabilities(&self) -> &[DecodeCapabilities] {
        &self.decode_capabilities
    }

    /// Get all encode capabilities.
    pub fn all_encode_capabilities(&self) -> &[EncodeCapabilities] {
        &self.encode_capabilities
    }

    /// Get device name.
    pub fn device_name(&self) -> String {
        let name = unsafe { CStr::from_ptr(self.device_properties.device_name.as_ptr()) };
        name.to_string_lossy().into_owned()
    }

    /// Get the Vulkan device.
    pub fn device(&self) -> &Arc<ash::Device> {
        &self.device
    }

    /// Get the physical device.
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// Get the Vulkan instance.
    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    /// Get the decode queue family index.
    pub fn decode_queue_family(&self) -> Option<u32> {
        self.decode_queue_family
    }

    /// Get the encode queue family index.
    pub fn encode_queue_family(&self) -> Option<u32> {
        self.encode_queue_family
    }

    /// Get the decode queue.
    pub fn decode_queue(&self) -> Option<vk::Queue> {
        self.decode_queue
    }

    /// Get the encode queue.
    pub fn encode_queue(&self) -> Option<vk::Queue> {
        self.encode_queue
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            // Device is dropped via Arc
            // Note: we can't destroy the device here because it's in an Arc
            // It will be destroyed when all references are dropped
        }
    }
}

// Manual Drop for the Arc<Device> - this is handled when the last Arc is dropped
// The instance should be destroyed after the device

unsafe impl Send for VulkanContext {}
unsafe impl Sync for VulkanContext {}
