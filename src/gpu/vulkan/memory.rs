//! Vulkan GPU memory management.
//!
//! Implements the GpuMemory trait for Vulkan, including DMA-BUF import/export.

use super::context::VulkanContext;
use super::error::VulkanError;
use crate::error::Result;
use crate::gpu::GpuUsage;
use crate::gpu::traits::{GpuBuffer, GpuBufferHandle, GpuMemory, GpuPixelFormat};

use ash::vk;
use std::os::fd::OwnedFd;
use std::sync::Arc;

/// Vulkan GPU memory allocator.
///
/// Manages GPU memory allocation and DMA-BUF import/export for video operations.
pub struct VulkanGpuMemory {
    /// Vulkan device.
    device: Arc<ash::Device>,
    /// Physical device for memory type queries.
    #[allow(dead_code)]
    physical_device: vk::PhysicalDevice,
    /// Vulkan instance for extension functions.
    instance: ash::Instance,
    /// Memory type index for device-local memory.
    device_local_memory_type: u32,
    /// Memory type index for host-visible memory.
    host_visible_memory_type: u32,
    /// Memory type index for external memory (DMA-BUF).
    external_memory_type: Option<u32>,
}

impl VulkanGpuMemory {
    /// Create a new Vulkan memory allocator from a context.
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        let device = ctx.device().clone();
        let physical_device = ctx.physical_device();
        let instance = ctx.instance().clone();

        // Query memory properties
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        // Find device-local memory type
        let device_local_memory_type = Self::find_memory_type(
            &memory_properties,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        )
        .ok_or_else(|| VulkanError::Other("No device-local memory type found".to_string()))?;

        // Find host-visible memory type
        let host_visible_memory_type = Self::find_memory_type(
            &memory_properties,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::empty(),
        )
        .ok_or_else(|| VulkanError::Other("No host-visible memory type found".to_string()))?;

        // Find external memory type (for DMA-BUF)
        // This requires querying external buffer properties
        let external_memory_type = Self::find_external_memory_type(&memory_properties);

        Ok(Self {
            device,
            physical_device,
            instance,
            device_local_memory_type,
            host_visible_memory_type,
            external_memory_type,
        })
    }

    /// Find a memory type index matching the requirements.
    fn find_memory_type(
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        required_flags: vk::MemoryPropertyFlags,
        preferred_flags: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        let mut best_match = None;
        let mut best_score = 0;

        for i in 0..memory_properties.memory_type_count {
            let memory_type = memory_properties.memory_types[i as usize];

            if memory_type.property_flags.contains(required_flags) {
                let mut score = 1;

                // Prefer memory types with preferred flags
                if memory_type.property_flags.contains(preferred_flags) {
                    score += 10;
                }

                // Prefer device-local memory
                if memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                {
                    score += 5;
                }

                if score > best_score {
                    best_score = score;
                    best_match = Some(i);
                }
            }
        }

        best_match
    }

    /// Find a memory type suitable for external memory (DMA-BUF).
    fn find_external_memory_type(
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
    ) -> Option<u32> {
        // For DMA-BUF, we typically need device-local memory
        // The actual external memory type is determined at import time
        Self::find_memory_type(
            memory_properties,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        )
    }

    /// Get the Vulkan device.
    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    /// Allocate a Vulkan image for video decode/encode.
    pub fn allocate_video_image(
        &mut self,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<(vk::Image, vk::DeviceMemory)> {
        // Create image
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe {
            self.device
                .create_image(&image_info, None)
                .map_err(|e| VulkanError::Other(format!("Failed to create image: {:?}", e)))?
        };

        // Get memory requirements
        let memory_requirements = unsafe { self.device.get_image_memory_requirements(image) };

        // Allocate memory
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(self.device_local_memory_type);

        let memory = unsafe {
            self.device
                .allocate_memory(&alloc_info, None)
                .map_err(|_e| {
                    self.device.destroy_image(image, None);
                    VulkanError::OutOfMemory
                })?
        };

        // Bind memory to image
        unsafe {
            self.device
                .bind_image_memory(image, memory, 0)
                .map_err(|e| {
                    self.device.free_memory(memory, None);
                    self.device.destroy_image(image, None);
                    VulkanError::Other(format!("Failed to bind image memory: {:?}", e))
                })?;
        }

        Ok((image, memory))
    }

    /// Convert GpuPixelFormat to Vulkan format.
    pub fn to_vk_format(format: GpuPixelFormat) -> vk::Format {
        match format {
            GpuPixelFormat::Nv12 => vk::Format::G8_B8R8_2PLANE_420_UNORM,
            GpuPixelFormat::P010 => vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
            GpuPixelFormat::I420 => vk::Format::G8_B8_R8_3PLANE_420_UNORM,
            GpuPixelFormat::I420p10 => vk::Format::G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16,
        }
    }
}

impl GpuMemory for VulkanGpuMemory {
    fn import_dmabuf(&mut self, fd: OwnedFd, size: usize) -> Result<GpuBuffer> {
        use std::os::fd::AsRawFd;

        let external_memory_type = self
            .external_memory_type
            .ok_or_else(|| VulkanError::DmaBufError("External memory not supported".to_string()))?;

        // Import memory from DMA-BUF fd
        let mut import_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(fd.as_raw_fd());

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(size as u64)
            .memory_type_index(external_memory_type)
            .push_next(&mut import_info);

        let memory = unsafe {
            self.device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| {
                    VulkanError::DmaBufError(format!("Failed to import DMA-BUF: {:?}", e))
                })?
        };

        // Vulkan now owns the fd, don't close it
        std::mem::forget(fd);

        Ok(GpuBuffer {
            handle: GpuBufferHandle::Vulkan {
                memory,
                image: None,
            },
            size,
            usage: GpuUsage::default(),
        })
    }

    fn export_dmabuf(&self, buffer: &GpuBuffer) -> Result<OwnedFd> {
        use std::os::fd::FromRawFd;

        let memory = match &buffer.handle {
            GpuBufferHandle::Vulkan { memory, .. } => *memory,
            _ => {
                return Err(
                    VulkanError::DmaBufError("Buffer is not a Vulkan buffer".to_string()).into(),
                );
            }
        };

        let get_fd_info = vk::MemoryGetFdInfoKHR::default()
            .memory(memory)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

        // Get the external memory FD function
        let external_memory_fd_fn =
            ash::khr::external_memory_fd::Device::new(&self.instance, &self.device);

        let fd = unsafe {
            external_memory_fd_fn
                .get_memory_fd(&get_fd_info)
                .map_err(|e| {
                    VulkanError::DmaBufError(format!("Failed to export DMA-BUF: {:?}", e))
                })?
        };

        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    fn allocate(&mut self, size: usize, usage: GpuUsage) -> Result<GpuBuffer> {
        let memory_type = if usage.transfer {
            self.host_visible_memory_type
        } else {
            self.device_local_memory_type
        };

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(size as u64)
            .memory_type_index(memory_type);

        let memory = unsafe {
            self.device
                .allocate_memory(&alloc_info, None)
                .map_err(|_| VulkanError::OutOfMemory)?
        };

        Ok(GpuBuffer {
            handle: GpuBufferHandle::Vulkan {
                memory,
                image: None,
            },
            size,
            usage,
        })
    }

    fn allocate_image(
        &mut self,
        width: u32,
        height: u32,
        format: GpuPixelFormat,
        usage: GpuUsage,
    ) -> Result<GpuBuffer> {
        let vk_format = Self::to_vk_format(format);

        let mut vk_usage = vk::ImageUsageFlags::empty();
        if usage.decode_dst {
            vk_usage |= vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR;
        }
        if usage.decode_src {
            vk_usage |= vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR;
        }
        if usage.encode_src {
            vk_usage |= vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR;
        }
        if usage.encode_dst {
            vk_usage |= vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR;
        }
        if usage.transfer {
            vk_usage |= vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        }

        let (image, memory) = self.allocate_video_image(width, height, vk_format, vk_usage)?;

        let size = format.frame_size(width, height);

        Ok(GpuBuffer {
            handle: GpuBufferHandle::Vulkan {
                memory,
                image: Some(image),
            },
            size,
            usage,
        })
    }

    fn free(&mut self, buffer: GpuBuffer) {
        if let GpuBufferHandle::Vulkan { memory, image } = buffer.handle {
            unsafe {
                if let Some(img) = image {
                    self.device.destroy_image(img, None);
                }
                self.device.free_memory(memory, None);
            }
        }
    }

    fn map(&self, buffer: &GpuBuffer) -> Result<*mut u8> {
        let memory = match &buffer.handle {
            GpuBufferHandle::Vulkan { memory, .. } => *memory,
            _ => return Err(VulkanError::Other("Not a Vulkan buffer".to_string()).into()),
        };

        let ptr = unsafe {
            self.device
                .map_memory(memory, 0, buffer.size as u64, vk::MemoryMapFlags::empty())
                .map_err(|e| VulkanError::Other(format!("Failed to map memory: {:?}", e)))?
        };

        Ok(ptr as *mut u8)
    }

    fn unmap(&self, buffer: &GpuBuffer) {
        if let GpuBufferHandle::Vulkan { memory, .. } = &buffer.handle {
            unsafe {
                self.device.unmap_memory(*memory);
            }
        }
    }
}

impl Drop for VulkanGpuMemory {
    fn drop(&mut self) {
        // Device is held by Arc, will be cleaned up when all references are dropped
    }
}
