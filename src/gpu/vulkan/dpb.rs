//! Decoded Picture Buffer (DPB) management.
//!
//! The DPB holds reference frames needed for inter-frame prediction in video
//! decoding. Vulkan Video requires explicit DPB management unlike VA-API.
//!
//! # DPB Modes
//!
//! Based on FFmpeg's implementation, there are three DPB configurations:
//!
//! 1. **Coincident Mode** (Intel): Output frames are directly usable as references.
//!    No separate DPB allocation needed.
//!
//! 2. **Distinct Separate Mode** (AMD, NVIDIA): Each reference needs a separate
//!    DPB image. We allocate a pool of DPB slots.
//!
//! 3. **Distinct Layered Mode**: A single multi-layer image holds all references.
//!    More memory efficient but requires upfront allocation.
//!
//! This implementation uses Distinct Separate Mode for maximum compatibility.

use super::error::VulkanError;
use super::session::VideoSession;
use crate::error::Result;

use ash::vk;
use std::sync::Arc;

/// A single DPB slot holding a reference picture.
pub struct DpbSlot {
    /// Image for this slot.
    pub image: vk::Image,
    /// Image view for this slot.
    pub image_view: vk::ImageView,
    /// Memory backing this slot.
    pub memory: vk::DeviceMemory,
    /// Slot index in the DPB.
    pub index: u32,
    /// Picture Order Count (for H.264/H.265).
    pub poc: i32,
    /// Frame number (for H.264).
    pub frame_num: u32,
    /// Is this slot currently in use as a reference?
    pub in_use: bool,
    /// Is this a long-term reference? (H.264)
    pub is_long_term: bool,
}

/// Decoded Picture Buffer manager.
///
/// Manages a pool of reference frames for video decoding.
pub struct Dpb {
    /// Vulkan device.
    device: Arc<ash::Device>,
    /// DPB slots.
    slots: Vec<DpbSlot>,
    /// Image format.
    format: vk::Format,
    /// Image extent.
    extent: vk::Extent2D,
    /// Maximum slots.
    max_slots: u32,
}

/// Reference to a DPB slot for use in decode operations.
#[derive(Debug, Clone, Copy)]
pub struct DpbReference {
    /// Slot index.
    pub slot_index: u32,
    /// Picture Order Count.
    pub poc: i32,
    /// Frame number.
    pub frame_num: u32,
    /// Is long-term reference.
    pub is_long_term: bool,
}

impl Dpb {
    /// Create a new DPB for the given session.
    ///
    /// Allocates `max_slots` reference frame images.
    pub fn new(session: &VideoSession, max_slots: Option<u32>) -> Result<Self> {
        let device = Arc::new(session.device().clone());
        let max_slots = max_slots
            .unwrap_or(session.max_dpb_slots())
            .min(session.max_dpb_slots());

        let format = session.picture_format();
        let extent = session.coded_extent();

        // Allocate DPB slots
        let mut slots = Vec::with_capacity(max_slots as usize);
        for i in 0..max_slots {
            let slot = Self::allocate_slot(&device, session, i, format, extent)?;
            slots.push(slot);
        }

        Ok(Self {
            device,
            slots,
            format,
            extent,
            max_slots,
        })
    }

    /// Allocate a single DPB slot.
    fn allocate_slot(
        device: &ash::Device,
        _session: &VideoSession,
        index: u32,
        format: vk::Format,
        extent: vk::Extent2D,
    ) -> Result<DpbSlot> {
        // Create image for DPB slot
        // DPB images need VIDEO_DECODE_DPB_KHR usage
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR
                    | vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe {
            device
                .create_image(&image_info, None)
                .map_err(VulkanError::from)?
        };

        // Get memory requirements
        let mem_requirements = unsafe { device.get_image_memory_requirements(image) };

        // For now, use memory type 0 with device local - proper implementation
        // would query physical device memory properties and find the best type.
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(0);

        let memory = unsafe {
            device.allocate_memory(&alloc_info, None).map_err(|e| {
                device.destroy_image(image, None);
                VulkanError::from(e)
            })?
        };

        // Bind memory to image
        unsafe {
            device.bind_image_memory(image, memory, 0).map_err(|e| {
                device.free_memory(memory, None);
                device.destroy_image(image, None);
                VulkanError::from(e)
            })?;
        }

        // Create image view
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let image_view = unsafe {
            device.create_image_view(&view_info, None).map_err(|e| {
                device.free_memory(memory, None);
                device.destroy_image(image, None);
                VulkanError::from(e)
            })?
        };

        Ok(DpbSlot {
            image,
            image_view,
            memory,
            index,
            poc: 0,
            frame_num: 0,
            in_use: false,
            is_long_term: false,
        })
    }

    /// Get a free DPB slot.
    ///
    /// Returns the index of a slot that is not currently in use.
    pub fn acquire_slot(&mut self) -> Option<u32> {
        for slot in &mut self.slots {
            if !slot.in_use {
                slot.in_use = true;
                return Some(slot.index);
            }
        }
        None
    }

    /// Mark a slot as containing a reference frame.
    pub fn mark_as_reference(
        &mut self,
        slot_index: u32,
        poc: i32,
        frame_num: u32,
        long_term: bool,
    ) {
        if let Some(slot) = self.slots.get_mut(slot_index as usize) {
            slot.poc = poc;
            slot.frame_num = frame_num;
            slot.is_long_term = long_term;
            slot.in_use = true;
        }
    }

    /// Release a slot (no longer needed as reference).
    pub fn release_slot(&mut self, slot_index: u32) {
        if let Some(slot) = self.slots.get_mut(slot_index as usize) {
            slot.in_use = false;
            slot.is_long_term = false;
        }
    }

    /// Clear all slots (e.g., at IDR frame).
    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            slot.in_use = false;
            slot.is_long_term = false;
            slot.poc = 0;
            slot.frame_num = 0;
        }
    }

    /// Get a slot by index.
    pub fn slot(&self, index: u32) -> Option<&DpbSlot> {
        self.slots.get(index as usize)
    }

    /// Get a mutable slot by index.
    pub fn slot_mut(&mut self, index: u32) -> Option<&mut DpbSlot> {
        self.slots.get_mut(index as usize)
    }

    /// Find a slot by POC (Picture Order Count).
    pub fn find_by_poc(&self, poc: i32) -> Option<&DpbSlot> {
        self.slots.iter().find(|s| s.in_use && s.poc == poc)
    }

    /// Find a slot by frame number (H.264).
    pub fn find_by_frame_num(&self, frame_num: u32) -> Option<&DpbSlot> {
        self.slots
            .iter()
            .find(|s| s.in_use && s.frame_num == frame_num)
    }

    /// Get all active reference slots.
    pub fn active_references(&self) -> Vec<DpbReference> {
        self.slots
            .iter()
            .filter(|s| s.in_use)
            .map(|s| DpbReference {
                slot_index: s.index,
                poc: s.poc,
                frame_num: s.frame_num,
                is_long_term: s.is_long_term,
            })
            .collect()
    }

    /// Get short-term references sorted by POC (for H.264 ref list construction).
    pub fn short_term_refs_by_poc(&self) -> Vec<DpbReference> {
        let mut refs: Vec<_> = self
            .slots
            .iter()
            .filter(|s| s.in_use && !s.is_long_term)
            .map(|s| DpbReference {
                slot_index: s.index,
                poc: s.poc,
                frame_num: s.frame_num,
                is_long_term: false,
            })
            .collect();
        refs.sort_by_key(|r| r.poc);
        refs
    }

    /// Get long-term references (for H.264).
    pub fn long_term_refs(&self) -> Vec<DpbReference> {
        self.slots
            .iter()
            .filter(|s| s.in_use && s.is_long_term)
            .map(|s| DpbReference {
                slot_index: s.index,
                poc: s.poc,
                frame_num: s.frame_num,
                is_long_term: true,
            })
            .collect()
    }

    /// Number of slots in use.
    pub fn active_count(&self) -> u32 {
        self.slots.iter().filter(|s| s.in_use).count() as u32
    }

    /// Maximum number of slots.
    pub fn max_slots(&self) -> u32 {
        self.max_slots
    }

    /// Image format.
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// Image extent.
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }
}

impl Drop for Dpb {
    fn drop(&mut self) {
        unsafe {
            for slot in &self.slots {
                self.device.destroy_image_view(slot.image_view, None);
                self.device.destroy_image(slot.image, None);
                self.device.free_memory(slot.memory, None);
            }
        }
    }
}

// Safety: Dpb manages its own resources
unsafe impl Send for Dpb {}
unsafe impl Sync for Dpb {}
