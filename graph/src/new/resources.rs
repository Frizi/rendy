use bitflags::bitflags;

/// Id of the buffer in graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferId(pub(super) usize);

/// Id of the image (or swapchain) in graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImageId(pub(super) usize);

/// Id of virtual resource in graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VirtualId(pub(super) usize);

#[derive(Debug, Clone, Copy)]
pub struct ImageInfo {
    kind: gfx_hal::image::Kind,
    levels: gfx_hal::image::Level,
    format: gfx_hal::format::Format,
    clear: Option<gfx_hal::command::ClearValue>,
}

#[derive(Debug, Clone, Copy)]
pub struct BufferInfo {
    size: u64,
    clear: Option<u32>,
}

bitflags!(
    /// Buffer access flags.
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct NodeBufferAccess: u32 {
        const INDIRECT_COMMAND_READ = 0x1;
        const INDEX_BUFFER_READ = 0x2;
        const VERTEX_BUFFER_READ = 0x4;
        const UNIFORM_BUFFER_READ = 0x8;
        const STORAGE_BUFFER_READ = 0x10;
        const STORAGE_BUFFER_WRITE = 0x20;
        const UNIFORM_TEXEL_BUFFER_READ = 0x40;
        const STORAGE_TEXEL_BUFFER_READ = 0x80;
        const STORAGE_TEXEL_BUFFER_WRITE = 0x100;
        const TRANSFER_READ = 0x200;
        const TRANSFER_WRITE = 0x400;
    }
);

bitflags!(
    /// Image access flags.
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct NodeImageAccess: u32 {
        const INPUT_ATTACHMENT_READ = 0x1;
        const SAMPLED_IMAGE_READ = 0x2;
        const STORAGE_IMAGE_READ = 0x4;
        const STORAGE_IMAGE_WRITE = 0x8;
        const COLOR_ATTACHMENT_READ = 0x10;
        const COLOR_ATTACHMENT_WRITE = 0x20;
        const DEPTH_STENCIL_ATTACHMENT_READ = 0x40;
        const DEPTH_STENCIL_ATTACHMENT_WRITE = 0x80;
        const TRANSFER_READ = 0x100;
        const TRANSFER_WRITE = 0x200;
    }
);

impl NodeImageAccess {
    pub fn is_write(&self) -> bool {
        self.contains(
            Self::STORAGE_IMAGE_WRITE
                | Self::COLOR_ATTACHMENT_WRITE
                | Self::DEPTH_STENCIL_ATTACHMENT_WRITE
                | Self::TRANSFER_WRITE,
        )
    }
}

impl NodeBufferAccess {
    pub fn is_write(&self) -> bool {
        self.contains(
            Self::STORAGE_BUFFER_WRITE | Self::STORAGE_TEXEL_BUFFER_WRITE | Self::TRANSFER_WRITE,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeVirtualAccess {
    Read,
    Write,
}

impl NodeVirtualAccess {
    pub fn is_write(&self) -> bool {
        *self == NodeVirtualAccess::Write
    }
}
