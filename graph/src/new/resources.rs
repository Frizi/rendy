use bitflags::bitflags;

/// Id of the buffer in graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferId(usize);

/// Id of the image (or swapchain) in graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImageId(usize);

/// Id of virtual resource in graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VirtualId(usize);

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
    fn is_write(&self) -> bool {
        self.contains(
            Self::STORAGE_IMAGE_WRITE
                | Self::COLOR_ATTACHMENT_WRITE
                | Self::DEPTH_STENCIL_ATTACHMENT_WRITE
                | Self::TRANSFER_WRITE,
        )
    }
}

impl NodeBufferAccess {
    fn is_write(&self) -> bool {
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
