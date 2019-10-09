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

bitflags!(
    /// Virtual resource access flags.
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct NodeVirtualAccess: u32 {
        const VIRTUAL_WRITE = 0x1;
        const VIRTUAL_READ = 0x2;
    }
);

impl NodeImageAccess {
    pub fn reads(&self) -> Self {
        *self
            & (Self::INPUT_ATTACHMENT_READ
                | Self::SAMPLED_IMAGE_READ
                | Self::STORAGE_IMAGE_READ
                | Self::COLOR_ATTACHMENT_READ
                | Self::DEPTH_STENCIL_ATTACHMENT_READ
                | Self::TRANSFER_READ)
    }

    pub fn writes(&self) -> Self {
        *self
            & (Self::STORAGE_IMAGE_WRITE
                | Self::COLOR_ATTACHMENT_WRITE
                | Self::DEPTH_STENCIL_ATTACHMENT_WRITE
                | Self::TRANSFER_WRITE)
    }

    pub fn is_attachment_only(&self) -> bool {
        self.is_attachment()
            && (Self::INPUT_ATTACHMENT_READ
                | Self::COLOR_ATTACHMENT_READ
                | Self::COLOR_ATTACHMENT_WRITE
                | Self::DEPTH_STENCIL_ATTACHMENT_READ
                | Self::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .contains(*self)
    }

    pub fn is_attachment(&self) -> bool {
        self.intersects(
            Self::INPUT_ATTACHMENT_READ
                | Self::COLOR_ATTACHMENT_READ
                | Self::COLOR_ATTACHMENT_WRITE
                | Self::DEPTH_STENCIL_ATTACHMENT_READ
                | Self::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )
    }
}

impl NodeBufferAccess {
    pub fn reads(&self) -> Self {
        *self
            & (Self::INDIRECT_COMMAND_READ
                | Self::INDEX_BUFFER_READ
                | Self::VERTEX_BUFFER_READ
                | Self::UNIFORM_BUFFER_READ
                | Self::STORAGE_BUFFER_READ
                | Self::UNIFORM_TEXEL_BUFFER_READ
                | Self::STORAGE_TEXEL_BUFFER_READ
                | Self::TRANSFER_READ)
    }

    pub fn writes(&self) -> Self {
        *self
            & (Self::STORAGE_BUFFER_WRITE | Self::STORAGE_TEXEL_BUFFER_WRITE | Self::TRANSFER_WRITE)
    }
}

impl NodeVirtualAccess {
    pub fn reads(&self) -> Self {
        *self & Self::VIRTUAL_READ
    }

    pub fn writes(&self) -> Self {
        *self & Self::VIRTUAL_WRITE
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceId {
    Image(ImageId),
    Buffer(BufferId),
    Virtual(VirtualId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ResourceUsage {
    Image(ImageId, NodeImageAccess),
    Buffer(BufferId, NodeBufferAccess),
    Virtual(VirtualId, NodeVirtualAccess),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResourceAccess {
    Image(NodeImageAccess),
    Buffer(NodeBufferAccess),
    Virtual(NodeVirtualAccess),
}

impl ResourceUsage {
    pub(crate) fn is_same_resource(&self, usage: &ResourceUsage) -> bool {
        match (self, usage) {
            (Self::Image(a, _), Self::Image(b, _)) if a == b => true,
            (Self::Buffer(a, _), Self::Buffer(b, _)) if a == b => true,
            (Self::Virtual(a, _), Self::Virtual(b, _)) if a == b => true,
            _ => false,
        }
    }
    pub(crate) fn merge_access(&mut self, usage: &ResourceUsage) {
        match (self, usage) {
            (Self::Image(_, mut a), Self::Image(_, b)) => a |= *b,
            (Self::Buffer(_, mut a), Self::Buffer(_, b)) => a |= *b,
            (Self::Virtual(_, mut a), Self::Virtual(_, b)) => a |= *b,
            _ => panic!("Trying to merge access flags of different resource kinds"),
        }
    }
    pub(crate) fn open(self) -> (ResourceId, ResourceAccess) {
        match self {
            Self::Image(id, access) => (ResourceId::Image(id), ResourceAccess::Image(access)),
            Self::Buffer(id, access) => (ResourceId::Buffer(id), ResourceAccess::Buffer(access)),
            Self::Virtual(id, access) => (ResourceId::Virtual(id), ResourceAccess::Virtual(access)),
        }
    }
}
impl ResourceAccess {
    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Self::Image(access) => access.is_empty(),
            Self::Buffer(access) => access.is_empty(),
            Self::Virtual(access) => access.is_empty(),
        }
    }
    pub(crate) fn is_attachment_only(&self) -> bool {
        match self {
            Self::Image(access) => access.is_attachment_only(),
            _ => false,
        }
    }
    pub(crate) fn is_attachment(&self) -> bool {
        match self {
            Self::Image(access) => access.is_attachment(),
            _ => false,
        }
    }

    pub(crate) fn split_rw(&self) -> (Self, Self) {
        match self {
            Self::Image(access) => (Self::Image(access.reads()), Self::Image(access.writes())),
            Self::Buffer(access) => (Self::Buffer(access.reads()), Self::Buffer(access.writes())),
            Self::Virtual(access) => (
                Self::Virtual(access.reads()),
                Self::Virtual(access.writes()),
            ),
        }
    }
}
