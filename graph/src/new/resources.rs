use bitflags::bitflags;
use gfx_hal::pso::PipelineStage;

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
    pub kind: gfx_hal::image::Kind,
    pub levels: gfx_hal::image::Level,
    pub format: gfx_hal::format::Format,
    pub clear: Option<gfx_hal::command::ClearValue>,
}

#[derive(Debug, Clone, Copy)]
pub struct BufferInfo {
    pub size: u64,
    pub clear: Option<u32>,
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

    pub fn is_write(&self) -> bool {
        !self.writes().is_empty()
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

    pub fn is_write(&self) -> bool {
        !self.writes().is_empty()
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
    Image(ImageId, NodeImageAccess, PipelineStage),
    Buffer(BufferId, NodeBufferAccess, PipelineStage),
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
            (Self::Image(a, _, _), Self::Image(b, _, _)) if a == b => true,
            (Self::Buffer(a, _, _), Self::Buffer(b, _, _)) if a == b => true,
            (Self::Virtual(a, _), Self::Virtual(b, _)) if a == b => true,
            _ => false,
        }
    }
    pub(crate) fn merge_access(&mut self, usage: &ResourceUsage) {
        match (self, usage) {
            (Self::Image(_, mut access_a, mut stage_a), Self::Image(_, access_b, stage_b)) => {
                access_a |= *access_b;
                stage_a |= *stage_b;
            }
            (Self::Buffer(_, mut access_a, mut stage_a), Self::Buffer(_, access_b, stage_b)) => {
                access_a |= *access_b;
                stage_a |= *stage_b;
            }
            (Self::Virtual(_, mut access_a), Self::Virtual(_, access_b)) => {
                access_a |= *access_b;
            }
            _ => panic!("Trying to merge access flags of different resource kinds"),
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

bitflags::bitflags!(
    // Those bits are exactly matching the pipeline stages

    /// Image access flags.
    pub struct ShaderUsage: u32 {
        /// Usage in vertex shader
        const VERTEX = 0x8;
        /// Usage in hull shader
        const HULL = 0x10;
        // Usage in domain shader
        const DOMAIN = 0x20;
        // Usage in geometry shader
        const GEOMETRY = 0x40;
        // Usage in fragment shader
        const FRAGMENT = 0x80;
        // Usage in compute shader
        const COMPUTE = 0x800;
    }
);

impl ShaderUsage {
    fn stage(&self) -> PipelineStage {
        // all possible values are known to be valid stages as well
        unsafe { PipelineStage::from_bits_unchecked(self.bits) }
    }
}

/// Declare how an image will be used in this rendering node
#[derive(Debug, Clone, Copy)]
pub enum ImageUsage {
    /// Image will be used as an input attachment
    InputAttachment,
    /// Image will be used as an color attachment for reading only
    ColorAttachmentRead,
    /// Image will be used as an color attachment for writing only
    ColorAttachmentWrite,
    /// Image will be used as an depth/stencil attachment for reading only
    DepthStencilAttachmentRead,
    /// Image will be used as an depth/stencil attachment for writing only
    DepthStencilAttachmentWrite,
    /// Image will be sampled in defined shader stages
    Sampled(ShaderUsage),
    /// Image will be used as a readonly storage in defined shader stages
    StorageRead(ShaderUsage),
    /// Image will be used as a writeable storage in defined shader stages
    StorageWrite(ShaderUsage),
    /// Image will be used in a memory transfer as a source
    TransferRead,
    /// Image will be used in a memory transfer as a destination
    TransferWrite,
    /// Custom image usage with manually provided access and pipeline stages
    Custom(NodeImageAccess, PipelineStage),
}

impl ImageUsage {
    pub(crate) fn stage(&self) -> PipelineStage {
        match self {
            // TODO: not sure about attachment reads
            Self::InputAttachment | Self::ColorAttachmentRead => PipelineStage::TOP_OF_PIPE,
            Self::ColorAttachmentWrite => PipelineStage::COLOR_ATTACHMENT_OUTPUT,
            Self::DepthStencilAttachmentWrite | Self::DepthStencilAttachmentRead => {
                PipelineStage::EARLY_FRAGMENT_TESTS | PipelineStage::LATE_FRAGMENT_TESTS
            }
            Self::Sampled(shader) => shader.stage(),
            Self::StorageRead(shader) => shader.stage(),
            Self::StorageWrite(shader) => shader.stage(),
            Self::TransferRead | Self::TransferWrite => PipelineStage::TRANSFER,
            Self::Custom(_, stage) => *stage,
        }
    }

    pub(crate) fn access(&self) -> NodeImageAccess {
        match self {
            Self::InputAttachment => NodeImageAccess::INPUT_ATTACHMENT_READ,
            Self::ColorAttachmentRead => NodeImageAccess::COLOR_ATTACHMENT_READ,
            Self::ColorAttachmentWrite => NodeImageAccess::COLOR_ATTACHMENT_WRITE,
            Self::DepthStencilAttachmentRead => NodeImageAccess::DEPTH_STENCIL_ATTACHMENT_READ,
            Self::DepthStencilAttachmentWrite => NodeImageAccess::DEPTH_STENCIL_ATTACHMENT_WRITE,
            Self::Sampled(_) => NodeImageAccess::SAMPLED_IMAGE_READ,
            Self::StorageRead(_) => NodeImageAccess::STORAGE_IMAGE_READ,
            Self::StorageWrite(_) => NodeImageAccess::STORAGE_IMAGE_WRITE,
            Self::TransferRead => NodeImageAccess::TRANSFER_READ,
            Self::TransferWrite => NodeImageAccess::TRANSFER_WRITE,
            Self::Custom(access, _) => *access,
        }
    }

    pub(crate) fn resource_usage(&self, id: ImageId) -> ResourceUsage {
        ResourceUsage::Image(id, self.access(), self.stage())
    }
}

impl std::ops::BitOr for ImageUsage {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        Self::Custom(self.access() | other.access(), self.stage() | other.stage())
    }
}

/// Declare how a buffer will be used in this rendering node
#[derive(Debug, Clone, Copy)]
pub enum BufferUsage {
    /// Buffer will be used as a indirect commad buffer
    IndirectCommand,
    // Buffer will be bound as index buffer
    Index,
    // Buffer will be bound as vertex attribute buffer
    Vertex,
    // Buffer will be bound as uniform buffer and read in defined shader stages
    Uniform(ShaderUsage),
    /// Buffer will be used as a readonly storage in defined shader stages
    StorageRead(ShaderUsage),
    /// Buffer will be used as a writeable storage in defined shader stages
    StorageWrite(ShaderUsage),
    /// Buffer will be used as a uniform texel buffer in defined shader stages
    UniformTexel(ShaderUsage),
    /// Buffer will be used as a readonly storage texel buffer in defined shader stages
    StorageTexelRead(ShaderUsage),
    /// Buffer will be used as a writeable storage texel buffer in defined shader stages
    StorageTexelWrite(ShaderUsage),
    /// Buffer will be used in a memory transfer as a source
    TransferRead,
    /// Buffer will be used in a memory transfer as a source
    TransferWrite,
    /// Custom buffer usage with manually provided access and pipeline stages
    Custom(NodeBufferAccess, PipelineStage),
}

impl BufferUsage {
    fn stage(&self) -> PipelineStage {
        match self {
            // TODO: not sure about attachment reads
            Self::IndirectCommand => PipelineStage::DRAW_INDIRECT,
            Self::Index => PipelineStage::VERTEX_INPUT,
            Self::Vertex => PipelineStage::VERTEX_INPUT,
            Self::Uniform(shader) => shader.stage(),
            Self::StorageRead(shader) => shader.stage(),
            Self::StorageWrite(shader) => shader.stage(),
            Self::UniformTexel(shader) => shader.stage(),
            Self::StorageTexelRead(shader) => shader.stage(),
            Self::StorageTexelWrite(shader) => shader.stage(),
            Self::TransferRead | Self::TransferWrite => PipelineStage::TRANSFER,
            Self::Custom(_, stage) => *stage,
        }
    }

    fn access(&self) -> NodeBufferAccess {
        match self {
            Self::IndirectCommand => NodeBufferAccess::INDIRECT_COMMAND_READ,
            Self::Index => NodeBufferAccess::INDEX_BUFFER_READ,
            Self::Vertex => NodeBufferAccess::VERTEX_BUFFER_READ,
            Self::Uniform(_) => NodeBufferAccess::UNIFORM_BUFFER_READ,
            Self::StorageRead(_) => NodeBufferAccess::STORAGE_BUFFER_READ,
            Self::StorageWrite(_) => NodeBufferAccess::STORAGE_BUFFER_WRITE,
            Self::UniformTexel(_) => NodeBufferAccess::UNIFORM_TEXEL_BUFFER_READ,
            Self::StorageTexelRead(_) => NodeBufferAccess::STORAGE_TEXEL_BUFFER_READ,
            Self::StorageTexelWrite(_) => NodeBufferAccess::STORAGE_TEXEL_BUFFER_WRITE,
            Self::TransferRead => NodeBufferAccess::TRANSFER_READ,
            Self::TransferWrite => NodeBufferAccess::TRANSFER_WRITE,
            Self::Custom(access, _) => *access,
        }
    }

    pub(crate) fn resource_usage(&self, id: BufferId) -> ResourceUsage {
        ResourceUsage::Buffer(id, self.access(), self.stage())
    }
}

impl std::ops::BitOr for BufferUsage {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        Self::Custom(self.access() | other.access(), self.stage() | other.stage())
    }
}
