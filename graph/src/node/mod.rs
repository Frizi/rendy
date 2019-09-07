//! Defines node - building block for framegraph.
//!

pub mod present;
pub mod render;

use {
    crate::{
        command::{Families, Family, FamilyId, Fence, Queue, Submission, Submittable},
        factory::{Factory, UploadError},
        frame::Frames,
        graph::GraphContext,
        util::{rendy_with_metal_backend, rendy_without_metal_backend},
        wsi::SwapchainError,
        BufferId, ImageId, NodeId,
    },
    gfx_hal::{queue::QueueFamilyId, Backend},
};

/// Buffer access node will perform.
/// Node must not perform any access to the buffer not specified in `access`.
/// All access must be between logically first and last `stages`.
#[derive(Clone, Copy, Debug)]
pub struct BufferAccess {
    /// Access flags.
    pub access: gfx_hal::buffer::Access,

    /// Intended usage flags for buffer.
    /// TODO: Could derive from access?
    pub usage: gfx_hal::buffer::Usage,

    /// Pipeline stages at which buffer is accessd.
    pub stages: gfx_hal::pso::PipelineStage,
}

/// Buffer pipeline barrier.
#[derive(Clone, Debug)]
pub struct BufferBarrier {
    /// State transition for the buffer.
    pub states: std::ops::Range<gfx_hal::buffer::State>,

    /// Stages at which buffer is accessd.
    pub stages: std::ops::Range<gfx_hal::pso::PipelineStage>,

    /// Transfer between families.
    pub families: Option<std::ops::Range<QueueFamilyId>>,
}

/// Buffer shared between nodes.
///
/// If Node doesn't actually use the buffer it can merge acquire and release barriers into one.
/// TODO: Make merge function.
#[derive(Clone, Debug)]
pub struct NodeBuffer {
    /// Id of the buffer.
    pub id: BufferId,

    /// Region of the buffer that is the transient resource.
    pub range: std::ops::Range<u64>,

    /// Acquire barrier.
    /// Node implementation must insert it before first command that uses the buffer.
    /// Barrier must be inserted even if this node doesn't use the buffer.
    pub acquire: Option<BufferBarrier>,

    /// Release barrier.
    /// Node implementation must insert it after last command that uses the buffer.
    /// Barrier must be inserted even if this node doesn't use the buffer.
    pub release: Option<BufferBarrier>,
}

/// Image access node wants to perform.
#[derive(Clone, Copy, Debug)]
pub struct ImageAccess {
    /// Access flags.
    pub access: gfx_hal::image::Access,

    /// Intended usage flags for image.
    /// TODO: Could derive from access?
    pub usage: gfx_hal::image::Usage,

    /// Preferred layout for access.
    /// Actual layout will be reported int `NodeImage`.
    /// Actual layout is guaranteed to support same operations.
    /// TODO: Could derive from access?
    pub layout: gfx_hal::image::Layout,

    /// Pipeline stages at which image is accessd.
    pub stages: gfx_hal::pso::PipelineStage,
}

/// Image pipeline barrier.
/// Node implementation must insert it before first command that uses the image.
/// Barrier must be inserted even if this node doesn't use the image.
#[derive(Clone, Debug)]
pub struct ImageBarrier {
    /// State transition for the image.
    pub states: std::ops::Range<gfx_hal::image::State>,

    /// Stages at which image is accessd.
    pub stages: std::ops::Range<gfx_hal::pso::PipelineStage>,

    /// Transfer between families.
    pub families: Option<std::ops::Range<QueueFamilyId>>,
}

/// Image shared between nodes.
#[derive(Clone, Debug)]
pub struct NodeImage {
    /// Id of the image.
    pub id: ImageId,

    /// Region of the image that is the transient resource.
    pub range: gfx_hal::image::SubresourceRange,

    /// Image state for node.
    pub layout: gfx_hal::image::Layout,

    /// Specify that node should clear image to this value.
    pub clear: Option<gfx_hal::command::ClearValue>,

    /// Acquire barrier.
    /// Node implementation must insert it before first command that uses the image.
    /// Barrier must be inserted even if this node doesn't use the image.
    pub acquire: Option<ImageBarrier>,

    /// Release barrier.
    /// Node implementation must insert it after last command that uses the image.
    /// Barrier must be inserted even if this node doesn't use the image.
    pub release: Option<ImageBarrier>,
}

/// Trait-object safe `Node`.
pub trait DynNode<B: Backend, T: ?Sized>: std::fmt::Debug + Send + Sync {
    /// Record commands required by node.
    /// Recorded buffers go into `submits`.
    unsafe fn run(&mut self, ctx: NodeContext<'_, B, T>);

    /// Dispose of the node.
    ///
    /// # Safety
    ///
    /// Must be called after waiting for device idle.
    unsafe fn dispose(self: Box<Self>, factory: &mut Factory<B>, aux: &T);
}

/// A context for rendergraph node execution. Contains all data that the node
/// get access to and contains ready-made methods for common operations.
#[derive(Debug)]
pub struct NodeContext<'a, B: Backend, T: ?Sized> {
    /// graph context
    pub graph_ctx: &'a GraphContext<B>,
    /// rendy Factory used by this graph
    pub factory: &'a Factory<B>,
    /// current rendering queue
    pub queue: &'a mut Queue<B>,
    /// user data provided to RenderGraph::run
    pub aux: &'a T,
    /// timeline of frames
    pub frames: &'a Frames<B>,
    /// semaphores to wait for on submission
    pub waits: smallvec::SmallVec<[(&'a B::Semaphore, gfx_hal::pso::PipelineStage); 16]>,
    /// semaphores to signal on submission
    pub signals: smallvec::SmallVec<[&'a B::Semaphore; 16]>,
    /// submitted fence
    pub fence: Option<&'a mut Fence<B>>,
}

impl<'a, B: Backend, T: ?Sized> NodeContext<'a, B, T> {
    /// Safety: Fence must be submitted
    pub unsafe fn submit<C>(&mut self, submits: C)
    where
        C: IntoIterator,
        C::Item: Submittable<B>,
    {
        self.queue.submit(
            Some(
                Submission::new()
                    .submits(submits)
                    .wait(self.waits.iter().cloned())
                    .signal(self.signals.iter().cloned()),
            ),
            self.fence.as_mut().map(|x| &mut **x),
        )
    }
}

/// Error building a node of the graph.
#[derive(Debug)]
pub enum NodeBuildError {
    /// Filed to uplaod the data.
    Upload(UploadError),
    /// Mismatched queue family.
    QueueFamily(FamilyId),
    /// Failed to create an imate view.
    View(gfx_hal::image::ViewError),
    /// Failed to create a pipeline.
    Pipeline(gfx_hal::pso::CreationError),
    /// Failed to create a swap chain.
    Swapchain(SwapchainError),
    /// Ran out of memory when creating something.
    OutOfMemory(gfx_hal::device::OutOfMemory),
}

/// Dynamic node builder that emits `DynNode`.
pub trait NodeBuilder<B: Backend, T: ?Sized>: std::fmt::Debug {
    /// Pick family for this node to be executed onto.
    fn family(&self, factory: &mut Factory<B>, families: &Families<B>) -> Option<FamilyId>;

    /// Get buffer accessed by the node.
    fn buffers(&self) -> Vec<(BufferId, BufferAccess)> {
        Vec::new()
    }

    /// Get images accessed by the node.
    fn images(&self) -> Vec<(ImageId, ImageAccess)> {
        Vec::new()
    }

    /// Indices of nodes this one dependes on.
    fn dependencies(&self) -> Vec<NodeId> {
        Vec::new()
    }

    /// Build node.
    fn build<'a>(
        self: Box<Self>,
        ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        family: &mut Family<B>,
        queue: usize,
        aux: &T,
        buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
    ) -> Result<Box<dyn DynNode<B, T>>, NodeBuildError>;
}

/// Builder for the node.
#[derive(derivative::Derivative)]
#[derivative(Debug(bound = "N: std::fmt::Debug"))]
pub struct DescBuilder<B: Backend, T: ?Sized, N> {
    desc: N,
    buffers: Vec<BufferId>,
    images: Vec<ImageId>,
    dependencies: Vec<NodeId>,
    marker: std::marker::PhantomData<fn(B, &T)>,
}

impl<B, T, N> DescBuilder<B, T, N>
where
    B: Backend,
    T: ?Sized,
{
    /// Create new builder out of desc
    pub fn new(desc: N) -> Self {
        DescBuilder {
            desc,
            buffers: Vec::new(),
            images: Vec::new(),
            dependencies: Vec::new(),
            marker: std::marker::PhantomData,
        }
    }
    /// Add buffer to the node.
    /// This method must be called for each buffer node uses.
    pub fn add_buffer(&mut self, buffer: BufferId) -> &mut Self {
        self.buffers.push(buffer);
        self
    }

    /// Add buffer to the node.
    /// This method must be called for each buffer node uses.
    pub fn with_buffer(mut self, buffer: BufferId) -> Self {
        self.add_buffer(buffer);
        self
    }

    /// Add image to the node.
    /// This method must be called for each image node uses.
    pub fn add_image(&mut self, image: ImageId) -> &mut Self {
        self.images.push(image);
        self
    }

    /// Add image to the node.
    /// This method must be called for each image node uses.
    pub fn with_image(mut self, image: ImageId) -> Self {
        self.add_image(image);
        self
    }

    /// Add dependency.
    /// Node will be placed after its dependencies.
    pub fn add_dependency(&mut self, dependency: NodeId) -> &mut Self {
        self.dependencies.push(dependency);
        self
    }

    /// Add dependency.
    /// Node will be placed after its dependencies.
    pub fn with_dependency(mut self, dependency: NodeId) -> Self {
        self.add_dependency(dependency);
        self
    }
}

/// Convert graph barriers into gfx barriers.
pub fn gfx_acquire_barriers<'a, 'b, B: Backend>(
    ctx: &'a GraphContext<B>,
    buffers: impl IntoIterator<Item = &'b NodeBuffer>,
    images: impl IntoIterator<Item = &'b NodeImage>,
) -> (
    std::ops::Range<gfx_hal::pso::PipelineStage>,
    Vec<gfx_hal::memory::Barrier<'a, B>>,
) {
    let mut bstart = gfx_hal::pso::PipelineStage::empty();
    let mut bend = gfx_hal::pso::PipelineStage::empty();

    let mut istart = gfx_hal::pso::PipelineStage::empty();
    let mut iend = gfx_hal::pso::PipelineStage::empty();

    let barriers: Vec<gfx_hal::memory::Barrier<'_, B>> = buffers
        .into_iter()
        .filter_map(|buffer| {
            buffer.acquire.as_ref().map(|acquire| {
                bstart |= acquire.stages.start;
                bend |= acquire.stages.end;

                gfx_hal::memory::Barrier::Buffer {
                    states: acquire.states.clone(),
                    families: acquire.families.clone(),
                    target: ctx
                        .get_buffer(buffer.id)
                        .expect("Buffer does not exist")
                        .raw(),
                    range: Some(buffer.range.start)..Some(buffer.range.end),
                }
            })
        })
        .chain(images.into_iter().filter_map(|image| {
            image.acquire.as_ref().map(|acquire| {
                istart |= acquire.stages.start;
                iend |= acquire.stages.end;

                gfx_hal::memory::Barrier::Image {
                    states: acquire.states.clone(),
                    families: acquire.families.clone(),
                    target: ctx.get_image(image.id).expect("Image does not exist").raw(),
                    range: image.range.clone(),
                }
            })
        }))
        .collect();

    (bstart | istart..bend | iend, barriers)
}

/// Convert graph barriers into gfx barriers.
pub fn gfx_release_barriers<'a, B: Backend>(
    ctx: &'a GraphContext<B>,
    buffers: impl IntoIterator<Item = &'a NodeBuffer>,
    images: impl IntoIterator<Item = &'a NodeImage>,
) -> (
    std::ops::Range<gfx_hal::pso::PipelineStage>,
    Vec<gfx_hal::memory::Barrier<'a, B>>,
) {
    let mut bstart = gfx_hal::pso::PipelineStage::empty();
    let mut bend = gfx_hal::pso::PipelineStage::empty();

    let mut istart = gfx_hal::pso::PipelineStage::empty();
    let mut iend = gfx_hal::pso::PipelineStage::empty();

    let barriers: Vec<gfx_hal::memory::Barrier<'_, B>> = buffers
        .into_iter()
        .filter_map(|buffer| {
            buffer.release.as_ref().map(|release| {
                bstart |= release.stages.start;
                bend |= release.stages.end;

                gfx_hal::memory::Barrier::Buffer {
                    states: release.states.clone(),
                    families: release.families.clone(),
                    target: ctx
                        .get_buffer(buffer.id)
                        .expect("Buffer does not exist")
                        .raw(),
                    range: Some(buffer.range.start)..Some(buffer.range.end),
                }
            })
        })
        .chain(images.into_iter().filter_map(|image| {
            image.release.as_ref().map(|release| {
                istart |= release.stages.start;
                iend |= release.stages.end;

                gfx_hal::memory::Barrier::Image {
                    states: release.states.clone(),
                    families: release.families.clone(),
                    target: ctx.get_image(image.id).expect("Image does not exist").raw(),
                    range: image.range.clone(),
                }
            })
        }))
        .collect();

    (bstart | istart..bend | iend, barriers)
}

rendy_with_metal_backend! {
    /// Check if backend is metal.
    pub fn is_metal<B: Backend>() -> bool {
        std::any::TypeId::of::<B>() == std::any::TypeId::of::<rendy_util::metal::Backend>()
    }
}

rendy_without_metal_backend! {
    /// Check if backend is metal.
    pub fn is_metal<B: Backend>() -> bool {
        false
    }
}
