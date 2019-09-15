use {
    super::resources::{
        BufferId, ImageId, NodeBufferAccess, NodeImageAccess, NodeVirtualAccess, VirtualId,
    },
    crate::{
        command::{Capability, Families, Family, FamilyId, Fence, Queue, Submission, Submittable},
        factory::{Factory, UploadError},
        frame::Frames,
        resource::{Buffer, Handle, Image},
        wsi::SwapchainError,
    },
    gfx_hal::{queue::QueueFamilyId, Backend},
    std::{
        any::{Any, TypeId},
        collections::HashMap,
        marker::PhantomData,
    },
};

/// Id of the node in graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(usize);

/// Node is a basic building block of rendering graph.
pub trait Node<B: Backend, T: ?Sized>: std::fmt::Debug + Send + Sync {
    type Outputs: OutputList;

    /// Construction phase of node, during which the usage of all graph resources is declared.
    /// Returns a rendering job that is going to be scheduled for execution if anything reads resources the node have declared to write.
    fn construct(
        &mut self,
        ctx: NodeContext<'_, B>,
        aux: &T,
    ) -> Result<(Self::Outputs, NodeExecution<B, T>), NodeConstructionError>;

    /// Dispose of the node.
    /// Called after device idle
    fn dispose(self: Box<Self>, factory: &mut Factory<B>, aux: &T);
}

/// Holds the output variable data of all constructed nodes.
pub struct OutputStore {
    outputs: HashMap<NodeId, Vec<(TypeId, Box<dyn Any>)>>,
}

/// Trait-object safe `Node`.
pub trait DynNode<B: Backend, T: ?Sized>: std::fmt::Debug + Send + Sync {
    fn construct(
        &mut self,
        output_store: &mut OutputStore,
        ctx: NodeContext<'_, B>,
        aux: &T,
    ) -> Result<NodeExecution<B, T>, NodeConstructionError>;

    /// # Safety
    ///
    /// Must be called after waiting for device idle.
    unsafe fn dispose(self: Box<Self>, factory: &mut Factory<B>, aux: &T);
}

impl<B: Backend, T: ?Sized, N: Node<B, T>> DynNode<B, T> for N {
    fn construct(
        &mut self,
        _output_store: &mut OutputStore,
        ctx: NodeContext<'_, B>,
        aux: &T,
    ) -> Result<NodeExecution<B, T>, NodeConstructionError> {
        let (_outs, _exec) = N::construct(self, ctx, aux)?;
        unimplemented!()
        // output_store.store(outs);
        // Ok(exec)
    }

    unsafe fn dispose(self: Box<Self>, factory: &mut Factory<B>, aux: &T) {
        N::dispose(self, factory, aux)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParameterId(NodeId, u32);
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct Parameter<T>(ParameterId, PhantomData<T>);

pub trait OutputList {
    type Data;

    fn size() -> u32;
}

impl<T> OutputList for Parameter<T> {
    type Data = T;
    fn size() -> u32 {
        1
    }
}

impl OutputList for () {
    type Data = ();
    fn size() -> u32 {
        0
    }
}

impl<A, B> OutputList for (A, B)
where
    A: OutputList,
    B: OutputList,
{
    type Data = (A::Data, B::Data);
    fn size() -> u32 {
        A::size() + B::size()
    }
}

impl<A, B, C> OutputList for (A, B, C)
where
    A: OutputList,
    B: OutputList,
    C: OutputList,
{
    type Data = (A::Data, B::Data, C::Data);
    fn size() -> u32 {
        A::size() + B::size() + C::size()
    }
}

pub enum NodeConstructionError {
    VariableReadFailed(ParameterId),
}

pub enum NodeExecutionError {
    UnscheduledImage(ImageId),
    UnscheduledBuffer(BufferId),
}

/// Error building a node of the graph.
#[derive(Debug)]
pub enum NodeBuildError {
    /// Filed to uplaod the data.
    Upload(UploadError),
    /// Family type required by note was not found,
    MissingFamily,
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

impl From<gfx_hal::device::OutOfMemory> for NodeBuildError {
    fn from(err: gfx_hal::device::OutOfMemory) -> Self {
        Self::OutOfMemory(err)
    }
}

pub trait NodeBuilder<B: Backend, T: ?Sized>: std::fmt::Debug + Sized {
    type Node: Node<B, T> + 'static;
    type Family: FamilySelector<B, T, Self>;

    fn build(
        self: Box<Self>,
        factory: &mut Factory<B>,
        family: &mut Family<B>,
        aux: &T,
    ) -> Result<Self::Node, NodeBuildError>;
}

pub trait FamilySelector<B: Backend, T: ?Sized, N: NodeBuilder<B, T>>: std::fmt::Debug {
    fn family(
        builder: &mut N,
        factory: &mut Factory<B>,
        families: &Families<B>,
    ) -> Option<FamilyId>;
}

impl<B, T, N, C> FamilySelector<B, T, N> for C
where
    B: Backend,
    T: ?Sized,
    N: NodeBuilder<B, T>,
    C: Capability,
{
    fn family(_builder: &mut N, _: &mut Factory<B>, families: &Families<B>) -> Option<FamilyId> {
        families.with_capability::<C>()
    }
}

pub trait DynNodeBuilder<B: Backend, T: ?Sized> {
    fn num_outputs(&self) -> u32;
    fn build(
        self: Box<Self>,
        factory: &mut Factory<B>,
        families: &mut Families<B>,
        aux: &T,
    ) -> Result<Box<dyn DynNode<B, T>>, NodeBuildError>;
}

impl<B: Backend, T: ?Sized, N: NodeBuilder<B, T>> DynNodeBuilder<B, T> for N {
    fn num_outputs(&self) -> u32 {
        <N::Node as Node<B, T>>::Outputs::size()
    }
    fn build(
        mut self: Box<Self>,
        factory: &mut Factory<B>,
        families: &mut Families<B>,
        aux: &T,
    ) -> Result<Box<dyn DynNode<B, T>>, NodeBuildError> {
        let family_id =
            N::Family::family(&mut self, factory, families).ok_or(NodeBuildError::MissingFamily)?;
        Ok(Box::new(N::build(
            self,
            factory,
            families.family_mut(family_id),
            aux,
        )?))
    }
}

pub type ExecResult = Result<(), NodeExecutionError>;

pub enum NodeExecution<'n, B: Backend, T: ?Sized> {
    RenderPass(Box<dyn for<'a> FnOnce(ExecPassContext<'a, B>, &'a T) -> ExecResult + 'n>),
    General(Box<dyn for<'a> FnOnce(ExecContext<'a, B>, &'a T) -> ExecResult + 'n>),
}

impl<'n, B: Backend, T: ?Sized> std::fmt::Debug for NodeExecution<'n, B, T> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_tuple(match &self {
            NodeExecution::RenderPass(_) => "RenderPass",
            NodeExecution::General(_) => "General",
        })
        .field(&format_args!(".."))
        .finish()
    }
}

impl<'n, B: Backend, T: ?Sized> NodeExecution<'n, B, T> {
    pub fn pass<F>(pass_closure: F) -> Self
    where
        F: for<'a> FnOnce(ExecPassContext<'a, B>, &'a T) -> ExecResult + 'n,
    {
        Self::RenderPass(Box::new(pass_closure))
    }

    pub fn general<F>(general_closure: F) -> Self
    where
        F: for<'a> FnOnce(ExecContext<'a, B>, &'a T) -> ExecResult + 'n,
    {
        Self::General(Box::new(general_closure))
    }
}

// pub trait NodeExecution<'a, B: Backend, T: ?Sized>: std::fmt::Debug + Send + Sync {
//     /// Record commands required by node.
//     /// Recorded buffers go into `submits`.
//     unsafe fn run(self: Box<Self>, ctx: ExecContext<'_, B>, aux: &T) -> ExecResult;
// }

/// A context for rendergraph node construction phase. Contains all data that the node
/// get access to and contains ready-made methods for common operations.
#[derive(Debug)]
pub struct NodeContext<'a, B: Backend> {
    factory: &'a Factory<B>,
    images: Vec<(ImageId, NodeImageAccess)>,
    buffers: Vec<(BufferId, NodeBufferAccess)>,
    virtuals: Vec<(VirtualId, NodeVirtualAccess)>,
}

impl<'a, B: Backend> NodeContext<'a, B> {
    pub fn get_parameter<T>(&self, _id: Parameter<T>) -> Result<&T, NodeConstructionError> {
        unimplemented!()
        // Err(NodeConstructionError::VariableReadFailed(id.0))
    }

    /// Create new image owned by graph.
    pub fn create_image(
        &mut self,
        _kind: gfx_hal::image::Kind,
        _levels: gfx_hal::image::Level,
        _format: gfx_hal::format::Format,
    ) -> ImageId {
        unimplemented!()
    }

    /// Create new buffer owned by graph.
    pub fn create_buffer(&mut self, _size: u64) -> BufferId {
        unimplemented!()
    }

    /// Create non-data dependency target. A virtual resource intended to
    /// describe dependencies between rendering nodes without carrying any data.
    pub fn create_virtual(&mut self) -> VirtualId {
        unimplemented!()
    }

    pub fn use_virtual(&mut self, id: VirtualId, access: NodeVirtualAccess) {
        self.virtuals.push((id, access));
    }

    /// Declare usage of image by the node
    pub fn use_image(&mut self, id: ImageId, access: NodeImageAccess) {
        self.images.push((id, access));
    }

    /// Declare usage of buffer by the node
    pub fn use_buffer(&mut self, id: BufferId, access: NodeBufferAccess) {
        self.buffers.push((id, access));
    }
}

pub struct ExecContext<'a, B: Backend> {
    factory: &'a Factory<B>,
    images: HashMap<ImageId, NodeImage<B>>,
    buffers: HashMap<BufferId, NodeBuffer<B>>,
    queue: &'a mut Queue<B>,
    frames: &'a Frames<B>,
    waits: smallvec::SmallVec<[(&'a B::Semaphore, gfx_hal::pso::PipelineStage); 16]>,
    signals: smallvec::SmallVec<[&'a B::Semaphore; 16]>,
    fence: Option<&'a mut Fence<B>>,
}

impl<'a, B: Backend> ExecContext<'a, B> {
    pub fn submit<C>(&mut self, submits: C)
    where
        C: IntoIterator,
        C::Item: Submittable<B>,
    {
        unsafe {
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

    pub fn image(&self, id: ImageId) -> Result<&NodeImage<B>, NodeExecutionError> {
        self.images
            .get(&id)
            .ok_or(NodeExecutionError::UnscheduledImage(id))
    }

    pub fn buffer(&self, id: BufferId) -> Result<&NodeBuffer<B>, NodeExecutionError> {
        self.buffers
            .get(&id)
            .ok_or(NodeExecutionError::UnscheduledBuffer(id))
    }

    pub fn frames(&self) -> &Frames<B> {
        self.frames
    }
}

pub struct ExecPassContext<'a, B: Backend> {
    general_ctx: ExecContext<'a, B>,
}

impl<'a, B: Backend> ExecContext<'a, B> {}

// struct PassData {

// }

//     ctx,
//     factory,
//     QueueId {
//         family: family.id(),
//         index: queue,
//     },
//     aux,
//     framebuffer_width,
//     framebuffer_height,
//     gfx_hal::pass::Subpass {
//         index,
//         main_pass: &render_pass,
//     },
//     buffers,
//     images,

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

/// Buffer pipeline barrier.
/// Node implementation must insert it before first command that uses the buffer.
/// Barrier must be inserted even if this node doesn't use the buffer.
#[derive(Clone, Debug)]
pub struct BufferBarrier {
    /// State transition for the buffer.
    pub states: std::ops::Range<gfx_hal::buffer::State>,

    /// Stages at which buffer is accessd.
    pub stages: std::ops::Range<gfx_hal::pso::PipelineStage>,

    /// Transfer between families.
    pub families: Option<std::ops::Range<QueueFamilyId>>,
}

/// Image shared between nodes.
#[derive(Clone, Debug)]
pub struct NodeImage<B: Backend> {
    /// The actual image object.
    pub image: Handle<Image<B>>,

    /// Region of the image that is the transient resource.
    pub range: gfx_hal::image::SubresourceRange,

    /// Image state for node.
    pub layout: gfx_hal::image::Layout,

    /// Acquire barrier.
    /// Node implementation must insert it before first command that uses the image.
    /// Barrier must be inserted even if this node doesn't use the image.
    pub acquire: Option<ImageBarrier>,

    /// Release barrier.
    /// Node implementation must insert it after last command that uses the image.
    /// Barrier must be inserted even if this node doesn't use the image.
    pub release: Option<ImageBarrier>,
}

/// Buffer shared between nodes.
///
/// If Node doesn't actually use the buffer it can merge acquire and release barriers into one.
#[derive(Clone, Debug)]
pub struct NodeBuffer<B: Backend> {
    /// The actual buffer object.
    pub buffer: Handle<Buffer<B>>,

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

/// Convert graph barriers into gfx barriers.
pub fn gfx_acquire_barriers<'a, B: Backend>(
    buffers: impl IntoIterator<Item = &'a NodeBuffer<B>>,
    images: impl IntoIterator<Item = &'a NodeImage<B>>,
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
                    target: buffer.buffer.raw(),
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
                    target: image.image.raw(),
                    range: image.range.clone(),
                }
            })
        }))
        .collect();

    (bstart | istart..bend | iend, barriers)
}

/// Convert graph barriers into gfx barriers.
pub fn gfx_release_barriers<'a, B: Backend>(
    buffers: impl IntoIterator<Item = &'a NodeBuffer<B>>,
    images: impl IntoIterator<Item = &'a NodeImage<B>>,
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
                    target: buffer.buffer.raw(),
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
                    target: image.image.raw(),
                    range: image.range.clone(),
                }
            })
        }))
        .collect();

    (bstart | istart..bend | iend, barriers)
}
