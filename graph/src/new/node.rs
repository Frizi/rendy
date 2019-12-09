use {
    super::{
        graph::{ExecContext, ExecPassContext, NodeContext, NodeCtx},
        resources::{BufferId, ImageId, ResourceUsage},
    },
    crate::{
        command::{
            Capability, EitherSubmit, Families, Family, FamilyId, RenderPassContinue,
            SecondaryLevel,
        },
        factory::{Factory, UploadError},
        wsi::SwapchainError,
    },
    rendy_core::hal::{queue::QueueFamilyId, Backend},
    std::{any::Any, collections::HashMap, marker::PhantomData},
};

/// Id of the node in graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub(super) usize);

pub type ConstructResult<'run, N, B, T> = Result<
    (
        <<N as Node<B, T>>::Outputs as OutputList>::Data,
        NodeExecution<'run, B>,
    ),
    NodeConstructionError,
>;

/// Node is a basic building block of rendering graph.
pub trait Node<B: Backend, T: ?Sized>: std::fmt::Debug + Send + Sync {
    type Outputs: OutputList;

    /// Construction phase of node, during which the usage of all graph resources is declared.
    /// Returns a rendering job that is going to be scheduled for execution if anything reads resources the node have declared to write.
    fn construct<'run>(
        &'run mut self,
        ctx: &mut impl NodeCtx<'run, B>,
        aux: &'run T,
    ) -> ConstructResult<'run, Self, B, T>;

    /// Dispose of the node.
    /// Called after device idle
    unsafe fn dispose(self: Box<Self>, factory: &mut Factory<B>, aux: &T) {
        let _ = (factory, aux);
    }
}

/// Holds the output variable data of all constructed nodes.
#[derive(Debug, Default)]
pub struct OutputStore {
    outputs: HashMap<NodeId, Vec<Box<dyn Any>>>,
}

impl OutputStore {
    /// Create new output store
    pub fn new() -> Self {
        Default::default()
    }

    pub(crate) fn get<T: Any>(&self, param: Parameter<T>) -> Option<&T> {
        let ParameterId(node_id, idx) = param.0;
        self.outputs
            .get(&node_id)
            .and_then(|vec| vec.get(idx as usize))
            .and_then(|v| v.downcast_ref())
    }

    pub(crate) fn set_all(&mut self, node: NodeId, vals: impl Iterator<Item = Box<dyn Any>>) {
        let vec = self.outputs.entry(node).or_insert_with(|| Vec::new());
        vec.clear();
        vec.extend(vals);
    }
}

/// Trait-object safe `Node`.
pub(crate) trait DynNode<B: Backend, T: ?Sized>: std::fmt::Debug + Send + Sync {
    fn construct<'run>(
        &'run mut self,
        ctx: &mut NodeContext<'_, 'run, '_, B>,
        aux: &'run T,
    ) -> Result<NodeExecution<B>, NodeConstructionError>;

    /// # Safety
    ///
    /// Must be called after waiting for device idle.
    unsafe fn dispose(self: Box<Self>, factory: &mut Factory<B>, aux: &T);
}

impl<B: Backend, T: ?Sized, N: Node<B, T>> DynNode<B, T> for N {
    fn construct<'run>(
        &'run mut self,
        ctx: &mut NodeContext<'_, 'run, '_, B>,
        aux: &'run T,
    ) -> Result<NodeExecution<B>, NodeConstructionError> {
        let (outs, exec) = N::construct(self, ctx, aux)?;
        ctx.set_outputs(N::Outputs::iter(outs));
        Ok(exec)
    }

    unsafe fn dispose(self: Box<Self>, factory: &mut Factory<B>, aux: &T) {
        N::dispose(self, factory, aux)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParameterId(NodeId, u32);

pub struct Parameter<T: Any>(pub(super) ParameterId, PhantomData<T>);

impl<T: Any> Copy for Parameter<T> {}
impl<T: Any> Clone for Parameter<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Any> std::fmt::Debug for Parameter<T> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_tuple("Parameter").field(&self.0).finish()
    }
}

impl<T: Any> Eq for Parameter<T> {}
impl<T: Any> PartialEq for Parameter<T> {
    fn eq(&self, other: &Parameter<T>) -> bool {
        self.0 == other.0
    }
}

impl<T: Any> std::hash::Hash for Parameter<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

#[doc(hidden)]
#[allow(missing_debug_implementations, missing_copy_implementations)]
pub struct InternalUse(pub(super) ());

pub trait OutputList {
    type Data: Any;
    type Iter: Iterator<Item = Box<dyn Any>>;
    /// Get number of outputs.
    fn size() -> u32;

    /// Create the parameters. Intended only for internal use.
    fn instance(node_id: NodeId, i: u32, _internal: InternalUse) -> Self;

    /// Convert output data into iterator over dynamic types;
    fn iter(data: Self::Data) -> Self::Iter;
}

impl<T: Any> OutputList for Parameter<T> {
    type Data = T;
    type Iter = std::iter::Once<Box<dyn Any>>;
    fn size() -> u32 {
        1
    }
    fn instance(node_id: NodeId, i: u32, _internal: InternalUse) -> Self {
        Parameter(ParameterId(node_id, i), PhantomData)
    }
    fn iter(data: Self::Data) -> Self::Iter {
        std::iter::once(Box::new(data))
    }
}

impl OutputList for () {
    type Data = ();
    type Iter = std::iter::Empty<Box<dyn Any>>;
    fn size() -> u32 {
        0
    }
    fn instance(_node_id: NodeId, _i: u32, _internal: InternalUse) -> Self {
        ()
    }
    fn iter(_data: Self::Data) -> Self::Iter {
        std::iter::empty()
    }
}

// TODO: implement the OutputList with macro.

// macro_rules! recursive_iter {
//     (@value $first:expr, $($rest:expr),*) => { $first.chain(recursive_iter!(@value $($rest),*)) };
//     (@value $last:expr) => { $last };
//     (@type $first:ty, $($rest:ty),*) => { std::iter::Chain<$first, recursive_iter!(@type $($rest),*)> };
//     (@type $last:ty) => { $last };
// }

impl<A, B> OutputList for (A, B)
where
    A: OutputList,
    B: OutputList,
{
    type Data = (A::Data, B::Data);
    type Iter = std::iter::Chain<A::Iter, B::Iter>;
    fn size() -> u32 {
        A::size() + B::size()
    }
    fn instance(node_id: NodeId, i: u32, _internal: InternalUse) -> Self {
        let a = A::instance(node_id, i, InternalUse(()));
        let i = i + A::size();
        let b = B::instance(node_id, i, InternalUse(()));
        // let i = i + B::size();
        (a, b)
    }
    fn iter(data: Self::Data) -> Self::Iter {
        let (a, b) = data;
        let a = A::iter(a);
        let b = B::iter(b);
        a.chain(b)
    }
}

impl<A, B, C> OutputList for (A, B, C)
where
    A: OutputList,
    B: OutputList,
    C: OutputList,
{
    type Data = (A::Data, B::Data, C::Data);
    type Iter = std::iter::Chain<A::Iter, std::iter::Chain<B::Iter, C::Iter>>;
    fn size() -> u32 {
        A::size() + B::size() + C::size()
    }
    fn instance(node_id: NodeId, i: u32, _internal: InternalUse) -> Self {
        let a = A::instance(node_id, i, InternalUse(()));
        let i = i + A::size();
        let b = B::instance(node_id, i, InternalUse(()));
        let i = i + B::size();
        let c = C::instance(node_id, i, InternalUse(()));
        // let i = i + C::size();
        (a, b, c)
    }
    fn iter(data: Self::Data) -> Self::Iter {
        let (a, b, c) = data;
        let a = A::iter(a);
        let b = B::iter(b);
        let c = C::iter(c);
        a.chain(b.chain(c))
    }
}

#[derive(Debug, Clone, Copy)]
/// Error during node construction phase in the graph.
pub enum NodeConstructionError {
    /// Node tried to read a variable that was never written to.
    /// This can only happen when node that produces this variable have failed to construct.
    VariableReadFailed(ParameterId),
    /// Tried to declare usage of a resource in invalid way, e.g. single image was used
    /// as two attachments.
    InvalidResourceUsage(ResourceUsage, ResourceUsage),
}

#[derive(Debug, Clone, Copy)]
/// Error during node execution phase in the graph.
pub enum NodeExecutionError {
    /// Node tried to access an image that was not registered in construction phase.
    UnscheduledImage(ImageId),
    /// Node tried to access a buffer that was not registered in construction phase.
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
    View(rendy_core::hal::image::ViewError),
    /// Failed to create a pipeline.
    Pipeline(rendy_core::hal::pso::CreationError),
    /// Failed to create a swap chain.
    Swapchain(SwapchainError),
    /// Ran out of memory when creating something.
    OutOfMemory(rendy_core::hal::device::OutOfMemory),
}

impl From<rendy_core::hal::device::OutOfMemory> for NodeBuildError {
    fn from(err: rendy_core::hal::device::OutOfMemory) -> Self {
        Self::OutOfMemory(err)
    }
}

impl From<rendy_core::hal::pso::CreationError> for NodeBuildError {
    fn from(err: rendy_core::hal::pso::CreationError) -> Self {
        Self::Pipeline(err)
    }
}

impl From<rendy_core::hal::image::ViewError> for NodeBuildError {
    fn from(err: rendy_core::hal::image::ViewError) -> Self {
        Self::View(err)
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

pub(crate) trait DynNodeBuilder<B: Backend, T: ?Sized>: std::fmt::Debug {
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

pub type PassSubmit<'a, B> = EitherSubmit<'a, B, SecondaryLevel, RenderPassContinue>;

pub type ExecResult<T> = Result<T, NodeExecutionError>;
pub type PassFn<'run, B> =
    Box<dyn for<'a> FnOnce(ExecPassContext<'a, B>) -> ExecResult<PassSubmit<'run, B>> + 'run>;
pub type SubmissionFn<'run, B> =
    Box<dyn FnOnce(&mut ExecContext<'run, B>) -> ExecResult<()> + 'run>;
pub type PostSubmitFn<'run, B> =
    Box<dyn FnOnce(&mut ExecContext<'run, B>) -> ExecResult<()> + 'run>;

#[derive(Debug, Clone, Copy)]
pub enum ExecutionPhase {
    Default,
    Output,
}

pub enum NodeExecution<'n, B: Backend> {
    /// This node has no evaluation phase.
    /// The resource access rules still apply as if the evaluation phase had to take place.
    None,
    /// Node evaluated in a context of rendering pass.
    /// The attachment layout is determined from declared image accesses (based on read/write flags), in order of their declaration.
    /// Can only operate on secondary command buffers.
    RenderPass(PassFn<'n, B>),
    /// A general rendering node, returns submittable work.
    Submission(ExecutionPhase, SubmissionFn<'n, B>),
    /// A general rendering node, operate after submission have happened.
    PostSubmit(ExecutionPhase, PostSubmitFn<'n, B>),
}

impl<'n, B: Backend> std::fmt::Debug for NodeExecution<'n, B> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            NodeExecution::None => fmt.debug_tuple("None").finish(),
            NodeExecution::RenderPass(_) => fmt.debug_tuple("RenderPass").finish(),
            NodeExecution::Submission(phase, _) => {
                fmt.debug_tuple("Submission").field(&phase).finish()
            }
            NodeExecution::PostSubmit(phase, _) => {
                fmt.debug_tuple("PostSubmit").field(&phase).finish()
            }
        }
    }
}

impl<'run, B: Backend> NodeExecution<'run, B> {
    pub fn is_output(&self) -> bool {
        match self {
            NodeExecution::Submission(ExecutionPhase::Output, _) => true,
            NodeExecution::PostSubmit(ExecutionPhase::Output, _) => true,
            _ => false,
        }
    }

    pub fn pass<F>(pass_closure: F) -> Self
    where
        F: for<'a> FnOnce(ExecPassContext<'a, B>) -> ExecResult<PassSubmit<'run, B>> + 'run,
    {
        Self::RenderPass(Box::new(pass_closure))
    }

    pub fn submission<F>(general_closure: F) -> Self
    where
        F: FnOnce(&mut ExecContext<'run, B>) -> ExecResult<()> + 'run,
    {
        Self::Submission(ExecutionPhase::Default, Box::new(general_closure))
    }

    pub fn output_submission<F>(general_closure: F) -> Self
    where
        F: FnOnce(&mut ExecContext<'run, B>) -> ExecResult<()> + 'run,
    {
        Self::Submission(ExecutionPhase::Output, Box::new(general_closure))
    }

    pub fn post_submit<F>(general_closure: F) -> Self
    where
        F: FnOnce(&mut ExecContext<'run, B>) -> ExecResult<()> + 'run,
    {
        Self::PostSubmit(ExecutionPhase::Default, Box::new(general_closure))
    }

    pub fn output_post_submit<F>(general_closure: F) -> Self
    where
        F: FnOnce(&mut ExecContext<'run, B>) -> ExecResult<()> + 'run,
    {
        Self::PostSubmit(ExecutionPhase::Output, Box::new(general_closure))
    }
}

/// Image pipeline barrier.
/// Node implementation must insert it before first command that uses the image.
/// Barrier must be inserted even if this node doesn't use the image.
#[derive(Clone, Debug)]
pub struct ImageBarrier {
    /// State transition for the image.
    pub states: std::ops::Range<rendy_core::hal::image::State>,

    /// Stages at which image is accessd.
    pub stages: std::ops::Range<rendy_core::hal::pso::PipelineStage>,

    /// Transfer between families.
    pub families: Option<std::ops::Range<QueueFamilyId>>,
}

/// Buffer pipeline barrier.
/// Node implementation must insert it before first command that uses the buffer.
/// Barrier must be inserted even if this node doesn't use the buffer.
#[derive(Clone, Debug)]
pub struct BufferBarrier {
    /// State transition for the buffer.
    pub states: std::ops::Range<rendy_core::hal::buffer::State>,

    /// Stages at which buffer is accessd.
    pub stages: std::ops::Range<rendy_core::hal::pso::PipelineStage>,

    /// Transfer between families.
    pub families: Option<std::ops::Range<QueueFamilyId>>,
}

// TODO: remove/replace
// /// Convert graph barriers into gfx barriers.
// pub fn gfx_acquire_barriers<'a, B: Backend>(
//     buffers: impl IntoIterator<Item = &'a NodeBuffer<B>>,
//     images: impl IntoIterator<Item = &'a NodeImage<B>>,
// ) -> (
//     std::ops::Range<rendy_core::hal::pso::PipelineStage>,
//     Vec<rendy_core::hal::memory::Barrier<'a, B>>,
// ) {
//     let mut bstart = rendy_core::hal::pso::PipelineStage::empty();
//     let mut bend = rendy_core::hal::pso::PipelineStage::empty();

//     let mut istart = rendy_core::hal::pso::PipelineStage::empty();
//     let mut iend = rendy_core::hal::pso::PipelineStage::empty();

//     let barriers: Vec<rendy_core::hal::memory::Barrier<'_, B>> = buffers
//         .into_iter()
//         .filter_map(|buffer| {
//             buffer.acquire.as_ref().map(|acquire| {
//                 bstart |= acquire.stages.start;
//                 bend |= acquire.stages.end;

//                 rendy_core::hal::memory::Barrier::Buffer {
//                     states: acquire.states.clone(),
//                     families: acquire.families.clone(),
//                     target: buffer.buffer.raw(),
//                     range: Some(buffer.range.start)..Some(buffer.range.end),
//                 }
//             })
//         })
//         .chain(images.into_iter().filter_map(|image| {
//             image.acquire.as_ref().map(|acquire| {
//                 istart |= acquire.stages.start;
//                 iend |= acquire.stages.end;

//                 rendy_core::hal::memory::Barrier::Image {
//                     states: acquire.states.clone(),
//                     families: acquire.families.clone(),
//                     target: image.image.raw(),
//                     range: image.range.clone(),
//                 }
//             })
//         }))
//         .collect();

//     (bstart | istart..bend | iend, barriers)
// }

// /// Convert graph barriers into gfx barriers.
// pub fn gfx_release_barriers<'a, B: Backend>(
//     buffers: impl IntoIterator<Item = &'a NodeBuffer<B>>,
//     images: impl IntoIterator<Item = &'a NodeImage<B>>,
// ) -> (
//     std::ops::Range<rendy_core::hal::pso::PipelineStage>,
//     Vec<rendy_core::hal::memory::Barrier<'a, B>>,
// ) {
//     let mut bstart = rendy_core::hal::pso::PipelineStage::empty();
//     let mut bend = rendy_core::hal::pso::PipelineStage::empty();

//     let mut istart = rendy_core::hal::pso::PipelineStage::empty();
//     let mut iend = rendy_core::hal::pso::PipelineStage::empty();

//     let barriers: Vec<rendy_core::hal::memory::Barrier<'_, B>> = buffers
//         .into_iter()
//         .filter_map(|buffer| {
//             buffer.release.as_ref().map(|release| {
//                 bstart |= release.stages.start;
//                 bend |= release.stages.end;

//                 rendy_core::hal::memory::Barrier::Buffer {
//                     states: release.states.clone(),
//                     families: release.families.clone(),
//                     target: buffer.buffer.raw(),
//                     range: Some(buffer.range.start)..Some(buffer.range.end),
//                 }
//             })
//         })
//         .chain(images.into_iter().filter_map(|image| {
//             image.release.as_ref().map(|release| {
//                 istart |= release.stages.start;
//                 iend |= release.stages.end;

//                 rendy_core::hal::memory::Barrier::Image {
//                     states: release.states.clone(),
//                     families: release.families.clone(),
//                     target: image.image.raw(),
//                     range: image.range.clone(),
//                 }
//             })
//         }))
//         .collect();

//     (bstart | istart..bend | iend, barriers)
// }
