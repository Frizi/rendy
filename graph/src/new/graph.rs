use {
    super::{
        node::{
            DynNode, DynNodeBuilder, ExecutionPhase, InternalUse, Node, NodeBuffer, NodeBuildError,
            NodeBuilder, NodeConstructionError, NodeExecution, NodeExecutionError, NodeId,
            NodeImage, OutputList, OutputStore, Parameter, PassFn, PostSubmitFn, SubmissionFn,
        },
        pipeline::Pipeline,
        resources::{
            AttachmentAccess, BufferId, BufferInfo, BufferUsage, ImageId, ImageInfo, ImageLoad,
            ImageUsage, NodeBufferAccess, NodeImageAccess, NodeVirtualAccess, ResourceId,
            ResourceUsage, VirtualId, WaitId,
        },
        walker::{GraphItem, TopoWithEdges},
    },
    crate::{
        command::{
            CommandBuffer, CommandPool, Compute, EitherSubmit, Families, Family, General, Graphics,
            IndividualReset, InitialState, Level, Queue, QueueId, QueueType, RenderPassEncoder,
            Supports,
        },
        factory::Factory,
        frame::{Fences, Frame, Frames},
        resource::{Buffer, Handle, Image},
    },
    graphy::{EdgeIndex, GraphAllocator, NodeIndex, Walker},
    rendy_core::{
        device_owned,
        hal::{pass::ATTACHMENT_UNUSED, Backend},
        DeviceId,
    },
    smallvec::{smallvec, SmallVec},
    std::{any::Any, collections::HashMap, marker::PhantomData, ops::Range},
};
/// A builder type for rendering graph.
pub struct GraphBuilder<B: Backend, T: ?Sized> {
    nodes: Vec<Box<dyn DynNodeBuilder<B, T>>>,
    frames_in_flight: u32,
}

impl<B: Backend, T: ?Sized> Default for GraphBuilder<B, T> {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            frames_in_flight: 2,
        }
    }
}

impl<B: Backend, T: ?Sized> std::fmt::Debug for GraphBuilder<B, T> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("GraphBuilder")
            .field("nodes", &self.nodes)
            .field("frames_in_flight", &self.frames_in_flight)
            .finish()
    }
}

/// Error that happened during graph building.
#[derive(Debug)]
pub enum GraphBuildError {
    /// Error that happened during node building.
    NodeBuildError(NodeId, NodeBuildError),
}

impl<B: Backend, T: ?Sized> GraphBuilder<B, T> {
    /// Create new `GraphBuilder`
    pub fn new() -> Self {
        GraphBuilder::default()
    }
    /// Add a node to `GraphBuilder`.
    pub fn add<N: NodeBuilder<B, T> + 'static>(
        &mut self,
        builder: N,
    ) -> <N::Node as Node<B, T>>::Outputs {
        let node_id = NodeId(self.nodes.len());
        self.nodes.push(Box::new(builder));
        OutputList::instance(node_id, 0, InternalUse(()))
    }

    /// Choose number of frames in flight for the graph
    pub fn with_frames_in_flight(mut self, frames_in_flight: u32) -> Self {
        self.frames_in_flight = frames_in_flight;
        self
    }

    /// Build rendering graph
    pub fn build(
        self,
        factory: &mut Factory<B>,
        families: &mut Families<B>,
        aux: &T,
    ) -> Result<Graph<B, T>, GraphBuildError> {
        Ok(Graph {
            device: factory.device().id(),
            pipeline: Pipeline::new(),
            nodes: GraphNodes(
                self.nodes
                    .into_iter()
                    .enumerate()
                    .map(|(i, builder)| {
                        builder
                            .build(factory, families, aux)
                            .map_err(|e| GraphBuildError::NodeBuildError(NodeId(i), e))
                    })
                    .collect::<Result<_, GraphBuildError>>()?,
            ),
            alloc: GraphAllocator::with_capacity(52_428_800),
            frames: Frames::new(),
            resources: GraphResourcePool::new(factory, families, self.frames_in_flight),
            inflight: self.frames_in_flight,
        })
    }
}

// A type of family to operate on in the graph
#[derive(Debug, Clone, Copy)]
pub enum FamilyType {
    Graphics,
    Compute,
    AsyncCompute,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RenderPassInfo {}

#[derive(Debug)]
struct GraphResourcePool<B: Backend> {
    // TODO: allocate, pool and recycle semaphores
    semaphores: Vec<B::Semaphore>,
    fences: Vec<Fences<B>>,
    render_passes: HashMap<RenderPassInfo, Vec<B::RenderPass>>,
    // general family might not be present, as opengl might not handle compute
    // This should always be available on conformant vulkan implementation,
    general_family: Option<FamilyResources<B>>,
    graphics_family: Option<FamilyResources<B>>,
    // @Incomplete: maybe handle dedicated transfer queue on the graph
    // transfer_family: Option<FamilyResources<B, Transfer>>,
    async_compute_family: Option<FamilyResources<B>>,
}

impl<B: Backend> GraphResourcePool<B> {
    fn new(factory: &mut Factory<B>, families: &mut Families<B>, frames_in_flight: u32) -> Self {
        // TODO: ensure that primary graphics family support presentation

        let general_family = families.with_capability::<General>();
        // select graphics-only family if general is not available (webgl case)
        let graphics_family = match general_family {
            None => families.with_capability::<Graphics>(),
            _ => None,
        };

        // some graphics family must be available
        assert!(general_family.is_some() || graphics_family.is_some());

        let async_compute_family =
            // find dedicated compute-only family
            families.find(|family| family.capability() == QueueType::Compute)
            .or_else(|| {
                // find compute-capable family different from other families
                families.find(|family| {
                    Some(family.id()) != general_family &&
                    Some(family.id()) != graphics_family &&
                    Supports::<Compute>::supports(&family.capability()).is_some()
                })
            });

        // TODO(braindump):
        // family ids are chosen now (still validate vkGetPhysicalDeviceSurfaceSupportKHR)
        // so acquire them one by one and create resources for them (FamilyResources::new)
        // then finish command buffer pool management and stuff
        // then manage semaphore pool
        // then finally fences
        // fences are only used directly by the graph on submissions.
        // semaphores are accessible for nodes somewhat, so we need to manage them.

        Self {
            semaphores: Vec::new(),
            fences: Vec::new(),
            render_passes: HashMap::new(),
            general_family: general_family
                .map(|id| FamilyResources::new(factory, families.family_mut(id), frames_in_flight)),
            graphics_family: graphics_family
                .map(|id| FamilyResources::new(factory, families.family_mut(id), frames_in_flight)),
            async_compute_family: async_compute_family
                .map(|id| FamilyResources::new(factory, families.family_mut(id), frames_in_flight)),
        }
    }

    // fn request_render_pass(&mut self, info: RenderPassInfo, compatible: bool) &B::RenderPass
    fn request_render_pass(
        &mut self,
        factory: &Factory<B>,
        pass_info: RenderPassInfo,
    ) -> &B::RenderPass {
        // let attachments = unimplemented!();
        // let subpasses = unimplemented!();
        // let deps = unimplemented!();
        // factory
        //     .device()
        //     .create_render_pass(attachments, subpasses, deps);
        unimplemented!()
    }

    fn family_data_mut(&mut self, family_type: FamilyType) -> Option<&mut FamilyResources<B>> {
        match family_type {
            FamilyType::Graphics => self
                .general_family
                .as_mut()
                .or(self.graphics_family.as_mut()),
            FamilyType::Compute => self
                .general_family
                .as_mut()
                .or(self.async_compute_family.as_mut()),
            FamilyType::AsyncCompute => self
                .async_compute_family
                .as_mut()
                .or(self.general_family.as_mut()),
        }
    }

    fn family_data(&self, family_type: FamilyType) -> Option<&FamilyResources<B>> {
        match family_type {
            FamilyType::Graphics => self
                .general_family
                .as_ref()
                .or(self.graphics_family.as_ref()),
            FamilyType::Compute => self
                .general_family
                .as_ref()
                .or(self.async_compute_family.as_ref()),
            FamilyType::AsyncCompute => self
                .async_compute_family
                .as_ref()
                .or(self.general_family.as_ref()),
        }
    }

    fn allocate_buffers<L: Level>(
        &mut self,
        family_type: FamilyType,
        count: usize,
    ) -> Vec<CommandBuffer<B, QueueType, InitialState, L, IndividualReset>> {
        self.family_data_mut(family_type)
            .unwrap_or_else(|| panic!("Family matching type {:?} not found", family_type))
            .pool
            .allocate_buffers(count)
    }

    fn queue(&self, family_type: FamilyType) -> QueueId {
        self.family_data(family_type)
            .unwrap_or_else(|| panic!("Family matching type {:?} not found", family_type))
            .queue_id
    }
}

#[derive(Debug)]
struct FamilyResources<B: Backend> {
    // @Incomplete: currently we use only one queue per family type
    queue_id: QueueId,
    // @Speed: use specialized non-reset pool for transient command buffers.
    // That will also require some per-frame state cleanup and fence synchronization
    pool: CommandPool<B, QueueType, IndividualReset>,
}

impl<B: Backend> FamilyResources<B> {
    fn new(factory: &mut Factory<B>, family: &mut Family<B>, frames_in_flight: u32) -> Self {
        // @Incomplete: Use frame_in_flights to preallocate per-frame per-family resources
        let _ = frames_in_flight;

        Self {
            queue_id: QueueId {
                family: family.id(),
                index: 0,
            },
            pool: factory
                .create_command_pool(family)
                .expect("Failed to initialize family command pool"),
        }
    }
}

/// A built runnable top-level rendering graph.
pub struct Graph<B: Backend, T: ?Sized> {
    device: DeviceId,
    nodes: GraphNodes<B, T>,
    pipeline: Pipeline<B>,
    alloc: GraphAllocator,
    frames: Frames<B>,
    resources: GraphResourcePool<B>,
    inflight: u32,
}

device_owned!(Graph<B, T: ?Sized>);

impl<B: Backend, T: ?Sized> std::fmt::Debug for Graph<B, T> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("Graph")
            .field("device", &self.device)
            .field("nodes", &self.nodes)
            .field("pipeline", &self.pipeline)
            .field("alloc", &self.alloc)
            .field("frames", &self.frames)
            .field("resources", &self.resources)
            .field("inflight", &self.inflight)
            .finish()
    }
}

struct GraphNodes<B: Backend, T: ?Sized>(Vec<Box<dyn DynNode<B, T>>>);

impl<B: Backend, T: ?Sized> std::fmt::Debug for GraphNodes<B, T> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_list().entries(self.0.iter()).finish()
    }
}

impl<B: Backend, T: ?Sized> Graph<B, T> {
    /// Construct, schedule and run all nodes of rendering graph.
    pub fn run(
        &mut self,
        factory: &mut Factory<B>,
        families: &mut Families<B>,
        aux: &T,
    ) -> Result<(), GraphRunError> {
        self.assert_device_owner(factory.device());

        unsafe {
            self.alloc.reset();
        }

        let nodes = &mut self.nodes;
        let mut run_ctx = ConstructContext::new(
            factory,
            families,
            &mut self.frames,
            &mut self.resources,
            &self.alloc,
        );
        nodes.run_construction_phase(&mut run_ctx, aux)?;

        self.pipeline.reduce(&mut run_ctx.graph.dag, &self.alloc);

        if run_ctx.frames.next().index() >= self.inflight as _ {
            let wait = Frame::with_index(run_ctx.frames.next().index() - self.inflight as u64);
            let res_fences = &mut run_ctx.resources.fences;
            run_ctx
                .frames
                .wait_complete(wait, factory, |mut frame_fences| {
                    factory.reset_fences(&mut frame_fences).unwrap();
                    res_fences.push(frame_fences);
                });
        }

        // TODO: provide fences and semaphores
        let fences = run_ctx.resources.fences.pop().unwrap_or_default();
        let fences_used = 0;
        // let semaphores = &run_ctx.resources.semaphores;

        run_ctx.run_execution_phase()?;

        self.resources.fences.truncate(fences_used);
        self.frames.advance(fences);
        Ok(())
    }
}

impl<B: Backend, T: ?Sized> GraphNodes<B, T> {
    #[inline(never)]
    fn run_construction_phase<'run, 'arena>(
        &'run mut self,
        run_ctx: &mut ConstructContext<'run, 'arena, B>,
        aux: &'run T,
    ) -> Result<(), GraphRunError> {
        // insert all nodes in their original order, except outputs that are inserted last
        let mut outputs = SmallVec::<[_; 8]>::new();

        for (i, node) in self.0.iter_mut().enumerate() {
            let mut seed = run_ctx.graph.seed();

            let execution = {
                let mut ctx = NodeContext::new(NodeId(i), &mut seed, run_ctx);
                node.construct(&mut ctx, aux)
                    .map_err(|e| GraphRunError::NodeConstruction(NodeId(i), e))?
            };

            if execution.is_output() {
                outputs.push((i, seed, execution));
            } else {
                run_ctx
                    .graph
                    .insert(NodeId(i), seed, execution)
                    .map_err(|e| GraphRunError::NodeConstruction(NodeId(i), e))?;
            }
        }

        for (i, seed, execution) in outputs.drain().rev() {
            run_ctx
                .graph
                .insert(NodeId(i), seed, execution)
                .map_err(|e| GraphRunError::NodeConstruction(NodeId(i), e))?;
        }

        Ok(())
    }
}

/// A context for rendergraph node construction phase. Contains all data that the node
/// get access to and contains ready-made methods for common operations.
#[derive(Debug)]
pub(crate) struct NodeContext<'ctx, 'run, 'arena, B: Backend> {
    id: NodeId,
    seed: &'ctx mut NodeSeed,
    run: &'ctx mut ConstructContext<'run, 'arena, B>,
    next_image_id: usize,
    next_buffer_id: usize,
}

/// A token that allows an image to be used in node execution
#[derive(Debug, Clone, Copy)]
pub struct ImageToken<'run>(ImageId, PhantomData<&'run ()>);

/// A token that allows a buffer to be used in node execution
#[derive(Debug, Clone, Copy)]
pub struct BufferToken<'run>(BufferId, PhantomData<&'run ()>);

/// A token that allows a waited semaphore to be used in node execution.
///
/// Semaphore must be used during the execution.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct WaitSemaphoreToken<'run>(WaitId, PhantomData<&'run ()>);

impl ImageToken<'_> {
    fn new(id: ImageId) -> Self {
        Self(id, PhantomData)
    }
    pub(crate) fn id(&self) -> ImageId {
        self.0
    }
}

impl BufferToken<'_> {
    fn new(id: BufferId) -> Self {
        Self(id, PhantomData)
    }
    pub(crate) fn id(&self) -> BufferId {
        self.0
    }
}

impl WaitSemaphoreToken<'_> {
    fn new(id: WaitId) -> Self {
        Self(id, PhantomData)
    }
    pub(crate) fn id(&self) -> WaitId {
        self.0
    }
}

pub trait NodeCtx<'run, B: Backend> {
    fn factory(&self) -> &'run Factory<B>;

    fn get_parameter<P: Any>(&self, id: Parameter<P>) -> Result<&P, NodeConstructionError>;

    /// Create new image owned by graph.
    fn create_image(&mut self, image_info: ImageInfo) -> ImageId;

    /// Provide node owned image into the graph for single frame.
    /// Provide `acquire` semaphore that must be waited on before the image is first accessed on any queue.
    fn provide_image(
        &mut self,
        image_info: ImageInfo,
        image: Handle<Image<B>>,
        acquire: Option<B::Semaphore>,
    ) -> ImageId;

    /// Provide node owned buffer into the graph for single frame.
    fn provide_buffer(&mut self, buffer_info: BufferInfo, buffer: Handle<Buffer<B>>) -> BufferId;

    /// Create new buffer owned by graph.
    fn create_buffer(&mut self, buffer_info: BufferInfo) -> BufferId;

    /// Create non-data dependency target. A virtual resource intended to
    /// describe dependencies between rendering nodes without carrying any data.
    fn create_virtual(&mut self) -> VirtualId;

    /// Declare usage of virtual resource by the node.
    fn use_virtual(
        &mut self,
        id: VirtualId,
        access: NodeVirtualAccess,
    ) -> Result<(), NodeConstructionError>;

    /// Declare usage of image by the node.
    fn use_image(
        &mut self,
        id: ImageId,
        usage: ImageUsage,
    ) -> Result<ImageToken<'run>, NodeConstructionError>;

    /// Declare usage of buffer by the node.
    fn use_buffer(
        &mut self,
        id: BufferId,
        usage: BufferUsage,
    ) -> Result<BufferToken<'run>, NodeConstructionError>;

    /// Declare usage of a color attachment in render pass node.
    ///
    /// This is mutually exclusive with `use_image` calls on the same image,
    /// and will cause construction error when non-renderpass execution is returned.
    fn use_color(&mut self, index: usize, image: ImageId) -> Result<(), NodeConstructionError>;
    /// Declare usage of a depth/stencil attachment in render pass node.
    /// Depth attachment access can be read-only when depth write is never enabled in this pass.
    ///
    /// This is mutually exclusive with `use_image` calls on the same image,
    /// and will cause construction error when non-renderpass execution is returned.
    fn use_depth(
        &mut self,
        image: ImageId,
        write_access: bool,
    ) -> Result<(), NodeConstructionError>;
    /// Declare usage of an input attachment in render pass node.
    /// Input attachment access is always read-only and limited to fragment shaders.
    ///
    /// This is mutually exclusive with `use_image` calls on the same image,
    /// and will cause construction error when non-renderpass execution is returned.
    fn use_input(&mut self, index: usize, image: ImageId) -> Result<(), NodeConstructionError>;

    fn wait_semaphore(
        &mut self,
        stages: rendy_core::hal::pso::PipelineStage,
    ) -> WaitSemaphoreToken<'run>;
}

impl<'ctx, 'run, 'arena, B: Backend> NodeContext<'ctx, 'run, 'arena, B> {
    fn new(
        id: NodeId,
        seed: &'ctx mut NodeSeed,
        run: &'ctx mut ConstructContext<'run, 'arena, B>,
    ) -> Self {
        Self {
            id,
            seed,
            run,
            next_image_id: 0,
            next_buffer_id: 0,
        }
    }

    pub(crate) fn set_outputs(&mut self, vals: impl Iterator<Item = Box<dyn Any>>) {
        self.run.output_store.set_all(self.id, vals);
    }
}

impl<'run, B: Backend> NodeCtx<'run, B> for NodeContext<'_, 'run, '_, B> {
    fn factory(&self) -> &'run Factory<B> {
        self.run.factory
    }

    fn get_parameter<P: Any>(&self, id: Parameter<P>) -> Result<&P, NodeConstructionError> {
        self.run
            .output_store
            .get(id)
            .ok_or(NodeConstructionError::VariableReadFailed(id.0))
    }

    fn create_image(&mut self, image_info: ImageInfo) -> ImageId {
        let id = ImageId(self.id, self.next_image_id);
        self.next_image_id += 1;
        self.run.graph.create_image(id, image_info);
        id
    }

    fn create_buffer(&mut self, buffer_info: BufferInfo) -> BufferId {
        let id = BufferId(self.id, self.next_buffer_id);
        self.next_buffer_id += 1;
        self.run.graph.create_buffer(id, buffer_info);
        id
    }

    fn provide_image(
        &mut self,
        image_info: ImageInfo,
        image: Handle<Image<B>>,
        acquire: Option<B::Semaphore>,
    ) -> ImageId {
        let id = self.create_image(image_info);
        self.run.provide_image(id, image, acquire);
        id
    }

    fn provide_buffer(&mut self, buffer_info: BufferInfo, buffer: Handle<Buffer<B>>) -> BufferId {
        let id = self.create_buffer(buffer_info);
        self.run.provide_buffer(id, buffer);
        id
    }

    fn create_virtual(&mut self) -> VirtualId {
        self.run.graph.create_virtual()
    }

    fn use_virtual(
        &mut self,
        id: VirtualId,
        access: NodeVirtualAccess,
    ) -> Result<(), NodeConstructionError> {
        self.run
            .graph
            .use_resource(self.seed, ResourceUsage::Virtual(id, access))
    }

    fn use_image(
        &mut self,
        id: ImageId,
        usage: ImageUsage,
    ) -> Result<ImageToken<'run>, NodeConstructionError> {
        self.run
            .graph
            .use_resource(self.seed, usage.resource_usage(id))?;
        Ok(ImageToken::new(id))
    }

    fn use_buffer(
        &mut self,
        id: BufferId,
        usage: BufferUsage,
    ) -> Result<BufferToken<'run>, NodeConstructionError> {
        self.run
            .graph
            .use_resource(self.seed, usage.resource_usage(id))?;
        Ok(BufferToken::new(id))
    }

    fn use_color(&mut self, index: usize, image: ImageId) -> Result<(), NodeConstructionError> {
        self.run
            .graph
            .use_resource(self.seed, ResourceUsage::ColorAttachment(image, index))
    }

    fn use_depth(
        &mut self,
        image: ImageId,
        write_access: bool,
    ) -> Result<(), NodeConstructionError> {
        let access = if write_access {
            AttachmentAccess::ReadWrite
        } else {
            AttachmentAccess::ReadOnly
        };
        self.run
            .graph
            .use_resource(self.seed, ResourceUsage::DepthAttachment(image, access))
    }

    fn use_input(&mut self, index: usize, image: ImageId) -> Result<(), NodeConstructionError> {
        self.run
            .graph
            .use_resource(self.seed, ResourceUsage::InputAttachment(image, index))
    }

    fn wait_semaphore(
        &mut self,
        stages: rendy_core::hal::pso::PipelineStage,
    ) -> WaitSemaphoreToken<'run> {
        let wait_id = self.run.graph.wait_semaphore(self.seed, stages);
        WaitSemaphoreToken::new(wait_id)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ImageNode {
    // The ID is only used to later retreive the image in execution phase
    pub(crate) id: ImageId,
    pub(crate) kind: rendy_core::hal::image::Kind,
    pub(crate) levels: rendy_core::hal::image::Level,
    pub(crate) format: rendy_core::hal::format::Format,
}

#[derive(Debug, Clone)]
pub(crate) struct BufferNode {
    pub(crate) id: BufferId,
    pub(crate) size: u64,
}

pub(crate) enum PlanNode<'n, B: Backend> {
    /// Construction phase execution
    // Execution(NodeExecution<'a, B, T>),
    /// Construction phase resource
    // Resource(ResourceId),
    Image(ImageNode),
    Buffer(BufferNode),
    ImageVersion,
    BufferVersion,
    /// Image clear operation
    ClearImage(rendy_core::hal::command::ClearValue),
    /// Buffer clear operation
    ClearBuffer(u32),
    /// Load image rendered in previous frame
    LoadImage(NodeId, usize),
    /// Store image for usage in next frame
    StoreImage(NodeId, usize),
    UndefinedImage,
    UndefinedBuffer,
    /// A subpass that might have multiple render groups.
    RenderSubpass(SmallVec<[(NodeId, PassFn<'n, B>); 4]>),
    /// A render pass - group of subpasses with dependencies between them
    RenderPass(RenderPassNode<'n, B>),
    /// A node representing arbitrary runnable operation.
    Submission(NodeId, SubmissionFn<'n, B>),
    PostSubmit(NodeId, PostSubmitFn<'n, B>),
    /// Resolve multisampled image into non-multisampled one.
    /// Currently this works under asumption that all layers of source image are resolved into first layer of destination.
    ResolveImage,
    /// A semaphore that is waited on before any child node can be executed
    WaitSemaphore(WaitId, rendy_core::hal::pso::PipelineStage),
    /// Graph root node. All incoming nodes are always evaluated.
    Root,
}

impl<'n, B: Backend> PlanNode<'n, B> {
    #[inline(always)]
    pub(crate) fn is_subpass(&self) -> bool {
        match self {
            Self::RenderSubpass(_) => true,
            _ => false,
        }
    }

    #[inline(always)]
    pub(crate) fn is_pass(&self) -> bool {
        match self {
            Self::RenderPass(_) => true,
            _ => false,
        }
    }

    pub(crate) fn pass_mut(&mut self) -> Option<&mut RenderPassNode<'n, B>> {
        match self {
            Self::RenderPass(pass) => Some(pass),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AttachmentRefEdge {
    Color(usize),
    Input(usize),
    DepthStencil(AttachmentAccess),
    // resolve attachments are not added at build time, but later based on graph transformation
    Resolve(usize),
}

impl AttachmentRefEdge {
    #[inline]
    pub(crate) fn is_write(&self) -> bool {
        match self {
            AttachmentRefEdge::Color(..) => true,
            AttachmentRefEdge::Resolve(..) => true,
            AttachmentRefEdge::Input(..) => false,
            AttachmentRefEdge::DepthStencil(access) => access.is_write(),
        }
    }
    #[inline]
    pub(crate) fn layout(&self) -> rendy_core::hal::image::Layout {
        match self {
            AttachmentRefEdge::Color(..) => rendy_core::hal::image::Layout::ColorAttachmentOptimal,
            AttachmentRefEdge::Resolve(..) => {
                rendy_core::hal::image::Layout::ColorAttachmentOptimal
            }
            AttachmentRefEdge::Input(..) => rendy_core::hal::image::Layout::ShaderReadOnlyOptimal,
            AttachmentRefEdge::DepthStencil(access) => match access {
                AttachmentAccess::ReadOnly => {
                    rendy_core::hal::image::Layout::DepthStencilReadOnlyOptimal
                }
                AttachmentAccess::ReadWrite => {
                    rendy_core::hal::image::Layout::DepthStencilAttachmentOptimal
                }
            },
        }
    }
    #[inline]
    pub(crate) fn accesses(&self) -> rendy_core::hal::image::Access {
        match self {
            AttachmentRefEdge::Color(..) => {
                rendy_core::hal::image::Access::COLOR_ATTACHMENT_READ
                    | rendy_core::hal::image::Access::COLOR_ATTACHMENT_WRITE
            }
            AttachmentRefEdge::Resolve(..) => {
                rendy_core::hal::image::Access::COLOR_ATTACHMENT_WRITE
            }
            AttachmentRefEdge::Input(..) => rendy_core::hal::image::Access::INPUT_ATTACHMENT_READ,
            AttachmentRefEdge::DepthStencil(access) => match access {
                AttachmentAccess::ReadOnly => {
                    rendy_core::hal::image::Access::DEPTH_STENCIL_ATTACHMENT_READ
                }
                AttachmentAccess::ReadWrite => {
                    rendy_core::hal::image::Access::DEPTH_STENCIL_ATTACHMENT_READ
                        | rendy_core::hal::image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE
                }
            },
        }
    }
    #[inline]
    pub(crate) fn stages(&self) -> rendy_core::hal::pso::PipelineStage {
        match self {
            // TODO: can it be declared if color is used as write-only?
            AttachmentRefEdge::Color(..) => {
                rendy_core::hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT
            }
            AttachmentRefEdge::Resolve(..) => {
                rendy_core::hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT
            }
            AttachmentRefEdge::Input(..) => rendy_core::hal::pso::PipelineStage::FRAGMENT_SHADER,
            AttachmentRefEdge::DepthStencil(_) => {
                // TODO: can it be declared which fragment test is used?
                rendy_core::hal::pso::PipelineStage::EARLY_FRAGMENT_TESTS
                    | rendy_core::hal::pso::PipelineStage::LATE_FRAGMENT_TESTS
            }
        }
    }
}

pub(crate) struct RenderPassSubpass<'n, B: Backend> {
    pub(crate) groups: SmallVec<[(NodeId, PassFn<'n, B>); 4]>,
    pub(crate) colors: SmallVec<[rendy_core::hal::pass::AttachmentRef; 8]>,
    pub(crate) inputs: SmallVec<[rendy_core::hal::pass::AttachmentRef; 8]>,
    pub(crate) resolves: SmallVec<[rendy_core::hal::pass::AttachmentRef; 4]>,
    pub(crate) depth_stencil: rendy_core::hal::pass::AttachmentRef,
}

impl<'n, B: Backend> std::fmt::Debug for RenderPassSubpass<'n, B> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("RenderPassSubpass")
            .field("groups", &self.groups.len())
            .field("colors", &self.colors)
            .field("inputs", &self.inputs)
            .field("resolves", &self.resolves)
            .field("depth_stencil", &self.depth_stencil)
            .finish()
    }
}

impl<'n, B: Backend> RenderPassSubpass<'n, B> {
    pub(crate) fn new(groups: SmallVec<[(NodeId, PassFn<'n, B>); 4]>) -> Self {
        Self {
            groups,
            colors: SmallVec::new(),
            inputs: SmallVec::new(),
            resolves: SmallVec::new(),
            depth_stencil: (ATTACHMENT_UNUSED, rendy_core::hal::image::Layout::Undefined),
        }
    }
    pub(crate) fn set_color(
        &mut self,
        ref_index: usize,
        attachment_ref: rendy_core::hal::pass::AttachmentRef,
    ) {
        if self.colors.len() <= ref_index {
            self.colors.resize(
                ref_index + 1,
                (ATTACHMENT_UNUSED, rendy_core::hal::image::Layout::Undefined),
            );
        }
        self.colors[ref_index] = attachment_ref;
    }
    pub(crate) fn set_resolve(
        &mut self,
        ref_index: usize,
        attachment_ref: rendy_core::hal::pass::AttachmentRef,
    ) {
        if self.resolves.len() <= ref_index {
            self.resolves.resize(
                ref_index + 1,
                (ATTACHMENT_UNUSED, rendy_core::hal::image::Layout::Undefined),
            );
        }
        self.colors[ref_index] = attachment_ref;
    }
    pub(crate) fn set_input(
        &mut self,
        ref_index: usize,
        attachment_ref: rendy_core::hal::pass::AttachmentRef,
    ) {
        if self.inputs.len() <= ref_index {
            self.inputs.resize(
                ref_index + 1,
                (ATTACHMENT_UNUSED, rendy_core::hal::image::Layout::Undefined),
            );
        }
        self.inputs[ref_index] = attachment_ref;
    }
    pub(crate) fn set_depth_stencil(
        &mut self,
        attachment_ref: rendy_core::hal::pass::AttachmentRef,
    ) {
        self.depth_stencil = attachment_ref;
    }
}

#[derive(Debug)]
pub(crate) struct RenderPassNode<'run, B: Backend> {
    pub(crate) attachments: SmallVec<[RenderPassAtachment; 8]>,
    pub(crate) subpasses: SmallVec<[RenderPassSubpass<'run, B>; 4]>,
    pub(crate) deps: SmallVec<[rendy_core::hal::pass::SubpassDependency; 32]>,
}

impl<'run, B: Backend> RenderPassNode<'run, B> {
    pub(crate) fn new() -> Self {
        Self {
            attachments: SmallVec::new(),
            subpasses: SmallVec::new(),
            deps: SmallVec::new(),
        }
    }

    fn run(self, general_ctx: ExecContext<'_, 'run, B>) {
        assert!(self.subpasses.len() > 0);

        let pass_info = unimplemented!();
        // let render_pass = general_ctx.request_render_pass(&self);

        // let queue: &mut Queue<B> = general_ctx.queue_mut(FamilyType::Graphics);

        // for (index, subpass) in self.subpasses.iter().enumerate() {
        //     let subpass = rendy_core::hal::pass::Subpass {
        //         index,
        //         main_pass: render_pass,
        //     };

        //     let pass_ctx = ExecPassContext {
        //         factory: general_ctx.factory(),
        //         queue,
        //         frames: general_ctx.frames(),
        //         // encoder: &'a mut RenderPassEncoder<'a, B>,
        //         subpass,
        //         images: HashMap::new(),
        //         buffers: HashMap::new(),
        //     };
        //     subpass(pass_ctx);
        // }
    }
}

// first_access/last_access is a way to approximate which subpasses need to preserve attachments.
// It might have false positives, but this should preserve correctness and is better than nothing.

// This is dogshit. Render pass doesn't have the "type" of attachment already figured out upfront.
// Same thing can be a color and an input (or even a resolve) for two different subpasses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RenderPassAtachment {
    pub(crate) ops: rendy_core::hal::pass::AttachmentOps,
    pub(crate) stencil_ops: rendy_core::hal::pass::AttachmentOps,
    pub(crate) clear: AttachmentClear,
    pub(crate) first_access: u8,
    pub(crate) last_access: u8,
    pub(crate) first_write: Option<(
        u8,
        rendy_core::hal::pso::PipelineStage,
        rendy_core::hal::image::Access,
    )>,
    pub(crate) last_write: Option<(
        u8,
        rendy_core::hal::pso::PipelineStage,
        rendy_core::hal::image::Access,
    )>,
    pub(crate) queued_reads: SmallVec<
        [(
            u8,
            rendy_core::hal::pso::PipelineStage,
            rendy_core::hal::image::Access,
        ); 4],
    >,
    pub(crate) first_layout: rendy_core::hal::image::Layout,
}

impl RenderPassAtachment {
    pub(crate) fn is_write(&self) -> bool {
        self.first_write.is_some()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AttachmentClear(pub(crate) Option<rendy_core::hal::command::ClearValue>);

impl Eq for AttachmentClear {}
impl PartialEq for AttachmentClear {
    fn eq(&self, other: &AttachmentClear) -> bool {
        match (self.0, other.0) {
            (Some(clear_a), Some(clear_b)) => unsafe {
                clear_a.color.uint32 == clear_b.color.uint32
            },
            (None, None) => true,
            _ => false,
        }
    }
}

impl AttachmentRefEdge {
    pub(crate) fn is_same_attachment(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Color(index_a), Self::Color(index_b)) => index_a == index_b,
            (Self::Input(index_a), Self::Input(index_b)) => index_a == index_b,
            (Self::DepthStencil(_), Self::DepthStencil(_)) => true,
            _ => false,
        }
    }
    pub(crate) fn merge(&mut self, other: &AttachmentRefEdge) {
        match (self, other) {
            (Self::DepthStencil(mut access_a), Self::DepthStencil(access_b)) => {
                access_a.merge(access_b);
            }
            _ => {}
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PlanEdge {
    AttachmentRef(AttachmentRefEdge),
    PassAttachment(RenderPassAtachment),
    /// Target node accesses connected source image version.
    BufferAccess(NodeBufferAccess, rendy_core::hal::pso::PipelineStage),
    ImageAccess(NodeImageAccess, rendy_core::hal::pso::PipelineStage),
    /// A node-to-node execution order dependency
    Effect,
    /// Resource version relation. Version increases in the edge direction
    Version,
    /// Link between a resource and a node that is responsible for it's creation.
    Origin,
}

impl<'n, B: Backend> std::fmt::Debug for PlanNode<'n, B> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanNode::Image(node) => fmt.debug_tuple("Image").field(node).finish(),
            PlanNode::Buffer(node) => fmt.debug_tuple("Buffer").field(node).finish(),
            PlanNode::LoadImage(i, c) => fmt.debug_tuple("LoadImage").field(i).field(c).finish(),
            PlanNode::StoreImage(i, c) => fmt.debug_tuple("StoreImage").field(i).field(c).finish(),
            PlanNode::ClearImage(c) => fmt.debug_tuple("ClearImage").field(c).finish(),
            PlanNode::ClearBuffer(c) => fmt.debug_tuple("ClearBuffer").field(c).finish(),
            PlanNode::ImageVersion => fmt.debug_tuple("ImageVersion").finish(),
            PlanNode::BufferVersion => fmt.debug_tuple("BufferVersion").finish(),
            PlanNode::UndefinedImage => fmt.debug_tuple("UndefinedImage").finish(),
            PlanNode::UndefinedBuffer => fmt.debug_tuple("UndefinedBuffer").finish(),
            PlanNode::WaitSemaphore(i, s) => {
                fmt.debug_tuple("WaitSemaphore").field(i).field(s).finish()
            }
            PlanNode::RenderSubpass(vec) => fmt
                .debug_tuple(&format!("RenderSubpass[{}]", vec.len()))
                .finish(),
            PlanNode::RenderPass(pass_node) => {
                fmt.write_str("RenderPass[")?;
                for (i, subpass) in pass_node.subpasses.iter().enumerate() {
                    if i != 0 {
                        fmt.write_str(", ")?;
                    }
                    write!(fmt, "{}", subpass.groups.len())?;
                }
                fmt.write_str("]")?;
                Ok(())
            }
            PlanNode::Submission(..) => fmt.debug_tuple("Submission").finish(),
            PlanNode::PostSubmit(..) => fmt.debug_tuple("PostSubmit").finish(),
            PlanNode::ResolveImage => fmt.debug_tuple("ResolveImage").finish(),
            PlanNode::Root => fmt.debug_tuple("Root").finish(),
        }
    }
}

impl PlanEdge {
    pub(crate) fn is_pass_attachment(&self) -> bool {
        match self {
            PlanEdge::PassAttachment(..) => true,
            _ => false,
        }
    }
    pub(crate) fn pass_attachment(&self) -> Option<&RenderPassAtachment> {
        match self {
            PlanEdge::PassAttachment(attachment) => Some(attachment),
            _ => None,
        }
    }
    pub(crate) fn pass_attachment_mut(&mut self) -> Option<&mut RenderPassAtachment> {
        match self {
            PlanEdge::PassAttachment(attachment) => Some(attachment),
            _ => None,
        }
    }
    pub(crate) fn is_attachment_ref(&self) -> bool {
        match self {
            PlanEdge::AttachmentRef(_) => true,
            _ => false,
        }
    }
    pub(crate) fn attachment_ref(&self) -> Option<&AttachmentRefEdge> {
        match self {
            PlanEdge::AttachmentRef(edge) => Some(edge),
            _ => None,
        }
    }
    pub(crate) fn is_version(&self) -> bool {
        match self {
            PlanEdge::Version => true,
            _ => false,
        }
    }
    pub(crate) fn is_origin(&self) -> bool {
        match self {
            PlanEdge::Origin => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
struct NodeSeed {
    resources: Range<usize>,
}

pub(crate) type PlanDag<'run, 'arena, B> = graphy::Graph<'arena, PlanNode<'run, B>, PlanEdge>;

#[derive(Debug)]
pub struct PlanGraph<'run, 'arena, B: Backend> {
    dag: PlanDag<'run, 'arena, B>,
    alloc: &'arena GraphAllocator,
    virtuals: usize,
    semaphores: usize,
    last_writes: HashMap<ResourceId, NodeIndex>,
    resource_usage: Vec<ResourceUsage>,
    retained_images: HashMap<(NodeId, usize), Handle<Image<B>>>,
}

impl<'run, 'arena, B: Backend> PlanGraph<'run, 'arena, B> {
    fn new(alloc: &'arena GraphAllocator) -> Self {
        let mut dag = PlanDag::new();
        // guaranteed to always be index 0
        dag.insert_node(alloc, PlanNode::Root);

        Self {
            dag,
            alloc,
            virtuals: 0,
            semaphores: 0,
            last_writes: HashMap::new(),
            resource_usage: Vec::new(),
            retained_images: HashMap::new(),
        }
    }

    fn seed(&self) -> NodeSeed {
        let len = self.resource_usage.len();
        NodeSeed {
            resources: len..len,
        }
    }

    fn use_resource(
        &mut self,
        seed: &mut NodeSeed,
        usage: ResourceUsage,
    ) -> Result<(), NodeConstructionError> {
        if let Some(res) = self.resource_usage[seed.resources.clone()]
            .iter_mut()
            .find(|r| r.is_same_resource(&usage))
        {
            res.merge_access(&usage)
        } else {
            self.resource_usage.push(usage);
            seed.resources.end += 1;
            Ok(())
        }
    }

    fn wait_semaphore(
        &mut self,
        seed: &mut NodeSeed,
        stages: rendy_core::hal::pso::PipelineStage,
    ) -> WaitId {
        let id = WaitId(self.semaphores);
        self.semaphores += 1;
        self.resource_usage
            .push(ResourceUsage::WaitSemaphore(id, stages));
        seed.resources.end += 1;
        id
    }

    /// Create new image owned by graph. The image lifecycle is totally controlled by the graph.
    pub(crate) fn create_image(&mut self, id: ImageId, info: ImageInfo) {
        let def_data = PlanNode::Image(ImageNode {
            id,
            kind: info.kind,
            levels: info.levels,
            format: info.format,
        });
        let init_data = match info.load {
            ImageLoad::DontCare => PlanNode::UndefinedImage,
            ImageLoad::Clear(clear) => PlanNode::ClearImage(clear),
            ImageLoad::Retain(index, clear) => {
                if self.retained_images.contains_key(&(id.0, index)) {
                    PlanNode::LoadImage(id.0, index)
                } else {
                    PlanNode::ClearImage(clear)
                }
            }
        };
        let def = self.dag.insert_node(self.alloc, def_data);
        let (_, init) = self
            .dag
            .insert_child(self.alloc, def, PlanEdge::Origin, init_data);
        self.last_writes.insert(ResourceId::Image(id), init);
    }
    /// Create new buffer owned by graph.
    pub(crate) fn create_buffer(&mut self, id: BufferId, info: BufferInfo) {
        let def_data = PlanNode::Buffer(BufferNode {
            id,
            size: info.size,
        });
        let init_data = if let Some(clear) = info.clear {
            PlanNode::ClearBuffer(clear)
        } else {
            PlanNode::UndefinedBuffer
        };

        let def = self.dag.insert_node(self.alloc, def_data);
        let (_, init) = self
            .dag
            .insert_child(self.alloc, def, PlanEdge::Origin, init_data);
        self.last_writes.insert(ResourceId::Buffer(id), init);
    }
    /// Create non-data dependency target. A virtual resource intended to
    /// describe dependencies between rendering nodes without carrying any data.
    pub(crate) fn create_virtual(&mut self) -> VirtualId {
        self.virtuals += 1;
        VirtualId(self.virtuals - 1)
    }

    fn insert_node(&mut self, node: PlanNode<'run, B>) -> NodeIndex {
        self.dag.insert_node(self.alloc, node)
    }

    fn insert_child(
        &mut self,
        parent: NodeIndex,
        edge: PlanEdge,
        node: PlanNode<'run, B>,
    ) -> NodeIndex {
        self.dag.insert_child(self.alloc, parent, edge, node).1
    }

    fn insert_edge(&mut self, from: NodeIndex, to: NodeIndex, edge: PlanEdge) -> EdgeIndex {
        self.dag
            .insert_edge_unchecked(self.alloc, from, to, edge)
            .unwrap()
        // TODO: implement checked version
        // self.dag
        //     .insert_edge(self.alloc, from, to, edge)
        //     .unwrap_or_else(|err| {
        //         panic!(
        //             "Trying to insert a cycle by adding an edge {} -> {}: {:?}",
        //             from.index(),
        //             to.index(),
        //             err,
        //         )
        //     })
    }

    fn insert(
        &mut self,
        node_id: NodeId,
        seed: NodeSeed,
        exec: NodeExecution<'run, B>,
    ) -> Result<(), NodeConstructionError> {
        // inserts should always happen in such order that resource usages are free to be drained
        debug_assert!(self.resource_usage.len() == seed.resources.end);

        // insert execution node
        let (node, allow_attachments) = match exec {
            NodeExecution::RenderPass(group) => (
                self.insert_node(PlanNode::RenderSubpass(smallvec![(node_id, group)])),
                true,
            ),
            NodeExecution::Submission(phase, closure) => {
                let node = self.insert_node(PlanNode::Submission(node_id, closure));
                match phase {
                    ExecutionPhase::Default => {}
                    ExecutionPhase::Output => {
                        self.insert_edge(node, NodeIndex::new(0), PlanEdge::Effect);
                    }
                }
                (node, false)
            }
            NodeExecution::PostSubmit(phase, closure) => {
                let node = self.insert_node(PlanNode::PostSubmit(node_id, closure));
                match phase {
                    ExecutionPhase::Default => {}
                    ExecutionPhase::Output => {
                        self.insert_edge(node, NodeIndex::new(0), PlanEdge::Effect);
                    }
                }
                (node, false)
            }
            NodeExecution::None => {
                // ignore resource access declared in this construction
                self.resource_usage.truncate(seed.resources.start);
                return Ok(());
            }
        };

        // insert resource reads and writes
        for res in self.resource_usage.drain(seed.resources) {
            match res {
                ResourceUsage::Image(id, access, stage) => {
                    if !access.is_empty() {
                        let last_version = self
                            .last_writes
                            .get(&ResourceId::Image(id))
                            .copied()
                            .expect("Accessing unproceessed image");
                        self.dag
                            .insert_edge_unchecked(
                                self.alloc,
                                last_version,
                                node,
                                PlanEdge::ImageAccess(access, stage),
                            )
                            .unwrap();
                        if !access.writes().is_empty() {
                            let (_, version) = self.dag.insert_child(
                                self.alloc,
                                node,
                                PlanEdge::Origin,
                                PlanNode::ImageVersion,
                            );
                            self.dag
                                .insert_edge_unchecked(
                                    self.alloc,
                                    last_version,
                                    version,
                                    PlanEdge::Version,
                                )
                                .unwrap();
                            self.last_writes.insert(ResourceId::Image(id), version);
                        }
                    }
                }
                ResourceUsage::Buffer(id, access, stage) => {
                    if !access.is_empty() {
                        let last_version = self
                            .last_writes
                            .get(&ResourceId::Buffer(id))
                            .copied()
                            .expect("Accessing unproceessed buffer");
                        self.dag
                            .insert_edge_unchecked(
                                self.alloc,
                                last_version,
                                node,
                                PlanEdge::BufferAccess(access, stage),
                            )
                            .unwrap();
                        if !access.writes().is_empty() {
                            let (_, version) = self.dag.insert_child(
                                self.alloc,
                                node,
                                PlanEdge::Origin,
                                PlanNode::BufferVersion,
                            );
                            self.dag
                                .insert_edge_unchecked(
                                    self.alloc,
                                    last_version,
                                    version,
                                    PlanEdge::Version,
                                )
                                .unwrap();
                            self.last_writes.insert(ResourceId::Buffer(id), version);
                        }
                    }
                }
                ResourceUsage::Virtual(id, access) => {
                    if !access.is_empty() {
                        if let Some(dep) = self.last_writes.get(&ResourceId::Virtual(id)) {
                            self.dag
                                .insert_edge_unchecked(self.alloc, node, *dep, PlanEdge::Effect)
                                .unwrap();
                        }
                    }
                    if !access.writes().is_empty() {
                        self.last_writes.insert(ResourceId::Virtual(id), node);
                    }
                }
                ResourceUsage::ColorAttachment(id, index) => {
                    if !allow_attachments {
                        continue;
                    }
                    let last_version = self
                        .last_writes
                        .get(&ResourceId::Image(id))
                        .copied()
                        .expect("Accessing unproceessed image");
                    self.dag
                        .insert_edge_unchecked(
                            self.alloc,
                            last_version,
                            node,
                            PlanEdge::AttachmentRef(AttachmentRefEdge::Color(index)),
                        )
                        .unwrap();
                    let (_, version) = self.dag.insert_child(
                        self.alloc,
                        node,
                        PlanEdge::Origin,
                        PlanNode::ImageVersion,
                    );
                    self.dag
                        .insert_edge_unchecked(self.alloc, last_version, version, PlanEdge::Version)
                        .unwrap();
                    self.last_writes.insert(ResourceId::Image(id), version);
                }
                ResourceUsage::InputAttachment(id, index) => {
                    if !allow_attachments {
                        continue;
                    }
                    let last_version = self
                        .last_writes
                        .get(&ResourceId::Image(id))
                        .copied()
                        .expect("Accessing unproceessed image");
                    self.dag
                        .insert_edge_unchecked(
                            self.alloc,
                            last_version,
                            node,
                            PlanEdge::AttachmentRef(AttachmentRefEdge::Input(index)),
                        )
                        .unwrap();
                }
                ResourceUsage::DepthAttachment(id, access) => {
                    if !allow_attachments {
                        continue;
                    }
                    let last_version = self
                        .last_writes
                        .get(&ResourceId::Image(id))
                        .copied()
                        .expect("Accessing unproceessed image");
                    self.dag
                        .insert_edge_unchecked(
                            self.alloc,
                            last_version,
                            node,
                            PlanEdge::AttachmentRef(AttachmentRefEdge::DepthStencil(access)),
                        )
                        .unwrap();
                    if access == AttachmentAccess::ReadWrite {
                        let (_, version) = self.dag.insert_child(
                            self.alloc,
                            node,
                            PlanEdge::Origin,
                            PlanNode::ImageVersion,
                        );
                        self.dag
                            .insert_edge_unchecked(
                                self.alloc,
                                last_version,
                                version,
                                PlanEdge::Version,
                            )
                            .unwrap();
                        self.last_writes.insert(ResourceId::Image(id), version);
                    }
                }
                ResourceUsage::WaitSemaphore(id, stage) => {
                    let semaphore_node = self
                        .dag
                        .insert_node(self.alloc, PlanNode::WaitSemaphore(id, stage));
                    self.dag
                        .insert_edge_unchecked(self.alloc, semaphore_node, node, PlanEdge::Effect)
                        .unwrap();
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct ExecContext<'ctx, 'run, B: Backend> {
    factory: &'ctx Factory<B>,
    families: &'ctx mut Families<B>,
    images: HashMap<ImageId, NodeImage<B>>,
    buffers: HashMap<BufferId, NodeBuffer<B>>,
    frames: &'ctx Frames<B>,
    resources: &'ctx mut GraphResourcePool<B>,
    submits: &'ctx mut Vec<EitherSubmit<'run, B>>,
    indexed_waits: &'ctx HashMap<WaitId, usize>,
    wait_semaphores: &'ctx [B::Semaphore],
}

impl<'ctx, 'run, B: Backend> ExecContext<'ctx, 'run, B> {
    pub(crate) fn new() -> Self {
        unimplemented!()
    }

    pub fn queue_mut(&mut self, family_type: FamilyType) -> &mut Queue<B> {
        self.families.queue_mut(self.resources.queue(family_type))
    }

    pub fn request_render_pass(&mut self, pass_info: RenderPassInfo) -> &B::RenderPass {
        self.resources.request_render_pass(self.factory, pass_info)
    }

    pub fn submit<C>(&mut self, submits: C)
    where
        C: IntoIterator,
        C::Item: Into<EitherSubmit<'run, B>>,
    {
        self.submits.extend(submits.into_iter().map(Into::into))
    }

    pub fn get_image(&self, token: ImageToken) -> &NodeImage<B> {
        self.images
            .get(&token.id())
            .expect("Somehow got a token to unscheduled image")
    }

    pub fn get_buffer(&self, token: BufferToken) -> &NodeBuffer<B> {
        self.buffers
            .get(&token.id())
            .expect("Somehow got a token to unscheduled buffer")
    }

    pub fn get_semaphore(&self, token: WaitSemaphoreToken) -> &'ctx B::Semaphore {
        let index = self
            .indexed_waits
            .get(&token.id())
            .expect("Somehow got a token to unscheduled semaphore");
        &self.wait_semaphores[*index]
    }

    pub fn frames(&self) -> &Frames<B> {
        self.frames
    }
}

#[derive(Debug)]
pub struct ExecPassContext<'a, B: Backend> {
    factory: &'a Factory<B>,
    queue: &'a mut Queue<B>,
    frames: &'a Frames<B>,
    encoder: &'a mut RenderPassEncoder<'a, B>,
    subpass: rendy_core::hal::pass::Subpass<'a, B>,
    images: HashMap<ImageId, Handle<Image<B>>>,
    buffers: HashMap<BufferId, Handle<Buffer<B>>>,
}

impl<'a, B: Backend> ExecPassContext<'a, B> {
    pub fn get_image(&self, token: ImageToken) -> Handle<Image<B>> {
        self.images
            .get(&token.id())
            .expect("Somehow got a token to unscheduled image")
            .clone()
    }

    pub fn get_buffer(&self, token: BufferToken) -> Handle<Buffer<B>> {
        self.buffers
            .get(&token.id())
            .expect("Somehow got a token to unscheduled buffer")
            .clone()
    }
}

#[derive(Debug)]
struct ConstructContext<'run, 'arena, B: Backend> {
    factory: &'run Factory<B>,
    families: &'run mut Families<B>,
    frames: &'run mut Frames<B>,
    resources: &'run mut GraphResourcePool<B>,
    output_store: OutputStore,
    images: HashMap<ImageId, (Handle<Image<B>>, Option<B::Semaphore>)>,
    buffers: HashMap<BufferId, Handle<Buffer<B>>>,
    graph: PlanGraph<'run, 'arena, B>,
}

// struct FrameContext<'run, B: Backend> {

// }

impl<'run, 'arena, B: Backend> ConstructContext<'run, 'arena, B> {
    fn new(
        factory: &'run Factory<B>,
        families: &'run mut Families<B>,
        frames: &'run mut Frames<B>,
        resources: &'run mut GraphResourcePool<B>,
        allocator: &'arena GraphAllocator,
    ) -> Self {
        Self {
            factory,
            families,
            frames,
            resources,
            output_store: OutputStore::new(),
            images: HashMap::new(),
            buffers: HashMap::new(),
            graph: PlanGraph::new(allocator),
        }
    }

    fn provide_image(
        &mut self,
        id: ImageId,
        image: Handle<Image<B>>,
        acquire: Option<B::Semaphore>,
    ) {
        self.images.insert(id, (image, acquire));
    }

    fn provide_buffer(&mut self, id: BufferId, buffer: Handle<Buffer<B>>) {
        self.buffers.insert(id, buffer);
    }

    fn run_execution_phase(self) -> Result<(), GraphRunError> {
        let topo: Vec<_> = TopoWithEdges::new(&self.graph.dag, NodeIndex::new(0))
            .iter(&self.graph.dag)
            .collect();

        let (mut nodes, mut edges) = self.graph.dag.into_items();

        // TODO: flushing and submits must be per family
        let mut flush_before_next = false;
        let mut submits = Vec::new();
        let mut wait_semaphores = Vec::new();
        let mut wait_stages = Vec::new();
        let mut flush_before_next = false;
        let mut indexed_waits = HashMap::new();

        let total_submits = nodes
            .iter()
            .filter(|node| match node {
                PlanNode::RenderPass(..) | PlanNode::Submission(..) => true,
                _ => false,
            })
            .count();
        let mut visited_submits = 0;

        for item in topo.into_iter().rev() {
            // :TakeShouldMove
            // @Cleanup: Those replaces should not be necessary
            match item {
                GraphItem::Node(node) => {
                    match std::mem::replace(nodes.take(node).unwrap(), PlanNode::Root) {
                        PlanNode::Image(..) => {}
                        PlanNode::Buffer(..) => {}
                        PlanNode::ClearImage(..) => {}
                        PlanNode::ClearBuffer(..) => {}
                        PlanNode::LoadImage(..) => {}
                        PlanNode::StoreImage(..) => {}
                        PlanNode::WaitSemaphore(index, stages) => {
                            flush_before_next = true;
                            // TODO: manage semaphore lifetimes
                            let semaphore = self.factory.create_semaphore().unwrap();
                            indexed_waits.insert(index, wait_semaphores.len());
                            wait_semaphores.push(semaphore);
                            wait_stages.push(stages);
                        }
                        PlanNode::RenderPass(pass) => {
                            visited_submits += 1;
                            let ctx = ExecContext {
                                factory: self.factory,
                                families: self.families,
                                images: HashMap::new(),  // TODO
                                buffers: HashMap::new(), // TODO
                                frames: self.frames,
                                resources: self.resources,
                                submits: &mut submits,
                                indexed_waits: &indexed_waits,
                                wait_semaphores: &wait_semaphores,
                            };

                            pass.run(ctx);
                        }
                        PlanNode::Submission(node_id, closure) => {
                            visited_submits += 1;
                            let ctx = ExecContext {
                                factory: self.factory,
                                families: self.families,
                                images: HashMap::new(),  // TODO
                                buffers: HashMap::new(), // TODO
                                frames: self.frames,
                                resources: self.resources,
                                submits: &mut submits,
                                indexed_waits: &indexed_waits,
                                wait_semaphores: &wait_semaphores,
                            };
                            closure(ctx).map_err(|e| GraphRunError::NodeExecution(node_id, e))?
                        }
                        PlanNode::PostSubmit(node_id, closure) => {
                            let last_submit = visited_submits == total_submits;
                            if flush_before_next || last_submit {
                                flush_before_next = false;
                                // TODO: split submits by family type. Now just assume everything is graphics
                                // let queue = self
                                //     .families
                                //     .queue_mut(self.resources.queue(FamilyType::Graphics));
                                // TODO: put fence if last submission
                                // queue.submit(submits.drain(..), None);
                                // queue.submit(
                                //     Some(
                                //         Submission::new()
                                //             .submits(submits.drain(..))
                                //             // .wait()
                                //             .signal(wait_semaphores.drain(..)),
                                //     ),
                                //     fence,
                                // );
                            }

                            let ctx = ExecContext {
                                factory: self.factory,
                                families: self.families,
                                images: HashMap::new(),  // TODO
                                buffers: HashMap::new(), // TODO
                                frames: self.frames,
                                resources: self.resources,
                                submits: &mut submits,
                                indexed_waits: &indexed_waits,
                                wait_semaphores: &wait_semaphores,
                            };
                            closure(ctx).map_err(|e| GraphRunError::NodeExecution(node_id, e))?
                        }
                        PlanNode::ResolveImage => {}
                        PlanNode::ImageVersion => {}
                        PlanNode::BufferVersion => {}
                        PlanNode::UndefinedImage => {}
                        PlanNode::UndefinedBuffer => {}
                        PlanNode::Root => {}
                        n @ PlanNode::RenderSubpass(..) => {
                            panic!("Unexpected unprocessed node {:?}", n)
                        }
                    }
                }
                GraphItem::Edge(edge) => {
                    match std::mem::replace(edges.take(edge).unwrap(), PlanEdge::Effect) {
                        PlanEdge::Effect | PlanEdge::Version | PlanEdge::Origin => {}
                        e @ PlanEdge::AttachmentRef(..)
                        | e @ PlanEdge::PassAttachment(..)
                        | e @ PlanEdge::BufferAccess(..)
                        | e @ PlanEdge::ImageAccess(..) => {
                            panic!("Unexpected unprocessed edge {:?}", e)
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
/// Error that happen during rendering graph run.
#[derive(Debug, Clone, Copy)]
pub enum GraphRunError {
    /// Error during node construction phase in the graph.
    NodeConstruction(NodeId, NodeConstructionError),
    /// Error during node execution phase in the graph.
    NodeExecution(NodeId, NodeExecutionError),
}

#[cfg(all(
    test,
    any(
        feature = "empty",
        feature = "dx12",
        feature = "metal",
        feature = "vulkan"
    )
))]
mod test {
    use super::*;
    use crate::{
        command::Family,
        factory::Factory,
        new::{
            node::ConstructResult,
            resources::ShaderUsage,
            test::{test_init, visualize_graph},
        },
    };

    impl<B: Backend, T: ?Sized> NodeBuilder<B, T> for ImageInfo {
        type Node = Self;
        type Family = crate::command::Transfer;
        fn build(
            self: Box<Self>,
            _: &mut Factory<B>,
            _: &mut Family<B>,
            _: &T,
        ) -> Result<Self::Node, NodeBuildError> {
            Ok(*self)
        }
    }

    impl<B: Backend, T: ?Sized> Node<B, T> for ImageInfo {
        type Outputs = Parameter<ImageId>;
        fn construct<'run>(
            &'run mut self,
            ctx: &mut impl NodeCtx<'run, B>,
            _aux: &'run T,
        ) -> ConstructResult<'run, Self, B, T> {
            let image = ctx.create_image(*self);
            Ok((image, NodeExecution::None))
        }
    }

    #[derive(Debug)]
    struct TestPass {
        color: Parameter<ImageId>,
        depth: Option<Parameter<ImageId>>,
    }

    impl TestPass {
        fn new(color: Parameter<ImageId>, depth: Option<Parameter<ImageId>>) -> Self {
            Self { color, depth }
        }
    }

    impl<B: Backend, T: ?Sized> NodeBuilder<B, T> for TestPass {
        type Node = Self;
        type Family = crate::command::Graphics;
        fn build(
            self: Box<Self>,
            _: &mut Factory<B>,
            _: &mut Family<B>,
            _: &T,
        ) -> Result<Self::Node, NodeBuildError> {
            Ok(*self)
        }
    }

    impl<B: Backend, T: ?Sized> Node<B, T> for TestPass {
        type Outputs = ();

        fn construct<'run>(
            &'run mut self,
            ctx: &mut impl NodeCtx<'run, B>,
            _aux: &'run T,
        ) -> ConstructResult<'run, Self, B, T> {
            let color = *ctx.get_parameter(self.color)?;
            ctx.use_color(0, color)?;
            if let Some(depth) = self.depth {
                let depth = *ctx.get_parameter(depth)?;
                ctx.use_depth(depth, true)?;
            }

            Ok(((), NodeExecution::pass(|_| Ok(()))))
        }
    }

    #[derive(Debug)]
    struct TestPass2 {
        color1: Option<Parameter<ImageId>>,
        color2: Option<Parameter<ImageId>>,
        depth: Option<Parameter<ImageId>>,
    }

    impl TestPass2 {
        fn new(
            color1: Option<Parameter<ImageId>>,
            color2: Option<Parameter<ImageId>>,
            depth: Option<Parameter<ImageId>>,
        ) -> Self {
            Self {
                color1,
                color2,
                depth,
            }
        }
    }

    impl<B: Backend, T: ?Sized> NodeBuilder<B, T> for TestPass2 {
        type Node = Self;
        type Family = crate::command::Graphics;
        fn build(
            self: Box<Self>,
            _: &mut Factory<B>,
            _: &mut Family<B>,
            _: &T,
        ) -> Result<Self::Node, NodeBuildError> {
            Ok(*self)
        }
    }

    impl<B: Backend, T: ?Sized> Node<B, T> for TestPass2 {
        type Outputs = ();

        fn construct<'run>(
            &'run mut self,
            ctx: &mut impl NodeCtx<'run, B>,
            _aux: &'run T,
        ) -> ConstructResult<'run, Self, B, T> {
            if let Some(color1) = self.color1 {
                let color1 = *ctx.get_parameter(color1)?;
                ctx.use_color(0, color1)?;
            }
            if let Some(color2) = self.color2 {
                let color2 = *ctx.get_parameter(color2)?;
                ctx.use_input(0, color2)?;
            }
            if let Some(depth) = self.depth {
                let depth = *ctx.get_parameter(depth)?;
                ctx.use_depth(depth, false)?;
            }

            Ok(((), NodeExecution::pass(|_| Ok(()))))
        }
    }

    #[derive(Debug)]
    struct TestCompute1 {
        color: Option<Parameter<ImageId>>,
    }

    impl TestCompute1 {
        fn new(color: Option<Parameter<ImageId>>) -> Self {
            Self { color }
        }
    }

    impl<B: Backend, T: ?Sized> NodeBuilder<B, T> for TestCompute1 {
        type Node = Self;
        type Family = crate::command::Graphics;
        fn build(
            self: Box<Self>,
            _: &mut Factory<B>,
            _: &mut Family<B>,
            _: &T,
        ) -> Result<Self::Node, NodeBuildError> {
            Ok(*self)
        }
    }

    impl<B: Backend, T: ?Sized> Node<B, T> for TestCompute1 {
        type Outputs = Parameter<BufferId>;

        fn construct<'run>(
            &'run mut self,
            ctx: &mut impl NodeCtx<'run, B>,
            _aux: &'run T,
        ) -> ConstructResult<'run, Self, B, T> {
            let buf_id = ctx.create_buffer(BufferInfo {
                size: 1,
                clear: None,
            });
            let buf_usage =
                ctx.use_buffer(buf_id, BufferUsage::StorageWrite(ShaderUsage::COMPUTE))?;

            if let Some(color) = self.color {
                let color = *ctx.get_parameter(color)?;
                ctx.use_color(0, color)?;
            }

            Ok((
                buf_id,
                NodeExecution::pass(move |_| {
                    println!("{:?}, {:?}", buf_usage, self);
                    Ok(())
                }),
            ))
        }
    }

    #[derive(Debug)]
    struct TestPass3 {
        color: Parameter<ImageId>,
        buffer: Parameter<BufferId>,
        write_buffer: bool,
    }

    impl TestPass3 {
        fn new(color: Parameter<ImageId>, buffer: Parameter<BufferId>, write_buffer: bool) -> Self {
            Self {
                color,
                buffer,
                write_buffer,
            }
        }
    }

    impl<B: Backend, T: ?Sized> NodeBuilder<B, T> for TestPass3 {
        type Node = Self;
        type Family = crate::command::Graphics;
        fn build(
            self: Box<Self>,
            _: &mut Factory<B>,
            _: &mut Family<B>,
            _: &T,
        ) -> Result<Self::Node, NodeBuildError> {
            Ok(*self)
        }
    }

    impl<B: Backend, T: ?Sized> Node<B, T> for TestPass3 {
        type Outputs = ();

        fn construct<'run>(
            &'run mut self,
            ctx: &mut impl NodeCtx<'run, B>,
            _aux: &'run T,
        ) -> ConstructResult<'run, Self, B, T> {
            let color = *ctx.get_parameter(self.color)?;
            let buffer = *ctx.get_parameter(self.buffer)?;

            ctx.use_color(0, color)?;
            if self.write_buffer {
                ctx.use_buffer(buffer, BufferUsage::StorageWrite(ShaderUsage::FRAGMENT))?;
            } else {
                ctx.use_buffer(buffer, BufferUsage::StorageRead(ShaderUsage::FRAGMENT))?;
            }

            Ok(((), NodeExecution::pass(|_| Ok(()))))
        }
    }

    #[derive(Debug)]
    struct TestOutput;

    impl<B: Backend, T: ?Sized> NodeBuilder<B, T> for TestOutput {
        type Node = Self;
        type Family = crate::command::Graphics;
        fn build(
            self: Box<Self>,
            _: &mut Factory<B>,
            _: &mut Family<B>,
            _: &T,
        ) -> Result<Self::Node, NodeBuildError> {
            Ok(*self)
        }
    }

    impl<B: Backend, T: ?Sized> Node<B, T> for TestOutput {
        type Outputs = Parameter<ImageId>;
        fn construct<'run>(
            &'run mut self,
            ctx: &mut impl NodeCtx<'run, B>,
            aux: &'run T,
        ) -> ConstructResult<'run, Self, B, T> {
            let output = ctx.create_image(ImageInfo {
                kind: rendy_core::hal::image::Kind::D2(1024, 1024, 1, 1),
                levels: 1,
                format: rendy_core::hal::format::Format::Rgba8Unorm,
                load: ImageLoad::DontCare,
            });
            let output_use = ctx.use_image(output, ImageUsage::ColorAttachmentRead)?;
            Ok((
                output,
                NodeExecution::output_submission(move |ctx| {
                    let _output = ctx.get_image(output_use);
                    Ok(())
                }),
            ))
        }
    }

    #[test]
    fn test_construction() {
        test_init();

        let config: crate::factory::Config = Default::default();
        let initialized = rendy_init::AnyRendy::init_auto(&config).unwrap();

        rendy_init::with_any_rendy!((initialized)
            (mut factory, mut families) => {
                let mut builder = GraphBuilder::new();

                let depth = builder.add(ImageInfo {
                    kind: rendy_core::hal::image::Kind::D2(1, 1, 1, 1),
                    levels: 1,
                    format: rendy_core::hal::format::Format::R32Sfloat,
                    load: ImageLoad::Clear(rendy_core::hal::command::ClearValue {
                        depth_stencil: rendy_core::hal::command::ClearDepthStencil {
                            depth: 0.0,
                            stencil: 0,
                        },
                    }),
                });
                let depth2 = builder.add(ImageInfo {
                    kind: rendy_core::hal::image::Kind::D2(1, 1, 1, 1),
                    levels: 1,
                    format: rendy_core::hal::format::Format::R32Sfloat,
                    load: ImageLoad::DontCare,
                });
                let color2 = builder.add(ImageInfo {
                    kind: rendy_core::hal::image::Kind::D2(1, 1, 1, 1),
                    levels: 1,
                    format: rendy_core::hal::format::Format::Rgba8Unorm,
                    load: ImageLoad::DontCare,
                });
                let color = builder.add(TestOutput);
                builder.add(TestPass2::new(Some(color2), None, None));
                builder.add(TestPass2::new(Some(color), Some(color2), None));
                builder.add(TestPass::new(color, Some(depth)));
                let buf1 = builder.add(TestCompute1::new(Some(color)));
                builder.add(TestPass::new(color2, Some(depth2)));
                builder.add(TestPass3::new(color, buf1, false));
                builder.add(TestPass3::new(color, buf1, true));
                builder.add(TestPass2::new(Some(color2), Some(color), None));
                builder.add(TestPass2::new(Some(color), Some(color2), None));
                builder.add(TestPass2::new(Some(color), Some(color2), Some(depth2)));
                builder.add(TestPass::new(color, Some(depth)));
                builder.add(TestPass::new(color, None));

                builder.add(TestPass2::new(Some(color2), None, None));
                builder.add(TestPass2::new(Some(color), Some(color2), None));
                builder.add(TestPass::new(color, None));

                let mut graph = builder.build(&mut factory, &mut families, &()).unwrap();
                unsafe {
                    graph.alloc.reset();
                }
                let mut run_ctx = ConstructContext::new(&factory, &mut families, &mut graph.frames, &mut graph.resources, &graph.alloc);
                graph
                    .nodes
                    .run_construction_phase(&mut run_ctx, &())
                    .unwrap();

                let mut file = std::fs::File::create("graph.dot").unwrap();
                visualize_graph(&mut file, &run_ctx.graph.dag, "raw");

                graph.pipeline.reduce(&mut run_ctx.graph.dag, &graph.alloc);
                visualize_graph(&mut file, &run_ctx.graph.dag, "opti");
                run_ctx.run_execution_phase().unwrap();

            }
        );
    }
}
