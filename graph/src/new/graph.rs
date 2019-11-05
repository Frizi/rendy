use {
    super::{
        node::{
            DynNode, DynNodeBuilder, ExecContext, GeneralFn, InternalUse, Node, NodeBuildError,
            NodeBuilder, NodeConstructionError, NodeExecution, NodeExecutionError, NodeId,
            OutputList, OutputStore, Parameter, PassFn,
        },
        pipeline::Pipeline,
        resources::{
            AttachmentAccess, BufferId, BufferInfo, BufferUsage, ImageId, ImageInfo, ImageLoad,
            ImageUsage, NodeBufferAccess, NodeImageAccess, NodeVirtualAccess, ResourceId,
            ResourceUsage, VirtualId,
        },
        walker::{GraphItem, TopoWithEdges},
    },
    crate::{
        command::Families,
        factory::Factory,
        resource::{Handle, Image},
    },
    gfx_hal::{pass::ATTACHMENT_UNUSED, Backend},
    graphy::{EdgeIndex, GraphAllocator, NodeIndex, Walker},
    smallvec::{smallvec, SmallVec},
    std::{any::Any, collections::HashMap, marker::PhantomData, ops::Range},
};
/// A builder type for rendering graph.
#[derive(derivative::Derivative)]
#[derivative(Default(bound = ""), Debug(bound = ""))]
pub struct GraphBuilder<B: Backend, T: ?Sized> {
    nodes: Vec<Box<dyn DynNodeBuilder<B, T>>>,
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
    /// TODO: Example with outputs
    pub fn add<N: NodeBuilder<B, T> + 'static>(
        &mut self,
        builder: N,
    ) -> <N::Node as Node<B, T>>::Outputs {
        let node_id = NodeId(self.nodes.len());
        self.nodes.push(Box::new(builder));
        OutputList::instance(node_id, 0, InternalUse(()))
    }
    /// Build rendering graph
    pub fn build(
        self,
        factory: &mut Factory<B>,
        families: &mut Families<B>,
        aux: &T,
    ) -> Result<Graph<B, T>, GraphBuildError> {
        Ok(Graph {
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
        })
    }
}
/// A built runnable top-level rendering graph.
#[derive(derivative::Derivative)]
#[derivative(Debug(bound = ""))]
pub struct Graph<B: Backend, T: ?Sized> {
    nodes: GraphNodes<B, T>,
    pipeline: Pipeline<B, T>,
    alloc: GraphAllocator,
}

#[derive(derivative::Derivative)]
#[derivative(Debug(bound = ""))]
struct GraphNodes<B: Backend, T: ?Sized>(Vec<Box<dyn DynNode<B, T>>>);

impl<B: Backend, T: ?Sized> Graph<B, T> {
    /// Construct, schedule and run all nodes of rendering graph.
    pub fn run(
        &mut self,
        factory: &mut Factory<B>,
        families: &mut Families<B>,
        aux: &T,
    ) -> Result<(), GraphRunError> {
        unsafe {
            self.alloc.reset();
        }

        let ref mut nodes = self.nodes;
        let mut run_ctx = RunContext::new(factory, families, &self.alloc);
        nodes.run_construction_phase(&mut run_ctx, aux)?;

        self.pipeline.reduce(&mut run_ctx.graph.dag, &self.alloc);

        run_ctx.run_execution_phase(aux)?;

        // Graph lowering
        // All graph excution types are eventually becoming a "General" nodes
        // Types at "higher level of abstraction" can be manipulated by reducers to perform some optimization
        // There are going ot be:
        // - Outputs: translated stright to General, just a resource management concept.
        // - render pass nodes -> lowered into grouped passes
        // - grouped passes -> eventually lowered into general
        //
        // All specialized general are actually just closures handling some specific case

        // Joining render passes:
        // - visit render pass node/folded pass node
        // - traverse through attachment resources
        // - if all attachments have a single common parent which is another pass/folded, combine
        // Needed operation:
        // - combine two connected nodes -> replace current node with other

        // TODO: Reduce graph (GraphReducer)
        // Apply "graph transformations" by pattern matching (optimization passes):
        //  - discard nodes that don't contribute to output
        //  - reorder render pass nodes to always be as close together as possible
        //    starting from output, for every renderpass node:
        //     - if there was a node with same attachment already visited and current node doesn't depend on it in other way than attachments, reorder that new node to be directly after current node
        //  - combine sequential renderpass nodes into single

        // TODO: allocate resources

        // TODO: schedule/run

        // TODO: traverse graph from resources accessed by output nodes.

        // # Some ideas about approaching this:
        // Treat resources and evals as dag nodes (TODO: currently resources are edges, this might be not ok)
        // Every resource mutation creates new output dag node, that points to parent resource.
        // - Resource mutations that effectively overwrite the whole resource without reading can be considered entirely new resources.
        //     - how to detect?, is this situation even possible with current API? If not, this can just be ignored.
        // The resource can know all nodes it's used in, that way it's vulkan object can be reused later
        // - that means graph "resources" are possibly just labels, decoupled from real underlying representation. We can do some "register allocation" on that.
        // - having a resource "span" allows to trivially reuse the same chunk of buffer/image for multiple resources.
        // - Resource that's never cleared, so used in next frame needs to be not overwritten. It can be detected and treated as "infinite span".
        //
        // First node that uses resource actually outputs the resource node. The use resource definitions are copied into the node, along with needed access pattern.
        //
        // The graph builder API guarantees that the nodes are topologically sorted, but might contain nodes not relevant to outputs.
        // Those should be first filtered out.
        // Every resource write creates conceptually new resource. Write based on `a` creates `a2` which shares it's resource info.
        // Nodes that write "a'" depend on all readers of "a", because the reads must be completed before overwrite.
        // - if this is really costly (how to estimate?), a resource copy operation can be inserted to decouple `a` from `a2`.
        // A resource copy nodes are really just kinda "resource renames" on surface api level.
        // That allows expressing graph that wants to use two versions of same resource using two separate resource labels.
        // That means resource copy nodes can be "noop" if the final node order doesn't actually require two copies to exist.

        unimplemented!()
    }
}

impl<B: Backend, T: ?Sized> GraphNodes<B, T> {
    #[inline(never)]
    fn run_construction_phase<'run, 'arena>(
        &'run mut self,
        run_ctx: &mut RunContext<'run, 'arena, B, T>,
        aux: &T,
    ) -> Result<(), GraphRunError> {
        // insert all nodes in their original order, except outputs that are inserted last
        let mut outputs = SmallVec::<[_; 8]>::new();

        for (i, node) in self.0.iter_mut().enumerate() {
            let mut seed = run_ctx.graph.seed();

            let execution = {
                let mut ctx = NodeContext::new(NodeId(i), &mut seed, run_ctx, aux);
                node.construct(&mut ctx)
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
pub(crate) struct NodeContext<'ctx, 'run, 'arena, B: Backend, T: ?Sized> {
    id: NodeId,
    seed: &'ctx mut NodeSeed,
    run: &'ctx mut RunContext<'run, 'arena, B, T>,
    next_image_id: usize,
    next_buffer_id: usize,
    aux: &'ctx T,
}

/// A token that allows an image to be used in node execution
#[derive(Debug, Clone, Copy)]
pub struct ImageToken<'run>(ImageId, PhantomData<&'run ()>);

/// A token that allows a buffer to be used in node execution
#[derive(Debug, Clone, Copy)]
pub struct BufferToken<'run>(BufferId, PhantomData<&'run ()>);

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

pub trait NodeCtx<'run, B: Backend, T: ?Sized> {
    fn factory(&self) -> &Factory<B>;
    fn aux(&self) -> &T;

    fn get_parameter<P: Any>(&self, id: Parameter<P>) -> Result<&P, NodeConstructionError>;

    /// Create new image owned by graph.
    fn create_image(&mut self, image_info: ImageInfo) -> ImageId;

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
}

impl<'ctx, 'run, 'arena, B: Backend, T: ?Sized> NodeContext<'ctx, 'run, 'arena, B, T> {
    fn new(
        id: NodeId,
        seed: &'ctx mut NodeSeed,
        run: &'ctx mut RunContext<'run, 'arena, B, T>,
        aux: &'ctx T,
    ) -> Self {
        Self {
            id,
            seed,
            run,
            next_image_id: 0,
            next_buffer_id: 0,
            aux,
        }
    }

    pub(crate) fn set_outputs(&mut self, vals: impl Iterator<Item = Box<dyn Any>>) {
        self.run.output_store.set_all(self.id, vals);
    }
}

impl<'run, B: Backend, T: ?Sized> NodeCtx<'run, B, T> for NodeContext<'_, 'run, '_, B, T> {
    fn factory(&self) -> &Factory<B> {
        &self.run.factory
    }

    fn aux(&self) -> &T {
        &self.aux
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
        self.run.graph.create_image(id, image_info)
    }

    fn create_buffer(&mut self, buffer_info: BufferInfo) -> BufferId {
        let id = BufferId(self.id, self.next_buffer_id);
        self.next_buffer_id += 1;
        self.run.graph.create_buffer(id, buffer_info)
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
}

#[derive(Debug, Clone)]
pub(crate) struct ImageNode {
    // The ID is only used to later retreive the image in execution phase
    pub(crate) id: ImageId,
    pub(crate) kind: gfx_hal::image::Kind,
    pub(crate) levels: gfx_hal::image::Level,
    pub(crate) format: gfx_hal::format::Format,
}

#[derive(Debug, Clone)]
pub(crate) struct BufferNode {
    pub(crate) id: BufferId,
    pub(crate) size: u64,
}

pub(crate) enum PlanNode<'n, B: Backend, T: ?Sized> {
    /// Construction phase execution
    // Execution(NodeExecution<'a, B, T>),
    /// Construction phase resource
    // Resource(ResourceId),
    Image(ImageNode),
    Buffer(BufferNode),
    ImageVersion,
    BufferVersion,
    /// Image clear operation
    ClearImage(gfx_hal::command::ClearValue),
    /// Buffer clear operation
    ClearBuffer(u32),
    /// Load image rendered in previous frame
    LoadImage(NodeId, usize),
    /// Store image for usage in next frame
    StoreImage(NodeId, usize),
    UndefinedImage,
    UndefinedBuffer,
    /// A subpass that might have multiple render groups.
    RenderSubpass(SmallVec<[(NodeId, PassFn<'n, B, T>); 4]>),
    /// A render pass - group of subpasses with dependencies between them
    RenderPass(RenderPassNode<'n, B, T>),
    /// A node representing arbitrary runnable operation.
    Run(NodeId, GeneralFn<'n, B, T>),
    /// Resolve multisampled image into non-multisampled one.
    /// Currently this works under asumption that all layers of source image are resolved into first layer of destination.
    ResolveImage,
    /// Graph root node. All incoming nodes are always evaluated.
    Root,
}

impl<'n, B: Backend, T: ?Sized> PlanNode<'n, B, T> {
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

    pub(crate) fn pass_mut(&mut self) -> Option<&mut RenderPassNode<'n, B, T>> {
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
    pub(crate) fn layout(&self) -> gfx_hal::image::Layout {
        match self {
            AttachmentRefEdge::Color(..) => gfx_hal::image::Layout::ColorAttachmentOptimal,
            AttachmentRefEdge::Resolve(..) => gfx_hal::image::Layout::ColorAttachmentOptimal,
            AttachmentRefEdge::Input(..) => gfx_hal::image::Layout::ShaderReadOnlyOptimal,
            AttachmentRefEdge::DepthStencil(access) => match access {
                AttachmentAccess::ReadOnly => gfx_hal::image::Layout::DepthStencilReadOnlyOptimal,
                AttachmentAccess::ReadWrite => {
                    gfx_hal::image::Layout::DepthStencilAttachmentOptimal
                }
            },
        }
    }
    #[inline]
    pub(crate) fn accesses(&self) -> gfx_hal::image::Access {
        match self {
            AttachmentRefEdge::Color(..) => {
                gfx_hal::image::Access::COLOR_ATTACHMENT_READ
                    | gfx_hal::image::Access::COLOR_ATTACHMENT_WRITE
            }
            AttachmentRefEdge::Resolve(..) => gfx_hal::image::Access::COLOR_ATTACHMENT_WRITE,
            AttachmentRefEdge::Input(..) => gfx_hal::image::Access::INPUT_ATTACHMENT_READ,
            AttachmentRefEdge::DepthStencil(access) => match access {
                AttachmentAccess::ReadOnly => gfx_hal::image::Access::DEPTH_STENCIL_ATTACHMENT_READ,
                AttachmentAccess::ReadWrite => {
                    gfx_hal::image::Access::DEPTH_STENCIL_ATTACHMENT_READ
                        | gfx_hal::image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE
                }
            },
        }
    }
    #[inline]
    pub(crate) fn stages(&self) -> gfx_hal::pso::PipelineStage {
        match self {
            // TODO: can it be declared if color is used as write-only?
            AttachmentRefEdge::Color(..) => gfx_hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
            AttachmentRefEdge::Resolve(..) => gfx_hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
            AttachmentRefEdge::Input(..) => gfx_hal::pso::PipelineStage::FRAGMENT_SHADER,
            AttachmentRefEdge::DepthStencil(_) => {
                // TODO: can it be declared which fragment test is used?
                gfx_hal::pso::PipelineStage::EARLY_FRAGMENT_TESTS
                    | gfx_hal::pso::PipelineStage::LATE_FRAGMENT_TESTS
            }
        }
    }
}

pub(crate) struct RenderPassSubpass<'n, B: Backend, T: ?Sized> {
    pub(crate) groups: SmallVec<[(NodeId, PassFn<'n, B, T>); 4]>,
    pub(crate) colors: SmallVec<[gfx_hal::pass::AttachmentRef; 8]>,
    pub(crate) inputs: SmallVec<[gfx_hal::pass::AttachmentRef; 8]>,
    pub(crate) resolves: SmallVec<[gfx_hal::pass::AttachmentRef; 4]>,
    pub(crate) depth_stencil: gfx_hal::pass::AttachmentRef,
}

impl<'n, B: Backend, T: ?Sized> std::fmt::Debug for RenderPassSubpass<'n, B, T> {
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
impl<'n, B: Backend, T: ?Sized> RenderPassSubpass<'n, B, T> {
    pub(crate) fn new(groups: SmallVec<[(NodeId, PassFn<'n, B, T>); 4]>) -> Self {
        Self {
            groups,
            colors: SmallVec::new(),
            inputs: SmallVec::new(),
            resolves: SmallVec::new(),
            depth_stencil: (ATTACHMENT_UNUSED, gfx_hal::image::Layout::Undefined),
        }
    }
    pub(crate) fn set_color(
        &mut self,
        ref_index: usize,
        attachment_ref: gfx_hal::pass::AttachmentRef,
    ) {
        if self.colors.len() <= ref_index {
            self.colors.resize(
                ref_index + 1,
                (ATTACHMENT_UNUSED, gfx_hal::image::Layout::Undefined),
            );
        }
        self.colors[ref_index] = attachment_ref;
    }
    pub(crate) fn set_resolve(
        &mut self,
        ref_index: usize,
        attachment_ref: gfx_hal::pass::AttachmentRef,
    ) {
        if self.resolves.len() <= ref_index {
            self.resolves.resize(
                ref_index + 1,
                (ATTACHMENT_UNUSED, gfx_hal::image::Layout::Undefined),
            );
        }
        self.colors[ref_index] = attachment_ref;
    }
    pub(crate) fn set_input(
        &mut self,
        ref_index: usize,
        attachment_ref: gfx_hal::pass::AttachmentRef,
    ) {
        if self.inputs.len() <= ref_index {
            self.inputs.resize(
                ref_index + 1,
                (ATTACHMENT_UNUSED, gfx_hal::image::Layout::Undefined),
            );
        }
        self.inputs[ref_index] = attachment_ref;
    }
    pub(crate) fn set_depth_stencil(&mut self, attachment_ref: gfx_hal::pass::AttachmentRef) {
        self.depth_stencil = attachment_ref;
    }
}

#[derive(Debug)]
pub(crate) struct RenderPassNode<'n, B: Backend, T: ?Sized> {
    pub(crate) subpasses: SmallVec<[RenderPassSubpass<'n, B, T>; 4]>,
    pub(crate) deps: SmallVec<[gfx_hal::pass::SubpassDependency; 32]>,
}

impl<'n, B: Backend, T: ?Sized> RenderPassNode<'n, B, T> {
    pub(crate) fn new() -> Self {
        Self {
            subpasses: SmallVec::new(),
            deps: SmallVec::new(),
        }
    }
}

// first_access/last_access is a way to approximate which subpasses need to preserve attachments.
// It might have false positives, but this should preserve correctness and is better than nothing.

// This is dogshit. Render pass doesn't have the "type" of attachment already figured out upfront.
// Same thing can be a color and an input (or even a resolve) for two different subpasses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RenderPassAtachment {
    pub(crate) ops: gfx_hal::pass::AttachmentOps,
    pub(crate) stencil_ops: gfx_hal::pass::AttachmentOps,
    pub(crate) clear: AttachmentClear,
    pub(crate) first_access: u8,
    pub(crate) last_access: u8,
    pub(crate) first_write: Option<(u8, gfx_hal::pso::PipelineStage, gfx_hal::image::Access)>,
    pub(crate) last_write: Option<(u8, gfx_hal::pso::PipelineStage, gfx_hal::image::Access)>,
    pub(crate) queued_reads:
        SmallVec<[(u8, gfx_hal::pso::PipelineStage, gfx_hal::image::Access); 4]>,
    pub(crate) first_layout: gfx_hal::image::Layout,
}

impl RenderPassAtachment {
    pub(crate) fn is_write(&self) -> bool {
        self.first_write.is_some()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AttachmentClear(pub(crate) Option<gfx_hal::command::ClearValue>);

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
    BufferAccess(NodeBufferAccess, gfx_hal::pso::PipelineStage),
    ImageAccess(NodeImageAccess, gfx_hal::pso::PipelineStage),
    /// A node-to-node execution order dependency
    Effect,
    /// Resource version relation. Version increases in the edge direction
    Version,
    /// Link between a resource and a node that is responsible for it's creation.
    Origin,
}

impl<'n, B: Backend, T: ?Sized> std::fmt::Debug for PlanNode<'n, B, T> {
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
            PlanNode::Run(..) => fmt.debug_tuple("Run").finish(),
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

pub(crate) type PlanDag<'run, 'arena, B, T> = graphy::Graph<'arena, PlanNode<'run, B, T>, PlanEdge>;

#[derive(Debug)]
pub struct PlanGraph<'run, 'arena, B: Backend, T: ?Sized> {
    dag: PlanDag<'run, 'arena, B, T>,
    alloc: &'arena GraphAllocator,
    virtuals: usize,
    last_writes: HashMap<ResourceId, NodeIndex>,
    resource_usage: Vec<ResourceUsage>,
    retained_images: HashMap<(NodeId, usize), Handle<Image<B>>>,
}

impl<'run, 'arena, B: Backend, T: ?Sized> PlanGraph<'run, 'arena, B, T> {
    fn new(alloc: &'arena GraphAllocator) -> Self {
        let mut dag = PlanDag::new();
        // guaranteed to always be index 0
        dag.insert_node(alloc, PlanNode::Root);

        Self {
            dag,
            alloc,
            virtuals: 0,
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

    /// Provide a node-managed resource to the graph.
    pub(crate) fn provide_image(
        &mut self,
        source: NodeId,
        info: ImageInfo,
        handle: Handle<Image<B>>,
    ) -> ImageId {
        unimplemented!()
    }

    /// Create new image owned by graph. The image lifecycle is totally controlled by the graph.
    pub(crate) fn create_image(&mut self, id: ImageId, info: ImageInfo) -> ImageId {
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
        id
    }
    /// Create new buffer owned by graph.
    pub(crate) fn create_buffer(&mut self, id: BufferId, info: BufferInfo) -> BufferId {
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
        id
    }
    /// Create non-data dependency target. A virtual resource intended to
    /// describe dependencies between rendering nodes without carrying any data.
    pub(crate) fn create_virtual(&mut self) -> VirtualId {
        self.virtuals += 1;
        VirtualId(self.virtuals - 1)
    }

    fn insert_node(&mut self, node: PlanNode<'run, B, T>) -> NodeIndex {
        self.dag.insert_node(self.alloc, node)
    }

    fn insert_child(
        &mut self,
        parent: NodeIndex,
        edge: PlanEdge,
        node: PlanNode<'run, B, T>,
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
        exec: NodeExecution<'run, B, T>,
    ) -> Result<(), NodeConstructionError> {
        // inserts should always happen in such order that resource usages are free to be drained
        debug_assert!(self.resource_usage.len() == seed.resources.end);

        // insert execution node
        let (node, allow_attachments) = match exec {
            NodeExecution::RenderPass(group) => (
                self.insert_node(PlanNode::RenderSubpass(smallvec![(node_id, group)])),
                true,
            ),
            NodeExecution::General(run) => (self.insert_node(PlanNode::Run(node_id, run)), false),
            NodeExecution::Output(run) => {
                let node = self.insert_node(PlanNode::Run(node_id, run));
                self.insert_edge(node, NodeIndex::new(0), PlanEdge::Effect);
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
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct RunContext<'run, 'arena, B: Backend, T: ?Sized> {
    // not really 'arena, but has to live at least as long, so it's fine
    pub(crate) factory: &'arena Factory<B>,
    pub(crate) families: &'arena Families<B>,
    pub(crate) output_store: OutputStore,
    graph: PlanGraph<'run, 'arena, B, T>,
}

impl<'run, 'arena, B: Backend, T: ?Sized> RunContext<'run, 'arena, B, T> {
    pub(crate) fn new(
        factory: &'arena Factory<B>,
        families: &'arena Families<B>,
        allocator: &'arena GraphAllocator,
    ) -> Self {
        Self {
            factory,
            families,
            output_store: OutputStore::new(),
            graph: PlanGraph::new(allocator),
        }
    }

    fn run_execution_phase(self, aux: &T) -> Result<(), GraphRunError> {
        let topo: Vec<_> = TopoWithEdges::new(&self.graph.dag, NodeIndex::new(0))
            .iter(&self.graph.dag)
            .collect();

        let (mut nodes, mut edges) = self.graph.dag.into_items();

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
                        PlanNode::RenderPass(..) => {}
                        PlanNode::Run(node_id, closure) => {
                            let ctx = ExecContext::new();
                            closure(ctx, aux)
                                .map_err(|e| GraphRunError::NodeExecution(node_id, e))?
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
            test::{test_init, visualize_graph, TestBackend},
        },
    };

    impl NodeBuilder<TestBackend, ()> for ImageInfo {
        type Node = Self;
        type Family = crate::command::Transfer;
        fn build(
            self: Box<Self>,
            _: &mut Factory<TestBackend>,
            _: &mut Family<TestBackend>,
            _: &(),
        ) -> Result<Self::Node, NodeBuildError> {
            Ok(*self)
        }
    }

    impl Node<TestBackend, ()> for ImageInfo {
        type Outputs = Parameter<ImageId>;
        fn construct<'run>(
            &'run mut self,
            ctx: &mut impl NodeCtx<'run, TestBackend, ()>,
        ) -> ConstructResult<'run, Self, TestBackend, ()> {
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

    impl NodeBuilder<TestBackend, ()> for TestPass {
        type Node = Self;
        type Family = crate::command::Graphics;
        fn build(
            self: Box<Self>,
            _: &mut Factory<TestBackend>,
            _: &mut Family<TestBackend>,
            _: &(),
        ) -> Result<Self::Node, NodeBuildError> {
            Ok(*self)
        }
    }

    impl Node<TestBackend, ()> for TestPass {
        type Outputs = ();

        fn construct<'run>(
            &'run mut self,
            ctx: &mut impl NodeCtx<'run, TestBackend, ()>,
        ) -> ConstructResult<'run, Self, TestBackend, ()> {
            let color = *ctx.get_parameter(self.color)?;
            ctx.use_color(0, color)?;
            if let Some(depth) = self.depth {
                let depth = *ctx.get_parameter(depth)?;
                ctx.use_depth(depth, true)?;
            }

            Ok(((), NodeExecution::pass(|_, _| Ok(()))))
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

    impl NodeBuilder<TestBackend, ()> for TestPass2 {
        type Node = Self;
        type Family = crate::command::Graphics;
        fn build(
            self: Box<Self>,
            _: &mut Factory<TestBackend>,
            _: &mut Family<TestBackend>,
            _: &(),
        ) -> Result<Self::Node, NodeBuildError> {
            Ok(*self)
        }
    }

    impl Node<TestBackend, ()> for TestPass2 {
        type Outputs = ();

        fn construct<'run>(
            &'run mut self,
            ctx: &mut impl NodeCtx<'run, TestBackend, ()>,
        ) -> ConstructResult<'run, Self, TestBackend, ()> {
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

            Ok(((), NodeExecution::pass(|_, _| Ok(()))))
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

    impl NodeBuilder<TestBackend, ()> for TestCompute1 {
        type Node = Self;
        type Family = crate::command::Graphics;
        fn build(
            self: Box<Self>,
            _: &mut Factory<TestBackend>,
            _: &mut Family<TestBackend>,
            _: &(),
        ) -> Result<Self::Node, NodeBuildError> {
            Ok(*self)
        }
    }

    impl Node<TestBackend, ()> for TestCompute1 {
        type Outputs = Parameter<BufferId>;

        fn construct<'run>(
            &'run mut self,
            ctx: &mut impl NodeCtx<'run, TestBackend, ()>,
        ) -> ConstructResult<'run, Self, TestBackend, ()> {
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
                NodeExecution::pass(move |_, _| {
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

    impl NodeBuilder<TestBackend, ()> for TestPass3 {
        type Node = Self;
        type Family = crate::command::Graphics;
        fn build(
            self: Box<Self>,
            _: &mut Factory<TestBackend>,
            _: &mut Family<TestBackend>,
            _: &(),
        ) -> Result<Self::Node, NodeBuildError> {
            Ok(*self)
        }
    }

    impl Node<TestBackend, ()> for TestPass3 {
        type Outputs = ();

        fn construct<'run>(
            &'run mut self,
            ctx: &mut impl NodeCtx<'run, TestBackend, ()>,
        ) -> ConstructResult<'run, Self, TestBackend, ()> {
            let color = *ctx.get_parameter(self.color)?;
            let buffer = *ctx.get_parameter(self.buffer)?;

            ctx.use_color(0, color)?;
            if self.write_buffer {
                ctx.use_buffer(buffer, BufferUsage::StorageWrite(ShaderUsage::FRAGMENT))?;
            } else {
                ctx.use_buffer(buffer, BufferUsage::StorageRead(ShaderUsage::FRAGMENT))?;
            }

            Ok(((), NodeExecution::pass(|_, _| Ok(()))))
        }
    }

    #[derive(Debug)]
    struct TestOutput;

    impl NodeBuilder<TestBackend, ()> for TestOutput {
        type Node = Self;
        type Family = crate::command::Graphics;
        fn build(
            self: Box<Self>,
            _: &mut Factory<TestBackend>,
            _: &mut Family<TestBackend>,
            _: &(),
        ) -> Result<Self::Node, NodeBuildError> {
            Ok(*self)
        }
    }

    impl Node<TestBackend, ()> for TestOutput {
        type Outputs = Parameter<ImageId>;
        fn construct<'run>(
            &'run mut self,
            ctx: &mut impl NodeCtx<'run, TestBackend, ()>,
        ) -> ConstructResult<'run, Self, TestBackend, ()> {
            let output = ctx.create_image(ImageInfo {
                kind: gfx_hal::image::Kind::D2(1024, 1024, 1, 1),
                levels: 1,
                format: gfx_hal::format::Format::Rgba8Unorm,
                load: ImageLoad::DontCare,
            });
            let output_use = ctx.use_image(output, ImageUsage::ColorAttachmentRead)?;
            Ok((
                output,
                NodeExecution::output(move |ctx, _| {
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
        let (mut factory, mut families): (Factory<TestBackend>, _) =
            crate::factory::init(config).unwrap();

        let mut builder = GraphBuilder::new();

        let depth = builder.add(ImageInfo {
            kind: gfx_hal::image::Kind::D2(1, 1, 1, 1),
            levels: 1,
            format: gfx_hal::format::Format::R32Sfloat,
            load: ImageLoad::Clear(gfx_hal::command::ClearValue {
                depth_stencil: gfx_hal::command::ClearDepthStencil {
                    depth: 0.0,
                    stencil: 0,
                },
            }),
        });
        let depth2 = builder.add(ImageInfo {
            kind: gfx_hal::image::Kind::D2(1, 1, 1, 1),
            levels: 1,
            format: gfx_hal::format::Format::R32Sfloat,
            load: ImageLoad::DontCare,
        });
        let color2 = builder.add(ImageInfo {
            kind: gfx_hal::image::Kind::D2(1, 1, 1, 1),
            levels: 1,
            format: gfx_hal::format::Format::Rgba8Unorm,
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
        let mut run_ctx = RunContext::new(&factory, &families, &graph.alloc);
        graph
            .nodes
            .run_construction_phase(&mut run_ctx, &())
            .unwrap();

        let mut file = std::fs::File::create("graph.dot").unwrap();
        visualize_graph(&mut file, &run_ctx.graph.dag, "raw");

        graph.pipeline.reduce(&mut run_ctx.graph.dag, &graph.alloc);
        visualize_graph(&mut file, &run_ctx.graph.dag, "opti");
        run_ctx.run_execution_phase(&()).unwrap();
    }
}
