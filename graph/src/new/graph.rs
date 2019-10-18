use {
    super::{
        node::{
            DynNode, DynNodeBuilder, GeneralFn, InternalUse, Node, NodeBuildError, NodeBuilder,
            NodeConstructionError, NodeExecution, NodeExecutionError, NodeId, OutputList,
            OutputStore, Parameter, PassFn,
        },
        pipeline::Pipeline,
        resources::{
            BufferId, BufferInfo, BufferUsage, ImageId, ImageInfo, ImageUsage, NodeBufferAccess,
            NodeImageAccess, NodeVirtualAccess, ResourceId, ResourceUsage, VirtualId,
        },
    },
    crate::{command::Families, factory::Factory},
    gfx_hal::Backend,
    graphy::{EdgeIndex, GraphAllocator, NodeIndex},
    smallvec::{smallvec, SmallVec},
    std::{any::Any, collections::HashMap, ops::Range},
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

        let mut run_ctx = RunContext::new(factory, families, &self.alloc);
        self.nodes.run_construction_phase(&mut run_ctx, aux)?;

        self.pipeline.reduce(&mut run_ctx.graph.dag, &self.alloc);

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
    fn run_construction_phase<'a: 'b, 'b>(
        &'a mut self,
        run_ctx: &mut RunContext<'b, B, T>,
        aux: &T,
    ) -> Result<(), GraphRunError> {
        // insert all nodes in their original order, except outputs that are inserted last
        let mut outputs = SmallVec::<[_; 8]>::new();

        for (i, node) in self.0.iter_mut().enumerate() {
            let mut seed = run_ctx.graph.seed();

            let execution = {
                let mut ctx = NodeContext {
                    id: NodeId(i),
                    run: run_ctx,
                    seed: &mut seed,
                };
                node.construct(&mut ctx, aux)
                    .map_err(|e| GraphRunError::NodeConstruction(NodeId(i), e))?
            };

            if execution.is_output() {
                outputs.push((seed, execution));
            } else {
                run_ctx.graph.insert(seed, execution);
            }
        }

        for (seed, execution) in outputs.drain().rev() {
            run_ctx.graph.insert(seed, execution);
        }

        Ok(())
    }
}

/// A context for rendergraph node construction phase. Contains all data that the node
/// get access to and contains ready-made methods for common operations.
#[derive(Debug)]
pub struct NodeContext<'a, 'b, B: Backend, T: ?Sized> {
    id: NodeId,
    seed: &'a mut NodeSeed,
    run: &'a mut RunContext<'b, B, T>,
}

impl<'a, 'b, B: Backend, T: ?Sized> NodeContext<'a, 'b, B, T> {
    pub(crate) fn set_outputs(&mut self, vals: impl Iterator<Item = Box<dyn Any>>) {
        self.run.output_store.set_all(self.id, vals);
    }

    pub fn factory(&self) -> &'a Factory<B> {
        self.run.factory
    }
    pub fn get_parameter<P: Any>(&self, id: Parameter<P>) -> Result<&P, NodeConstructionError> {
        self.run
            .output_store
            .get(id)
            .ok_or(NodeConstructionError::VariableReadFailed(id.0))
    }
    /// Create new image owned by graph.
    pub fn create_image(&mut self, image_info: ImageInfo) -> ImageId {
        self.run.graph.create_image(image_info)
    }
    /// Create new buffer owned by graph.
    pub fn create_buffer(&mut self, buffer_info: BufferInfo) -> BufferId {
        self.run.graph.create_buffer(buffer_info)
    }
    /// Create non-data dependency target. A virtual resource intended to
    /// describe dependencies between rendering nodes without carrying any data.
    pub fn create_virtual(&mut self) -> VirtualId {
        self.run.graph.create_virtual()
    }
    pub fn use_virtual(&mut self, id: VirtualId, access: NodeVirtualAccess) {
        self.run
            .graph
            .use_resource(self.seed, ResourceUsage::Virtual(id, access));
    }
    /// Declare usage of image by the node
    pub fn use_image(&mut self, id: ImageId, usage: ImageUsage) {
        self.run
            .graph
            .use_resource(self.seed, usage.resource_usage(id));
    }
    /// Declare usage of buffer by the node
    pub fn use_buffer(&mut self, id: BufferId, usage: BufferUsage) {
        self.run
            .graph
            .use_resource(self.seed, usage.resource_usage(id));
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ImageNode {
    pub(crate) kind: gfx_hal::image::Kind,
    pub(crate) levels: gfx_hal::image::Level,
    pub(crate) format: gfx_hal::format::Format,
}

#[derive(Debug, Clone)]
pub(crate) struct BufferNode {
    pub(crate) size: u64,
}

// struct SubpassDesc {}

// struct PassNode {
//     groups: SmallVec<[PassFn<'n, B, T>; 4]>,
// }

pub(crate) enum PlanNodeData<'n, B: Backend, T: ?Sized> {
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
    UndefinedImage,
    UndefinedBuffer,
    /// A subpass that might have multiple render groups.
    RenderSubpass(SmallVec<[PassFn<'n, B, T>; 4]>),
    // RenderPass(SmallVec<[PassFn<'n, B, T>; 4]>),
    /// A node representing arbitrary runnable operation. All Op nodes are eventually lowered into it.
    Run(GeneralFn<'n, B, T>),
    /// Resolve multisampled image into non-multisampled one.
    /// Currently this works under asumption that all layers of source image are resolved into first layer of destination.
    ResolveImage,
    /// Placeholder value required to take out the node out of graph just before replacing it
    Tombstone,
    /// Graph root node. All incoming nodes are always evaluated.
    Root,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Attachment {
    Input {
        index: usize,
        ops: gfx_hal::pass::AttachmentOps,
    },
    DepthStencil {
        index: usize,
        ops: gfx_hal::pass::AttachmentOps,
        stecil_ops: gfx_hal::pass::AttachmentOps,
    },
    Color {
        index: usize,
        ops: gfx_hal::pass::AttachmentOps,
    },
    // TODO: Resolve attachments should be inferred from...
    // - using multiple samples on non-msaa targets?
    // - using SINGLE sample on msaa targets (resolve in subpass before)
    // - using msaa on multisampling
    Resolve {
        index: usize,
        ops: gfx_hal::pass::AttachmentOps,
    },
    // TODO: preserve attachments should be autoinferred based on subpass joining
    Preserve,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PlanEdge {
    /// Target node accesses connected source image version.
    ImageAccess(NodeImageAccess, gfx_hal::pso::PipelineStage),
    BufferAccess(NodeBufferAccess, gfx_hal::pso::PipelineStage),
    /// A node-to-node execution order dependency
    Effect,
    /// Resource version relation. Version increases in the edge direction
    Version,
    /// Link between a resource and a node that is responsible for it's creation.
    /// TODO: can this be just an effect?
    Origin,
}

impl<'n, B: Backend, T: ?Sized> std::fmt::Debug for PlanNodeData<'n, B, T> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanNodeData::Image(node) => fmt.debug_tuple("Image").field(node).finish(),
            PlanNodeData::Buffer(node) => fmt.debug_tuple("Buffer").field(node).finish(),
            PlanNodeData::ClearImage(c) => fmt.debug_tuple("ClearImage").field(c).finish(),
            PlanNodeData::ClearBuffer(c) => fmt.debug_tuple("ClearBuffer").field(c).finish(),
            PlanNodeData::ImageVersion => fmt.debug_tuple("ImageVersion").finish(),
            PlanNodeData::BufferVersion => fmt.debug_tuple("BufferVersion").finish(),
            PlanNodeData::UndefinedImage => fmt.debug_tuple("UndefinedImage").finish(),
            PlanNodeData::UndefinedBuffer => fmt.debug_tuple("UndefinedBuffer").finish(),
            PlanNodeData::RenderSubpass(vec) => fmt
                .debug_tuple(&format!("RenderSubpass[{}]", vec.len()))
                .finish(),
            PlanNodeData::Run(_) => fmt.debug_tuple("Run").finish(),
            PlanNodeData::ResolveImage => fmt.debug_tuple("ResolveImage").finish(),
            PlanNodeData::Tombstone => fmt.debug_tuple("Tombstone").finish(),
            PlanNodeData::Root => fmt.debug_tuple("Root").finish(),
        }
    }
}

impl PlanEdge {
    pub(crate) fn is_attachment_only(&self) -> bool {
        match self {
            PlanEdge::ImageAccess(usage, _) => usage.is_attachment_only(),
            _ => false,
        }
    }

    pub(crate) fn is_attachment(&self) -> bool {
        match self {
            PlanEdge::ImageAccess(usage, _) => usage.is_attachment(),
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
struct NodeSeed {
    resources: Range<usize>,
}

pub(crate) type PlanDag<'a, B, T> = graphy::Graph<'a, PlanNodeData<'a, B, T>, PlanEdge>;

#[derive(Debug)]
pub struct PlanGraph<'a, B: Backend, T: ?Sized> {
    dag: PlanDag<'a, B, T>,
    alloc: &'a GraphAllocator,
    images: Vec<ImageInfo>,
    buffers: Vec<BufferInfo>,
    processed_images: usize,
    processed_buffers: usize,
    virtuals: usize,
    last_writes: HashMap<ResourceId, NodeIndex>,
    resource_usage: Vec<ResourceUsage>,
}

impl<'a, B: Backend, T: ?Sized> PlanGraph<'a, B, T> {
    fn new(alloc: &'a GraphAllocator) -> Self {
        let mut dag = PlanDag::new();
        // guaranteed to always be index 0
        dag.insert_node(alloc, PlanNodeData::Root);

        Self {
            dag,
            alloc,
            images: Vec::new(),
            buffers: Vec::new(),
            processed_images: 0,
            processed_buffers: 0,
            virtuals: 0,
            last_writes: HashMap::new(),
            resource_usage: Vec::new(),
        }
    }

    fn seed(&self) -> NodeSeed {
        let len = self.resource_usage.len();
        NodeSeed {
            resources: len..len,
        }
    }

    fn use_resource(&mut self, seed: &mut NodeSeed, usage: ResourceUsage) {
        if let Some(res) = self.resource_usage[seed.resources.clone()]
            .iter_mut()
            .find(|r| r.is_same_resource(&usage))
        {
            res.merge_access(&usage);
        } else {
            self.resource_usage.push(usage);
            seed.resources.end += 1;
        }
    }

    /// Create new image owned by graph.
    pub(crate) fn create_image(&mut self, image_info: ImageInfo) -> ImageId {
        self.images.push(image_info);
        ImageId(self.processed_images + self.images.len() - 1)
    }
    /// Create new buffer owned by graph.
    pub(crate) fn create_buffer(&mut self, buffer_info: BufferInfo) -> BufferId {
        self.buffers.push(buffer_info);
        BufferId(self.processed_buffers + self.buffers.len() - 1)
    }
    /// Create non-data dependency target. A virtual resource intended to
    /// describe dependencies between rendering nodes without carrying any data.
    pub(crate) fn create_virtual(&mut self) -> VirtualId {
        self.virtuals += 1;
        VirtualId(self.virtuals - 1)
    }

    fn insert_node(&mut self, node: PlanNodeData<'a, B, T>) -> NodeIndex {
        self.dag.insert_node(self.alloc, node)
    }

    fn insert_child(
        &mut self,
        parent: NodeIndex,
        edge: PlanEdge,
        node: PlanNodeData<'a, B, T>,
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

    fn process_resources(&mut self) {
        for info in self.images.drain(..) {
            let id = self.processed_images;
            let init_data = if let Some(clear) = info.clear {
                PlanNodeData::ClearImage(clear)
            } else {
                PlanNodeData::UndefinedImage
            };

            let def = self.dag.insert_node(
                self.alloc,
                PlanNodeData::Image(ImageNode {
                    kind: info.kind,
                    levels: info.levels,
                    format: info.format,
                }),
            );
            let (_, init) = self
                .dag
                .insert_child(self.alloc, def, PlanEdge::Origin, init_data);
            self.last_writes
                .insert(ResourceId::Image(ImageId(id)), init);
            self.processed_images += 1;
        }
        for info in self.buffers.drain(..) {
            let id = self.processed_buffers;
            let init_data = if let Some(clear) = info.clear {
                PlanNodeData::ClearBuffer(clear)
            } else {
                PlanNodeData::UndefinedBuffer
            };

            let def = self.dag.insert_node(
                self.alloc,
                PlanNodeData::Buffer(BufferNode { size: info.size }),
            );
            let (_, init) = self
                .dag
                .insert_child(self.alloc, def, PlanEdge::Origin, init_data);
            self.last_writes
                .insert(ResourceId::Image(ImageId(id)), init);
            self.processed_buffers += 1;
        }
    }

    fn insert(&mut self, seed: NodeSeed, exec: NodeExecution<'a, B, T>) {
        // inserts should always happen in such order that resource usages are free to be drained
        debug_assert!(self.resource_usage.len() == seed.resources.end);

        // insert execution node
        let node = match exec {
            NodeExecution::RenderPass(group) => {
                self.insert_node(PlanNodeData::RenderSubpass(smallvec![group]))
            }
            NodeExecution::General(run) => self.insert_node(PlanNodeData::Run(run)),
            NodeExecution::Output(run) => {
                let node = self.insert_node(PlanNodeData::Run(run));
                self.insert_edge(node, NodeIndex::new(0), PlanEdge::Effect);
                node
            }
            NodeExecution::None => return,
        };

        // process resources created up to this point
        self.process_resources();

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
                                PlanNodeData::ImageVersion,
                            );
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
                                PlanNodeData::BufferVersion,
                            );
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
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct RunContext<'a, B: Backend, T: ?Sized> {
    pub(crate) factory: &'a Factory<B>,
    pub(crate) families: &'a Families<B>,
    // images: Vec<ImageInfo>,
    // buffers: Vec<BufferInfo>,
    // virtuals: usize,
    pub(crate) output_store: OutputStore,
    graph: PlanGraph<'a, B, T>,
}

impl<'a, B: Backend, T: ?Sized> RunContext<'a, B, T> {
    pub(crate) fn new(
        factory: &'a Factory<B>,
        families: &'a Families<B>,
        allocator: &'a GraphAllocator,
    ) -> Self {
        Self {
            factory,
            families,
            // images: Vec::new(),
            // buffers: Vec::new(),
            // virtuals: 0,
            output_store: OutputStore::new(),
            graph: PlanGraph::new(allocator),
        }
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
        new::test::{test_init, visualize_graph, TestBackend},
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
        fn construct(
            &mut self,
            ctx: &mut NodeContext<TestBackend, ()>,
            _: &(),
        ) -> Result<(ImageId, NodeExecution<'_, TestBackend, ()>), NodeConstructionError> {
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

        fn construct(
            &mut self,
            ctx: &mut NodeContext<TestBackend, ()>,
            _: &(),
        ) -> Result<((), NodeExecution<'_, TestBackend, ()>), NodeConstructionError> {
            let color = *ctx.get_parameter(self.color)?;
            ctx.use_image(color, ImageUsage::ColorAttachmentWrite);
            if let Some(depth) = self.depth {
                let depth = *ctx.get_parameter(depth)?;
                ctx.use_image(depth, ImageUsage::DepthStencilAttachmentWrite);
            }

            Ok(((), NodeExecution::pass(|_, _| Ok(()))))
        }
    }

    #[derive(Debug)]
    struct TestPass2 {
        color1: Parameter<ImageId>,
        color2: Parameter<ImageId>,
        depth: Option<Parameter<ImageId>>,
    }

    impl TestPass2 {
        fn new(
            color1: Parameter<ImageId>,
            color2: Parameter<ImageId>,
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

        fn construct(
            &mut self,
            ctx: &mut NodeContext<TestBackend, ()>,
            _: &(),
        ) -> Result<((), NodeExecution<'_, TestBackend, ()>), NodeConstructionError> {
            let color1 = *ctx.get_parameter(self.color1)?;
            let color2 = *ctx.get_parameter(self.color2)?;
            ctx.use_image(color1, ImageUsage::ColorAttachmentWrite);
            ctx.use_image(color2, ImageUsage::ColorAttachmentRead);
            if let Some(depth) = self.depth {
                let depth = *ctx.get_parameter(depth)?;
                ctx.use_image(depth, ImageUsage::DepthStencilAttachmentRead);
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
        fn construct(
            &mut self,
            ctx: &mut NodeContext<TestBackend, ()>,
            _: &(),
        ) -> Result<(ImageId, NodeExecution<'_, TestBackend, ()>), NodeConstructionError> {
            let output = ctx.create_image(ImageInfo {
                kind: gfx_hal::image::Kind::D2(1024, 1024, 1, 1),
                levels: 1,
                format: gfx_hal::format::Format::Rgba8Unorm,
                clear: None,
            });
            ctx.use_image(output, ImageUsage::ColorAttachmentRead);
            Ok((output, NodeExecution::output(|_, _| Ok(()))))
        }
    }

    #[test]
    fn test_construction() {
        test_init();
        let mut file = std::fs::File::create("graph.dot").unwrap();

        let config: crate::factory::Config = Default::default();
        let (mut factory, mut families): (Factory<TestBackend>, _) =
            crate::factory::init(config).unwrap();

        let mut builder = GraphBuilder::new();

        let depth = builder.add(ImageInfo {
            kind: gfx_hal::image::Kind::D2(1, 1, 1, 1),
            levels: 1,
            format: gfx_hal::format::Format::R32Sfloat,
            clear: Some(gfx_hal::command::ClearValue {
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
            clear: None,
        });
        let color2 = builder.add(ImageInfo {
            kind: gfx_hal::image::Kind::D2(1, 1, 1, 1),
            levels: 1,
            format: gfx_hal::format::Format::Rgba8Unorm,
            clear: None,
        });
        let color = builder.add(TestOutput);
        builder.add(TestPass::new(color, Some(depth)));
        builder.add(TestPass::new(color2, Some(depth2)));
        builder.add(TestPass::new(color, Some(depth)));
        builder.add(TestPass::new(color2, Some(depth2)));
        builder.add(TestPass2::new(color2, color, None));
        builder.add(TestPass2::new(color, color2, Some(depth2)));
        builder.add(TestPass2::new(color, color2, Some(depth2)));
        builder.add(TestPass::new(color, Some(depth)));
        builder.add(TestPass::new(color, Some(depth)));
        builder.add(TestPass::new(color, None));

        let mut graph = builder.build(&mut factory, &mut families, &()).unwrap();

        let mut run_ctx = RunContext::new(&factory, &families, &graph.alloc);
        graph
            .nodes
            .run_construction_phase(&mut run_ctx, &())
            .unwrap();
        visualize_graph(&mut file, &run_ctx.graph.dag, "raw");
        graph.pipeline.reduce(&mut run_ctx.graph.dag, &graph.alloc);
        visualize_graph(&mut file, &run_ctx.graph.dag, "opti");
    }
}
