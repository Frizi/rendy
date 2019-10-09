use {
    super::{
        node::{
            DynNode, DynNodeBuilder, InternalUse, Node, NodeBuildError, NodeBuilder,
            NodeConstructionError, NodeExecution, NodeExecutionError, NodeId, OutputList,
            OutputStore, Parameter, PassFn,
        },
        pipeline::Pipeline,
        resources::{
            BufferId, BufferInfo, ImageId, ImageInfo, NodeBufferAccess, NodeImageAccess,
            NodeVirtualAccess, ResourceAccess, ResourceId, ResourceUsage, VirtualId,
        },
    },
    crate::{command::Families, factory::Factory},
    daggy::{Dag, NodeIndex},
    gfx_hal::Backend,
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
        })
    }
}
/// A built runnable top-level rendering graph.
#[derive(derivative::Derivative)]
#[derivative(Debug(bound = ""))]
pub struct Graph<B: Backend, T: ?Sized> {
    nodes: GraphNodes<B, T>,
    pipeline: Pipeline<B, T>,
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
        let mut run_ctx = RunContext::new(factory, families);
        self.nodes.run_construction_phase(&mut run_ctx, aux)?;

        self.pipeline.optimize(&mut run_ctx.graph.dag);

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
        let mut outputs = smallvec::SmallVec::<[_; 8]>::new();

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
                run_ctx.graph.insert(seed, execution, false);
            }
        }

        for (seed, execution) in outputs.drain().rev() {
            run_ctx.graph.insert(seed, execution, true);
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
        self.run.create_image(image_info)
    }
    /// Create new buffer owned by graph.
    pub fn create_buffer(&mut self, buffer_info: BufferInfo) -> BufferId {
        self.run.create_buffer(buffer_info)
    }
    /// Create non-data dependency target. A virtual resource intended to
    /// describe dependencies between rendering nodes without carrying any data.
    pub fn create_virtual(&mut self) -> VirtualId {
        self.run.create_virtual()
    }
    pub fn use_virtual(&mut self, id: VirtualId, access: NodeVirtualAccess) {
        self.run
            .graph
            .use_resource(self.seed, ResourceUsage::Virtual(id, access));
    }
    /// Declare usage of image by the node
    pub fn use_image(&mut self, id: ImageId, access: NodeImageAccess) {
        self.run
            .graph
            .use_resource(self.seed, ResourceUsage::Image(id, access));
    }
    /// Declare usage of buffer by the node
    pub fn use_buffer(&mut self, id: BufferId, access: NodeBufferAccess) {
        self.run
            .graph
            .use_resource(self.seed, ResourceUsage::Buffer(id, access));
    }
}

pub(crate) enum PlanNodeData<'a, B: Backend, T: ?Sized> {
    /// Construction phase execution
    Execution(NodeExecution<'a, B, T>),
    /// Construction phase resource
    Resource(ResourceId),
    // /// A resource usage that is used as an attachment read or write.
    // Attachment(ResourceId),
    /// A lowered subpass that might have multiple render groups.
    RenderSubpass(Vec<PassFn<'a, B, T>>),
    /// Placeholder value required to take out the node out of graph just before replacing it
    Tombstone,
    /// Graph root node. All incoming nodes are always evaluated.
    Root,
}

impl<'n, B: Backend, T: ?Sized> std::fmt::Debug for PlanNodeData<'n, B, T> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            PlanNodeData::Execution(exec) => {
                fmt.debug_struct("Execution").field("exec", exec).finish()
            }
            PlanNodeData::Resource(res) => fmt.debug_tuple("Resource").field(res).finish(),
            // PlanNodeData::Attachment(res) => fmt.debug_tuple("Attachment").field(res).finish(),
            PlanNodeData::RenderSubpass(vec) => {
                fmt.debug_tuple("RenderSubpass").field(&vec.len()).finish()
            }
            PlanNodeData::Tombstone => fmt.debug_tuple("Tombstone").finish(),
            PlanNodeData::Root => fmt.debug_tuple("Root").finish(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PlanEdge {
    /// generic data acccess of any type.
    /// When directed towards resource node, it is a write access.
    Data(ResourceAccess),
    // edge created only as a node-to-node re
    Effect,
}

impl PlanEdge {
    pub(crate) fn is_data(&self) -> bool {
        match self {
            PlanEdge::Data(_) => true,
            _ => false,
        }
    }

    pub(crate) fn is_effect(&self) -> bool {
        match self {
            PlanEdge::Effect => true,
            _ => false,
        }
    }

    pub(crate) fn is_attachment_only(&self) -> bool {
        match self {
            PlanEdge::Data(usage) => usage.is_attachment_only(),
            _ => false,
        }
    }

    pub(crate) fn is_attachment(&self) -> bool {
        match self {
            PlanEdge::Data(usage) => usage.is_attachment(),
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
struct NodeSeed {
    resources: Range<usize>,
}

pub(crate) type PlanDag<'a, B, T> = Dag<PlanNodeData<'a, B, T>, PlanEdge>;

#[derive(Debug)]
pub struct PlanGraph<'a, B: Backend, T: ?Sized> {
    dag: PlanDag<'a, B, T>,
    last_writes: HashMap<ResourceId, NodeIndex>,
    resource_usage: Vec<ResourceUsage>,
}

impl<'a, B: Backend, T: ?Sized> Default for PlanGraph<'a, B, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, B: Backend, T: ?Sized> PlanGraph<'a, B, T> {
    fn new() -> Self {
        let mut dag = PlanDag::new();
        // guaranteed to always be index 0
        dag.add_node(PlanNodeData::Root);

        Self {
            dag,
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

    fn insert(&mut self, seed: NodeSeed, exec: NodeExecution<'a, B, T>, root: bool) {
        // inserts should always happen in such order that resource usages are free to be drained
        debug_assert!(self.resource_usage.len() == seed.resources.end);

        let resources = seed.resources;

        let node_data = PlanNodeData::Execution(exec);

        let n = self.dag.add_node(node_data);

        let mut connect_node_root = root;

        for res in self.resource_usage.drain(resources) {
            let (id, access) = res.open();

            let (reads, writes) = access.split_rw();

            // insert read
            if let Some(prev_write) = self.last_writes.get(&id) {
                self.dag
                    .add_edge(*prev_write, n, PlanEdge::Data(reads))
                    .expect("Trying to insert a cycle");
            } else {
                // Node reads something that was never written.
                // It is either dead or reads from previous frame state (only if the resource is not cleared).
                // TODO: check what to do about it
            }

            // insert write
            if !writes.is_empty() {
                let (_, res_node) =
                    self.dag
                        .add_child(n, PlanEdge::Data(writes), PlanNodeData::Resource(id));
                self.last_writes.insert(id, res_node);
                if root {
                    self.dag
                        .add_edge(res_node, NodeIndex::new(0), PlanEdge::Effect)
                        .expect("Cycle through root");
                    connect_node_root = false;
                }
            }
        }

        if connect_node_root {
            self.dag
                .add_edge(n, NodeIndex::new(0), PlanEdge::Effect)
                .expect("Cycle through root");
        }
    }
}

#[derive(Debug)]
pub(crate) struct RunContext<'a, B: Backend, T: ?Sized> {
    pub(crate) factory: &'a Factory<B>,
    pub(crate) families: &'a Families<B>,
    images: Vec<ImageInfo>,
    buffers: Vec<BufferInfo>,
    virtuals: usize,
    pub(crate) output_store: OutputStore,
    graph: PlanGraph<'a, B, T>,
}

impl<'a, B: Backend, T: ?Sized> RunContext<'a, B, T> {
    pub(crate) fn new(factory: &'a Factory<B>, families: &'a Families<B>) -> Self {
        Self {
            factory,
            families,
            images: Vec::new(),
            buffers: Vec::new(),
            virtuals: 0,
            output_store: OutputStore::new(),
            graph: PlanGraph::new(),
        }
    }
    /// Create new image owned by graph.
    pub(crate) fn create_image(&mut self, image_info: ImageInfo) -> ImageId {
        self.images.push(image_info);
        ImageId(self.images.len() - 1)
    }
    /// Create new buffer owned by graph.
    pub(crate) fn create_buffer(&mut self, buffer_info: BufferInfo) -> BufferId {
        self.buffers.push(buffer_info);
        BufferId(self.buffers.len() - 1)
    }
    /// Create non-data dependency target. A virtual resource intended to
    /// describe dependencies between rendering nodes without carrying any data.
    pub(crate) fn create_virtual(&mut self) -> VirtualId {
        self.virtuals += 1;
        VirtualId(self.virtuals - 1)
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
