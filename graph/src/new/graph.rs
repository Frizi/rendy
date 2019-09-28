use {
    super::{
        node::{
            DynNode, DynNodeBuilder, InternalUse, Node, NodeBuildError, NodeBuilder,
            NodeConstructionError, NodeExecution, NodeExecutionError, NodeId, OutputList,
            OutputStore, Parameter,
        },
        resources::{
            BufferId, BufferInfo, ImageId, ImageInfo, NodeBufferAccess, NodeImageAccess,
            NodeVirtualAccess, VirtualId,
        },
    },
    crate::{command::Families, factory::Factory},
    gfx_hal::Backend,
    std::{any::Any, collections::HashMap},
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
            nodes: self
                .nodes
                .into_iter()
                .enumerate()
                .map(|(i, builder)| {
                    builder
                        .build(factory, families, aux)
                        .map_err(|e| GraphBuildError::NodeBuildError(NodeId(i), e))
                })
                .collect::<Result<_, GraphBuildError>>()?,
        })
    }
}
/// A built runnable rendering graph.
#[derive(derivative::Derivative)]
#[derivative(Debug(bound = ""))]
pub struct Graph<B: Backend, T: ?Sized> {
    nodes: Vec<Box<dyn DynNode<B, T>>>,
}

impl<B: Backend, T: ?Sized> Graph<B, T> {
    /// Construct, schedule and run all nodes of rendering graph.
    pub fn run(
        &mut self,
        factory: &mut Factory<B>,
        families: &mut Families<B>,
        aux: &T,
    ) -> Result<(), GraphRunError> {
        let mut run_ctx = RunContext {
            factory: &factory,
            families: &families,
            images: Vec::new(),
            buffers: Vec::new(),
            virtuals: 0,
            output_store: OutputStore::new(),
        };
        let mut graph = daggy::Dag::new();
        let mut last_writes: HashMap<ResourceId, daggy::NodeIndex> = HashMap::new();
        // a faux node that's only purpose to be the origin of all resources not yet written to.
        let root = graph.add_node(NodeExecution::None);
        let mut usage = ResourceUsage::default();
        let mut outputs = smallvec::SmallVec::<[_; 8]>::new();
        // construction phase
        for (i, node) in self.nodes.iter_mut().enumerate() {
            let execution = {
                let mut ctx = NodeContext {
                    id: NodeId(i),
                    run: &mut run_ctx,
                    resources: &mut usage,
                };
                node.construct(&mut ctx, aux)
                    .map_err(|e| GraphRunError::NodeConstruction(NodeId(i), e))?
            };
            let is_output = execution.is_output();
            let node = graph.add_node(execution);
            if is_output {
                outputs.push((
                    node,
                    std::mem::replace(&mut usage, ResourceUsage::default()),
                ))
            } else {
                usage.add_to_graph(node, root, &mut graph, &mut last_writes);
            }
        }
        for (node, mut usage) in outputs {
            usage.add_to_graph(node, root, &mut graph, &mut last_writes);
        }
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
        // 
        // ## Idea of an algorithm:
        // We scan nodes in two alternating phases, "ungroupable" and "groupable"
        // always Starting with Ungroupable:
        // - remove non-pass leafs (nodes without dependency) out of the graph, schedule them for execution
        // - sort collected nodes by some heurestic and schedule
        // - repeat until there are some non-pass leafs
        // groupable:
        // - remove renderpass leafs out of the graph. If there was a node with identical attachments and it's written resources are not read by anything else, join the nodes as single pass.
        // - repeat until there are no more renderpass leafs.
        // - Sort collected passes by some heurestic and schedule
        // - once there are no more nodes, schedule is done

        unimplemented!()
    }
}
/// A context for rendergraph node construction phase. Contains all data that the node
/// get access to and contains ready-made methods for common operations.
#[derive(Debug)]
pub struct NodeContext<'a, 'b, B: Backend> {
    pub(crate) id: NodeId,
    pub(crate) run: &'a mut RunContext<'b, B>,
    pub(crate) resources: &'a mut ResourceUsage,
    // images: Vec<(ImageId, NodeImageAccess)>,
    // buffers: Vec<(BufferId, NodeBufferAccess)>,
    // virtuals: Vec<(VirtualId, NodeVirtualAccess)>,
}

impl<'a, 'b, B: Backend> NodeContext<'a, 'b, B> {
    pub fn factory(&self) -> &'a Factory<B> {
        self.run.factory
    }
    pub fn get_parameter<T: Any>(&self, id: Parameter<T>) -> Result<&T, NodeConstructionError> {
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
        self.resources.virtuals.push((id, access));
    }
    /// Declare usage of image by the node
    pub fn use_image(&mut self, id: ImageId, access: NodeImageAccess) {
        self.resources.images.push((id, access));
    }
    /// Declare usage of buffer by the node
    pub fn use_buffer(&mut self, id: BufferId, access: NodeBufferAccess) {
        self.resources.buffers.push((id, access));
    }
}
#[derive(Debug, Default)]
pub(crate) struct ResourceUsage {
    pub(crate) images: Vec<(ImageId, NodeImageAccess)>,
    pub(crate) buffers: Vec<(BufferId, NodeBufferAccess)>,
    pub(crate) virtuals: Vec<(VirtualId, NodeVirtualAccess)>,
}

impl ResourceUsage {
    fn add_to_graph<B: Backend, T: ?Sized>(
        &mut self,
        node: daggy::NodeIndex,
        root: daggy::NodeIndex,
        graph: &mut PlanGraph<'_, B, T>,
        last_writes: &mut HashMap<ResourceId, daggy::NodeIndex>,
    ) {
        let images = self
            .images
            .drain(..)
            .map(|(id, access)| (ResourceId::Image(id), Edge::Image(id, access)));
        let buffers = self
            .buffers
            .drain(..)
            .map(|(id, access)| (ResourceId::Buffer(id), Edge::Buffer(id, access)));
        let virtuals = self
            .virtuals
            .drain(..)
            .map(|(id, access)| (ResourceId::Virtual(id), Edge::Virtual(id, access)));
        let edges = images
            .chain(buffers)
            .chain(virtuals)
            .map(|(resource, edge)| {
                let writer = *last_writes.get(&resource).unwrap_or(&root);
                if edge.access_is_write() {
                    last_writes.insert(resource, node);
                }
                (node, writer, edge)
            });
        graph.add_edges(edges);
    }
}
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
enum ResourceId {
    Image(ImageId),
    Buffer(BufferId),
    Virtual(VirtualId),
}

enum Edge {
    Image(ImageId, NodeImageAccess),
    Buffer(BufferId, NodeBufferAccess),
    Virtual(VirtualId, NodeVirtualAccess),
}

impl Edge {
    fn access_is_write(&self) -> bool {
        match self {
            Edge::Image(_, access) => access.is_write(),
            Edge::Buffer(_, access) => access.is_write(),
            Edge::Virtual(_, access) => access.is_write(),
        }
    }
}

type PlanGraph<'a, B, T> = daggy::Dag<NodeExecution<'a, B, T>, Edge>;
#[derive(Debug)]
pub(crate) struct RunContext<'a, B: Backend> {
    pub(crate) factory: &'a Factory<B>,
    pub(crate) families: &'a Families<B>,
    pub(crate) images: Vec<ImageInfo>,
    pub(crate) buffers: Vec<BufferInfo>,
    pub(crate) virtuals: usize,
    pub(crate) output_store: OutputStore,
}

impl<'a, B: Backend> RunContext<'a, B> {
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
