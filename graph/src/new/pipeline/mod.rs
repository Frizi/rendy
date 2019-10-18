use {
    crate::new::{graph::PlanDag, graph_reducer::GraphReducer},
    gfx_hal::Backend,
    graphy::{GraphAllocator, NodeIndex},
};

mod subpasses;
use subpasses::{CombineSubpassesReducer, OrderWritesReducer};

#[derive(derivative::Derivative)]
#[derivative(Debug(bound = ""))]
pub(crate) struct Pipeline<B: Backend, T: ?Sized> {
    stage1: GraphReducer<B, T>,
}

impl<B: Backend, T: ?Sized> Pipeline<B, T> {
    pub(crate) fn new() -> Self {
        Self {
            stage1: GraphReducer::new()
                .with_reducer(CombineSubpassesReducer)
                .with_reducer(OrderWritesReducer),
        }
    }
    pub(crate) fn reduce<'a>(&mut self, graph: &mut PlanDag<'a, B, T>, alloc: &'a GraphAllocator) {
        self.stage1.reduce_graph(graph, alloc);
        graph.trim(NodeIndex::new(0)).unwrap();
    }
}
