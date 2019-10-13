use {
    crate::new::{graph::PlanDag, graph_reducer::GraphReducer},
    gfx_hal::Backend,
};

// macro_rules! try_reduce {
//     ($x:expr) => {
//         match $x {
//             Some(v) => v,
//             _ => return crate::new::graph_reducer::Reduction::NoChange,
//         }
//     };
// }
//

mod subpasses;
use subpasses::CombineSubpassesReducer;

#[derive(derivative::Derivative)]
#[derivative(Debug(bound = ""))]
pub(crate) struct Pipeline<B: Backend, T: ?Sized> {
    stage1: GraphReducer<B, T>,
}

impl<B: Backend, T: ?Sized> Pipeline<B, T> {
    pub(crate) fn new() -> Self {
        Self {
            stage1: GraphReducer::new().with_reducer(CombineSubpassesReducer),
        }
    }
    pub(crate) fn reduce(&mut self, graph: &mut PlanDag<'_, B, T>) {
        self.stage1.reduce_graph(graph);
    }
}
