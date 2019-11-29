use {
    crate::new::{
        graph::{PlanEdge, PlanNode},
        graph_reducer::{GraphEditor, Reducer, Reduction},
        pipeline::node_ext::NodeExt,
    },
    gfx_hal::Backend,
    graphy::{NodeIndex, Walker},
};

#[derive(Debug)]
pub(super) struct CombineVersionsReducer;

impl<'a, B: Backend, T: ?Sized> Reducer<B, T> for CombineVersionsReducer {
    fn reduce(&mut self, editor: &mut GraphEditor<B, T>, node: NodeIndex) -> Reduction {
        if let PlanNode::ImageVersion = editor.graph()[node] {
            // combine two version nodes if those are consecutive and have the same origin
            if let Some(version) = node.version(editor.graph()) {
                if version.origin(editor.graph()) == node.origin(editor.graph()) {
                    return Reduction::Replace(version);
                }
            }
        }
        Reduction::NoChange
    }
}

// #[derive(Debug)]
// pub(super) struct InsertStoresReducer;
// impl<'a, B: Backend, T: ?Sized> Reducer<B, T> for InsertStoresReducer {
//     fn reduce(&mut self, editor: &mut GraphEditor<B, T>, node: NodeIndex) -> Reduction {
//         if let PlanNode::LoadImage(node_id, index) = editor.graph()[node] {
//             let mut head = node;
//             while let Some(version) = head.child_version(editor.graph()) {
//                 head = version;
//             }
//             editor.insert_edge_unchecked(head, NodeIndex::new(0), PlanEdge::Effect);
//         }
//         Reduction::NoChange
//     }
// }

#[derive(Debug)]
pub(super) struct OrderWritesReducer;
impl<B: Backend, T: ?Sized> Reducer<B, T> for OrderWritesReducer {
    fn reduce(&mut self, editor: &mut GraphEditor<B, T>, node: NodeIndex) -> Reduction {
        let mut write_accesses = node
            .children()
            .filter(|graph, &(edge, _)| match graph[edge] {
                PlanEdge::ImageAccess(access, _) => access.is_write(),
                PlanEdge::BufferAccess(access, _) => access.is_write(),
                _ => false,
            });

        let mut read_accesses = node
            .children()
            .filter(|graph, &(edge, _)| match graph[edge] {
                PlanEdge::ImageAccess(access, _) => !access.is_write(),
                PlanEdge::BufferAccess(access, _) => !access.is_write(),
                _ => false,
            });

        // there should ever be at most one write access
        let write_access = write_accesses.walk_next(editor.graph());
        assert_eq!(None, write_accesses.walk_next(editor.graph()));

        if let Some((_, write)) = write_access {
            let mut next = read_accesses.walk_next(editor.graph());
            while let Some((_, read)) = next {
                editor
                    .insert_edge_unchecked(read, write, PlanEdge::Effect)
                    .unwrap();
                editor.revisit(read);
                next = read_accesses.walk_next(editor.graph());
            }
        }

        Reduction::NoChange
    }
}
