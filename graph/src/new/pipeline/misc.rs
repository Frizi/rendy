use {
    crate::new::{
        graph::{PlanEdge, PlanNode},
        graph_reducer::{GraphEditor, Reducer, Reduction},
        pipeline::node_ext::NodeExt,
    },
    graphy::{NodeIndex, Walker},
    rendy_core::hal::Backend,
};

#[derive(Debug)]
pub(super) struct CombineVersionsReducer;

impl<'a, B: Backend> Reducer<B> for CombineVersionsReducer {
    fn reduce(&mut self, editor: &mut GraphEditor<B>, node: NodeIndex) -> Reduction {
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

#[derive(Debug)]
pub(super) struct InsertStoresReducer;
impl<'a, B: Backend> Reducer<B> for InsertStoresReducer {
    fn reduce(&mut self, editor: &mut GraphEditor<B>, node: NodeIndex) -> Reduction {
        if let PlanNode::LoadImage(..) = editor.graph()[node] {
            let mut head = node;
            while let Some(version) = head.child_version(editor.graph()) {
                head = version;
            }
            editor
                .insert_edge_unchecked(head, NodeIndex::new(0), PlanEdge::Effect)
                .unwrap();
        }
        Reduction::NoChange
    }
}

#[derive(Debug)]
pub(super) struct OrderWritesReducer;
impl<B: Backend> Reducer<B> for OrderWritesReducer {
    fn reduce(&mut self, editor: &mut GraphEditor<B>, node: NodeIndex) -> Reduction {
        let mut write_accesses = node
            .children()
            .filter(|graph, &(edge, _)| match &graph[edge] {
                PlanEdge::ImageAccess(access, _) => access.is_write(),
                PlanEdge::BufferAccess(access, _) => access.is_write(),
                PlanEdge::PassAttachment(att) => att.is_write(),
                _ => false,
            });

        let mut read_accesses = node
            .children()
            .filter(|graph, &(edge, _)| match &graph[edge] {
                PlanEdge::ImageAccess(access, _) => !access.is_write(),
                PlanEdge::BufferAccess(access, _) => !access.is_write(),
                PlanEdge::PassAttachment(att) => !att.is_write(),
                _ => false,
            });

        // there should ever be at most one write access
        let write_access = write_accesses.walk_next(editor.graph());
        assert_eq!(None, write_accesses.walk_next(editor.graph()));

        if let Some((_, write)) = write_access {
            let mut next = read_accesses.walk_next(editor.graph());
            while let Some((_, read)) = next {
                // do not insert redundant edge if usage relation is already present
                // Note that `contributions.uses` can have false negatives at this point,
                // because contributions are not recomputed before this pass.
                if !editor.context().contributions.uses(write, read) {
                    editor
                        .insert_edge_unchecked(read, write, PlanEdge::Effect)
                        .unwrap();
                    editor.revisit(read);
                }
                next = read_accesses.walk_next(editor.graph());
            }
        }

        Reduction::NoChange
    }
}
