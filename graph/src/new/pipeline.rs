use {
    super::{
        graph::{PlanDag, PlanEdge, PlanNodeData},
        graph_reducer::{GraphEditor, GraphReducer, Reducer, Reduction},
        node::NodeExecution,
        resources::{ImageId, ResourceId},
        walker::WalkerExt,
    },
    daggy::{EdgeIndex, NodeIndex, Walker},
    gfx_hal::Backend,
};

// impl reducers pipeline

// macro_rules! try_reduce {
//     ($x:expr) => {
//         match $x {
//             Some(v) => v,
//             _ => return Reduction::NoChange,
//         }
//     };
// }

#[derive(derivative::Derivative)]
#[derivative(Debug(bound = ""))]
pub(crate) struct Pipeline<B: Backend, T: ?Sized> {
    reducer: GraphReducer<B, T>,
}

impl<B: Backend, T: ?Sized> Pipeline<B, T> {
    pub(crate) fn new() -> Self {
        Self {
            reducer: GraphReducer::new()
                .with_reducer(LowerRenderPassReducer)
                .with_reducer(CombineSubpassesReducer),
        }
    }
    pub(crate) fn optimize(&mut self, graph: &mut PlanDag<'_, B, T>) {
        self.reducer.reduce_graph(graph);
    }
}

// // Find resource nodes that qualify as an attachment
// #[derive(Debug)]
// struct FindAttachmentsReducer;

// impl<'a, B: Backend, T: ?Sized> Reducer<PlanNodeData<'a, B, T>, PlanEdge>
//     for FindAttachmentsReducer
// {
//     fn reduce(
//         &mut self,
//         editor: &mut GraphEditor<'_, PlanNodeData<'a, B, T>, PlanEdge>,
//         node: NodeIndex,
//     ) -> Reduction {
//         let replacement = match editor.graph().node_weight(node) {
//             Some(PlanNodeData::Resource(ResourceId::Image(image))) => {
//                 // Merge passes if this node represents an attachment between exactly two render passes

//                 let mut writers = walk_data(editor.graph().parents(node));
//                 let mut readers = walk_data(editor.graph().children(node));

//                 let (writer_edge, _) = try_reduce!(writers.walk_next(editor.graph()));
//                 let (reader_edge, _) = try_reduce!(readers.walk_next(editor.graph()));

//                 // there is more than one edge in either direction
//                 if readers.walk_next(editor.graph()).is_some()
//                     || writers.walk_next(editor.graph()).is_some()
//                 {
//                     return Reduction::NoChange;
//                 }

//                 let read_is_attachment =
//                     try_reduce!(editor.graph().edge_weight(reader_edge)).is_attachment_only();
//                 let write_is_attachment =
//                     try_reduce!(editor.graph().edge_weight(writer_edge)).is_attachment_only();

//                 if read_is_attachment && write_is_attachment {
//                     PlanNodeData::Attachment(ResourceId::Image(*image))
//                 } else {
//                     return Reduction::NoChange;
//                 }
//             }
//             _ => return Reduction::NoChange,
//         };

//         *editor.graph_mut().node_weight_mut(node).unwrap() = replacement;
//         Reduction::Changed
//     }
// }

#[derive(Debug)]
struct LowerRenderPassReducer;

impl<'a, B: Backend, T: ?Sized> Reducer<B, T> for LowerRenderPassReducer {
    fn reduce(&mut self, editor: &mut GraphEditor<B, T>, node: NodeIndex) -> Reduction {
        match editor.graph_mut().node_weight_mut(node) {
            Some(data @ PlanNodeData::Execution(NodeExecution::RenderPass(_))) => {
                // to avoid copy, replace with an empty node first, then push moved closure from old node into a vector.
                let old_node =
                    std::mem::replace(data, PlanNodeData::RenderSubpass(Vec::with_capacity(1)));
                if let PlanNodeData::RenderSubpass(ref mut vec) = data {
                    if let PlanNodeData::Execution(NodeExecution::RenderPass(pass_fn)) = old_node {
                        vec.push(pass_fn);
                        Reduction::Changed
                    } else {
                        unreachable!()
                    }
                } else {
                    unreachable!()
                }
            }
            _ => Reduction::NoChange,
        }
    }
}

// Find render pass nodes that writes only exclusive attachments
#[derive(Debug)]
struct CombineSubpassesReducer;

impl<'a, B: Backend, T: ?Sized> Reducer<B, T> for CombineSubpassesReducer {
    fn reduce(&mut self, editor: &mut GraphEditor<B, T>, node: NodeIndex) -> Reduction {
        let mut candidate_merge = None;

        match editor.graph().node_weight(node) {
            Some(PlanNodeData::RenderSubpass(_)) => {
                let mut read_attachments = walk_attachments(editor.graph().parents(node));

                // Do all attachments I read come from the same node? If so, which node?
                while let Some((_, resource_node, _)) = read_attachments.walk_next(editor.graph()) {
                    // take attachment writer
                    let (_, writer) = editor
                        .graph()
                        .parents(resource_node)
                        .walk_next(editor.graph())
                        .expect("Attachment node must have one child");

                    match candidate_merge {
                        Some(candidate) if candidate != writer => {
                            // this subpass attachments are written by more than one subpass
                            log::trace!("CombineSubpassesReducer bailout: found another writer");
                            return Reduction::NoChange;
                        }
                        None => {
                            match editor.graph().node_weight(writer) {
                                Some(PlanNodeData::RenderSubpass(_)) => {
                                    candidate_merge = Some(writer)
                                }
                                _ => {
                                    log::trace!(
                                        "CombineSubpassesReducer bailout: written by non-subpass"
                                    );
                                    // Written by not a subpass
                                    return Reduction::NoChange;
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => return Reduction::NoChange,
        };

        match candidate_merge {
            Some(merge_target) => {
                // append this pass data to other pass and replace current node with it,
                let this_node = std::mem::replace(
                    editor.graph_mut().node_weight_mut(node).unwrap(),
                    PlanNodeData::Tombstone,
                );

                let mut reads = editor.graph().parents(node);
                while let Some((read_edge, read_node)) = reads.walk_next(editor.graph()) {
                    if editor
                        .graph()
                        .edge_weight(read_edge)
                        .unwrap()
                        .is_attachment()
                    {
                        editor.kill(read_node);
                    } else if let Some(edge_data) = editor.graph_mut().remove_edge(read_edge) {
                        editor
                            .graph_mut()
                            .add_edge(read_node, merge_target, edge_data)
                            .unwrap();
                    }
                }

                let merge_node = editor.graph_mut().node_weight_mut(merge_target).unwrap();
                match (this_node, merge_node) {
                    (
                        PlanNodeData::RenderSubpass(ref mut this_groups),
                        PlanNodeData::RenderSubpass(ref mut merge_groups),
                    ) => {
                        merge_groups.append(this_groups);
                    }
                    _ => unreachable!(),
                }

                Reduction::Replace(merge_target)
            }
            _ => Reduction::NoChange,
        }
    }
}

fn walk_attachments<'w, 'a: 'w, B: Backend, T: ?Sized + 'w>(
    walker: impl Walker<&'w PlanDag<'a, B, T>, Item = (EdgeIndex, NodeIndex)>,
) -> impl Walker<&'w PlanDag<'a, B, T>, Item = (EdgeIndex, NodeIndex, ImageId)> {
    walker.filter_map(|graph, &(edge, node)| {
        if graph.edge_weight(edge).map_or(false, |e| match e {
            PlanEdge::Data(data) => data.is_attachment(),
            _ => false,
        }) {
            match graph.node_weight(node) {
                Some(PlanNodeData::Resource(ResourceId::Image(id))) => Some((edge, node, *id)),
                _ => unreachable!(),
            }
        } else {
            None
        }
    })
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
    use crate::new::{
        resources::{ImageId, NodeImageAccess, ResourceAccess},
        test::{assert_topo_eq, test_init},
    };

    #[test]
    fn test_compound_reducer() {
        test_init();

        let mut graph = graph! {
            pass1 = PlanNodeData::Execution(NodeExecution::RenderPass(Box::new(|_, _| Ok(()))));
            res1 = PlanNodeData::Resource(ResourceId::Image(ImageId(0)));
            pass1 -> res1 = PlanEdge::Data(ResourceAccess::Image(NodeImageAccess::COLOR_ATTACHMENT_WRITE));

            pass2 = PlanNodeData::Execution(NodeExecution::RenderPass(Box::new(|_, _| Ok(()))));
            res1 -> pass2 = PlanEdge::Data(ResourceAccess::Image(NodeImageAccess::COLOR_ATTACHMENT_READ));
            @ res2 = PlanNodeData::Resource(ResourceId::Image(ImageId(0)));
            pass2 -> res2 = PlanEdge::Data(ResourceAccess::Image(NodeImageAccess::COLOR_ATTACHMENT_WRITE));
        };

        let expected_graph = graph! {
            pass = PlanNodeData::RenderSubpass(vec![
                Box::new(|_, _| Ok(())),
                Box::new(|_, _| Ok(())),
            ]);
            @ attachment = PlanNodeData::Resource(ResourceId::Image(ImageId(0)));
            pass -> attachment = PlanEdge::Data(ResourceAccess::Image(NodeImageAccess::COLOR_ATTACHMENT_WRITE));
        };

        let mut reducer = GraphReducer::new()
            .with_reducer(LowerRenderPassReducer)
            .with_reducer(CombineSubpassesReducer);
        reducer.reduce_graph(&mut graph);

        assert_topo_eq(&expected_graph, &graph);
    }
}
