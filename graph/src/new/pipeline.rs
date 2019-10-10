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
        node::PassFn,
        resources::{ImageId, NodeImageAccess, ResourceAccess},
        test::{assert_topo_eq, test_init, TestBackend},
    };

    fn subpass_node(num_groups: usize) -> PlanNodeData<'static, TestBackend, ()> {
        let mut vec: Vec<PassFn<'static, TestBackend, ()>> = Vec::with_capacity(num_groups);
        for _ in 0..num_groups {
            vec.push(Box::new(|_, _| Ok(())))
        }
        PlanNodeData::RenderSubpass(vec)
    }

    fn group_node() -> PlanNodeData<'static, TestBackend, ()> {
        PlanNodeData::Execution(NodeExecution::RenderPass(Box::new(|_, _| Ok(()))))
    }

    fn image_node(id: usize) -> PlanNodeData<'static, TestBackend, ()> {
        PlanNodeData::Resource(ResourceId::Image(ImageId(id)))
    }

    fn image_edge(access: NodeImageAccess) -> PlanEdge {
        PlanEdge::Data(ResourceAccess::Image(access))
    }

    #[test]
    fn test_combine_two_passes() {
        test_init();

        let mut graph = graph! {
            group1 = group_node();
            group2 = group_node();

            color_a = image_node(0);
            @ color_b = image_node(0);

            group1 -> color_a = image_edge(NodeImageAccess::COLOR_ATTACHMENT_WRITE);
            color_a -> group2 = image_edge(NodeImageAccess::COLOR_ATTACHMENT_READ);
            group2 -> color_b = image_edge(NodeImageAccess::COLOR_ATTACHMENT_WRITE);
        };

        let expected_graph = graph! {
            pass = subpass_node(2);
            @ color_a = image_node(0);
            pass -> color_a = image_edge(NodeImageAccess::COLOR_ATTACHMENT_WRITE);
        };

        let mut reducer = GraphReducer::new()
            .with_reducer(LowerRenderPassReducer)
            .with_reducer(CombineSubpassesReducer);
        reducer.reduce_graph(&mut graph);

        assert_topo_eq(&expected_graph, &graph);
    }

    #[test]
    fn test_combine_two_passes_texture_access() {
        test_init();

        let mut graph = graph! {
            group1 = group_node();
            group2 = group_node();

            color1 = image_node(0);
            @ color2 = image_node(1);

            group1 -> color1 = image_edge(NodeImageAccess::COLOR_ATTACHMENT_WRITE);
            color1 -> group2 = image_edge(NodeImageAccess::SAMPLED_IMAGE_READ);
            group2 -> color2 = image_edge(NodeImageAccess::COLOR_ATTACHMENT_WRITE);
        };

        let expected_graph = graph! {
            pass1 = subpass_node(1);
            pass2 = subpass_node(1);

            color1 = image_node(0);
            @ color2 = image_node(1);

            pass1 -> color1 = image_edge(NodeImageAccess::COLOR_ATTACHMENT_WRITE);
            color1 -> pass2 = image_edge(NodeImageAccess::SAMPLED_IMAGE_READ);
            pass2 -> color2 = image_edge(NodeImageAccess::COLOR_ATTACHMENT_WRITE);
        };

        let mut reducer = GraphReducer::new()
            .with_reducer(LowerRenderPassReducer)
            .with_reducer(CombineSubpassesReducer);
        reducer.reduce_graph(&mut graph);

        assert_topo_eq(&expected_graph, &graph);
    }

    #[test]
    fn test_combine_three_passes_with_depth_read() {
        test_init();

        let mut graph = graph! {
            group1 = group_node();
            group2 = group_node();
            group3 = group_node();

            color_a = image_node(0);
            color_b = image_node(0);
            @ color_c = image_node(0);
            depth_a = image_node(1);
            @ depth_b = image_node(1);

            group1 -> color_a = image_edge(NodeImageAccess::COLOR_ATTACHMENT_WRITE);
            group1 -> depth_a = image_edge(NodeImageAccess::DEPTH_STENCIL_ATTACHMENT_WRITE);
            color_a -> group2 = image_edge(NodeImageAccess::COLOR_ATTACHMENT_READ);
            depth_a -> group2 = image_edge(NodeImageAccess::DEPTH_STENCIL_ATTACHMENT_READ);
            group2 -> color_b = image_edge(NodeImageAccess::COLOR_ATTACHMENT_WRITE);
            color_b -> group3 = image_edge(NodeImageAccess::COLOR_ATTACHMENT_READ);
            depth_a -> group3 = image_edge(NodeImageAccess::DEPTH_STENCIL_ATTACHMENT_READ);
            group3 -> color_c = image_edge(NodeImageAccess::COLOR_ATTACHMENT_WRITE);
            group3 -> depth_b = image_edge(NodeImageAccess::DEPTH_STENCIL_ATTACHMENT_WRITE);
        };

        let expected_graph = graph! {
            pass = subpass_node(3);
            @ color_a = image_node(0);
            @ depth_a = image_node(1);
            pass -> color_a = image_edge(NodeImageAccess::COLOR_ATTACHMENT_WRITE);
            pass -> depth_a = image_edge(NodeImageAccess::DEPTH_STENCIL_ATTACHMENT_WRITE);
        };

        let mut reducer = GraphReducer::new()
            .with_reducer(LowerRenderPassReducer)
            .with_reducer(CombineSubpassesReducer);
        reducer.reduce_graph(&mut graph);

        assert_topo_eq(&expected_graph, &graph);
    }

    #[test]
    fn test_combine_three_passes_with_external_read() {
        test_init();

        let mut graph = graph! {
            group1 = group_node();
            group2 = group_node();
            group3 = group_node();
            group4 = group_node();

            color_a = image_node(0);
            color_b = image_node(0);
            @ color_c = image_node(0);
            depth_a = image_node(1);
            @ depth_b = image_node(1);
            external_a = image_node(2);

            group4 -> external_a = image_edge(NodeImageAccess::COLOR_ATTACHMENT_WRITE);
            group1 -> color_a = image_edge(NodeImageAccess::COLOR_ATTACHMENT_WRITE);
            group1 -> depth_a = image_edge(NodeImageAccess::DEPTH_STENCIL_ATTACHMENT_WRITE);
            color_a -> group2 = image_edge(NodeImageAccess::COLOR_ATTACHMENT_READ);
            depth_a -> group2 = image_edge(NodeImageAccess::DEPTH_STENCIL_ATTACHMENT_READ);
            external_a -> group2 = image_edge(NodeImageAccess::SAMPLED_IMAGE_READ);
            group2 -> color_b = image_edge(NodeImageAccess::COLOR_ATTACHMENT_WRITE);
            color_b -> group3 = image_edge(NodeImageAccess::COLOR_ATTACHMENT_READ);
            depth_a -> group3 = image_edge(NodeImageAccess::DEPTH_STENCIL_ATTACHMENT_READ);
            group3 -> color_c = image_edge(NodeImageAccess::COLOR_ATTACHMENT_WRITE);
            group3 -> depth_b = image_edge(NodeImageAccess::DEPTH_STENCIL_ATTACHMENT_WRITE);
        };

        let expected_graph = graph! {
            main_pass = subpass_node(3);
            depth_a = image_node(1);
            color_a = image_node(0);
            ext_pass = subpass_node(1);
            external_a = image_node(2);

            @ color_a;
            @ depth_a;

            ext_pass -> external_a = image_edge(NodeImageAccess::COLOR_ATTACHMENT_WRITE);
            main_pass -> depth_a = image_edge(NodeImageAccess::DEPTH_STENCIL_ATTACHMENT_WRITE);
            main_pass -> color_a = image_edge(NodeImageAccess::COLOR_ATTACHMENT_WRITE);
            external_a -> main_pass = image_edge(NodeImageAccess::SAMPLED_IMAGE_READ);
        };

        let mut reducer = GraphReducer::new()
            .with_reducer(LowerRenderPassReducer)
            .with_reducer(CombineSubpassesReducer);
        reducer.reduce_graph(&mut graph);

        dbg!(&graph);
        dbg!(&expected_graph);

        assert_topo_eq(&expected_graph, &graph);
    }
}
