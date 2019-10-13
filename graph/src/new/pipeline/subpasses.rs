use {
    crate::new::{
        graph::{PlanDag, PlanEdge, PlanNodeData},
        graph_reducer::{GraphEditor, Reducer, Reduction},
        walker::WalkerExt,
    },
    daggy::{EdgeIndex, NodeIndex, Walker},
    gfx_hal::Backend,
};

// Find render pass nodes that writes only exclusive attachments
#[derive(Debug)]
pub(super) struct CombineSubpassesReducer;

impl<'a, B: Backend, T: ?Sized> Reducer<B, T> for CombineSubpassesReducer {
    fn reduce(&mut self, editor: &mut GraphEditor<B, T>, node: NodeIndex) -> Reduction {
        let mut candidate_merge = None;

        match editor.graph().node_weight(node) {
            Some(PlanNodeData::RenderSubpass(_)) => {
                let mut read_attachments = walk_attachments(editor.graph().parents(node));

                // Do all attachments I read come from the same node? If so, which node?
                while let Some((_, access_node)) = read_attachments.walk_next(editor.graph()) {
                    // take attachment writer
                    let (_, writer) = walk_origins(editor.graph().parents(access_node))
                        .walk_next(editor.graph())
                        .expect("Attachment node must have an orign");

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
                        merge_groups.extend(this_groups.drain());
                    }
                    _ => unreachable!(),
                }

                Reduction::Replace(merge_target)
            }
            _ => Reduction::NoChange,
        }
    }
}

fn walk_origins<'w, 'a: 'w, B: Backend, T: ?Sized + 'w>(
    walker: impl Walker<&'w PlanDag<'a, B, T>, Item = (EdgeIndex, NodeIndex)>,
) -> impl Walker<&'w PlanDag<'a, B, T>, Item = (EdgeIndex, NodeIndex)> {
    walker.filter(|graph, &(edge, _)| {
        graph.edge_weight(edge).map_or(false, |e| match e {
            PlanEdge::Origin => true,
            _ => false,
        })
    })
}

fn walk_attachments<'w, 'a: 'w, B: Backend, T: ?Sized + 'w>(
    walker: impl Walker<&'w PlanDag<'a, B, T>, Item = (EdgeIndex, NodeIndex)>,
) -> impl Walker<&'w PlanDag<'a, B, T>, Item = (EdgeIndex, NodeIndex)> {
    walker.filter(|graph, &(edge, _)| {
        graph.edge_weight(edge).map_or(false, |e| match e {
            PlanEdge::ImageAccess(access, _) => access.is_attachment(),
            _ => false,
        })
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
        graph::ImageNode,
        graph_reducer::GraphReducer,
        node::PassFn,
        resources::{ImageUsage, ShaderUsage},
        test::{assert_topo_eq, test_init, TestBackend},
    };
    use smallvec::SmallVec;

    fn subpass_node(num_groups: usize) -> PlanNodeData<'static, TestBackend, ()> {
        let mut vec: SmallVec<[PassFn<'static, TestBackend, ()>; 4]> =
            SmallVec::with_capacity(num_groups);
        for _ in 0..num_groups {
            vec.push(Box::new(|_, _| Ok(())))
        }
        PlanNodeData::RenderSubpass(vec)
    }

    fn group_node() -> PlanNodeData<'static, TestBackend, ()> {
        subpass_node(1)
    }

    fn image_edge(usage: ImageUsage) -> PlanEdge {
        PlanEdge::ImageAccess(usage.access(), usage.stage())
    }

    fn color_node() -> PlanNodeData<'static, TestBackend, ()> {
        PlanNodeData::Image(ImageNode {
            kind: gfx_hal::image::Kind::D2(1024, 1024, 1, 1),
            levels: 1,
            format: gfx_hal::format::Format::Rgba8Unorm,
        })
    }

    fn depth_node() -> PlanNodeData<'static, TestBackend, ()> {
        PlanNodeData::Image(ImageNode {
            kind: gfx_hal::image::Kind::D2(1024, 1024, 1, 1),
            levels: 1,
            format: gfx_hal::format::Format::R32Sfloat,
        })
    }

    #[test]
    fn test_combine_two_passes() {
        test_init();

        let mut graph = graph! {
            group1 = group_node();
            @ group2 = group_node();

            color_def = color_node();
            color_init = PlanNodeData::UndefinedImage;
            color_a = PlanNodeData::ImageVersion;
            color_b = PlanNodeData::ImageVersion;
            color_c = PlanNodeData::ImageVersion;

            color_def -> color_init = PlanEdge::Origin;
            color_init -> color_a = PlanEdge::Origin;
            color_a -> color_b = PlanEdge::Version;

            color_a -> group1 = image_edge(ImageUsage::ColorAttachmentWrite);
            group1 -> color_b = PlanEdge::Origin;
            color_b -> group2 = image_edge(ImageUsage::ColorAttachmentWrite);
            group2 -> color_c = PlanEdge::Origin;
        };

        let expected_graph = graph! {
            @ pass = subpass_node(2);

            color_def = color_node();
            color_init = PlanNodeData::UndefinedImage;
            color_a = PlanNodeData::ImageVersion;
            color_b = PlanNodeData::ImageVersion;

            color_def -> color_init = PlanEdge::Origin;
            color_init -> color_a = PlanEdge::Origin;

            pass -> color_a = image_edge(ImageUsage::ColorAttachmentWrite);
            pass -> color_b = PlanEdge::Origin;
        };

        let mut reducer = GraphReducer::new().with_reducer(CombineSubpassesReducer);
        reducer.reduce_graph(&mut graph);

        assert_topo_eq(&expected_graph, &graph);
    }

    #[test]
    fn test_combine_two_passes_texture_access() {
        test_init();

        let mut graph = graph! {
            group1 = group_node();
            @ group2 = group_node();

            color1_def = color_node();
            color1_init = PlanNodeData::UndefinedImage;
            color1_a = PlanNodeData::ImageVersion;
            color1_b = PlanNodeData::ImageVersion;
            color1_def -> color1_init = PlanEdge::Origin;
            color1_init -> color1_a = PlanEdge::Origin;
            color1_a -> color1_b = PlanEdge::Version;

            color2_def = color_node();
            color2_init = PlanNodeData::UndefinedImage;
            color2_a = PlanNodeData::ImageVersion;
            color2_b = PlanNodeData::ImageVersion;
            color2_def -> color2_init = PlanEdge::Origin;
            color2_init -> color2_a = PlanEdge::Origin;
            color2_a -> color2_b = PlanEdge::Version;

            color1_a -> group1 = image_edge(ImageUsage::ColorAttachmentWrite);
            group1 -> color1_b = PlanEdge::Origin;
            color1_b -> group2 = image_edge(ImageUsage::Sampled(ShaderUsage::FRAGMENT));
            color2_a -> group2 = image_edge(ImageUsage::ColorAttachmentWrite);
            group2 -> color2_b = PlanEdge::Origin;
        };

        let expected_graph = graph! {
            group1 = subpass_node(1);
            @ group2 = subpass_node(1);

            color1_def = color_node();
            color1_init = PlanNodeData::UndefinedImage;
            color1_a = PlanNodeData::ImageVersion;
            color1_b = PlanNodeData::ImageVersion;
            color1_def -> color1_init = PlanEdge::Origin;
            color1_init -> color1_a = PlanEdge::Origin;
            color1_a -> color1_b = PlanEdge::Version;

            color2_def = color_node();
            color2_init = PlanNodeData::UndefinedImage;
            color2_a = PlanNodeData::ImageVersion;
            color2_b = PlanNodeData::ImageVersion;
            color2_def -> color2_init = PlanEdge::Origin;
            color2_init -> color2_a = PlanEdge::Origin;
            color2_a -> color2_b = PlanEdge::Version;

            color1_a -> group1 = image_edge(ImageUsage::ColorAttachmentWrite);
            group1 -> color1_b = PlanEdge::Origin;
            color1_b -> group2 = image_edge(ImageUsage::Sampled(ShaderUsage::FRAGMENT));
            color2_a -> group2 = image_edge(ImageUsage::ColorAttachmentWrite);
            group2 -> color2_b = PlanEdge::Origin;
        };

        let mut reducer = GraphReducer::new().with_reducer(CombineSubpassesReducer);
        reducer.reduce_graph(&mut graph);

        assert_topo_eq(&expected_graph, &graph);
    }

    #[test]
    fn test_combine_three_passes_with_depth_read() {
        test_init();

        let mut graph = graph! {
            group1 = group_node();
            group2 = group_node();
            @ group3 = group_node();

            color_def = color_node();
            color_init = PlanNodeData::UndefinedImage;
            color_a = PlanNodeData::ImageVersion;
            color_b = PlanNodeData::ImageVersion;
            color_c = PlanNodeData::ImageVersion;
            color_def -> color_init = PlanEdge::Origin;
            color_init -> color_a = PlanEdge::Origin;
            color_a -> color_b = PlanEdge::Version;
            color_b -> color_c = PlanEdge::Version;

            depth_def = depth_node();
            depth_init = PlanNodeData::UndefinedImage;
            depth_a = PlanNodeData::ImageVersion;
            depth_b = PlanNodeData::ImageVersion;
            depth_def -> depth_init = PlanEdge::Origin;
            depth_init -> depth_a = PlanEdge::Origin;
            depth_a -> depth_b = PlanEdge::Version;

            color_a -> group1 = image_edge(ImageUsage::ColorAttachmentWrite);
            depth_a -> group1 = image_edge(ImageUsage::DepthStencilAttachmentWrite);
            group1 -> color_b = PlanEdge::Origin;
            group1 -> depth_b = PlanEdge::Origin;

            color_b -> group2 = image_edge(ImageUsage::ColorAttachmentWrite);
            depth_b -> group2 = image_edge(ImageUsage::DepthStencilAttachmentRead);
            group2 -> color_c = PlanEdge::Origin;

            color_c -> group3 = image_edge(ImageUsage::ColorAttachmentWrite);
            depth_b -> group3 = image_edge(ImageUsage::DepthStencilAttachmentWrite);
        };

        let expected_graph = graph! {
            @ pass = subpass_node(3);

            depth_a = PlanNodeData::ImageVersion;
            depth_def = depth_node();
            color_def = color_node();
            color_init = PlanNodeData::UndefinedImage;
            color_a = PlanNodeData::ImageVersion;
            depth_init = PlanNodeData::UndefinedImage;

            color_def -> color_init = PlanEdge::Origin;
            color_init -> color_a = PlanEdge::Origin;
            depth_def -> depth_init = PlanEdge::Origin;
            depth_init -> depth_a = PlanEdge::Origin;
            depth_a -> pass = image_edge(ImageUsage::DepthStencilAttachmentWrite);
            color_a -> pass = image_edge(ImageUsage::ColorAttachmentWrite);
        };

        let mut reducer = GraphReducer::new().with_reducer(CombineSubpassesReducer);
        reducer.reduce_graph(&mut graph);

        dbg!(&graph);
        dbg!(&expected_graph);

        assert_topo_eq(&expected_graph, &graph);
    }

    #[test]
    fn test_combine_three_passes_with_external_read() {
        test_init();

        let mut graph = graph! {
            group1 = group_node();
            group2 = group_node();
            group3 = group_node();
            @ group4 = group_node();

            color_a = PlanNodeData::ImageVersion;
            color_b = PlanNodeData::ImageVersion;
            color_c = PlanNodeData::ImageVersion;
            depth_a = PlanNodeData::ImageVersion;
            depth_b = PlanNodeData::ImageVersion;
            external_a = PlanNodeData::ImageVersion;

            group4 -> external_a = image_edge(ImageUsage::ColorAttachmentWrite);
            group1 -> color_a = image_edge(ImageUsage::ColorAttachmentWrite);
            group1 -> depth_a = image_edge(ImageUsage::DepthStencilAttachmentWrite);
            color_a -> group2 = image_edge(ImageUsage::ColorAttachmentRead);
            depth_a -> group2 = image_edge(ImageUsage::DepthStencilAttachmentRead);
            external_a -> group2 = image_edge(ImageUsage::Sampled(ShaderUsage::FRAGMENT));
            group2 -> color_b = image_edge(ImageUsage::ColorAttachmentWrite);
            color_b -> group3 = image_edge(ImageUsage::ColorAttachmentRead);
            depth_a -> group3 = image_edge(ImageUsage::DepthStencilAttachmentRead);
            group3 -> color_c = image_edge(ImageUsage::ColorAttachmentWrite);
            group3 -> depth_b = image_edge(ImageUsage::DepthStencilAttachmentWrite);
        };

        let expected_graph = graph! {
            main_pass = subpass_node(3);
            color_a = PlanNodeData::ImageVersion;
            depth_a = PlanNodeData::ImageVersion;
            ext_pass = subpass_node(1);
            external_a = PlanNodeData::ImageVersion;

            ext_pass -> external_a = image_edge(ImageUsage::ColorAttachmentWrite);
            main_pass -> color_a = image_edge(ImageUsage::ColorAttachmentWrite);
            main_pass -> depth_a = image_edge(ImageUsage::DepthStencilAttachmentWrite);
            external_a -> main_pass = image_edge(ImageUsage::Sampled(ShaderUsage::FRAGMENT));
        };

        let mut reducer = GraphReducer::new().with_reducer(CombineSubpassesReducer);
        reducer.reduce_graph(&mut graph);

        assert_topo_eq(&expected_graph, &graph);
    }
}
