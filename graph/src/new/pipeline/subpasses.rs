use {
    crate::new::{
        graph::{PlanEdge, PlanNodeData},
        graph_reducer::{GraphEditor, Reducer, Reduction},
    },
    gfx_hal::Backend,
    graphy::{Direction, NodeIndex, Walker},
};

// Find render pass nodes that writes only exclusive attachments
#[derive(Debug)]
pub(super) struct CombineSubpassesReducer;

impl<'a, B: Backend, T: ?Sized> Reducer<B, T> for CombineSubpassesReducer {
    fn reduce(&mut self, editor: &mut GraphEditor<B, T>, node: NodeIndex) -> Reduction {
        let mut candidate_merge = None;

        if let Some(PlanNodeData::RenderSubpass(_)) = editor.graph().get_node(node) {
            let mut read_attachments =
                node.parents()
                    .filter(|graph, &(edge, _)| match graph[edge] {
                        PlanEdge::ImageAccess(access, _) => access.is_attachment(),
                        _ => false,
                    });

            // Do all attachments I read come from the same node? If so, which node?
            while let Some((_, access_node)) = read_attachments.walk_next(editor.graph()) {
                let (_, writer) = access_node
                    .parents()
                    .filter(|graph, &(edge, _)| graph[edge] == PlanEdge::Origin)
                    .walk_next(editor.graph())
                    .expect("Attachment node must have an orign");

                match candidate_merge {
                    Some(candidate) if candidate != writer => {
                        log::trace!("CombineSubpassesReducer bailout: more than one writer");
                        return Reduction::NoChange;
                    }
                    None => match &editor.graph()[writer] {
                        PlanNodeData::RenderSubpass(_) => candidate_merge = Some(writer),
                        _ => {
                            log::trace!("CombineSubpassesReducer bailout: written by non-subpass");
                            return Reduction::NoChange;
                        }
                    },
                    _ => {}
                }
            }
        } else {
            return Reduction::NoChange;
        }

        if let Some(merge_target) = candidate_merge {
            editor
                .graph_mut()
                .rewire_where(Direction::Incoming, node, merge_target, |edge, _| {
                    // TODO: actually rewire attachments that don't exist yet.
                    // This is now not required only because we merge only exact matches.
                    !edge.is_attachment()
                })
                .unwrap();

            if let Ok((
                PlanNodeData::RenderSubpass(ref mut this_groups),
                PlanNodeData::RenderSubpass(ref mut merge_groups),
            )) = editor.graph_mut().node_pair_mut(node, merge_target)
            {
                merge_groups.extend(this_groups.drain());
            } else {
                unreachable!();
            }

            Reduction::Replace(merge_target)
        } else {
            Reduction::NoChange
        }
    }
}

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
        test::{assert_equivalent, assert_topo_eq, test_init, TestBackend},
    };
    use smallvec::SmallVec;

    fn subpass_node<'a>(num_groups: usize) -> PlanNodeData<'a, TestBackend, ()> {
        let mut vec: SmallVec<[PassFn<'a, TestBackend, ()>; 4]> =
            SmallVec::with_capacity(num_groups);
        for _ in 0..num_groups {
            vec.push(Box::new(|_, _| Ok(())))
        }
        PlanNodeData::RenderSubpass(vec)
    }

    fn group_node<'a>() -> PlanNodeData<'a, TestBackend, ()> {
        subpass_node(1)
    }

    fn image_edge(usage: ImageUsage) -> PlanEdge {
        PlanEdge::ImageAccess(usage.access(), usage.stage())
    }

    fn color_node<'a>() -> PlanNodeData<'a, TestBackend, ()> {
        PlanNodeData::Image(ImageNode {
            kind: gfx_hal::image::Kind::D2(1024, 1024, 1, 1),
            levels: 1,
            format: gfx_hal::format::Format::Rgba8Unorm,
        })
    }

    fn depth_node<'a>() -> PlanNodeData<'a, TestBackend, ()> {
        PlanNodeData::Image(ImageNode {
            kind: gfx_hal::image::Kind::D2(1024, 1024, 1, 1),
            levels: 1,
            format: gfx_hal::format::Format::R32Sfloat,
        })
    }

    #[test]
    fn test_combine_two_passes() {
        test_init();
        let alloc = graphy::GraphAllocator::with_capacity(655360);

        let mut graph = graph! {
            [&alloc],
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
            [&alloc],
            @ pass = subpass_node(2);

            color_def = color_node();
            color_init = PlanNodeData::UndefinedImage;
            color_a = PlanNodeData::ImageVersion;

            color_def -> color_init = PlanEdge::Origin;
            color_init -> color_a = PlanEdge::Origin;
            color_a -> pass = image_edge(ImageUsage::ColorAttachmentWrite);
        };

        GraphReducer::new()
            .with_reducer(CombineSubpassesReducer)
            .reduce_graph(&mut graph, &alloc);
        graph.trim(NodeIndex::new(0)).unwrap();

        assert_topo_eq(
            &graph,
            &[
                PlanNodeData::Root,
                subpass_node(2),
                PlanNodeData::ImageVersion,
                PlanNodeData::UndefinedImage,
                color_node(),
            ],
        );

        assert_equivalent(&expected_graph, &graph);
    }

    #[test]
    fn test_combine_two_passes_texture_access() {
        test_init();
        let alloc = graphy::GraphAllocator::with_capacity(655360);

        let mut graph = graph! {
            [&alloc],
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

        let mut expected_graph = graph! {
            [&alloc],
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
        reducer.reduce_graph(&mut graph, &alloc);
        graph.trim(NodeIndex::new(0)).unwrap();
        expected_graph.trim(NodeIndex::new(0)).unwrap();

        assert_equivalent(&expected_graph, &graph);
    }

    #[test]
    fn test_combine_three_passes_with_depth_read() {
        test_init();
        let alloc = graphy::GraphAllocator::with_capacity(655360);

        let mut graph = graph! {
            [&alloc],
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
            [&alloc],
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
            color_a -> pass = image_edge(ImageUsage::ColorAttachmentWrite);
            depth_a -> pass = image_edge(ImageUsage::DepthStencilAttachmentWrite);
        };

        let mut reducer = GraphReducer::new().with_reducer(CombineSubpassesReducer);
        reducer.reduce_graph(&mut graph, &alloc);
        graph.trim(NodeIndex::new(0)).unwrap();

        assert_equivalent(&expected_graph, &graph);
    }

    #[test]
    fn test_combine_three_passes_with_external_read() {
        test_init();
        let alloc = graphy::GraphAllocator::with_capacity(655360);

        let mut graph = graph! {
            [&alloc],
            group1 = group_node();
            group2 = group_node();
            @ group3 = group_node();
            group4 = group_node();
            @ debug_pass = PlanNodeData::Run(Box::new(|_, _| Ok(())));

            depth_def = depth_node();
            color_def = color_node();
            extra_def = color_node();

            color_init = PlanNodeData::UndefinedImage;
            depth_init = PlanNodeData::UndefinedImage;
            extra_init = PlanNodeData::UndefinedImage;

            color_a = PlanNodeData::ImageVersion;
            color_b = PlanNodeData::ImageVersion;
            color_c = PlanNodeData::ImageVersion;
            color_d = PlanNodeData::ImageVersion;
            depth_a = PlanNodeData::ImageVersion;
            depth_b = PlanNodeData::ImageVersion;
            depth_c = PlanNodeData::ImageVersion;
            extra_a = PlanNodeData::ImageVersion;
            extra_b = PlanNodeData::ImageVersion;
            extra_c = PlanNodeData::ImageVersion;

            color_def -> color_init = PlanEdge::Origin;
            color_init -> color_a = PlanEdge::Origin;
            depth_def -> depth_init = PlanEdge::Origin;
            depth_init -> depth_a = PlanEdge::Origin;
            extra_def -> extra_init = PlanEdge::Origin;
            extra_init -> extra_a = PlanEdge::Origin;
            color_a -> color_b = PlanEdge::Version;
            color_b -> color_c = PlanEdge::Version;
            color_c -> color_d = PlanEdge::Version;
            depth_a -> depth_b = PlanEdge::Version;
            depth_b -> depth_c = PlanEdge::Version;
            extra_a -> extra_b = PlanEdge::Version;
            extra_b -> extra_c = PlanEdge::Version;

            color_a -> group1 = image_edge(ImageUsage::ColorAttachmentWrite);
            depth_a -> group1 = image_edge(ImageUsage::DepthStencilAttachmentWrite);
            group1 -> color_b = PlanEdge::Origin;
            group1 -> depth_b = PlanEdge::Origin;

            extra_a -> group4 = image_edge(ImageUsage::ColorAttachmentWrite);
            group4 -> extra_b = PlanEdge::Origin;

            color_b -> group2 = image_edge(ImageUsage::ColorAttachmentWrite);
            depth_b -> group2 = image_edge(ImageUsage::DepthStencilAttachmentRead);
            extra_b -> group2 = image_edge(ImageUsage::Sampled(ShaderUsage::FRAGMENT));
            group2 -> color_c = PlanEdge::Origin;

            color_c -> group3 = image_edge(ImageUsage::ColorAttachmentRead);
            depth_b -> group3 = image_edge(ImageUsage::DepthStencilAttachmentRead);
            group3 -> color_c = image_edge(ImageUsage::ColorAttachmentWrite);
            group3 -> depth_b = image_edge(ImageUsage::DepthStencilAttachmentWrite);

            extra_b -> debug_pass = image_edge(ImageUsage::ColorAttachmentWrite);
            debug_pass -> extra_c = PlanEdge::Origin;
        };

        let mut expected_graph = graph! {
            [&alloc],
            @ main_pass = subpass_node(3);
            ext_pass = subpass_node(1);
            @ debug_pass = PlanNodeData::Run(Box::new(|_, _| Ok(())));

            depth_def = depth_node();
            color_def = color_node();
            extra_def = color_node();

            color_init = PlanNodeData::UndefinedImage;
            depth_init = PlanNodeData::UndefinedImage;
            extra_init = PlanNodeData::UndefinedImage;

            color_a = PlanNodeData::ImageVersion;
            depth_a = PlanNodeData::ImageVersion;
            extra_a = PlanNodeData::ImageVersion;
            extra_b = PlanNodeData::ImageVersion;
            extra_c = PlanNodeData::ImageVersion;

            color_def -> color_init = PlanEdge::Origin;
            color_init -> color_a = PlanEdge::Origin;
            depth_def -> depth_init = PlanEdge::Origin;
            depth_init -> depth_a = PlanEdge::Origin;
            extra_def -> extra_init = PlanEdge::Origin;
            extra_init -> extra_a = PlanEdge::Origin;
            extra_a -> extra_b = PlanEdge::Version;
            extra_b -> extra_c = PlanEdge::Version;

            extra_a -> ext_pass = image_edge(ImageUsage::ColorAttachmentWrite);
            ext_pass -> extra_b = PlanEdge::Origin;
            color_a -> main_pass = image_edge(ImageUsage::ColorAttachmentWrite);
            depth_a -> main_pass = image_edge(ImageUsage::DepthStencilAttachmentWrite);
            extra_b -> main_pass = image_edge(ImageUsage::Sampled(ShaderUsage::FRAGMENT));
            extra_b -> debug_pass = image_edge(ImageUsage::ColorAttachmentWrite);
            debug_pass -> extra_c = PlanEdge::Origin;
            main_pass -> debug_pass = PlanEdge::Effect;
        };

        GraphReducer::new()
            .with_reducer(CombineSubpassesReducer)
            .with_reducer(OrderWritesReducer)
            .reduce_graph(&mut graph, &alloc);

        graph.trim(NodeIndex::new(0)).unwrap();
        expected_graph.trim(NodeIndex::new(0)).unwrap();

        assert_equivalent(&expected_graph, &graph);
    }
}
