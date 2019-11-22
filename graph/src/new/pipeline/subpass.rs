use {
    crate::new::{
        graph::{AttachmentRefEdge, PlanDag, PlanEdge, PlanNode},
        graph_reducer::{GraphEditor, Reducer, Reduction},
        pipeline::{node_ext::NodeExt, Contributions},
        resources::{AttachmentAccess, NodeBufferAccess, NodeImageAccess},
    },
    graphy::{Direction, EdgeIndex, NodeIndex, Walker},
    rendy_core::hal::Backend,
    smallvec::{smallvec, SmallVec},
};

// Find render pass nodes that writes only exclusive attachments
#[derive(Debug)]
pub(super) struct CombineSubpassesReducer;

impl<'a, B: Backend> Reducer<B> for CombineSubpassesReducer {
    fn reduce(&mut self, editor: &mut GraphEditor<B>, node: NodeIndex) -> Reduction {
        if !editor.graph()[node].is_subpass() {
            return Reduction::NoChange;
        }

        let mut potential_merges: SmallVec<[(NodeIndex, bool); 32]> = SmallVec::new();

        // only consider attachments and image/buffer accesses in the first place
        // @Improvement: Merging BufferAccess and ImageAccess is theoretically possible,
        // but let's not deal with generating in-subpass barriers for now.
        let mut inputs = node
            .parents()
            .filter(|graph: &PlanDag<B>, &(edge, _)| graph[edge].is_attachment_ref());

        while let Some((_, input)) = inputs.walk_next(editor.graph()) {
            let origin = input
                .origin(editor.graph())
                .expect("Subpass input node must have an origin");

            if !editor.graph()[origin].is_subpass() {
                continue;
            }

            let merges_index = potential_merges.iter().position(|m| m.0 == origin);

            if let Some(index) = merges_index {
                // already disallowed
                if potential_merges[index].1 == false {
                    continue;
                }
            }

            // do not combine when input is used by other nodes. The combining would effectively
            // destroy that input transient state, but it is required for other nodes to function.
            let allow = input.has_single_user(editor);

            // previous version of attachment/access must either:
            // - be used *directly* by the node
            // - not exist at all
            let allow = allow
                && if let Some(parent_version) = input.version(editor.graph()) {
                    origin.directly_uses(
                        editor.graph(),
                        &editor.context().contributions,
                        parent_version,
                    )
                } else {
                    // previous version doesn't exist
                    true
                };

            if let Some(index) = merges_index {
                potential_merges[index].1 = allow;
            } else {
                potential_merges.push((origin, allow));
            }
        }
        potential_merges.retain(|m| m.1);

        if potential_merges.is_empty() {
            // TODO: maybe consider joining empty subpasses?
            return Reduction::NoChange;
        }

        let current_attachments = UsesSnapshot::build(editor.graph(), node);
        let candidate_merge = potential_merges.into_iter().find_map(|(writer, _)| {
            let writer_attachments = UsesSnapshot::build(editor.graph(), writer);
            if current_attachments.mergeable(
                &writer_attachments,
                editor.graph(),
                &editor.context().contributions,
            ) {
                Some((writer, writer_attachments))
            } else {
                None
            }
        });

        if let Some((merge_target, target_attachments)) = candidate_merge {
            editor
                .graph_mut()
                .rewire_where(Direction::Incoming, node, merge_target, |edge, node| {
                    target_attachments.rewrire(edge, node)
                })
                .unwrap();

            target_attachments.patch(&current_attachments, editor.graph_mut());

            if let Ok((
                PlanNode::RenderSubpass(ref mut this_groups),
                PlanNode::RenderSubpass(ref mut merge_groups),
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

#[derive(Debug, PartialEq, Eq)]
struct UsesSnapshot {
    node: NodeIndex,
    depth: Option<(NodeIndex, EdgeIndex, AttachmentAccess)>,
    // length hardcoded for convenience, must be big enough to contain all possible slots.
    // At the time of writing, max number of supported colors in vulkan gpu database is 8.
    colors: SmallVec<[Option<NodeIndex>; 8]>,
    resolves: SmallVec<[Option<NodeIndex>; 8]>,
    inputs: SmallVec<[Option<NodeIndex>; 8]>,
    buf_accesses: SmallVec<
        [(
            NodeIndex,
            EdgeIndex,
            NodeBufferAccess,
            rendy_core::hal::pso::PipelineStage,
        ); 4],
    >,
    img_accesses: SmallVec<
        [(
            NodeIndex,
            EdgeIndex,
            NodeImageAccess,
            rendy_core::hal::pso::PipelineStage,
        ); 4],
    >,
}

impl UsesSnapshot {
    fn build<B: Backend>(graph: &PlanDag<B>, node: NodeIndex) -> Self {
        let mut snapshot = UsesSnapshot {
            node,
            depth: None,
            colors: smallvec![None; 8],
            resolves: smallvec![None; 8],
            inputs: smallvec![None; 8],
            buf_accesses: smallvec![],
            img_accesses: smallvec![],
        };

        let mut num_colors = 0;
        let mut num_resolves = 0;
        let mut num_inputs = 0;

        for (edge_id, node_id) in node.parents().iter(graph) {
            match &graph[edge_id] {
                PlanEdge::AttachmentRef(edge) => match &edge {
                    AttachmentRefEdge::Color(i) => {
                        snapshot.colors[*i] = Some(node_id);
                        num_colors = num_colors.max(*i + 1);
                    }
                    AttachmentRefEdge::Resolve(i) => {
                        snapshot.resolves[*i] = Some(node_id);
                        num_resolves = num_resolves.max(*i + 1);
                    }
                    AttachmentRefEdge::Input(i) => {
                        snapshot.inputs[*i] = Some(node_id);
                        num_inputs = num_inputs.max(*i + 1);
                    }
                    AttachmentRefEdge::DepthStencil(access) => {
                        snapshot.depth = Some((node_id, edge_id, *access))
                    }
                },
                PlanEdge::BufferAccess(access, stage) => {
                    snapshot
                        .buf_accesses
                        .push((node_id, edge_id, *access, *stage));
                }
                PlanEdge::ImageAccess(access, stage) => {
                    snapshot
                        .img_accesses
                        .push((node_id, edge_id, *access, *stage));
                }
                _ => {}
            }
        }
        snapshot.colors.truncate(num_colors);
        snapshot.resolves.truncate(num_resolves);
        snapshot.inputs.truncate(num_inputs);
        snapshot
    }

    fn rewrire(&self, edge: &PlanEdge, node: NodeIndex) -> bool {
        match edge {
            PlanEdge::AttachmentRef(edge) => match edge {
                // Rewire attachment if it doesn't already exist.
                // Merge into existing ones if needed.
                AttachmentRefEdge::Color(i) => self.colors.get(*i).map_or(true, |i| i.is_none()),
                AttachmentRefEdge::Resolve(i) => {
                    self.resolves.get(*i).map_or(true, |i| i.is_none())
                }
                AttachmentRefEdge::Input(i) => self.inputs.get(*i).map_or(true, |i| i.is_none()),
                AttachmentRefEdge::DepthStencil(..) => self.depth.is_none(),
            },
            PlanEdge::ImageAccess(..) => self.img_accesses.iter().find(|i| i.0 == node).is_none(),
            PlanEdge::BufferAccess(..) => self.buf_accesses.iter().find(|i| i.0 == node).is_none(),
            e => panic!("Trying to rewiring unknown edge {:?}", e),
        }
    }

    fn patch<B: Backend>(self, source: &Self, graph: &mut PlanDag<B>) {
        if let (Some((_, edge, access_a)), Some((_, _, access_b))) = (self.depth, source.depth) {
            if access_a == AttachmentAccess::ReadWrite && access_b == AttachmentAccess::ReadOnly {
                match graph[edge] {
                    PlanEdge::AttachmentRef(AttachmentRefEdge::DepthStencil(ref mut access)) => {
                        *access = AttachmentAccess::ReadWrite;
                    }
                    _ => unreachable!(),
                }
            }
        }

        for (node, edge, access_a, stage_a) in self.buf_accesses {
            if let Some((_, _, access_b, stage_b)) =
                source.buf_accesses.iter().find(|i| i.0 == node)
            {
                if access_a != *access_b || stage_a != *stage_b {
                    match graph[edge] {
                        PlanEdge::BufferAccess(ref mut access, ref mut stage) => {
                            *access = access_a | *access_b;
                            *stage = stage_a | *stage_b;
                        }
                        _ => unreachable!(),
                    }
                }
            }
        }

        for (node, edge, access_a, stage_a) in self.img_accesses {
            if let Some((_, _, access_b, stage_b)) =
                source.img_accesses.iter().find(|i| i.0 == node)
            {
                if access_a != *access_b || stage_a != *stage_b {
                    match graph[edge] {
                        PlanEdge::ImageAccess(ref mut access, ref mut stage) => {
                            *access = access_a | *access_b;
                            *stage = stage_a | *stage_b;
                        }
                        _ => unreachable!(),
                    }
                }
            }
        }
    }

    fn mergeable<B: Backend>(
        &self,
        other: &Self,
        graph: &PlanDag<B>,
        contributions: &Contributions,
    ) -> bool {
        if !match (self.depth, other.depth) {
            (Some((depth_a, _, access_a)), Some((depth_b, _, _))) => {
                if access_a == AttachmentAccess::ReadOnly {
                    depth_a == depth_b
                } else {
                    depth_a.version(graph) == Some(depth_b)
                }
            }
            _ => true,
        } {
            return false;
        }

        let mut colors = self.colors.iter().zip(&other.colors);
        let mut resolves = self.resolves.iter().zip(&other.resolves);
        let mut inputs = self.inputs.iter().zip(&other.inputs);

        if !colors.all(|(color_a, color_b)| match (color_a, color_b) {
            (Some(color_a), Some(color_b)) => color_a.version(graph) == Some(*color_b),
            _ => true,
        }) {
            return false;
        }

        if !resolves.all(|(resolve_a, resolve_b)| match (resolve_a, resolve_b) {
            (Some(resolve_a), Some(resolve_b)) => resolve_a.version(graph) == Some(*resolve_b),
            _ => true,
        }) {
            return false;
        }

        if !inputs.all(|(input_a, input_b)| match (input_a, input_b) {
            (Some(input_a), Some(input_b)) => input_a == input_b,
            _ => true,
        }) {
            return false;
        }

        if !other.img_accesses.iter().all(|i| {
            self.node
                .directly_uses_or_not_at_all(graph, contributions, i.0)
        }) {
            return false;
        }

        if !other.buf_accesses.iter().all(|i| {
            self.node
                .directly_uses_or_not_at_all(graph, contributions, i.0)
        }) {
            return false;
        }

        return true;
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
        graph_reducer::{GraphReducer, ReductionContext},
        node::{NodeId, PassFn},
        resources::{ImageId, ImageUsage, ShaderUsage},
        test::{assert_equivalent, assert_topo_eq, test_init, TestBackend},
    };
    use smallvec::SmallVec;

    fn subpass_node<'a>(num_groups: usize) -> PlanNode<'a, TestBackend> {
        let mut vec: SmallVec<[(NodeId, PassFn<'a, TestBackend>); 4]> =
            SmallVec::with_capacity(num_groups);
        for _ in 0..num_groups {
            vec.push((NodeId(0), Box::new(|_| Ok(()))))
        }
        PlanNode::RenderSubpass(vec)
    }

    fn group_node<'a>() -> PlanNode<'a, TestBackend> {
        subpass_node(1)
    }

    fn image_edge(usage: ImageUsage) -> PlanEdge {
        PlanEdge::ImageAccess(usage.access(), usage.stage())
    }

    fn color_node<'a>(id: usize) -> PlanNode<'a, TestBackend> {
        PlanNode::Image(ImageNode {
            id: ImageId(NodeId(0), id),
            kind: rendy_core::hal::image::Kind::D2(1024, 1024, 1, 1),
            levels: 1,
            format: rendy_core::hal::format::Format::Rgba8Unorm,
        })
    }

    fn depth_node<'a>(id: usize) -> PlanNode<'a, TestBackend> {
        PlanNode::Image(ImageNode {
            id: ImageId(NodeId(0), id),
            kind: rendy_core::hal::image::Kind::D2(1024, 1024, 1, 1),
            levels: 1,
            format: rendy_core::hal::format::Format::R32Sfloat,
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

            color_def = color_node(0);
            color_init = PlanNode::UndefinedImage;
            color_a = PlanNode::ImageVersion;
            color_b = PlanNode::ImageVersion;
            color_c = PlanNode::ImageVersion;

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

            color_def = color_node(0);
            color_init = PlanNode::UndefinedImage;
            color_a = PlanNode::ImageVersion;

            color_def -> color_init = PlanEdge::Origin;
            color_init -> color_a = PlanEdge::Origin;
            color_a -> pass = image_edge(ImageUsage::ColorAttachmentWrite);
        };

        let ctx = ReductionContext::new(Contributions::collect(&graph, &alloc));
        GraphReducer::new()
            .with_reducer(CombineSubpassesReducer)
            .reduce_graph(&mut graph, &ctx, &alloc);
        graph.trim(NodeIndex::new(0)).unwrap();

        assert_topo_eq(
            &graph,
            &[
                PlanNode::Root,
                subpass_node(2),
                PlanNode::ImageVersion,
                PlanNode::UndefinedImage,
                color_node(0),
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

            color1_def = color_node(0);
            color1_init = PlanNode::UndefinedImage;
            color1_a = PlanNode::ImageVersion;
            color1_b = PlanNode::ImageVersion;
            color1_def -> color1_init = PlanEdge::Origin;
            color1_init -> color1_a = PlanEdge::Origin;
            color1_a -> color1_b = PlanEdge::Version;

            color2_def = color_node(1);
            color2_init = PlanNode::UndefinedImage;
            color2_a = PlanNode::ImageVersion;
            color2_b = PlanNode::ImageVersion;
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

            color1_def = color_node(0);
            color1_init = PlanNode::UndefinedImage;
            color1_a = PlanNode::ImageVersion;
            color1_b = PlanNode::ImageVersion;
            color1_def -> color1_init = PlanEdge::Origin;
            color1_init -> color1_a = PlanEdge::Origin;
            color1_a -> color1_b = PlanEdge::Version;

            color2_def = color_node(1);
            color2_init = PlanNode::UndefinedImage;
            color2_a = PlanNode::ImageVersion;
            color2_b = PlanNode::ImageVersion;
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
        let ctx = ReductionContext::new(Contributions::collect(&graph, &alloc));
        expected_graph.trim(NodeIndex::new(0)).unwrap();
        reducer.reduce_graph(&mut graph, &ctx, &alloc);
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

            color_def = color_node(0);
            color_init = PlanNode::UndefinedImage;
            color_a = PlanNode::ImageVersion;
            color_b = PlanNode::ImageVersion;
            color_c = PlanNode::ImageVersion;
            color_def -> color_init = PlanEdge::Origin;
            color_init -> color_a = PlanEdge::Origin;
            color_a -> color_b = PlanEdge::Version;
            color_b -> color_c = PlanEdge::Version;

            depth_def = depth_node(1);
            depth_init = PlanNode::UndefinedImage;
            depth_a = PlanNode::ImageVersion;
            depth_b = PlanNode::ImageVersion;
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

            depth_a = PlanNode::ImageVersion;
            depth_def = depth_node(1);
            color_def = color_node(0);
            color_init = PlanNode::UndefinedImage;
            color_a = PlanNode::ImageVersion;
            depth_init = PlanNode::UndefinedImage;

            color_def -> color_init = PlanEdge::Origin;
            color_init -> color_a = PlanEdge::Origin;
            depth_def -> depth_init = PlanEdge::Origin;
            depth_init -> depth_a = PlanEdge::Origin;
            color_a -> pass = image_edge(ImageUsage::ColorAttachmentWrite);
            depth_a -> pass = image_edge(ImageUsage::DepthStencilAttachmentWrite);
        };

        let mut reducer = GraphReducer::new().with_reducer(CombineSubpassesReducer);
        graph.trim(NodeIndex::new(0)).unwrap();
        let ctx = ReductionContext::new(Contributions::collect(&graph, &alloc));
        reducer.reduce_graph(&mut graph, &ctx, &alloc);

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
            @ debug_pass = PlanNode::PostSubmit(NodeId(0), Box::new(|_| Ok(())));

            depth_def = depth_node(0);
            color_def = color_node(1);
            extra_def = color_node(2);

            color_init = PlanNode::UndefinedImage;
            depth_init = PlanNode::UndefinedImage;
            extra_init = PlanNode::UndefinedImage;

            color_a = PlanNode::ImageVersion;
            color_b = PlanNode::ImageVersion;
            color_c = PlanNode::ImageVersion;
            color_d = PlanNode::ImageVersion;
            depth_a = PlanNode::ImageVersion;
            depth_b = PlanNode::ImageVersion;
            depth_c = PlanNode::ImageVersion;
            extra_a = PlanNode::ImageVersion;
            extra_b = PlanNode::ImageVersion;
            extra_c = PlanNode::ImageVersion;

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
            @ debug_pass = PlanNode::PostSubmit(NodeId(0), Box::new(|_| Ok(())));

            depth_def = depth_node(0);
            color_def = color_node(1);
            extra_def = color_node(1);

            color_init = PlanNode::UndefinedImage;
            depth_init = PlanNode::UndefinedImage;
            extra_init = PlanNode::UndefinedImage;

            color_a = PlanNode::ImageVersion;
            depth_a = PlanNode::ImageVersion;
            extra_a = PlanNode::ImageVersion;
            extra_b = PlanNode::ImageVersion;
            extra_c = PlanNode::ImageVersion;

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
            // main_pass -> debug_pass = PlanEdge::Effect;
        };

        let ctx = ReductionContext::new(Contributions::collect(&graph, &alloc));
        GraphReducer::new()
            .with_reducer(CombineSubpassesReducer)
            // .with_reducer(OrderWritesReducer)
            .reduce_graph(&mut graph, &ctx, &alloc);

        graph.trim(NodeIndex::new(0)).unwrap();
        expected_graph.trim(NodeIndex::new(0)).unwrap();

        assert_equivalent(&expected_graph, &graph);
    }
}
