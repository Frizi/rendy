use {
    crate::new::{
        graph::{
            AttachmentClear, AttachmentRefEdge, PlanDag, PlanEdge, PlanNode, RenderPassAtachment,
            RenderPassNode, RenderPassSubpass,
        },
        graph_reducer::{GraphEditor, Reducer, Reduction},
        pipeline::node_ext::NodeExt,
    },
    gfx_hal::{
        pass::{AttachmentLoadOp, AttachmentOps, AttachmentStoreOp},
        Backend,
    },
    graphy::{NodeIndex, Walker},
    smallvec::SmallVec,
};

#[derive(Debug)]
pub(super) struct CombinePassesReducer;

impl<'a, B: Backend, T: ?Sized> Reducer<B, T> for CombinePassesReducer {
    fn reduce(&mut self, editor: &mut GraphEditor<B, T>, node: NodeIndex) -> Reduction {
        // combining passes is done from top to bottom.
        // Subpass can only be translated into pass when it doesn't dpeend on other subpasses.
        //
        // First stage - convert subpass to pass in place.
        // Second stage - append subpass to a pass it uses.

        if !editor.graph()[node].is_subpass() {
            return Reduction::NoChange;
        }

        // only consider attachments and image/buffer accesses in the first place
        // @Improvement: merge buffers/images, take them into account when constructing subpass dependencies
        let mut attachments = node
            .parents()
            .filter(|graph: &PlanDag<B, T>, &(edge, _)| graph[edge].is_attachment_ref());

        let num_subpasses = attachments
            .clone()
            .iter(editor.graph())
            .filter(|(_, node)| {
                let origin = node
                    .origin(editor.graph())
                    .expect("Attachment source node must have an orign");
                editor.graph()[origin].is_subpass()
            })
            .count();

        // this forces the reduction to happen from leaves to root.
        // That way we will only merge with passes that are already reduced.
        if num_subpasses != 0 {
            return Reduction::NoChange;
        }

        let mut potential_merges: SmallVec<[PotentialMerge; 32]> = SmallVec::new();

        for (edge, input) in attachments.clone().iter(editor.graph()) {
            let origin = input
                .origin(editor.graph())
                .expect("Attachment source node must have an orign");

            if !editor.graph()[origin].is_pass() {
                continue;
            }

            let merges_index = potential_merges.iter().position(|m| m.origin() == origin);

            if let Some(index) = merges_index {
                if !potential_merges[index].is_allowed() {
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

            if !allow {
                match merges_index {
                    Some(index) => potential_merges[index].forbid(),
                    None => potential_merges.push(PotentialMerge::Forbid(origin)),
                };
            } else {
                let merge = match merges_index {
                    Some(index) => &mut potential_merges[index],
                    None => {
                        potential_merges.push(PotentialMerge::new_allowed(origin));
                        let last = potential_merges.len() - 1;
                        &mut potential_merges[last]
                    }
                };
                let merge = merge.allowed_mut().unwrap();
                let attachment = editor.graph()[edge]
                    .attachment_ref()
                    .expect("Filtered attachments contains non-attachment");

                merge.num_attachments += 1;
                match attachment {
                    AttachmentRefEdge::Input(_) => merge.num_inputs += 1,
                    _ => {}
                }
            }
        }

        let merge_pass: Option<NodeIndex> = potential_merges
            .into_iter()
            .filter_map(|m| m.into_allowed())
            .filter(|allowed| {
                // only allow merging with nodes that do not contribute through
                // anything else than attachment edges we are about to remove
                let total_contributions = node.count_contributrions(editor, allowed.origin);
                total_contributions == allowed.num_attachments as usize
            })
            .max_by_key(|merge| {
                // score mergeable passes by, in order:
                // - has the most common INPUT attachments
                // - has the most common attachments overall
                merge.num_inputs as usize * 256 + merge.num_attachments as usize
            })
            .map(|m| m.origin);

        let mut subpass = {
            let groups = match editor.graph_mut()[node] {
                PlanNode::RenderSubpass(ref mut groups) => {
                    std::mem::replace(groups, Default::default())
                }
                _ => unreachable!(),
            };
            RenderPassSubpass::new(groups)
        };

        if let Some(pass_node) = merge_pass {
            // this subpass is being merged with existing render pass

            // gather all existing attachments in the pass
            let pass_attachments = pass_node
                .parents()
                .filter(|graph: &PlanDag<B, T>, &(edge, _)| graph[edge].is_pass_attachment());

            let mut num_attachments = pass_attachments.clone().iter(editor.graph()).count();
            let subpass_index = editor.graph_mut()[pass_node]
                .pass_mut()
                .unwrap()
                .subpasses
                .len();

            let mut new_deps = SmallVec::new();

            // enumerate attachments used by this subpass
            while let Some((edge, writer)) = attachments.walk_next(editor.graph()) {
                let comp_version = if writer.origin(editor.graph()).unwrap() == pass_node {
                    writer.version(editor.graph()).unwrap()
                } else {
                    writer
                };

                if let Some((existing_index, existing_edge)) = pass_attachments
                    .clone()
                    .iter(editor.graph())
                    .enumerate()
                    .find(|(_, (_, node))| *node == comp_version)
                    .map(|(i, (edge, _))| (i, edge))
                {
                    let (pass_att_edge, subpass_att_edge) = editor
                        .graph_mut()
                        .edge_pair_mut(existing_edge, edge)
                        .unwrap();

                    let pass_att = pass_att_edge.pass_attachment_mut().unwrap();
                    let subpass_att = subpass_att_edge.attachment_ref().unwrap();
                    process_existing_attachment(
                        subpass_att,
                        &mut subpass,
                        subpass_index,
                        &mut new_deps,
                        pass_att,
                        existing_index,
                    );
                } else {
                    let (ops, clear) = attachment_ops(&mut editor.graph_mut()[writer]);
                    let new_edge = process_new_attachment(
                        &mut editor.graph_mut()[edge].attachment_ref().unwrap(),
                        &mut subpass,
                        subpass_index,
                        ops,
                        clear,
                        num_attachments,
                    );
                    num_attachments += 1;
                    editor
                        .insert_edge_unchecked(
                            writer,
                            pass_node,
                            PlanEdge::PassAttachment(new_edge),
                        )
                        .unwrap();
                };
            }

            let pass = editor.graph_mut()[pass_node].pass_mut().unwrap();
            pass.subpasses.push(subpass);
            pass.deps.extend(new_deps.drain());
            Reduction::Replace(pass_node)
        } else {
            let mut pass_node = RenderPassNode::new();
            // we've got a beginning of brand new render pass
            let mut num_attachments = 0;
            while let Some((edge, writer)) = attachments.walk_next(editor.graph()) {
                let (ops, clear) = attachment_ops(&mut editor.graph_mut()[writer]);
                let new_edge = process_new_attachment(
                    &mut editor.graph_mut()[edge].attachment_ref().unwrap(),
                    &mut subpass,
                    0,
                    ops,
                    clear,
                    num_attachments,
                );
                editor.graph_mut()[edge] = PlanEdge::PassAttachment(new_edge);
                num_attachments += 1;
            }

            pass_node.subpasses.push(subpass);
            editor.graph_mut()[node] = PlanNode::RenderPass(pass_node);
            Reduction::Changed
        }
    }
}

fn attachment_ops<B: Backend, T: ?Sized>(
    writer_node: &mut PlanNode<B, T>,
) -> (AttachmentOps, AttachmentClear) {
    let (clear, load) = match writer_node {
        PlanNode::ImageVersion => (AttachmentClear(None), AttachmentLoadOp::Load),
        PlanNode::LoadImage(..) => (AttachmentClear(None), AttachmentLoadOp::Load),
        PlanNode::UndefinedImage => (AttachmentClear(None), AttachmentLoadOp::DontCare),
        PlanNode::ClearImage(clear) => {
            let clear = *clear;
            *writer_node = PlanNode::UndefinedImage;
            (AttachmentClear(Some(clear)), AttachmentLoadOp::Clear)
        }
        _ => unreachable!(),
    };

    let ops = AttachmentOps {
        load,
        // Store ops are pesimistically set to `Store` as a safe default.
        // Further reduction passes can reduce the strenght of this when suitable conditions are met
        store: AttachmentStoreOp::Store,
    };
    (ops, clear)
}

fn process_new_attachment<B: Backend, T: ?Sized>(
    ref_edge: &AttachmentRefEdge,
    subpass: &mut RenderPassSubpass<B, T>,
    subpass_index: usize,
    ops: AttachmentOps,
    clear: AttachmentClear,
    attachment_index: usize,
) -> RenderPassAtachment {
    let attachment_ref = (attachment_index, ref_edge.layout());

    let mut attachment = RenderPassAtachment {
        ops,
        stencil_ops: AttachmentOps::DONT_CARE,
        clear,
        first_access: subpass_index as _,
        last_access: subpass_index as _,
        first_write: None,
        last_write: None,
        queued_reads: SmallVec::new(),
        first_layout: attachment_ref.1,
    };

    match ref_edge {
        AttachmentRefEdge::Color(id) => {
            subpass.set_color(*id, attachment_ref);
        }
        AttachmentRefEdge::Resolve(id) => {
            subpass.set_resolve(*id, attachment_ref);
        }
        AttachmentRefEdge::Input(id) => {
            subpass.set_input(*id, attachment_ref);
        }
        AttachmentRefEdge::DepthStencil(_) => {
            subpass.set_depth_stencil(attachment_ref);
            attachment.stencil_ops = attachment.ops;
        }
    }

    let stages = ref_edge.stages();
    let accesses = ref_edge.accesses();

    if ref_edge.is_write() {
        attachment.first_write = Some((subpass_index as _, stages, accesses));
        attachment.last_write = attachment.first_write;
    } else {
        attachment
            .queued_reads
            .push((subpass_index as _, stages, accesses));
    }

    attachment
}

fn process_existing_attachment<B: Backend, T: ?Sized>(
    ref_edge: &AttachmentRefEdge,
    subpass: &mut RenderPassSubpass<B, T>,
    subpass_index: usize,
    deps: &mut SmallVec<[gfx_hal::pass::SubpassDependency; 32]>,
    attachment: &mut RenderPassAtachment,
    attachment_index: usize,
) {
    let attachment_ref = (attachment_index, ref_edge.layout());

    match ref_edge {
        AttachmentRefEdge::Color(id) => {
            subpass.set_color(*id, attachment_ref);
        }
        AttachmentRefEdge::Resolve(id) => {
            subpass.set_resolve(*id, attachment_ref);
        }
        AttachmentRefEdge::Input(id) => {
            subpass.set_input(*id, attachment_ref);
        }
        AttachmentRefEdge::DepthStencil(_) => {
            subpass.set_depth_stencil(attachment_ref);
            attachment.stencil_ops = attachment.ops;
        }
    };

    let stages = ref_edge.stages();
    let accesses = ref_edge.accesses();
    attachment.last_access = subpass_index as _;
    if ref_edge.is_write() {
        attachment.last_write = Some((subpass_index as _, stages, accesses));
        if attachment.first_write.is_none() {
            attachment.first_write = attachment.last_write;
        }
        deps.extend(attachment.queued_reads.drain().map(
            |(read_index, read_stages, read_accesses)| gfx_hal::pass::SubpassDependency {
                passes: gfx_hal::pass::SubpassRef::Pass(read_index as _)
                    ..gfx_hal::pass::SubpassRef::Pass(subpass_index),
                stages: read_stages..stages,
                accesses: read_accesses..accesses,
            },
        ));
    } else {
        if let Some((write_index, write_stages, write_accesses)) = attachment.last_write {
            deps.push(gfx_hal::pass::SubpassDependency {
                passes: gfx_hal::pass::SubpassRef::Pass(write_index as _)
                    ..gfx_hal::pass::SubpassRef::Pass(subpass_index),
                stages: write_stages..stages,
                accesses: write_accesses..accesses,
            });
        }
        attachment
            .queued_reads
            .push((subpass_index as _, stages, accesses));
    }
}

struct AllowedMerge {
    origin: NodeIndex,
    num_attachments: u8,
    num_inputs: u8,
}
enum PotentialMerge {
    Allow(AllowedMerge),
    Forbid(NodeIndex),
}
impl PotentialMerge {
    fn new_allowed(origin: NodeIndex) -> Self {
        Self::Allow(AllowedMerge {
            origin,
            num_attachments: 0,
            num_inputs: 0,
        })
    }
    fn is_allowed(&self) -> bool {
        match self {
            Self::Allow(_) => true,
            _ => false,
        }
    }
    fn allowed_mut(&mut self) -> Option<&mut AllowedMerge> {
        match self {
            Self::Allow(inner) => Some(inner),
            _ => None,
        }
    }
    fn into_allowed(self) -> Option<AllowedMerge> {
        match self {
            Self::Allow(inner) => Some(inner),
            _ => None,
        }
    }

    fn origin(&self) -> NodeIndex {
        match self {
            Self::Allow(inner) => inner.origin,
            Self::Forbid(origin) => *origin,
        }
    }

    fn forbid(&mut self) {
        let origin = self.origin();
        *self = Self::Forbid(origin);
    }
}
