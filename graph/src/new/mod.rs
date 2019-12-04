// TODO: after all required nodes are implemented, disallow dead code and remove what's left
#![allow(dead_code)]

//! A new implementation of rendering graph

// #[cfg(test)]
#[macro_use]
mod test;

mod graph;
mod graph_reducer;
mod node;
mod nodes;
mod pipeline;
mod resources;
mod track;
mod walker;

pub use graph::{
    ExecPassContext, FamilyType, Graph, GraphBuildError, GraphBuilder, GraphRunError, NodeCtx,
};
pub use node::{ConstructResult, Node, NodeBuildError, NodeBuilder, NodeExecution, Parameter};
pub use nodes::*;
pub use resources::{BufferId, ImageId, RenderPassId, SubpassId, VirtualId, WaitId};
pub use track::Track;

// fn setup_graph(g: GraphBuilder) {
//     let color = g.add(OutputToWindow::new());
//     let depth = g.add(DepthPrepass::new());

//     let partitions = g.add(LogReduceDepthPartitions::new(depth));
//     let shadow_maps = g.add(RenderShadowDepth::new(partitions).sized(1024).multisample(4));
//     let evsm_shadows = g.add(EvsmShadowReduce::new(shadow_maps));
//     g.add(GenerateMipmaps::new(evsm_shadows).mip_levels(4));
//     g.add(MipGaussianBlurW::new(evsm_shadows).kernel_size(4));
//     g.add(MipGaussianBlurH::new(evsm_shadows).kernel_size(4));

//     g.add(RenderOpaquePbr::new(color, depth, evsm_shadows));
//     g.add(RenderOpaquePbrSkinned::new(color, depth, evsm_shadows));
//     g.add(RenderTransparentPbr::new(color, depth, evsm_shadows));
//     g.add(RenderTransparentPbrSkinned::new(color, depth, evsm_shadows));
// }
