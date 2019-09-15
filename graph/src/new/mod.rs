mod node;
mod nodes;
mod resources;
mod track;

use {
    crate::{command::Families, factory::Factory},
    gfx_hal::Backend,
    node::{
        DynNode, DynNodeBuilder, Node, NodeBuilder, NodeConstructionError, NodeExecutionError,
        NodeId,
    },
};

pub struct GraphBuilder<B: Backend, T: ?Sized> {
    nodes: Vec<Box<dyn DynNodeBuilder<B, T>>>,
}

pub enum GraphBuildError {}

impl<B: Backend, T: ?Sized> GraphBuilder<B, T> {
    pub fn add<N: NodeBuilder<B, T> + 'static>(
        &mut self,
        builder: N,
    ) -> <N::Node as Node<B, T>>::Outputs {
        let _node_id = self.nodes.len();
        self.nodes.push(Box::new(builder));

        unimplemented!()
        // (
        //     ParameterId(node_id, 0),
        //     ParameterId(node_id, 1),
        //     ParameterId(node_id, 2),
        // )
    }

    pub fn build(
        self,
        _factory: &mut Factory<B>,
        _families: &mut Families<B>,
        _aux: &T,
    ) -> Result<Graph<B, T>, GraphBuildError> {
        unimplemented!()
    }
}

pub struct Graph<B: Backend, T: ?Sized> {
    nodes: Vec<Box<dyn DynNode<B, T>>>,
}

impl<B: Backend, T: ?Sized> Graph<B, T> {
    fn run(
        &self,
        _factory: &mut Factory<B>,
        _families: &mut Families<B>,
        _aux: &T,
    ) -> Result<(), GraphRunError> {
        unimplemented!()
    }
}

enum GraphRunError {
    NodeConstruction(NodeId, NodeConstructionError),
    NodeExecution(NodeId, NodeExecutionError),
}

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
//     g.add(RenderOpaquePbrSkinned::new(color, depth, evsm_shadows));
//     g.add(RenderTransparentPbr::new(color, depth, evsm_shadows));
//     g.add(RenderTransparentPbrSkinned::new(color, depth, evsm_shadows));
// }
