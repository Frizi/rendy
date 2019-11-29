use {
    crate::{
        command::{
            CommandPool, Encoder, Family, IndividualReset, Level, SimultaneousUse, Submit,
            Supports, Transfer,
        },
        factory::Factory,
        frame::cirque::CommandCirqueOverlap,
        new::{
            graph::NodeCtx,
            node::{
                gfx_acquire_barriers, gfx_release_barriers, ConstructResult, Node, NodeBuildError,
                NodeBuilder, NodeExecution, NodeImage, Parameter,
            },
            resources::{ImageId, ImageUsage},
            track::Track,
        },
    },
    rendy_core::hal::Backend,
};

/// Rendering node that copies one image onto another.
#[derive(Debug, Clone, Copy)]
pub struct CopyImageBuilder {
    src: Parameter<ImageId>,
    dst: Parameter<ImageId>,
}

impl<B: Backend> CopyImage<B> {
    /// Create `CopyImage` node builder.
    pub fn builder(src: Parameter<ImageId>, dst: Parameter<ImageId>) -> CopyImageBuilder {
        CopyImageBuilder { src, dst }
    }
}

impl<B: Backend, T: ?Sized> NodeBuilder<B, T> for CopyImageBuilder {
    type Node = CopyImage<B>;
    type Family = Transfer;

    fn build(
        self: Box<Self>,
        factory: &mut Factory<B>,
        family: &mut Family<B>,
        _aux: &T,
    ) -> Result<Self::Node, NodeBuildError> {
        let pool = factory
            .create_command_pool(family)?
            .with_capability::<Transfer>()
            .expect("Graph builder must provide family with Graphics capability");

        Ok(CopyImage {
            src: self.src,
            dst: self.dst,
            pool,
            cirque: CommandCirqueOverlap::new(),
            submit: Track::new(),
        })
    }
}

/// Rendering node that copies one image onto another.
// #[derive(Debug)]
// pub struct CopyImage<B: Backend> {
//     src: Parameter<ImageId>,
//     dst: Parameter<ImageId>,
//     pool: CommandPool<B, Transfer, IndividualReset>,
//     cirque: CommandCirqueOverlap<B, Transfer>,
//     submit: Track<(usize, usize), Submit<B, SimultaneousUse>>,
// }

// impl<B: Backend, T: ?Sized> Node<B, T> for CopyImage<B> {
//     type Outputs = ();

//     fn construct<'run>(
//         &'run mut self,
//         ctx: &mut impl NodeCtx<'run, B>,
//         _aux: &'run T,
//     ) -> ConstructResult<'run, Self, B, T> {
//         let src_id = *ctx.get_parameter(self.src)?;
//         let dst_id = *ctx.get_parameter(self.dst)?;

//         let src = ctx.use_image(src_id, ImageUsage::TransferRead)?;
//         let dst = ctx.use_image(dst_id, ImageUsage::TransferWrite)?;

//         let pool = &mut self.pool;
//         let submit = &mut self.submit;
//         let cirque = &mut self.cirque;

//         Ok((
//             (),
//             NodeExecution::submission(move |mut ctx| {
//                 let src_image = ctx.get_image(src);
//                 let dst_image = ctx.get_image(dst);

//                 let submit = submit.track(
//                     (src_image.image.instance(), dst_image.image.instance()),
//                     |_| unsafe {
//                         cirque.encode(ctx.frames(), pool, (), |recording| {
//                             encode_copy(&mut recording.encoder(), src_image, dst_image);
//                         })
//                     },
//                 );
                
//                 // TODO: submit node owned buffer
//                 // ctx.submit(Some(submit));
//                 Ok(())
//             }),
//         ))
//     }

//     unsafe fn dispose(mut self: Box<Self>, factory: &mut Factory<B>, _aux: &T) {
//         self.cirque.dispose(&mut self.pool);
//         self.pool.dispose(factory);
//     }
// }

// unsafe fn encode_copy<B, C, L>(
//     encoder: &mut Encoder<'_, B, C, L>,
//     src_image: &NodeImage<B>,
//     dst_image: &NodeImage<B>,
// ) where
//     B: Backend,
//     C: Supports<Transfer>,
//     L: Level,
// {
//     let (mut stages, mut barriers) = gfx_acquire_barriers(None, Some(src_image));
//     stages.start |= rendy_core::hal::pso::PipelineStage::TRANSFER;
//     stages.end |= rendy_core::hal::pso::PipelineStage::TRANSFER;
//     barriers.push(rendy_core::hal::memory::Barrier::Image {
//         states: (
//             rendy_core::hal::image::Access::empty(),
//             rendy_core::hal::image::Layout::Undefined,
//         )
//             ..(
//                 rendy_core::hal::image::Access::TRANSFER_WRITE,
//                 rendy_core::hal::image::Layout::TransferDstOptimal,
//             ),
//         families: None,
//         target: dst_image.image.raw(),
//         range: rendy_core::hal::image::SubresourceRange {
//             aspects: rendy_core::hal::format::Aspects::COLOR,
//             levels: 0..1,
//             layers: 0..1,
//         },
//     });
//     encoder.pipeline_barrier(
//         stages,
//         rendy_core::hal::memory::Dependencies::empty(),
//         barriers,
//     );
//     encoder.copy_image(
//         src_image.image.raw(),
//         src_image.layout,
//         dst_image.image.raw(),
//         rendy_core::hal::image::Layout::TransferDstOptimal,
//         Some(rendy_core::hal::command::ImageCopy {
//             src_subresource: rendy_core::hal::image::SubresourceLayers {
//                 aspects: src_image.range.aspects,
//                 level: 0,
//                 layers: src_image.range.layers.start..src_image.range.layers.start + 1,
//             },
//             src_offset: rendy_core::hal::image::Offset::ZERO,
//             dst_subresource: rendy_core::hal::image::SubresourceLayers {
//                 aspects: rendy_core::hal::format::Aspects::COLOR,
//                 level: 0,
//                 layers: 0..1,
//             },
//             dst_offset: rendy_core::hal::image::Offset::ZERO,
//             extent: rendy_core::hal::image::Extent {
//                 width: dst_image.image.kind().extent().width,
//                 height: dst_image.image.kind().extent().height,
//                 depth: 1,
//             },
//         }),
//     );
//     let (mut stages, mut barriers) = gfx_release_barriers(None, Some(src_image));
//     stages.start |= rendy_core::hal::pso::PipelineStage::TRANSFER;
//     stages.end |= rendy_core::hal::pso::PipelineStage::BOTTOM_OF_PIPE;
//     barriers.push(rendy_core::hal::memory::Barrier::Image {
//         states: (
//             rendy_core::hal::image::Access::TRANSFER_WRITE,
//             rendy_core::hal::image::Layout::TransferDstOptimal,
//         )
//             ..(
//                 rendy_core::hal::image::Access::empty(),
//                 rendy_core::hal::image::Layout::Present, // TODO: get dst layout
//             ),
//         families: None,
//         target: dst_image.image.raw(),
//         range: rendy_core::hal::image::SubresourceRange {
//             aspects: rendy_core::hal::format::Aspects::COLOR,
//             levels: 0..1,
//             layers: 0..1,
//         },
//     });
//     encoder.pipeline_barrier(
//         stages,
//         rendy_core::hal::memory::Dependencies::empty(),
//         barriers,
//     );
// }
