use {
    crate::{
        command::{
            CommandPool, Encoder, Family, IndividualReset, Level, SimultaneousUse, Submit,
            Supports, Transfer,
        },
        factory::Factory,
        frame::cirque::CommandCirqueOverlap,
        new::{
            graph::NodeContext,
            node::{
                gfx_acquire_barriers, gfx_release_barriers, Node, NodeBuildError, NodeBuilder,
                NodeConstructionError, NodeExecution, NodeImage, OutputList, Parameter,
            },
            resources::{ImageId, NodeImageAccess},
            track::Track,
        },
    },
    gfx_hal::Backend,
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
#[derive(Debug)]
pub struct CopyImage<B: Backend> {
    src: Parameter<ImageId>,
    dst: Parameter<ImageId>,
    pool: CommandPool<B, Transfer, IndividualReset>,
    cirque: CommandCirqueOverlap<B, Transfer>,
    submit: Track<(usize, usize), Submit<B, SimultaneousUse>>,
}

impl<B: Backend, T: ?Sized> Node<B, T> for CopyImage<B> {
    type Outputs = ();

    fn construct(
        &mut self,
        ctx: &mut NodeContext<'_, '_, B, T>,
        _aux: &T,
    ) -> Result<(<Self::Outputs as OutputList>::Data, NodeExecution<'_, B, T>), NodeConstructionError>
    {
        let src_id = *ctx.get_parameter(self.src)?;
        let dst_id = *ctx.get_parameter(self.dst)?;

        ctx.use_image(src_id, NodeImageAccess::TRANSFER_READ);
        ctx.use_image(dst_id, NodeImageAccess::TRANSFER_WRITE);

        let pool = &mut self.pool;
        let submit = &mut self.submit;
        let cirque = &mut self.cirque;

        Ok((
            (),
            NodeExecution::general(move |mut ctx, _| {
                let src_image = ctx.image(src_id)?;
                let dst_image = ctx.image(dst_id)?;

                let submit = submit.track(
                    (src_image.image.instance(), dst_image.image.instance()),
                    |_| unsafe {
                        cirque.encode(ctx.frames(), pool, (), |recording| {
                            encode_copy(&mut recording.encoder(), src_image, dst_image);
                        })
                    },
                );

                ctx.submit(Some(submit));
                Ok(())
            }),
        ))
    }

    fn dispose(mut self: Box<Self>, factory: &mut Factory<B>, _aux: &T) {
        unsafe {
            self.cirque.dispose(&mut self.pool);
            self.pool.dispose(factory);
        }
    }
}

unsafe fn encode_copy<B, C, L>(
    encoder: &mut Encoder<'_, B, C, L>,
    src_image: &NodeImage<B>,
    dst_image: &NodeImage<B>,
) where
    B: Backend,
    C: Supports<Transfer>,
    L: Level,
{
    let (mut stages, mut barriers) = gfx_acquire_barriers(None, Some(src_image));
    stages.start |= gfx_hal::pso::PipelineStage::TRANSFER;
    stages.end |= gfx_hal::pso::PipelineStage::TRANSFER;
    barriers.push(gfx_hal::memory::Barrier::Image {
        states: (
            gfx_hal::image::Access::empty(),
            gfx_hal::image::Layout::Undefined,
        )
            ..(
                gfx_hal::image::Access::TRANSFER_WRITE,
                gfx_hal::image::Layout::TransferDstOptimal,
            ),
        families: None,
        target: dst_image.image.raw(),
        range: gfx_hal::image::SubresourceRange {
            aspects: gfx_hal::format::Aspects::COLOR,
            levels: 0..1,
            layers: 0..1,
        },
    });
    encoder.pipeline_barrier(stages, gfx_hal::memory::Dependencies::empty(), barriers);
    encoder.copy_image(
        src_image.image.raw(),
        src_image.layout,
        dst_image.image.raw(),
        gfx_hal::image::Layout::TransferDstOptimal,
        Some(gfx_hal::command::ImageCopy {
            src_subresource: gfx_hal::image::SubresourceLayers {
                aspects: src_image.range.aspects,
                level: 0,
                layers: src_image.range.layers.start..src_image.range.layers.start + 1,
            },
            src_offset: gfx_hal::image::Offset::ZERO,
            dst_subresource: gfx_hal::image::SubresourceLayers {
                aspects: gfx_hal::format::Aspects::COLOR,
                level: 0,
                layers: 0..1,
            },
            dst_offset: gfx_hal::image::Offset::ZERO,
            extent: gfx_hal::image::Extent {
                width: dst_image.image.kind().extent().width,
                height: dst_image.image.kind().extent().height,
                depth: 1,
            },
        }),
    );
    let (mut stages, mut barriers) = gfx_release_barriers(None, Some(src_image));
    stages.start |= gfx_hal::pso::PipelineStage::TRANSFER;
    stages.end |= gfx_hal::pso::PipelineStage::BOTTOM_OF_PIPE;
    barriers.push(gfx_hal::memory::Barrier::Image {
        states: (
            gfx_hal::image::Access::TRANSFER_WRITE,
            gfx_hal::image::Layout::TransferDstOptimal,
        )
            ..(
                gfx_hal::image::Access::empty(),
                gfx_hal::image::Layout::Present,
            ),
        families: None,
        target: dst_image.image.raw(),
        range: gfx_hal::image::SubresourceRange {
            aspects: gfx_hal::format::Aspects::COLOR,
            levels: 0..1,
            layers: 0..1,
        },
    });
    encoder.pipeline_barrier(stages, gfx_hal::memory::Dependencies::empty(), barriers);
}
