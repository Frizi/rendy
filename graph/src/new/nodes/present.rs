use {
    crate::{
        command::{Family, Graphics},
        factory::Factory,
        new::{
            graph::{FamilyType, NodeCtx},
            node::{ConstructResult, Node, NodeBuildError, NodeBuilder, NodeExecution, Parameter},
            resources::{ImageId, ImageInfo, ImageLoad, ImageUsage},
        },
        wsi::Target,
    },
    rendy_core::hal::{self, Backend},
};

pub trait TargetGetter<'a, B: Backend, T: ?Sized>: Send + Sync + 'static {
    type TargetMutRef: std::ops::DerefMut<Target = Target<B>> + 'a;
    fn target(&'a mut self, aux: &'a T) -> Self::TargetMutRef;
}

impl<'a, B: Backend, T: ?Sized + 'a, F, R> TargetGetter<'a, B, T> for F
where
    R: std::ops::DerefMut<Target = Target<B>> + 'a,
    F: FnMut(&'a T) -> R + Send + Sync + 'static,
{
    type TargetMutRef = R;
    fn target(&'a mut self, aux: &'a T) -> Self::TargetMutRef {
        self(aux)
    }
}

impl<'a, B: Backend, T: ?Sized> TargetGetter<'a, B, T> for Target<B> {
    type TargetMutRef = &'a mut Self;
    fn target(&'a mut self, _: &'a T) -> Self::TargetMutRef {
        self
    }
}

/// Node for presenting rendered image onto a surface
pub struct Present<F> {
    get_target: F,
    clear: Option<hal::command::ClearColor>,
}

impl<F> std::fmt::Debug for Present<F> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("Present").finish()
    }
}

impl<F> Present<F> {
    pub fn new(get_target: F, clear: Option<hal::command::ClearColor>) -> Self {
        Self { get_target, clear }
    }
}

impl<B: Backend, T: ?Sized, F> NodeBuilder<B, T> for Present<F>
where
    F: for<'a> TargetGetter<'a, B, T>,
{
    type Node = PresentNode<F>;
    type Family = Graphics;

    fn build(
        self: Box<Self>,
        factory: &mut Factory<B>,
        _family: &mut Family<B>,
        _aux: &T,
    ) -> Result<Self::Node, NodeBuildError> {
        Ok(PresentNode {
            // free_acquire: factory.create_semaphore().unwrap(),
            // per_image: Vec::new(),
            get_target: self.get_target,
            clear: self.clear,
        })
    }
}

/// Node for presenting rendered image onto a surface
pub struct PresentNode<F> {
    get_target: F,
    clear: Option<hal::command::ClearColor>,
}

impl<F> std::fmt::Debug for PresentNode<F> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("PresentNode").finish()
    }
}

impl<B: Backend, T: ?Sized, F> Node<B, T> for PresentNode<F>
where
    F: for<'a> TargetGetter<'a, B, T>,
{
    type Outputs = Parameter<ImageId>;
    fn construct<'run>(
        &'run mut self,
        ctx: &mut impl NodeCtx<'run, B>,
        aux: &'run T,
    ) -> ConstructResult<'run, Self, B, T> {
        let mut target = self.get_target.target(aux);

        // let num_images = target.backbuffer().len();
        // while self.per_image.len() < num_images {
        //     self.per_image.push(PresentPerImage {
        //         // acquire: ctx.factory().create_semaphore().unwrap(),
        //         // release: ctx.factory().create_semaphore().unwrap(),
        //         // view: Track::new(),
        //     })
        // }

        // for (index, image) in target.backbuffer().iter().enumerate() {
        //     self.per_image[index].view.track(image.instance(), |_| {
        //         let format = image.format();
        //         let view = unsafe {
        //             ctx.factory().device().create_image_view(
        //                 image.raw(),
        //                 hal::image::ViewKind::D2,
        //                 format,
        //                 hal::format::Swizzle::NO,
        //                 hal::image::SubresourceRange {
        //                     aspects: format.surface_desc().aspects,
        //                     levels: 0..1,
        //                     layers: 0..1,
        //                 },
        //             )
        //         }
        //         .unwrap();
        //         view
        //     });
        // }

        // @Robustness
        // The image providing code should actually be a execution time thing.
        // Only then we actually know how the image is going to be used, and also
        // the semaphore thing wouldn't be so awkward.

        let extent = target.extent();
        let semaphore = ctx.create_semaphore();
        let next = unsafe { target.next_image(&semaphore) };

        let (output, index, wait) = match &next {
            Ok(next) => {
                log::trace!("Presentable image acquired: {:#?}", next);
                let index = next[0];

                let image = next.image(0);
                let image_info = ImageInfo {
                    kind: hal::image::Kind::D2(extent.width, extent.height, 1, 1),
                    levels: 1,
                    format: image.format(),
                    load: match self.clear {
                        None => ImageLoad::DontCare,
                        Some(color) => ImageLoad::Clear(hal::command::ClearValue { color }),
                    },
                };

                let output = ctx.provide_image(image_info, image, Some(semaphore));
                ctx.use_image(output, ImageUsage::Present)?;
                let wait = ctx.wait_semaphore(output, ImageUsage::Present.stage());
                (output, index, wait)
            }
            Err(err) => {
                log::debug!("Swapchain acquisition error: {:#?}", err);
                ctx.recycle_semaphores(Some(semaphore));
                // just create a dummy image and do nothing
                let image_info = ImageInfo {
                    kind: hal::image::Kind::D2(extent.width, extent.height, 1, 1),
                    levels: 1,
                    format: hal::format::Format::Rgba8Unorm,
                    load: ImageLoad::DontCare,
                };
                let output = ctx.create_image(image_info);
                return Ok((output, NodeExecution::None));
            }
        };

        drop(next);
        drop(target);

        Ok((
            output,
            NodeExecution::output_post_submit(move |ctx| {
                use hal::queue::CommandQueue;
                let target = self.get_target.target(aux);
                let swapchain = target.swapchain();

                let semaphore = &ctx.semaphores[wait];
                let queue_id = ctx.queue_id(FamilyType::Graphics);
                let queue = ctx.families.queue_mut(queue_id).raw();

                if let Err(err) =
                    unsafe { queue.present(Some((swapchain, index)), Some(semaphore)) }
                {
                    // TODO: maybe propagate this error
                    log::debug!("Swapchain presentation error: {:#?}", err);
                }

                Ok(())
            }),
        ))
    }

    // unsafe fn dispose(self: Box<Self>, _factory: &mut Factory<B>, _aux: &T) {}
}
