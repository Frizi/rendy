use {
    super::*,
    crate::command::{
        BeginInfo, Capability, CommandBuffer, CommandPool, ExecutableState, IndividualReset,
        InitialState, Level, MultiShot, NoSimultaneousUse, OutsideRenderPass, PendingState,
        PrimaryLevel, RecordingState, RenderPassRelation, SimultaneousUse, Submit,
    },
};

///
pub type CommandCirque<B, C, P = OutsideRenderPass, L = PrimaryLevel> = Cirque<
    CommandBuffer<B, C, ExecutableState<MultiShot, P>, L, IndividualReset>,
    CommandBuffer<B, C, InitialState, L, IndividualReset>,
    CommandBuffer<B, C, PendingState<ExecutableState<MultiShot, P>>, L, IndividualReset>,
>;

///
pub type CommandCirqueRef<'a, B, C, P = OutsideRenderPass, L = PrimaryLevel> = CirqueRef<
    'a,
    CommandBuffer<B, C, ExecutableState<MultiShot, P>, L, IndividualReset>,
    CommandBuffer<B, C, InitialState, L, IndividualReset>,
    CommandBuffer<B, C, PendingState<ExecutableState<MultiShot, P>>, L, IndividualReset>,
>;

///
pub type CommandInitialRef<'a, B, C, P = OutsideRenderPass, L = PrimaryLevel> = InitialRef<
    'a,
    CommandBuffer<B, C, ExecutableState<MultiShot, P>, L, IndividualReset>,
    CommandBuffer<B, C, InitialState, L, IndividualReset>,
    CommandBuffer<B, C, PendingState<ExecutableState<MultiShot, P>>, L, IndividualReset>,
>;

///
pub type CommandReadyRef<'a, B, C, P = OutsideRenderPass, L = PrimaryLevel> = ReadyRef<
    'a,
    CommandBuffer<B, C, ExecutableState<MultiShot, P>, L, IndividualReset>,
    CommandBuffer<B, C, InitialState, L, IndividualReset>,
    CommandBuffer<B, C, PendingState<ExecutableState<MultiShot, P>>, L, IndividualReset>,
>;

impl<B, C, P, L> CommandCirque<B, C, P, L>
where
    B: rendy_core::hal::Backend,
    L: Level,
    C: Capability,
    P: RenderPassRelation<L>,
{
    /// Encode and submit.
    /// Safety: Returned `Submit` instance can only be submitted for current frame.
    pub unsafe fn encode<'a>(
        &'a mut self,
        frames: &Frames<B>,
        pool: &mut CommandPool<B, C, IndividualReset>,
        encode: impl FnOnce(CommandCirqueRef<'a, B, C, P, L>) -> CommandReadyRef<'a, B, C, P, L>,
    ) -> Submit<B, NoSimultaneousUse, L, P> {
        let cr = self.get(
            frames,
            || pool.allocate_buffers(1).pop().unwrap(),
            |pending| pending.mark_complete(),
        );

        let ready = encode(cr);

        let mut slot = None;

        ready.finish(|executable| {
            let (submit, pending) = executable.submit();
            slot = Some(submit);
            pending
        });

        slot.unwrap()
    }
}

type ExecOverlap<P> = ExecutableState<MultiShot<SimultaneousUse>, P>;
type RecOverlap<P> = RecordingState<MultiShot<SimultaneousUse>, P>;

///
type CommandCirqueOverlapInner<B, C, P = OutsideRenderPass, L = PrimaryLevel> = Cirque<
    CommandBuffer<B, C, ExecOverlap<P>, L, IndividualReset>,
    CommandBuffer<B, C, InitialState, L, IndividualReset>,
    CommandBuffer<B, C, PendingState<ExecOverlap<P>>, L, IndividualReset>,
>;

///
pub type CommandCirqueOverlapRef<'a, B, C, P = OutsideRenderPass, L = PrimaryLevel> = CirqueRef<
    'a,
    CommandBuffer<B, C, ExecOverlap<P>, L, IndividualReset>,
    CommandBuffer<B, C, InitialState, L, IndividualReset>,
    CommandBuffer<B, C, PendingState<ExecOverlap<P>>, L, IndividualReset>,
>;

///
pub type CommandReadyOverlapRef<'a, B, C, P = OutsideRenderPass, L = PrimaryLevel> = ReadyRef<
    'a,
    CommandBuffer<B, C, ExecOverlap<P>, L, IndividualReset>,
    CommandBuffer<B, C, InitialState, L, IndividualReset>,
    CommandBuffer<B, C, PendingState<ExecOverlap<P>>, L, IndividualReset>,
>;

///
type CommandFinishOverlapRef<B, C, P = OutsideRenderPass, L = PrimaryLevel> = DetachedFinishRef<
    CommandBuffer<B, C, ExecOverlap<P>, L, IndividualReset>,
    CommandBuffer<B, C, InitialState, L, IndividualReset>,
    CommandBuffer<B, C, PendingState<ExecOverlap<P>>, L, IndividualReset>,
>;

/// Specialized cirque for multishot simulateneous use command buffers.
/// Allows for reusing the encoded command buffer for multiple frames
/// and reencoding as needed.
#[derive(Debug, derivative::Derivative)]
#[derivative(Default(bound = ""))]
pub struct CommandCirqueOverlap<B, C, P = OutsideRenderPass, L = PrimaryLevel>
where
    B: gfx_hal::Backend,
    L: Level,
    C: Capability,
    P: RenderPassRelation<L>,
{
    cirque: CommandCirqueOverlapInner<B, C, P, L>,
    detached: Option<CommandFinishOverlapRef<B, C, P, L>>,
}

impl<B, C, P, L> CommandCirqueOverlap<B, C, P, L>
where
    B: gfx_hal::Backend,
    L: Level,
    C: Capability,
    P: RenderPassRelation<L>,
{
    /// Create new `CommandCirqueOverlap` instance.
    pub fn new() -> Self {
        Default::default()
    }

    /// Encode and submit.
    /// Safety: the `Submit` instance returned from previous invocation of `encode` (if any) must not be submitted again.
    pub unsafe fn encode<'a, 'b>(
        &'a mut self,
        frames: &Frames<B>,
        pool: &mut CommandPool<B, C, IndividualReset>,
        info: impl BeginInfo<'b, B, L, PassRelation = P>,
        encode: impl FnOnce(&mut CommandBuffer<B, C, RecOverlap<P>, L, IndividualReset>),
    ) -> Submit<B, SimultaneousUse, L, P> {
        if let Some(returned) = self.detached.take() {
            returned.attach(&mut self.cirque, frames);
        }

        let cr = self.cirque.get(
            frames,
            || pool.allocate_buffers(1).pop().unwrap(),
            |pending| pending.mark_complete(),
        );

        let ready = cr.or_reset(|cbuf| cbuf.reset()).init(move |cbuf| {
            let mut recording = cbuf.begin(MultiShot(SimultaneousUse), info);
            encode(&mut recording);
            recording.finish()
        });

        let mut slot = None;
        let detached = ready.detach().finish(|executable| {
            let (submit, pending) = executable.submit();
            slot = Some(submit);
            pending
        });

        self.detached.replace(detached);

        slot.unwrap()
    }

    /// Dispose of the `SimulataneousCommandCirque`.
    /// Safety: Must use the pool used in `encode` calls.
    pub unsafe fn dispose(self, pool: &mut CommandPool<B, C, IndividualReset>) {
        if let Some(detached) = self.detached {
            detached.dispose(|pending| {
                let executable = pending.mark_complete();
                pool.free_buffers(Some(executable))
            });
        }

        self.cirque.dispose(|buffer| {
            buffer.either_with(
                &mut *pool,
                |pool, executable| pool.free_buffers(Some(executable)),
                |pool, pending| {
                    let executable = pending.mark_complete();
                    pool.free_buffers(Some(executable))
                },
            );
        });
    }
}
