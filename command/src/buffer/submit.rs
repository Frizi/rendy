use {
    super::{
        level::PrimaryLevel,
        state::{ExecutableState, InvalidState, PendingState},
        usage::{MultiShot, NoSimultaneousUse, OneShot, OutsideRenderPass, SimultaneousUse},
        CommandBuffer,
    },
    crate::family::FamilyId,
    rendy_core::hal,
};

/// Structure contains command buffer ready for submission.
#[derive(Debug)]
pub struct Submit<B: hal::Backend, S = NoSimultaneousUse, L = PrimaryLevel, P = OutsideRenderPass> {
    raw: std::ptr::NonNull<B::CommandBuffer>,
    family: FamilyId,
    simultaneous: S,
    level: L,
    pass_continue: P,
}

unsafe impl<B, S, L, P> Send for Submit<B, S, L, P>
where
    B: hal::Backend,
    B::CommandBuffer: Send + Sync,
    FamilyId: Send,
    S: Send,
    L: Send,
    P: Send,
{
}

unsafe impl<B, S, L, P> Sync for Submit<B, S, L, P>
where
    B: hal::Backend,
    B::CommandBuffer: Send + Sync,
    S: Sync,
    L: Sync,
    P: Sync,
{
}

/// Submittable object.
/// Values that implement this trait can be submitted to the queues
/// or executed as part of primary buffers (in case of `Submittable<B, SecondaryLevel>`).
pub unsafe trait Submittable<B: hal::Backend, L = PrimaryLevel, P = OutsideRenderPass> {
    /// Get family that this submittable is belong to.
    fn family(&self) -> FamilyId;

    /// Get raw command buffer.
    /// This function is intended for submitting command buffer into raw queue.
    ///
    /// # Safety
    ///
    /// This function returns unbound reference to the raw command buffer.
    /// The actual lifetime of the command buffer is tied to the original `CommandBuffer` wrapper.
    /// `CommandBuffer` must not destroy raw command buffer or give access to it before submitted command is complete so
    /// using this funcion to submit command buffer into queue must be valid.
    unsafe fn raw<'a>(self) -> &'a B::CommandBuffer;
}

unsafe impl<B, S, L, P> Submittable<B, L, P> for Submit<B, S, L, P>
where
    B: hal::Backend,
{
    fn family(&self) -> FamilyId {
        self.family
    }

    unsafe fn raw<'a>(self) -> &'a B::CommandBuffer {
        &*self.raw.as_ptr()
    }
}

unsafe impl<'a, B, L, P> Submittable<B, L, P> for &'a Submit<B, SimultaneousUse, L, P>
where
    B: hal::Backend,
{
    fn family(&self) -> FamilyId {
        self.family
    }

    unsafe fn raw<'b>(self) -> &'b B::CommandBuffer {
        &*self.raw.as_ptr()
    }
}

impl<B, C, P, L, R> CommandBuffer<B, C, ExecutableState<OneShot, P>, L, R>
where
    B: hal::Backend,
    P: Copy,
    L: Copy,
{
    /// Produce `Submit` object that can be used to populate submission.
    pub fn submit_once(
        self,
    ) -> (
        Submit<B, NoSimultaneousUse, L, P>,
        CommandBuffer<B, C, PendingState<InvalidState>, L, R>,
    ) {
        let pass_continue = self.state.1;
        let level = self.level;

        let buffer = unsafe { self.change_state(|_| PendingState(InvalidState)) };

        let submit = Submit {
            raw: buffer.raw,
            family: buffer.family,
            pass_continue,
            simultaneous: NoSimultaneousUse,
            level,
        };

        (submit, buffer)
    }
}

impl<B, C, S, L, P, R> CommandBuffer<B, C, ExecutableState<MultiShot<S>, P>, L, R>
where
    B: hal::Backend,
    P: Copy,
    S: Copy,
    L: Copy,
{
    /// Produce `Submit` object that can be used to populate submission.
    pub fn submit(
        self,
    ) -> (
        Submit<B, S, L, P>,
        CommandBuffer<B, C, PendingState<ExecutableState<MultiShot<S>, P>>, L, R>,
    ) {
        let MultiShot(simultaneous) = self.state.0;
        let pass_continue = self.state.1;
        let level = self.level;

        let buffer = unsafe { self.change_state(|state| PendingState(state)) };

        let submit = Submit {
            raw: buffer.raw,
            family: buffer.family,
            pass_continue,
            simultaneous,
            level,
        };

        (submit, buffer)
    }
}

/// Structure contains command buffer ready for submission that either allows simulateneous use or doesn't.
#[derive(Debug)]
pub enum EitherSubmit<'a, B: hal::Backend, L = PrimaryLevel, P = OutsideRenderPass> {
    Simultaneous(&'a Submit<B, SimultaneousUse, L, P>),
    NoSimultaneous(Submit<B, NoSimultaneousUse, L, P>),
}

impl<B: hal::Backend, L, P> From<Submit<B, NoSimultaneousUse, L, P>> for EitherSubmit<'_, B, L, P> {
    fn from(submit: Submit<B, NoSimultaneousUse, L, P>) -> Self {
        Self::NoSimultaneous(submit)
    }
}

impl<'a, B: hal::Backend, L, P> From<&'a Submit<B, SimultaneousUse, L, P>>
    for EitherSubmit<'a, B, L, P>
{
    fn from(submit: &'a Submit<B, SimultaneousUse, L, P>) -> Self {
        Self::Simultaneous(submit)
    }
}

unsafe impl<B: hal::Backend, L, P> Submittable<B, L, P> for EitherSubmit<'_, B, L, P> {
    fn family(&self) -> FamilyId {
        match self {
            Self::Simultaneous(submit) => submit.family(),
            Self::NoSimultaneous(submit) => submit.family(),
        }
    }

    unsafe fn raw<'a>(self) -> &'a B::CommandBuffer {
        match self {
            Self::Simultaneous(submit) => submit.raw(),
            Self::NoSimultaneous(submit) => submit.raw(),
        }
    }
}
