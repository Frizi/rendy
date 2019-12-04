pub use rendy_core::hal::image::*;
use std::sync::atomic::{AtomicUsize, Ordering};

use {
    crate::core::{
        device_owned,
        hal::{
            device::Device as _,
            pso::{ComputePipelineDesc, CreationError, GraphicsPipelineDesc},
            Backend,
        },
        Device, DeviceId,
    },
    relevant::Relevant,
};

static GRAPHICS_INSTANCE_COUNTER: AtomicUsize = AtomicUsize::new(0);
static COMPUTE_INSTANCE_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Generic graphics pipeline resource wrapper.
///
/// # Parameters
///
/// `B` - raw image type.
#[derive(Debug)]
pub struct GraphicsPipeline<B: Backend> {
    device: DeviceId,
    instance: usize,
    raw: B::GraphicsPipeline,
    relevant: Relevant,
}

device_owned!(GraphicsPipeline<B>);

impl<B: Backend> GraphicsPipeline<B> {
    /// Create a new graphics pipeline.
    pub fn create(
        device: &Device<B>,
        desc: &GraphicsPipelineDesc<B>,
        cache: Option<&B::PipelineCache>,
    ) -> Result<Self, CreationError> {
        let raw = unsafe { device.create_graphics_pipeline(desc, cache)? };
        Ok(Self {
            device: device.id(),
            instance: GRAPHICS_INSTANCE_COUNTER.fetch_add(1, Ordering::Relaxed),
            raw,
            relevant: Relevant,
        })
    }
    /// Retreive unique instance identifier
    pub fn instance(&self) -> usize {
        self.instance
    }

    /// Access raw pipeline backend object.
    pub fn raw(&self) -> &B::GraphicsPipeline {
        &self.raw
    }
    
    /// Dispose of a graphics pipeline.
    /// Safety:
    /// - graphics pipeline must not be used in any unfinished submit.
    pub unsafe fn dispose(self, device: &Device<B>) {
        device.destroy_graphics_pipeline(self.raw);
        self.relevant.dispose();
    }
}

/// Generic compute pipeline resource wrapper.
///
/// # Parameters
///
/// `B` - raw image type.
#[derive(Debug)]
pub struct ComputePipeline<B: Backend> {
    device: DeviceId,
    instance: usize,
    raw: B::ComputePipeline,
    relevant: Relevant,
}

device_owned!(ComputePipeline<B>);

impl<B: Backend> ComputePipeline<B> {
    /// Create a new compute pipeline.
    pub fn create(
        device: &Device<B>,
        desc: &ComputePipelineDesc<B>,
        cache: Option<&B::PipelineCache>,
    ) -> Result<Self, CreationError> {
        let raw = unsafe { device.create_compute_pipeline(desc, cache)? };
        Ok(Self {
            device: device.id(),
            instance: COMPUTE_INSTANCE_COUNTER.fetch_add(1, Ordering::Relaxed),
            raw,
            relevant: Relevant,
        })
    }
    /// Retreive unique instance identifier
    pub fn instance(&self) -> usize {
        self.instance
    }

    /// Access raw pipeline backend object.
    pub fn raw(&self) -> &B::ComputePipeline {
        &self.raw
    }

    /// Dispose of a compute pipeline.
    /// Safety:
    /// - compute pipeline must not be used in any unfinished submit.
    pub unsafe fn dispose(self, device: &Device<B>) {
        device.destroy_compute_pipeline(self.raw);
        self.relevant.dispose();
    }
}
