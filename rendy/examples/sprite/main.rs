//!
//! A simple sprite example.
//! This examples shows how to render a sprite on a white background.
//!

use rendy::{
    command::{
        CommandPool, Families, Family, Graphics, IndividualReset, QueueId, QueueType,
        RenderPassContinue, RenderPassEncoder, SecondaryLevel, SimultaneousUse, Submit,
    },
    factory::{
        BasicDevicesConfigure, BasicHeapsConfigure, Config, Factory, GraphOptimizedQueues,
        ImageState,
    },
    frame::cirque::CommandCirqueOverlap,
    graph::{
        new::{
            ConstructResult, ExecPassContext, FamilyType, Graph, GraphBuilder, ImageId, Node,
            NodeBuildError, NodeBuilder, NodeCtx, NodeExecution, Parameter, Present, SubpassId,
            Track,
        },
        present::PresentNode,
    },
    hal::{self, device::Device as _, Backend},
    init::winit::{
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::WindowBuilder,
    },
    init::AnyWindowedRendy,
    memory::Dynamic,
    mesh::PosTex,
    resource::{
        Buffer, BufferInfo, DescriptorSet, DescriptorSetLayout, Escape, GraphicsPipeline, Handle,
    },
    shader::{ShaderKind, SourceLanguage, SourceShaderInfo, SpirvShader},
    texture::{image::ImageTextureConfig, Texture},
};

#[cfg(feature = "spirv-reflection")]
use rendy::shader::SpirvReflection;

#[cfg(not(feature = "spirv-reflection"))]
use rendy::mesh::AsVertex;

use std::{fs::File, io::BufReader};

lazy_static::lazy_static! {
    static ref VERTEX: SpirvShader = SourceShaderInfo::new(
        include_str!("shader.vert"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/examples/sprite/shader.vert").into(),
        ShaderKind::Vertex,
        SourceLanguage::GLSL,
        "main",
    ).precompile().unwrap();

    static ref FRAGMENT: SpirvShader = SourceShaderInfo::new(
        include_str!("shader.frag"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/examples/sprite/shader.frag").into(),
        ShaderKind::Fragment,
        SourceLanguage::GLSL,
        "main",
    ).precompile().unwrap();

    static ref SHADERS: rendy::shader::ShaderSetBuilder = rendy::shader::ShaderSetBuilder::default()
        .with_vertex(&*VERTEX).unwrap()
        .with_fragment(&*FRAGMENT).unwrap();
}

#[cfg(feature = "spirv-reflection")]
lazy_static::lazy_static! {
    static ref SHADER_REFLECTION: SpirvReflection = SHADERS.reflect().unwrap();
}

#[derive(Debug)]
struct SpriteGraphicsPipeline {
    color: Parameter<ImageId>,
}

impl SpriteGraphicsPipeline {
    fn new(color: Parameter<ImageId>) -> Self {
        Self { color }
    }
}

impl<B: Backend, T: ?Sized> NodeBuilder<B, T> for SpriteGraphicsPipeline {
    type Node = SpriteGraphicsPipelineImpl<B>;
    type Family = Graphics;

    fn build(
        self: Box<Self>,
        factory: &mut Factory<B>,
        family: &mut Family<B>,
        _: &T,
    ) -> Result<Self::Node, NodeBuildError> {
        #[cfg(feature = "spirv-reflection")]
        let layout = SHADER_REFLECTION.layout().unwrap();
        #[cfg(not(feature = "spirv-reflection"))]
        let layout = Layout {
            sets: vec![SetLayout {
                bindings: vec![
                    hal::pso::DescriptorSetLayoutBinding {
                        binding: 0,
                        ty: hal::pso::DescriptorType::SampledImage,
                        count: 1,
                        stage_flags: hal::pso::ShaderStageFlags::FRAGMENT,
                        immutable_samplers: false,
                    },
                    hal::pso::DescriptorSetLayoutBinding {
                        binding: 1,
                        ty: hal::pso::DescriptorType::Sampler,
                        count: 1,
                        stage_flags: hal::pso::ShaderStageFlags::FRAGMENT,
                        immutable_samplers: false,
                    },
                ],
            }],
            push_constants: Vec::new(),
        };

        let set_layouts = layout
            .sets
            .into_iter()
            .map(|set| {
                factory
                    .create_descriptor_set_layout(set.bindings)
                    .map(Handle::from)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let pipeline_layout = unsafe {
            factory
                .device()
                .create_pipeline_layout(set_layouts.iter().map(|l| l.raw()), layout.push_constants)
                .unwrap()
        };

        // This is how we can load an image and create a new texture.
        let image_reader = BufReader::new(
            File::open(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/examples/sprite/logo.png"
            ))
            .map_err(|e| {
                log::error!("Unable to open {}: {:?}", "/examples/sprite/logo.png", e);
                hal::pso::CreationError::Other
            })?,
        );

        let texture_builder = rendy::texture::image::load_from_image(
            image_reader,
            ImageTextureConfig {
                generate_mips: true,
                ..Default::default()
            },
        )
        .map_err(|e| {
            log::error!("Unable to load image: {:?}", e);
            hal::pso::CreationError::Other
        })?;

        let texture = texture_builder
            .build(
                ImageState {
                    queue: QueueId {
                        family: family.id(),
                        index: 0,
                    },
                    stage: hal::pso::PipelineStage::FRAGMENT_SHADER,
                    access: hal::image::Access::SHADER_READ,
                    layout: hal::image::Layout::ShaderReadOnlyOptimal,
                },
                factory,
            )
            .unwrap();

        let descriptor_set = factory
            .create_descriptor_set(set_layouts[0].clone())
            .unwrap();

        unsafe {
            factory.device().write_descriptor_sets(vec![
                hal::pso::DescriptorSetWrite {
                    set: descriptor_set.raw(),
                    binding: 0,
                    array_offset: 0,
                    descriptors: vec![hal::pso::Descriptor::Image(
                        texture.view().raw(),
                        hal::image::Layout::ShaderReadOnlyOptimal,
                    )],
                },
                hal::pso::DescriptorSetWrite {
                    set: descriptor_set.raw(),
                    binding: 1,
                    array_offset: 0,
                    descriptors: vec![hal::pso::Descriptor::Sampler(texture.sampler().raw())],
                },
            ]);
        }

        let mut pool = factory
            .create_command_pool(family)
            .map_err(NodeBuildError::OutOfMemory)?;

        let shader_set = SHADERS.build(factory, Default::default()).unwrap();

        #[cfg(feature = "spirv-reflection")]
        let vbuf_size = SHADER_REFLECTION.attributes_range(..).unwrap().stride as u64 * 6;

        #[cfg(not(feature = "spirv-reflection"))]
        let vbuf_size = PosTex::vertex().stride as u64 * 6;

        let mut vbuf = factory
            .create_buffer(
                BufferInfo {
                    size: vbuf_size,
                    usage: hal::buffer::Usage::VERTEX,
                },
                Dynamic,
            )
            .unwrap();

        unsafe {
            // Fresh buffer.
            factory
                .upload_visible_buffer(
                    &mut vbuf,
                    0,
                    &[
                        PosTex {
                            position: [-0.5, 0.33, 0.0].into(),
                            tex_coord: [0.0, 1.0].into(),
                        },
                        PosTex {
                            position: [0.5, 0.33, 0.0].into(),
                            tex_coord: [1.0, 1.0].into(),
                        },
                        PosTex {
                            position: [0.5, -0.33, 0.0].into(),
                            tex_coord: [1.0, 0.0].into(),
                        },
                        PosTex {
                            position: [-0.5, 0.33, 0.0].into(),
                            tex_coord: [0.0, 1.0].into(),
                        },
                        PosTex {
                            position: [0.5, -0.33, 0.0].into(),
                            tex_coord: [1.0, 0.0].into(),
                        },
                        PosTex {
                            position: [-0.5, -0.33, 0.0].into(),
                            tex_coord: [0.0, 0.0].into(),
                        },
                    ],
                )
                .unwrap();
        }

        Ok(SpriteGraphicsPipelineImpl {
            color: self.color,
            descriptor_set,
            set_layouts,
            pipeline_layout,
            pool,
            submit_data: Track::new(),
            shader_set,
            cirque: CommandCirqueOverlap::new(),
            texture,
            vbuf,
        })
    }
}

#[derive(Debug)]
struct SpriteGraphicsPipelineImpl<B: Backend> {
    color: Parameter<ImageId>,
    descriptor_set: Escape<DescriptorSet<B>>,
    set_layouts: Vec<Handle<DescriptorSetLayout<B>>>,
    pipeline_layout: B::PipelineLayout,
    pool: CommandPool<B, QueueType, IndividualReset>,
    submit_data: Track<
        SubpassId,
        (
            Escape<GraphicsPipeline<B>>,
            Submit<B, SimultaneousUse, SecondaryLevel, RenderPassContinue>,
        ),
    >,
    shader_set: rendy_shader::ShaderSet<B>,
    cirque: CommandCirqueOverlap<B, QueueType, RenderPassContinue, SecondaryLevel>,
    texture: Texture<B>,
    vbuf: Escape<Buffer<B>>,
}

impl<B: Backend, T: ?Sized> Node<B, T> for SpriteGraphicsPipelineImpl<B> {
    type Outputs = ();

    fn construct<'run>(
        &'run mut self,
        ctx: &mut impl NodeCtx<'run, B>,
        _aux: &'run T,
    ) -> ConstructResult<'run, Self, B, T> {
        let color = *ctx.get_parameter(self.color)?;
        ctx.use_color(0, color)?;

        Ok((
            (),
            NodeExecution::<B>::pass(move |ctx| {
                let shader_set = &self.shader_set;
                let layout = &self.pipeline_layout;
                let descriptor_set = self.descriptor_set.raw();
                let vbuf = self.vbuf.raw();
                let pool = &mut self.pool;
                let cirque = &mut self.cirque;

                let (_, ref submit) = self.submit_data.track(ctx.subpass_id(), |_| {
                    let shaders = shader_set.raw().unwrap();

                    #[cfg(feature = "spirv-reflection")]
                    let (elemets, stride, rate) = SHADER_REFLECTION
                        .attributes_range(..)
                        .unwrap()
                        .gfx_vertex_input_desc(hal::pso::VertexInputRate::Vertex);

                    #[cfg(not(feature = "spirv-reflection"))]
                    let (elemets, stride, rate) =
                        PosTex::vertex().gfx_vertex_input_desc(hal::pso::VertexInputRate::Vertex);

                    let mut vertex_buffers = Vec::new();
                    let mut attributes = Vec::new();
                    push_vertex_desc(&elemets, stride, rate, &mut vertex_buffers, &mut attributes);

                    let rect = ctx.viewport_rect();

                    let subpass = ctx.subpass();
                    let pso: Escape<GraphicsPipeline<B>> = ctx
                        .factory()
                        .create_graphics_pipeline(&rendy_core::hal::pso::GraphicsPipelineDesc {
                            shaders,
                            rasterizer: rendy_core::hal::pso::Rasterizer::FILL,
                            vertex_buffers,
                            attributes,
                            input_assembler: rendy_core::hal::pso::InputAssemblerDesc {
                                primitive: rendy_core::hal::pso::Primitive::TriangleList,
                                with_adjacency: false,
                                restart_index: None,
                            },
                            blender: rendy_core::hal::pso::BlendDesc {
                                logic_op: None,
                                targets: vec![rendy_core::hal::pso::ColorBlendDesc {
                                    mask: rendy_core::hal::pso::ColorMask::ALL,
                                    blend: Some(rendy_core::hal::pso::BlendState::ALPHA),
                                }],
                            },
                            depth_stencil: rendy_core::hal::pso::DepthStencilDesc::default(),
                            multisampling: None,
                            baked_states: rendy_core::hal::pso::BakedStates {
                                viewport: Some(rendy_core::hal::pso::Viewport {
                                    rect,
                                    depth: 0.0..1.0,
                                }),
                                scissor: Some(rect),
                                blend_color: None,
                                depth_bounds: None,
                            },
                            layout,
                            subpass,
                            flags: rendy_core::hal::pso::PipelineCreationFlags::empty(),
                            parent: rendy_core::hal::pso::BasePipeline::None,
                        })
                        .unwrap();

                    let frames = ctx.frames();

                    let submit = unsafe {
                        cirque.encode(frames, pool, subpass, |recording| {
                            let mut encoder = recording.render_pass_encoder();
                            encoder.bind_graphics_descriptor_sets(
                                layout,
                                0,
                                std::iter::once(descriptor_set),
                                std::iter::empty::<u32>(),
                            );
                            encoder.bind_vertex_buffers(0, Some((vbuf, 0)));
                            encoder.draw(0..6, 0..1);
                        })
                    };

                    (pso, submit)
                });

                Ok(submit.into())
            }),
        ))
    }

    unsafe fn dispose(mut self: Box<Self>, factory: &mut Factory<B>, _aux: &T) {
        // TODO: need to put resources into dispose
        self.cirque.dispose(&mut self.pool);
        factory.destroy_command_pool(self.pool);

        if let Some((pso, submit)) = self.submit_data.into_inner() {
            drop(submit);
            factory.destroy_relevant_graphics_pipeline(Escape::unescape(pso));
        }
        factory
            .device()
            .destroy_pipeline_layout(self.pipeline_layout);
        drop(self.set_layouts);
    }
}

fn run<B: Backend>(
    event_loop: EventLoop<()>,
    mut factory: Factory<B>,
    mut families: Families<B>,
    graph: Graph<B, ()>,
) {
    let started = std::time::Instant::now();

    std::thread::spawn(move || {
        while started.elapsed() < std::time::Duration::new(30, 0) {
            std::thread::sleep(std::time::Duration::new(1, 0));
        }

        std::process::abort();
    });

    let mut frame = 0u64;
    let mut elapsed = started.elapsed();
    let mut graph = Some(graph);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                _ => {}
            },
            Event::EventsCleared => {
                factory.maintain(&mut families);
                if let Some(ref mut graph) = graph {
                    graph.run(&mut factory, &mut families, &()).unwrap();
                    frame += 1;
                }

                elapsed = started.elapsed();
                if elapsed >= std::time::Duration::new(5, 0) {
                    *control_flow = ControlFlow::Exit
                }
            }
            _ => {}
        }

        if *control_flow == ControlFlow::Exit && graph.is_some() {
            let elapsed_ns = elapsed.as_secs() * 1_000_000_000 + elapsed.subsec_nanos() as u64;

            log::info!(
                "Elapsed: {:?}. Frames: {}. FPS: {}",
                elapsed,
                frame,
                frame * 1_000_000_000 / elapsed_ns
            );

            graph.take().unwrap().dispose(&mut factory, &());
        }
    });
}

fn main() {
    env_logger::Builder::from_default_env()
        .filter_module("sprite", log::LevelFilter::Trace)
        .init();

    let config: Config<BasicDevicesConfigure, BasicHeapsConfigure, GraphOptimizedQueues> =
        Default::default();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Rendy example")
        .with_inner_size((960, 640).into());

    let rendy = AnyWindowedRendy::init_auto(&config, window, &event_loop).unwrap();
    rendy::with_any_windowed_rendy!((rendy)
        (mut factory, mut families, surface, window) => {
            let size = window.inner_size().to_physical(window.hidpi_factor());

            let caps = factory.get_surface_capabilities(&surface);
            let image_count = 2
                .min(*caps.image_count.end())
                .max(*caps.image_count.start());

            use hal::window::PresentMode;

            let present_mode = match () {
                _ if caps.present_modes.contains(PresentMode::IMMEDIATE) => PresentMode::IMMEDIATE,
                _ if caps.present_modes.contains(PresentMode::RELAXED) => PresentMode::RELAXED,
                _ if caps.present_modes.contains(PresentMode::MAILBOX) => PresentMode::MAILBOX,
                _ if caps.present_modes.contains(PresentMode::FIFO) => PresentMode::FIFO,
                _ => panic!("No known present modes found"),
            };

            let mut graph_builder = GraphBuilder::<_, ()>::new();

            let target = factory
            .create_target(
                surface,
                hal::window::Extent2D {
                    width: size.width as _,
                    height: size.height as _,
                },
                image_count,
                present_mode,
                rendy_core::hal::image::Usage::COLOR_ATTACHMENT,
            ).unwrap();

            let clear = hal::command::ClearColor {
                float32: [1.0, 1.0, 1.0, 1.0],
            };

            let output = graph_builder.add(Present::new(target, Some(clear)));
            graph_builder.add(SpriteGraphicsPipeline::new(output));

            let graph = graph_builder
                .build(&mut factory, &mut families, &())
                .unwrap();

            run(event_loop, factory, families, graph);
    })
}

fn push_vertex_desc(
    elements: &[rendy_core::hal::pso::Element<rendy_core::hal::format::Format>],
    stride: rendy_core::hal::pso::ElemStride,
    rate: rendy_core::hal::pso::VertexInputRate,
    vertex_buffers: &mut Vec<rendy_core::hal::pso::VertexBufferDesc>,
    attributes: &mut Vec<rendy_core::hal::pso::AttributeDesc>,
) {
    let index = vertex_buffers.len() as rendy_core::hal::pso::BufferIndex;

    vertex_buffers.push(rendy_core::hal::pso::VertexBufferDesc {
        binding: index,
        stride,
        rate,
    });

    let mut location = attributes.last().map_or(0, |a| a.location + 1);
    for &element in elements {
        attributes.push(rendy_core::hal::pso::AttributeDesc {
            location,
            binding: index,
            element,
        });
        location += 1;
    }
}
