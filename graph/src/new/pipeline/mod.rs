use {
    crate::new::{
        graph::PlanDag,
        graph_reducer::{GraphReducer, ReductionContext},
        walker::Topo,
    },
    gfx_hal::Backend,
    graphy::{GraphAllocator, NodeIndex, Walker},
    purple::collections::{BitVec, Vec},
};

mod misc;
mod node_ext;
mod pass;
mod subpass;

use misc::{CombineVersionsReducer, OrderWritesReducer};
use pass::CombinePassesReducer;
use subpass::CombineSubpassesReducer;

#[derive(derivative::Derivative)]
#[derivative(Debug(bound = ""))]
pub(crate) struct Pipeline<B: Backend, T: ?Sized> {
    stage1: GraphReducer<B, T>,
    stage2: GraphReducer<B, T>,
    stage3: GraphReducer<B, T>,
}

impl<B: Backend, T: ?Sized> Pipeline<B, T> {
    pub(crate) fn new() -> Self {
        Self {
            stage1: GraphReducer::new()
                .with_reducer(CombineSubpassesReducer)
                .with_reducer(CombineVersionsReducer),
            stage2: GraphReducer::new()
                .with_reducer(CombinePassesReducer)
                .with_reducer(CombineVersionsReducer),
            stage3: GraphReducer::new().with_reducer(OrderWritesReducer),
        }
    }
    #[inline(never)]
    pub(crate) fn reduce<'a>(&mut self, graph: &mut PlanDag<'a, B, T>, alloc: &'a GraphAllocator) {
        let start = std::time::Instant::now();
        // let mut file = std::fs::File::create("graph.dot").unwrap();

        graph.trim(NodeIndex::new(0)).unwrap();
        let trim1 = std::time::Instant::now();
        // crate::new::test::visualize_graph(&mut file, graph, "trim");

        let ctx = ReductionContext::new(Contributions::collect(graph, alloc));
        let collect = std::time::Instant::now();

        self.stage1.reduce_graph(graph, &ctx, alloc);
        let stage1 = std::time::Instant::now();
        // crate::new::test::visualize_graph(&mut file, graph, "stage1");

        self.stage2.reduce_graph(graph, &ctx, alloc);
        let stage2 = std::time::Instant::now();
        // crate::new::test::visualize_graph(&mut file, graph, "stage2");

        graph.trim(NodeIndex::new(0)).unwrap();
        let trim2 = std::time::Instant::now();

        self.stage3.reduce_graph(graph, &ctx, alloc);
        let stage3 = std::time::Instant::now();

        eprintln!(
            "Reduce timings:\n\ttrim1:  {}μs\n\tcollect:{}μs\n\tstage1: {}μs\n\tstage2: {}μs\n\ttrim2:  {}μs\n\tstage3: {}μs\n\ttotal:  {}μs",
            (trim1 - start).as_micros(),
            (collect - trim1).as_micros(),
            (stage1 - collect).as_micros(),
            (stage2 - stage1).as_micros(),
            (trim2 - stage2).as_micros(),
            (stage3 - trim2).as_micros(),
            (stage3 - start).as_micros()
        );
    }
}

pub(crate) struct Contributions<'a> {
    row_size: u32,
    end_index: u32,
    slab: BitVec<'a>,
}

impl<'a> Contributions<'a> {
    fn collect<B: Backend, T: ?Sized>(
        graph: &PlanDag<'a, B, T>,
        alloc: &'a GraphAllocator,
    ) -> Self {
        let align = 128;
        let row_size = (graph.node_count() + align - 1) & !(align - 1);
        let total_size = row_size * graph.node_count();
        let slab = BitVec::with_size_in(&alloc.0, total_size);

        let mut contribs = Self {
            row_size,
            end_index: graph.node_count(),
            slab,
        };

        let mut nodes = Vec::with_capacity_in(&alloc.0, graph.node_count() as usize);
        for node in Topo::new(graph, NodeIndex::new(0)).iter(graph) {
            nodes.push(node).unwrap();
        }

        for node in nodes.iter().rev() {
            for (_, parent) in node.parents().iter(graph) {
                contribs.insert_use(*node, parent);
            }
        }

        contribs
    }

    pub fn uses(&self, user_index: NodeIndex, dep_index: NodeIndex) -> bool {
        assert!(user_index.inner() < self.end_index && dep_index.inner() < self.end_index);
        // bounds checked in assert above
        unsafe {
            self.slab
                .get_unchecked(user_index.inner() * self.row_size + dep_index.inner())
        }
    }

    fn insert_use(&mut self, user_index: NodeIndex, dep_index: NodeIndex) {
        assert!(user_index.inner() < self.end_index && dep_index.inner() < self.end_index);
        assert_ne!(user_index, dep_index);

        // bounds checked in asserts above
        unsafe {
            self.slab
                .set_unchecked(user_index.inner() * self.row_size + dep_index.inner(), true);
        }
        let src_base = dep_index.inner() * self.row_size;
        let dst_base = user_index.inner() * self.row_size;
        self.slab
            .or_subslice_aligned(dst_base, src_base, self.row_size);
    }
}
