#[cfg(all(
    feature = "empty",
    not(any(feature = "dx12", feature = "metal", feature = "vulkan"))
))]
pub(crate) use rendy_util::empty::Backend as TestBackend;

#[cfg(all(feature = "vulkan", not(any(feature = "dx12", feature = "metal"))))]
pub(crate) use rendy_util::vulkan::Backend as TestBackend;

#[cfg(all(feature = "metal", not(any(feature = "dx12"))))]
pub(crate) use rendy_util::metal::Backend as TestBackend;

#[cfg(all(feature = "dx12"))]
pub(crate) use rendy_util::dx12::Backend as TestBackend;

#[allow(unused_macros)]
macro_rules! graph_decl {
        (@decl [$graph:ident],) => {};
        (@decl [$graph:ident], $name:ident = $node:expr; $($tail:tt)*) => {
            let $name = $graph.add_node($node);
            graph_decl!(@decl [$graph], $($tail)*);
        };
        (@decl [$graph:ident], @ $name:ident = $node:expr; $($tail:tt)*) => {
            let $name = $graph.add_node($node);
            $graph
                .add_edge($name, daggy::NodeIndex::new(0), $crate::new::graph::PlanEdge::Effect)
                .unwrap();
            graph_decl!(@decl [$graph], $($tail)*);
        };
        (@decl [$graph:ident], @ $name:ident; $($tail:tt)*) => {
            $graph
                .add_edge($name, daggy::NodeIndex::new(0), $crate::new::graph::PlanEdge::Effect)
                .unwrap();
            graph_decl!(@decl [$graph], $($tail)*);
        };
        (@decl [$graph:ident], $a:ident -> $b:ident = $edge:expr; $($tail:tt)*) => {
            $graph.add_edge($a, $b, $edge).unwrap_or_else(|err| {
                panic!(
                    "Failed inserting link ({} -> {}): {}",
                    stringify!($a),
                    stringify!($b),
                    err,
                )
            });
            graph_decl!(@decl [$graph], $($tail)*);
        };
    }

#[cfg(any(
    feature = "empty",
    feature = "dx12",
    feature = "metal",
    feature = "vulkan"
))]
#[allow(unused_macros)]
macro_rules! graph {
        ($($tail:tt)*) => {{
            let mut graph = PlanDag::<$crate::new::test::TestBackend, ()>::new();
            graph.add_node(PlanNodeData::Root);
            graph_decl!(@decl [graph], $($tail)*);
            graph
        }};
    }

#[allow(dead_code)]
pub(crate) fn test_init() {
    let _ = env_logger::builder().is_test(true).try_init();
}

#[allow(dead_code)]
pub fn assert_topo_eq<A: std::fmt::Debug, B: PartialEq>(
    expected: &daggy::Dag<A, B>,
    actual: &daggy::Dag<A, B>,
) {
    use crate::new::walker::Topo;
    use daggy::{NodeIndex, Walker};

    let topo_expected: Vec<NodeIndex> = Topo::new(expected, NodeIndex::new(0))
        .iter(expected)
        .collect();
    let topo_actual: Vec<NodeIndex> = Topo::new(actual, NodeIndex::new(0)).iter(actual).collect();

    for (expected_id, actual_id) in topo_expected.iter().zip(&topo_actual).rev() {
        let expected_node = expected.node_weight(*expected_id).unwrap();
        let actual_node = expected.node_weight(*actual_id).unwrap();
        let expected_fmt = format!("[{}] {:?}", expected_id.index(), expected_node);
        let actual_fmt = format!("[{}] {:?}", actual_id.index(), actual_node);
        assert_eq!(expected_fmt, actual_fmt);
    }
    assert_eq!(topo_expected.len(), topo_actual.len());
}
