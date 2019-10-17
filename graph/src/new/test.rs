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

use crate::new::graph::PlanDag;

#[allow(unused_macros)]
macro_rules! graph_decl {
        (@decl [$graph:ident, $alloc:expr],) => {};
        (@decl [$graph:ident, $alloc:expr], $name:ident = $node:expr; $($tail:tt)*) => {
            let $name = $graph.insert_node($alloc, $node);
            graph_decl!(@decl [$graph, $alloc], $($tail)*);
        };
        (@decl [$graph:ident, $alloc:expr], @ $name:ident = $node:expr; $($tail:tt)*) => {
            let $name = $graph.insert_node($alloc, $node);
            $graph
                .insert_edge_unchecked($alloc, $name, graphy::NodeIndex::new(0), $crate::new::graph::PlanEdge::Effect)
                .unwrap();
            graph_decl!(@decl [$graph, $alloc], $($tail)*);
        };
        (@decl [$graph:ident, $alloc:expr], @ $name:ident; $($tail:tt)*) => {
            $graph
                .insert_edge_unchecked($alloc, $name, graphy::NodeIndex::new(0), $crate::new::graph::PlanEdge::Effect)
                .unwrap();
            graph_decl!(@decl [$graph, $alloc], $($tail)*);
        };
        (@decl [$graph:ident, $alloc:expr], $a:ident -> $b:ident = $edge:expr; $($tail:tt)*) => {
            $graph.insert_edge_unchecked($alloc, $a, $b, $edge).unwrap_or_else(|err| {
                panic!(
                    "Failed inserting link ({} -> {}): {:?}",
                    stringify!($a),
                    stringify!($b),
                    err,
                )
            });
            graph_decl!(@decl [$graph, $alloc], $($tail)*);
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

        ([$alloc:expr], $($tail:tt)*) => {{
            let mut graph = $crate::new::graph::PlanDag::<$crate::new::test::TestBackend, ()>::new();
            graph.insert_node($alloc, PlanNodeData::Root);
            graph_decl!(@decl [graph, $alloc], $($tail)*);
            graph
        }};
    }

#[allow(dead_code)]
pub(crate) fn test_init() {
    let _ = env_logger::builder().is_test(true).try_init();
}

/// Check if a graph returns specified topological order.
/// This comparison ignores edge values.
#[allow(dead_code)]
pub fn assert_topo_eq<A: std::fmt::Debug, B>(graph: &graphy::Graph<A, B>, topo: &[A]) {
    use crate::new::walker::Topo;
    use graphy::{NodeIndex, Walker};

    let actual_topo: Vec<NodeIndex> = Topo::new(graph, NodeIndex::new(0)).iter(graph).collect();
    let actual_topo: Vec<String> = actual_topo
        .into_iter()
        .map(|id| format!("{:?}", graph.get_node(id).unwrap()))
        .collect();
    let expected_topo: Vec<String> = topo.iter().map(|t| format!("{:?}", t)).collect();
    assert_eq!(expected_topo, actual_topo);
}

/// graphs are equivalent when their topological ordering of both edges and nodes is the same.
#[allow(dead_code)]
pub fn assert_equivalent<A: std::fmt::Debug, B: PartialEq + std::fmt::Debug>(
    expected_graph: &graphy::Graph<A, B>,
    actual_graph: &graphy::Graph<A, B>,
) {
    use crate::new::walker::{GraphItem, TopoWithEdges};
    use graphy::{NodeIndex, Walker};

    let expected_topo: Vec<GraphItem> = TopoWithEdges::new(expected_graph, NodeIndex::new(0))
        .iter(expected_graph)
        .collect();
    let actual_topo: Vec<GraphItem> = TopoWithEdges::new(actual_graph, NodeIndex::new(0))
        .iter(actual_graph)
        .collect();

    for (expected_item, actual_item) in expected_topo.iter().zip(actual_topo.iter()) {
        match (expected_item, actual_item) {
            (GraphItem::Node(expected), GraphItem::Node(actual)) => {
                let expected = expected_graph.get_node(*expected).unwrap();
                let actual = actual_graph.get_node(*actual).unwrap();
                let expected = format!("{:?}", expected);
                let actual = format!("{:?}", actual);
                log::trace!("n: {} == {}", expected, actual);
                assert_eq!(expected, actual);
            }
            (GraphItem::Edge(expected), GraphItem::Edge(actual)) => {
                let expected = expected_graph.get_edge(*expected).unwrap();
                let actual = actual_graph.get_edge(*actual).unwrap();
                log::trace!("e: {:?} == {:?}", expected, actual);
                assert_eq!(expected, actual);
            }
            (GraphItem::Node(expected), GraphItem::Edge(actual)) => {
                let expected = expected_graph.get_node(*expected).unwrap();
                let actual = actual_graph.get_edge(*actual).unwrap();
                panic!(
                    "Expected item of different kind. \n left: `Node({:?})`,\nright: Edge({:?})",
                    expected, actual
                );
            }
            (GraphItem::Edge(expected), GraphItem::Node(actual)) => {
                let expected = expected_graph.get_edge(*expected).unwrap();
                let actual = actual_graph.get_node(*actual).unwrap();
                panic!(
                    "Expected item of different kind. \n left: `Edge({:?})`,\nright: `Node({:?})`",
                    expected, actual
                );
            }
        }
    }
    assert_eq!(expected_topo.len(), actual_topo.len());
}

pub(crate) fn visualize_graph<B: gfx_hal::Backend, T: ?Sized>(
    write: &mut impl std::io::Write,
    graph: &PlanDag<B, T>,
    name: &str,
) {
    use crate::new::graph::{PlanEdge, PlanNodeData};
    use graphy::{EdgeIndex, NodeIndex};

    struct Visualize<'a, 'b, B: gfx_hal::Backend, T: ?Sized>(&'b PlanDag<'a, B, T>, String);

    impl<'a, 'b, B: gfx_hal::Backend, T: ?Sized> Visualize<'a, 'b, B, T> {
        fn edge_color(&self, index: EdgeIndex) -> &'static str {
            match self.0.get_edge(index).unwrap() {
                PlanEdge::Effect => "#ff615d",
                PlanEdge::Origin => "#ff950f",
                PlanEdge::ImageAccess(_, _) => "#aaaaff",
                PlanEdge::BufferAccess(_, _) => "#aaffaa",
                _ => "#aaaaaa",
            }
        }
        fn node_color(&self, index: NodeIndex) -> &'static str {
            match self.0.get_node(index).unwrap() {
                PlanNodeData::Root => "#99aaff",
                PlanNodeData::Image { .. } => "#0000ff",
                PlanNodeData::ImageVersion => "#0000aa",
                PlanNodeData::UndefinedImage => "#2266ff",
                PlanNodeData::ClearImage(_) => "#0000cc",
                _ => "black",
            }
        }
        fn node_label(&self, node: &PlanNodeData<'a, B, T>) -> String {
            match node {
                PlanNodeData::Root => format!("<font color=\"white\">{}</font>", self.1),
                PlanNodeData::Image(im) => format!("<table border='0' cellborder='1' cellspacing='0'><tr><td><b>Image</b></td></tr><tr><td>{:?}</td></tr><tr><td>{:?}, levels: {:?}</td></tr></table>", im.kind, im.format, im.levels),
                n => format!("{:?}", n),
            }
        }
        fn edge_label(&self, edge: &PlanEdge) -> String {
            match edge {
                PlanEdge::ImageAccess(a, s) => format!("<table border='0' cellborder='1' cellspacing='0'><tr><td><b>ImageAccess</b></td></tr><tr><td>{:?}</td></tr><tr><td>{:?}</td></tr></table>", a, s),
                PlanEdge::BufferAccess(a, s) => format!("<table border='0' cellborder='1' cellspacing='0'><tr><td><b>BufferAccess</b></td></tr><tr><td>{:?}</td></tr><tr><td>{:?}</td></tr></table>", a, s),
                e => format!("{:?}", e),
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    enum Item {
        Node(NodeIndex),
        Edge(EdgeIndex),
    }

    #[derive(Debug, Clone, Copy)]
    enum Edge {
        Start(NodeIndex, EdgeIndex),
        End(EdgeIndex, NodeIndex),
    }

    impl Edge {
        fn node(&self) -> NodeIndex {
            match self {
                Edge::Start(n, _) => *n,
                Edge::End(_, n) => *n,
            }
        }
        fn edge(&self) -> EdgeIndex {
            match self {
                Edge::Start(_, e) => *e,
                Edge::End(e, _) => *e,
            }
        }
    }

    impl<'a, 'b, B: gfx_hal::Backend, T: ?Sized> dot::Labeller<'b, Item, Edge>
        for Visualize<'a, 'b, B, T>
    {
        fn graph_id(&'b self) -> dot::Id<'b> {
            dot::Id::new(std::borrow::Cow::Borrowed(self.1.as_str())).unwrap()
        }
        fn node_id(&self, n: &Item) -> dot::Id<'b> {
            let id = match n {
                Item::Node(n) => format!("N{}", n.index()),
                Item::Edge(e) => format!("E{}", e.index()),
            };
            dot::Id::new(id).unwrap()
        }
        fn node_label(&'b self, n: &Item) -> dot::LabelText<'b> {
            match n {
                Item::Node(n_id) => match self.0.get_node(*n_id).unwrap() {
                    n => dot::LabelText::HtmlStr(
                        format!(
                            "<font color=\"{}\">{}</font>",
                            self.node_color(*n_id),
                            self.node_label(n)
                        )
                        .into(),
                    ),
                },
                Item::Edge(e) => dot::LabelText::HtmlStr(
                    format!(
                        "<font color=\"{}\">{}</font>",
                        self.edge_color(*e),
                        self.edge_label(self.0.get_edge(*e).unwrap()),
                    )
                    .into(),
                ),
            }
        }
        fn edge_end_arrow(&self, e: &Edge) -> dot::Arrow {
            match e {
                Edge::Start(_, _) => dot::Arrow::none(),
                Edge::End(_, _) => dot::Arrow::normal(),
            }
        }
        fn node_shape(&'b self, n: &Item) -> Option<dot::LabelText<'b>> {
            let shape = match n {
                Item::Node(n) => match self.0.get_node(*n).unwrap() {
                    PlanNodeData::Root => "circle",
                    PlanNodeData::Image(_) => "plain",
                    _ => "box",
                },
                Item::Edge(_) => "plain",
            };
            Some(dot::LabelText::LabelStr(shape.into()))
        }
        fn node_style(&self, n: &Item) -> dot::Style {
            match n {
                Item::Node(n) => match self.0.get_node(*n).unwrap() {
                    PlanNodeData::Root => dot::Style::Filled,
                    _ => dot::Style::Bold,
                },
                Item::Edge(_) => dot::Style::Rounded,
            }
        }
        fn edge_style(&self, e: &Edge) -> dot::Style {
            match self.0.get_edge(e.edge()).unwrap() {
                PlanEdge::Effect => dot::Style::Dotted,
                _ => dot::Style::Filled,
            }
        }
        fn edge_color(&self, e: &Edge) -> Option<dot::LabelText<'b>> {
            Some(dot::LabelText::LabelStr(self.edge_color(e.edge()).into()))
        }
        fn node_color(&self, n: &Item) -> Option<dot::LabelText<'b>> {
            let color = match n {
                Item::Edge(e) => self.edge_color(*e),
                Item::Node(n) => self.node_color(*n),
            };
            Some(dot::LabelText::LabelStr(color.into()))
        }
    }

    impl<'a, 'b, B: gfx_hal::Backend, T: ?Sized> dot::GraphWalk<'b, Item, Edge>
        for Visualize<'a, 'b, B, T>
    {
        fn nodes(&'b self) -> dot::Nodes<'b, Item> {
            let nodes = self.0.nodes_indices_iter().map(|n| Item::Node(n));
            let edges = self.0.edges_indices_iter().map(|e| Item::Edge(e));
            nodes.chain(edges).collect()
        }
        fn edges(&'b self) -> dot::Edges<'b, Edge> {
            let all_edges = self.0.edges_indices_iter().collect::<Vec<_>>();
            all_edges
                .into_iter()
                .flat_map(|e| {
                    let ends = self.0.get_edge_endpoints(e).unwrap();
                    Some(Edge::Start(ends.from, e))
                        .into_iter()
                        .chain(Some(Edge::End(e, ends.to)))
                })
                .collect()
        }
        fn source(&self, e: &Edge) -> Item {
            match e {
                Edge::Start(n, _) => Item::Node(*n),
                Edge::End(e, _) => Item::Edge(*e),
            }
        }
        fn target(&self, e: &Edge) -> Item {
            match e {
                Edge::Start(_, e) => Item::Edge(*e),
                Edge::End(_, n) => Item::Node(*n),
            }
        }
    }

    dot::render(&Visualize(&graph, name.into()), write).unwrap();
}
