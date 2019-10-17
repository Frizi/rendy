use fixedbitset::FixedBitSet;
use graphy::{EdgeIndex, Graph, NodeIndex, Walker};

/// A reverse topological order traversal for a graph, starting from a known root node.
pub struct Topo {
    tovisit: Vec<NodeIndex>,
    ordered: FixedBitSet,
}

impl Topo {
    /// Create a new `Topo`, using the graph's visitor map, and put all
    /// initial nodes in the to visit list.
    pub fn new<N, E>(graph: &Graph<N, E>, root: NodeIndex) -> Self {
        Topo {
            ordered: FixedBitSet::with_capacity(graph.node_count() as _),
            tovisit: vec![root],
        }
    }

    /// Reset topological sort starting from passed root node.
    pub fn reset(&mut self, root: NodeIndex) {
        self.tovisit.clear();
        self.tovisit.push(root);
    }

    /// Return the next node in the current reverse topological order traversal, or
    /// `None` if the traversal is at the end.
    pub fn next<N, E>(&mut self, g: &Graph<N, E>) -> Option<NodeIndex> {
        // Take an unvisited element and find which of its neighbors are next
        while let Some(node) = self.tovisit.pop() {
            if self.ordered.contains(node.index()) {
                continue;
            }
            self.ordered.put(node.index());
            for (_, neigh) in node.parents().iter(g) {
                // Look at each neighbor, and those that only have incoming edges
                // from the already ordered list, they are the next to visit.
                if neigh
                    .children()
                    .iter(g)
                    .all(|(_, b)| self.ordered.contains(b.index()))
                {
                    self.tovisit.push(neigh);
                }
            }
            return Some(node);
        }
        None
    }
}

impl<'a, N, E> Walker<&Graph<'a, N, E>> for Topo {
    type Item = NodeIndex;
    fn walk_next(&mut self, context: &Graph<'a, N, E>) -> Option<Self::Item> {
        self.next(context)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum GraphItem {
    Edge(EdgeIndex),
    Node(NodeIndex),
}

/// A reverse topological order traversal for a graph, starting from a known root node.
/// Also traverses edges.
pub struct TopoWithEdges {
    tovisit: Vec<GraphItem>,
    ordered_nodes: FixedBitSet,
    ordered_edges: FixedBitSet,
}

impl TopoWithEdges {
    /// Create a new `Topo`, using the graph's visitor map, and put all
    /// initial nodes in the to visit list.
    pub fn new<N, E>(graph: &Graph<N, E>, root: NodeIndex) -> Self {
        TopoWithEdges {
            tovisit: vec![GraphItem::Node(root)],
            ordered_nodes: FixedBitSet::with_capacity(graph.node_count() as _),
            ordered_edges: FixedBitSet::with_capacity(graph.edge_count() as _),
        }
    }

    /// Reset topological sort starting from passed root node.
    pub fn reset(&mut self, root: NodeIndex) {
        self.tovisit.clear();
        self.tovisit.push(GraphItem::Node(root));
    }

    /// Return the next node in the current reverse topological order traversal, or
    /// `None` if the traversal is at the end.
    pub fn next<N, E>(&mut self, g: &Graph<N, E>) -> Option<GraphItem> {
        // Take an unvisited element and find which of its neighbors are next
        while let Some(item) = self.tovisit.pop() {
            match item {
                GraphItem::Node(node) => {
                    if self.ordered_nodes.contains(node.index()) {
                        continue;
                    }
                    self.ordered_nodes.put(node.index());
                    // this node just got visited, visit all incoming edges next
                    self.tovisit.extend(
                        node.parents()
                            .iter(g)
                            .map(|(edge, _)| GraphItem::Edge(edge)),
                    );
                }
                GraphItem::Edge(edge) => {
                    if self.ordered_edges.contains(edge.index()) {
                        continue;
                    }
                    self.ordered_edges.put(edge.index());

                    // this node just got visited, visit it's origin if all it's outgoing edges were already visited
                    let origin = g.get_edge_endpoints(edge).unwrap().from;
                    if origin
                        .children()
                        .iter(g)
                        .all(|(edge, _)| self.ordered_edges.contains(edge.index()))
                    {
                        self.tovisit.push(GraphItem::Node(origin));
                    }
                }
            }
            return Some(item);
        }
        None
    }
}

impl<'a, N, E> Walker<&Graph<'a, N, E>> for TopoWithEdges {
    type Item = GraphItem;
    fn walk_next(&mut self, context: &Graph<'a, N, E>) -> Option<Self::Item> {
        self.next(context)
    }
}
