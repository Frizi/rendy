use daggy::Walker;
use petgraph::visit::{IntoNeighbors, IntoNeighborsDirected, Reversed, VisitMap, Visitable};
use std::marker::PhantomData;
pub(crate) trait WalkerExt<G>: Walker<G> {
    fn filter<P>(self, predicate: P) -> Filter<G, Self, P>
    where
        Self: Sized,
        P: FnMut(G, &Self::Item) -> bool,
    {
        Filter {
            walker: self,
            predicate,
            _marker: PhantomData,
        }
    }

    fn filter_map<T, P>(self, predicate: P) -> FilterMap<G, Self, P>
    where
        Self: Sized,
        P: FnMut(G, &Self::Item) -> Option<T>,
    {
        FilterMap {
            walker: self,
            predicate,
            _marker: PhantomData,
        }
    }
}

impl<G, W> WalkerExt<G> for W where W: Walker<G> {}

#[derive(Clone, Debug)]
pub struct FilterMap<G, W, P> {
    walker: W,
    predicate: P,
    _marker: PhantomData<G>,
}

impl<G, W, I, P> Walker<G> for FilterMap<G, W, P>
where
    G: Copy,
    W: Walker<G>,
    P: FnMut(G, &W::Item) -> Option<I>,
{
    type Item = I;
    #[inline]
    fn walk_next(&mut self, graph: G) -> Option<Self::Item> {
        while let Some(item) = self.walker.walk_next(graph) {
            if let Some(mapped) = (self.predicate)(graph, &item) {
                return Some(mapped);
            }
        }
        None
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Filter<G, W, P> {
    walker: W,
    predicate: P,
    _marker: PhantomData<G>,
}

impl<G, W, P> Walker<G> for Filter<G, W, P>
where
    G: Copy,
    W: Walker<G>,
    P: FnMut(G, &W::Item) -> bool,
{
    type Item = W::Item;
    #[inline]
    fn walk_next(&mut self, graph: G) -> Option<Self::Item> {
        while let Some(item) = self.walker.walk_next(graph) {
            if (self.predicate)(graph, &item) {
                return Some(item);
            }
        }
        None
    }
}
/// A reverse topological order traversal for a graph, starting from a known root note.
pub struct Topo<N, VM> {
    tovisit: Vec<N>,
    ordered: VM,
}

impl<N, VM> Topo<N, VM>
where
    N: Copy + PartialEq,
    VM: VisitMap<N>,
{
    /// Create a new `Topo`, using the graph's visitor map, and put all
    /// initial nodes in the to visit list.
    pub fn new<G>(graph: G, root: N) -> Self
    where
        G: Visitable<NodeId = N, Map = VM>,
    {
        Topo {
            ordered: graph.visit_map(),
            tovisit: vec![root],
        }
    }

    /// Reset topological sort starting from passed root node.
    pub fn reset(&mut self, root: N) {
        self.tovisit.clear();
        self.tovisit.push(root);
    }

    /// Return the next node in the current reverse topological order traversal, or
    /// `None` if the traversal is at the end.
    pub fn next<G>(&mut self, g: G) -> Option<N>
    where
        G: IntoNeighborsDirected + Visitable<NodeId = N, Map = VM>,
    {
        // Take an unvisited element and find which of its neighbors are next
        while let Some(nix) = self.tovisit.pop() {
            if self.ordered.is_visited(&nix) {
                continue;
            }
            self.ordered.visit(nix);
            for neigh in Reversed(g).neighbors(nix) {
                // Look at each neighbor, and those that only have incoming edges
                // from the already ordered list, they are the next to visit.
                if g.neighbors(neigh).all(|b| self.ordered.is_visited(&b)) {
                    self.tovisit.push(neigh);
                }
            }
            return Some(nix);
        }
        None
    }
}

impl<G> Walker<G> for Topo<G::NodeId, G::Map>
where
    G: IntoNeighborsDirected + Visitable,
{
    type Item = G::NodeId;
    fn walk_next(&mut self, context: G) -> Option<Self::Item> {
        self.next(context)
    }
}
