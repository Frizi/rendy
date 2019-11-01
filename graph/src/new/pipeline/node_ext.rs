use crate::new::{
    graph::{PlanDag, PlanEdge},
    graph_reducer::GraphEditor,
    pipeline::Contributions,
};
use gfx_hal::Backend;
use graphy::{NodeIndex, Walker};

pub(crate) trait NodeExt<B: Backend, T: ?Sized> {
    fn origin(&self, graph: &PlanDag<B, T>) -> Option<NodeIndex>;
    fn version(&self, graph: &PlanDag<B, T>) -> Option<NodeIndex>;
    fn child_version(&self, graph: &PlanDag<B, T>) -> Option<NodeIndex>;
    fn directly_uses(
        &self,
        graph: &PlanDag<B, T>,
        contributions: &Contributions,
        other: NodeIndex,
    ) -> bool;
    fn directly_uses_or_not_at_all(
        &self,
        graph: &PlanDag<B, T>,
        contributions: &Contributions,
        other: NodeIndex,
    ) -> bool;
    fn has_single_user(&self, editor: &GraphEditor<B, T>) -> bool;
    fn count_contributrions(&self, editor: &GraphEditor<B, T>, from: NodeIndex) -> usize;
}

impl<B: Backend, T: ?Sized> NodeExt<B, T> for NodeIndex {
    fn origin(&self, graph: &PlanDag<B, T>) -> Option<NodeIndex> {
        self.parents()
            .filter(|g: &PlanDag<B, T>, (e, _)| g[*e].is_origin())
            .walk_next(graph)
            .map(|t| t.1)
    }
    fn version(&self, graph: &PlanDag<B, T>) -> Option<NodeIndex> {
        self.parents()
            .filter(|g: &PlanDag<B, T>, (e, _)| g[*e].is_version())
            .walk_next(graph)
            .map(|t| t.1)
    }
    fn child_version(&self, graph: &PlanDag<B, T>) -> Option<NodeIndex> {
        self.children()
            .filter(|g: &PlanDag<B, T>, (e, _)| g[*e].is_version())
            .walk_next(graph)
            .map(|t| t.1)
    }
    fn directly_uses(
        &self,
        graph: &PlanDag<B, T>,
        contributions: &Contributions,
        other: NodeIndex,
    ) -> bool {
        if contributions.uses(*self, other) {
            self.parents()
                .iter(graph)
                .find(|(_, n)| *n == other)
                .is_some()
        } else {
            false
        }
    }
    fn directly_uses_or_not_at_all(
        &self,
        graph: &PlanDag<B, T>,
        contributions: &Contributions,
        other: NodeIndex,
    ) -> bool {
        if contributions.uses(*self, other) {
            self.parents()
                .iter(graph)
                .find(|(_, n)| *n == other)
                .is_some()
        } else {
            true
        }
    }
    fn has_single_user(&self, editor: &GraphEditor<B, T>) -> bool {
        self.children()
            .filter(|g, (e, n)| !editor.is_dead(*n) && g[*e] != PlanEdge::Version)
            .iter(editor.graph())
            .take(2)
            .count()
            == 1
    }
    fn count_contributrions(&self, editor: &GraphEditor<B, T>, from: NodeIndex) -> usize {
        self.parents()
            .iter(editor.graph())
            .filter(|(_, n)| editor.context().contributions.uses(*n, from))
            .count()
    }
}
