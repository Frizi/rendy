use crate::new::{
    graph::{PlanDag, PlanEdge},
    graph_reducer::GraphEditor,
    pipeline::Contributions,
};
use graphy::{NodeIndex, Walker};
use rendy_core::hal::Backend;

pub(crate) trait NodeExt<B: Backend> {
    fn origin(&self, graph: &PlanDag<B>) -> Option<NodeIndex>;
    fn child_origin(&self, graph: &PlanDag<B>) -> Option<NodeIndex>;
    fn version(&self, graph: &PlanDag<B>) -> Option<NodeIndex>;
    fn child_version(&self, graph: &PlanDag<B>) -> Option<NodeIndex>;
    fn directly_uses(
        &self,
        graph: &PlanDag<B>,
        contributions: &Contributions,
        other: NodeIndex,
    ) -> bool;
    fn directly_uses_or_not_at_all(
        &self,
        graph: &PlanDag<B>,
        contributions: &Contributions,
        other: NodeIndex,
    ) -> bool;
    fn has_single_user(&self, editor: &GraphEditor<B>) -> bool;
    fn count_contributrions(&self, editor: &GraphEditor<B>, from: NodeIndex) -> usize;
}

impl<B: Backend> NodeExt<B> for NodeIndex {
    fn origin(&self, graph: &PlanDag<B>) -> Option<NodeIndex> {
        self.parents()
            .filter(|g: &PlanDag<B>, (e, _)| g[*e].is_origin())
            .walk_next(graph)
            .map(|t| t.1)
    }

    fn child_origin(&self, graph: &PlanDag<B>) -> Option<NodeIndex> {
        self.children()
            .filter(|g: &PlanDag<B>, (e, _)| g[*e].is_origin())
            .walk_next(graph)
            .map(|t| t.1)
    }

    fn version(&self, graph: &PlanDag<B>) -> Option<NodeIndex> {
        self.parents()
            .filter(|g: &PlanDag<B>, (e, _)| g[*e].is_version())
            .walk_next(graph)
            .map(|t| t.1)
    }

    fn child_version(&self, graph: &PlanDag<B>) -> Option<NodeIndex> {
        self.children()
            .filter(|g: &PlanDag<B>, (e, _)| g[*e].is_version())
            .walk_next(graph)
            .map(|t| t.1)
    }

    fn directly_uses(
        &self,
        graph: &PlanDag<B>,
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
        graph: &PlanDag<B>,
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
    fn has_single_user(&self, editor: &GraphEditor<B>) -> bool {
        self.children()
            .filter(|g, (e, n)| !editor.is_dead(*n) && g[*e] != PlanEdge::Version)
            .iter(editor.graph())
            .take(2)
            .count()
            == 1
    }
    fn count_contributrions(&self, editor: &GraphEditor<B>, from: NodeIndex) -> usize {
        self.parents()
            .iter(editor.graph())
            .filter(|(_, n)| editor.context().contributions.uses(*n, from))
            .count()
    }
}
