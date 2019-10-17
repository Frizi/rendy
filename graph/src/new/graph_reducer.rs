use super::graph::PlanDag;
use gfx_hal::Backend;
use graphy::{Direction, NodeIndex, Walker};

#[derive(Debug)]
pub enum Reduction {
    /// Reduction have no effect
    NoChange,
    /// Signal that node was changed
    Changed,
    /// Signal that current node was replaced by other node
    Replace(NodeIndex),
}

/// A reducer can reduce or simplify a given node based on its operator and inputs.
/// It can also edit the graphs by changing and replacing nodes other than the one
/// currently being reduced by using the provided editor directly.

pub(crate) trait Reducer<B: Backend, T: ?Sized>: std::fmt::Debug {
    /// Try to reduce a node if possible.
    fn reduce(&mut self, editor: &mut GraphEditor<'_, '_, B, T>, node: NodeIndex) -> Reduction;
    /// Invoked once when a round of reductions is finished. Can be used to
    /// do additional reductions at the end, which in turn can cause a new round
    /// of reductions.
    fn finalize(&mut self, editor: &mut GraphEditor<'_, '_, B, T>) {
        let _ = editor;
    }

    fn reducer_name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

#[derive(Debug, Clone, Copy)]
struct NodeProgress {
    node: NodeIndex,
    input_index: u32,
}

impl NodeProgress {
    fn new(node: NodeIndex, input_index: u32) -> Self {
        Self { node, input_index }
    }
}

type Reducers<B, T> = Vec<Box<dyn Reducer<B, T> + 'static>>;

#[derive(derivative::Derivative)]
#[derivative(Debug(bound = ""))]
pub struct GraphReducer<B: Backend, T: ?Sized> {
    reducers: Reducers<B, T>,
    state: ReductionState,
}

#[derive(Debug)]
struct ReductionState {
    stack: Vec<NodeProgress>,
    revisit: Vec<NodeIndex>,
    node_flags: Vec<State>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Unvisited,
    Revisit,
    OnStack,
    Visited,
    Dead,
}

impl ReductionState {
    fn state(&self, node: NodeIndex) -> State {
        self.node_flags
            .get(node.index())
            .copied()
            .unwrap_or(State::Unvisited)
    }

    fn set_state(&mut self, node: NodeIndex, state: State) {
        if self.node_flags.len() <= node.index() {
            self.node_flags.resize(node.index() + 1, State::Unvisited);
        }
        self.node_flags[node.index()] = state;
    }

    fn push(&mut self, node: NodeIndex) {
        let state = self.state(node);
        debug_assert_ne!(State::OnStack, state);
        debug_assert_ne!(State::Dead, state);

        self.set_state(node, State::OnStack);
        self.stack.push(NodeProgress::new(node, 0));
    }

    fn pop(&mut self) {
        let node = self.stack.pop().unwrap().node;
        self.set_state(node, State::Visited);
    }

    fn recurse(&mut self, node: NodeIndex) -> bool {
        match self.state(node) {
            State::OnStack | State::Visited | State::Dead => false,
            State::Unvisited | State::Revisit => {
                self.push(node);
                true
            }
        }
    }

    fn revisit(&mut self, node: NodeIndex) {
        if self.state(node) == State::Visited {
            self.set_state(node, State::Revisit);
            self.revisit.push(node);
        }
    }

    fn kill(&mut self, node: NodeIndex) {
        self.set_state(node, State::Dead);
    }

    fn is_dead(&self, node: NodeIndex) -> bool {
        self.state(node) == State::Dead
    }

    fn replace<'a, B: Backend, T: ?Sized>(
        &mut self,
        graph: &mut PlanDag<'a, B, T>,
        node: NodeIndex,
        replacement: NodeIndex,
        max_id: NodeIndex,
    ) {
        if replacement <= max_id {
            // `replacement` is an old node, so unlink `node` and assume that
            // `replacement` was already reduced and finish.
            for (_, user) in node.children().iter(&graph) {
                // Don't revisit this node if it refers to itself.
                if user != node {
                    self.revisit(user);
                }
            }
            graph.rewire_children(node, replacement).unwrap();
            self.kill(node);
        } else {
            // Replace all old uses of `node` with `replacement`, but allow new nodes
            // created by this reduction to use `node`.
            for (_, user) in node.children().iter(&graph) {
                // Don't revisit this node if it refers to itself.
                if user <= max_id && user != node {
                    self.revisit(user);
                }
            }
            graph
                .rewire_where(Direction::Outgoing, node, replacement, |_, user| {
                    user <= max_id
                })
                .unwrap();
            // Unlink `node` if it's no longer used.
            if node.children().walk_next(&graph).is_none() {
                self.kill(node);
            }

            // If there was a replacement, reduce it after popping {node}.
            self.recurse(replacement);
        }
    }

    fn reduce<'a, B: Backend, T: ?Sized>(
        &mut self,
        reducers: &mut Reducers<B, T>,
        graph: &mut PlanDag<'a, B, T>,
        node: NodeIndex,
    ) -> Reduction {
        let mut skip = reducers.len();
        let mut i = 0;

        loop {
            if i == reducers.len() {
                break;
            }
            if i == skip {
                i += 1;
                continue;
            }
            let mut editor = GraphEditor { state: self, graph };
            let reduction: Reduction = reducers[i].reduce(&mut editor, node);
            match reduction {
                Reduction::NoChange => {
                    // No change from this reducer, check next.
                    i += 1;
                }
                Reduction::Changed => {
                    log::trace!(
                        "In-place update of {:?} by reducer {}",
                        node,
                        reducers[i].reducer_name()
                    );
                    skip = i;
                    i = 0;
                    continue;
                }
                Reduction::Replace(replacement) => {
                    log::trace!(
                        "Replacement of {:?} with {:?} by reducer {}",
                        node,
                        replacement,
                        reducers[i].reducer_name()
                    );
                    return reduction;
                }
            }
        }

        if skip == reducers.len() {
            // No change from any reducer.
            return Reduction::NoChange;
        }
        // At least one reducer did some in-place reduction.
        return Reduction::Changed;
    }

    fn reduce_node<'a, B: Backend, T: ?Sized>(
        &mut self,
        reducers: &mut Reducers<B, T>,
        graph: &mut PlanDag<'a, B, T>,
        node: NodeIndex,
    ) {
        debug_assert!(self.stack.is_empty());
        debug_assert!(self.revisit.is_empty());
        debug_assert!(self.node_flags.is_empty());

        self.push(node);

        loop {
            if !self.stack.is_empty() {
                // Process the node on the top of the stack, potentially pushing more or
                // popping the node off the stack.
                self.reduce_top(reducers, graph);
            } else if let Some(node) = self.revisit.pop() {
                // If the stack becomes empty, revisit any nodes in the revisit queue.
                if self.state(node) == State::Revisit {
                    // state can change while in queue.
                    self.push(node);
                }
            } else {
                // Run all finalizers.
                let mut editor = GraphEditor { state: self, graph };
                for reducer in reducers.iter_mut() {
                    reducer.finalize(&mut editor);
                }

                // Check if we have new nodes to revisit.
                if self.revisit.is_empty() {
                    break;
                }
            }
        }

        // remove dead nodes
        for (id, state) in self.node_flags.drain(..).enumerate().rev() {
            if state == State::Dead {
                graph.remove_node(NodeIndex::new(id as _));
            }
        }

        debug_assert!(self.stack.is_empty());
        debug_assert!(self.revisit.is_empty());
        debug_assert!(self.node_flags.is_empty());
    }

    fn recurse_top_inputs<'a, B: Backend, T: ?Sized>(
        &mut self,
        graph: &mut PlanDag<'a, B, T>,
        reset: bool,
    ) -> bool {
        let NodeProgress { node, input_index } = self.stack.last().copied().unwrap();

        let offset = if !reset && node.parents().iter(&graph).len() < input_index as usize {
            input_index as usize
        } else {
            0
        };

        let parents_after = node.parents().iter(&graph).skip(offset);
        for (i, (_, input)) in parents_after.enumerate() {
            if input != node && self.recurse(input) {
                self.stack.last_mut().unwrap().input_index = (i + offset + 1) as u32;
                return true;
            }
        }

        let parents_upto = node.parents().iter(&graph).take(offset);
        for (i, (_, input)) in parents_upto.enumerate() {
            if input != node && self.recurse(input) {
                self.stack.last_mut().unwrap().input_index = i as u32 + 1;
                return true;
            }
        }

        return false;
    }

    fn reduce_top<'a, B: Backend, T: ?Sized>(
        &mut self,
        reducers: &mut Reducers<B, T>,
        graph: &mut PlanDag<'a, B, T>,
    ) {
        let node = self.stack.last().unwrap().node;

        if self.is_dead(node) {
            return self.pop(); // Node was killed while on stack.
        }

        log::trace!("Reduce top [{}]: {:?}", node.index(), graph.get_node(node));

        debug_assert_eq!(State::OnStack, self.state(node));

        // Recurse on an input if necessary.
        if self.recurse_top_inputs(graph, false) {
            return;
        }

        // Remember the max node id before reduction.
        let max_id = NodeIndex::new(graph.node_count() - 1);

        // All inputs should be visited or on stack. Apply reductions to node.
        match self.reduce(reducers, graph, node) {
            Reduction::NoChange => {
                // If there was no reduction, pop {node} and continue.
                self.pop();
            }
            Reduction::Changed => {
                // In-place update of {node}, may need to recurse on an input.
                if self.recurse_top_inputs(graph, true) {
                    return;
                }
                self.pop();

                // Revisit all uses of the node.
                for (_, user) in node.children().iter(graph) {
                    // Don't revisit this node if it refers to itself.
                    // TODO: impossible because Dag, do we want to keep it like that?
                    if user != node {
                        self.revisit(user);
                    }
                }
            }
            Reduction::Replace(replacement) => {
                self.pop();
                self.replace(graph, node, replacement, max_id);
            }
        }
    }
}

pub(crate) struct GraphEditor<'a, 'b: 'a, B: Backend, T: ?Sized> {
    state: &'a mut ReductionState,
    graph: &'a mut PlanDag<'b, B, T>,
}

impl<'a, 'b: 'a, B: Backend, T: ?Sized> GraphEditor<'a, 'b, B, T> {
    pub(crate) fn graph(&self) -> &PlanDag<'b, B, T> {
        self.graph
    }

    pub(crate) fn graph_mut(&mut self) -> &mut PlanDag<'b, B, T> {
        self.graph
    }

    pub(crate) fn replace(&mut self, node: NodeIndex, replacement: NodeIndex) {
        self.state
            .replace(self.graph, node, replacement, NodeIndex::end());
    }

    pub(crate) fn revisit(&mut self, node: NodeIndex) {
        self.state.revisit(node);
    }

    pub(crate) fn kill(&mut self, node: NodeIndex) {
        self.state.kill(node);
    }

    pub(crate) fn is_dead(&self, node: NodeIndex) -> bool {
        self.state.is_dead(node)
    }
}

impl<B: Backend, T: ?Sized> GraphReducer<B, T> {
    pub(crate) fn new() -> Self {
        Self {
            reducers: Vec::new(),
            state: ReductionState {
                stack: Vec::new(),
                revisit: Vec::new(),
                node_flags: Vec::new(),
            },
        }
    }

    pub(crate) fn add_reducer(&mut self, reducer: impl Reducer<B, T> + 'static) {
        self.reducers.push(Box::new(reducer));
    }

    pub(crate) fn with_reducer(mut self, reducer: impl Reducer<B, T> + 'static) -> Self {
        self.add_reducer(reducer);
        self
    }

    pub(crate) fn reduce_graph<'a>(&mut self, graph: &mut PlanDag<'a, B, T>) {
        self.state
            .reduce_node(&mut self.reducers, graph, NodeIndex::new(0))
    }
}
