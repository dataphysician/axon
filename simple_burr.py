from __future__ import annotations

from typing import Any, Callable, AsyncGenerator, Generator, Protocol, Mapping
from burr.core import action, State, ApplicationBuilder
from burr.core.application import ApplicationContext
from burr.core.parallelism import MapStates, RunnableGraph
from burr.core.graph import GraphBuilder
from rich import print as rprint
import asyncio
from random import Random

# --------------------------------------------------------------------------------------
# Selector Types

class Selector(Protocol):
    def __call__(
        self,
        context: str | None,
        candidates: list[str],
        feedback: str | None
    ) -> tuple[list[str], Any | None]:
        ...

def random_selector(
    context: str | None,
    candidates: list[str],
    feedback: str | None,
    *,
    rng: Random | None = None
) -> tuple[list[str], Any | None]:
    """Simple random selector with optional reproducibility via rng."""
    rng = rng or Random()
    # For debugging purposes, let's select all candidates instead of filtering
    ids = candidates
    if not ids:
        return [], None
    if len(ids) == 1:
        return [ids[0]], None
    k: int = rng.choice([1, 2])
    k = min(k, len(ids))
    rng.shuffle(ids)
    # keep chosen order; do NOT sort here or you lose the RNG ordering
    return ids[:k], None

# Switch the selector with your real selection logic
def selection_agent(
    context: str | Mapping[str, Any] | None,
    candidates: list[str] | None = None,
    feedbacks: Mapping[str, Any] | None = None,
    *,
    rng: Random | None = None,
) -> tuple[list[str], Any | None]:
    """
    Adapter that normalizes context/feedbacks to strings for Selector.
    """
    candidates = candidates or []
    ctx_str = context if isinstance(context, str) else ""
    fb_str = None if not isinstance(feedbacks, Mapping) else feedbacks.get('default')

    selected_candidates, reasoning = random_selector(ctx_str, candidates, fb_str, rng=rng)
    reasoning = "", ""  # NOTE: demo/debug-only, remove later
    return selected_candidates, reasoning

# --------------------------------------------------------------------------------------
# Helpers / schema utilities

def _make_batch_name(node_id: str, kind: str) -> str:
    return f"{node_id}_-_{kind}"

def _iter_batches_for_node(node: dict[str, Any]) -> Generator[tuple[str, list[str]]]:
    """
    Yield (batch_name, candidates) pairs for a node.
    - metadata keys: codeFirst, codeAlso, useAdditionalCode (values: list[str])
    - children: dict[str, str] -> list of child node_ids (list[str])
    """
    node_id: str = node["id"]
    md: dict[str, Any] = node.get("metadata", {})
    for key in ["codeFirst", "codeAlso", "useAdditionalCode"]:
        vals = md.get(key)
        if isinstance(vals, list) and all(isinstance(x, str) for x in vals):
            yield _make_batch_name(node_id, key), list(vals)

    children = node.get("children", {})
    if isinstance(children, dict) and children:
        yield _make_batch_name(node_id, "children"), list(children.keys())

def _is_node_id(index: dict[str, Any], item: str) -> bool:
    """A selected item is a node_id if it exists in the flat index."""
    return item in index

def _split_batch_name(name: str) -> tuple[str, str]:
    node_id, _, kind = name.partition("_-_")
    return node_id, kind

def _feedback_for_batch(batch_name: str, feedbacks: dict[str, Any]) -> dict[str, Any]:
    """
    Supports either flattened keys: {"A_-_children": {...}}
    or nested keys: {"A": {"children": {...}}}
    """
    if not isinstance(feedbacks, dict):
        return {}
    node_id, kind = _split_batch_name(batch_name)
    if batch_name in feedbacks and isinstance(feedbacks[batch_name], dict):
        return feedbacks[batch_name]
    nested = feedbacks.get(node_id)
    if isinstance(nested, dict):
        kval = nested.get(kind)
        if isinstance(kval, dict):
            return kval
    return {}

# --------------------------------------------------------------------------------------
# Actions

@action(
    reads=["index", "queue", "visited", "visited_set", "batches", "end_nodes"],
    writes=["queue", "visited", "visited_set", "batches", "end_nodes"],
)
def spawn_batches(state: State) -> State:
    """
    DFS step: pop from the END of queue (LIFO) to expand a node and create up to four batches.
    Mark node as terminal iff expansion produced no batches.
    """
    index: dict[str, dict[str, Any]] = state["index"]
    queue: list[str] = list(state["queue"])
    visited: list[str] = list(state.get("visited", []))
    visited_set = set(state.get("visited_set", []))
    batches: dict[str, list[str]] = dict(state["batches"])
    end_nodes: list[str] = list(state.get("end_nodes", []))

    if not queue:
        return state

    node_id = queue.pop()  # DFS
    if node_id in visited_set:
        return state.update(queue=queue)

    visited.append(node_id)
    visited_set.add(node_id)

    # --- permissive mode for non-index items (temporary) ---
    if node_id not in index:
        rprint(f"[yellow]<traversal_exited>[/] [bold]Cannot expand node:[/] {node_id}")
        if node_id not in end_nodes:
            end_nodes.append(node_id)  # treat as terminal so it appears in final output
        return state.update(
            queue=queue,
            visited=visited,
            visited_set=list(visited_set),
            batches=batches,
            end_nodes=end_nodes,
        )
    # -------------------------------------------------------

    node = index[node_id]
    before = len(batches)
    for batch_name, candidates in _iter_batches_for_node(node):
        batches[batch_name] = candidates
    after = len(batches)

    # terminal iff expansion produced no work
    if after == before and node_id not in end_nodes:
        end_nodes.append(node_id)

    return state.update(
        queue=queue,
        visited=visited,
        visited_set=list(visited_set),
        batches=batches,
        end_nodes=end_nodes,
    )

@action(
    reads=["context", "batches", "batch_name", "batch_feedback", "rng_seed"],
    writes=["batch_result"],
)
async def select_candidates(state: State) -> State:
    """
    Per-batch selection. Writes batch_result = {batch_name, selected}.
    """
    batch_name: str = state.get("batch_name", "UNKNOWN")
    candidates: list[str] = state.get("batches", {}).get(batch_name, [])
    context = state.get("context")
    # IMPORTANT: ProcessBatches.states() writes the resolved per-batch dict onto 'feedbacks'
    feedbacks: dict[str, Any] = state.get("batch_feedback", {})

    # optional reproducibility
    seed = state.get("rng_seed")
    rng = Random(seed) if seed is not None else None

    selected, reasoning = selection_agent(context, candidates, feedbacks=feedbacks, rng=rng)
    reason, citation = ("", "") if not (isinstance(reasoning, (list, tuple)) and len(reasoning) == 2) else reasoning

    # # NOTE: debug printing
    # rprint(f"({batch_name}) Enumerating {len(candidates)} candidates: {candidates}")
    # reason, citation = "Test", None  # NOTE: demo logging, remove later 
    # if not selected:
    #     rprint("No candidates selected.\n")
    # elif len(selected) == 1:
    #     rprint(f"Selected candidate: {selected[0]}")
    #     rprint(f"Reason: {reason}" if reason else "")
    #     rprint(f"Citation: {citation}\n" if citation else "")
    # else:
    #     rprint(f"Selected candidates: {list(selected)}")
    #     rprint(f"Reason: {reason}" if reason else "")
    #     rprint(f"Citation: {citation}\n" if citation else "")

    return state.update(batch_result={"batch_name": batch_name, "selected": selected})

@action(reads=["traversal_depths"], writes=["traversal_depths"])
def ensure_traversal_depths(state: State) -> State:
    return state if "traversal_depths" in state else state.update(traversal_depths={})

@action(reads=["traversal_kind"], writes=["traversal_kind"])
def ensure_traversal_kind(state: State) -> State:
    return state if "traversal_kind" in state else state.update(traversal_kind={})

class ProcessBatches(MapStates):
    """
    Process all batches in parallel and reduce results into queue/batches/end_nodes.
    """
    def action(self, state: State, inputs: dict[str, Any] | None = None):
        return select_candidates

    async def states(self, state: State, context: ApplicationContext, inputs: dict[str, Any] | None = None) -> AsyncGenerator[State, None]:
        batches: dict[str, list[str]] = state["batches"]
        all_feedbacks: dict[str, Any] = state.get("feedbacks", {})
        for batch_name in list(batches.keys()):
            fb = _feedback_for_batch(batch_name, all_feedbacks)
            # For each fan-out state, attach just this batch's feedback dict under 'batch_feedback'
            yield state.update(batch_name=batch_name, batch_feedback=fb)

    async def reduce(self, state: State, states: AsyncGenerator[State, None]) -> State:
        """
        Reduce results: enqueue selected node_ids (recurse) or collect terminal artifacts into end_nodes.
        """
        index: dict[str, Any] = state["index"]
        queue: list[str] = list(state["queue"])
        batches: dict[str, list[str]] = dict(state["batches"])
        end_nodes: list[str] = list(state["end_nodes"])
        traversal_depths: dict[str, int] = dict(state.get("traversal_depths", {}))
        traversal_kind: dict[str, str | None] = dict(state.get("traversal_kind", {}))

        async for result_state in states:
            batch_name: str = result_state["batch_name"]
            selected: list[str] = result_state["batch_result"]["selected"] or []

            parent_id, _kind = _split_batch_name(batch_name)
            parent_depth = traversal_depths.get(parent_id, 0)

            # for item in selected:
            #     if _is_node_id(index, item):
            #         queue.append(item)                      # DFS enqueue
            #         traversal_depths.setdefault(item, parent_depth + 1)
            #         traversal_kind.setdefault(item, _kind)  # record how we reached this node
            #     else:
            #         if item not in end_nodes:               # artifact string
            #             end_nodes.append(item)

            for item in selected:
                if item not in index and item not in end_nodes: # Mark non-indexed nodes as an end_node
                    end_nodes.append(item)
                queue.append(item)                          # DFS enqueue regardless of index membership
                traversal_depths.setdefault(item, parent_depth + 1) # Depth of Traversal
                traversal_kind.setdefault(item, _kind) # Batch of Selected Candidate

            if batch_name in batches:
                del batches[batch_name]

        return state.update(
            queue=queue, 
            batches=batches, 
            end_nodes=end_nodes, 
            traversal_depths=traversal_depths, 
            traversal_kind=traversal_kind, 
            batch_result={}
        )

    def is_async(self) -> bool:
        return True

    @property
    def reads(self) -> list[str]:
        return [
            "context", "batches", "index", "queue", "end_nodes",
            "feedbacks", "rng_seed",
            "batch_name", "batch_feedback", "traversal_depths", "traversal_kind", "batch_result",
        ]
    @property
    def writes(self) -> list[str]:
        return [
            "queue", "batches", "end_nodes",
            "batch_result", "traversal_depths", "traversal_kind",     # NOTE: batch_result is cleared in reduce
        ]

# Register the action with Burr
process_batches = ProcessBatches()

# --------------------------------------------------------------------------------------
# Round graph: expand one node -> process all batches
def round_graph() -> RunnableGraph:
    graph = (
        GraphBuilder()
        .with_actions(
            expand=spawn_batches,
            process=process_batches,
        )
        .with_transitions(("expand", "process"), ("process", "expand"))
        .build()
    )
    return RunnableGraph(graph=graph, entrypoint="expand", halt_after=["process"])

# --------------------------------------------------------------------------------------
# Driver helpers (loop until fixed point)

@action(reads=["end_nodes"], writes=["end_nodes"])
def ensure_end_nodes(state: State) -> State:
    """Idempotently ensure the end_nodes key exists."""
    return state if "end_nodes" in state else state.update(end_nodes=[])

async def run_until_converged(base_state: State) -> State:
    # Ensure once up-front
    state = ensure_end_nodes(base_state)
    state = ensure_traversal_depths(state)
    state = ensure_traversal_kind(state)

    while True:
        if not state.get("queue", []) and not state.get("batches", {}):
            return state

        sub = round_graph()

        app = await (
            ApplicationBuilder()
            .with_graph(sub.graph)
            .with_entrypoint(sub.entrypoint)
            .with_state(state)
            .with_typing(None)
            .abuild()
        )

        _action, _result, state = await app.arun(halt_after=sub.halt_after)

# --------------------------------------------------------------------------------------
# Example usage / quick test harness
if __name__ == "__main__":
    from simple_icd import build_index_flat
    index = build_index_flat("icd10cm_tabular_2026.xml")

    def tabbed_output(visited: list[str], traversal_depths: dict[str, int], traversal_kind: dict[str, str | None], tab: str = "\t") -> str:
        lines: list[str] = []
        for node in visited:
            depth = traversal_depths.get(node, 0)
            kind = traversal_kind.get(node)
            kind = "" if not kind or kind == "children" else f"({kind})"
            suffix = f" {kind}"
            lines.append(f"{tab * depth}{node}{suffix}")
        return "\n".join(lines)

    def tabbed_output(visited: list[str], traversal_depths: dict[str, int], traversal_kind: dict[str, str | None], index: dict[str, dict[str, Any]], tab: str = "    ") -> str:
        lines: list[str] = []
        for node in visited:
            depth = traversal_depths.get(node, 0)
            kind = traversal_kind.get(node)

            # get label if this is a real node id in the index
            label = None
            if isinstance(index, dict) and node in index:
                node_obj = index.get(node) or {}
                # typical shape: node_obj.get("label")
                maybe = node_obj.get("label")
                label = maybe if isinstance(maybe, str) and maybe.strip() else None

            # build parts: node_id [— label] [(kind)]
            label_part = f" — {label}" if label else ""
            kind_str = "" if not kind or kind == "children" else f"({kind})"
            kind_part = f" {kind_str}" if kind_str else ""

            lines.append(f"{tab * depth}{node}{label_part}{kind_part}")
        return "\n".join(lines)


    init = State().update(
        index=index,
        queue=["ROOT"],   # stack top is the END; expand pops from END
        traversal_depths={"ROOT": 0},
        traversal_kind={"ROOT": None},
        batches={},
        visited=[],
        # visited_set may be omitted; action will initialize it
        end_nodes=[],
        context="Hello world",
    )

    final_state = asyncio.run(run_until_converged(init))
    rprint("=== DONE ===")
    # rprint("Visited:", final_state["visited"])
    rprint(
        "ICD Code Traversal:\n" + 
        tabbed_output(final_state["visited"], final_state["traversal_depths"], final_state.get("traversal_kind", {}), index=index, tab="    ")
    )
    rprint("End nodes:", tuple(final_state["end_nodes"]))
