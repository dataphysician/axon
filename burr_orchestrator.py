from __future__ import annotations

from typing import Any, Callable, AsyncGenerator, Generator, Protocol, Mapping
from burr.core import action, State, ApplicationBuilder
from burr.core.application import ApplicationContext
from burr.core.parallelism import MapStates, RunnableGraph
from burr.core.graph import GraphBuilder
from rich import print as rprint
from random import Random
import asyncio
from llm_selector import llm_agent, configure_llm
# import dspy
# from dspy_selector import NodeProgram, ensure_dspy_configured

# --------------------------------------------------------------------------------------
# Selector Types
class AsyncSelector(Protocol):
    async def __call__(
        self, 
        context: str | None, 
        candidates: list[str], 
        feedback: str | None,
        batch_name: str,
    ) -> tuple[list[str], Any | None]: 
        ...

async def random_selector(
    context: str | None,
    candidates: list[str],
    feedback: str | None,
) -> tuple[list[str], Any | None]:
    """Simple random selector with optional reproducibility via rng."""
    rng = Random()
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

###

# # Switch the selector with your real selection logic
# async def async_dspy_selector(
#     batch_name: str,
#     context: str | None = None,    
#     candidates: dict[str, str] | None = None,
#     instructions: str | None = None,
#     feedback: str | None = None,
# ) -> tuple[list[str], Any | None]:
#     """
#     Async variant using DSPy's async helpers. Returns the same (list[str], reasoning) tuple.
#     """
#     ensure_dspy_configured()

#     prog = NodeProgram(
#         batch_name=batch_name or "selection",
#         candidates=candidates,
#         instruction=instructions,
#         feedback=feedback,
#         auto_load=True,
#     )
#     async_prog = dspy.asyncify(prog)  # wraps Module -> awaitable call
#     selected_ids, reasoning = await async_prog(context=context or "", candidates=candidates)

#     return selected_ids, (reasoning or None)

async def async_llm_selector(
    batch_name: str,
    context: str | None = None,    
    candidates: dict[str, str] | None = None,
    instructions: str | None = None,
    feedback: str | None = None,
) -> tuple[list[str], Any | None]:
    """
    Async variant using the LLM agent directly. Returns the same (list[str], reasoning) tuple.
    Can serve as a drop-in replacement for async_dspy_selector.
    """
    # Call llm_agent with the appropriate parameters
    selected_ids, reasoning = await llm_agent(
        batch_name=batch_name,
        context=context or "",
        candidates=candidates,
        instructions=instructions,
        feedback=feedback,
    )

    return selected_ids, (reasoning or None)

###



#############################################
async def selection_agent(
    context: str | dict[str, Any] | None,
    candidates: list[str] | dict[str, str] | None = None,
    feedbacks: dict[str, Any] | None = None,
    batch_name: str | None = None,
    **kwargs: Any,
) -> tuple[list[str], Any | None]:
    """
    Adapter that normalizes context/feedbacks to strings for Selector.
    """
    candidates = candidates or {}
    ctx_str = context if isinstance(context, str) else ""
    fb_str = None if not isinstance(feedbacks, dict) else feedbacks.get('default')

    # selected_candidates, reasoning = await random_selector(ctx_str, candidates, fb_str)
    # selected_candidates, reasoning = await async_dspy_selector(
    selected_candidates, reasoning = await async_llm_selector(
        batch_name=batch_name or None, 
        context=ctx_str, 
        candidates=candidates, 
        instructions=None,
        feedback=fb_str,
    )
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

def _build_candidate_texts(
    index: Mapping[str, Any],
    batch_name: str,
    ids: list[str],
) -> dict[str, str]:
    """
    Return {id: label/text}. For '..._-_children', prefer the node's children dict if present.
    Fallback to index[id]['label'] or the id itself.
    """
    texts: dict[str, str] = {}
    parent_id, kind = _split_batch_name(batch_name)

    # Prefer the parent's children dict when kind == "children"
    if kind == "children":
        try:
            children = (index.get(parent_id, {}) or {}).get("children") or {}
            if isinstance(children, dict):
                for cid in ids:
                    txt = children.get(cid)
                    if isinstance(txt, str) and txt.strip():
                        texts[cid] = txt
        except Exception:
            pass

    # Fallbacks per id
    for cid in ids:
        if cid in texts:
            continue
        label = (index.get(cid, {}) or {}).get("label")
        texts[cid] = label if isinstance(label, str) and label.strip() else cid

    return texts

# --------------------------------------------------------------------------------------
# Hierarchy-preserving DFS data structure for Marimo consumption

def collect_batch_info(
    batch_name: str,
    candidates: dict[str, str],
    selected: list[str],
    reasoning: str | None,
    citation: str | None,
    traversal_depths: dict[str, int],
    traversal_kind: dict[str, str | None],
) -> dict[str, Any]:
    """
    Collect batch information in a hierarchy-preserving DFS data structure for Marimo consumption.
    """
    parent_id, kind = _split_batch_name(batch_name)
    depth = traversal_depths.get(parent_id, 0)
    
    batch_info = {
        "batch_name": batch_name,
        "parent_id": parent_id,
        "depth": depth,
        "kind": kind,
        "candidates": candidates,
        "selected": selected,
        "reasoning": reasoning,
        "citation": citation,
    }
    
    # Store in a global or state-level structure for later access by Marimo
    if not hasattr(collect_batch_info, "collected_data"):
        collect_batch_info.collected_data = []
    
    collect_batch_info.collected_data.append(batch_info)
    
    return batch_info

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
    reads=["index", "context", "batches", "batch_name", "batch_feedback", "rng_seed"],
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

    index: dict[str, Any] = state["index"]
    candidates = _build_candidate_texts(index, batch_name, candidates)
    selected, reasoning = await selection_agent(context, candidates, feedbacks=feedbacks, batch_name=batch_name)

    # reason, citation = ("", "") if not (isinstance(reasoning, (list, tuple)) and len(reasoning) == 2) else reasoning
    reason, citation = None, None
    if isinstance(reasoning, tuple) and len(reasoning) == 2:
        reason, citation = reasoning
    if isinstance(reasoning, str):
        reason = reasoning

    # Collect batch information in hierarchy-preserving DFS structure
    traversal_depths = state.get("traversal_depths", {})
    traversal_kind = state.get("traversal_kind", {})
    batch_info = collect_batch_info(
        batch_name=batch_name,
        candidates=candidates,
        selected=selected,
        reasoning=reason,
        citation=citation,
        traversal_depths=traversal_depths,
        traversal_kind=traversal_kind
    )
    
    # NOTE: debug printing (now using collected data)
    # Only print if not in marimo environment
    import sys
    if 'marimo' not in sys.modules:
        rprint("#"*45)
        rprint(f"[yellow]({batch_name}) Selection:[/] Enumerating {len(candidates)} candidates:")
        for cid, label in candidates.items():
            rprint(f"\t{cid}: {label}")
        # reason, citation = "Test", None  # NOTE: demo logging, remove later 
        if not selected:
            rprint("No candidates selected.\n")
        elif len(selected) == 1:
            rprint(f"Selected candidate:\n\t{selected[0]}")
        else:
            rprint(f"Selected candidates:\n\t{list(selected)}")

        rprint(f"[yellow]({batch_name}) Reasoning:[/]\n\t{reason}" if reason else "")
        rprint(f"[yellow]({batch_name}) Citation:[/]\n\t{citation}" if citation else "")

    #############################################

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

            if _kind == "children" and not selected:
                if parent_id not in end_nodes:
                    end_nodes.append(parent_id)

            for item in selected:
                if item not in index and item not in end_nodes: # Mark non-indexed nodes as an end_node
                    end_nodes.append(item)
                queue.append(item)                          # DFS enqueue regardless of index membership
                # For metadata-based selections, keep the same depth as parent
                # For children, use parent_depth + 1
                if _kind == "children":
                    traversal_depths.setdefault(item, parent_depth + 1) # Depth of Traversal
                else:
                    traversal_depths.setdefault(item, parent_depth) # Same depth for metadata refs
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

# Global variable to track round completion for Marimo UI
_round_completion_tracker = []

def get_round_completion_data():
    """Get round completion data for Marimo UI."""
    global _round_completion_tracker
    return _round_completion_tracker

def reset_round_completion_data():
    """Reset round completion data for Marimo UI."""
    global _round_completion_tracker
    _round_completion_tracker = []

async def run_until_converged(base_state: State) -> State:
    # Ensure once up-front
    state = ensure_end_nodes(base_state)
    state = ensure_traversal_depths(state)
    state = ensure_traversal_kind(state)
    
    # Reset round completion tracker
    reset_round_completion_data()

    while True:
        # Check if we've reached a fixed point (no more work to do)
        if not state.get("queue", []) and not state.get("batches", {}):
            # Add formatted codes (node_id\tlabel\tBillable/Non-Billable) to the final state
            index = state.get("index", {})
            end_nodes = state.get("end_nodes", [])
            formatted_codes = []
            for node_id in end_nodes:
                # Get node data from index
                node_data = index.get(node_id, {})
                # Get label from index, fallback to node_id if not found
                label = node_data.get("label", node_id)
                # Determine if node is leaf (billable) or not
                children = node_data.get("children", {})
                is_leaf = not children or len(children) == 0
                billing_status = "Billable" if is_leaf else "Non-Billable"
                formatted_codes.append(f"{node_id}\t{label}\t{billing_status}")
            state = state.update(formatted_end_nodes=formatted_codes)
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
        
        # Track round completion by appending current batch info data
        if hasattr(collect_batch_info, "collected_data"):
            _round_completion_tracker.append(list(collect_batch_info.collected_data))

# --------------------------------------------------------------------------------------
# Example usage / quick test harness
if __name__ == "__main__":
    import os
    from icd_index import build_index_flat
    index = build_index_flat("icd10cm_tabular_2026.xml")
    configure_llm(
        model="gpt-4o-mini",
        max_tokens=200,
    )


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
        traversal_kind={"ROOT": None}, # The kind of batch to spawn: 'children', 'codeFirst', 'codeAlso', 'useAdditional', None
        batches={},
        visited=[],        # visited_set may be omitted; action will initialize it
        end_nodes=[],
        context="Diabetes Mellitus (not sure if Type 1 or Type 2) with ketoacidosis, GCS=3",
    )

    final_state = asyncio.run(run_until_converged(init))
    rprint("=== DONE ===")
    rprint(
        "ICD Code Traversal:\n" + 
        tabbed_output(final_state["visited"], final_state["traversal_depths"], final_state.get("traversal_kind", {}), index=index, tab="    ")
    )
    rprint("End nodes:", tuple(final_state["end_nodes"]) if len(final_state["end_nodes"]) >1 else f"({final_state["end_nodes"][0] or ''})")
