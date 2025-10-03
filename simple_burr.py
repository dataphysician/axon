# deterministic_burr.py
from __future__ import annotations

from typing import Any, Literal, Hashable, TypeVar
from pydantic import BaseModel, Field, field_validator

from burr.core import ApplicationBuilder, State, action, default, expr
from burr.integrations.pydantic import PydanticTypingSystem
from burr.core.parallelism import MapStates, RunnableGraph

# ---- Import your ICD graph definitions/builders from flat_icd.py ----
from simple_icd import GenericNode, ICDCode, Index, build_index_flat


# ─────────────────────────────────────────────────────────────────────
# Fixed types for traversal
# ─────────────────────────────────────────────────────────────────────

BatchId = Literal["code_first", "children", "code_also", "use_additional"]
BATCH_ORDER: list[BatchId] = ["code_first", "children", "code_also", "use_additional"]
FeedbackType = Literal["positive", "negative", "default"]


# ─────────────────────────────────────────────────────────────────────
# Burr-side Pydantic models (normalized by validators)
# ─────────────────────────────────────────────────────────────────────

class NodeData(BaseModel):
    """
    Graph node (Burr-side). `metadata` drives batches & feedback.

    Validators normalize:
      - parent: dict[id,str] | str | None  -> str | None (first key if dict)
      - children: dict[id,str] | list[str] | None -> list[str] (ids only)
      - metadata: ensure dict[str, Any]
    """
    node_id: str
    label: str
    depth: int
    parent: str | None
    children: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("parent", mode="before")
    @classmethod
    def _coerce_parent(cls, v: Any) -> str | None:
        if v is None or isinstance(v, str):
            return v
        if isinstance(v, dict) and v:
            return str(next(iter(v.keys())))
        return None

    @field_validator("children", mode="before")
    @classmethod
    def _coerce_children(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, dict):
            return [str(k) for k in v.keys()]
        if isinstance(v, list):
            return [str(i) for i in v]
        return [str(v)]

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_meta(cls, v: Any) -> dict[str, Any]:
        return dict(v or {})


class CandidateBatch(BaseModel):
    """
    One batch to run selection against for a given `current_node` (the queue key).
    Validators guarantee:
      - candidates always a list[str]
      - feedback always has keys {positive, negative, default} with list[str] values
    """
    path: list[str]                     # root → current_node (inclusive)
    batch: BatchId
    candidates: list[str] = Field(default_factory=list)
    feedback: dict[FeedbackType, list[str]] = Field(default_factory=dict)

    @field_validator("candidates", mode="before")
    @classmethod
    def _coerce_candidates(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(i) for i in v]
        return [str(v)]

    @field_validator("feedback", mode="before")
    @classmethod
    def _normalize_feedback(cls, v: Any) -> dict[FeedbackType, list[str]]:
        base: dict[FeedbackType, list[str]] = {
            "positive": [],
            "negative": [],
            "default": [],
        }
        if not v:
            return base
        if isinstance(v, dict):
            for k in ("positive", "negative", "default"):
                val = v.get(k)
                if val is None:
                    continue
                if isinstance(val, list):
                    base[k].extend([str(i) for i in val])
                else:
                    base[k].append(str(val))
        else:
            base["default"].append(str(v))
        return base


class StopNode(BaseModel):
    """Terminal result for reporting."""
    node: str
    path: list[str]
    reason: str   # 'leaf' | 'depth_cap' | 'no_candidates' | 'no_selection'


class AppState(BaseModel):
    """
    Minimal, typed, normalized state.

    - index: the flat domain index (node_id -> NodeData)
    - root: entry node id
    - queue: current wave work, keyed by current_node -> list[CandidateBatch]
    - stops: accumulated terminal results
    - current_node/current_batch/chosen_nodes: per-MapStates scratch
    """
    index: dict[str, NodeData]
    root: str

    queue: dict[str, list[CandidateBatch]] = Field(default_factory=dict)
    stops: list[StopNode] = Field(default_factory=list)

    max_depth: int = 999

    current_node: str | None = None
    current_batch: CandidateBatch | None = None
    chosen_nodes: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Adapters: dataclass (GenericNode[ICDCode]) -> Pydantic (NodeData map)
# ─────────────────────────────────────────────────────────────────────

T = TypeVar("T", bound=Hashable)

def node_to_pydantic(n: GenericNode[ICDCode]) -> NodeData:
    # NodeData validators handle parent/children/metadata shape for us.
    return NodeData(
        node_id=str(n.node_id),
        label=n.node_label,
        depth=n.depth,
        parent=n.parent,           # validator normalizes
        children=n.children,       # validator normalizes
        metadata=n.metadata,       # validator normalizes
    )

def index_to_flat_map(index: Index) -> dict[str, NodeData]:
    return {str(k): node_to_pydantic(v) for k, v in index.items()}


# ─────────────────────────────────────────────────────────────────────
# Small helper
# ─────────────────────────────────────────────────────────────────────

def _list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    return [str(x)]


# ─────────────────────────────────────────────────────────────────────
# Actions (Burr) — all single-step; MapStates provides async parallelism
# ─────────────────────────────────────────────────────────────────────

@action(reads=["index", "root", "queue", "stops", "max_depth"], writes=["queue", "stops"])
def load_node(state: State) -> State:
    """
    Open nodes in `queue` (or seed root if empty), create per-node CandidateBatch items.
    - Apply feedback ONLY to the `children` batch of the *current* node.
    - Rule batches do NOT inherit the current node's feedback.
    - All batches carry the same base path (selection extends it).
    - If all batches are empty for a node, emit a StopNode.
    """
    index: dict[str, NodeData] = state["index"]
    queue: dict[str, list[CandidateBatch]] = dict(state.get("queue") or {})
    stops: list[StopNode] = list(state.get("stops") or [])

    # Seed first wave with a path placeholder
    if not queue:
        queue = {state["root"]: [CandidateBatch(path=[state["root"]], batch="children")]}

    next_queue: dict[str, list[CandidateBatch]] = {}

    for node_id, incoming in list(queue.items()):
        node = index[node_id]
        base_path = incoming[0].path if incoming else [node_id]

        # depth cap
        if node.depth >= state["max_depth"]:
            stops.append(StopNode(node=node_id, path=base_path, reason="depth_cap"))
            continue

        meta = node.metadata  # already normalized

        # ---- Build feedback ONLY for the children batch ----
        fb_dict = meta.get("feedback") or {}
        children_feedback = {
            "positive": [*(_list(meta.get("includes"))), *(_list(meta.get("inclusionTerm"))), *(_list(fb_dict.get("positive")))],
            "negative": [*(_list(meta.get("excludes2"))), *(_list(fb_dict.get("negative")))],
            "default":  [*(_list(fb_dict.get("default")))],
        }

        # ---- Children batch (override if meta['children'] present) ----
        children_vals = meta.get("children", node.children)
        children_batch = CandidateBatch(
            path=base_path,
            batch="children",
            candidates=children_vals,     # validator -> list[str]
            feedback=children_feedback,   # validator -> 3 keys with list[str]
        )

        # ---- Rule batches (NO feedback inherited) ----
        batches: list[CandidateBatch] = [children_batch]
        for rule in ("code_first", "code_also", "use_additional"):
            if rule in meta:
                batches.append(
                    CandidateBatch(
                        path=base_path,                 # same path; reducer extends on select
                        batch=rule,                     # type: ignore[arg-type]
                        candidates=meta.get(rule),      # validator coerces to list[str]
                        # feedback intentionally omitted for rule batches
                    )
                )

        # ---- Terminal vs work ----
        if all(len(b.candidates) == 0 for b in batches):
            reason = "leaf" if len(node.children) == 0 and "children" not in meta else "no_candidates"
            stops.append(StopNode(node=node_id, path=base_path, reason=reason))
            continue

        kept = [b for b in batches if b.candidates]
        if kept:
            next_queue[node_id] = kept
        else:
            stops.append(StopNode(node=node_id, path=base_path, reason="no_candidates"))

    return state.update(queue=next_queue, stops=stops)

import random
# Non-streaming selection action for ONE (current_node, current_batch)
@action(reads=["current_node", "current_batch"], writes=["chosen_nodes"])
def select_candidates(state: State) -> State:
    """
    Choose 0..N candidates for this (current_node, batch).
    Replace the stub with an LLM/HITL policy if desired.
    """
    b: CandidateBatch = state["current_batch"]
    
    # chosen: list[str] = b.candidates[:2] if len(b.candidates)>3 else b.candidates[:1]  # deterministic stub: pick the first
    _rng = random.Random()
    if len(b.candidates) == 1:
        chosen = [b.candidates[0]]
    k: int = _rng.choice([1, 2])
    k = min(k, len(b.candidates))
    _rng.shuffle(b.candidates)
    chosen = sorted(b.candidates[:k])
    print(b.path)
    return state.update(chosen_nodes=chosen)


@action(reads=["queue", "stops"], writes=[])
def done(state: State) -> State:
    """Terminal action; app halts when transitions route here."""
    return state


# ─────────────────────────────────────────────────────────────────────
# Parallel MapStates: map select_candidates, reduce into next queue
# ─────────────────────────────────────────────────────────────────────

class SelectPhase(MapStates):
    """Map over all CandidateBatch items across the queue; reduce to next queue or stops."""

    @property
    def reads(self) -> list[str]:
        return ["index", "queue", "stops", "current_node", "current_batch", "chosen_nodes"]

    @property
    def writes(self) -> list[str]:
        return ["queue", "stops", "current_node", "current_batch", "chosen_nodes"]

    def action(self, state, inputs: dict[str, object]) -> RunnableGraph:
        # single-step, non-streaming action
        return RunnableGraph.create(select_candidates)

    def states(self, state, context, inputs: dict[str, object]):
        queue: dict[str, list[CandidateBatch]] = state["queue"]
        for node_id, batches in queue.items():
            for b in batches:
                yield state.update(current_node=node_id, current_batch=b, chosen_nodes=[])

    def reduce(self, state, states):
        stops: list[StopNode] = list(state.get("stops") or [])
        index: dict[str, NodeData] = state["index"]

        any_sel: dict[str, bool] = {}
        path_by_node: dict[str, list[str]] = {}
        next_entries: list[tuple[str, list[str]]] = []

        for s in states:
            node_id: str = s["current_node"]
            b: CandidateBatch = s["current_batch"]
            chosen: list[str] = s.get("chosen_nodes") or []

            any_sel.setdefault(node_id, False)
            if chosen:
                any_sel[node_id] = True

            path_by_node.setdefault(node_id, b.path[:])

            for n in chosen:
                # optional cycle guard
                if n in b.path:
                    continue
                next_entries.append((n, b.path + [n]))

        # Emit 'no_selection' stops where a node had no picks across all its batches
        for node_id, had in any_sel.items():
            if not had:
                stops.append(StopNode(node=node_id, path=path_by_node[node_id], reason="no_selection"))

        # Deduplicate by node; seed next queue with path-carrying placeholders
        seen: set[str] = set()
        next_queue: dict[str, list[CandidateBatch]] = {}
        for nid, path in next_entries:
            if nid not in seen and nid in index:
                seen.add(nid)
                next_queue[nid] = [CandidateBatch(path=path, batch="children")]

        return state.update(
            queue=next_queue,
            stops=stops,
            current_node=None,
            current_batch=None,
            chosen_nodes=[],
        )


# ─────────────────────────────────────────────────────────────────────
# Builder (ApplicationBuilder + typing before state)
# ─────────────────────────────────────────────────────────────────────

def build_app(index_map: dict[str, NodeData], root: str, *, max_depth: int = 999):
    """Construct an async Burr application with typed state and the simplified abstractions."""
    return (
        ApplicationBuilder()
        .with_actions(
            load_node=load_node,
            select_phase=SelectPhase(),
            done=done,
        )
        .with_transitions(
            ("load_node", "done", expr("len(queue) == 0")),          # no work → finished
            ("load_node", "select_phase", default),                  # select over this wave
            ("select_phase", "load_node", expr("len(queue) > 0")),  # next wave to open
            ("select_phase", "done", default),                       # traversal finished
        )
        .with_typing(PydanticTypingSystem(AppState))                                # typing FIRST
        .with_state(AppState(index=index_map, root=root, max_depth=max_depth))      # then typed instance
        .with_entrypoint("load_node")
        .abuild()
    )


# ─────────────────────────────────────────────────────────────────────
# Example run
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    # 1) Build your flat ICD index (dataclass world)
    raw_index: Index = build_index_flat("icd10cm_tabular_2026.xml")

    # 2) Adapt to Burr’s Pydantic NodeData flat map
    index_map: dict[str, NodeData] = index_to_flat_map(raw_index)
    root_id = "ROOT"

    async def main():
        app = await build_app(index_map, root=root_id, max_depth=999)
        # Run until 'done'
        _, _, state = await app.arun(halt_after=["done"])
        typed = AppState.model_validate(state)

        from rich import print as rprint
        print("Stops:")
        for s in typed.stops:
            rprint(" -", s.path[-1])
        rprint(typed.model_dump().keys())
        rprint(typed.model_dump(exclude={"index"}))
    asyncio.run(main())