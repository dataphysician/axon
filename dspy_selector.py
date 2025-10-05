from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal
from rich import print as rprint
import re
import dspy

class NodeSig(dspy.Signature):
    """Select ALL correct candidate IDs for this node. Return [] if none."""
    context: str  = dspy.InputField(desc="Provenance / prior decisions")
    choices: str  = dspy.InputField(desc="One per line: 'candidate_id: text'")
    selected_ids: list[str] = dspy.OutputField(desc="Subset of IDs")
    reasoning: str = dspy.OutputField(desc="Brief primary reason for the selection")

def make_nodesig(
    *,
    batch_name: str,
    candidates: dict[str, str] | None = None,
    instruction: str | None = None,
    feedback: str | None = None,
) -> type[dspy.Signature]:
    ids_tuple: tuple[str, ...] = tuple(candidates.keys() or ())
    literal_list = list[Literal[*ids_tuple]] if ids_tuple else list[str]
    choices = "\n".join(f"{cid}: {txt.strip().replace('\n', ' ')}" for cid, txt in candidates.items()) 
    base_template = (
        "Given the provided context, choose only the MOST APPROPRIATE candidates from the list provided." +
        f"Select 0..N candidates from the following choices:\n{choices}\n" + 
        "Return [] if none or when no candidates were appropriate." + 
        "Then supply a brief reason for the primary selection/s."
    )
    doc = instruction.strip() if instruction else base_template
    if feedback:
        doc = f"{doc}\nNote: {feedback}"


    return type(
        batch_name,
        (dspy.Signature,),
        {
            "__doc__": doc,
            "context": dspy.InputField(desc="Provenance / prior decisions"),
            "choices": dspy.InputField(desc="One per line: 'candidate_id: text'"),
            "selected_ids": dspy.OutputField(desc="IDs of selected candidates"),
            "reasoning": dspy.OutputField(desc="Brief primary reason for the selection"),
            "__annotations__": {
                "context": str,
                "choices": str,
                "selected_ids": literal_list,  # narrowed list[Literal[...]] when provided
                "reasoning": str,          # <-- IMPORTANT: reasoning is a plain string
            },
        },
    )

class NodeProgram(dspy.Module):
    """Minimal DSPy program that selects child IDs for a node and explains why."""
    def __init__(
        self,
        *,
        batch_name: str,
        candidates: dict[str, str] | None = None,
        instruction: str | None = None,
        feedback: str | None = None,
        program_dir: Path | str = "dspy_programs",
        auto_load: bool = True,
    ):
        super().__init__()
        self._batch_name = batch_name
        self._program_dir: Path = Path(program_dir)

        Sig = make_nodesig(
            batch_name=batch_name,
            candidates=candidates,
            instruction=instruction,
            feedback=feedback,
        )
        self.decide = dspy.Predict(Sig)  # class-based Signature supported. :contentReference[oaicite:0]{index=0}

        if auto_load:
            self.load()

    def forward(self, *, context: str, candidates: dict[str, str]) -> tuple[list[str], str | None]:
        choices = "\n".join(f"{cid}: {txt.strip().replace('\n', ' ')}" for cid, txt in candidates.items()) # choices = "id: text" (one per line)
        result = self.decide(context=context, choices=choices)
        return (result.selected_ids or [], (result.reasoning or None))

    # whole-program load if saved with save_program=True. :contentReference[oaicite:1]{index=1}
    def load(self):
        path = self._program_dir / self._batch_name
        try:
            if path.exists():
                loaded = dspy.load(str(path))
                if hasattr(loaded, "decide"):
                    self.decide = loaded.decide
        except Exception:
            pass
        return self

    def save(self) -> None:
        path = self._program_dir / self._batch_name
        path.parent.mkdir(parents=True, exist_ok=True)
        dspy.save(self, str(path))


def ensure_dspy_configured() -> None:
    if getattr(dspy.settings, "lm", None) is None:
        # dspy.configure(lm=dspy.LM(provider="openai", model="gpt-4o", temperature=0, cache=False))

        import os
        dspy.configure(
            lm=dspy.LM(
                # model="cerebras/llama3.1-8b", 
                model="sambanova/Qwen3-32B", # gpt-oss-120b", # Meta-Llama-3.1-8B-Instruct", # Meta-Llama-3.1-8B-Instruct
                temperature=0, 
                cache=False,
            )
        )


# ---- Example -----------------------------------------------------------------
if __name__ == "__main__":
    from icd_index import build_index_flat
    index = build_index_flat("icd10cm_tabular_2026.xml")
    
    # ---- Tree helpers ------------------------------------------------------------
    def get_children(index: dict, node_id: str) -> dict[str, str]:
        """Return {child_id: label/text} or {} if leaf/missing."""
        try:
            children = index[node_id].get("children") or {}
            if isinstance(children, dict):
                return children
            return {cid: index[cid].get("label", cid) for cid in children}
        except Exception:
            return {}

    def select_children(index: dict, node_id: str, *, context: str) -> tuple[list[str], str | None]:
        """Run a NodeProgram on this node's children and return (selected_ids, reasoning)."""
        children = get_children(index, node_id)
        if not children:
            return ([], None)
        prog = NodeProgram(
            batch_name=f"{node_id}_-_children",
            candidates=children,
        )
        return prog(context=context, candidates=children)  # -> (ids, reasoning)

    def walk_and_print(index: dict, root_id: str, *, context: str) -> None:
        """
        DFS from root → leaves.
        Prints each node once (no duplicates). If the parent selected this node,
        its reasoning is printed one extra tab beneath the node line.
        """
        rprint("="*45)
        rprint("[red]ICD-10-CM Traversal - Parent to Child[/]")
        rprint("="*45)
        def dfs(node_id: str, depth: int, reason_from_parent: str | None) -> None:
            # print the node itself
            label = index.get(node_id, {}).get("label", node_id)
            rprint(f"{'\t' * depth}[yellow]Selection ({node_id}):[/] {label}")

            # print why the *parent* selected this node (if provided)
            if reason_from_parent:
                rprint(f"{'\t' * (depth)}[yellow]Reasoning ({node_id}):[/] {reason_from_parent}")

            # get children for this node
            children = get_children(index, node_id)
            if not children:
                return

            # ask LM which children to follow; one reasoning string for this selection step
            selected_ids, reasoning = select_children(index, node_id, context=context)
            if not selected_ids:
                return

            # recurse into each selected child; do NOT print the child here
            for cid in selected_ids:
                dfs(cid, depth + 1, reasoning)

        dfs(root_id, 0, None)

    # -----------------------------------------------------------------------------
    dspy.configure(lm=dspy.LM(provider="openai", model="gpt-4o", temperature=0, cache=True))
    ROOT = "ROOT"
    ctx = "Diabetes Mellitus (not sure if Type 1 or Type 2) with ketoacidosis, GCS=3"

    walk_and_print(index, ROOT, context=ctx)