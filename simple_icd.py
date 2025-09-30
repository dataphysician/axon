#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
import sys
import xml.etree.ElementTree as ET
from typing import Any, Awaitable, Callable
from xml.etree.ElementTree import Element

# ──────────────────────────────────────────────────────────────────────────────
# Types (PEP 585/604)
# ──────────────────────────────────────────────────────────────────────────────
NodeId = str
ContextPath = list[str]
Candidates = dict[str, str]          # id -> label
Metadata = dict[str, Any]

# ──────────────────────────────────────────────────────────────────────────────
# ICD code normalization (blocks, codes, comma-separated lists)
# ──────────────────────────────────────────────────────────────────────────────
NORMALIZED_CODE_RE = r"""
^
(?:
    [A-Z]\d[A-Z0-9]-[A-Z]\d[A-Z0-9]                # block-level range (3 chars - 3 chars)
  |
    [A-Z]\d[A-Z0-9](?:\.[A-Z0-9]{1,4})?-?          # standard code (3 chars + optional .+1–4, optional trailing -)
)
(?:\s*,\s*                                        # comma separator (optional spaces)
    (?:
        [A-Z]\d[A-Z0-9]-[A-Z]\d[A-Z0-9]            # another block-level range
      |
        [A-Z]\d[A-Z0-9](?:\.[A-Z0-9]{1,4})?-?      # another standard code
    )
)*
$
"""
code_re = re.compile(NORMALIZED_CODE_RE, re.VERBOSE)


def normalize_code(s: str) -> str:
    """Normalize a string to a valid ICD code/range/list; allow ROOT and Chapter_XX."""
    s = (s or "").strip()
    if not s:
        raise ValueError("Empty ICD code")
    if s.upper().startswith("ROOT"):
        return "ROOT"
    if s.startswith("Chapter_"):
        return s
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    if s.endswith("-"):
        s = s[:-1]
    if not code_re.match(s):
        raise ValueError(f"Invalid ICD code/range/list: {s!r}")
    return s


def to_block_form(code: str) -> str:
    """Ensure a code is in block/range form: single -> single-single (e.g., C50 -> C50-C50)."""
    if "-" in code:
        return code
    return f"{code}-{code}"


# ──────────────────────────────────────────────────────────────────────────────
# Metadata extraction (only known ICD rule-like tags)
# ──────────────────────────────────────────────────────────────────────────────
KNOWN_TAGS: set[str] = {
    "includes",
    "inclusionTerm",
    "excludes1",
    "excludes2",
    "codeFirst",
    "codeAlso",
    "useAdditionalCode",
    "sevenChrNote",
    "sevenChrDef",  # SPECIAL handling: list[dict] mapping extension @id -> text
}


def _local(tag: str) -> str:
    """Strip XML namespace, if any."""
    return tag.split("}", 1)[-1]


def _flat_text(e: Element) -> str:
    """Join all text inside the element (handles nested inline tags)."""
    return "".join(e.itertext()).strip()


def get_metadata(elem: Element) -> dict[str, str | list[str] | list[dict[str, str]]]:
    """
    Return ONLY the known ICD rule-like tags and their text content.
    - Normal tags -> str or list[str] if repeated.
    - sevenChrDef -> list[dict[str, str]] where each dict is {<extension id>: <text>}.
    """
    meta: dict[str, str | list[str] | list[dict[str, str]]] = {}

    for child in elem:
        key = _local(child.tag)
        if key not in KNOWN_TAGS:
            continue

        # Special structure for sevenChrDef:
        if key == "sevenChrDef":
            items: list[dict[str, str]] = []
            for ext in child.findall(".//extension"):
                ext_id = (ext.get("char") or "").strip()
                ext_text = _flat_text(ext)
                if ext_id:
                    items.append({ext_id: ext_text})
                else:
                    # If no id, still preserve the value under a placeholder key
                    items.append({"": ext_text})
            if not items:
                # Fallback if there were no <extension> children: keep whole text as a single unnamed item
                text = _flat_text(child)
                if text:
                    items.append({"": text})
            # Merge into meta (append if multiple sevenChrDef sections exist)
            if key in meta:
                assert isinstance(meta[key], list)
                meta[key].extend(items)  # type: ignore[arg-type]
            else:
                meta[key] = items
            continue

        # Default: string value for the tag
        val = _flat_text(child)

        if key in meta:
            if isinstance(meta[key], list):
                # already a list[str] (or list[dict], but sevenChrDef handled above)
                meta[key] = list(meta[key])  # type: ignore[assignment]
                meta[key].append(val)        # type: ignore[arg-type]
            else:
                # promote single str to list[str]
                meta[key] = [meta[key], val]  # type: ignore[list-item]
        else:
            meta[key] = val

    return meta


# ──────────────────────────────────────────────────────────────────────────────
# Flat index construction
# ──────────────────────────────────────────────────────────────────────────────
def _add_node(
    index: dict[str, dict[str, object]],
    *,
    node_id: NodeId,
    node_label: str,
    parent: NodeId | None,
    level: int,
    metadata: dict[str, str | list[str] | list[dict[str, str]]],
) -> None:
    """Insert/merge a node record into the flat index."""
    if node_id in index:
        rec = index[node_id]
    else:
        rec = {
            "node_id": node_id,
            "node_label": node_label,
            "parent": parent,
            "children": [],   # list[NodeId]
            "level": level,
            "metadata": {},
        }
        index[node_id] = rec

    if node_label:
        rec["node_label"] = node_label

    if rec.get("parent") is None and parent is not None:
        rec["parent"] = parent

    existing: dict[str, Any] = rec.get("metadata", {})  # type: ignore[assignment]
    for k, v in (metadata or {}).items():
        if k not in existing:
            existing[k] = v
        else:
            a = existing[k]
            if k == "sevenChrDef":
                # Always merge as list[list[dict]] → list[dict]
                if isinstance(a, list) and isinstance(v, list):
                    existing[k] = [*a, *v]
                elif isinstance(a, list):
                    existing[k] = [*a, v]
                elif isinstance(v, list):
                    existing[k] = [a, *v]
                else:
                    existing[k] = [a, v]
                continue

            # For other keys: merge str/list[str]
            if isinstance(a, list) and isinstance(v, list):
                existing[k] = [*a, *v]
            elif isinstance(a, list):
                existing[k] = [*a, v]
            elif isinstance(v, list):
                existing[k] = [a, *v]
            else:
                existing[k] = [a, v]
    rec["metadata"] = existing


def _link_parent_child(index: dict[str, dict[str, object]], parent: NodeId, child: NodeId) -> None:
    """Ensure the parent's children list contains the child."""
    if parent not in index:
        index[parent] = {
            "node_id": parent,
            "node_label": parent,
            "parent": None,
            "children": [],
            "level": 0,
            "metadata": {},
        }
    kids: list[str] = index[parent]["children"]  # type: ignore[assignment]
    if child not in kids:
        kids.append(child)


def _safe_code_from_diag(diag: Element) -> NodeId:
    """Prefer <name>; fall back to stable synthetic id if not normalizable."""
    name = (diag.findtext("name") or "").strip()
    if name:
        try:
            return normalize_code(name)
        except ValueError:
            pass
    desc = (diag.findtext("desc") or "").strip()
    data = (name + "::" + desc).encode("utf-8")
    h = hashlib.sha1(data).hexdigest()[:12]
    return f"diag::{h}"


def _add_diag_tree(
    index: dict[str, dict[str, object]],
    diag: Element,
    parent: NodeId,
    start_level: int,
) -> None:
    """Attach a <diag> node (and its diag children) under parent with proper metadata."""
    node_id = _safe_code_from_diag(diag)
    node_label = (diag.findtext("desc") or node_id).strip()
    _add_node(
        index,
        node_id=node_id,
        node_label=node_label,
        parent=parent,
        level=start_level,
        metadata=get_metadata(diag),
    )
    _link_parent_child(index, parent, node_id)
    for sub in diag.findall("diag"):
        _add_diag_tree(index, sub, node_id, start_level=start_level + 1)


def build_index_flat(xml_path: str) -> dict[str, dict[str, object]]:
    """
    Build a flat dictionary:
      node_id -> {
        node_id: str, node_label: str, parent: str|None,
        children: list[str], level: int, metadata: dict
      }
    """
    et = ET.parse(xml_path)
    root = et.getroot()
    index: dict[str, dict[str, object]] = {}

    # ROOT
    _add_node(index, node_id="ROOT", node_label="ICD-10-CM Root", parent=None, level=0, metadata={})

    # Chapters
    for chapter_i, chapter in enumerate(root.findall("chapter"), start=1):
        chap_num = f"{chapter_i:02d}"
        chap_id = f"Chapter_{chap_num}"
        chap_label = (chapter.findtext("desc") or f"Chapter {chap_num}").strip()
        _add_node(index, node_id=chap_id, node_label=chap_label, parent="ROOT", level=1, metadata=get_metadata(chapter))
        _link_parent_child(index, "ROOT", chap_id)

        # Formal blocks
        formal_blocks: set[str] = set()
        sidx = chapter.find("sectionIndex")
        if sidx is not None:
            for sref in sidx.findall("sectionRef"):
                raw = (sref.get("id") or "").strip()
                if not raw:
                    continue
                try:
                    bid = normalize_code(raw)
                except ValueError:
                    continue
                bid = to_block_form(bid)
                blabel = (sref.text or "").strip() or bid
                _add_node(index, node_id=bid, node_label=blabel, parent=chap_id, level=2, metadata=get_metadata(sref))
                _link_parent_child(index, chap_id, bid)
                formal_blocks.add(bid)

        # Recurse into section/diag (skip section node itself)
        for section in chapter.findall("section"):
            sec_id_raw = (section.get("id") or "").strip()
            parent_for_diags = chap_id
            if sec_id_raw:
                try:
                    sec_norm = to_block_form(normalize_code(sec_id_raw))
                    if sec_norm in formal_blocks:
                        parent_for_diags = sec_norm
                except ValueError:
                    pass

            for diag in section.findall("diag"):
                _add_diag_tree(index, diag, parent_for_diags, start_level=3)

    return index


# ──────────────────────────────────────────────────────────────────────────────
# Phase 0 contracts (pure, no Orchestrator Frameworks yet)
# ──────────────────────────────────────────────────────────────────────────────
async def enumerate_candidates(index: dict[str, dict[str, object]], node_id: NodeId) -> Candidates:
    """Return {child_id: child_label} for a node."""
    rec = index.get(node_id)
    if not rec:
        return {}
    kids: list[str] = rec.get("children", [])  # type: ignore[assignment]
    out: dict[str, str] = {}
    for cid in kids:
        child = index.get(cid)
        out[cid] = (child.get("node_label") if child else cid)  # type: ignore[union-attr]
    return out


class Selector:
    """Randomized candidate selector that prints its decision and feedback."""

    def __init__(self, *, seed: int | None = None) -> None:
        import random
        self._rng = random.Random(seed)

    async def __call__(
        self,
        *,
        context_path: ContextPath | None,
        candidates: Candidates,
        feedback: str = "",
        k_choices: tuple[int, ...] = (1, 2),
    ) -> list[NodeId]:
        ids: list[NodeId] = list(candidates.keys())
        path_str = " > ".join(context_path or [])
        print(f"[Selector] path={path_str if path_str else 'ROOT'}")
        print(f"  candidates={ids}")
        fb = (feedback or "").strip()
        if not fb:
            fb = "avoid loops"
        print(f"  feedback='{fb}'")

        if not ids:
            print("  chosen=[]")
            return []

        visited = set(context_path or [])
        head = (context_path or [None])[-1]
        ids = [i for i in ids if i != head and i not in visited]
        if not ids:
            print("  chosen=[]  # (all blocked by loop-avoidance)")
            return []

        if len(ids) == 1:
            print(f"  chosen={ids}")
            return [ids[0]]

        k = min(self._rng.choice(k_choices), len(ids))
        chosen = self._rng.sample(ids, k)
        print(f"  chosen={sorted(chosen)}")
        return chosen


async def terminate_node(node_id: NodeId, *, state: dict[str, Any]) -> bool:
    """Default terminator: never force-stop; traversal stops naturally on leaves."""
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Async fan-out traversal (with safety max-depth)
# ──────────────────────────────────────────────────────────────────────────────
async def traverse_fanout(
    *,
    index: dict[str, dict[str, object]],
    root_id: NodeId,
    selector: Callable[..., Awaitable[list[NodeId]]],
    terminator: Callable[..., Awaitable[bool]] = terminate_node,
    max_parallel: int = 200,
    max_depth: int = 8,
) -> dict[str, Any]:
    terminal_nodes: list[str] = []
    all_paths: list[ContextPath] = []
    deepest_path: ContextPath | None = None
    sem = asyncio.Semaphore(max_parallel)

    async def mark_terminal(path: ContextPath) -> None:
        node = path[-1]
        if path not in all_paths:
            all_paths.append(list(path))
        if node not in terminal_nodes:
            terminal_nodes.append(node)

    async def walk(node_id: NodeId, path: ContextPath) -> None:
        nonlocal deepest_path
        if deepest_path is None or len(path) > len(deepest_path):
            deepest_path = list(path)

        # depth safety
        if len(path) - 1 >= max_depth:
            await mark_terminal(path)
            return

        if await terminator(node_id, state={"path": path}):
            await mark_terminal(path)
            return

        enum = await enumerate_candidates(index, node_id)
        if not enum:
            await mark_terminal(path)
            return

        chosen = await selector(context_path=path, candidates=enum, feedback="broad search | avoid loops")
        if not chosen:
            await mark_terminal(path)
            return

        async with sem:
            await asyncio.gather(*[walk(child, [*path, child]) for child in chosen])

    await walk(root_id, [root_id])

    return {
        "terminal_nodes": terminal_nodes,
        "all_paths": all_paths,
        "deepest_path": deepest_path or [],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node inspection
# ──────────────────────────────────────────────────────────────────────────────
def get_node(index: dict[str, dict[str, object]], node_id: str) -> dict[str, object] | None:
    return index.get(node_id)


def print_node(index: dict[str, dict[str, object]], node_id: str) -> None:
    rec = get_node(index, node_id)
    if not rec:
        print(f"[not found] {node_id}")
        return
    print(f"node_id:    {rec['node_id']}")
    print(f"label:      {rec['node_label']}")
    print(f"parent:     {rec['parent']}")
    print(f"level:      {rec['level']}")
    print(f"children:   {rec['children']}")
    print("metadata:")
    meta: dict[str, Any] = rec.get("metadata", {})  # type: ignore[assignment]
    if not meta:
        print("  <empty>")
    else:
        for k, v in meta.items():
            print(f"  {k}: {v}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI demo
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="ICD Phase 0: flat index + async parallel fan-out demo")
    p.add_argument("--xml", default="icd10cm_tabular_2026.xml", help="Path to icd10cm_tabular_<year>.xml")
    p.add_argument("--start", default="ROOT", help="Start node id (default: ROOT)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for selector")
    p.add_argument("--max-parallel", type=int, default=200, help="Max concurrent fan-out tasks")
    p.add_argument("--max-depth", type=int, default=8, help="Max traversal depth (edges) from start")
    p.add_argument("--print-node", dest="print_node_id", default="S52", help="Node id to pretty-print after traversal")
    args = p.parse_args()

    # Build index
    index = build_index_flat(args.xml)

    # Run async fan-out traversal
    sel = Selector(seed=args.seed)
    result = asyncio.run(
        traverse_fanout(
            index=index,
            root_id=args.start,
            selector=sel,
            terminator=terminate_node,
            max_parallel=args.max_parallel,
            max_depth=args.max_depth,
        )
    )

    # Pretty print traversal results
    def show(pth: ContextPath) -> str:
        return ", ".join(pth)

    print("\n=== RESULTS ===")
    print("TERMINATED NODES:", result["terminal_nodes"])
    print("TERMINATED PATHS:", [show(p) for p in result["all_paths"]])
    print("DEEPEST PATH:", show(result["deepest_path"]))

    # Optional node inspection
    if args.print_node_id:
        print("\n=== NODE INSPECTION ===")
        print_node(index, args.print_node_id)


if __name__ == "__main__":
    main()
