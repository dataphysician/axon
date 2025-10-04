#!/usr/bin/env python3
from __future__ import annotations

import re
from rich import print as rprint
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from collections.abc import Hashable
from typing import TypeVar

# ──────────────────────────────────────────────────────────────────────────────
# Generic type parameter
# ──────────────────────────────────────────────────────────────────────────────
T = TypeVar("T", bound=Hashable)

# ──────────────────────────────────────────────────────────────────────────────
# Library-style dataclass used ONLY when reporting a node (not stored in index)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class GenericNode[T]:
    """
    Reusable node record used only for presentation/reporting.

    T: the node-id type (must be Hashable so it can serve as a dict key)
    """
    node_id: T
    node_label: str
    parent: dict[T, str] | None = None          # single-entry mapping or None (for root)
    children: dict[T, str] = field(default_factory=dict)
    depth: int = 0                               # 0: root, 1: chapter, 2: block, 3+ : diag depth
    metadata: dict[str, str | list[str] | dict[str, str | list[str]]] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return not self.children

# ──────────────────────────────────────────────────────────────────────────────
# Node/index dict types (pure-dict index)
# ──────────────────────────────────────────────────────────────────────────────
# A Node is a dict with string keys and values that are str | list[str] | dict.
type NodeDict = dict[str, str | list[str] | int | dict]
type Index = dict[str, NodeDict]

# ──────────────────────────────────────────────────────────────────────────────
# Code Patterns
# ──────────────────────────────────────────────────────────────────────────────
_DIAG_CODE_RE = re.compile(r"^[A-Z][A-Z0-9]{2}(?:\.[A-Z0-9]{1,4})?$", re.IGNORECASE)
_CODE_INPUT_RE = re.compile(
    r"""^(?:
        (?P<root>ROOT) | 
        (?P<chapter>Chapter_(?P<chapnum>0[1-9]|1[0-9]|2[0-2])) | 
        (?P<range>(?P<left>[A-Z][A-Z0-9]{2})-(?P<right>[A-Z][A-Z0-9]{2})) | 
        (?P<diag>(?P<d1>[A-Z][A-Z0-9]{2})(?:\.(?P<d2>[A-Z0-9]{1,4}))?)
    )$""",
    re.I | re.VERBOSE,
)

# ──────────────────────────────────────────────────────────────────────────────
# ICD Code canonicalizer
# ──────────────────────────────────────────────────────────────────────────────
class ICDCode(str):
    __slots__ = ()
    def __new__(cls, s: str) -> ICDCode:
        if not isinstance(s, str):
            raise TypeError("ICDCode must be constructed from a string")
        m = _CODE_INPUT_RE.fullmatch(s.strip())
        if not m:
            raise ValueError(
                "Invalid ICD code. Expecting: 'ROOT' or 'Chapter_01' or standard ICD code like "
                "'A00-B99', 'C01', 'E11.621' or 'S06.0X1A'."
            )
        g = m.groupdict()
        if g["root"]:
            return str.__new__(cls, "ROOT")
        if g["chapter"]:
            return str.__new__(cls, f"Chapter_{g['chapnum']}")
        if g["range"]:
            return str.__new__(cls, f"{g['left'].upper()}-{g['right'].upper()}")
        d1 = g["d1"].upper()
        d2 = g.get("d2")
        return str.__new__(cls, d1 if not d2 else f"{d1}.{d2.upper()}")

# ──────────────────────────────────────────────────────────────────────────────
# Code Input normalization
# ──────────────────────────────────────────────────────────────────────────────
def normalize_code(s: str) -> str:
    s = (s or "").strip()
    if not s:
        raise ValueError("Empty diagnosis code")
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    if s.endswith("-."):
        s = s[:-2]
    if s.endswith("-"):
        s = s[:-1]
    if s.endswith("."):
        s = s[:-1]
    su = s.upper()
    if "-" in su or "," in su or ";" in su or "~" in su:
        raise ValueError(f"Expected a single diagnosis code, got: {s!r}")
    if not _DIAG_CODE_RE.fullmatch(su):
        raise ValueError(f"Invalid diagnosis code: {s!r}")
    return su

def to_block_form(code: str) -> str:
    return code if "-" in code else f"{code}-{code}"

# ──────────────────────────────────────────────────────────────────────────────
# Metadata extraction (NO concatenation)
#   - Non-7th char tags: str or list[str]
#   - sevenChrDef: dict[str, str | list[str]]  (char -> text or list of texts)
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
    "sevenChrDef",
}

def _local(tag: str) -> str:
    return tag.split("}", 1)[-1]

def _lines(elem: ET.Element) -> list[str]:
    out: list[str] = []
    for chunk in elem.itertext():
        for line in str(chunk).splitlines():
            s = line.strip()
            if s:
                out.append(s)
    return out

def _merge_listlike(a: str | list[str], b: str | list[str]) -> list[str]:
    la = [a] if isinstance(a, str) else list(a)
    lb = [b] if isinstance(b, str) else list(b)
    return [*la, *lb]

def _merge_meta_value(
    cur: str | list[str] | dict[str, str | list[str]],
    new: str | list[str] | dict[str, str | list[str]],
) -> str | list[str] | dict[str, str | list[str]]:
    # strings/lists merge to list; dicts merge per-key, list-merging for collisions
    if isinstance(cur, dict) and isinstance(new, dict):
        out: dict[str, str | list[str]] = dict(cur)
        for k, v in new.items():
            if k in out:
                out[k] = _merge_meta_value(out[k], v)  # type: ignore[arg-type]
            else:
                out[k] = v
        return out
    if isinstance(cur, (str, list)) and isinstance(new, (str, list)):
        return _merge_listlike(cur, new)
    # fallback: prefer new
    return new  # type: ignore[return-value]

def get_metadata(elem: ET.Element) -> dict[str, str | list[str] | dict[str, str | list[str]]]:
    meta: dict[str, str | list[str] | dict[str, str | list[str]]] = {}
    for child in elem:
        key = _local(child.tag)
        if key not in KNOWN_TAGS:
            continue

        if key == "sevenChrDef":
            mapping: dict[str, str | list[str]] = {}
            found_ext = False
            for ext in child.findall("./extension"):
                found_ext = True
                ch = (ext.get("char") or "").strip()
                if not ch:
                    continue
                lines = _lines(ext)
                if not lines:
                    continue
                val: str | list[str] = lines[0] if len(lines) == 1 else lines
                if ch in mapping:
                    mapping[ch] = _merge_meta_value(mapping[ch], val)  # type: ignore[arg-type]
                else:
                    mapping[ch] = val
            if not found_ext:
                lines = _lines(child)
                if lines:
                    mapping["_text"] = lines[0] if len(lines) == 1 else lines
            if "sevenChrDef" in meta and isinstance(meta["sevenChrDef"], dict):
                meta["sevenChrDef"] = _merge_meta_value(meta["sevenChrDef"], mapping)  # type: ignore[arg-type]
            else:
                meta["sevenChrDef"] = mapping
            continue

        notes = child.findall("./note")
        gathered: list[str] = []
        if notes:
            for n in notes:
                gathered.extend(_lines(n))
        else:
            gathered = _lines(child)

        if not gathered:
            continue
        meta[key] = gathered[0] if len(gathered) == 1 else gathered
    return meta

# ──────────────────────────────────────────────────────────────────────────────
# Index helpers (pure dicts)
# ──────────────────────────────────────────────────────────────────────────────
def _ensure_node(index: Index, node_id: str) -> NodeDict:
    if node_id not in index:
        index[node_id] = {
            "id": node_id,                 # str
            "label": node_id,              # str
            "depth": int(0),               # int
            "parent": {},                  # dict[str, str]
            "children": {},                # dict[str, str]
            "metadata": {},                # dict[str, str|list[str]|dict]
        }
    return index[node_id]

def _add_node(
    index: Index,
    *,
    node_id: str,
    node_label: str,
    parent: str | None,
    depth: int,
    metadata: dict[str, str | list[str] | dict[str, str | list[str]]],
) -> None:
    rec = _ensure_node(index, node_id)
    rec["label"] = node_label or str(rec.get("label") or node_id)
    rec["depth"] = depth

    if parent:
        _ensure_node(index, parent)
        parent_map = rec.get("parent")
        if not parent_map:  # None or empty -> set single-entry map
            rec["parent"] = {parent: str(index[parent]["label"])}

    # merge metadata
    if metadata:
        cur_meta = rec.get("metadata") or {}
        if cur_meta:
            rec["metadata"] = _merge_meta_value(cur_meta, metadata)  # type: ignore[arg-type]
        else:
            rec["metadata"] = metadata

def _link_parent_child(index: Index, parent: str, child: str) -> None:
    _ensure_node(index, parent)
    _ensure_node(index, child)
    # update parent's children
    children = index[parent].get("children") or {}
    children[child] = str(index[child]["label"])
    index[parent]["children"] = children
    # set child's parent if missing
    parent_map = index[child].get("parent")
    if not parent_map:
        index[child]["parent"] = {parent: str(index[parent]["label"])}

def _add_diag_tree(index: Index, diag: ET.Element, parent: str, start_depth: int) -> None:
    node_id = normalize_code((diag.findtext("name") or "").strip())
    node_label = (diag.findtext("desc") or node_id).strip()
    _add_node(
        index,
        node_id=node_id,
        node_label=node_label,
        parent=parent,
        depth=start_depth,
        metadata=get_metadata(diag),
    )
    _link_parent_child(index, parent, node_id)
    for sub in diag.findall("diag"):
        _add_diag_tree(index, sub, node_id, start_depth=start_depth + 1)

def _section_has_diag(section_elem: ET.Element) -> bool:
    for e in section_elem.iter():
        if _local(e.tag) == "diag":
            return True
    return False

# ──────────────────────────────────────────────────────────────────────────────
# Builder (returns pure-dict Index)
# ──────────────────────────────────────────────────────────────────────────────
def build_index_flat(xml_path: str) -> Index:
    et = ET.parse(xml_path)
    root = et.getroot()
    index: Index = {}

    # ROOT
    index["ROOT"] = {
        "id": "ROOT",
        "label": "ICD-10-CM",
        "depth": 0,
        "parent": {},
        "children": {},
        "metadata": {},
    }

    # Chapter
    for chapter_i, chapter in enumerate(root.findall("chapter"), start=1):
        chap_num = f"{chapter_i:02d}"
        chap_id = f"Chapter_{chap_num}"
        chap_label = (chapter.findtext("desc") or f"Chapter {chap_num}").strip()

        _add_node(
            index,
            node_id=chap_id,
            node_label=chap_label,
            parent="ROOT",
            depth=1,
            metadata=get_metadata(chapter),
        )
        _link_parent_child(index, "ROOT", chap_id)

        # Block
        for section in chapter.findall("section"):
            sec_id_raw = (section.get("id") or "").strip()
            if not sec_id_raw:
                continue
            block_id = to_block_form(sec_id_raw)
            if not _section_has_diag(section):
                continue
            block_label = (section.findtext("desc") or "").strip() or block_id

            _add_node(
                index,
                node_id=block_id,
                node_label=block_label,
                parent=chap_id,
                depth=2,
                metadata=get_metadata(section),
            )
            _link_parent_child(index, chap_id, block_id)

            # ICD Codes
            for diag in section.findall("diag"):
                _add_diag_tree(index, diag, block_id, start_depth=3)

    return index

# ──────────────────────────────────────────────────────────────────────────────
# Reporting helpers: convert dict → dataclass on demand
# ──────────────────────────────────────────────────────────────────────────────
def parse_to_dataclass(index: Index, node_id: str) -> GenericNode[str] | None:
    raw = index.get(node_id)
    if not raw:
        return None
    depth_val = raw.get("depth", 0)
    return GenericNode[str](
        node_id=str(raw.get("id") or node_id),
        node_label=str(raw.get("label") or node_id),
        parent=raw.get("parent") or None,  # dict[str, str] | None
        children=raw.get("children") or {},  # dict[str, str]
        depth=int(depth_val) if not isinstance(depth_val, int) else depth_val,
        metadata=raw.get("metadata") or {},
    )

def print_node(index: Index, node_id: str) -> None:
    nid = str(ICDCode(str(node_id)))
    rec = parse_to_dataclass(index, nid)
    if not rec:
        rprint(
            f"No ICD code found for '{node_id}'. Please use codes like: "
            f"'ROOT', 'Chapter_01', 'A00-B99', 'C01', 'E11.621' or 'S06.0X1A'."
        )
        return
    rprint(f"node_id   : ", rec.node_id)
    rprint(f"label     : ", rec.node_label)
    rprint(f"depth     : ", rec.depth)
    rprint(f"parent    : ", rec.parent)
    rprint(f"children  : ", rec.children)
    rprint(f"is_leaf   : ", rec.is_leaf)
    rprint(f"metadata  : ", rec.metadata)

# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from functools import partial

    idx = build_index_flat("icd10cm_tabular_2026.xml")
    view_node = partial(print_node, idx)

    for nid in [
        "S52", "S52.1", "S52.112",
        "S72", "T20", "A54.23",
        "E11.621", "L89.312", "D81.82", 
        "ROOT", "Chapter_04", "B35-B49", 123
    ]:
        try:
            view_node(str(nid))
        except Exception as e:
            rprint(f"[red]Error for {nid!r}: {e}[/red]")
        rprint("\n", "="*80, "\n")

    from collections import Counter
    depth_counts = Counter(v.get("depth", 0) for v in idx.values())

    rprint("ICD Code Counter")
    for depth in sorted(k for k in depth_counts if k):
        match depth:
            case 1: icd_code = "Chapter"
            case 2: icd_code = "Block"
            case 3: icd_code = "Category"
            case 4: icd_code = "Subcategory"
            case 5: icd_code = "Classification"
            case 6: icd_code = "Subclassification"
            case 7: icd_code = "Extension"
            case _: icd_code = f"Depth {depth}"
        rprint(f"{icd_code}: {depth_counts[depth]}")