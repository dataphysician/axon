#!/usr/bin/env python3
from __future__ import annotations

import re
from rich import print as rprint
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from collections.abc import Hashable
from typing import TypeVar

# ──────────────────────────────────────────────────────────────────────────────
# Generic Types
# ──────────────────────────────────────────────────────────────────────────────

# PEP 695 generic type parameter
T = TypeVar("T", bound=Hashable)

@dataclass(slots=True)
class GenericNode[T]:
    """
    Library-style, reusable node record.

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
        return self.children is None or not self.children


# ──────────────────────────────────────────────────────────────────────────────
# Code Patterns
# ──────────────────────────────────────────────────────────────────────────────

# AAA or AAA.BBBB (1–4 after the dot); letters/digits allowed (e.g., X placeholders, 7th chars A/D/S)
_DIAG_CODE_RE = re.compile(r"^[A-Z][A-Z0-9]{2}(?:\.[A-Z0-9]{1,4})?$", re.IGNORECASE)
# ROOT  | Chapter_XX | CODE-RANGE | ICD.CODE
_CODE_INPUT_RE = re.compile(
    r"""^(?:
        (?P<root>ROOT) | 
        (?P<chapter>Chapter_(?P<chapnum>0[1-9]|1[0-9]|2[0-2])) | 
        (?P<range>(?P<left>[A-Z][A-Z0-9]{2})-(?P<right>[A-Z][A-Z0-9]{2})) | 
        (?P<diag>(?P<d1>[A-Z][A-Z0-9]{2})(?:\.(?P<d2>[A-Z0-9]{1,4}))?)
    )$""",
    re.I,
)


# ──────────────────────────────────────────────────────────────────────────────
# ICD Types
# ──────────────────────────────────────────────────────────────────────────────

class ICDCode(str):
    """
    Valid forms (any input case):
      - ROOT
      - Chapter_01 to Chapter_22
      - Block range: A00-B99, Q00-QA0, I30-I5A
      - Diagnosis: C01, E11.621, S06.0X1A

    Canonicalizes to:
      - 'ROOT'
      - 'Chapter_XX'
      - 'A00-B99' (uppercased)
      - Uppercased diagnosis (e.g., 'E11.621')
    """
    __slots__ = ()

    def __new__(cls, s: str) -> "ICDCode":
        if not isinstance(s, str):
            raise TypeError("ICDCode must be constructed from a string")
        m = _CODE_INPUT_RE.fullmatch(s.strip())
        if not m:
            raise ValueError("Invalid ICD code. Expecting: 'ROOT' or 'Chapter_01' or standard ICD code like 'A00-B99', 'C01', 'E11.621' or 'S06.0X1A'.")

        g = m.groupdict()
        if g["root"]: # 'ROOT'
            return str.__new__(cls, "ROOT")
        if g["chapter"]: # 'Chapter_XX'
            return str.__new__(cls, f"Chapter_{g['chapnum']}")
        if g["range"]: # Range like 'A00-B99'
            return str.__new__(cls, f"{g['left'].upper()}-{g['right'].upper()}")

        # diagnosis
        d1 = g["d1"].upper()
        d2 = g.get("d2")
        return str.__new__(cls, d1 if not d2 else f"{d1}.{d2.upper()}")


NodeData = GenericNode[ICDCode] # assigns NodeData class to use the ICDCode as the look-up key of GenericNode
Index = dict[ICDCode, NodeData]  # assigns Index class flat lookup table

# ──────────────────────────────────────────────────────────────────────────────
# Code Input normalization
# ──────────────────────────────────────────────────────────────────────────────
def normalize_code(s: str) -> str:
    """
    Normalize a single ICD-10-CM diagnosis code from <diag><name>.
    Examples accepted: C01, E11.621, S06.0X1A, Z99, T36.0X5S
    Normalizations:
      - Trim whitespace
      - Strip surrounding parentheses: "(E11.621)" -> "E11.621"
      - Treat 'S82.-' style placeholders as the category: 'S82.-' -> 'S82'
      - Strip any trailing '-' and/or '.' left dangling: 'S82.' -> 'S82'
      - Uppercase
      - Reject ranges/lists (no '-' ranges, no ',' or ';')
    """
    s = (s or "").strip()
    if not s:
        raise ValueError("Empty diagnosis code")

    # Surrounding parentheses
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()

    # Handle roll-up shorthand like 'S82.-' -> 'S82'
    if s.endswith("-."):
        s = s[:-2]

    # Strip a trailing hyphen (e.g., 'T36.0X5S-' -> 'T36.0X5S')
    if s.endswith("-"):
        s = s[:-1]

    # Strip a trailing dot if left dangling (e.g., 'S82.' -> 'S82')
    if s.endswith("."):
        s = s[:-1]

    su = s.upper()

    # Disallow ranges/lists for <diag><name>
    if "-" in su or "," in su or ";" in su or "~" in su:
        raise ValueError(f"Expected a single diagnosis code, got: {s!r}")

    if not _DIAG_CODE_RE.fullmatch(su):
        raise ValueError(f"Invalid diagnosis code: {s!r}")

    return su


def to_block_form(code: str) -> str:
    """Ensure code is in block form ("C50" -> "C50-C50")."""
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
    """Flatten all text under elem and split into non-empty, stripped lines."""
    out: list[str] = []
    for chunk in elem.itertext():
        for line in str(chunk).splitlines():
            s = line.strip()
            if s:
                out.append(s)
    return out

def get_metadata(elem: ET.Element) -> dict[str, str | list[str] | dict[str, str | list[str]]]:
    """
    Extract ONLY standardized rule-like content:

      • For all KNOWN_TAGS except sevenChrDef:
          - Collect the immediate child <note> elements.
          - Each <note> may contain multiple lines (via <br/> etc.); split into lines.
          - If the total collected lines == 1, store a str; else store list[str].

      • For sevenChrDef:
          - Build dict[char -> str | list[str]] from <extension char="…">…</extension>.
          - If an extension’s text has multiple lines, store list[str] for that char.
          - If no <extension> children but raw text exists, store under "_text".
          - If multiple sevenChrDef blocks repeat a char, merge (concatenate) lines.
    """
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
                lines = _lines(ext)  # 0, 1, or many lines
                if not lines:
                    continue
                val: str | list[str] = lines[0] if len(lines) == 1 else lines

                if ch in mapping:
                    # merge with existing value
                    cur = mapping[ch]
                    if isinstance(cur, list) and isinstance(val, list):
                        mapping[ch] = [*cur, *val]
                    elif isinstance(cur, list):
                        mapping[ch] = [*cur, val]
                    elif isinstance(val, list):
                        mapping[ch] = [cur, *val]
                    else:
                        mapping[ch] = [cur, val]
                else:
                    mapping[ch] = val

            if not found_ext:
                lines = _lines(child)
                if lines:
                    mapping["_text"] = lines[0] if len(lines) == 1 else lines

            # merge sevenChrDef blocks if multiple
            if "sevenChrDef" in meta and isinstance(meta["sevenChrDef"], dict):
                existing = meta["sevenChrDef"]
                for ch, val in mapping.items():
                    if ch in existing:
                        cur = existing[ch]
                        if isinstance(cur, list) and isinstance(val, list):
                            existing[ch] = [*cur, *val]
                        elif isinstance(cur, list):
                            existing[ch] = [*cur, val]
                        elif isinstance(val, list):
                            existing[ch] = [cur, *val]
                        else:
                            existing[ch] = [cur, val]
                    else:
                        existing[ch] = val
            else:
                meta["sevenChrDef"] = mapping

            continue

        # All other rule-like tags: immediate <note> children
        notes = child.findall("./note")
        gathered: list[str] = []
        if notes:
            for n in notes:
                gathered.extend(_lines(n))
        else:
            # Some containers have raw text without <note>
            gathered = _lines(child)

        if not gathered:
            continue
        meta[key] = gathered[0] if len(gathered) == 1 else gathered

    return meta


# ──────────────────────────────────────────────────────────────────────────────
# Index helpers
# ──────────────────────────────────────────────────────────────────────────────
def _add_node(
    index: Index,
    *,
    node_id: ICDCode,
    node_label: str,
    parent: ICDCode | None,
    depth: int,
    metadata: dict[str, str | list[str] | dict[str, str | list[str]]],
) -> None:
    """Insert or merge a node (and carefully merge metadata)."""
    if node_id not in index:
        index[node_id] = NodeData(
            node_id=node_id,
            node_label=node_label,
            parent=None,
            depth=depth,
        )

    rec = index[node_id]
    rec.node_label = node_label or rec.node_label
    rec.depth = depth

    if parent and rec.parent is None:
        rec.parent = {parent: index[parent].node_label if parent in index else parent}

    # merge metadata
    for k, v in (metadata or {}).items():
        if k not in rec.metadata:
            rec.metadata[k] = v
        else:
            rec.metadata[k] = _merge_meta_value(rec.metadata[k], v)  # type: ignore[arg-type]

def _link_parent_child(index: Index, parent: ICDCode, child: ICDCode) -> None:
    """Populate parent→children and child's parent map (if missing)."""
    if parent not in index:
        index[parent] = NodeData(node_id=parent, node_label=parent, parent=None, depth=0)
    if child not in index:
        index[child] = NodeData(node_id=child, node_label=child, parent=None, depth=0)

    index[parent].children[child] = index[child].node_label
    if index[child].parent is None:
        index[child].parent = {parent: index[parent].node_label}

def _add_diag_tree(index: Index, diag: ET.Element, parent: ICDCode, start_depth: int) -> None:
    """Attach a <diag> node (and nested <diag>) under given parent."""
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
    # namespace-agnostic check for any <diag> descendant
    for e in section_elem.iter():
        if _local(e.tag) == "diag":
            return True
    return False

# ──────────────────────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────────────────────
def build_index_flat(xml_path: str) -> Index:
    et = ET.parse(xml_path)
    root = et.getroot()
    index: Index = {}

    # ROOT
    index["ROOT"] = NodeData(node_id="ROOT", node_label="ICD-10-CM", parent=None, depth=0)

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
        formal_blocks: set[str] = set()
        for section in chapter.findall("section"):
            sec_id_raw = (section.get("id") or "").strip()
            if not sec_id_raw:
                continue
            block_id = to_block_form(sec_id_raw)
            # include only if this section has at least one <diag>
            if not _section_has_diag(section):
                continue

            block_label = (section.findtext("desc") or "").strip() or block_id

            _add_node(
                index,
                node_id=block_id,
                node_label=block_label,
                parent=chap_id,
                depth=2,
                metadata=get_metadata(section),  # merge rule-like tags from the section
            )
            _link_parent_child(index, chap_id, block_id)
            formal_blocks.add(block_id)

            # ICD Codes
            for diag in section.findall("diag"):
                _add_diag_tree(index, diag, block_id, start_depth=3)


    return index

# ──────────────────────────────────────────────────────────────────────────────
# Convenience
# ──────────────────────────────────────────────────────────────────────────────
def print_node(index: Index, node_id: ICDCode) -> None:
    rec = index.get(node_id)
    if not rec:
        rprint(f"No ICD code found for '{node_id}'. Please use codes like: 'ROOT', 'Chapter_01', 'A00-B99', 'C01', 'E11.621' or 'S06.0X1A'.")
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
    # xml_path = sys.argv[1] if len(sys.argv) > 1 else "icd10cm_tabular_2026.xml"
    from functools import partial

    idx = build_index_flat("icd10cm_tabular_2026.xml")
    view_node = partial(print_node, idx)

    for nid in [
        "S52", "S52.1", "S52.112",
        "S72", "T20", "A54.23",
        "E11.621", "L89.312", "D81.82", 
        "ROOT", "Chapter_04", "B35-B49", 123
    ]:
        view_node(nid)
        rprint("\n", "="*80, "\n")


    from collections import Counter
    depth_counts = Counter(node.depth for node in idx.values())
    rprint("ICD Code Counter")
    for depth in sorted(depth_counts):
        match depth:
            case 0:
                continue
            case 1:
                icd_code = "Chapter"
            case 2:
                icd_code = "Block"
            case 3:
                icd_code = "Category"
            case 4:
                icd_code = "Subcategory"
            case 5:
                icd_code = "Classification"
            case 6:
                icd_code = "Subclassification"
            case 7:
                icd_code = "Extension"

        rprint(f"{icd_code}: {depth_counts[depth]}")