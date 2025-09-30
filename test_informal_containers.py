#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from typing import Any

# ─────────────────────────────
# Types
# ─────────────────────────────
NodeId = str
Metadata = dict[str, Any]

# ─────────────────────────────
# Code normalization
# ─────────────────────────────

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
    s = (s or "").strip()
    if not s:
        raise ValueError("Empty ICD code")
    if s.upper().startswith("ROOT"):
        return "ROOT"
    if s.startswith("chapter_"):
        return s
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    if s.endswith("-"):
        s = s.rstrip("-")
    if not re.match(NORMALIZED_CODE_RE, s):
        raise ValueError(f"Invalid ICD code/range/list: {s!r}")
    return s

def to_block_form(code: str) -> str:
    if "-" in code:
        return code
    return f"{code}-{code}"

# ─────────────────────────────
# Verification
# ─────────────────────────────
def verify_informal_sections(xml_path: str, print_mode: str = "both") -> int:
    """
    Returns 0 if all informal sections contain zero <diag> descendants.
    Returns non-zero and prints a report if any violations are found.
    print_mode ∈ {"formal", "informal", "both"} controls what block IDs are listed.
    """
    et = ET.parse(xml_path)
    root = et.getroot()

    total_chapters: int = 0
    total_sections: int = 0
    informal_sections: int = 0
    violations: list[dict[str, Any]] = []
    all_blocks: dict[str, set[str]] = {}
    informal_blocks: dict[str, list[str]] = {}

    for chapter_i, chapter in enumerate(root.findall("chapter"), start=1):
        total_chapters += 1
        chap_num = f"{chapter_i:02d}"

        block_ids: set[str] = set()
        sidx = chapter.find("sectionIndex")
        if sidx is not None:
            for sref in sidx.findall("sectionRef"):
                bid_raw = (sref.get("id") or "").strip()
                if not bid_raw:
                    continue
                try:
                    bid_norm = normalize_code(bid_raw)
                except ValueError:
                    print(f"Problem with Block: {bid_raw}") # TODO: Remove guard later
                    continue
                block_ids.add(to_block_form(bid_norm)) # Add blocks to set in range format
        all_blocks[chap_num] = block_ids

        for section in chapter.findall("section"):
            total_sections += 1
            sec_id_raw = section.get("id") or ""
            try:
                sec_norm = normalize_code(sec_id_raw)
                sec_block = to_block_form(sec_norm)
            except ValueError:
                sec_norm = sec_id_raw.strip() or "<NO-ID>"
                sec_block = "<NO-BLOCK>"

            is_informal = sec_block not in block_ids
            diag_descendants = section.findall(".//diag")
            diag_count = len(diag_descendants)

            if is_informal:
                informal_sections += 1
                informal_blocks.setdefault(chap_num, []).append(sec_block)
                if diag_count > 0:
                    samples: list[str] = []
                    for d in diag_descendants[:5]:
                        name = (d.findtext("name") or "").strip()
                        desc = (d.findtext("desc") or "").strip()
                        if desc and name:
                            samples.append(f"{name}: {desc[:80]}")
                        elif desc:
                            samples.append(desc[:80])
                        elif name:
                            samples.append(name)
                        else:
                            samples.append("<diag>")
                    violations.append(
                        {
                            "chapter": chap_num,
                            "section_id": sec_norm,
                            "diag_count": diag_count,
                            "samples": samples,
                        }
                    )

    # Reporting
    print("=== Informal Section Verification ===")
    print(f"XML: {xml_path}")
    print(f"Chapters: {total_chapters}")
    print(f"Total Sections: {total_sections}")
    print(f"Informal sections (sections not in sectionIndex): {informal_sections}")
    print(f"Formal block sections: {total_sections - informal_sections}")
    print()

    if print_mode in {"formal", "both"}:
        print("Formal Block IDs")
        for chap_num, block_ids in all_blocks.items():
            print(f"Chapter {chap_num}: {str(len(block_ids)) + ' blocks' if block_ids else ''} {sorted(block_ids)}")
        print()

    if print_mode in {"informal", "both"}:
        print("Informal Section IDs:")
        for chap_num, blocks in informal_blocks.items():
            print(f"Chapter {chap_num}: {str(len(blocks)) + ' containers' if blocks else ''} {sorted(blocks)}")
        print()

    if print_mode.lower() in {"none", ""}:
        pass

    if not violations:
        print("✅ PASS: No informal sections containing <diag> descendants.")
        return 0

    print(f"❌ FAIL: {len(violations)} informal section(s) contain <diag> descendants.\n")
    for v in violations:
        print(f"- Chapter {v['chapter']} | Section ID: {v['section_id']}")
        print(f"  diag descendants: {v['diag_count']}")
        if v["samples"]:
            print("  sample diags:")
            for s in v["samples"]:
                print(f"    • {s}")
        print()
    return 2

# ─────────────────────────────
# CLI
# ─────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that informal <chapter>/<section> containers have no <diag> descendants."
    )
    parser.add_argument(
        "--xml",
        default="icd10cm_tabular_2026.xml",
        help="Path to icd10cm_tabular_<year>.xml",
    )
    parser.add_argument(
        "--print-mode",
        choices=["formal", "informal", "both"],
        default="none",
        help="Which block IDs to print: only formal, only informal, or both (default).",
    )
    args = parser.parse_args()
    rc = verify_informal_sections(args.xml, print_mode=args.print_mode)
    sys.exit(rc)

if __name__ == "__main__":
    main()

