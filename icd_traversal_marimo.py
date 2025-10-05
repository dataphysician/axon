# icd_traversal_marimo.py

import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    import os as _os
    
    def create_context_input():
        return mo.ui.text_area(
            value="Diabetes Mellitus (likely Type 2) with ketoacidosis, GCS=3",
            # label="Clinical Case",
            full_width=True,
        )

    def create_base_url_input():
        return mo.ui.text(
            value="https://api.openai.com/v1",
            label="LLM Base URL (optional)",
            full_width=True,
        )

    def create_api_key_input():
        # Use environment variable if set, otherwise use placeholder
        default_api_key = _os.environ.get("OPENAI_API_KEY", "hailtaumu")
        return mo.ui.text(
            value=default_api_key,
            kind="password",
            label="LLM API Key (optional)",
            full_width=True,
        )

    def create_model_input():
        return mo.ui.text(
            value="gpt-4o-mini",
            label="LLM Model",
            full_width=True,
        )

    def create_temperature_input():
        return mo.ui.slider(
            0.0, 1.0, 
            value=0.7, 
            step=0.1, 
            label="Temperature"
        )

    def create_max_tokens_input():
        return mo.ui.number(
            1, 32000, 
            value=20, 
            label="Max Tokens"
        )

    def create_transpile_button():
        return mo.ui.run_button(label="Execute")

    # Create all UI elements (removed xml_upload)
    context_input = create_context_input()
    base_url = create_base_url_input()
    api_key = create_api_key_input()
    model_name = create_model_input()
    temperature = create_temperature_input()
    max_tokens = create_max_tokens_input()
    transpile_btn = create_transpile_button()

    # Create the main UI layout with tabs (removed ICD XML Source section)
    clinical_tab = mo.vstack([
        mo.md("##Clinical Context"),  
        context_input,
        mo.md("##Transpile to ICD-10-CM"),  
        transpile_btn,
    ], align="start")

    # Create two columns for the settings
    left_column = mo.vstack([
        mo.md("##LLM Configuration"),  
        base_url,
        api_key,
        model_name,
    ], align="start")
    
    right_column = mo.vstack([
        mo.md("##Parameters"),  
        temperature,
        max_tokens,
    ], align="start")
    
    # Arrange columns in a horizontal stack and stretch
    settings_tab = mo.hstack([
        left_column,
        right_column
    ]).style(width="100%")

    # Create tabs widget
    tabs = mo.ui.tabs({
        "Clinical": clinical_tab,
        "Settings": settings_tab
    })

    main_ui = mo.vstack([
        mo.md("#ICD-10-CM Agentic Traversal"),
        tabs,
    ], align="start").style(width="100%")
    
    # Return the main UI container for display
    main_ui
    
    return (
        create_context_input,
        create_base_url_input, create_api_key_input, create_model_input,
        create_temperature_input, create_max_tokens_input, create_transpile_button,
        context_input,
        base_url, api_key, model_name, temperature, max_tokens,
        transpile_btn, tabs, clinical_tab, settings_tab
    )


@app.cell
def _(mo, transpile_btn):
    from icd_index import build_index_flat

    index: dict = {}

    if transpile_btn.value:
        # Use default XML file
        index = build_index_flat("icd10cm_tabular_2026.xml")
        mo.md(f"✅ Built index with **{len(index)}** nodes")
    return (index,)


@app.cell
def _():
    # Global variable to track report codes for Marimo UI
    _global_report_codes = []
    return


@app.cell
def _(mo, transpile_btn):
    global _global_report_codes
    # Clear the report codes when Execute button is pressed
    if transpile_btn.value:
        _global_report_codes = []
    return


@app.cell
def _(api_key, base_url, max_tokens, mo, model_name, temperature, transpile_btn):
    import os as _os
    from llm_selector import configure_llm
    if transpile_btn.value:
        # Use environment variable if UI field contains placeholder or is empty
        api_key_value = None
        if api_key.value and api_key.value != "hailtaumu":
            api_key_value = api_key.value
        else:
            api_key_value = _os.environ.get("OPENAI_API_KEY")
            
        configure_llm(
            base_url=base_url.value if base_url.value else None,
            api_key=api_key_value,
            model=model_name.value,
            temperature=temperature.value,
            max_tokens=max_tokens.value
        )
        mo.md(f"✅ LLM configured: `{model_name.value}` (temp={temperature.value}, max_tokens={int(max_tokens.value)})")
    return


@app.cell
def _(context_input, index: dict, mo, transpile_btn):
    from burr.core import State

    init = None
    if transpile_btn.value and index:
        init = (
            State()
            .update(
                index=index,
                queue=["ROOT"],                 # stack top is END; expand pops from END
                traversal_depths={"ROOT": 0},
                traversal_kind={"ROOT": None},  # 'children','codeFirst','codeAlso','useAdditional', None
                batches={},
                visited=[],
                end_nodes=[],
                context=context_input.value,
            )
        )
        mo.md("🔧 Initialized traversal state")
    return (init,)


@app.cell
def _(mo):
    from typing import Any, Iterable
    from collections import OrderedDict

    ARROW = "&nbsp;&nbsp;"  # invisible spacer; repeat per depth

    def _indent(depth: int) -> str:
        # depth=0 -> "", depth=1 -> "❯❯ ", depth=2 -> "❯❯❯❯ ", …
        return (ARROW * 2 * max(depth, 0)) + (" " if depth > 0 else "")

    def _title_for(node_id: str, index: dict[str, dict], depth: int, kind: str | None) -> str:
        base_label = index.get(node_id, {}).get("label", node_id)
        label = f"{base_label} ({kind})" if kind and kind != "children" else base_label
        return f"{_indent(depth)}{node_id} — {label}"

    def _ui_dict(d: dict[str, str]) -> mo.ui.dictionary:
        # Wrap values with mo.md for nicer rendering
        return mo.ui.dictionary({k: mo.md(v) for k, v in (d or {}).items()})

    def build_stream_tree_markdown(batch_info_data: Iterable[dict[str, Any]], index: dict[str, dict]) -> str:
        """
        Streamed DFS preview in EXACT batch arrival order.
        For each batch b, append each selected node at b['depth'] with optional kind annotation.
        Expected batch fields: depth:int, selected:list[str], kind:str|None
        """
        if not batch_info_data:
            return "_Traversal pending…_"
        lines: list[str] = ["**ICD Code Traversal:**"]
        for b in batch_info_data:
            depth = int(b.get("depth", 0))
            kind = b.get("kind")  # 'children', 'codeFirst', 'codeAlso', 'useAdditional', etc.
            for nid in (b.get("selected") or []):
                lines.append(_title_for(str(nid), index, depth, kind))
        return "\n\n".join(lines)  # Use double newlines for proper markdown line breaks

    def build_stream_accordion(batch_info_data: Iterable[dict[str, Any]], index: dict[str, dict], traversal_depths: dict[str, int] | None = None, traversal_kind: dict[str, str | None] | None = None):
        """
        Single accordion: append items strictly in encounter order.
        Each batch becomes an item; body shows that batch's candidates & reasoning.
        """
        items_list: list[tuple[str, Any]] = []
        traversal_depths = traversal_depths or {}
        traversal_kind = traversal_kind or {}

        for b in (batch_info_data or []):
            parent_id = b.get("parent_id", "")
            batch_depth = int(b.get("depth", 0))
            kind = b.get("kind")
            candidates = b.get("candidates") or {}
            reasoning = str(b.get("reasoning") or "")
            selected = b.get("selected", [])

            # Build content lazily for performance
            def _content_factory(candidates=candidates, reasoning=reasoning, selected=selected):
                # Create a simple markdown element with all the information
                content_parts = []

                # Add selected items section
                content_parts.append(f"**Selected:** {', '.join(selected) if selected else 'None'}")

                # Add candidates section
                if candidates:
                    candidates_md = "\n".join([f"- {k}: {v}" for k, v in candidates.items()])
                    content_parts.append(f"**Candidates:**\n{candidates_md}")
                else:
                    content_parts.append("_No candidates_")

                # Add reasoning section
                if reasoning:
                    content_parts.append(f"**Reasoning:**\n{reasoning}")
                else:
                    content_parts.append("_No reasoning provided_")

                return mo.md("\n\n".join(content_parts))

            # Create title for the batch
            base_title = _title_for(parent_id, index, batch_depth, kind)
            # Keep titles unique without altering order
            title = base_title
            seen_titles = {t for t, _ in items_list}
            suffix = 1
            while title in seen_titles:
                suffix += 1
                title = f"{base_title} ·{suffix}"
            items_list.append((title, _content_factory))

        if not items_list:
            return mo.accordion(OrderedDict([("(no selections yet)", mo.md("_Waiting for first selection…_"))]), lazy=True)

        ordered = OrderedDict(items_list)  # preserve exact encounter order
        return mo.accordion(ordered, lazy=True)
    return (build_stream_accordion,)


@app.cell
async def _(build_stream_accordion, index: dict, init, mo):
    """
    Run traversal and refresh the tree/accordion whenever collect_batch_info grows.
    No sorting; order is exactly as produced by the orchestrator.
    """
    import asyncio
    from burr_orchestrator import run_until_converged, collect_batch_info

    # Initialize return variables
    final_state = None
    report_codes = list()
    batch_info_data: list[dict] = []
    logs = ""

    if init is None:
        mo.md("_Click **Run traversal** to begin._")
    else:
        # Clear any existing batch info collection before starting a new traversal
        from burr_orchestrator import clear_batch_info_collection
        clear_batch_info_collection()
        
        # Store final values here (avoid nonlocal issues)
        results = {"final_state": None, "logs": ""}

        # Output shell we keep replacing as new batches arrive
        mo.output.replace(mo.status.spinner("Starting traversal…"))

        # Render a given snapshot immediately (preserving encounter order)
        def _render_snapshot(snapshot: list[dict]):
            # Extract traversal_depths and traversal_kind from the batch data
            traversal_depths = {}
            traversal_kind = {}
            for batch in snapshot:
                parent_id = batch.get("parent_id")
                depth = batch.get("depth")
                kind = batch.get("kind")
                selected = batch.get("selected", [])
                # Set depths for selected nodes
                for node_id in selected:
                    # Only set if not already set (to preserve the first occurrence)
                    if node_id not in traversal_depths:
                        traversal_depths[node_id] = depth
                    # For traversal_kind, we'll use the kind from the batch that selected this node
                    if node_id not in traversal_kind:
                        traversal_kind[node_id] = kind

            # Build the accordion
            accordion = build_stream_accordion(snapshot, index or {}, traversal_depths, traversal_kind)

            # Replace the output with the accordion
            mo.output.replace(mo.vstack([
                mo.md("##ICD Traversal"),  
                accordion
            ]))

        # Wrap the traversal so we can capture stdout in the task
        async def _run_and_capture():
            with mo.capture_stdout() as buf:
                results["final_state"] = await run_until_converged(init)
            results["logs"] = buf.getvalue()

        # Start traversal as a task
        task = asyncio.create_task(_run_and_capture())

        # Live-poll for new batches and update UI incrementally
        last_len = 0
        try:
            while not task.done():
                if hasattr(collect_batch_info, "collected_data"):
                    current = collect_batch_info.collected_data or []
                    if len(current) != last_len:
                        batch_info_data = list(current)  # exact order from orchestrator
                        last_len = len(current)
                        _render_snapshot(batch_info_data)
                await asyncio.sleep(0.1)  # keep UI responsive

            # Task finished; ensure final state & final snapshot
            await task
            if hasattr(collect_batch_info, "collected_data"):
                batch_info_data = list(collect_batch_info.collected_data or [])
            _render_snapshot(batch_info_data)

            # Append end nodes (if any) after final render
            try:
                fs = results["final_state"]
                if isinstance(fs, dict) and fs.get("end_nodes"):
                    final_codes = fs["end_nodes"]

            except Exception:
                pass

            # Update return variables with results
            final_codes = results["final_state"]
            logs = results["logs"]

        except Exception as e:
            mo.output.replace(mo.md(f"**Error:** {e!r}"))
        if isinstance(report_codes, str):
            report_codes = [report_codes]
        else:
            # Use formatted_end_nodes if available, otherwise fall back to end_nodes
            if "formatted_end_nodes" in results["final_state"]:
                report_codes = results["final_state"]["formatted_end_nodes"]
            else:
                report_codes = results["final_state"]["end_nodes"] or []
    return final_codes, logs, report_codes


@app.cell
def _(final_codes, logs, report_codes, mo):
    global _global_report_codes
    # Update global report codes with the latest results
    _global_report_codes = report_codes
    
    if logs:
        mo.md("##Traversal logs")  
        mo.md(f"```\n{logs}\n```")

    ui_codes = mo.md("Empty")
    if _global_report_codes and len(_global_report_codes) > 0:
        # Display codes in a table format
        codes_table = "| Code | Label | Billing Status |\n|------|-------|----------------|\n"
        for code in _global_report_codes:
            parts = code.split('\t')
            if len(parts) == 3:
                node_id, label, billing_status = parts
                codes_table += f"| {node_id} | {label} | {billing_status} |\n"
            else:
                # Fallback for any codes that don't match the expected format
                codes_table += f"| {code} | | |\n"
        ui_codes = mo.md(f"**Final Code List:**\n\n{codes_table}")
    mo.vstack([
        mo.md("##Codes Identified"),  
        ui_codes,        
    ])
    return


if __name__ == "__main__":
    app.run()
