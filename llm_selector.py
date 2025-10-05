import os
import re
from typing import Any
from openai import AsyncOpenAI
from pydantic import BaseModel
from rich import print as rprint


schema_pattern = r"""
(?s)                              # DOTALL so . matches newlines
\{\s*["']selected_ids["']\s*:\s*\[
\s*(?P<group1>                    # group1: zero or more comma-separated tokens
    (?:
        ["'][^"']*["']            # a quoted string
        |                         # or
        [A-Za-z0-9_.-]+           # an unquoted bare token
    )
    (?:\s*,\s*
        (?:
            ["'][^"']*["']|[A-Za-z0-9_.-]+
        )
    )*
)?
\s*\]
\s*,\s*["']reasoning["']\s*:\s*
(?P<group2>                       # group2: the reasoning value
    (?:
        ["'][^"']*["']            # a quoted string
        |                         # or: anything up to the closing brace, lazily
        [^}]*
    )
)
"""

def extract_selected_and_reasoning(text: str):
    m = re.search(schema_pattern, text, flags=re.VERBOSE)
    if m:
        g1 = m.group('group1')
        g2 = m.group('group2')
        # selected_ids
        selected_ids = []
        if g1:  # split on commas outside of quotes isn't needed if your items never contain commas
            selected_ids = [t.strip().strip('\'"') for t in g1.split(',')]
        # reasoning: strip surrounding quotes if present
        reasoning = g2.strip().strip('\'"')
        return {'selected_ids': selected_ids, 'reasoning': reasoning}

    # If no match (no selected_ids / no shape), fallback: whole string becomes reasoning
    return {'selected_ids': [], 'reasoning': text.strip()}

def configure_llm(
        base_url: str | None = None, 
        api_key: str | None = None, 
        model: str | None = None,
        temperature: float | int | None = 0.0,
        max_tokens: int | None = None,
    ):
    """Set LLM configuration through environment variables."""
    # Set default values if not configured
    if base_url:
        os.environ["LLM_BASE_URL"] = base_url
    if api_key:
        os.environ["LLM_API_KEY"] = api_key
    if model:
        os.environ["LLM_MODEL"] = model
    if temperature:
        os.environ["LLM_TEMPERATURE"] = str(temperature)
    if max_tokens:
        os.environ["LLM_MAX_TOKENS"] = str(max_tokens)

async def llm_agent(
    batch_name: str,
    context: str,
    candidates: dict[str, str] | None = None,
    instructions: str | None = None,
    feedback: str | None = None,
) -> tuple[list[str], Any | None]:
    
    # Ensure LLM is configured
    aclient = AsyncOpenAI(
        base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY")),
    )
    choices = "\n".join(f"{cid}: {txt.strip().replace('\n', ' ')}" for cid, txt in candidates.items()) 
    base_template = (
        "Given the provided context, choose only the most appropriate candidates from the list provided." +
        f"Select 0..N candidates from the following choices:\n{choices}\n" + 
        "Return [] if none or when no candidates were appropriate." + 
        "Then supply a brief reason for the primary selection/s.\n" +
        "Structured output: {'selected_ids': [...], 'reasoning': ...}"
    )
    if feedback:
        base_template += f"\nNote: {feedback}"
    
    ##### Structured Generation #####
    #################################
    ids = list(candidates.keys())
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selected_ids": {
                "type": "array",
                "items": (
                    {"type": "string", "enum": ids} if ids else {"type": "string"}
                ),
                "description": "IDs of selected candidates"
            },
            "reasoning": {
                "type": "string",
                "description": "Brief primary reason for the selection"
            }
        },
        "required": ["selected_ids", "reasoning"]
    }

    # Sanitize batch_name to match the required pattern ^[a-zA-Z0-9_-]+$
    import re
    sanitized_batch_name = re.sub(r'[^a-zA-Z0-9_-]', '_', batch_name)
    
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": sanitized_batch_name,
            "schema": schema
        }
    }

    #################################
    #################################
    payload = {
        "model": os.environ.get("LLM_MODEL", "gpt-4o"),
        "messages": [
            {"role": "system", "content": instructions.strip() if instructions else base_template},
            {"role": "user", "content": context},
        ],
        "temperature": float(os.environ.get("LLM_TEMPERATURE", 0.0)),
    }
    if response_format:
        payload.update({"response_format": response_format})

    max_tokens = os.environ.get("LLM_MAX_TOKENS", None)
    if max_tokens:
        payload.update({"max_tokens": int(max_tokens)})

    result = await aclient.chat.completions.create(**payload)
    
    # Handle different response types from OpenAI API
    try:
        # For ChatCompletion objects, extract the content and parse JSON
        if hasattr(result, 'choices') and len(result.choices) > 0:
            content = result.choices[0].message.content
            if content:
                import json
                parsed_result = json.loads(content)
                selected_ids = parsed_result.get("selected_ids", [])
                reasoning = parsed_result.get("reasoning", None)
                return selected_ids, reasoning
        
        # For BaseModel objects (if any)
        if isinstance(result, BaseModel):
            return result.selected_ids, result.reasoning
            
        # For string responses with choices
        if isinstance(result, str) and "choices" in result:
            content = result.choices[0].message.content if hasattr(result, 'choices') else str(result)
            return extract_selected_and_reasoning(content)
            
        # Fallback for other response types
        content = result.choices[0].message.content if hasattr(result, 'choices') else str(result)
        extracted = extract_selected_and_reasoning(content)
        return extracted["selected_ids"], extracted["reasoning"]
    except (json.JSONDecodeError, AttributeError, IndexError, KeyError) as e:
        # If any error occurs, fallback to regex extraction
        try:
            content = result.choices[0].message.content if hasattr(result, 'choices') and len(result.choices) > 0 else str(result)
            extracted = extract_selected_and_reasoning(content)
            return extracted["selected_ids"], extracted["reasoning"]
        except Exception:
            # Last resort fallback
            return [], None
    

# Add get_children function
def get_children(index: dict, node_id: str) -> dict[str, str]:
    """Return {child_id: label/text} or {} if leaf/missing."""
    try:
        children = index[node_id].get("children") or {}
        if isinstance(children, dict):
            return children
        return {cid: index[cid].get("label", cid) for cid in children}
    except Exception:
        return {}

# Add select_children function
async def select_children(index: dict, node_id: str, *, context: str) -> tuple[list[str], str | None]:
    """Run a NodeProgram on this node's children and return (selected_ids, reasoning)."""
    children = get_children(index, node_id)
    if not children:
        return ([], None)
    
    # Use the llm_selector function to select children
    selected_ids, reasoning = await llm_agent(
        batch_name=f"{node_id}_-_children",
        context=context,
        candidates=children
    )
    return (selected_ids or [], reasoning or None)

# Add walk_and_print function
async def walk_and_print(index: dict, root_id: str, *, context: str) -> None:
    """
    DFS from root → leaves.
    Prints each node once (no duplicates). If the parent selected this node,
    its reasoning is printed one extra tab beneath the node line.
    """
    rprint("="*45)
    rprint("[red]ICD-10-CM Traversal - Parent to Child[/]")
    rprint("="*45)
    
    async def dfs(node_id: str, depth: int, reason_from_parent: str | None) -> None:
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
        selected_ids, reasoning = await select_children(index, node_id, context=context)
        if not selected_ids:
            return

        # recurse into each selected child; do NOT print the child here
        for cid in selected_ids:
            await dfs(cid, depth + 1, reasoning)

    await dfs(root_id, 0, None)



# Example usage
if __name__ == "__main__":
    import asyncio
    from icd_index import build_index_flat

    # Build the ICD index
    index = build_index_flat("icd10cm_tabular_2026.xml")

    # Configure the LLM
    # Note: This example assumes you have set up your OpenAI API key in the environment variables
    
    # Run the tree traversal
    ROOT = "ROOT"
    ctx = "Diabetes Mellitus (not sure if Type 1 or Type 2) with ketoacidosis, GCS=3"

    configure_llm(model="gpt-4o-mini", max_tokens=200)    
    # Since walk_and_print is async, we need to run it in an event loop
    asyncio.run(walk_and_print(index, ROOT, context=ctx))
