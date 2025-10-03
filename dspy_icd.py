from pathlib import Path
import dspy

class NodeSig(dspy.Signature):
    """Select ALL correct candidate IDs for this node. Return [] if none."""
    context: str  = dspy.InputField(desc="Provenance / prior decisions")
    choices: str  = dspy.InputField(desc="One per line: 'candidate_id: text'")
    selected_ids: list[str] = dspy.OutputField(desc="Subset of IDs")

class NodeProgram(dspy.Module):
    def __init__(self, run_id: str | None = None, artifact_dir: Path | str = "artifacts", auto_load: bool = True):
        super().__init__()
        self._artifact_dir: Path = Path(artifact_dir)
        self._run_id: str | None = run_id
        self.decide = dspy.Predict(NodeSig)
        if run_id is not None and auto_load:
            self.load()

    def forward(self, context: str, candidates: dict[str, str]) -> list[str]:
        choices_str = "\n".join(f"{cid}: {txt.strip().replace('\n', ' ')}" for cid, txt in candidates.items())
        result = self.decide(context=context, choices=choices_str)
        return result.selected_ids or [] # NOTE: Temporary guard against faulty LLM output or failed parsing.

    def load(self) -> "NodeProgram":
        if self._run_id is None:
            return self
        path = self._artifact_dir / self._run_id
        try:
            if path.exists():
                loaded = dspy.load(str(path))
                self.decide = loaded.decide
        except Exception:
            pass
        return self

    def save(self) -> None:
        if self._run_id is None:
            raise ValueError("run_id is required to save a node program.")
        path = self._artifact_dir / self._run_id
        path.parent.mkdir(parents=True, exist_ok=True)
        dspy.save(self, str(path))
