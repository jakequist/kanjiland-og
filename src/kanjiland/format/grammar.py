"""Load grammar rule definitions from docs/GRAMMAR_RULES.md.

The inventory lives in the doc file so that spec and code stay in sync
(the doc is the single source of truth per ADR-011). We extract the fenced
YAML blocks and keep entries that look like rule definitions (i.e. have a
'roles' key). Only ``grammar-0.1`` exists today; a future ruleset version
will need its own source (separate file or headed section).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_YAML_BLOCK = re.compile(r"```yaml\s*\n(.*?)```", re.DOTALL)
_DOC_PATH = Path(__file__).resolve().parents[3] / "docs" / "GRAMMAR_RULES.md"


@lru_cache(maxsize=None)
def load_ruleset(version: str) -> dict[str, dict[str, Any]]:
    """Return {rule_id: {name, level, roles, description}} for the given
    ruleset version. Empty dict if the ruleset is unknown or the file is
    missing (linter treats unknown ruleset as an invariant-1 violation)."""
    if version != "grammar-0.1":
        return {}
    if not _DOC_PATH.exists():
        return {}
    text = _DOC_PATH.read_text(encoding="utf-8")
    rules: dict[str, dict[str, Any]] = {}
    for match in _YAML_BLOCK.finditer(text):
        try:
            block = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            continue
        if not isinstance(block, dict):
            continue
        for rule_id, spec in block.items():
            if rule_id == "RULE_ID" or not isinstance(spec, dict):
                continue
            if "roles" not in spec or not isinstance(spec["roles"], dict):
                continue
            rules[rule_id] = spec
    return rules
