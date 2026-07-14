"""Import-guard for CLAUDE.md rule #1 / ADR-007.

The shipped inference path (everything under ``src/kanjiland``) may NOT import
classical NLP tooling. Those tools are permitted for OFFLINE silver-data
generation, but only under ``tools/`` — never in the runtime package.

This is a *static* check: it parses the AST of every module under
``src/kanjiland`` and inspects its ``import`` statements. It therefore catches
forbidden dependencies even when they are imported lazily/conditionally, and
works without the forbidden packages being installed.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Top-level package names for runtime-NLP tooling. Matched against the first
# dotted component of every imported module. Keep lowercase.
FORBIDDEN_TOP_LEVEL = frozenset(
    {
        "mecab",  # mecab-python3
        "fugashi",
        "natto",  # natto-py (MeCab binding)
        "sudachipy",
        "sudachidict_core",
        "sudachidict_full",
        "sudachidict_small",
        "unidic",
        "unidic_lite",
        "ipadic",
        "jumandic",
        "pyknp",
        "janome",
        "nagisa",
        "konoha",
        "spacy",
        "ginza",
    }
)

SRC_PKG = Path(__file__).resolve().parent.parent / "src" / "kanjiland"


def _imported_top_levels(tree: ast.AST):
    """Yield (top_level_module, lineno) for every import in an AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0].lower(), node.lineno
        elif isinstance(node, ast.ImportFrom):
            # Ignore relative imports (node.level > 0); they are intra-package.
            if node.level == 0 and node.module:
                yield node.module.split(".")[0].lower(), node.lineno


def test_runtime_package_imports_no_classical_nlp():
    py_files = sorted(SRC_PKG.rglob("*.py"))
    assert py_files, f"no python files found under {SRC_PKG}"

    violations: list[str] = []
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for top_level, lineno in _imported_top_levels(tree):
            if top_level in FORBIDDEN_TOP_LEVEL:
                rel = path.relative_to(SRC_PKG.parent.parent)
                violations.append(f"{rel}:{lineno} imports '{top_level}'")

    assert not violations, (
        "Runtime NLP dependency imported in src/kanjiland (CLAUDE.md rule #1, "
        "ADR-007). Move offline-only tooling under tools/:\n  "
        + "\n  ".join(violations)
    )
