# tools/ — offline-only helpers

Nothing in this directory may be imported by src/kanjiland (enforce with a test
once the first tool lands). This is where silver-data generation and teacher
distillation live.

ADR-007 (ACCEPTED — hybrid): classical NLP tools ARE allowed here, offline
only. MeCab + UniDic generate the deterministic labels (segmentation, ruby,
lemma, pos→§6 mapping); an LLM teacher generates the judgment labels
(contextual glosses, translations, grammar roles). None of it may be imported
by src/kanjiland — the runtime inference path stays NLP-dependency-free
(rule #1). An import-guard test enforces this once the first tool lands.
