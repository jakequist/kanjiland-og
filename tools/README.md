# tools/ — offline-only helpers

Nothing in this directory may be imported by src/kanjiland (enforce with a test
once the first tool lands). This is where silver-data generation and teacher
distillation live.

NOTE (ADR-007, OPEN): whether classical NLP tools (MeCab/UniDic) are allowed
here for offline label generation is undecided. Do not add them without an
explicit decision recorded in docs/DECISIONS.md.
