# M6 — Teacher bake-off

Ja→En on 120 held-out kftt-test sentences (seed 1), scored vs the human reference. **COMET** (semantic, ADR-008 headline) is the trusted ranking; chrF/BLEU are shown but *undersell fluent paraphrase* — e.g. the gpt-5.6 models translate Japanese proper names correctly (立松和平→Wahei Tatematsu) where the cheaper models hallucinate readings (→'Kazuhei Tatsumatsu'), yet score lower on n-gram overlap. Cost from measured tokens; gpt-5.6 prices pending.

| model | COMET | chrF | BLEU | in tok | out tok | errors |
|:--|--:|--:|--:|--:|--:|--:|
| gpt-5.6-sol | 0.8017 | 48.14 | 18.71 | 84 | 39 | 0 |
| gpt-5.6-luna | 0.7998 | 48.50 | 18.63 | 84 | 40 | 0 |
| gpt-5.6-terra | 0.7990 | 46.89 | 17.53 | 84 | 39 | 0 |
| gpt-4.1-mini | 0.7968 | 48.91 | 20.45 | 85 | 36 | 0 |
| gpt-5-mini | 0.7812 | 46.37 | 17.37 | 84 | 46 | 0 |

**Read:** all three gpt-5.6 models > gpt-4.1-mini on COMET; gpt-5-mini dominated (worst COMET + wasted reasoning tokens) — dropped. Teacher choice is luna vs sol vs gpt-4.1-mini, decided by gpt-5.6 pricing + latency (luna 19s vs sol 44s per 120 @ 8 workers).

