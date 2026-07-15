"""Evaluation harness (M4): chrF / SacreBLEU / COMET + a seed-aware results store.

from kanjiland.eval import metrics, results
from kanjiland.eval.translate import translate
"""

from . import baseline, metrics, results, translate

__all__ = ["metrics", "results", "translate", "baseline"]
