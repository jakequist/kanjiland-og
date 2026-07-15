"""fastText language identification for corpus filtering (offline only).

Cheap script checks (``filters.script_ok``) catch swapped columns and romaji,
but they can't tell "this Japanese-script line is actually mangled" or catch
code-switched English lines that happen to contain a stray kanji. fastText's
``lid.176`` model — a tiny, fast linear classifier over character n-grams,
trained on 176 languages — does the real language ID. We use it to require
that the Japanese side classifies as `ja` and the English side as `en`, each
above a confidence threshold.

**Offline-only, never on the runtime path.** This is a data-generation tool
(cf. ADR-014). It is imported by the pipeline, which lives under
``src/kanjiland/data`` but is only ever run offline to prepare corpora; the
shipped model/inference code never imports it. fastText is *not* a Japanese
morphological analyzer, so it is unaffected by the "no runtime NLP deps" rule
(ADR-007 / rule #1) either way.

The model file (~126 MB) is downloaded on first use and cached under
``data/models/``; it is not committed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Facebook's official pretrained language-ID model. The .bin is the full,
# most-accurate model; the .ftz quantized variant is ~1 MB but slightly less
# accurate — we prefer accuracy here since this runs offline.
MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
DEFAULT_MODEL_PATH = Path("data/models/lid.176.bin")


def ensure_model(path: Path = DEFAULT_MODEL_PATH, url: str = MODEL_URL) -> Path:
    """Download the lid.176 model to ``path`` if not already present."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    import requests  # part of the `data` extra; keep the import local

    print(f"downloading fastText lid.176 -> {path} ...")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return path


@dataclass
class LangIDConfig:
    enabled: bool = True
    model_path: Path = DEFAULT_MODEL_PATH
    # Minimum classifier confidence to accept a side as its expected language.
    # 0.5 is deliberately lenient: short sentences are genuinely hard to ID, and
    # the script/ratio filters already removed the obvious junk — we mainly want
    # to catch confident *wrong*-language predictions.
    min_confidence: float = 0.5
    # Skip language ID entirely when either side is shorter than this many
    # characters. fastText confuses short kanji-heavy Japanese ("何て？") with
    # Chinese, so on very short text it produces confident *wrong* answers and
    # would drop good pairs. Below this length we trust the script/ratio filters
    # instead. (See ADR-013.)
    min_chars: int = 10


class LanguageIdentifier:
    """Lazily-loaded wrapper around the fastText lid.176 model."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self._model = None  # loaded on first predict()

    def _load(self):
        if self._model is None:
            import fasttext  # heavy/optional dep; import only when actually used

            # fastText prints a spurious load warning to stderr; silence it.
            fasttext.FastText.eprint = staticmethod(lambda *a, **k: None)
            self._model = fasttext.load_model(str(ensure_model(self.model_path)))
        return self._model

    def predict(self, text: str) -> tuple[str, float]:
        """Return (language_code, confidence) for a single line of text."""
        model = self._load()
        # fastText treats a newline as a document separator and errors on it;
        # our normalizer already strips newlines, but be defensive.
        labels, probs = model.predict(text.replace("\n", " "), k=1)
        lang = labels[0].removeprefix("__label__")
        return lang, float(probs[0])


def langid_ok(
    ja: str,
    en: str,
    identifier: LanguageIdentifier,
    cfg: LangIDConfig,
) -> bool:
    """True if the ja side reads as Japanese and the en side as English, each
    at or above the confidence threshold. Short sentences bypass the check
    (see ``LangIDConfig.min_chars``)."""
    if len(ja) < cfg.min_chars or len(en) < cfg.min_chars:
        return True
    ja_lang, ja_conf = identifier.predict(ja)
    if ja_lang != "ja" or ja_conf < cfg.min_confidence:
        return False
    en_lang, en_conf = identifier.predict(en)
    if en_lang != "en" or en_conf < cfg.min_confidence:
        return False
    return True
