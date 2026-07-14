"""The Tokenizer: special tokens + byte-level BPE, with encode/decode/save/load.

Layers three things on top of the raw BPE core (``bpe.py``):

1. **Special tokens** occupy the lowest ids and are atomic. PAD/BOS/EOS are
   control tokens with no text surface; the nine PUA separators from the wire
   format (FORMAT_SPEC §2) are special tokens whose surface is their PUA
   character. Encoding splits on those characters *first*, so a separator can
   never be merged into a neighbouring byte and always maps to one id.

2. **The id offset.** Final id layout::

       [0 .. S-1]          special tokens (S of them)
       [S .. S+255]        the 256 raw bytes
       [S+256 .. vocab-1]  learned merges

   so ``final_id = byte_offset + internal_id`` for every non-special token,
   where ``internal_id`` is the byte-id-space id the BPE core uses.

3. **Round-trip decode.** Byte ids are accumulated and UTF-8 decoded at
   special-token boundaries, so multi-byte characters that straddle several
   BPE tokens reconstruct exactly.

Guarantee: ``decode(encode(text)) == text`` for any ``str`` (see tests).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from ..format import separators as sep
from .bpe import BYTE_BASE, Pair, train_bpe, word_freqs_from_pretokens
from .pretokenize import pretokenize

# Canonical special-token table. Order fixes the ids and must stay stable
# (changing it invalidates every trained tokenizer). Control tokens first,
# then the PUA separators in wire-format order. ``surface`` is the text a
# token stands for, or None for control tokens that never appear in text.
_CONTROL_TOKENS = ("PAD", "BOS", "EOS")
_PUA_TOKENS = (
    ("HEADER", sep.HEADER),
    ("TOKEN", sep.TOKEN),
    ("WORD", sep.WORD),
    ("SENTENCE", sep.SENTENCE),
    ("GRAMMAR", sep.GRAMMAR),
    ("PARAGRAPH", sep.PARAGRAPH),
    ("LIST_SEP", sep.LIST_SEP),
    ("FIELD_SEP", sep.FIELD_SEP),
    ("RECORD_END", sep.RECORD_END),
)


def _default_special_tokens() -> list[dict]:
    specials: list[dict] = []
    for name in _CONTROL_TOKENS:
        specials.append({"name": name, "id": len(specials), "surface": None})
    for name, ch in _PUA_TOKENS:
        specials.append({"name": name, "id": len(specials), "surface": ch})
    return specials


class Tokenizer:
    """Byte-level BPE tokenizer with wire-format special tokens."""

    VERSION = 1
    PRETOKENIZER = "script_aware"

    def __init__(self, merges: list[Pair], special_tokens: list[dict] | None = None):
        self.special_tokens = special_tokens or _default_special_tokens()
        self.byte_offset = len(self.special_tokens)
        self.merges: list[Pair] = [tuple(m) for m in merges]

        # Encode side: (a, b) -> rank (== which merge, lowest applies first).
        self.merge_ranks: dict[Pair, int] = {m: i for i, m in enumerate(self.merges)}

        # Special-token lookups.
        self._char_to_special: dict[str, int] = {}
        self._id_to_surface: dict[int, str | None] = {}
        self._id_to_name: dict[int, str] = {}
        for spec in self.special_tokens:
            self._id_to_surface[spec["id"]] = spec["surface"]
            self._id_to_name[spec["id"]] = spec["name"]
            if spec["surface"] is not None:
                self._char_to_special[spec["surface"]] = spec["id"]
        self.pad_id = self._name_to_id("PAD")
        self.bos_id = self._name_to_id("BOS")
        self.eos_id = self._name_to_id("EOS")

        # Decode side: internal id -> the exact bytes it expands to.
        self._id_to_bytes: dict[int, bytes] = {b: bytes([b]) for b in range(BYTE_BASE)}
        for k, (a, b) in enumerate(self.merges):
            self._id_to_bytes[BYTE_BASE + k] = self._id_to_bytes[a] + self._id_to_bytes[b]

        self._encode_cache: dict[str, list[int]] = {}

    def _name_to_id(self, name: str) -> int:
        for spec in self.special_tokens:
            if spec["name"] == name:
                return spec["id"]
        raise KeyError(name)

    # --- construction ----------------------------------------------------

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        *,
        verbose: bool = False,
    ) -> "Tokenizer":
        """Train on an iterable of raw text lines to a target ``vocab_size``.

        ``vocab_size`` counts everything: special tokens + 256 bytes + merges.
        """
        specials = _default_special_tokens()
        num_merges = vocab_size - len(specials) - BYTE_BASE
        if num_merges < 0:
            raise ValueError(
                f"vocab_size={vocab_size} too small; need at least "
                f"{len(specials) + BYTE_BASE} for specials + bytes"
            )

        def _pretokens() -> Iterable[str]:
            for line in texts:
                yield from pretokenize(line)

        word_freqs = word_freqs_from_pretokens(_pretokens())
        merges = train_bpe(word_freqs, num_merges, verbose=verbose)
        return cls(merges=merges, special_tokens=specials)

    # --- encode / decode -------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self.byte_offset + BYTE_BASE + len(self.merges)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode ``text`` to token ids. Splits on special-token characters."""
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)

        buf: list[str] = []
        for ch in text:
            sid = self._char_to_special.get(ch)
            if sid is None:
                buf.append(ch)
                continue
            if buf:
                self._encode_chunk("".join(buf), ids)
                buf.clear()
            ids.append(sid)
        if buf:
            self._encode_chunk("".join(buf), ids)

        if add_eos:
            ids.append(self.eos_id)
        return ids

    def _encode_chunk(self, chunk: str, out: list[int]) -> None:
        offset = self.byte_offset
        for tok in pretokenize(chunk):
            for internal in self._bpe(tok):
                out.append(offset + internal)

    def _bpe(self, tok: str) -> list[int]:
        """Byte-level BPE encode of one pre-token, in internal id space."""
        cached = self._encode_cache.get(tok)
        if cached is not None:
            return cached

        seq = list(tok.encode("utf-8"))
        while len(seq) >= 2:
            # Merge the lowest-rank (earliest-learned) adjacent pair present.
            best_rank: int | None = None
            best_pair: Pair | None = None
            for pair in zip(seq, seq[1:]):
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, pair
            if best_pair is None:
                break
            a, b = best_pair
            new_id = BYTE_BASE + best_rank
            merged: list[int] = []
            i, n = 0, len(seq)
            while i < n:
                if i < n - 1 and seq[i] == a and seq[i + 1] == b:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(seq[i])
                    i += 1
            seq = merged

        self._encode_cache[tok] = seq
        return seq

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        """Decode token ids back to text. Inverse of ``encode`` on valid ids."""
        pieces: list[str] = []
        byte_buf = bytearray()

        def flush() -> None:
            if byte_buf:
                pieces.append(byte_buf.decode("utf-8", errors="replace"))
                byte_buf.clear()

        for i in ids:
            if i in self._id_to_surface:  # special token
                flush()
                surface = self._id_to_surface[i]
                if surface is not None:
                    pieces.append(surface)
                elif not skip_special:
                    pieces.append(f"<{self._id_to_name[i]}>")
                continue
            internal = i - self.byte_offset
            b = self._id_to_bytes.get(internal)
            if b is None:  # out-of-range id (e.g. malformed model output)
                flush()
                pieces.append("�")
            else:
                byte_buf += b
        flush()
        return "".join(pieces)

    # --- persistence -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": self.VERSION,
            "model": "byte-bpe",
            "pretokenizer": self.PRETOKENIZER,
            "vocab_size": self.vocab_size,
            "byte_offset": self.byte_offset,
            "special_tokens": self.special_tokens,
            "merges": [list(m) for m in self.merges],
        }

    def save(self, path: str | Path) -> None:
        """Write the tokenizer as JSON — metadata indented, one merge per line.

        Standard ``json.dump(indent=2)`` explodes each ``[a, b]`` merge across
        four lines; on a 32k vocab that is ~127k lines of noise. Emitting one
        pair per line keeps the artifact compact and greppable while staying
        valid JSON. Deterministic output for reproducible diffs.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = self.to_dict()
        head_keys = ["version", "model", "pretokenizer", "vocab_size", "byte_offset"]
        parts = [f"  {json.dumps(k)}: {json.dumps(d[k])}" for k in head_keys]
        parts.append('  "special_tokens": ' + json.dumps(d["special_tokens"], ensure_ascii=False))
        merge_lines = ",\n".join(f"    [{a}, {b}]" for a, b in d["merges"])
        parts.append('  "merges": [\n' + merge_lines + "\n  ]" if merge_lines else '  "merges": []')
        path.write_text("{\n" + ",\n".join(parts) + "\n}\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        with Path(path).open(encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != cls.VERSION:
            raise ValueError(f"unsupported tokenizer version: {data.get('version')}")
        merges = [tuple(m) for m in data["merges"]]
        return cls(merges=merges, special_tokens=data["special_tokens"])
