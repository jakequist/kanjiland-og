"""Batch annotation driver (M7, offline). Runs the full silver-data pipeline over
a large sentence set via the OpenAI Batch API (half rate, 24h window):

  submit: sentence -> deterministic ⟨T⟩ -> chat request (annotation prompt) -> batch
  fetch:  download -> RE-TAG each sentence deterministically -> assemble teacher
          JSON into a Document -> LINTER GATE -> keep only clean docs as silver

Morphs are re-derived at fetch time rather than persisted: UniDic tagging is
deterministic, so tag(sentence) reproduces the exact ⟨T⟩ skeleton the request was
built from. The linter is the data gate (ADR-007); its pass-rate is a headline M7
metric, and the drop reasons tell us where the teacher struggles.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "teacher"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from assemble import assemble_and_lint  # noqa: E402
from batch import (  # noqa: E402 — reuse the M6 Batch plumbing
    MAX_PER_BATCH, create_batch, download_content, get_batch, upload_file,
)
from deterministic import tag  # noqa: E402
from kanjiland.format.serializer import serialize  # noqa: E402
from teacher import system_prompt, user_prompt  # noqa: E402


def _request(i: int, sentence: str, model: str) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_prompt(sentence, tag(sentence))},
        ],
        "response_format": {"type": "json_object"},
    }
    if model.startswith("gpt-5.6"):
        body["reasoning_effort"] = "none"
    return {"custom_id": f"row-{i}", "method": "POST", "url": "/v1/chat/completions", "body": body}


def submit(sentences: list[str], model: str, work: Path, key: str) -> None:
    work.mkdir(parents=True, exist_ok=True)
    (work / "sentences.txt").write_text("\n".join(sentences), encoding="utf-8")
    manifest = {"model": model, "total": len(sentences), "batches": []}
    for start in range(0, len(sentences), MAX_PER_BATCH):
        chunk = sentences[start:start + MAX_PER_BATCH]
        jl = work / f"requests_{start}.jsonl"
        with jl.open("w", encoding="utf-8") as f:
            for j, s in enumerate(chunk):
                f.write(json.dumps(_request(start + j, s, model), ensure_ascii=False) + "\n")
        fid = upload_file(jl, key)
        batch = create_batch(fid, key)
        manifest["batches"].append({"id": batch["id"], "start": start, "count": len(chunk)})
        print(f"  chunk {start}-{start+len(chunk)} | batch={batch['id']} | {batch['status']}")
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"submitted {len(sentences)} annotation requests -> {work}/manifest.json")


def fetch(work: Path, out: Path, key: str, poll: int) -> None:
    man = json.loads((work / "manifest.json").read_text())
    sentences = (work / "sentences.txt").read_text(encoding="utf-8").splitlines()
    raw: dict[int, str] = {}
    for bm in man["batches"]:
        while True:
            b = get_batch(bm["id"], key)
            print(f"  {bm['id']} {b['status']} {b.get('request_counts')}", flush=True)
            if b["status"] in ("completed", "failed", "expired", "cancelled"):
                break
            time.sleep(poll)
        if b["status"] != "completed":
            print(f"  WARNING {bm['id']} ended {b['status']}")
            continue
        for line in download_content(b["output_file_id"], key).decode().splitlines():
            o = json.loads(line)
            i = int(o["custom_id"].split("-")[1])
            try:
                raw[i] = o["response"]["body"]["choices"][0]["message"]["content"]
            except Exception:  # noqa: BLE001
                pass

    npass = 0
    drops: dict[str, int] = {}
    out.parent.mkdir(parents=True, exist_ok=True)
    audit = []
    with out.open("w", encoding="utf-8") as fh:
        for i, s in enumerate(sentences):
            content = raw.get(i)
            if content is None:
                drops["api_fail"] = drops.get("api_fail", 0) + 1
                continue
            try:
                ann = json.loads(content)
                doc, viols = assemble_and_lint(s, tag(s), ann)
            except Exception:  # noqa: BLE001
                drops["assemble_err"] = drops.get("assemble_err", 0) + 1
                continue
            if viols:
                for v in viols:
                    drops[f"inv{v.invariant}"] = drops.get(f"inv{v.invariant}", 0) + 1
                continue
            npass += 1
            fh.write(json.dumps({"ja": s, "wire": serialize(doc)}, ensure_ascii=False) + "\n")
            if len(audit) < 40:
                audit.append(s)
    total = len(sentences)
    (out.parent / "gate_report.json").write_text(json.dumps(
        {"total": total, "passed": npass, "pass_rate": round(npass / max(total, 1), 4),
         "drops": drops, "audit_sample": audit}, ensure_ascii=False, indent=2))
    print(f"\nGATE: {npass}/{total} passed ({100*npass/max(total,1):.1f}%) -> {out}")
    print("drop reasons:", drops)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--input", required=True, type=Path)
    s.add_argument("--n", type=int, default=10000)
    s.add_argument("--model", default="gpt-5.6-luna")
    s.add_argument("--work", type=Path, default=Path("data/processed/m7_annot/batch"))
    g = sub.add_parser("fetch")
    g.add_argument("--work", type=Path, default=Path("data/processed/m7_annot/batch"))
    g.add_argument("--out", type=Path, default=Path("data/processed/m7_annot/silver.jsonl"))
    g.add_argument("--poll", type=int, default=180)
    args = ap.parse_args()
    key = os.environ.get("OPENAI_API_KEY") or sys.exit("set OPENAI_API_KEY")

    if args.cmd == "submit":
        sents = [ln for ln in args.input.read_text(encoding="utf-8").splitlines() if ln.strip()][: args.n]
        submit(sents, args.model, args.work, key)
    else:
        fetch(args.work, args.out, key, args.poll)


if __name__ == "__main__":
    main()
