"""OpenAI Batch API driver for teacher translation (M6, offline — ADR-007).

Batch is the right tool for bulk KD data generation: you upload ALL requests as
one file, OpenAI processes them within a 24h window, and it bills at HALF the
synchronous rate (for the gpt-5.x flagships, Batch == Flex == 0.5× Standard).
We have no latency requirement offline, so that discount is free — and there's no
rate-limit juggling, the queue absorbs the whole job.

Flow (all raw HTTP, no SDK):
  1. write a JSONL where each line is one chat-completions request (custom_id = row
     index, so we can re-align the results to the input order)
  2. upload it via the Files API (purpose="batch")
  3. create a batch over that file
  4. poll until status == completed
  5. download the output file, map custom_id -> translation

Submit and fetch are separate CLI verbs because a 500k-sentence batch can take
minutes to hours — you submit, note the batch id, and fetch later.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from translate import SYSTEM_PROMPT, _effort_for  # noqa: E402 — reuse the exact prompt/effort logic

BASE = "https://api.openai.com/v1"


def _req(method: str, url: str, api_key: str, *, data=None, headers=None, timeout=120.0):
    h = {"Authorization": f"Bearer {api_key}"}
    if headers:
        h.update(headers)
    r = urllib.request.Request(url, data=data, headers=h, method=method)
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.load(resp)


def build_requests_jsonl(sentences: list[str], model: str, path: Path) -> None:
    """One chat-completions request per sentence; custom_id encodes input order."""
    effort = _effort_for(model)
    with path.open("w", encoding="utf-8") as f:
        for i, s in enumerate(sentences):
            body = {"model": model, "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": s},
            ]}
            if effort is not None:
                body["reasoning_effort"] = effort
            f.write(json.dumps({
                "custom_id": f"row-{i}", "method": "POST",
                "url": "/v1/chat/completions", "body": body,
            }, ensure_ascii=False) + "\n")


def upload_file(path: Path, api_key: str) -> str:
    """multipart/form-data upload to the Files API (purpose=batch) -> file id.

    Built by hand to avoid a `requests` dependency: a multipart body is just
    boundary-delimited parts, one for the `purpose` field and one for the file.
    """
    boundary = f"----kanjiland{uuid.uuid4().hex}"
    payload = path.read_bytes()
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n".encode(),
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
         f"filename=\"{path.name}\"\r\nContent-Type: application/jsonl\r\n\r\n").encode(),
        payload, b"\r\n", f"--{boundary}--\r\n".encode(),
    ]
    body = b"".join(parts)
    r = _req("POST", f"{BASE}/files", api_key, data=body,
             headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return r["id"]


def create_batch(file_id: str, api_key: str) -> dict:
    body = json.dumps({
        "input_file_id": file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
    }).encode()
    return _req("POST", f"{BASE}/batches", api_key, data=body,
                headers={"Content-Type": "application/json"})


def get_batch(batch_id: str, api_key: str) -> dict:
    return _req("GET", f"{BASE}/batches/{batch_id}", api_key)


def download_content(file_id: str, api_key: str) -> bytes:
    r = urllib.request.Request(f"{BASE}/files/{file_id}/content",
                               headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(r, timeout=300) as resp:
        return resp.read()


def parse_results(jsonl: bytes) -> dict[int, str | None]:
    """custom_id 'row-N' -> translation (None on per-request error)."""
    out: dict[int, str | None] = {}
    for line in jsonl.decode().splitlines():
        o = json.loads(line)
        i = int(o["custom_id"].split("-")[1])
        try:
            out[i] = o["response"]["body"]["choices"][0]["message"]["content"].strip()
        except Exception:  # noqa: BLE001 — record the failure, don't abort the whole file
            out[i] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="upload sentences + create a batch")
    s.add_argument("--input", required=True, type=Path, help="one Japanese sentence per line")
    s.add_argument("--model", required=True)
    s.add_argument("--work", type=Path, default=Path("data/processed/_m6_batch"))

    f = sub.add_parser("status", help="print batch status + request counts")
    f.add_argument("--batch", required=True)

    g = sub.add_parser("fetch", help="poll until done, download, write translations")
    g.add_argument("--batch", required=True)
    g.add_argument("--out", required=True, type=Path)
    g.add_argument("--poll", type=int, default=60, help="seconds between status polls")

    args = ap.parse_args()
    key = os.environ.get("OPENAI_API_KEY") or sys.exit("set OPENAI_API_KEY")

    if args.cmd == "submit":
        args.work.mkdir(parents=True, exist_ok=True)
        sents = [ln for ln in args.input.read_text(encoding="utf-8").splitlines() if ln.strip()]
        jl = args.work / "requests.jsonl"
        build_requests_jsonl(sents, args.model, jl)
        fid = upload_file(jl, key)
        batch = create_batch(fid, key)
        (args.work / "batch_id.txt").write_text(batch["id"])
        print(f"submitted {len(sents)} sentences | file={fid} | batch={batch['id']} | status={batch['status']}")
        print(f"fetch later: uv run python tools/teacher/batch.py fetch --batch {batch['id']} --out <path>")

    elif args.cmd == "status":
        b = get_batch(args.batch, key)
        print(f"status={b['status']} counts={b.get('request_counts')}")

    elif args.cmd == "fetch":
        while True:
            b = get_batch(args.batch, key)
            st = b["status"]
            print(f"  status={st} counts={b.get('request_counts')}", flush=True)
            if st in ("completed", "failed", "expired", "cancelled"):
                break
            time.sleep(args.poll)
        if b["status"] != "completed":
            sys.exit(f"batch ended as {b['status']}")
        res = parse_results(download_content(b["output_file_id"], key))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            for i in range(len(res)):
                fh.write(json.dumps({"i": i, "en": res.get(i)}, ensure_ascii=False) + "\n")
        n_ok = sum(1 for v in res.values() if v)
        print(f"wrote {n_ok}/{len(res)} translations -> {args.out}")


if __name__ == "__main__":
    main()
