"""Measure DocGraph on a real PDF and write proof files.

Usage:
    python benchmark.py path/to/resume.pdf
    python benchmark.py path/to/resume.pdf --pause 3      # more delay between questions (rate limits)

Writes docs/benchmark.md and docs/benchmark_results.json.
The golden set below is written for Mratunjai's resume. For another PDF, edit GOLDEN.
Only quote numbers that this script printed on your machine.
"""
import argparse
import json
import math
import platform
import statistics
import sys
import tempfile
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import rag_graph as rg

if hasattr(sys.stdout, "reconfigure"):          # Windows consoles choke on unicode answers otherwise
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# kind "answer": passes if the answer contains ANY of the `expect` strings (case-insensitive)
# kind "idk"   : the document does not contain the answer, so the app must refuse
GOLDEN = [
    {"q": "What is the candidate's name?", "lang": "en", "kind": "answer", "expect": ["mratunjai"]},
    {"q": "Which university did he do his MCA from?", "lang": "en", "kind": "answer", "expect": ["upes", "petroleum"]},
    {"q": "What throughput did the load balancer achieve?", "lang": "en", "kind": "answer", "expect": ["15k", "15,000", "15 k"]},
    {"q": "What was the p99 latency of the load balancer?", "lang": "en", "kind": "answer", "expect": ["22 ms", "22ms"]},
    {"q": "Which routing algorithm is the primary one in the load balancer?", "lang": "en", "kind": "answer", "expect": ["round robin"]},
    {"q": "Which library does the DPI system use to capture packets?", "lang": "en", "kind": "answer", "expect": ["libpcap"]},
    {"q": "How many algorithms does the 3D graph visualizer simulate?", "lang": "en", "kind": "answer", "expect": ["16"]},
    {"q": "How many LeetCode problems has he solved?", "lang": "en", "kind": "answer", "expect": ["400"]},
    {"q": "How much faster is the vector database than brute force?", "lang": "en", "kind": "answer", "expect": ["60%", "60 %", "60 percent"]},
    {"q": "Which embedding model does the vector database RAG pipeline use?", "lang": "en", "kind": "answer", "expect": ["nomic"]},
    {"q": "लोड बैलेंसर कितने requests per second संभालता है?", "lang": "auto", "kind": "answer", "expect": ["15"]},
    {"q": "How many years of work experience does he have?", "lang": "en", "kind": "idk", "expect": []},
    {"q": "What is his salary?", "lang": "en", "kind": "idk", "expect": []},
    {"q": "Does he know Rust?", "lang": "en", "kind": "idk", "expect": []},
    {"q": "What is his date of birth?", "lang": "en", "kind": "idk", "expect": []},
    {"q": "उसके पास कितने साल का काम का अनुभव है?", "lang": "auto", "kind": "idk", "expect": []},
]


def pct(values, p):
    """Nearest-rank percentile."""
    if not values:
        return None
    s = sorted(values)
    return s[max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))]


def norm(text: str) -> str:
    """Fold typographic variants (narrow no-break space, non-breaking hyphen...) so matching tests content, not formatting."""
    text = unicodedata.normalize("NFKC", text).lower().replace("\u2011", "-").replace("'", "").replace("\u2019", "")
    return " ".join(text.split())


def judge(item: dict, answer: str) -> bool:
    a = norm(answer)
    if item["kind"] == "idk":
        return any(norm(m) in a for m in rg._IDK_MARKERS)
    return any(norm(e) in a for e in item["expect"])


def run_question(question: str, lang: str, doc_id: str):
    start = time.perf_counter()
    first, text, resets = None, "", 0
    for ev in rg.stream_answer(question, [], doc_id, lang):
        if ev["type"] == "reset":
            resets += 1
            text = ""
            continue
        if first is None:
            first = time.perf_counter() - start
        text += ev["text"]
    return text, first, time.perf_counter() - start, resets


def fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between questions")
    ap.add_argument("--out", default="docs")
    args = ap.parse_args(argv)

    pdf_path = Path(args.pdf)
    pdf_bytes = pdf_path.read_bytes()

    # ---- indexing: cold (no cache) then warm (from disk cache) ----
    rg.INDEX_DIR = Path(tempfile.mkdtemp(prefix="docgraph_bench_"))
    rg._STORES.clear()
    rate_limited = []

    def on_progress(msg, frac):
        if "Rate limit" in msg:
            rate_limited.append(msg)

    t0 = time.perf_counter()
    doc_id = rg.index_pdf(pdf_bytes, progress=on_progress)
    cold = time.perf_counter() - t0
    n_chunks = len(rg._STORES[doc_id].store)

    rg._STORES.clear()
    t0 = time.perf_counter()
    rg.index_pdf(pdf_bytes, progress=lambda m, f: None)
    warm = time.perf_counter() - t0
    store = rg.get_store(doc_id)
    print(f"indexed {n_chunks} chunks: cold {cold:.1f}s (rate-limit waits: {len(rate_limited)}), warm {warm:.2f}s")

    # ---- warm-up call (not counted): opens connections ----
    run_question("What is the candidate's name?", "en", doc_id)

    # ---- questions ----
    rows = []
    for item in GOLDEN:
        q = item["q"]
        retrieval, hit = None, None
        if item["lang"] == "en":
            t0 = time.perf_counter()
            docs = store.similarity_search(q, k=rg.DEFAULT_K)
            retrieval = time.perf_counter() - t0
            if item["kind"] == "answer":
                blob = norm(" ".join(d.page_content for d in docs))
                hit = any(norm(e) in blob for e in item["expect"])
        text, ttft, total, resets = run_question(q, item["lang"], doc_id)
        ok = judge(item, text)
        rows.append({"question": q, "kind": item["kind"], "lang": item["lang"], "ok": ok,
                     "retrieval_hit": hit, "retrieval_s": retrieval, "ttft_s": ttft,
                     "total_s": total, "retried": resets > 0, "answer": text[:300]})
        print(f"[{'PASS' if ok else 'FAIL'}] ttft={fmt(ttft)}s total={fmt(total)}s retry={resets > 0} | {q[:60]}")
        if not ok:
            print("        answer was:", ascii(text[:200]))
        time.sleep(args.pause)

    # ---- summary ----
    answerable = [r for r in rows if r["kind"] == "answer"]
    refusals = [r for r in rows if r["kind"] == "idk"]
    ret_rows = [r for r in answerable if r["retrieval_hit"] is not None]
    ttfts = [r["ttft_s"] for r in rows if r["ttft_s"] is not None]
    totals = [r["total_s"] for r in rows]
    retrievals = [r["retrieval_s"] for r in rows if r["retrieval_s"] is not None]
    summary = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "pdf": pdf_path.name, "chunks": n_chunks,
        "index_cold_s": cold, "index_cold_rate_limit_waits": len(rate_limited), "index_warm_s": warm,
        "retrieval_median_s": statistics.median(retrievals) if retrievals else None,
        "retrieval_p95_s": pct(retrievals, 95),
        "ttft_median_s": statistics.median(ttfts) if ttfts else None, "ttft_p95_s": pct(ttfts, 95),
        "total_median_s": statistics.median(totals), "total_p95_s": pct(totals, 95),
        "answerable_correct": sum(r["ok"] for r in answerable), "answerable_total": len(answerable),
        "refused_correct": sum(r["ok"] for r in refusals), "refused_total": len(refusals),
        "retrieval_hits": sum(bool(r["retrieval_hit"]) for r in ret_rows), "retrieval_total": len(ret_rows),
        "retries_triggered": sum(r["retried"] for r in rows),
        "config": {"llm": rg.LLM_MODEL, "reasoning_effort": rg.LLM_REASONING_EFFORT if "gpt-oss" in rg.LLM_MODEL else "n/a", "embeddings": rg.EMBED_MODEL, "chunk_size": rg.CHUNK_SIZE,
                   "chunk_overlap": rg.CHUNK_OVERLAP, "k": rg.DEFAULT_K, "retry_k": rg.RETRY_K},
        "machine": f"{platform.system()} {platform.release()}, Python {platform.python_version()}",
    }

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "benchmark_results.json").write_text(
        json.dumps({"summary": summary, "questions": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    s = summary
    retrieval_cell = (f"n/a (document has only {s['chunks']} chunks, so all are retrieved)"
                      if s['chunks'] <= s['config']['k'] else f"{s['retrieval_hits']}/{s['retrieval_total']}")
    md = f"""# DocGraph benchmark

Run on {s['date']} ({s['machine']}), free-tier Gemini and Groq APIs, over the internet.
Document: `{s['pdf']}` ({s['chunks']} chunks). Config: {s['config']['llm']} (reasoning effort: {s['config']['reasoning_effort']}), {s['config']['embeddings']},
chunk {s['config']['chunk_size']}/{s['config']['chunk_overlap']}, k={s['config']['k']} (retry k={s['config']['retry_k']}).
Reproduce: `python benchmark.py <pdf>`. Numbers depend on network and free-tier load; sample is small ({len(rows)} questions). Answers are Unicode-normalised (NFKC) before matching, so typographic spaces and hyphens do not count as errors.

| Metric | Result |
|---|---|
| Index a new PDF (embed, cold) | {fmt(s['index_cold_s'], 1)} s ({s['index_cold_rate_limit_waits']} rate-limit waits included) |
| Load a cached index (warm) | {fmt(s['index_warm_s'])} s |
| Retrieval incl. query embedding (median / p95) | {fmt(s['retrieval_median_s'])} s / {fmt(s['retrieval_p95_s'])} s |
| Time to first token (median / p95) | {fmt(s['ttft_median_s'])} s / {fmt(s['ttft_p95_s'])} s |
| Full answer time (median / p95) | {fmt(s['total_median_s'])} s / {fmt(s['total_p95_s'])} s |
| Correct answers on answerable questions | {s['answerable_correct']}/{s['answerable_total']} |
| Correct refusals ("I dont know") on out-of-scope questions | {s['refused_correct']}/{s['refused_total']} |
| Retrieval returned the needed fact in top-{s['config']['k']} | {retrieval_cell} |
| Questions where the retry edge fired | {s['retries_triggered']}/{len(rows)} |

## Per question

| Result | Question | Time to first token | Total | Retry |
|---|---|---|---|---|
""" + "\n".join(
        f"| {'PASS' if r['ok'] else 'FAIL'} | {r['question']} | {fmt(r['ttft_s'])} s | {fmt(r['total_s'])} s | {'yes' if r['retried'] else 'no'} |"
        for r in rows) + "\n"
    (out / "benchmark.md").write_text(md, encoding="utf-8")

    print("\n" + md)
    print(f"Saved {out / 'benchmark.md'} and {out / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
