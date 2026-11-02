# DocGraph benchmark

Run on 2026-11-02 09:31 (Windows 10, Python 3.14.3), free-tier Gemini and Groq APIs, over the internet.
Document: `Mratunjai_singh_Resume.pdf` (8 chunks). Config: openai/gpt-oss-120b (reasoning effort: low), models/gemini-embedding-001,
chunk 800/100, k=8 (retry k=12).
Reproduce: `python benchmark.py <pdf>`. Numbers depend on network and free-tier load; sample is small (16 questions). Answers are Unicode-normalised (NFKC) before matching, so typographic spaces and hyphens do not count as errors.

| Metric | Result |
|---|---|
| Index a new PDF (embed, cold) | 2.1 s (0 rate-limit waits included) |
| Load a cached index (warm) | 0.05 s |
| Retrieval incl. query embedding (median / p95) | 0.89 s / 0.92 s |
| Time to first token (median / p95) | 1.13 s / 1.96 s |
| Full answer time (median / p95) | 1.14 s / 2.02 s |
| Correct answers on answerable questions | 11/11 |
| Correct refusals ("I dont know") on out-of-scope questions | 5/5 |
| Retrieval returned the needed fact in top-8 | n/a (document has only 8 chunks, so all are retrieved) |
| Questions where the retry edge fired | 0/16 |

## Per question

| Result | Question | Time to first token | Total | Retry |
|---|---|---|---|---|
| PASS | What is the candidate's name? | 1.04 s | 1.06 s | no |
| PASS | Which university did he do his MCA from? | 0.96 s | 1.06 s | no |
| PASS | What throughput did the load balancer achieve? | 1.01 s | 1.05 s | no |
| PASS | What was the p99 latency of the load balancer? | 1.13 s | 1.17 s | no |
| PASS | Which routing algorithm is the primary one in the load balancer? | 1.16 s | 1.19 s | no |
| PASS | Which library does the DPI system use to capture packets? | 0.93 s | 0.96 s | no |
| PASS | How many algorithms does the 3D graph visualizer simulate? | 1.14 s | 1.17 s | no |
| PASS | How many LeetCode problems has he solved? | 1.22 s | 1.25 s | no |
| PASS | How much faster is the vector database than brute force? | 1.08 s | 1.12 s | no |
| PASS | Which embedding model does the vector database RAG pipeline use? | 1.10 s | 1.14 s | no |
| PASS | लोड बैलेंसर कितने requests per second संभालता है? | 1.96 s | 2.02 s | no |
| PASS | How many years of work experience does he have? | 1.12 s | 1.13 s | no |
| PASS | What is his salary? | 1.23 s | 1.24 s | no |
| PASS | Does he know Rust? | 1.12 s | 1.13 s | no |
| PASS | What is his date of birth? | 1.13 s | 1.14 s | no |
| PASS | उसके पास कितने साल का काम का अनुभव है? | 1.77 s | 1.78 s | no |
