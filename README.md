# AIMO controlled-robustness inference: run package

This package runs `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` on the dataset `Manourimer/aimo-controlled-robustness-v1`. It is a port of our working saturation-pilot vLLM code, with the same:
- checkpoint;
- tokenizer;
- chat template;
- `solve.txt` system prompt;
- decoding settings.

It has no Modal or cluster-specific dependencies. Each process uses one GPU and runs one vLLM engine. GPUs are allocated by your own scheduler.

## 1. Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

- **Tested environment:** Linux, CUDA, and **one GPU with ≥ 48 GB memory per process**. We used A100-80GB, and 80 GB is recommended.
- **Why 48 GB:** the weights take about 16 GB. vLLM also needs room for at least one full 131,072-token sequence (about 19.3 GB of KV cache), so 40 GB cards will not start.

## 2. Get the dataset (pinned revision)

```bash
hf download Manourimer/aimo-controlled-robustness-v1 --repo-type dataset \
  --revision ba7e4679b3e49cff82ddfd956338b3887d6169cd --local-dir data
```

- **Access:** the repository is private. Log in first (`hf auth login`, or set `HF_TOKEN`) with an account that has access, or ask us for the files.
- **Required files:** the script needs `problems.csv`, `perturbations.csv`, `pilot_selection.csv` and `inference_config.json` in `--data-dir`.
- **Integrity:** the script refuses to start if text hashes, counts, prompt or decoding settings differ from what is documented.

## 3. Run the pilot (1,408 generations)

```bash
python run_inference.py --mode pilot --data-dir data --out-dir results
```

The pilot covers the 18 saturation-pilot problems × 16 samples, plus 140 selected variants × 8 samples.

## 4. Run the full experiment (6,304 generations)

```bash
python run_inference.py --mode full --data-dir data --out-dir results
```

- **Contents:** 40 problems × 32 samples, plus all 628 valid variants × 8 samples. The 12 invalid variants are never run.
- **Pilot reuse:** the pilot is a strict subset (same job IDs, seeds and settings). With the same `--out-dir`, finished pilot generations are skipped.

## 5. Several GPUs in parallel

Start independent processes, one per GPU, each with its own shard index. Every shard runs a disjoint subset of jobs and writes to its own directory:

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python run_inference.py --mode full --data-dir data --out-dir results \
    --shard-index $i --num-shards 4 > shard$i.log 2>&1 &
done
wait
```

Use the same `--num-shards` for all processes of one launch. Do not run two launches with different shard counts at the same time.

## 6. Sampled-token log-probabilities

- **Default: on.** Every generated token's log-probability is saved, aligned with its token ID. This uses vLLM's standard `SamplingParams(logprobs=0)` with `logprobs_mode="raw_logprobs"`, i.e. the model's untruncated distribution, before top-k/top-p.
- **No effect on sampling:** collection does not change sampling, prompts or seeds; only the `logprobs` request field changes.
- **Preflight:** before any real work, a 4-token test checks that log-probabilities come back aligned.
  - If they don't, the run **stops** with an explanation.
  - Add `--allow-logprob-fallback` to continue without them instead. Each record then has `logprobs.status = "missing"`, `"misaligned"` or `"disabled"`.
- **To turn them off:** `--no-collect-logprobs`. Everything else stays identical.
- **Optional, off by default:** `--top-logprobs 5` also stores the top-5 log-probabilities per token. This adds about 8 GB to the full run.
- **Entropy:** per-token entropy is not available from standard vLLM without the full distribution, so it is not collected.

## 7. Where results go

`results/<mode>_shard<i>of<n>/`:
- `generations/<generation_id>.json`: one file per finished generation, written atomically. `generation_id` is `<problem_id or variant_id>__s<k>`.
- `run_config_<timestamp>.json`, one per launch, recording:
  - model and tokenizer revision, chat template, sampling parameters;
  - logprob settings and preflight result;
  - vLLM, torch and transformers versions, GPU;
  - data-file hashes.

To decode the arrays in a generation record:

```python
import base64, json, numpy as np
r = json.load(open("results/full_shard0of1/generations/aime2025_I-3__s0.json"))
ids = np.frombuffer(base64.b64decode(r["completion_token_ids"]), dtype="<i4")
lp  = np.frombuffer(base64.b64decode(r["logprobs"]["sampled"]), dtype="<f4") if r["logprobs"]["status"] == "ok" else None
```

## 8. Resume and check progress

- **Resume:** re-run the same command. Generations with a completed file anywhere under `--out-dir`, from any mode or shard, are skipped; nothing is ever overwritten.
- **What is lost:** sequences that were in flight when a job died are re-run, with the same seed.
- **Check progress (no GPU needed):**

  ```bash
  python run_inference.py --mode full --data-dir data --out-dir results --status
  ```

  This reports done and pending jobs, unreadable files, duplicate generation IDs (also saved under `duplicates/`, never counted twice), log-probability status, and truncations.
- **Optional quick smoke test:** `--max-jobs 4`. Its outputs are real results and count toward the run.
- **CPU-only checks:**

  ```bash
  python test_run_inference.py --data-dir data
  ```

## 9. What is saved per generation

- **Identifiers:** `problem_id`, `variant_id` (null for originals), `family`, `sample_index`, `seed`, `input_text_sha256`.
- **Prompt:** the exact rendered prompt, its token IDs and token count.
- **Completion:**
  - full text, including reasoning and final answer, with special tokens kept;
  - generated token IDs (int32, base64);
  - `n_generated_tokens`, `finish_reason`, `stop_reason`, `truncated`, `reasoning_tokens` (position of `</think>`).
- **Log-probabilities:** the sampled-token values (float32, base64, aligned with the token IDs) and their status.
- **Answer convenience fields:** `answer_method`, `answer_raw`, `answer_parsed`, `parse_status`, `correct_lenient` (`ANSWER:` or `\boxed{}`), `correct_strict` (`ANSWER:` only). We recompute these from the saved text anyway.

No activations are collected.

## 10. Expected storage

These figures are based on the saturation pilot's lengths: mean 25.1k generated tokens, median 22.9k, max 58k, and 2.65 bytes of text per token.

| | Pilot (35M tokens) | Full (158M tokens) |
|---|---:|---:|
| Completion text | 0.09 GB | 0.42 GB |
| + token IDs (int32, base64) | +0.19 GB | +0.84 GB |
| + sampled logprobs (float32, base64) | +0.19 GB | +0.84 GB |
| **Total with logprobs (default)** | **≈ 0.5 GB** | **≈ 2.1 GB** |
| Total with `--no-collect-logprobs` | ≈ 0.3 GB | ≈ 1.3 GB |

- **Scaling:** if the new problems run longer (e.g. a mean of 35k tokens), multiply by about 1.4.
- **Hard ceiling:** if every generation hit the 100k cap, the full run would take about 8.5 GB.
- **Format choice:** the same logprobs written as JSON decimal floats would take about 3.2 GB for the full run, so we use base64 instead. Full vocabulary distributions are never saved.

## 11. Known limitations

- **Untested on a GPU in this form.** The engine loop is the pilot's tested code, but this script itself has only passed CPU checks: job definitions, sharding, resume, serialization, and prompt rendering identical to the pilot.
- **Logprob overhead is estimated, not measured.**
  - GPU: one extra log-softmax, gather and rank pass over the vocabulary per decode step, against a 151,936-token vocabulary. That is small next to the forward pass, so we expect at most a few percent slowdown, and tens of MB of temporary GPU memory.
  - CPU: about 0.4 KB of host RAM per in-flight token, about 0.6 GB at 64 sequences of 25k tokens.
  - Disk: about +0.84 GB for the full run.
- **Not bit-reproducible:** seeds are fixed per generation, but vLLM's batched GPU kernels are not batch-invariant, so a re-run gives different (equally valid) samples.
- **Shard collisions:** concurrent launches with different shard counts could run the same job twice. The second copy goes to `duplicates/` and is reported, not counted.
