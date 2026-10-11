#!/usr/bin/env python3
"""Generation run for the AIMO controlled-robustness dataset (Manourimer/aimo-controlled-robustness-v1).

A portable adaptation of the saturation-pilot vLLM worker (same engine loop, tokenizer, chat template,
checkpoint and decoding); no Modal, no queue service. One process = one GPU = one vLLM replica.

  python run_inference.py --mode pilot --data-dir DATA --out-dir OUT              # 1,408 generations
  python run_inference.py --mode full  --data-dir DATA --out-dir OUT              # 6,304 (reuses pilot results)
  python run_inference.py --mode full  --data-dir DATA --out-dir OUT --shard-index 0 --num-shards 4
  python run_inference.py --mode full  --data-dir DATA --out-dir OUT --status     # progress, no GPU
  python run_inference.py --mode full  --data-dir DATA --out-dir OUT --dry-run    # checks only, no GPU

Outputs: OUT/<mode>_shard<i>of<n>/generations/<generation_id>.json (one file per finished generation,
written atomically) plus run_config.json per launch. A generation whose file exists anywhere under OUT is
never run again and never overwritten, so pilot results count toward the full run and a crashed run is
resumed by relaunching the same command.
"""
import argparse
import base64
import csv
import hashlib
import json
import os
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
REVISION = "6e8885a6ff5c1dc5201574c8fd700323f23c25fa"
DATASET_ID = "Manourimer/aimo-controlled-robustness-v1"
DATASET_REVISION = "ba7e4679b3e49cff82ddfd956338b3887d6169cd"
VLLM_VERSION = "0.11.0"
SAMPLING = dict(temperature=1.0, top_k=40, top_p=0.95, max_tokens=100_000)
MAX_MODEL_LEN = 131_072
LOGPROBS_MODE = "raw_logprobs"  # vLLM default: log-softmax of the model logits, before top-k/top-p truncation
SYSTEM_PROMPT_SHA256 = "72283c58f504d7c3d32d66066fcb06aecc20d14e93136cd6b1589f7f3ce7b002"  # baselines/prompts/solve.txt
# Rendered prompts of the saturation pilot (same problems, same tokenizer): sha256 of the prompt string,
# sha256 of json.dumps(prompt_token_ids), token count. The run refuses to start if rendering differs.
PILOT_PROMPT_REFS = {
    "aime2025_I-3": ("dd0a9b2333e920105bcdf2fdc554388b6b776e28eb3446934221c1b9c656dc6e",
                     "8c6b78e72c0594af6a806afc0da5cbce2c9a2255abe5e148de32a54f516db7d6", 170),
    "brumo2025_p22": ("53050d25a71dcbede9566c265266d0daa14b812a958cc003e88b1c276349c1f0",
                      "f8969c0e753c404b5387f33003d1515efa1b5c2cc5670ef05b7ba673db1b09be", 113),
    "hmmtfeb2025_round1-9": ("5f6c2945b39b919764067ac3550e7cd8db92f7e3adbb35fb97f0ce629a0ee25f",
                             "94183f6800a517fb7e26da1ca291cdb078f49e3590bfdffcc09c2378a988e3e0", 245),
}
MODES = {  # problems, originals' samples, variant source, variants' samples, expected total
    "pilot": dict(n_problems=18, k_original=16, k_variant=8, n_variants=140, total=1408),
    "full": dict(n_problems=40, k_original=32, k_variant=8, n_variants=628, total=6304),
}
SCHEMA = "crb40v1-generation-v1"


def sha256(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def seed_for(prompt_id, k):
    """inference_config.json 'seeds.rule'."""
    return int(sha256(f"crb40v1|{prompt_id}|{k}")[:8], 16)


# ---------------------------------------------------------------- data and jobs (CPU only)

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_data(data_dir):
    d = Path(data_dir)
    files = ["problems.csv", "perturbations.csv", "pilot_selection.csv", "inference_config.json"]
    missing = [f for f in files if not (d / f).exists()]
    if missing:
        sys.exit(f"--data-dir {d} is missing {missing}; download the dataset first (see README)")
    cfg = json.loads((d / "inference_config.json").read_text())
    problems = read_csv(d / "problems.csv")
    perts = read_csv(d / "perturbations.csv")
    pilot_sel = read_csv(d / "pilot_selection.csv")
    for p in problems:
        assert sha256(p["problem_text"]) == p["problem_text_sha256"], f"text hash mismatch {p['problem_id']}"
    for v in perts:
        assert sha256(v["perturbed_text"]) == v["perturbed_text_sha256"], f"text hash mismatch {v['variant_id']}"
    # configuration that must not drift from what was documented
    sp = cfg["prompt"]["system_prompt"]
    assert sha256(sp) == SYSTEM_PROMPT_SHA256, "system prompt differs from solve.txt"
    assert cfg["model"]["id"] == MODEL_ID and cfg["model"]["revision"] == REVISION, "checkpoint mismatch"
    dec = cfg["decoding"]
    assert (dec["temperature"], dec["top_k"], dec["top_p"], dec["max_new_tokens"]) == \
        (SAMPLING["temperature"], SAMPLING["top_k"], SAMPLING["top_p"], SAMPLING["max_tokens"]), "decoding mismatch"
    assert cfg["model"]["max_model_len"] == MAX_MODEL_LEN
    valid = [v for v in perts if v["validity_status"] == "valid"]
    n_invalid = len(perts) - len(valid)
    assert (len(problems), len(perts), len(valid), n_invalid) == (40, 640, 628, 12), "unexpected dataset counts"
    valid_ids = {v["variant_id"] for v in valid}
    assert all(s["variant_id"] in valid_ids for s in pilot_sel), "pilot selection contains a non-valid variant"
    return dict(cfg=cfg, problems=problems, valid=valid, pilot_sel=pilot_sel, system_prompt=sp,
                file_sha256={f: sha256((d / f).read_bytes()) for f in files})


def build_jobs(mode, data):
    """All jobs of a mode, independent of sharding and of logprob settings."""
    m = MODES[mode]
    probs = {p["problem_id"]: p for p in data["problems"]}
    if mode == "pilot":
        pids = sorted(p["problem_id"] for p in data["problems"] if p["in_saturation_pilot"] == "True")
        sel = {s["variant_id"] for s in data["pilot_sel"]}
        variants = [v for v in data["valid"] if v["variant_id"] in sel]
    else:
        pids = sorted(probs)
        variants = list(data["valid"])
    assert len(pids) == m["n_problems"] and len(variants) == m["n_variants"]
    jobs = []
    for pid in pids:
        p = probs[pid]
        for k in range(m["k_original"]):
            jobs.append(dict(prompt_id=pid, problem_id=pid, variant_id=None, family="original", sample_index=k,
                             text=p["problem_text"], reference_answer=int(p["reference_answer"])))
    for v in variants:
        for k in range(m["k_variant"]):
            jobs.append(dict(prompt_id=v["variant_id"], problem_id=v["problem_id"], variant_id=v["variant_id"],
                             family=v["family"], sample_index=k, text=v["perturbed_text"],
                             reference_answer=int(v["reference_answer"])))
    for j in jobs:
        j["generation_id"] = f"{j['prompt_id']}__s{j['sample_index']}"
        j["seed"] = seed_for(j["prompt_id"], j["sample_index"])
        j["input_text_sha256"] = sha256(j["text"])
    ids = [j["generation_id"] for j in jobs]
    assert len(ids) == len(set(ids)) == m["total"], (len(ids), m["total"])
    # sample-major order (as in the pilot): a partial run covers every prompt evenly
    return sorted(jobs, key=lambda j: (j["sample_index"], j["prompt_id"]))


def sampling_kwargs(seed, n_logprobs, max_tokens=SAMPLING["max_tokens"]):
    """vLLM SamplingParams arguments. Logprob collection changes only the 'logprobs' entry (None = off,
    0 = sampled token only, K = sampled token + top-K); it does not change how tokens are sampled."""
    return dict(SAMPLING, max_tokens=max_tokens, seed=seed, logprobs=n_logprobs,
                detokenize=False, skip_special_tokens=False)


def shard_of(generation_id, num_shards):
    return int(sha256(generation_id)[:8], 16) % num_shards


# ---------------------------------------------------------------- existing results (resume)

def scan_results(out_dir):
    """generation_id -> list of result files (any mode, any shard). Partial .tmp files are ignored."""
    found = {}
    for f in Path(out_dir).glob("*/generations/*.json"):
        try:
            rec = json.loads(f.read_text())
            ok = rec.get("status") == "complete" and rec.get("generation_id") == f.stem
        except (json.JSONDecodeError, OSError):
            ok = False
        found.setdefault(f.stem, []).append((str(f), ok))
    return found


def status_report(mode, data, out_dir):
    jobs = build_jobs(mode, data)
    found = scan_results(out_dir)
    want = {j["generation_id"] for j in jobs}
    done = {g for g in want if any(ok for _, ok in found.get(g, []))}
    corrupt = sorted(g for g in want if g in found and g not in done)
    dups = sorted(g for g, fs in found.items() if len(fs) > 1)
    lp, trunc = {}, 0
    for g in done:
        rec = json.loads(Path(next(p for p, ok in found[g] if ok)).read_text())
        lp[rec["logprobs"]["status"]] = lp.get(rec["logprobs"]["status"], 0) + 1
        trunc += bool(rec["truncated"])
    rep = dict(mode=mode, expected=len(want), done=len(done), pending=len(want) - len(done),
               unreadable_or_incomplete_files=corrupt, duplicate_generation_ids=dups,
               logprob_status=lp, truncated=trunc)
    print(json.dumps(rep, indent=2))
    return rep


# ---------------------------------------------------------------- answers (copied from the pilot's satpilot_core.py)

def parse_answer(text):
    s = (text or '').strip()
    s = re.sub(r'^\s*ANSWER\s*:\s*', '', s, flags=re.I).strip()
    s = s.strip('$').strip().rstrip('.').strip()
    m = re.fullmatch(r'\\boxed\{(.*)\}', s)
    if m: s = m.group(1).strip()
    m = re.fullmatch(r'([+-]?)\\d?frac\{(\d+)\}\{(\d+)\}', s)
    if m: s = f'{m.group(1)}{m.group(2)}/{m.group(3)}'
    if re.fullmatch(r'[+-]?\d{1,3}(,\d{3})+', s): s = s.replace(',', '')
    if re.fullmatch(r'[+-]?\d+(/[1-9]\d*)?', s): return Fraction(s), 'parsed', s
    if re.fullmatch(r'[+-]?\d+\.\d+', s): return Fraction(s), 'parsed_decimal', s
    return None, 'unparsed', s


def last_boxed(s):
    i = s.rfind("\\boxed{")
    while i != -1:
        j, depth = i + len("\\boxed{"), 1
        while j < len(s) and depth:
            depth += {"{": 1, "}": -1}.get(s[j], 0)
            j += 1
        if depth == 0:
            return s[i + len("\\boxed{"): j - 1]
        i = s.rfind("\\boxed{", 0, i)
    return None


def extract_answer(completion, reference_answer, truncated):
    """Convenience fields only; the full completion is saved so these can be recomputed."""
    think_closed = "</think>" in completion
    post = completion.rsplit("</think>", 1)[1] if think_closed else ""
    post = re.sub(r"<｜[^｜]*｜>", "", post)
    lines = re.findall(r"ANSWER\s*:([^\n]*)", post)
    method, raw, val, status = "none", "", None, "no_final_answer"
    if lines:
        method, raw = "answer_line", lines[-1].replace("*", "").strip()
        val, status, _ = parse_answer(raw)
        if val is None and (b := last_boxed(post)) is not None:
            v2, s2, _ = parse_answer(b)
            if v2 is not None:
                method, raw, val, status = "answer_line_boxed", b, v2, s2
    elif (b := last_boxed(post)) is not None:
        method, raw = "boxed_fallback", b
        val, status, _ = parse_answer(b)
    match = val is not None and not truncated and val == Fraction(int(reference_answer))
    return dict(think_closed=think_closed, answer_method=method, answer_raw=raw,
                answer_parsed=None if val is None else str(val), parse_status=status,
                correct_lenient=bool(match), correct_strict=bool(match and method.startswith("answer_line")))


# ---------------------------------------------------------------- serialization

def b64(values, dtype):
    import numpy as np
    return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode("ascii")


def logprob_fields(o, ids, collect, top_k):
    """Sampled-token logprobs aligned with ids; never raises (the generation is kept either way)."""
    if not collect:
        return dict(status="disabled")
    lp = o.logprobs
    if lp is None:
        return dict(status="missing", note="vLLM returned no logprobs for this request")
    if len(lp) != len(ids) or any(t not in d for t, d in zip(ids, lp)):
        return dict(status="misaligned", note=f"{len(lp)} logprob positions for {len(ids)} tokens")
    rec = dict(status="ok", mode=LOGPROBS_MODE, dtype="<f4", encoding="base64",
               sampled=b64([d[t].logprob for t, d in zip(ids, lp)], "<f4"),
               cumulative=o.cumulative_logprob)
    if top_k > 0:  # entries ranked 1..top_k at each position (the sampled token may or may not be among them)
        tid, tlp = [], []
        for d in lp:
            top = sorted(((v.rank, t, v.logprob) for t, v in d.items() if v.rank is not None and v.rank <= top_k))[:top_k]
            top += [(None, -1, float("nan"))] * (top_k - len(top))
            tid += [t for _, t, _ in top]; tlp += [x for _, _, x in top]
        rec.update(top_k=top_k, top_ids=b64(tid, "<i4"), top_logprobs=b64(tlp, "<f4"), top_shape=[len(ids), top_k])
    return rec


def write_atomic(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False))
    os.replace(tmp, path)


# ---------------------------------------------------------------- generation (GPU)

def run(args, data, jobs):
    import torch
    import transformers
    import vllm
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    if vllm.__version__ != VLLM_VERSION:
        print(f"WARNING: vllm {vllm.__version__} != tested {VLLM_VERSION}", flush=True)
    out_root = Path(args.out_dir)
    run_dir = out_root / f"{args.mode}_shard{args.shard_index}of{args.num_shards}"
    gen_dir = run_dir / "generations"

    snap = args.model_path or __import__("huggingface_hub").snapshot_download(
        MODEL_ID, revision=REVISION, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    tok = transformers.AutoTokenizer.from_pretrained(snap)
    sp = data["system_prompt"]

    def render(text):
        s = tok.apply_chat_template([{"role": "system", "content": sp}, {"role": "user", "content": text}],
                                    tokenize=False, add_generation_prompt=True)
        return s, tok(s)["input_ids"]

    probs = {p["problem_id"]: p["problem_text"] for p in data["problems"]}
    for pid, (s_sha, ids_sha, n) in PILOT_PROMPT_REFS.items():  # identical to the saturation pilot
        s, ids = render(probs[pid])
        if (sha256(s), sha256(json.dumps(ids)), len(ids)) != (s_sha, ids_sha, n):
            sys.exit(f"prompt rendering differs from the saturation pilot for {pid}: wrong tokenizer/transformers "
                     f"version or model revision (transformers {transformers.__version__}); refusing to run")

    engine = LLMEngine.from_engine_args(EngineArgs(
        model=snap, tokenizer=snap, dtype="bfloat16", max_model_len=MAX_MODEL_LEN, seed=0,
        gpu_memory_utilization=args.gpu_memory_utilization, generation_config="vllm", enable_prefix_caching=True,
        logprobs_mode=LOGPROBS_MODE, max_logprobs=max(20, args.top_logprobs)))
    n_logprobs = args.top_logprobs if args.collect_logprobs else None

    def params(seed, max_tokens=SAMPLING["max_tokens"]):
        return SamplingParams(**sampling_kwargs(seed, n_logprobs, max_tokens), output_kind=RequestOutputKind.FINAL_ONLY)

    # logprob preflight: a 4-token request (not saved as a result) before any expensive work
    collect, preflight = args.collect_logprobs, "disabled"
    if collect:
        s, ids = render("Compute $1+1$.")
        engine.add_request("preflight", TokensPrompt(prompt_token_ids=ids), params(1, max_tokens=4))
        o = None
        while engine.has_unfinished_requests():
            for out in engine.step():
                if out.finished:
                    o = out.outputs[0]
        pf = logprob_fields(o, list(o.token_ids), True, args.top_logprobs)
        preflight = pf["status"]
        if preflight != "ok":
            msg = f"sampled-token logprobs unavailable in preflight ({pf})."
            if not args.allow_logprob_fallback:
                sys.exit(msg + " Rerun with --no-collect-logprobs, or --allow-logprob-fallback to continue without them.")
            print("WARNING: " + msg + " Continuing WITHOUT logprobs (recorded per generation).", flush=True)
            collect, n_logprobs = False, None

    config = dict(
        schema=SCHEMA, mode=args.mode, shard_index=args.shard_index, num_shards=args.num_shards,
        dataset_id=DATASET_ID, dataset_revision_expected=DATASET_REVISION, data_file_sha256=data["file_sha256"],
        model_id=MODEL_ID, model_revision=REVISION, tokenizer_revision=REVISION, model_path_given=bool(args.model_path),
        dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
        rope_scaling=json.loads(Path(snap, "config.json").read_text()).get("rope_scaling"),
        sampling=SAMPLING, system_prompt=sp, chat_template=tok.chat_template,
        chat_template_sha256=sha256(tok.chat_template or ""),
        collect_logprobs=collect, logprobs_mode=LOGPROBS_MODE if collect else None,
        top_logprobs=args.top_logprobs if collect else 0, logprob_preflight=preflight,
        allow_logprob_fallback=args.allow_logprob_fallback,
        engine=f"vllm {vllm.__version__}", torch=torch.__version__, transformers=transformers.__version__,
        gpu=torch.cuda.get_device_name(0), gpu_count=torch.cuda.device_count(),
        gpu_memory_utilization=args.gpu_memory_utilization, max_inflight=args.max_inflight,
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write_atomic(run_dir / f"run_config_{time.strftime('%Y%m%dT%H%M%S')}.json", config)

    found = scan_results(out_root)
    pending = [j for j in jobs if not any(ok for _, ok in found.get(j["generation_id"], []))]
    print(f"[{run_dir.name}] {len(jobs)} jobs in shard, {len(jobs) - len(pending)} already done, "
          f"{len(pending)} to run; logprobs={'on' if collect else 'off'}", flush=True)
    think_close = tok.convert_tokens_to_ids("</think>")
    queue, inflight, n_done = list(pending), {}, 0

    def refill():
        while queue and len(inflight) < args.max_inflight:
            j = queue.pop(0)
            s, ids = render(j["text"])
            assert len(ids) + SAMPLING["max_tokens"] <= MAX_MODEL_LEN
            engine.add_request(j["generation_id"], TokensPrompt(prompt_token_ids=ids), params(j["seed"]))
            inflight[j["generation_id"]] = dict(job=j, prompt=s, prompt_ids=ids, t_submit=time.time())

    refill()
    while inflight:
        for out in engine.step():
            if not out.finished:
                continue
            info = inflight.pop(out.request_id)
            j, o = info["job"], out.outputs[0]
            ids = list(o.token_ids)
            completion = tok.decode(ids, skip_special_tokens=False)
            truncated = o.finish_reason == "length"
            rec = dict(
                schema=SCHEMA, status="complete", generation_id=j["generation_id"], run_mode=args.mode,
                shard=f"{args.shard_index}of{args.num_shards}",
                problem_id=j["problem_id"], variant_id=j["variant_id"], family=j["family"],
                sample_index=j["sample_index"], seed=j["seed"], input_text_sha256=j["input_text_sha256"],
                reference_answer=j["reference_answer"],
                model_id=MODEL_ID, model_revision=REVISION, engine=f"vllm {vllm.__version__}",
                rendered_prompt=info["prompt"], rendered_prompt_sha256=sha256(info["prompt"]),
                prompt_token_ids=info["prompt_ids"], prompt_token_count=len(info["prompt_ids"]),
                completion_text=completion, n_generated_tokens=len(ids),
                completion_token_ids=b64(ids, "<i4"), completion_token_ids_dtype="<i4", completion_token_ids_encoding="base64",
                finish_reason=o.finish_reason, stop_reason=o.stop_reason, truncated=truncated,
                reasoning_tokens=ids.index(think_close) if think_close in ids else None,
                logprobs=logprob_fields(o, ids, collect, args.top_logprobs),
                **extract_answer(completion, j["reference_answer"], truncated),
                t_submit=info["t_submit"], t_done=time.time(), seconds=time.time() - info["t_submit"],
                timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            dest = gen_dir / f"{j['generation_id']}.json"
            if any(ok for _, ok in scan_one(out_root, j["generation_id"])):  # another launch finished it meanwhile
                dest = run_dir / "duplicates" / f"{j['generation_id']}.{int(time.time())}.json"
                print(f"DUPLICATE {j['generation_id']} -> {dest} (not counted)", flush=True)
            write_atomic(dest, rec)
            n_done += 1
            print(json.dumps(dict(n=n_done, id=j["generation_id"], tokens=len(ids), finish=o.finish_reason,
                                  logprobs=rec["logprobs"]["status"], s=round(rec["seconds"]))), flush=True)
            refill()
    print(f"[{run_dir.name}] finished: {n_done} generations written this launch", flush=True)


def scan_one(out_root, generation_id):
    res = []
    for f in Path(out_root).glob(f"*/generations/{generation_id}.json"):
        try:
            rec = json.loads(f.read_text())
            res.append((str(f), rec.get("status") == "complete"))
        except (json.JSONDecodeError, OSError):
            res.append((str(f), False))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=list(MODES))
    ap.add_argument("--data-dir", required=True, help="directory with the dataset files (pinned revision)")
    ap.add_argument("--out-dir", required=True, help="results root; reuse the same one for pilot and full")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    lp = ap.add_mutually_exclusive_group()
    lp.add_argument("--collect-logprobs", dest="collect_logprobs", action="store_true", default=True,
                    help="save the log-probability of every sampled token (default)")
    lp.add_argument("--no-collect-logprobs", dest="collect_logprobs", action="store_false")
    ap.add_argument("--allow-logprob-fallback", action="store_true",
                    help="if logprobs are unavailable, continue without them (recorded) instead of stopping")
    ap.add_argument("--top-logprobs", type=int, default=0, help="optional: also save top-K logprobs per token (default 0 = off)")
    ap.add_argument("--max-inflight", type=int, default=64, help="concurrent sequences submitted to vLLM")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--model-path", default=None, help="local snapshot of the pinned revision (else downloaded)")
    ap.add_argument("--max-jobs", type=int, default=None, help="run only the first N pending jobs of this shard (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="validate data and job definitions, no GPU")
    ap.add_argument("--status", action="store_true", help="report progress under --out-dir, no GPU")
    a = ap.parse_args(argv)
    if not 0 <= a.shard_index < a.num_shards:
        sys.exit("--shard-index must be in [0, --num-shards)")
    if not 0 <= a.top_logprobs <= 20:
        sys.exit("--top-logprobs must be in [0, 20]")
    data = load_data(a.data_dir)
    if a.status:
        return status_report(a.mode, data, a.out_dir)
    jobs = [j for j in build_jobs(a.mode, data) if shard_of(j["generation_id"], a.num_shards) == a.shard_index]
    if a.dry_run:
        print(json.dumps(dict(mode=a.mode, shard=f"{a.shard_index}of{a.num_shards}", jobs_in_shard=len(jobs),
                              total_jobs=MODES[a.mode]["total"], data_file_sha256=data["file_sha256"]), indent=2))
        return jobs
    if a.max_jobs is not None:
        done = scan_results(a.out_dir)
        jobs = [j for j in jobs if not any(ok for _, ok in done.get(j["generation_id"], []))][:a.max_jobs]
    run(a, data, jobs)


if __name__ == "__main__":
    main()
