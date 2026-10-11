"""CPU-only checks for run_inference.py (no GPU, no vLLM needed).

  python test_run_inference.py --data-dir DATA [--tokenizer-path SNAPSHOT]

--tokenizer-path (a local snapshot of the pinned model revision, transformers==4.57.1 installed) additionally
checks that prompts render exactly as in the saturation pilot.
"""
import argparse
import base64
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import run_inference as ri


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("ok  ", msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--tokenizer-path", default=None)
    a = ap.parse_args()
    data = ri.load_data(a.data_dir)
    pilot, full = ri.build_jobs("pilot", data), ri.build_jobs("full", data)
    check(len(pilot) == 1408 and len(full) == 6304, "job counts: pilot 1,408, full 6,304")
    fk = {j["generation_id"]: j for j in full}
    check(all(j["generation_id"] in fk and fk[j["generation_id"]]["seed"] == j["seed"]
              and fk[j["generation_id"]]["input_text_sha256"] == j["input_text_sha256"] for j in pilot),
          "pilot jobs (ids, seeds, inputs) are a strict subset of full")
    check(len({j["seed"] for j in full}) == len(full), "seeds unique across the full run")
    valid = {v["variant_id"] for v in data["valid"]}
    var_jobs = [j for j in full if j["variant_id"]]
    check({j["variant_id"] for j in var_jobs} == valid and all(j["variant_id"] in valid for j in pilot if j["variant_id"]),
          "only the 628 valid variants are used; none of the 12 invalid")
    check(sum(j["family"] == "original" for j in pilot) == 18 * 16 and sum(j["family"] == "original" for j in full) == 40 * 32,
          "originals: 18x16 (pilot), 40x32 (full)")
    check(all(j["seed"] == ri.seed_for(j["prompt_id"], j["sample_index"]) for j in full), "seed rule as in inference_config.json")
    for n in (1, 2, 4, 7):
        parts = [{j["generation_id"] for j in full if ri.shard_of(j["generation_id"], n) == i} for i in range(n)]
        check(sum(map(len, parts)) == len(full) and set().union(*parts) == set(fk), f"{n} shards: disjoint and complete")
    on, off = ri.sampling_kwargs(123, 0), ri.sampling_kwargs(123, None)
    check({k for k in on if on[k] != off[k]} == {"logprobs"}, "logprob flag changes only SamplingParams.logprobs")
    j1 = ri.main(["--mode", "full", "--data-dir", a.data_dir, "--out-dir", "/nonexistent", "--dry-run", "--collect-logprobs"])
    j2 = ri.main(["--mode", "full", "--data-dir", a.data_dir, "--out-dir", "/nonexistent", "--dry-run", "--no-collect-logprobs"])
    check(j1 == j2, "job definitions identical with --collect-logprobs and --no-collect-logprobs")

    # serialization of logprobs
    import numpy as np
    ids = [5, 9, 2]
    lp = [{5: SimpleNamespace(logprob=-0.1, rank=1), 7: SimpleNamespace(logprob=-2.0, rank=2)},
          {9: SimpleNamespace(logprob=-1.5, rank=3), 1: SimpleNamespace(logprob=-0.2, rank=1), 4: SimpleNamespace(logprob=-1.0, rank=2)},
          {2: SimpleNamespace(logprob=-0.01, rank=1)}]
    o = SimpleNamespace(logprobs=lp, cumulative_logprob=-1.61)
    r = ri.logprob_fields(o, ids, True, 0)
    dec = np.frombuffer(base64.b64decode(r["sampled"]), dtype="<f4")
    check(r["status"] == "ok" and np.allclose(dec, [-0.1, -1.5, -0.01]), "sampled logprobs aligned with token ids (round trip)")
    r2 = ri.logprob_fields(o, ids, True, 2)
    tid = np.frombuffer(base64.b64decode(r2["top_ids"]), dtype="<i4").reshape(3, 2)
    check(tid.tolist() == [[5, 7], [1, 4], [2, -1]], "top-k ids by rank, padded with -1")
    check(ri.logprob_fields(o, [5, 8, 2], True, 0)["status"] == "misaligned", "misaligned logprobs detected, not saved")
    check(ri.logprob_fields(SimpleNamespace(logprobs=None), ids, True, 0)["status"] == "missing", "missing logprobs recorded")
    check(ri.logprob_fields(o, ids, False, 0) == {"status": "disabled"}, "disabled logprobs recorded")
    check(np.frombuffer(base64.b64decode(ri.b64([151643, 0, 151935], "<i4")), dtype="<i4").tolist() == [151643, 0, 151935],
          "token-id round trip")
    ans = ri.extract_answer("x</think>\nso \\boxed{12}<｜end▁of▁sentence｜>", 12, False)
    check(ans["correct_lenient"] and not ans["correct_strict"], "boxed-only answer: lenient yes, strict no")
    ans = ri.extract_answer("x</think>\nANSWER: 12<｜end▁of▁sentence｜>", 12, False)
    check(ans["correct_lenient"] and ans["correct_strict"], "ANSWER: line: lenient and strict")
    check(not ri.extract_answer("x</think>\nANSWER: 12", 12, True)["correct_lenient"], "truncated never correct")

    # resume / duplicates with mock outputs
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        for j in pilot[:1000]:
            ri.write_atomic(out / "pilot_shard0of2" / "generations" / f"{j['generation_id']}.json",
                            dict(status="complete", generation_id=j["generation_id"], logprobs={"status": "ok"}, truncated=False))
        (out / "pilot_shard0of2" / "generations" / f"{pilot[1000]['generation_id']}.tmp").write_text("{partial")
        (out / "pilot_shard0of2" / "generations" / f"{pilot[1001]['generation_id']}.json").write_text("{corrupt")
        rp = ri.status_report("pilot", data, td)
        check(rp["done"] == 1000 and rp["pending"] == 408 and rp["unreadable_or_incomplete_files"] == [pilot[1001]["generation_id"]],
              "resume: completed counted, .tmp ignored, corrupt file reported and re-run")
        rf = ri.status_report("full", data, td)
        check(rf["done"] == 1000 and rf["pending"] == 5304, "full run reuses completed pilot generations")
        ri.write_atomic(out / "full_shard1of4" / "generations" / f"{pilot[0]['generation_id']}.json",
                        dict(status="complete", generation_id=pilot[0]["generation_id"], logprobs={"status": "ok"}, truncated=False))
        check(ri.status_report("full", data, td)["duplicate_generation_ids"] == [pilot[0]["generation_id"]], "duplicates reported")

    if a.tokenizer_path:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer_path)
        probs = {p["problem_id"]: p["problem_text"] for p in data["problems"]}
        for pid, (s_sha, ids_sha, n) in ri.PILOT_PROMPT_REFS.items():
            s = tok.apply_chat_template([{"role": "system", "content": data["system_prompt"]}, {"role": "user", "content": probs[pid]}],
                                        tokenize=False, add_generation_prompt=True)
            ids = tok(s)["input_ids"]
            check((ri.sha256(s), ri.sha256(json.dumps(ids)), len(ids)) == (s_sha, ids_sha, n), f"prompt rendering == saturation pilot ({pid})")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
