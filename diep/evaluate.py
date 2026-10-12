"""Evaluation: the GA/DPMA protocol, so the three methods are scored identically.

The paper reports zero-shot ``acc`` on MMLU/BoolQ/OBQA/RTE (spec §6). This
study instead reuses the protocol every GA and DPMA number was produced with,
copied from ``GA/MoE_compression_GA/examples/run_harness.py``:

* lm-evaluation-harness, **1-shot**, ``bootstrap_iters=0``, batch size ``auto``;
* tasks ``all`` = 8 multiple-choice (arc_challenge, arc_easy, boolq, hellaswag,
  mmlu, openbookqa, piqa, winogrande) + 2 generative (gsm8k, mbpp);
* per task the first available of ``acc_norm, acc, exact_match...``;
* headline ``mean_mc`` = mean over the 8 multiple-choice tasks;
* the same JSON schema, so one loader reads GA, DPMA and DiEP results.

``tests/test_protocol.py`` parses the GA file and checks these constants are
identical, so the protocols cannot silently drift apart.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

MC_TASKS = [
    "arc_challenge", "arc_easy", "boolq", "hellaswag", "mmlu",
    "openbookqa", "piqa", "winogrande",
]
GEN_TASKS = ["gsm8k", "mbpp"]
TASK_GROUPS = {"mc": MC_TASKS, "gen": GEN_TASKS, "all": MC_TASKS + GEN_TASKS}
CODE_EXECUTION_TASKS = {"mbpp", "humaneval"}
PREFERRED_METRICS = ("acc_norm,none", "acc,none", "exact_match,strict-match",
                     "exact_match,flexible-extract", "exact_match,none",
                     "pass_at_1,none", "pass@1,none")

__all__ = ["MC_TASKS", "GEN_TASKS", "TASK_GROUPS", "resolve_tasks", "pick_metrics",
           "result_stem", "run_harness"]


def resolve_tasks(requested: list[str]) -> list[str]:
    tasks: list[str] = []
    for name in requested:
        tasks.extend(TASK_GROUPS.get(name, [name]))
    return list(dict.fromkeys(tasks))


def pick_metrics(scores: dict) -> dict:
    """The headline metric per task, plus its stderr (the GA's rule)."""
    chosen = {}
    for key in PREFERRED_METRICS:
        if key in scores:
            chosen["metric"] = key
            chosen["value"] = scores[key]
            stderr = scores.get(key.replace(",", "_stderr,", 1))
            chosen["stderr"] = stderr if isinstance(stderr, (int, float)) else None
            break
    return chosen


def result_stem(method: str, keep_ratio: float, replicate: int, num_fewshot: int) -> str:
    return f"{method}-keep{keep_ratio:.3f}-rep{replicate}-{num_fewshot}shot"


def run_harness(model, tokenizer, *, tasks, num_fewshot=1, eval_batch_size="auto", limit=None,
                allow_unsafe_code=False) -> tuple[dict, float, list[str]]:
    """Score ``model`` in memory. Returns the harness output, minutes, and tasks."""
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    tasks = resolve_tasks(tasks)
    unsafe = sorted(CODE_EXECUTION_TASKS.intersection(tasks))
    if unsafe and not allow_unsafe_code:
        raise SystemExit(f"{', '.join(unsafe)} execute model-generated code; "
                         "re-run with --allow-unsafe-code if that is acceptable")
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=eval_batch_size)
    started = time.time()
    evaluation = lm_eval.simple_evaluate(
        model=lm, tasks=tasks, num_fewshot=num_fewshot, limit=limit,
        bootstrap_iters=0, verbosity="ERROR", confirm_run_unsafe_code=bool(unsafe))
    return evaluation, (time.time() - started) / 60, tasks


def build_record(evaluation, tasks, *, base: dict) -> dict:
    """The GA's result record: per-task headline + all metrics, and the means."""
    record = dict(base)
    record["tasks"] = {
        task: {"headline": pick_metrics(scores),
               "all": {k: v for k, v in scores.items() if isinstance(v, (int, float))}}
        for task, scores in evaluation["results"].items()
    }
    headlines = {}
    for task in tasks:
        headline = record["tasks"].get(task, {}).get("headline")
        if headline:
            headlines[task] = headline["value"]
    record["headline"] = headlines
    mc = [v for t, v in headlines.items() if t in MC_TASKS]
    gen = [v for t, v in headlines.items() if t in GEN_TASKS]
    record["mean_mc"] = round(float(np.mean(mc)), 6) if mc else None
    record["mean_gen"] = round(float(np.mean(gen)), 6) if gen else None
    record["mean_headline"] = (round(float(np.mean(list(headlines.values()))), 6)
                               if headlines else None)
    return record


def print_record(record: dict, tasks: list[str]) -> None:
    print(f"{'task':<18}{'metric':<28}{'value':>9}")
    for task in tasks:
        headline = record["tasks"].get(task, {}).get("headline")
        if headline:
            print(f"{task:<18}{headline['metric']:<28}{headline['value']:>9.4f}")
    for label, key in (("mean (MC)", "mean_mc"), ("mean (generative)", "mean_gen"),
                       ("mean (all)", "mean_headline")):
        if record.get(key) is not None:
            print(f"{label:<46}{record[key]:>9.4f}")


def write_record(record: dict, output_dir, stem: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.json"
    path.write_text(json.dumps(record, indent=2, default=str))
    return path
