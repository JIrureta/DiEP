"""The ``diep`` command line (spec §11)::

    python -m diep search     --model M --calibration-file F --out RUN        # RUN/scores.pt
    python -m diep prune      --model M --scores RUN/scores.pt --sparsity R --out RUN/pruned [--merge]
    python -m diep calib-skip --model RUN/pruned --calibration-file F --out RUN/skip_config.json
    python -m diep eval       --model RUN/pruned [--skip RUN/skip_config.json] --method diep --output DIR
    python -m diep bench      --model RUN/pruned [--skip ...] [--baseline M]

The search does not depend on the sparsity -- ``r`` enters only at pruning -- so
one search serves every ratio. ``--sparsity`` on ``search`` is optional and only
adds that ratio's mask and keep lists to ``scores.pt``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

DEFAULT_MODEL = "allenai/OLMoE-1B-7B-0924"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def load_base_model(name, device, dtype=torch.bfloat16):
    import transformers
    from transformers import AutoModelForCausalLM

    key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(name, **{key: dtype}).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def calibration_for(args, model_name):
    from .data import load_calibration
    return load_calibration(args.calibration_file, expect_tokenizer=model_name,
                            allow_other_file=getattr(args, "allow_other_calibration", False))


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1, default=str))


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------
def cmd_search(args):
    from .adapters import get_adapter
    from .provenance import card
    from .prune import compute_scores, keep_lists, select_global
    from .search import SearchConfig, search

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = SearchConfig(search_forward=args.search_forward, epochs=args.epochs,
                       batch_size=args.batch_size, micro_batch=args.micro_batch,
                       lr_alpha=args.lr_alpha, lr_beta=args.lr_beta,
                       update_ratio=args.update_ratio, lam=args.lam,
                       recon_target=args.recon_target, score_mode=args.score_mode,
                       beta_init=args.beta_init, beta_min=args.beta_min, seed=args.seed,
                       ablation=args.ablation, fixed_beta=args.fixed_beta)
    print(f"[1/3] calibration: {args.calibration_file}")
    calibration = calibration_for(args, args.model)
    print("      " + calibration.describe())
    print(f"[2/3] loading {args.model} ...")
    model = load_base_model(args.model, args.device)
    adapter = get_adapter(model)
    print(f"[3/3] search (seed {cfg.seed}) ...")
    result = search(model, adapter, calibration, cfg, device=args.device)
    blocks = adapter.moe_blocks(model)
    scores = compute_scores(result["alpha"], result["beta"], score_mode=cfg.score_mode,
                            ablation=cfg.ablation, frequency=result["frequency"])
    payload = {"alpha": result["alpha"], "beta": result["beta"], "scores": scores,
               "frequency": result["frequency"], "config": result["config"],
               "steps": result["steps"], "logs": result["logs"], "model": args.model,
               "calibration": calibration.path, "top_k": adapter.top_k(blocks[0]),
               "num_experts": adapter.num_experts(blocks[0])}
    if args.sparsity is not None:
        mask = select_global(scores, args.sparsity, args.m_min or adapter.top_k(blocks[0]))
        payload.update(mask=mask, keep=keep_lists(mask), sparsity=args.sparsity)
    torch.save(payload, out / "scores.pt")
    write_json(out / "search_log.json", result["logs"])
    write_json(out / "run_card.json", card(stage="search", config=result["config"],
                                           steps=result["steps"], model=args.model,
                                           calibration=calibration.path,
                                           calibration_metadata=calibration.metadata))
    print(f"\nwrote {out / 'scores.pt'}  (beta {result['beta'].min():.4f}..{result['beta'].max():.4f})")


def cmd_prune(args):
    from transformers import AutoTokenizer

    from .adapters import get_adapter
    from .bench import param_bytes
    from .merge import merge_pruned_experts
    from .provenance import card
    from .prune import keep_lists, prune_model, save_pruned, select_global, select_uniform

    scores_file = torch.load(args.scores, map_location="cpu", weights_only=False)
    model_name = args.model or scores_file["model"]
    model = load_base_model(model_name, args.device)
    adapter = get_adapter(model)
    blocks = adapter.moe_blocks(model)
    m_min = args.m_min or adapter.top_k(blocks[0])
    scores = scores_file["scores"]
    mask = (select_global(scores, args.sparsity, m_min) if args.selection == "global"
            else select_uniform(scores, args.sparsity))
    keep = keep_lists(mask)
    per_layer = [len(k) for k in keep]
    print(f"      {args.selection} selection at r={args.sparsity}: kept "
          f"{sum(per_layer)}/{mask.numel()}, per layer {min(per_layer)}..{max(per_layer)}")
    parameters_before, bytes_before = sum(p.numel() for p in model.parameters()), param_bytes(model)

    merge_report = None
    if args.merge:
        calibration = calibration_for(args, model_name)
        print("      merging pruned experts into retained ones (spec §8) ...")
        merge_report = merge_pruned_experts(model, adapter, calibration, mask, args.device,
                                            n_tokens=args.cka_tokens, seed=args.cka_seed)
    prune_model(model, adapter, keep)
    parameters_after = sum(p.numel() for p in model.parameters())
    meta = {"sparsity": args.sparsity, "selection": args.selection, "m_min": m_min,
            "merge": bool(args.merge), "scores": str(Path(args.scores).resolve()),
            "seed": scores_file["config"].get("seed"), "base_model": model_name,
            "parameters_before": parameters_before, "parameters_after": parameters_after,
            "param_bytes_before": bytes_before, "param_bytes_after": param_bytes(model),
            "search_config": scores_file["config"]}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    save_pruned(model, adapter, keep, args.out, tokenizer=tokenizer, meta=meta)
    write_json(Path(args.out) / "prune_card.json", card(stage="prune", **meta,
                                                         per_layer=per_layer,
                                                         merge_report=merge_report))
    print(f"      wrote {args.out}  ({parameters_after / parameters_before:.1%} of the parameters)")


def cmd_calib_skip(args):
    from .adapters import get_adapter
    from .prune import load_pruned
    from .skip import calibrate_skip, save_skip_config

    model, raw = load_pruned(args.model, device=args.device)
    adapter = get_adapter(model)
    calibration = calibration_for(args, raw["diep"]["base_model"])
    config = calibrate_skip(model, adapter, calibration, args.device,
                            n_tokens=args.cka_tokens, seed=args.cka_seed)
    save_skip_config(config, args.out)
    g1 = [round(v["gamma1"][0], 3) for v in config.values()]
    print(f"      wrote {args.out}; gamma1 at rank 1 per layer: {g1}")


def _skip_state(model, adapter, skip_path, device):
    from .moe_forward import DiEPState, install
    from .skip import build_gamma, load_skip_config
    state = DiEPState(mode="skip", gamma=build_gamma(load_skip_config(skip_path), device))
    return install(model, adapter, state), state


def cmd_eval(args):
    from transformers import AutoTokenizer

    from .adapters import get_adapter
    from .evaluate import build_record, print_record, result_stem, run_harness, write_record
    from .provenance import card
    from .prune import load_pruned

    model, raw = load_pruned(args.model, device=args.device)
    meta = raw["diep"]
    adapter = get_adapter(model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    kept = sum(len(k) for k in meta["keep"])
    total = len(meta["keep"]) * raw["num_experts"]
    replicate = args.replicate if args.replicate is not None else int(meta.get("seed") or 0)
    stem = result_stem(args.method, kept / total, replicate, args.num_fewshot)
    target = Path(args.output) / f"{stem}.json"
    if target.is_file() and not args.force:
        print(f"{target} exists; pass --force to redo it")
        return
    state = None
    if args.skip:
        _, state = _skip_state(model, adapter, args.skip, args.device)
        print(f"      adaptive skipping on ({args.skip})")
    evaluation, minutes, tasks = run_harness(
        model, tokenizer, tasks=args.tasks, num_fewshot=args.num_fewshot,
        eval_batch_size=args.eval_batch_size, limit=args.limit,
        allow_unsafe_code=args.allow_unsafe_code)
    base = {"method": args.method, "replicate": replicate, "keep_ratio": round(kept / total, 6),
            "requested_keep_ratio": round(1 - meta["sparsity"], 6),
            "num_fewshot": args.num_fewshot, "model": meta["base_model"],
            "individual": str(Path(args.model).resolve()), "experts_kept": kept,
            "experts_total": total, "parameters_after": meta["parameters_after"],
            "parameter_ratio": round(meta["parameters_after"] / meta["parameters_before"], 6),
            "layer_budgets": [len(k) for k in meta["keep"]], "elapsed_minutes": round(minutes, 2),
            "diep": {k: meta[k] for k in ("sparsity", "selection", "m_min", "merge", "seed")}
            | {"skip": str(args.skip) if args.skip else None,
               "skip_fraction": (state.skip_stats["skipped"] / max(state.skip_stats["pairs"], 1)
                                 if state else None)}}
    record = build_record(evaluation, tasks, base=base)
    record["provenance"] = card(stage="eval", tasks=tasks, num_fewshot=args.num_fewshot,
                                eval_batch_size=args.eval_batch_size, limit=args.limit)
    print_record(record, tasks)
    print(f"written to {write_record(record, args.output, stem)}")


def cmd_bench(args):
    from .adapters import get_adapter
    from .bench import param_bytes, throughput
    from .data import load_calibration
    from .moe_forward import DiEPState, install
    from .prune import load_pruned

    model, raw = load_pruned(args.model, device=args.device)
    adapter = get_adapter(model)
    calibration = load_calibration(args.calibration_file,
                                   allow_other_file=args.allow_other_calibration)
    prompts = [calibration.input_ids[i:i + 1, :args.prompt_tokens].to(args.device)
               for i in range(args.prompts)]
    results = {"pruned_native": throughput(model, prompts, new_tokens=args.new_tokens)}
    state = DiEPState(mode="skip", gamma=None)          # the DiEP forward, nothing skipped
    dmodel = install(model, adapter, state)
    results["pruned_diep_forward"] = throughput(model, prompts, new_tokens=args.new_tokens)
    if args.skip:
        from .skip import build_gamma, load_skip_config
        state.gamma = build_gamma(load_skip_config(args.skip), args.device)
        state.skip_stats = {"pairs": 0, "skipped": 0}
        results["pruned_skip"] = throughput(model, prompts, new_tokens=args.new_tokens)
        results["pruned_skip"]["skip_fraction"] = (state.skip_stats["skipped"]
                                                   / max(state.skip_stats["pairs"], 1))
    dmodel.restore()
    results["param_bytes_pruned"] = param_bytes(model)
    del model
    torch.cuda.empty_cache()
    if args.baseline:
        full = load_base_model(args.baseline, args.device)
        results["full_native"] = throughput(full, prompts, new_tokens=args.new_tokens)
        results["param_bytes_full"] = param_bytes(full)
        results["memory_ratio"] = results["param_bytes_pruned"] / results["param_bytes_full"]
        for key in ("pruned_native", "pruned_diep_forward", "pruned_skip"):
            if key in results:
                results[key]["speedup_vs_full"] = (results[key]["tokens_per_s"]
                                                   / results["full_native"]["tokens_per_s"])
    if "pruned_skip" in results:
        results["pruned_skip"]["speedup_vs_same_forward"] = (
            results["pruned_skip"]["tokens_per_s"] / results["pruned_diep_forward"]["tokens_per_s"])
    write_json(args.out, results)
    print(json.dumps(results, indent=1))


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(prog="diep", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, calibration=True):
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        if calibration:
            p.add_argument("--calibration-file", default=None)
            p.add_argument("--allow-other-calibration", action="store_true",
                           help="only for tests: this study uses c4-32x1024-seed0.pt")

    s = sub.add_parser("search", help="stage 1: learn alpha and beta")
    common(s)
    s.add_argument("--model", default=DEFAULT_MODEL)
    s.add_argument("--out", required=True)
    s.add_argument("--sparsity", type=float, default=None, help="optional: add this r's mask")
    s.add_argument("--m-min", type=int, default=None)
    s.add_argument("--search-forward", choices=["routed", "dense"], default="routed")
    s.add_argument("--epochs", type=int, default=10)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--micro-batch", type=int, default=1)
    s.add_argument("--lr-alpha", type=float, default=5e-3)
    s.add_argument("--lr-beta", type=float, default=5e-3)
    s.add_argument("--update-ratio", default="3:1")
    s.add_argument("--lam", type=float, default=0.01)
    s.add_argument("--recon-target", choices=["hidden", "logits"], default="hidden")
    s.add_argument("--score-mode", choices=["softmax", "raw"], default="softmax")
    s.add_argument("--beta-init", type=float, default=1.0)
    s.add_argument("--beta-min", type=float, default=1e-3)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--ablation", choices=["none", "alpha_only", "beta_only"], default="none")
    s.add_argument("--fixed-beta", default=None, help='"a:b" or "random" (spec §9)')
    s.set_defaults(func=cmd_search)

    p = sub.add_parser("prune", help="stage 2: global ranking and physical pruning")
    common(p)
    p.add_argument("--model", default=None, help="default: the model the search used")
    p.add_argument("--scores", required=True)
    p.add_argument("--sparsity", type=float, required=True, help="fraction of experts removed")
    p.add_argument("--out", required=True)
    p.add_argument("--selection", choices=["global", "uniform"], default="global")
    p.add_argument("--m-min", type=int, default=None, help="per-layer floor (default: top-k)")
    p.add_argument("--merge", action="store_true", help="spec §8 (off by default)")
    p.add_argument("--cka-tokens", type=int, default=2048)
    p.add_argument("--cka-seed", type=int, default=0)
    p.set_defaults(func=cmd_prune)

    c = sub.add_parser("calib-skip", help="stage 3: calibrate adaptive skipping")
    common(c)
    c.add_argument("--model", required=True, help="a pruned checkpoint directory")
    c.add_argument("--out", required=True)
    c.add_argument("--cka-tokens", type=int, default=2048)
    c.add_argument("--cka-seed", type=int, default=0)
    c.set_defaults(func=cmd_calib_skip)

    e = sub.add_parser("eval", help="lm-eval with the GA/DPMA protocol")
    common(e, calibration=False)
    e.add_argument("--model", required=True, help="a pruned checkpoint directory")
    e.add_argument("--skip", default=None, help="skip_config.json: evaluate with skipping")
    e.add_argument("--method", default="diep")
    e.add_argument("--output", required=True)
    e.add_argument("--tasks", nargs="+", default=["all"])
    e.add_argument("--num-fewshot", type=int, default=1)
    e.add_argument("--eval-batch-size", default="auto")
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--replicate", type=int, default=None, help="default: the search seed")
    e.add_argument("--allow-unsafe-code", action="store_true")
    e.add_argument("--force", action="store_true")
    e.set_defaults(func=cmd_eval)

    b = sub.add_parser("bench", help="memory ratio and generation throughput")
    common(b)
    b.add_argument("--model", required=True, help="a pruned checkpoint directory")
    b.add_argument("--skip", default=None)
    b.add_argument("--baseline", default=None, help="the full model, for speedup and memory")
    b.add_argument("--out", required=True)
    b.add_argument("--prompts", type=int, default=20)
    b.add_argument("--prompt-tokens", type=int, default=128)
    b.add_argument("--new-tokens", type=int, default=256)
    b.set_defaults(func=cmd_bench)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, "calibration_file", "unset") is None and args.command != "eval":
        if args.command in ("search", "calib-skip", "bench") or getattr(args, "merge", False):
            sys.exit("--calibration-file is required for this command")
    args.func(args)


if __name__ == "__main__":
    main()
