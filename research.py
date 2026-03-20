#!/usr/bin/env python3
"""
Autoresearch orchestrator for nanoGPT × MPP.

Two-phase experiment pipeline:
  1. QUICK — cheap T4 run (500 iters, small model) to filter bad ideas fast
  2. FULL  — real run (5000 iters, bigger model) to validate promising ones

Usage:
    python research.py --baseline                          # establish baseline
    python research.py "increase LR to 6e-4"              # test with default train.py
    python research.py "cosine warmup" --train-py exp.py   # test custom train.py
    python research.py "bigger model" --full-only          # skip quick, go straight to full
    python research.py "small tweak" --quick-only          # quick only, no full validation
"""
import argparse, json, os, sys, time
from run import run_experiment

RESULTS = "results.tsv"

# Quick phase: fast filter (~30s on H100, ~$0.20)
QUICK = dict(max_iters=1000, n_layer=6, n_head=6, n_embd=384, batch_size=128, block_size=256, timeout=600)

# Full phase: real validation (~5min on H100, ~$1.50)
FULL = dict(max_iters=5000, n_layer=8, n_head=8, n_embd=512, batch_size=128, block_size=256, timeout=1800)


def read_best():
    """Read results.tsv, return (rows, best_val_loss_by_phase)."""
    if not os.path.exists(RESULTS):
        return [], {}
    rows = []
    best = {}  # phase -> best val_loss among "keep" rows
    with open(RESULTS) as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 7:
                continue
            row = dict(zip(
                ["run", "val_loss", "gpu", "phase", "elapsed", "status", "description"],
                parts[:7]
            ))
            rows.append(row)
            if row["status"] == "keep":
                try:
                    vl = float(row["val_loss"])
                    phase = row["phase"]
                    if phase not in best or vl < best[phase]:
                        best[phase] = vl
                except ValueError:
                    pass
    return rows, best


def next_run():
    rows, _ = read_best()
    return len(rows) + 1


def log(run_num, val_loss, gpu, phase, elapsed, status, description):
    """Append a result row to results.tsv."""
    exists = os.path.exists(RESULTS)
    with open(RESULTS, "a") as f:
        if not exists:
            f.write("run\tval_loss\tgpu\tphase\telapsed\tstatus\tdescription\n")
        vl = f"{val_loss:.6f}" if val_loss is not None else "NaN"
        f.write(f"{run_num}\t{vl}\t{gpu}\t{phase}\t{elapsed:.0f}s\t{status}\t{description}\n")


def main():
    p = argparse.ArgumentParser(description="nanoGPT autoresearch via MPP")
    p.add_argument("hypothesis", nargs="?", help="What are you testing?")
    p.add_argument("--train-py", help="Path to modified train.py")
    p.add_argument("--gpu", default="H100")
    p.add_argument("--baseline", action="store_true", help="Run baseline (no custom train.py)")
    p.add_argument("--quick-only", action="store_true", help="Skip full validation phase")
    p.add_argument("--full-only", action="store_true", help="Skip quick, run full directly")
    p.add_argument("--threshold", type=float, default=1.02,
                   help="Quick val_loss must be < best * threshold to proceed to full")
    # Override defaults
    p.add_argument("--quick-iters", type=int, help="Override quick phase iterations")
    p.add_argument("--full-iters", type=int, help="Override full phase iterations")
    p.add_argument("--lr", type=float, help="Learning rate override")
    p.add_argument("--dropout", type=float, help="Dropout override")
    args = p.parse_args()

    if not args.hypothesis and not args.baseline:
        p.error("Provide a hypothesis or --baseline")

    desc = "baseline" if args.baseline else args.hypothesis
    gpu = args.gpu
    rows, best = read_best()

    print(f"{'='*60}")
    print(f"  EXPERIMENT: {desc}")
    print(f"  GPU: {gpu} | train.py: {args.train_py or 'upstream default'}")
    for phase, vl in best.items():
        print(f"  Best {phase} val_loss: {vl:.6f}")
    print(f"{'='*60}\n")

    quick_cfg = {**QUICK, "gpu": gpu, "lr": args.lr, "dropout": args.dropout}
    full_cfg = {**FULL, "gpu": gpu, "lr": args.lr, "dropout": args.dropout, "eval_interval": 500}
    if args.quick_iters:
        quick_cfg["max_iters"] = args.quick_iters
    if args.full_iters:
        full_cfg["max_iters"] = args.full_iters
        full_cfg["eval_interval"] = min(500, args.full_iters)

    # ---- QUICK PHASE ----
    if not args.full_only:
        qi = quick_cfg["max_iters"]
        print(f"  [QUICK] {qi} iters, {quick_cfg['n_layer']}L/{quick_cfg['n_head']}H/{quick_cfg['n_embd']}E\n")

        result = run_experiment(train_py=args.train_py, **quick_cfg)
        qv = result["val_loss"]
        run_num = next_run()

        if qv is None:
            print(f"\n  ✗ QUICK FAILED — no val_loss extracted")
            log(run_num, None, gpu, "quick", result["elapsed"], "crash", desc)
            tail = result["log"].strip().split("\n")[-20:]
            print("\n  --- log tail ---")
            for l in tail:
                print(f"  {l}")
            sys.exit(1)

        print(f"\n  [QUICK] val_loss: {qv:.6f}  ({result['elapsed']:.0f}s)")

        best_quick = best.get("quick")
        if best_quick and not args.baseline:
            if qv < best_quick:
                pct = (1 - qv / best_quick) * 100
                print(f"  ✓ Improved over best quick ({best_quick:.6f}) by {pct:.2f}%")
            elif qv / best_quick <= args.threshold:
                print(f"  ~ Within threshold ({qv/best_quick:.3f}x), proceeding to full")
            else:
                pct = (qv / best_quick - 1) * 100
                print(f"  ✗ Worse than best quick by {pct:.1f}% — discarding")
                log(run_num, qv, gpu, "quick", result["elapsed"], "discard", desc)
                print(f"\nval_loss: {qv:.6f}")
                print("status: discard")
                return

        status = "keep" if args.quick_only else "quick-pass"
        log(run_num, qv, gpu, "quick", result["elapsed"], status, desc)

        if args.quick_only:
            print(f"\nval_loss: {qv:.6f}")
            print("status: keep (quick-only)")
            return

    # ---- FULL PHASE ----
    fi = full_cfg["max_iters"]
    nl, nh, ne = full_cfg["n_layer"], full_cfg["n_head"], full_cfg["n_embd"]
    print(f"\n  [FULL] {fi} iters, {nl}L/{nh}H/{ne}E\n")

    result = run_experiment(train_py=args.train_py, **full_cfg)
    fv = result["val_loss"]
    run_num = next_run()

    if fv is None:
        print(f"\n  ✗ FULL FAILED — no val_loss extracted")
        log(run_num, None, gpu, "full", result["elapsed"], "crash", desc)
        tail = result["log"].strip().split("\n")[-20:]
        print("\n  --- log tail ---")
        for l in tail:
            print(f"  {l}")
        sys.exit(1)

    best_full = best.get("full")
    if best_full and not args.baseline:
        if fv < best_full:
            pct = (1 - fv / best_full) * 100
            status = "keep"
            print(f"\n  ✓ NEW BEST: {fv:.6f} (was {best_full:.6f}, -{pct:.2f}%)")
        else:
            status = "discard"
            print(f"\n  ✗ No improvement: {fv:.6f} (best: {best_full:.6f})")
    else:
        status = "keep"
        print(f"\n  Baseline full val_loss: {fv:.6f}")

    log(run_num, fv, gpu, "full", result["elapsed"], status, desc)

    print(f"\nval_loss: {fv:.6f}")
    print(f"status: {status}")


if __name__ == "__main__":
    main()
