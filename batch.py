#!/usr/bin/env python3
"""
Run multiple quick-only experiments in parallel batches.
Usage: python3 batch.py
"""
import subprocess, sys, time, threading, json

EXPERIMENTS = [
    # LR sweep
    {"desc": "LR=3e-4", "args": ["--lr", "3e-4"]},
    {"desc": "LR=5e-4", "args": ["--lr", "5e-4"]},
    {"desc": "LR=1.5e-3", "args": ["--lr", "1.5e-3"]},
    {"desc": "LR=2e-3", "args": ["--lr", "2e-3"]},
    {"desc": "LR=3e-3", "args": ["--lr", "3e-3"]},
    # Dropout sweep
    {"desc": "dropout=0.05", "args": ["--dropout", "0.05"]},
    {"desc": "dropout=0.1", "args": ["--dropout", "0.1"]},
    {"desc": "dropout=0.3", "args": ["--dropout", "0.3"]},
    {"desc": "dropout=0.4", "args": ["--dropout", "0.4"]},
    # Combos
    {"desc": "LR=1.5e-3 + dropout=0.1", "args": ["--lr", "1.5e-3", "--dropout", "0.1"]},
    {"desc": "LR=2e-3 + dropout=0.3", "args": ["--lr", "2e-3", "--dropout", "0.3"]},
    {"desc": "LR=5e-4 + dropout=0.3", "args": ["--lr", "5e-4", "--dropout", "0.3"]},
    {"desc": "LR=1e-3 + dropout=0.15", "args": ["--lr", "1e-3", "--dropout", "0.15"]},
    # Iter count
    {"desc": "2000 quick iters", "args": ["--quick-iters", "2000"]},
    {"desc": "500 quick iters", "args": ["--quick-iters", "500"]},
]

BATCH_SIZE = 5
STAGGER_SECS = 3
GPU = "A10G"

results = {}
lock = threading.Lock()


def run_one(exp):
    desc = exp["desc"]
    cmd = [
        sys.executable, "research.py", desc,
        "--gpu", GPU, "--quick-only",
    ] + exp["args"]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        output = r.stdout + r.stderr
        elapsed = time.time() - t0
        # Extract val_loss from output
        val = None
        for line in output.split("\n"):
            if line.strip().startswith("val_loss:"):
                try:
                    val = float(line.strip().split(":")[1].strip())
                except ValueError:
                    pass
        with lock:
            results[desc] = {"val_loss": val, "elapsed": f"{elapsed:.0f}s", "ok": r.returncode == 0}
        status = f"val_loss={val:.6f}" if val else "FAILED"
        print(f"  ✓ {desc}: {status} ({elapsed:.0f}s)")
    except Exception as e:
        with lock:
            results[desc] = {"val_loss": None, "elapsed": "?", "ok": False, "error": str(e)}
        print(f"  ✗ {desc}: {e}")


def main():
    print(f"Running {len(EXPERIMENTS)} experiments in batches of {BATCH_SIZE} on {GPU}\n")

    for batch_start in range(0, len(EXPERIMENTS), BATCH_SIZE):
        batch = EXPERIMENTS[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(EXPERIMENTS) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"--- Batch {batch_num}/{total_batches} ({len(batch)} experiments) ---")

        threads = []
        for i, exp in enumerate(batch):
            t = threading.Thread(target=run_one, args=(exp,))
            threads.append(t)
            t.start()
            if i < len(batch) - 1:
                time.sleep(STAGGER_SECS)

        for t in threads:
            t.join()
        print()

    # Summary
    print("=" * 70)
    print(f"{'Experiment':<40} {'val_loss':>10} {'time':>8} {'ok':>4}")
    print("-" * 70)
    for exp in EXPERIMENTS:
        r = results.get(exp["desc"], {})
        vl = f"{r['val_loss']:.6f}" if r.get("val_loss") else "FAILED"
        print(f"{exp['desc']:<40} {vl:>10} {r.get('elapsed','?'):>8} {'  ✓' if r.get('ok') else '  ✗'}")

    # Best
    valid = [(d, r["val_loss"]) for d, r in results.items() if r.get("val_loss")]
    if valid:
        best = min(valid, key=lambda x: x[1])
        print(f"\n🏆 Best: {best[0]} → val_loss={best[1]:.6f}")


if __name__ == "__main__":
    main()
