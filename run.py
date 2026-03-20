#!/usr/bin/env python3
"""
Run a single nanoGPT experiment on a Modal GPU sandbox, paid via MPP.

Usage:
    python run.py                                          # baseline demo
    python run.py --max-iters 500 --n-layer 4              # quick experiment
    python run.py --train-py experiments/cosine.py --gpu T4 # custom train.py
"""
import argparse, base64, json, os, re, subprocess, sys, time

TEMPO = os.path.expanduser("~/.local/bin/tempo")
MODAL = "https://modal.mpp.tempo.xyz"
IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"

B, D, G, Y, C, M, X = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[36m", "\033[35m", "\033[0m"


def tempo(path, data, quiet=False, retries=3):
    for attempt in range(retries):
        if not quiet:
            print(f"  {D}→ POST {path}{X}", file=sys.stderr)
        r = subprocess.run(
            [TEMPO, "request", "-t", "-X", "POST", "--json", json.dumps(data), MODAL + path],
            capture_output=True, text=True, timeout=1200,
        )
        out = r.stdout.strip()
        if r.returncode != 0:
            is_payment_err = "E_PAYMENT" in out or "payment" in out.lower()
            if is_payment_err and attempt < retries - 1:
                wait = 5 * (attempt + 1)
                if not quiet:
                    print(f"  {Y}⚠ Payment error, retrying in {wait}s (attempt {attempt+2}/{retries})...{X}", file=sys.stderr)
                time.sleep(wait)
                continue
            raise RuntimeError(f"tempo error (rc={r.returncode}): {r.stderr.strip()}\n{out[:500]}")
        parsed = {}
        for line in out.split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, _, v = line.partition(":")
            k = k.strip()
            if k in ("sandbox_id", "stdout", "stderr", "returncode", "status"):
                parsed[k] = v.strip().strip('"').replace("\\n", "\n")
        return parsed
    raise RuntimeError("tempo: all retries exhausted")


def ex(sb, cmd, quiet=False):
    return tempo("/sandbox/exec", {"sandbox_id": sb, "command": cmd}, quiet=quiet)


def run_experiment(
    train_py=None, gpu="T4", max_iters=500,
    n_layer=4, n_head=4, n_embd=128,
    batch_size=32, block_size=256,
    timeout=900, eval_interval=None, eval_iters=20,
    lr=None, dropout=None, quiet=False,
):
    """
    Run one nanoGPT experiment on a Modal GPU sandbox.

    Returns dict:
        val_loss     - final validation loss (float or None)
        train_loss   - final training loss (float or None)
        elapsed      - wall clock seconds (float)
        gpu          - GPU type used
        max_iters    - iterations trained
        log          - full training log text
    """
    if eval_interval is None:
        eval_interval = max_iters

    sb = None
    try:
        # 1. Create sandbox
        if not quiet:
            print(f"  {C}Creating {gpu} sandbox...{X}", file=sys.stderr)
        t0 = time.time()
        r = tempo("/sandbox/create", {"gpu": gpu, "timeout": timeout, "image": IMAGE}, quiet=quiet)
        sb = r.get("sandbox_id")
        if not sb:
            raise RuntimeError(f"No sandbox_id in response: {r}")
        if not quiet:
            print(f"  {G}✓ Sandbox {sb} ({time.time()-t0:.0f}s){X}", file=sys.stderr)

        # 2. Build the training script
        parts = [
            "set -e",
            "apt-get update -qq && apt-get install -y -qq git >/dev/null 2>&1",
            "pip install -q tiktoken",
            "git clone --depth 1 https://github.com/karpathy/nanoGPT.git /workspace 2>/dev/null",
            "cd /workspace",
        ]

        # GPU-specific patches
        # Always disable torch.compile — container image lacks a C compiler
        parts.append('sed -i "s/compile = True/compile = False/" train.py')
        if gpu == "T4":
            parts.append("sed -i \"s/dtype = 'bfloat16'/dtype = 'float16'/\" train.py")

        # Upload custom train.py if provided
        if train_py:
            with open(train_py, "r") as f:
                content = f.read()
            b64 = base64.b64encode(content.encode()).decode()
            parts.append(f'echo "{b64}" | base64 -d > /workspace/train.py')

        # Determine dtype for sampling
        dtype = "float16" if gpu == "T4" else "bfloat16"

        # Training arguments
        train_args = [
            f"--device=cuda",
            f"--max_iters={max_iters}",
            f"--lr_decay_iters={max_iters}",
            f"--eval_interval={eval_interval}",
            f"--log_interval=100",
            f"--n_layer={n_layer}",
            f"--n_head={n_head}",
            f"--n_embd={n_embd}",
            f"--batch_size={batch_size}",
            f"--block_size={block_size}",
            f"--eval_iters={eval_iters}",
        ]
        if lr is not None:
            train_args.append(f"--learning_rate={lr}")
        if dropout is not None:
            parts.append(f"sed -i \"s/dropout = [0-9.]*/dropout = {dropout}/\" config/train_shakespeare_char.py")
            train_args.append(f"--dropout={dropout}")

        parts.append("python data/shakespeare_char/prepare.py")
        parts.append(f"python train.py config/train_shakespeare_char.py {' '.join(train_args)}")
        parts.append(
            f"python sample.py --out_dir=out-shakespeare-char --device=cuda"
            f" --num_samples=1 --max_new_tokens=200 --dtype={dtype}"
        )

        script = "\n".join(parts)

        # 3. Launch in background
        if not quiet:
            print(f"  {C}Training {max_iters} iters ({n_layer}L/{n_head}H/{n_embd}E, bs={batch_size})...{X}", file=sys.stderr)
        ex(sb, ["bash", "-c", f"({script}) > /tmp/run.log 2>&1 && echo DONE > /tmp/ok &\necho started"], quiet=quiet)

        # 4. Poll for completion
        poll_interval = 15 if max_iters <= 1000 else 30
        session_expired = False
        while time.time() - t0 < timeout:
            time.sleep(poll_interval)
            try:
                r = ex(sb, ["bash", "-c", "cat /tmp/ok 2>/dev/null || echo WAIT"], quiet=True)
            except RuntimeError as e:
                if "E_PAYMENT" in str(e) or "payment" in str(e).lower():
                    if not quiet:
                        print(f"  {Y}⚠ Payment session expired during polling{X}", file=sys.stderr)
                    session_expired = True
                    break
                raise
            done = "DONE" in (r.get("stdout", "") or "")

            if not quiet:
                try:
                    r2 = ex(sb, ["bash", "-c", "tail -1 /tmp/run.log 2>/dev/null"], quiet=True)
                    tail = (r2.get("stdout", "") or "").strip().split("\n")[-1].strip()
                except RuntimeError:
                    tail = "?"
                elapsed = int(time.time() - t0)
                mark = f"{G}✓ DONE{X}" if done else f"{D}{tail[:80]}{X}"
                print(f"  {D}[{elapsed:>4}s]{X} {mark}", file=sys.stderr)

            if done:
                break

        elapsed = time.time() - t0

        # 5. Read full log
        log = ""
        if not session_expired:
            try:
                r = ex(sb, ["cat", "/tmp/run.log"], quiet=True)
                log = r.get("stdout", "") or ""
            except RuntimeError:
                if not quiet:
                    print(f"  {Y}⚠ Could not read log (session expired){X}", file=sys.stderr)

        # 6. Extract metrics from log
        # nanoGPT prints: "step N: train loss X.XXXX, val loss Y.YYYY"
        val_loss = None
        train_loss = None
        for line in log.split("\n"):
            m = re.search(r"train loss ([\d.]+),?\s*val loss ([\d.]+)", line)
            if m:
                train_loss = float(m.group(1))
                val_loss = float(m.group(2))

        return {
            "val_loss": val_loss,
            "train_loss": train_loss,
            "elapsed": round(elapsed, 1),
            "gpu": gpu,
            "max_iters": max_iters,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "log": log,
        }

    finally:
        if sb:
            if not quiet:
                print(f"  {D}Terminating sandbox...{X}", file=sys.stderr)
            try:
                tempo("/sandbox/terminate", {"sandbox_id": sb}, quiet=True)
            except Exception:
                pass


def main():
    p = argparse.ArgumentParser(description="Run a nanoGPT experiment on Modal GPU via MPP")
    p.add_argument("--train-py", help="Path to custom train.py (default: upstream nanoGPT)")
    p.add_argument("--gpu", default="T4")
    p.add_argument("--max-iters", type=int, default=500)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--eval-interval", type=int)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--lr", type=float)
    p.add_argument("--dropout", type=float)
    p.add_argument("--quiet", "-q", action="store_true")
    args = p.parse_args()

    result = run_experiment(
        train_py=args.train_py, gpu=args.gpu, max_iters=args.max_iters,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        batch_size=args.batch_size, block_size=args.block_size,
        timeout=args.timeout, eval_interval=args.eval_interval,
        eval_iters=args.eval_iters, lr=args.lr, dropout=args.dropout, quiet=args.quiet,
    )

    # Structured output to stdout (log excluded — too large)
    out = {k: v for k, v in result.items() if k != "log"}
    print(json.dumps(out, indent=2))

    # Key metric on its own line for easy grepping
    if result["val_loss"] is not None:
        print(f"\nval_loss: {result['val_loss']:.6f}")
    else:
        print("\nval_loss: FAILED (check log)")
        # Dump tail of log for debugging
        log_lines = result["log"].strip().split("\n")
        print("\n--- last 30 lines of log ---")
        for line in log_lines[-30:]:
            print(line)
        sys.exit(1)


if __name__ == "__main__":
    main()
