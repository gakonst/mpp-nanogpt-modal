#!/usr/bin/env python3
"""
nanoGPT × MPP: Train a GPT on Shakespeare, paid with stablecoins.
No API keys. No signup. Just HTTP 402.
"""
import json, os, subprocess, sys, time

TEMPO = os.path.expanduser("~/.local/bin/tempo")
MODAL = "https://modal.mpp.tempo.xyz"
IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"

B, D, G, Y, C, M, X = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[36m", "\033[35m", "\033[0m"


def tempo(path, data):
    r = subprocess.run(
        [TEMPO, "request", "-t", "-X", "POST", "--json", json.dumps(data), MODAL + path],
        capture_output=True, text=True, timeout=1200,
    )
    out = r.stdout.strip()
    if r.returncode != 0:
        print(f"  {Y}✗ {r.stderr.strip() or out}{X}")
        sys.exit(1)
    parsed = {}
    for line in out.split("\n"):
        if ":" in line:
            k, _, v = line.partition(":")
            parsed[k.strip()] = v.strip().strip('"').replace("\\n", "\n")
    return parsed


def ex(sb, cmd):
    return tempo("/sandbox/exec", {"sandbox_id": sb, "command": cmd})


# Everything the sandbox needs to do, in one script.
# pytorch image already has: torch, numpy, requests. We just need git + tiktoken.
SETUP_AND_TRAIN = r"""
set -e

# Install git + tiktoken (only missing pieces)
apt-get update -qq && apt-get install -y -qq git >/dev/null 2>&1
pip install -q tiktoken

# Clone nanoGPT
git clone --depth 1 https://github.com/karpathy/nanoGPT.git /workspace 2>/dev/null
cd /workspace

# Disable torch.compile (T4 compat) + use float16
sed -i "s/compile = True/compile = False/" train.py
sed -i "s/dtype = 'bfloat16'/dtype = 'float16'/" train.py

# Prepare Shakespeare data (~1MB download)
python data/shakespeare_char/prepare.py

# Train baby GPT (200 iters, ~15s on T4)
python train.py config/train_shakespeare_char.py \
    --device=cuda --max_iters=500 --lr_decay_iters=500 \
    --eval_interval=100 --log_interval=50 \
    --n_layer=4 --n_head=4 --n_embd=128 \
    --batch_size=32 --block_size=256 --eval_iters=10

# Generate Shakespeare
python sample.py --out_dir=out-shakespeare-char --device=cuda \
    --num_samples=1 --max_new_tokens=500 --dtype=float16
"""


def main():
    print(f"""
{B}{M}╔══════════════════════════════════════════════════╗
║      nanoGPT × MPP × Modal                       ║
║  Train GPT on Shakespeare, paid in stablecoins.  ║
║  No API keys. No signup. Just HTTP 402.          ║
╚══════════════════════════════════════════════════╝{X}
""")

    sb = None
    try:
        # 1. Create GPU sandbox (payment happens here)
        print(f"  {B}💰 Creating T4 sandbox — paying USDC via Tempo...{X}")
        t0 = time.time()
        r = tempo("/sandbox/create", {"gpu": "T4", "timeout": 900, "image": IMAGE})
        sb = r.get("sandbox_id")
        print(f"  {G}✓ Sandbox {B}{sb}{X} {D}({time.time()-t0:.0f}s){X}")
        print(f"  {D}  Image: {IMAGE}{X}")
        print(f"  {D}  GPU: Tesla T4 (16GB){X}\n")

        # 2. Run everything in one background exec
        print(f"  {B}🚀 Launching experiment...{X}")
        print(f"  {D}  git clone nanoGPT → prepare shakespeare → train 500 iters → sample{X}\n")

        ex(sb, ["bash", "-c",
            f"({SETUP_AND_TRAIN}) > /tmp/run.log 2>&1 && echo DONE > /tmp/ok &\necho started"])

        # 3. Poll + stream log tail
        t0 = time.time()
        while time.time() - t0 < 600:
            time.sleep(12)
            elapsed = int(time.time() - t0)

            r = ex(sb, ["bash", "-c", "cat /tmp/ok 2>/dev/null || echo WAIT"])
            done = "DONE" in (r.get("stdout", "") or "")

            r2 = ex(sb, ["bash", "-c", "tail -2 /tmp/run.log 2>/dev/null"])
            tail = (r2.get("stdout", "") or "").strip().split("\n")[-1].strip()
            print(f"  {D}  [{elapsed:>3}s] {tail[:90]}{X}")

            if done:
                break

        dt = time.time() - t0
        print(f"\n  {G}{B}✓ Done in {dt:.0f}s{X}\n")

        # 4. Read full log
        r = ex(sb, ["cat", "/tmp/run.log"])
        log = (r.get("stdout", "") or "")

        print(f"  {B}{'═' * 52}{X}")
        in_sample = False
        skip_prefixes = ("Overriding", "/", "Loading", "Initializing", "num ", "using ",
                         "found ", "tokens ", "saving ", "step ", "#", "out_dir", "eval_",
                         "log_", "always_", "wandb_", "dataset", "gradient_", "batch_",
                         "block_", "n_layer", "n_head", "n_embd", "dropout", "learning_",
                         "max_iters", "lr_decay", "min_lr", "beta2", "warmup_", "device",
                         "compile", "scaler", "$ python")
        for line in log.split("\n"):
            s = line.strip()
            if not s:
                if in_sample:
                    print()
                continue
            if "FutureWarning" in s or "torch.load" in s or "deprecated" in s:
                continue
            if "length of dataset" in s or "vocab size" in s or "has" in s and "tokens" in s:
                print(f"  {D}{s}{X}")
            elif s.startswith("number of parameters"):
                print(f"  {C}{s}{X}")
            elif "train loss" in s and "val loss" in s:
                print(f"  {G}{B}{s}{X}")
            elif s.startswith("iter"):
                print(f"  {D}{s}{X}")
            elif "------" in s:
                in_sample = not in_sample
                if in_sample:
                    print(f"\n  {B}Generated Shakespeare:{X}")
            elif in_sample:
                print(f"  {M}{s}{X}")
            elif not s.startswith(skip_prefixes):
                pass  # skip config noise
        print(f"  {B}{'═' * 52}{X}")

        print(f"""
{B}{M}╔══════════════════════════════════════════════════╗
║  GPU compute paid via Machine Payments Protocol  ║
║  No API keys. No signup.  https://mpp.dev        ║
╚══════════════════════════════════════════════════╝{X}
""")

    except KeyboardInterrupt:
        print(f"\n  {Y}Interrupted.{X}")
    except Exception as e:
        print(f"\n  {Y}Error: {e}{X}")
        import traceback; traceback.print_exc()
    finally:
        if sb:
            print(f"  {D}Terminating sandbox...{X}")
            try:
                tempo("/sandbox/terminate", {"sandbox_id": sb})
                print(f"  {G}✓ Terminated{X}")
            except Exception:
                pass


if __name__ == "__main__":
    main()
