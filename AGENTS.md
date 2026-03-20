# AGENTS.md — Modal MPP on Tempo

## What this repo does

Runs nanoGPT training on Modal GPU sandboxes, paid via MPP (Machine Payments Protocol) using Tempo stablecoins. No API keys, no signup.

## Modal MPP API (`https://modal.mpp.tempo.xyz`)

All calls go through `tempo request -t -X POST --json '...'`.

### Endpoints

| Endpoint | Request | Response |
|---|---|---|
| `/sandbox/create` | `{"gpu": "T4", "timeout": 900, "image": "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"}` | `{"sandbox_id": "sb-xxx"}` |
| `/sandbox/exec` | `{"sandbox_id": "sb-xxx", "command": ["bash", "-c", "..."]}` | `{"stdout": "...", "stderr": "...", "returncode": 0}` |
| `/sandbox/status` | `{"sandbox_id": "sb-xxx"}` | `{"sandbox_id": "...", "status": "running"}` |
| `/sandbox/terminate` | `{"sandbox_id": "sb-xxx"}` | `{"status": "terminated"}` |

### Key learnings

1. **Use pre-built Docker images via `"image"` field.** The `image` field accepts any Docker registry string. `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` comes with torch, numpy, requests pre-installed. This avoids the 60s+ pip install torch penalty.

2. **Exec has a ~30s timeout.** Long-running commands (pip install torch, training) will return `returncode: -1` if they exceed the exec timeout. Use the **background + poll pattern**:
   ```bash
   # Start in background
   {"command": ["bash", "-c", "my_long_command > /tmp/out.log 2>&1 && echo DONE > /tmp/ok &\necho started"]}
   
   # Poll for completion
   {"command": ["bash", "-c", "cat /tmp/ok 2>/dev/null || echo WAIT"]}
   
   # Read results
   {"command": ["cat", "/tmp/out.log"]}
   ```

3. **Each tempo request costs time + money.** Minimize API calls. Batch work into single bash scripts. Don't make unnecessary status checks.

4. **GPU options:** `T4` (16GB), `A10G` (24GB), `A100-40GB`, `A100-80GB`, `H100`. Use `T4` for demos.

5. **No git in pytorch image.** Install with `apt-get update -qq && apt-get install -y -qq git >/dev/null 2>&1`.

6. **T4 + PyTorch 2.5 gotchas:**
   - `torch.compile` fails on T4 (Turing arch). Always disable: `sed -i "s/compile = True/compile = False/" train.py`
   - T4 doesn't support bfloat16 natively. Use float16: `sed -i "s/dtype = 'bfloat16'/dtype = 'float16'/" train.py`

7. **Image contents** (`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`):
   - ✅ torch 2.5.1+cu124, numpy 2.1.2, requests
   - ❌ git, curl, tiktoken, tqdm (need to install)

8. **Session-based billing.** `/sandbox/create` opens a payment session. Subsequent exec/status/terminate calls use that session. If the session expires, exec calls return `E_PAYMENT` error — create a new sandbox.

9. **Don't install packages via `{"image": {"python_packages": [...]}}`.** This causes 500 errors. Use Docker image string or pip install inside the sandbox.

## Working recipe for nanoGPT on T4

```bash
# 1. Create sandbox (~14s)
tempo request -t -X POST --json '{"gpu":"T4","timeout":900,"image":"pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"}' https://modal.mpp.tempo.xyz/sandbox/create

# 2. Setup + train in one background exec
tempo request -t -X POST --json '{"sandbox_id":"sb-xxx","command":["bash","-c","(apt-get update -qq && apt-get install -y -qq git >/dev/null 2>&1 && pip install -q tiktoken && git clone --depth 1 https://github.com/karpathy/nanoGPT.git /workspace && cd /workspace && sed -i \"s/compile = True/compile = False/\" train.py && sed -i \"s/dtype = .bfloat16./dtype = .float16./\" train.py && python data/shakespeare_char/prepare.py && python train.py config/train_shakespeare_char.py --device=cuda --max_iters=500 --n_layer=4 --n_head=4 --n_embd=128 --batch_size=32 && python sample.py --out_dir=out-shakespeare-char --device=cuda --num_samples=1 --max_new_tokens=500 --dtype=float16) > /tmp/run.log 2>&1 && echo DONE > /tmp/ok &"]}' https://modal.mpp.tempo.xyz/sandbox/exec

# 3. Poll (every 12s)
tempo request -t -X POST --json '{"sandbox_id":"sb-xxx","command":["bash","-c","cat /tmp/ok 2>/dev/null || echo WAIT; tail -2 /tmp/run.log"]}' https://modal.mpp.tempo.xyz/sandbox/exec

# 4. Read results
tempo request -t -X POST --json '{"sandbox_id":"sb-xxx","command":["cat","/tmp/run.log"]}' https://modal.mpp.tempo.xyz/sandbox/exec

# 5. Terminate
tempo request -t -X POST --json '{"sandbox_id":"sb-xxx"}' https://modal.mpp.tempo.xyz/sandbox/terminate
```

Total time: ~90-110s. Loss: 4.19 → 2.37 (500 iters). Generates babbling Shakespeare.

## Tempo stdout parsing

The `-t` flag outputs YAML-like format. `stdout` is a single-line string with `\n` escape sequences:
```
stdout: "line1\nline2\nline3\n"
stderr: ""
returncode: 0
```

**Gotcha:** Long stdout (>~5000 chars) gets truncated. Use `head`/`tail` to read large log files in chunks instead of `cat`.
