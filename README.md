# mpp-nanogpt-modal

*One day, machines no longer needed human credit cards to rent compute. An agent could walk up to any GPU in the cloud, pay with a stablecoin, train a model, and walk away. No accounts. No API keys. No invoices. Just HTTP 402 and a receipt on-chain. This repo is one way that story begins. —March 2026*

Train [nanoGPT](https://github.com/karpathy/nanoGPT) on a cloud GPU, paid with stablecoins via [MPP](https://mpp.dev). No API keys. No signup. Just HTTP 402.

## Quick start

```bash
# 1. Install Tempo CLI + wallet
curl -L https://tempo.xyz/install | bash
tempo add request
tempo wallet login

# 2. Train a GPT on Shakespeare, paid with stablecoins
python run.py
```

## Output

```
╔══════════════════════════════════════════════════╗
║      nanoGPT × MPP × Modal                       ║
║  Train GPT on Shakespeare, paid in stablecoins.  ║
║  No API keys. No signup. Just HTTP 402.          ║
╚══════════════════════════════════════════════════╝

  💰 Creating T4 sandbox — paying USDC via Tempo...
  ✓ Sandbox sb-i0wAVkwVmw1acszCAnUoh8 (15s)
    Image: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
    GPU: Tesla T4 (16GB)

  🚀 Launching experiment...
    git clone nanoGPT → prepare shakespeare → train 500 iters → sample

    [ 12s] ---------------
    [ 61s] ---------------

  ✓ Done in 97s

  ════════════════════════════════════════════════════
  length of dataset in characters: 1,115,394
  vocab size: 65
  number of parameters: 0.80M
  step 0: train loss 4.1920, val loss 4.1879
  iter 50: loss 2.8890, time 10.79ms, mfu 1.55%
  step 100: train loss 2.5571, val loss 2.5548
  step 200: train loss 2.4717, val loss 2.4678
  step 300: train loss 2.4222, val loss 2.4234
  step 400: train loss 2.3789, val loss 2.4059
  step 500: train loss 2.3615, val loss 2.3745

  Generated Shakespeare:
  CANIIO:
  Ror wcowind to layer thadise myobe t eranthand my dalatangs
  ar hapar us he he. F dilasoate Iwice my.
  DELOY:
  Gorou wat thertof isth ble mil ndill, ath iree senghin lat
  Herid ov the and th theanoureransesel lind te l.
  ════════════════════════════════════════════════════

╔══════════════════════════════════════════════════╗
║  GPU compute paid via Machine Payments Protocol  ║
║  No API keys. No signup.  https://mpp.dev        ║
╚══════════════════════════════════════════════════╝
```

**97 seconds.** Loss 4.19 → 2.37. Babbling Shakespeare from a 0.8M parameter GPT, trained on a cloud T4 GPU paid with USDC stablecoins.

## How it works

1. `run.py` calls `tempo request` to create a Modal GPU sandbox — the HTTP 402 payment flow is handled automatically by the Tempo CLI
2. One background exec runs everything: install git + tiktoken → clone nanoGPT → prepare Shakespeare data (1MB) → train 500 iters → generate sample text
3. Polls for completion every 12s, streams status
4. Reads training log + generated text, terminates sandbox

The pre-built Docker image (`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`) comes with torch + numpy + requests. Only `git` and `tiktoken` need installing at runtime (~10s).

## What happens under the hood

```
run.py                              Modal (via MPP)
  │                                     │
  ├── POST /sandbox/create ────────────►│
  │◄── 402 Payment Required ───────────┤
  │                                     │
  ├── [pay USDC on Tempo] ────────────►│
  │◄── 200 {sandbox_id} ──────────────┤
  │                                     │
  ├── POST /sandbox/exec ──────────────►│  git clone, prepare, train, sample
  │    (background + poll)              │  (~75s on T4)
  │◄── {stdout, val_loss} ────────────┤
  │                                     │
  ├── POST /sandbox/terminate ─────────►│
  └── done.                             └──
```

## Links

- **[MPP](https://mpp.dev)** — Machine Payments Protocol (open standard, co-authored by Stripe & Tempo)
- **[nanoGPT](https://github.com/karpathy/nanoGPT)** — The simplest, fastest repository for training/finetuning medium-sized GPTs
- **[Tempo](https://tempo.xyz)** — Infrastructure for machine payments
- **[Modal](https://modal.com)** — Serverless GPU compute

## License

MIT
