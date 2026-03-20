# mpp-nanogpt-modal

*One day, machines no longer needed human credit cards to rent compute. An agent could walk up to any GPU in the cloud, pay with a stablecoin, train a model, and walk away. No accounts. No API keys. No invoices. Just HTTP 402 and a receipt on-chain. This repo is one way that story begins. —March 2026*

Train [nanoGPT](https://github.com/karpathy/nanoGPT) on a cloud GPU, paid with stablecoins via [MPP](https://mpp.dev). No API keys. No signup. Just HTTP 402.

## What happens

```
run.py                              Modal (via MPP)
  │                                     │
  ├── POST /sandbox/create ────────────►│
  │◄── 402 Payment Required ───────────┤
  │                                     │
  ├── [pay USDC on Tempo] ────────────►│
  │◄── 200 {sandbox_id} ──────────────┤
  │                                     │
  ├── POST /sandbox/exec ──────────────►│  clone nanoGPT, prepare data,
  │    (background + poll)              │  train 500 iters, sample text
  │◄── {stdout, val_loss} ────────────┤
  │                                     │
  ├── POST /sandbox/terminate ─────────►│
  └── done.                             └──
```

An agent pays for a T4 GPU sandbox on [Modal](https://modal.com) through the [Machine Payments Protocol](https://mpp.dev). Payment settles instantly on the [Tempo](https://tempo.xyz) blockchain in USDC. The sandbox runs nanoGPT on Shakespeare — 500 training iterations, then generates text. ~90 seconds end-to-end.

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
╚══════════════════════════════════════════════════╝

  💰 Creating T4 sandbox — paying USDC via Tempo...
  ✓ Sandbox sb-XfsQooNM352iel0w1QkWvZ (14s)

  🚀 Launching experiment...
    [ 12s] scaler = torch.cuda.amp.GradScaler(...)
    [ 82s] ---------------

  ✓ Done in 90s

  ════════════════════════════════════════════════════
  vocab size: 65
  number of parameters: 0.80M

  step 0: train loss 4.1920, val loss 4.1879
  step 100: train loss 2.5571, val loss 2.5548
  step 200: train loss 2.4717, val loss 2.4678
  step 300: train loss 2.4222, val loss 2.4234
  step 400: train loss 2.3789, val loss 2.4059
  step 500: train loss 2.3615, val loss 2.3745

  Generated Shakespeare:
  CANIIO:
  Ror wcowind to layer thadise myobe t eranthand my dalatangs...
  ════════════════════════════════════════════════════
```

Loss: **4.19 → 2.37** in 500 iterations. Babbling Shakespeare from a 0.8M parameter GPT, trained on a cloud GPU paid with stablecoins.

## How it works

1. `run.py` calls `tempo request` to create a Modal GPU sandbox (HTTP 402 → pay USDC → get sandbox)
2. One background exec runs everything: `git clone nanoGPT` → prepare Shakespeare data → train → sample
3. Polls for completion, reads the training log
4. Terminates the sandbox

The pre-built Docker image (`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`) has torch + numpy + requests already installed. Only git and tiktoken need installing at runtime.

## Links

- **[MPP](https://mpp.dev)** — Machine Payments Protocol (open standard, co-authored by Stripe & Tempo)
- **[nanoGPT](https://github.com/karpathy/nanoGPT)** — The simplest, fastest repository for training/finetuning medium-sized GPTs
- **[Tempo](https://tempo.xyz)** — Infrastructure for machine payments
- **[Modal](https://modal.com)** — Serverless GPU compute

## License

MIT
