# autoresearch-mpp

*One day, machines no longer needed human credit cards to rent compute. An agent could walk up to any GPU in the cloud, pay with a stablecoin, train a model, and walk away. No accounts. No API keys. No invoices. Just HTTP 402 and a receipt on-chain. This repo is one way that story begins. —March 2026*

Run [autoresearch](https://github.com/karpathy/autoresearch) experiments on cloud GPUs, paid per-experiment with stablecoins via [MPP](https://mpp.dev) (Machine Payments Protocol). No API keys. No signup. Just HTTP 402.

## How it works

```
Your agent                         Modal (via MPP)
  │                                     │
  ├── POST /sandbox/create ────────────►│
  │◄── 402 Payment Required ───────────┤
  │                                     │
  ├── [pay via Tempo stablecoin] ──────►│
  │◄── 200 {sandbox_id} ──────────────┤
  │                                     │
  ├── POST /sandbox/exec ──────────────►│  ← install deps, prepare data
  ├── POST /sandbox/exec ──────────────►│  ← python train.py (5 min)
  │◄── {val_bpb: 1.482} ──────────────┤
  │                                     │
  ├── POST /sandbox/terminate ─────────►│
  └── done.                             └──
```

The agent pays for a GPU sandbox on [Modal](https://modal.com) through the [Machine Payments Protocol](https://mpp.dev). Payment is settled instantly on the [Tempo](https://tempo.xyz) blockchain in USDC. The agent gets a sandbox, runs the experiment, gets `val_bpb`, and walks away.

## Quick start

**Requirements:** [Tempo CLI](https://tempo.xyz) with a funded wallet.

```bash
# 1. Install Tempo CLI + wallet
curl -L https://tempo.xyz/install | bash
tempo add request
tempo wallet login

# 2. Run an experiment
python run.py
```

That's it. The script creates a T4 GPU sandbox on Modal, installs autoresearch, downloads training data, runs a 2-minute training experiment, and returns `val_bpb`.

## Options

```bash
python run.py                              # defaults: T4, 2min, depth=4
python run.py --gpu A10G --depth 6         # bigger GPU, deeper model
python run.py --time-budget 300            # full 5-minute run
python run.py --train-py my_train.py       # your own train.py
```

## What you see

```
╔══════════════════════════════════════════════════════════════╗
║          autoresearch × MPP × Modal                          ║
╚══════════════════════════════════════════════════════════════╝

[1/6] Creating Modal GPU sandbox via MPP
  💰 Paying via Tempo stablecoin (USDC)...
  ✓ Sandbox created: sb-N6V1EQlpWzYyaWK6BGK5rt (3.2s)
  🔗 Payment settled on Tempo blockchain

[2/6] Installing dependencies
  ✓ PyTorch installed
  ✓ All dependencies installed
  🖥 GPU: Tesla T4 (15GB)

[3/6] Cloning karpathy/autoresearch
  ✓ Repository cloned to /workspace

[4/6] Configuring for T4 (depth=4, budget=120s)
  ✓ Configuration applied

[5/6] Preparing data (download shards + train tokenizer)
  ✓ Data prepared (45s)

[6/6] 🚀 Training (120s budget, depth=4)
  ──────────────────────────────────────────────────────────
  ⭐ val_bpb:          1.482310
     training_seconds: 120.1
     peak_vram_mb:     8240.2
     num_params_M:     6.2
  ──────────────────────────────────────────────────────────

  ━━━ val_bpb: 1.482310 ━━━
```

## The autoresearch loop, paid by machines

The real power: point your coding agent at `program.md` and let it run the [Karpathy loop](https://github.com/karpathy/autoresearch) — modify `train.py`, pay for a GPU, run the experiment, check if `val_bpb` improved, keep or discard, repeat. Each experiment costs a few cents in USDC. The agent pays. You sleep.

## Project structure

```
run.py          — orchestrator: create sandbox, run experiment, get results
program.md      — agent instructions for autonomous experimentation via MPP
```

## Links

- **[MPP](https://mpp.dev)** — Machine Payments Protocol (open standard, co-authored by Stripe & Tempo)
- **[autoresearch](https://github.com/karpathy/autoresearch)** — Karpathy's autonomous ML research loop
- **[Tempo](https://tempo.xyz)** — Infrastructure for machine payments
- **[Modal](https://modal.com)** — Serverless GPU compute

## License

MIT
