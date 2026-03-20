# autoresearch-mpp

Autonomous ML research on cloud GPUs, paid via MPP stablecoins. No API keys, no signup.

Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — but instead of a local GPU, experiments run on Modal GPU sandboxes paid with USDC via Tempo.

## Setup

1. **Verify wallet**: `tempo wallet -t whoami`
2. **Read the code**: Read `run.py` (experiment runner) and `research.py` (orchestrator).
3. **Establish baseline**: `python research.py --baseline`
4. **Create branch**: `git checkout -b autoresearch/<tag>` from main.

## Architecture

```
research.py                     run.py                      Modal (via MPP)
  │                               │                            │
  ├── QUICK phase ───────────────►├── POST /sandbox/create ──►│ (pay USDC)
  │   (500 iters, small model)   │◄── sandbox_id ────────────┤
  │                               ├── POST /sandbox/exec ────►│ (train)
  │◄── val_loss ─────────────────┤◄── stdout ────────────────┤
  │                               ├── POST /sandbox/terminate►│
  │   if promising:               │                            │
  │                               │                            │
  ├── FULL phase ────────────────►├── POST /sandbox/create ──►│ (pay USDC)
  │   (5000 iters, bigger model) │   ... same flow ...        │
  │◄── val_loss ─────────────────┤                            │
  │                               │                            │
  ├── log to results.tsv          │                            │
  └── keep / discard              │                            │
```

## The experiment loop

LOOP FOREVER:

1. Read `results.tsv` — understand what's been tried, what worked, current best val_loss
2. Form a hypothesis (architecture change, hyperparameter tweak, optimizer modification, etc.)
3. If your change requires modifying train.py:
   - Read the upstream nanoGPT train.py (clone it or `curl` from GitHub)
   - Create your modified version at `experiments/<name>.py`
   - Run: `python research.py "<hypothesis>" --train-py experiments/<name>.py`
4. If your change is just hyperparameters (LR, batch size, etc.):
   - Run: `python research.py "<hypothesis>" --lr 0.001` (or other flags)
5. Read the output — `val_loss:` and `status:` lines tell you the result
6. If status=keep, your change improved things. Build on it.
7. If status=discard, move on to the next idea.

## Two-phase pipeline

Every experiment runs in two phases:

| Phase | Iters | Model | GPU | Cost | Time | Purpose |
|-------|-------|-------|-----|------|------|---------|
| QUICK | 1000 | 6L/6H/384E (~10M params) | H100 | ~$0.20 | ~30s | Fast filter — does this idea help at all? |
| FULL | 5000 | 8L/8H/512E (~25M params) | H100 | ~$1.50 | ~5min | Real validation — does it hold at scale? |

Default GPU is **H100** (80GB VRAM). Native bfloat16 + torch.compile, no patches needed.

The quick phase gates the full phase. If quick val_loss is >2% worse than the current best, the experiment is discarded without running full. This saves ~$1.50 and ~5 minutes per bad idea.

### Flags

```bash
python research.py "hypothesis" --train-py experiments/foo.py   # full pipeline
python research.py "hypothesis" --quick-only                     # quick only (rapid iteration)
python research.py "hypothesis" --full-only                      # skip quick (scale-dependent ideas)
python research.py "hypothesis" --gpu A10G                       # use a beefier GPU
python research.py "hypothesis" --lr 0.001                       # override learning rate
python research.py "hypothesis" --quick-iters 200                # faster quick phase
python research.py "hypothesis" --threshold 1.05                 # more lenient quick→full gate
```

## What you CAN modify

- `train.py` (nanoGPT's) — architecture, optimizer, hyperparameters, everything
- Create files in `experiments/` — modified train.py variants
- `run.py` flags via `research.py` — GPU type, iteration count, model size

## What you CANNOT modify

- `prepare.py` in upstream nanoGPT (fixed evaluation harness)
- The evaluation metric: val_loss from the final eval step is ground truth
- `research.py` / `run.py` core logic (the experiment infrastructure)

## Logging results

Results are logged to `results.tsv` (tab-separated):

```
run	val_loss	gpu	phase	elapsed	status	description
1	2.374500	T4	quick	85s	quick-pass	baseline
2	2.120300	T4	full	890s	keep	baseline
3	2.350100	T4	quick	92s	quick-pass	increase LR to 6e-4
4	2.098700	T4	full	905s	keep	increase LR to 6e-4
5	2.410000	T4	quick	88s	discard	switch to GeLU activation
```

## Tips

- **Start with quick-only**: Use `--quick-only` for rapid hypothesis testing (~90s per experiment)
- **Scale up winners**: Once a quick experiment shows promise, let it run the full pipeline
- **GPU consistency**: val_loss results are only comparable within the same GPU type
- **Cost**: Quick ~$0.20, Full ~$1.50. Budget ~$25-30 for a 20-experiment session.
- **Custom train.py**: Read nanoGPT's train.py first, then modify surgically. Small diffs = easier debugging.
- **Combine wins**: After finding 2-3 improvements that each help individually, combine them.

## Ideas to try

- Learning rate schedules (cosine, warmup, cyclical)
- Optimizer changes (weight decay, beta values, gradient clipping)
- Architecture (more/fewer layers, wider/narrower, different activation functions)
- Regularization (dropout, layer norm vs RMS norm)
- Training tricks (gradient accumulation, mixed precision settings)
- Positional encoding variants
- Attention modifications (multi-query, grouped-query, sliding window)

## NEVER STOP

Once the loop begins, do NOT pause to ask the human. Run experiments autonomously. The human might be asleep. If you run out of ideas, think harder — try combining previous near-misses, try radical architectural changes, read upstream autoresearch discussions for inspiration. The loop runs until you are manually stopped.
