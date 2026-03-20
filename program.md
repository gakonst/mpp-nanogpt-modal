# autoresearch-mpp

This is an experiment to have the LLM do its own research, paying for GPU compute via MPP.

## Setup

To set up a new experiment session:

1. **Verify Tempo wallet**: Run `tempo wallet -t whoami` to confirm wallet is funded.
2. **Read the code**: Read `run.py` to understand how experiments are executed via Modal MPP.
3. **Establish baseline**: Run `python run.py` to get the baseline val_bpb. Record it.
4. **Create experiment branch**: `git checkout -b autoresearch/<tag>` from main.
5. **Initialize results.tsv**: Create `results.tsv` with the header row.

## How experiments work

Unlike standard autoresearch (which runs on a local GPU), experiments here run on **cloud GPUs paid via MPP**:

1. You modify `train.py` locally (or create a variant)
2. `run.py` creates a Modal GPU sandbox (paid via Tempo/USDC)
3. It uploads your train.py, runs the experiment, returns val_bpb
4. You compare against the current best and keep or discard

Each experiment costs a few cents in USDC, settled instantly on the Tempo blockchain.

## The experiment loop

LOOP FOREVER:

1. Look at the current best val_bpb and the experiment history in results.tsv
2. Come up with an experimental idea (architecture change, hyperparameter, optimizer tweak)
3. Save your modified train.py (or create a copy)
4. Run: `python run.py --train-py train.py > run.log 2>&1`
5. Read the results: `grep "^val_bpb:" run.log`
6. If val_bpb improved (lower), keep the change. If not, discard.
7. Log to results.tsv

## What you CAN modify

- `train.py` — architecture, optimizer, hyperparameters, batch size, model size, everything
- `run.py` flags — `--gpu`, `--depth`, `--time-budget`, `--num-shards`

## What you CANNOT modify

- `prepare.py` in the upstream repo (it's the fixed evaluation harness)
- The evaluation metric: val_bpb from `evaluate_bpb()` is the ground truth

## Logging results

Log to `results.tsv` (tab-separated):

```
run	val_bpb	gpu	cost_usd	status	description
1	1.482310	T4	0.05	keep	baseline (depth=4, budget=120s)
2	1.451200	T4	0.05	keep	increase LR to 0.06
3	1.490000	T4	0.05	discard	switch to GeLU activation
```

## Tips

- **Start small**: Use T4 + 120s budget for fast iteration (~3 min per experiment including setup)
- **Scale up when promising**: Use `--gpu A10G --depth 6 --time-budget 300` for serious runs
- **GPU matters**: val_bpb results are only comparable across runs on the same GPU type
- **Cost awareness**: Each T4 experiment costs ~$0.05-0.10 in USDC. Budget accordingly.

## NEVER STOP

Once the loop begins, do NOT pause to ask the human. Run experiments autonomously. The human might be asleep. If you run out of ideas, think harder — try combining previous near-misses, try radical architectural changes, read the upstream autoresearch discussions for inspiration. The loop runs until you are manually stopped.
