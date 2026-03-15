# Deep RL from Human Preferences — PyTorch Reimplementation

PyTorch reimplementation of the paper:

> **Deep Reinforcement Learning from Human Preferences**
> Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei
> NeurIPS 2017 — https://arxiv.org/abs/1706.03741

Based on the original TensorFlow 1 implementation by Matthew Rahtz:
https://github.com/mrahtz/learning-from-human-preferences

---

## What this repo implements

The paper proposes a method to train RL agents **without access to a reward function**, using instead preferences expressed by a human between pairs of short trajectory segments. The pipeline consists of three components running in parallel:

1. **A2C policy** — generates rollouts and sends segments to the preference interface
2. **Preference interface** — shows segment pairs to a human (or simulates preferences synthetically) and forwards labels to a database
3. **Reward predictor** — a CNN ensemble trained on labeled pairs; its output replaces environment rewards during policy training

---

## Installation

Requires Python 3.10+ and a virtual environment.

```bash
git clone <repo-url>
cd learning-from-human-preferences

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

Accept the Atari ROM license when prompted by `ale-py`.

---

## Reproducing the paper

The entry point is `scripts/run.py`. It supports four modes that map directly to the paper pipeline.

### Option A — Full pipeline in one command (recommended)

Runs all three phases sequentially and automatically.
Use `--synthetic_prefs` to simulate human preferences (the agent prefers the segment with higher cumulative environment reward).

```bash
python scripts/run.py train_policy_with_preferences ALE/Pong-v5 \
    --log_dir runs/pong_preferences \
    --synthetic_prefs \
    --n_envs 4 \
    --n_initial_prefs 500 \
    --max_prefs 3000 \
    --n_initial_epochs 50 \
    --million_timesteps 10
```

### Option B — Step by step

**Step 1 — Collect initial preferences**

Runs A2C with environment rewards and sends segment pairs to the preference interface.
Stops automatically once `--n_initial_prefs` labeled pairs have been collected.

```bash
python scripts/run.py gather_initial_prefs ALE/Pong-v5 \
    --log_dir runs/pong_run \
    --synthetic_prefs \
    --n_envs 4 \
    --n_initial_prefs 500 \
    --max_prefs 3000 \
    --million_timesteps 1
```

Produces `runs/pong_run/train.pkl.gz` and `runs/pong_run/val.pkl.gz`.

**Step 2 — Pretrain the reward predictor**

```bash
python scripts/run.py pretrain_reward_predictor ALE/Pong-v5 \
    --log_dir runs/pong_run \
    --load_prefs_dir runs/pong_run \
    --n_initial_epochs 50 \
    --reward_predictor_ckpt_interval 10
```

**Step 3 — Train the policy with the reward predictor**

```bash
python scripts/run.py train_policy_with_preferences ALE/Pong-v5 \
    --log_dir runs/pong_run \
    --synthetic_prefs \
    --n_envs 4 \
    --n_initial_prefs 500 \
    --max_prefs 3000 \
    --n_initial_epochs 50 \
    --million_timesteps 10
```

### Baseline — Train with environment rewards only

```bash
python scripts/run.py train_policy_with_original_rewards ALE/Pong-v5 \
    --log_dir runs/pong_baseline \
    --n_envs 4 \
    --million_timesteps 10
```

### Human preferences (no `--synthetic_prefs`)

Omitting `--synthetic_prefs` opens a video window showing two segments side by side.
Type your choice and press Enter:

| Key | Meaning |
|-----|---------|
| `L` | Left segment is better |
| `R` | Right segment is better |
| `E` | Equal / cannot tell |
| _(empty)_ | Skip this pair |

---

## Output files

After running, `runs/<run_name>/` contains:

```
runs/<run_name>/
├── args.txt                              # CLI arguments used
├── train.pkl.gz                          # Preference database — training split
├── val.pkl.gz                            # Preference database — validation split
├── policy_checkpoints/
│   └── policy_<step>.pt                  # A2C policy checkpoints
├── reward_predictor_checkpoints/
│   └── reward_predictor_<step>.pt        # Reward predictor checkpoints
├── a2c/                                  # TensorBoard — policy training
├── reward_predictor/
│   ├── train/                            # TensorBoard — reward predictor train
│   └── test/                             # TensorBoard — reward predictor val
├── pref_interface/                       # TensorBoard — segments collected
└── pref_buffer/                          # TensorBoard — preferences collected
```

---

## Monitoring with TensorBoard

```bash
tensorboard --logdir runs/
```

Then open `http://localhost:6006`.

| Metric | Where | Meaning |
|--------|-------|---------|
| `a2c/policy_loss` | `a2c/` | Policy gradient loss |
| `a2c/value_loss` | `a2c/` | Value function MSE |
| `a2c/entropy` | `a2c/` | Policy entropy (exploration) |
| `a2c/explained_variance` | `a2c/` | How well the value function predicts returns |
| `a2c/learning_rate` | `a2c/` | Current learning rate |
| `reward_predictor_loss` | `reward_predictor/train/` | Cross-entropy on training preferences |
| `reward_predictor_accuracy` | `reward_predictor/train/` | % preferences correctly predicted |
| `reward_predictor_val_loss` | `reward_predictor/test/` | Cross-entropy on validation preferences |
| `reward_predictor_val_accuracy` | `reward_predictor/test/` | Validation accuracy |
| `preferences/total_received` | `pref_buffer/` | Total labeled pairs collected |
| `preferences/train_db_size` | `pref_buffer/` | Preferences in training database |
| `segments/buffer_size` | `pref_interface/` | Segments available for labeling |

---

## Key hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `--n_envs` | 1 | Parallel environments for A2C |
| `--n_initial_prefs` | 500 | Preferences to collect before reward predictor pretraining |
| `--max_prefs` | 3000 | Maximum preference buffer size (circular) |
| `--n_initial_epochs` | 200 | Reward predictor pretraining epochs |
| `--million_timesteps` | 10 | Total A2C training timesteps (×10⁶) |
| `--lr` | 7e-4 | A2C learning rate |
| `--reward_predictor_learning_rate` | 2e-4 | Reward predictor Adam learning rate |
| `--synthetic_prefs` | off | Use environment rewards to simulate human preferences |
| `--dropout` | 0.0 | Dropout probability in reward predictor CNN |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Main process                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  RewardPredictorEnsemble  (trains on preferences)   │   │
│  │  PrefBuffer thread        (receives labeled pairs)  │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────┬──────────────────────────┬───────────────────┘
               │ seg_pipe                 │ pref_pipe
               ▼                          ▼
┌──────────────────────┐    ┌─────────────────────────────┐
│  Trainer subprocess  │    │  PrefInterface subprocess   │
│  (A2C + Runner)      │    │  (human / synthetic labels) │
│                      │    │  VideoRenderer subprocess   │
│  loads RP checkpoint │    │  (pyglet window)            │
│  every N updates     │    │                             │
└──────────────────────┘    └─────────────────────────────┘
```

The reward predictor is shared between processes via **filesystem checkpoints**: the main process saves a `.pt` file after each training epoch; the trainer subprocess reloads it every `--reward_predictor_ckpt_interval` A2C updates.

---

## Reference

```bibtex
@inproceedings{christiano2017deep,
  title     = {Deep Reinforcement Learning from Human Preferences},
  author    = {Christiano, Paul and Leike, Jan and Brown, Tom B. and
               Martic, Miljan and Legg, Shane and Amodei, Dario},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2017}
}
```
