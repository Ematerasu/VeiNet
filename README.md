# Vei – DNN based AI agent for Scripts of Tribute

Here is a self-play reinforcement learning framework for the deck-building card game Scripts of Tribute (Tales of Tribute, ESO). It orchestrates parallel AI-vs-AI matches, logs full game trajectories, trains a policy/value network via PPO, and evaluates against reference bots.

Basically built as a Proof of Concept for future NN-enhanced approaches, as a result we've got here Vei (Vectorized Embedded Intelligence) bot based on VeiNet architecture.

---

## Table of Contents

1. [Project Structure](#project-structure)  
2. [Installation](#installation)  
3. [Starting Self-Play and Training](#starting-self-play-and-training)  
4. [Vei Bot Architecture](#vei-bot-architecture)  
5. [Neural Network: VeiNet](#neural-network-veinet)  
6. [Encoders and Registry](#encoders-and-registry)  
7. [Self-Play & PPO Training Modules](#self-play--ppo-training-modules)

---

## Project Structure
```
VEI/
├─ AI/
│ └─ vei.py # Bot implementation (Vei)
├─ checkpoints/ # Model checkpoints (weights_*.pt)
├─ data/
│ ├─ cards.json # Card definitions (ID, name, metadata)
│ └─ card_embeddings.npy # Precomputed 65-dim embeddings per card
├─ models/
│ ├─ card_registry.py # Maps card IDs ↔ embeddings
│ ├─ move_encoder.py # Encodes legal moves → fixed vectors
│ ├─ state_encoder.py # Encodes GameState → feature tensors
│ └─ VeiNet.py # Policy/value network definition
├─ replay/ # JSONL logs of game steps (state, action, logp, reward)
├─ vei_train/
│ ├─ launch_workers.py # Launches self-play workers; handles checkpointing and evaluation
│ ├─ selfplay_worker.py # Single‐process self-play match generator
│ ├─ ppo_learner.py # PPO optimization loop
│ └─ vei_eval.py # Model evaluation vs reference bots
├─ main.py # Optional entry point to coordinate training pipeline
└─ README.md # This file
```

---

## Installation

1. Ensure **Python 3.10+** is installed.  
2. Create and activate a virtual environment.  
3. Install dependencies:
   ```bash
   pip install torch numpy scripts-of-tribute
4. Place the following files in `data/`:
    * `cards.json`
    * `card_embeddings.npy`


## Starting Self-Play and Training
### 1. Launch Parallel Self-Play Workers
```bash
python -m vei_train.launch_workers \
  --num-workers 4 \
  --initial-weights ./checkpoints/weights_0.pt \
  --replay-dir ./replay \
  --eval-every 50 \
  --eval-games 200
```
* `--num-workers`: number of concurrent self-play processes

* `--initial-weights`: path to starting model weights

* `--replay-dir`: directory for JSONL trajectory logs

* `--eval-every`: checkpoint interval (in iterations) to trigger evaluation

* `--eval-games`: number of games per evaluation session

### 2. Run a Single Self-Play Worker
```bash
python -m vei_train.selfplay_worker \
  --worker-id 0 \
  --weights ./checkpoints/latest.pt \
  --replay-dir ./replay
```

### 3. Execute PPO Training
```
python -m vei_train.ppo_learner \
  --weights ./checkpoints/latest.pt \
  --replay-dir ./replay
```
### 4. Evaluate a Checkpoint

```bash
python -m vei_train.vei_eval \
  --weights ./checkpoints/weights_150.pt \
  --enemy RandomBot \
  --games 100
```

Schema of the process:
```
                                    ┌────────────────────────┐
                                    │     launch_workers     │
                                    │ (orchestrator & guard) │
                                    └────────────────────────┘
                                               │
                        ┌──────────────────────┼──────────────────────┐
                        │                      │                      │
                Spawn N │                      │                      │
                workers │                      │                      │
                        ▼                      ▼                      ▼

                ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
                │ selfplay_worker1 │     │ selfplay_worker2 │ …   │ selfplay_workerN │
                │ (loop)           │     │ (loop)           │     │ (loop)           │
                └──────────────────┘     └──────────────────┘     └──────────────────┘
                        │                      │                      │
                        │ load weights         │ load weights         │ load weights
                        │ from checkpoints/    │ from checkpoints/    │ from checkpoints/
                        ▼                      ▼                      ▼

                ┌─────────┐              ┌─────────┐                ┌─────────┐
                │   Vei   │              │   Vei   │                │   Vei   │
                │  Agent  │              │  Agent  │                │  Agent  │
                └─────────┘              └─────────┘                └─────────┘
                        │                      │                         │
                        │ interacts via        │ interacts via           │ interacts via
                        │ scripts-of-tribute   │ scripts-of-tribute      │ scripts-of-tribute
                        ▼                      ▼                         ▼

                ┌──────────────────────────────────────────────────────────┐
                │                 Game Runner / Engine                     │
                └──────────────────────────────────────────────────────────┘
                                │ ←──── game state & legal moves ────┐
                                │                                    │
                                └─ play() returns action decisions ─┘
                                │                                   
                                ▼                                   
                        ┌──────────────────┐                       
                        │  game_end()      │                       
                        │ (compute reward) │                       
                        └──────────────────┘                       
                                │                                   
                                │ append step JSON to `replay/`     
                                ▼                                   

                        ┌──────────────────┐
                        │     replay/      │
                        │  *.jsonl logs    │
                        └──────────────────┘
                                ↑
                                │ batched read
                                │
                                │
                        ┌────────────────────────┐
                        │     ppo_learner        │ <-- spawn this process in separate terminal
                        │  (loop)                │
                        └────────────────────────┘
                                    │
                    consume JSONL   │
                    compute losses  │
                    update VeiNet   │
                                    ▼
                        ┌────────────────────────┐
                        │   checkpoints/         │
                        │ weights_*.pt           │
                        └────────────────────────┘
                                        │
                            new checkpoint detected
                            by launch_workers
                                        │
                                        │
                                        ▼
                                ┌───────────────────┐
                                │ launch_workers    │
                                │ (distribute       │
                                │  updated weights) │
                                └───────────────────┘
```
And the loops continue:
- **selfplay_worker**: load weights → play games → log → repeat  
- **ppo_learner**: read logs → train → save weights → repeat  
- **launch_workers**: spawn/manage workers, watch checkpoints, re-deploy weights  


---

## Vei Bot Architecture

The `Vei` class in `AI/vei.py` implements the agent:

1. **Initialization (`pregame_prepare`)**  
   - Load `VeiNet` weights from checkpoint files.  
   - Instantiate `StateEncoder` and `MoveEncoder`.  

2. **Action Selection (`play`)**  
   - Encode the current `GameState` into feature tensors.  
   - Encode all legal moves into action embeddings.  
   - Pass state features through `VeiNet` to obtain latent representation and value estimate.  
   - Compute policy logits over move embeddings via dot-product with the state latent.  
   - Sample or greedy-select an action; return it to the game engine.  

3. **Post-Game Logging (`game_end`)**  
   - Compute reward for each timestep.  
   - Serialize each step as JSON with fields:  
     ```json
     {
       "state": { …encoded features… },
       "action": { …move encoding… },
       "logp": float,
       "reward": float,
       "return": float
     }
     ```  
   - Append to `replay/` as newline-delimited JSON (JSONL).

---

## Neural Network: VeiNet

VeiNet is a Transformer-based policy/value network that aggregates sets of card and agent embeddings, scalar game features, and phase information:

- **SetPool Modules**  
  - One for each card group: hand, played, cooldown, draw pile, tavern.  
  - Two for agents: self and enemy.  
  - Each uses a learned seed vector and `MultiheadAttention` to pool variable-size sets.

- **Feature Encoders**  
  - `card_proj`: projects 65-dim card embeddings → 256-dim.  
  - `agent_proj`: projects 67-dim agent features → 256-dim.  
  - `scalar_enc`: maps numeric game scalars (11 dims) → 256-dim.  
  - `patron_enc`: maps patron favors (10 dims) → 256-dim.  
  - `phase_emb`: learnable embedding for four game phases.
  - `deck_pct_enc`: vector to represent distribution of decks in Vei's card pool.

- **Transformer Trunk**  
    - Interfer all pooled outputs → 10 × 256 dims.  
    - `pre_trunk`: linear + ReLU → 256 dims.  
    - `trans_enc`: two layers of `TransformerEncoder` (8 heads, GELU, feedforward 1024).  
    - `post_proj`: linear + ReLU → 256 dims.  

- **Value Head**  
  - Single linear layer maps 256-dim trunk output → scalar V(s).

```python
# Forward pass sketch
feats = state_encoder(game_state)
move_vecs = move_encoder(possible_moves)
latent, value = VeiNet.forward_state(feats, move_vecs)
```

---

## Encoders and Registry

- **CardRegistry** (`models/card_registry.py`):  
  - Loads `cards.json` and `card_embeddings.npy`.  
  - Provides mapping: `unique_card_id ↔ index`, and embedding lookup.

- **StateEncoder** (`models/state_encoder.py`):  
  - Converts `GameState` into a dict of tensors:  
    - Card sets: `hand`, `played`, `cooldown`, `draw`, `tavern`.  
    - Agent trackers: `agents_self`, `agents_enemy`.  
    - Numeric scalars: `scalars`.  
    - Patron affinities: `patrons`.  
    - Phase index: `phase`.

- **MoveEncoder** (`models/move_encoder.py`):  
  - Enumerates legal moves (type, involved card indices, patrons, targets) and encodes each into a fixed-size vector.

---

## Self-Play & PPO Training Modules

- **selfplay_worker.py**  
  - Loops: select opponent, run a match via `Vei` vs reference bot, write JSONL replay.

- **launch_workers.py**  
  - Spawns multiple `selfplay_worker` processes.  
  - Monitors `checkpoints/` for new weights; copies to each worker.  
  - Triggers `vei_eval.py` at configured intervals.  
  - Aggregates evaluation metrics into `metrics.csv`.

- **ppo_learner.py**  
  1. Load batches of transitions from `replay/`.  
  2. Compute advantages, PPO clipped objective, value loss, entropy bonus.  
  3. Update `VeiNet` parameters; optionally freeze encoders or adjust learning rates.  
  4. Save new checkpoint in `checkpoints/`.

- **vei_eval.py**  
  - Loads specified checkpoint.  
  - Plays a fixed number of matches vs selected reference bots.  
  - Reports win rates.

---
