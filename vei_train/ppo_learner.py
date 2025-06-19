import argparse, glob, json, math, os, random, time, pathlib
import torch, torch.nn.functional as F
from torch import nn
import numpy as np
import gc
import itertools
from tqdm import tqdm

from models.VeiNet import SimpleVeiNet, VeiNet
from models.move_encoder import MoveEncoder 
from scripts_of_tribute.enums import MoveEnum, PatronId
from scripts_of_tribute.move import (
    BasicMove,
    MakeChoiceMoveUniqueCard,
    MakeChoiceMoveUniqueEffect,
    SimpleCardMove,
    SimplePatronMove,
)

"""
DISCLAIMER:
This file became quite a mess, because I had to constantly tweak, debug etc.
Overall it works, but be aware that jumping into this and trying to understand logic behind it might be hard task.
All params to tweak things are here at the top, simply run this file and adjust params to logs you receive in the terminal
"""


# ──────────────────────  hyperparam  ───────────────────── #
BATCH       = 1024
MINI        = 128
EPOCHS      = 4
LR          = 1e-6
CLIP_EPS    = 0.2
CLIP_VF     = 0.2
TEMPERATURE = 1.0

BETA_LOW    = 0.1
BETA_HIGH   = 0.3
KL_TARGET   = 0.01
# ────────────────────────────────────────────────────────── #
UPDATE_EVERY = 10
CURRENT_TAG = 0.0

def feats_to_torch(feats_json: dict, device: torch.device):
    CARD_KEYS   = {"hand","played","cooldown","draw","tavern"}
    AGENT_KEYS  = {"agents_self","agents_enemy"}
    out = {}
    for k, arr in feats_json.items():
        if k == "phase":
            out[k] = torch.as_tensor(arr, dtype=torch.long,  device=device)
        else:
            if len(arr) == 0:
                if k in CARD_KEYS:
                    out[k] = torch.empty((0, 65), dtype=torch.float32, device=device)
                elif k in AGENT_KEYS:
                    out[k] = torch.empty((0, 67), dtype=torch.float32, device=device)
                else:
                    out[k] = torch.as_tensor(arr, dtype=torch.float32, device=device)
            else:
                out[k] = torch.as_tensor(arr, dtype=torch.float32, device=device)
    return out

def move_from_dict(d: dict) -> BasicMove:
    cmd = MoveEnum(d["type"])
    move_id = random.randint(0, 10000)
    if cmd == MoveEnum.MAKE_CHOICE:
        if "cuid" in d:
            return MakeChoiceMoveUniqueCard(
                move_id,
                command=cmd,
                cardsUniqueIds=d["cuid"]
            )
        elif "eff" in d:
            return MakeChoiceMoveUniqueEffect(
                move_id,
                command=cmd,
                effects=d["eff"]
            )
        else:
            raise ValueError("Unknown MAKE_CHOICE format")

    elif cmd in (MoveEnum.PLAY_CARD, MoveEnum.ACTIVATE_AGENT, MoveEnum.ATTACK, MoveEnum.BUY_CARD):
        return SimpleCardMove(
            move_id,
            command=cmd,
            cardUniqueId=d["uid"]
        )

    elif cmd == MoveEnum.CALL_PATRON:
        return SimplePatronMove(
            move_id,
            command=cmd,
            patronId=PatronId(d["patron"])
        )

    elif cmd == MoveEnum.END_TURN:
        return BasicMove(move_id, command=cmd)

    else:
        raise ValueError(f"Unknown move type: {cmd}")

def load_batch(replay_dir: str, want: int = BATCH):
    files = glob.glob(os.path.join(replay_dir, "*.jsonl"))
    random.shuffle(files)
    steps = []
    for fp in files:
        with open(fp) as f:
            for line in f:
                s = json.loads(line)
                if s.get("tag") == CURRENT_TAG:
                    steps.append(s)
        os.remove(fp)
        if len(steps) >= want:
            break
    return steps

def collate(steps):
    feat_jsons = [s["feats"] for s in steps]
    move_lists = [s["moves"] for s in steps]
    old_lp   = torch.tensor([s["old_logp"] for s in steps], dtype=torch.float32)
    old_val    = torch.tensor([s.get("old_value", 0.0) for s in steps], dtype=torch.float32)
    a_idx    = torch.tensor([s["action_idx"] for s in steps], dtype=torch.int64)
    R        = torch.tensor([s["reward"]   for s in steps], dtype=torch.float32)

    return feat_jsons, move_lists, old_lp, old_val, a_idx, R

def main():
    global CURRENT_TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",    default="weights.pt")
    ap.add_argument("--replay-dir", default="replay")
    args = ap.parse_args()
    if os.path.isfile(args.weights):
        CURRENT_TAG = os.path.getmtime(args.weights)

    os.makedirs(args.replay_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class LearnerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone     = VeiNet()
            self.move_encoder = MoveEncoder(mode='stub')

        def forward(self, feats_batch, move_vecs, mask):
            B, K, D = move_vecs.shape

            logits_batch = []
            values_batch = []
            for b in range(B):
                feats = feats_batch[b]
                mv_vecs = move_vecs[b]
                logit_b, v_b = self.backbone.forward_state(feats, mv_vecs)
                logit_b = logit_b.masked_fill(~mask[b], float('-inf'))
                logits_batch.append(logit_b)
                values_batch.append(v_b)

            logits = torch.stack(logits_batch, dim=0)  # (B, K)
            values = torch.stack(values_batch, dim=0)  # (B,)

            return logits, values

    net = LearnerNet().to(device)

    opt = torch.optim.Adam(
        [
            {"params": net.backbone.parameters(), "lr": LR},
            {"params": net.move_encoder.parameters(), "lr": LR * 5},
        ],
        betas=(0.9, 0.999), eps=1e-8
    )

    if not os.path.isfile(args.weights):
        torch.save(net.state_dict(), args.weights)
        print("[Learner] wrote initial random weights")

    else:
        ckpt = torch.load(args.weights, map_location=device)
        model_sd = net.state_dict()

        compatible = {k: v for k, v in ckpt.items()
                    if k in model_sd and v.shape == model_sd[k].shape}

        print(f"Loading {len(compatible)}/{len(model_sd)} tensors from checkpoint")
        model_sd.update(compatible)
        net.load_state_dict(model_sd)
        print("[Learner] loaded existing weights")

    step_cnt = 0
    pbar = tqdm(desc="Training steps", unit="step")
    buffer = []
    kl_beta  = BETA_LOW
    while True:
        buffer.extend(load_batch(args.replay_dir, want=BATCH))
        if len(buffer) < BATCH:
            time.sleep(1)
            continue
        steps   = buffer[:BATCH]
        buffer  = buffer[BATCH:]
        feat_jsons, move_lists, old_lp, old_val, a_idx, R = collate(steps)
        feats_batch = [feats_to_torch(f, device) for f in feat_jsons]
        old_lp, old_val, a_idx, R = [t.to(device) for t in (old_lp, old_val, a_idx, R)]

        kl_sum, kl_count = 0.0, 0
        ent_sum, ent_count = 0.0, 0

        for _ in range(EPOCHS):
            perm = torch.randperm(BATCH)
            opt.zero_grad()
            for i in range(0, BATCH, MINI):
                idx = perm[i:i+MINI]

                feats_mb = [feats_batch[j] for j in idx.tolist()]
                move_mb     = [move_lists[j] for j in idx.tolist()]
                old_lp_mb, old_val_mb = old_lp[idx], old_val[idx]
                ret_mb       = R[idx]
                act       = a_idx[idx]
                act = act.view(-1, 1).long()

                mv_s, mask_s = net.move_encoder.forward_batch(move_mb)
                mv_s, mask_s = mv_s.to(device), mask_s.to(device)

                logits, V = net(feats_mb, mv_s, mask_s)
                logits = logits / TEMPERATURE
                logits = torch.clamp(logits, -20.0, 20.0)

                log_probs = F.log_softmax(logits.masked_fill(~mask_s, -1e9), dim=-1)
                probs     = log_probs.exp()
                lp_new = log_probs.gather(1, act).squeeze(1)

                adv   = ret_mb - V.detach()
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                
                ratio = (lp_new - old_lp_mb).exp()
                clipped = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv
                loss_pi = -(torch.min(ratio * adv, clipped)).mean()

                V_clip    = old_val_mb + torch.clamp(V - old_val_mb, -CLIP_VF, CLIP_VF)
                loss_v1   = (V - ret_mb).pow(2)
                loss_v2   = (V_clip - ret_mb).pow(2)
                loss_v    = 0.5 * torch.max(loss_v1, loss_v2).mean()

                entropy   = -(probs * log_probs).masked_select(mask_s).sum() / mask_s.sum()

                approx_kl = (old_lp_mb - lp_new).mean().detach()

                loss    = 1.0*loss_pi + 1.0*loss_v - 0.02*entropy + kl_beta*approx_kl
                loss.backward()

                ent_sum   += entropy.item()   * idx.size(0)
                ent_count += idx.size(0)
                kl_sum    += approx_kl.item() * idx.size(0)
                kl_count  += idx.size(0)
                
                if ((i // MINI) + 1) * MINI >= BATCH:
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()

        avg_kl  = kl_sum  / kl_count
        avg_ent = ent_sum / ent_count
        kl_beta = float(min(BETA_HIGH, max(BETA_LOW, avg_kl / KL_TARGET)))

        step_cnt += 1
        pbar.update(1)
        pbar.set_postfix({"step": step_cnt})

        if step_cnt % UPDATE_EVERY == 0:
            torch.save(net.state_dict(), args.weights)
            print(
                f"[Learner] step {step_cnt} | R̄={R.float().mean():+.3f} | val head: {V.float().mean():.3f} | V.std={V.std():.3f} | "
                f"KL≈{avg_kl:.4f} | H={avg_ent:.4f} | Loss={loss.item():.4f}| saved."
            )
            CURRENT_TAG = os.path.getmtime(args.weights)

if __name__ == "__main__":
    main()
