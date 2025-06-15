import argparse, glob, json, math, os, random, time, pathlib
import torch, torch.nn.functional as F
from torch import nn
import numpy as np
import gc
import itertools

from models.VeiNet import VeiNet
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
BATCH        = 256
EPOCHS       = 5
CLIP_EPS     = 0.1
LR           = 1e-6
UPDATE_EVERY = 100
KL_BETA = 0.5

FREEZE_K = 1000
RESYNC_K = 500
ME_LR_RESYNC = 5e-7

KL_TARGET   = 0.03
LR_MIN      = 5e-7
LR_MAX      = 2e-4
LR_GAIN_UP  = 1.15
LR_GAIN_DOWN= 0.7
# ────────────────────────────────────────────────────────── #

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
    a_idx    = torch.tensor([s["action_idx"] for s in steps], dtype=torch.int64)
    old_lp   = torch.tensor([s["old_logp"] for s in steps], dtype=torch.float32)
    R        = torch.tensor([s["reward"]   for s in steps], dtype=torch.float32)

    move_lists = [s["moves"] for s in steps]
    return feat_jsons, move_lists, old_lp, a_idx, R

def main():
    global CURRENT_TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",    default="weights.pt")
    ap.add_argument("--replay-dir", default="replay")
    args = ap.parse_args()

    CURRENT_TAG = os.path.getmtime(args.weights)

    os.makedirs(args.replay_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class LearnerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone     = VeiNet()
            self.move_encoder = MoveEncoder(mode='stub')
        def forward(self, feats_batch, move_vecs):
            trunk_vecs, values = [], []
            for f in feats_batch:
                h, v = self.backbone.forward_state(f)
                trunk_vecs.append(h)
                values.append(v)
            H = torch.stack(trunk_vecs)           # (B,256)
            V = torch.stack(values).squeeze(-1)   # (B,)
            logits = (H.unsqueeze(1) * move_vecs).sum(-1)
            return logits, V

    net = LearnerNet().to(device)
    for p in net.move_encoder.parameters():
        p.requires_grad = False
    
    phase_end = FREEZE_K
    mlp_params   = net.backbone.pre_trunk.parameters()
    head_params = itertools.chain(
        net.backbone.value_head.parameters()
    )
    me_params = net.move_encoder.parameters()
    for p in net.move_encoder.parameters():
        p.requires_grad = False
    for p in net.backbone.trans_enc.parameters():
        p.requires_grad = False
    for p in net.backbone.post_proj.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(
        [
            {"params": mlp_params,  "lr": LR},
            {"params": head_params, "lr": LR},
            {"params": me_params,   "lr": LR * 0.1},
            {"params": net.backbone.trans_enc.parameters(),  "lr": 0.0, "initial_lr": 5e-6},
            {"params": net.backbone.post_proj.parameters(),  "lr": 0.0, "initial_lr": 5e-6},
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
    buffer = []
    BETA_LOW  = 0.1
    BETA_HIGH = 0.3
    kl_beta = BETA_LOW
    while True:
        if step_cnt == 0:
            for p in net.backbone.trans_enc.parameters():
                p.requires_grad = False
            for p in net.backbone.post_proj.parameters():
                p.requires_grad = False

        if step_cnt == 500:
            for p in net.backbone.trans_enc.parameters():
                p.requires_grad = True
            for p in net.backbone.post_proj.parameters():
                p.requires_grad = True

            for g in opt.param_groups:
                if g.get("initial_lr") is not None:
                    g["lr"] = g["initial_lr"]

            print(f"[Scheduler] Transformer unfreezed @ step {step_cnt} (lr=5e-6)")

        buffer.extend(load_batch(args.replay_dir, want=BATCH))
        if len(buffer) < BATCH:
            print(f'Not enough for learn {len(buffer)}/{BATCH}', end='\r')
            time.sleep(1)
            continue
        steps   = buffer[:BATCH]
        buffer  = buffer[BATCH:]
        feat_jsons, move_lists, old_lp, a_idx, R = collate(steps)
        feats_batch = [feats_to_torch(f, device) for f in feat_jsons]
        old_lp, a_idx, R = [t.to(device) for t in (old_lp, a_idx, R)]

        kl_sum   = 0.0
        kl_count = 0
        ent_sum   = 0.0
        ent_count = 0
        MINI = 64
        GRAD_ACC = BATCH // MINI
        for _ in range(EPOCHS):
            perm = torch.randperm(BATCH)
            opt.zero_grad()
            for i in range(0, BATCH, MINI):
                idx = perm[i:i+MINI]

                feats_mb = [feats_batch[j] for j in idx.tolist()]
                lp_old    = old_lp[idx]
                ret       = R[idx]
                act       = a_idx[idx]
                act = act.view(-1, 1).long()

                mv_s, mask_s = net.move_encoder.encode_stub_batch([move_lists[j] for j in idx.tolist()])

                mv_s, mask_s = mv_s.to(device), mask_s.to(device)
                logits, V = net(feats_mb, mv_s)
                lp_new    = logits.log_softmax(-1).gather(1, act).squeeze(1)

                adv   = ret - V.detach()
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                #with torch.no_grad():
                    #print("adv mean:", adv.abs().mean().item())
                ratio = (lp_new - lp_old).exp()
                loss_pi = -(torch.min(ratio*adv, torch.clamp(ratio,1-CLIP_EPS,1+CLIP_EPS)*adv)).mean()
                loss_v  = 0.5*F.mse_loss(V, ret)
                ent     = -(logits.softmax(-1)*logits.log_softmax(-1)).masked_select(mask_s).mean()
                approx_kl = (lp_old - lp_new).mean().detach()

                loss    = 1.1*loss_pi + 0.3*loss_v - 0.005*ent + kl_beta * approx_kl

                ent_sum   += ent.item() * len(idx)
                ent_count += len(idx)

                kl_sum   += approx_kl.item() * len(idx)
                kl_count += len(idx)
                loss.backward()
                if (i // MINI + 1) % GRAD_ACC == 0:
                    nn.utils.clip_grad_norm_(mlp_params, 10.)
                    nn.utils.clip_grad_norm_(net.backbone.trans_enc.parameters(), 2.)
                    nn.utils.clip_grad_norm_(net.backbone.value_head.parameters(), 1.)
                    nn.utils.clip_grad_norm_(net.backbone.post_proj.parameters(), 2.)
                    opt.step()
                    opt.zero_grad()

        avg_kl = kl_sum / kl_count
        avg_ent = ent_sum / ent_count

        trunk_group = opt.param_groups[0]
        trunk_lr    = trunk_group["lr"]
        if avg_kl < 0.5 * KL_TARGET:
            trunk_lr = min(trunk_lr * LR_GAIN_UP, LR_MAX)
        elif avg_kl > 2.0 * KL_TARGET:
            trunk_lr = max(trunk_lr * LR_GAIN_DOWN, LR_MIN)
        trunk_group["lr"] = trunk_lr
        #print(trunk_group["lr"], loss.item())
        kl_beta = BETA_LOW if avg_kl < 0.01 else BETA_HIGH
        step_cnt += 1
        if step_cnt % UPDATE_EVERY == 0:
            torch.save(net.state_dict(), args.weights)
            print(f"[Learner] step {step_cnt} | R̄={R.float().mean():+.3f} | val head: {V.float().mean():.3f} | V.std={V.std():.3f} | KL≈{avg_kl:.4f} | H={avg_ent:.4f} | Loss={loss.item():.4f}| saved.")
            CURRENT_TAG = os.path.getmtime(args.weights)

        if step_cnt == phase_end:
            if any(p.requires_grad for p in net.move_encoder.parameters()):
                for p in net.move_encoder.parameters():
                    p.requires_grad = False
                phase_end += FREEZE_K
                print(f"[Scheduler] freeze ME @ {step_cnt}")
            else:
                for p in net.move_encoder.parameters():
                    p.requires_grad = True
                for g in opt.param_groups:
                    if set(g['params']) & set(me_params):
                        g['lr'] = ME_LR_RESYNC
                phase_end += RESYNC_K
                print(f"[Scheduler] resync ME @ {step_cnt}")
        del feats_batch, old_lp, a_idx, R
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
