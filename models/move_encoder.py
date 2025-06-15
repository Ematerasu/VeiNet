from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import torch.nn as nn

from scripts_of_tribute.move import BasicMove, MakeChoiceMoveUniqueCard, MakeChoiceMoveUniqueEffect
from scripts_of_tribute.enums import PatronId, MoveEnum
from models.card_registry import CardRegistry


class MoveEncoder(nn.Module):

    MOVE_TYPES = [
        MoveEnum.PLAY_CARD,
        MoveEnum.ACTIVATE_AGENT,
        MoveEnum.ATTACK,
        MoveEnum.BUY_CARD,
        MoveEnum.CALL_PATRON,
        MoveEnum.MAKE_CHOICE,
        MoveEnum.END_TURN,
    ]
    EFFECT_NAMES = [
        "GAIN_COIN",
        "GAIN_POWER",
        "GAIN_PRESTIGE",
        "OPP_LOSE_PRESTIGE",
        "REPLACE_TAVERN",
        "ACQUIRE_TAVERN",
        "DESTROY_CARD",
        "DRAW",
        "OPP_DISCARD",
        "RETURN_TOP",
        "RETURN_AGENT_TOP",
        "TOSS",
        "KNOCKOUT",
        "PATRON_CALL",
        "CREATE_SUMMERSET_SACKING",
        "HEAL",
        "KNOCKOUT_ALL",
        "DONATE",
    ]
    EFFECT2ID: Dict[str, int] = {name: i for i, name in enumerate(EFFECT_NAMES)}
    NUM_EFFECTS = len(EFFECT_NAMES)  # = 18
    NUM_PATRONS = len(PatronId)
    MAX_EFFECT_AMOUNT = 10.0

    def __init__(self, d_model: int = 256, device: str | torch.device = "cpu", mode="live"):
        super().__init__()
        self.d_model = d_model
        self.device = torch.device(device)
        self.reg = CardRegistry()
        self.mode   = mode

        self.type_emb   = nn.Embedding(len(self.MOVE_TYPES), d_model)
        self.patron_emb = nn.Embedding(self.NUM_PATRONS, 10)
        self.effect_emb = nn.Embedding(self.NUM_EFFECTS, d_model)

        # ─── feed-forward:  in_dim = d_model + 65 + 10 + d_model + 1 ────
        in_dim = d_model + 65 + 10 + d_model + 1
        self.ff = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    # ------------------------------------------------------------------ #
    def forward(self, moves: List[object]) -> torch.Tensor:
        rows = [self._encode_one(mv) for mv in moves]
        return (
            torch.stack(rows, 0)
            if rows
            else torch.empty((0, self.d_model), device=self.device)
        )

    # ------------------------------------------------------------------ #
    def _encode_one(self, mv):
        if self.mode == "live":
            return self._encode_live(mv)   # BasicMove
        else:
            return self._encode_stub(mv)   # dict / MoveStub

    def _encode_live(self, mv: BasicMove) -> torch.Tensor:
        t_idx     = self.MOVE_TYPES.index(mv.command)
        vec_type  = self.type_emb.weight[t_idx]              # (d_model,)

        vec_card  = torch.zeros(65, device=self.device)      # card
        vec_pat   = torch.zeros(10, device=self.device)      # patron
        vec_choice= torch.zeros(self.d_model, device=self.device)
        flag_att  = torch.zeros(1, device=self.device)       # 1-bit

        if mv.command in (MoveEnum.PLAY_CARD,
                            MoveEnum.ACTIVATE_AGENT,
                            MoveEnum.BUY_CARD):
            cid = self.reg.uid2cid[mv.cardUniqueId]
            vec_card = torch.as_tensor(self.reg.embedding[cid], device=self.device)

        elif mv.command == MoveEnum.ATTACK:
            cid = self.reg.uid2cid[mv.cardUniqueId]
            vec_card = torch.as_tensor(self.reg.embedding[cid], device=self.device)
            flag_att[0] = 1.0               # attack-flag

        elif mv.command == MoveEnum.CALL_PATRON:
            pid_idx = list(PatronId).index(mv.patronId)
            vec_pat = self.patron_emb.weight[pid_idx]

        elif mv.command == MoveEnum.MAKE_CHOICE:
            if isinstance(mv, MakeChoiceMoveUniqueCard):
                if mv.cardsUniqueIds:
                    cid_list = [
                        self.reg.uid2cid[uid] for uid in mv.cardsUniqueIds
                    ]
                    emb = torch.as_tensor(
                        self.reg.embedding[cid_list], device=self.device
                    )  # (N,65)
                    vec_card = emb.mean(0)  # 65-d

            elif isinstance(mv, MakeChoiceMoveUniqueEffect) and mv.effects:
                eff_vecs = []
                for eff_str in mv.effects:
                    eff_id, amt = self._parse_effect(eff_str)
                    scale = 1 + amt / self.MAX_EFFECT_AMOUNT
                    eff_vecs.append(self.effect_emb.weight[eff_id] * scale)
                vec_choice = torch.stack(eff_vecs, 0).mean(0)


        concat = torch.cat([vec_type, vec_card, vec_pat, vec_choice, flag_att])
        return self.ff(concat)

    def _parse_effect(self, eff_str: str) -> Tuple[int, int]:
        """
        'GAIN_COIN 3'  →  (id, 3)
        'KNOCKOUT 1' → (id, 1)
        """
        type, amt = eff_str.split()
        return self.EFFECT2ID[type], int(amt)
    
    def _encode_stub(self, d: dict) -> torch.Tensor:
        t_idx    = d["type"]
        vec_type = self.type_emb.weight[t_idx]
        dev      = vec_type.device

        vec_card  = torch.zeros(65, device=dev)
        vec_pat   = torch.zeros(10, device=dev)
        vec_choice= torch.zeros(self.d_model, device=dev)
        flag_att  = torch.zeros(1, device=dev)

        if t_idx in (0,1,3) and "cid" in d:
            vec_card = torch.as_tensor(self.reg.embedding[d["cid"]],
                                    device=dev)
        elif t_idx == 2 and "cid" in d:
            vec_card = torch.as_tensor(self.reg.embedding[d["cid"]],
                                    device=dev)
            flag_att[0] = 1.0

        if t_idx == 4 and "patron" in d:
            vec_pat = self.patron_emb.weight[d["patron"]]

        if t_idx == 5 and "cids" in d:
            emb = torch.as_tensor(self.reg.embedding[d["cids"]],
                                device=dev)
            vec_card = emb.mean(0)

        if t_idx == 5 and "eff" in d:
            eff_vecs = []
            for eff in d["eff"]:
                eid, amt = self._parse_effect(eff)
                eff_vecs.append(self.effect_emb.weight[eid].to(dev) *
                                (1 + amt/self.MAX_EFFECT_AMOUNT))
            vec_choice = torch.stack(eff_vecs).mean(0)

        return self.ff(torch.cat([vec_type, vec_card,
                                vec_pat, vec_choice, flag_att]))

    def encode_stub_batch(self, move_batch: List[List[dict]]) -> Tuple[torch.Tensor, torch.Tensor]:
        self.device = next(self.parameters()).device
        B = len(move_batch)
        K = max(len(moves) for moves in move_batch)
        D = self.d_model

        flat_moves = []
        bk_index = []
        for b, moves in enumerate(move_batch):
            for k, d in enumerate(moves):
                flat_moves.append(d)
                bk_index.append((b, k))

        vec_type_list = []
        vec_card_list = []
        vec_pat_list = []
        vec_choice_list = []
        flag_att_list = []

        for d in flat_moves:
            t_idx = d["type"]
            vec_type = self.type_emb.weight[t_idx].to(self.device)
            vec_card = torch.zeros(65, device=self.device)
            vec_pat = torch.zeros(10, device=self.device)
            vec_choice = torch.zeros(self.d_model, device=self.device)
            flag_att = torch.zeros(1, device=self.device)

            if t_idx in (0, 1, 3) and "cid" in d:
                vec_card = torch.as_tensor(self.reg.embedding[d["cid"]], device=self.device)
            elif t_idx == 2 and "cid" in d:
                vec_card = torch.as_tensor(self.reg.embedding[d["cid"]], device=self.device)
                flag_att[0] = 1.0

            if t_idx == 4 and "patron" in d:
                vec_pat = self.patron_emb.weight[d["patron"]].to(self.device)

            if t_idx == 5 and "cids" in d:
                emb = torch.as_tensor(self.reg.embedding[d["cids"]], device=self.device)
                vec_card = emb.mean(0)

            if t_idx == 5 and "eff" in d:
                eff_vecs = []
                for eff in d["eff"]:
                    eid, amt = self._parse_effect(eff)
                    eff_vecs.append(self.effect_emb.weight[eid] * (1 + amt / self.MAX_EFFECT_AMOUNT))
                vec_choice = torch.stack(eff_vecs).mean(0)

            vec_type_list.append(vec_type)
            vec_card_list.append(vec_card)
            vec_pat_list.append(vec_pat)
            vec_choice_list.append(vec_choice)
            flag_att_list.append(flag_att)

        concat = torch.cat([
            torch.stack(vec_type_list),
            torch.stack(vec_card_list),
            torch.stack(vec_pat_list),
            torch.stack(vec_choice_list),
            torch.stack(flag_att_list)
        ], dim=1)

        out_vecs = self.ff(concat)  # (N, D)

        out = torch.zeros(B, K, D, device=self.device)
        mask = torch.zeros(B, K, dtype=torch.bool, device=self.device)
        for (b, k), vec in zip(bk_index, out_vecs):
            out[b, k] = vec
            mask[b, k] = True

        return out, mask