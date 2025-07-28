from __future__ import annotations

from enum import Enum
from typing import List, TypedDict

import numpy as np
import torch
from scripts_of_tribute.board import CurrentPlayer, GameState, UniqueCard, Choice
from scripts_of_tribute.enums import PatronId

from models.card_registry import CardRegistry


class StateTensors(TypedDict):
    hand:     torch.Tensor        # (Nh, 65)
    played:   torch.Tensor        # (Np, 65)
    cooldown: torch.Tensor        # (Nc, 65)
    draw:     torch.Tensor        # (Nd, 65)
    tavern:   torch.Tensor        # (Nt, 65)
    agents_self:  torch.Tensor    # (Na, 67)  – 65 + hp + activated
    agents_enemy: torch.Tensor    # (Ne, 67)
    scalars: torch.Tensor         # (4,)
    patrons: torch.Tensor         # (NUM_PATRONS,)
    phase:   torch.Tensor         # (1,)
    deck_pct: torch.Tensor        # (D,)
    choice_followup: torch.Tensor # (1,)

class ChoiceFollowUpEnum(Enum):
    NONE = 0

    ENACT_CHOSEN_EFFECT = 1
    REPLACE_CARDS_IN_TAVERN = 2
    DESTROY_CARDS = 3
    DISCARD_CARDS = 4
    REFRESH_CARDS = 5
    TOSS_CARDS = 6
    KNOCKOUT_AGENTS = 7
    ACQUIRE_CARDS = 8
    COMPLETE_HLAALU = 9
    COMPLETE_PELLIN = 10
    COMPLETE_PSIJIC = 11
    COMPLETE_TREASURY = 12
    DONATE = 13

CHOICE_FOLLOWUP_MAP = {e.name: e.value for e in ChoiceFollowUpEnum}

class StateEncoder:
    COINS_MAX      = 20
    PRESTIGE_MAX   = 80
    POWER_MAX      = 30
    CALLS_MAX      = 3

    def __init__(self, device: torch.device):
        self.reg = CardRegistry()
        self.device = device


    def __call__(self, gs: GameState) -> StateTensors:
        feats = {
            "hand":     self._cards_tensor(gs.current_player.hand),
            "played":   self._cards_tensor(gs.current_player.played),
            "cooldown": self._cards_tensor(gs.current_player.cooldown_pile),
            "draw":     self._cards_tensor(gs.current_player.draw_pile),
            "tavern":   self._cards_tensor(gs.tavern_available_cards),
            "agents_self":  self._agents_tensor(gs.current_player.agents),
            "agents_enemy": self._agents_tensor(gs.enemy_player.agents),
            "scalars": self._scalar_tensor(gs.current_player, gs.enemy_player),
            "patrons": self._patron_tensor(gs.patron_states,
                                            gs.current_player.player_id,
                                            gs.enemy_player.player_id),
            "phase":   torch.tensor([gs.board_state.value],
                                    dtype=torch.long, device=self.device),
        }

        feats["choice_followup"] = torch.tensor(
            [self._encode_choice_followup(gs.pending_choice)],
            dtype=torch.long,
            device=self.device
        )

        feats["deck_pct"] = self._deck_distribution(gs.current_player)

        return feats

    def _cards_tensor(self, card_list: List[UniqueCard]) -> torch.Tensor:
        if not card_list:
            return torch.empty((0, 65), dtype=torch.float32,
                                device=self.device)

        cid_list = []
        for c in card_list:
            cid = self.reg.cid_from_uid(c.unique_id, c.name)
            cid_list.append(cid)
            self.reg.uid2cid[c.unique_id] = cid
        vecs = self.reg.embedding[cid_list]           # (N, 65)  (numpy)
        return torch.as_tensor(vecs, dtype=torch.float32, device=self.device)

    def _agents_tensor(self, agent_list) -> torch.Tensor:
        if not agent_list:
            return torch.empty((0, 67), dtype=torch.float32,
                                device=self.device)

        rows: List[np.ndarray] = []
        for ag in agent_list:
            cid = self.reg.cid_from_uid(ag.representing_card.unique_id, ag.representing_card.name)
            self.reg.uid2cid[ag.representing_card.unique_id] = cid
            base = self.reg.embedding[cid]                   # (65,)
            hp_norm = np.asarray([ag.currentHP / ag.representing_card.hp], dtype=np.float32)
            ex_flag = np.asarray([float(ag.activated)], dtype=np.float32)
            rows.append(np.concatenate([base, hp_norm, ex_flag]))
        arr = np.stack(rows, axis=0)
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    def _scalar_tensor(self, me, opp) -> torch.Tensor:
        mc, oc = me.coins/self.COINS_MAX,      opp.coins/self.COINS_MAX
        mp, op = me.prestige/self.PRESTIGE_MAX,opp.prestige/self.PRESTIGE_MAX
        mw, ow = me.power/self.POWER_MAX,      opp.power/self.POWER_MAX
        mcl, ocl = me.patron_calls/self.CALLS_MAX, 1/self.CALLS_MAX # opp.patron_calls are always 1 when its our turn

        arr = np.asarray([
            mc, oc, mc-oc,
            mp, op, mp-op,
            mw, ow, mw-ow,
            mcl, ocl
        ], dtype=np.float32)
        return torch.from_numpy(arr).to(self.device)

    def _patron_tensor(self, patron_states, curr_player_id, opp_player_id) -> torch.Tensor:
        vals = torch.zeros(len(PatronId), dtype=torch.float32, device=self.device)

        for pid in PatronId:
            owner = patron_states.patrons.get(pid, None)
            if owner is None:
                continue
            if owner == curr_player_id:
                vals[pid.value] = 1.0
            elif owner == opp_player_id:
                vals[pid.value] = -1.0
        return vals

    def _deck_distribution(self, player: CurrentPlayer) -> torch.Tensor:
        groups = [
            *player.hand,
            *player.played,
            *player.cooldown_pile,
            *player.draw_pile,
            *[agent.representing_card for agent in player.agents]
        ]

        counts = {pid: 0 for pid in PatronId}
        total = 0

        for card in groups:
            counts[card.deck] += 1
            total += 1

        if total > 0:
            pct = [counts[pid] / total for pid in PatronId]
        else:
            pct = [0.0 for _ in PatronId]

        return torch.tensor(pct, dtype=torch.float32, device=self.device)

    def _encode_choice_followup(self, choice: Choice | None) -> int:
        if not choice:
            return CHOICE_FOLLOWUP_MAP["NONE"]
        return CHOICE_FOLLOWUP_MAP.get(choice.choice_follow_up, CHOICE_FOLLOWUP_MAP["NONE"])
