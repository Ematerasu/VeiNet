import pathlib
import random
from typing import List
import numpy as np
from scripts_of_tribute.base_ai import BaseAI
from scripts_of_tribute.enums import MoveEnum, PatronId
from scripts_of_tribute.move import BasicMove
import torch
import sys, traceback
from collections import Counter

from models.VeiNet import VeiNet
from models.card_registry import CardRegistry
from models.move_encoder import MoveEncoder
from models.state_encoder import StateEncoder

class Vei(BaseAI):

    def serialize_move(self, mv: BasicMove) -> dict:
        d = {"type": int(mv.command.value)}
        if hasattr(mv, "cardUniqueId"):
            d["uid"] = self.card_registry.uid2cid[mv.cardUniqueId]
        if hasattr(mv, "patronId"):
            d["patron"] = int(mv.patronId.value)
        if hasattr(mv, "cardsUniqueIds"):
            d["cuid"] = [self.card_registry.uid2cid[u] for u in mv.cardsUniqueIds]
        if hasattr(mv, "effects"):
            d["eff"]  = mv.effects
        return d
    
    _PATRON_PRIORITY = [
        PatronId.DUKE_OF_CROWS,
        PatronId.HLAALU,
        PatronId.PELIN,
        PatronId.ANSEI,
        PatronId.SAINT_ALESSIA,
        PatronId.RED_EAGLE,
        PatronId.RAJHIN,
        PatronId.ORGNUM,
        PatronId.PSIJIC,
    ]

    def __init__(
            self,
            bot_name: str,
            weights: str | None = None,
            traj_path: str | None = None,
            tag: float = 0.0
        ):
        super().__init__(bot_name)
        self.device = torch.device("cpu")
        self.card_registry = CardRegistry()
        self.weights = weights
        self._steps: list[dict] = []
        self._traj_path = traj_path
        self.player_id = None
        self._tag = tag
        self.crashed = False

    def pregame_prepare(self):
        self._steps.clear()
        if hasattr(self, "net"):
            return
        try:
            self.net = VeiNet().to(self.device)
            self.encoder = StateEncoder(self.device)
            self.move_encoder = MoveEncoder(device=self.device).to(self.device)
            if self.weights and pathlib.Path(self.weights).is_file():
                raw = torch.load(self.weights, map_location=self.device)

                # 1) ——— MoveEncoder ———
                me_ckpt = {k[13:]: v for k, v in raw.items()
                        if k.startswith("move_encoder.")}
                self.move_encoder.load_state_dict(me_ckpt, strict=False)

                # 2) ——— VeiNet ———
                net_ckpt = {k.replace("backbone.", ""): v
                            for k, v in raw.items()
                            if not k.startswith("move_encoder.")}
                model_sd = self.net.state_dict()

                compatible = {k: v for k, v in net_ckpt.items()
                            if k in model_sd and v.shape == model_sd[k].shape}

                print(f"[Load] VeiNet tensors   ok:{len(compatible)} / {len(model_sd)}")
                model_sd.update(compatible)
                self.net.load_state_dict(model_sd)
            self.net.eval()
        except Exception as e:
            print("Exception in Vei.pregame_prepare():", e, file=sys.stderr, flush=True)
            self.crashed = True
            traceback.print_exc()
            return


    def select_patron(self, available_patrons):
        for pid in self._PATRON_PRIORITY:
            if pid in available_patrons:
                return pid
        return random.choice(available_patrons)
    
    def play(self, game_state, possible_moves, remaining_time):
        if self.crashed:
            return possible_moves[-1]
        if self.player_id is None:
            self.player_id = game_state.current_player.player_id
        try:
            feats = self.encoder(game_state)
            with torch.no_grad():
                state_vec, _ = self.net.forward_state(feats)         # (256,)
                move_vecs = self.move_encoder(possible_moves)        # (K,256)
                logits = (state_vec.unsqueeze(0) * move_vecs).sum(-1) # dot
            try:
                end_idx = next(i for i, m in enumerate(possible_moves) if m.command == MoveEnum.END_TURN)
                if end_idx < logits.size(0):
                    logits[end_idx] -= 1.0
            except (StopIteration, IndexError):
                pass
            probs = logits.softmax(0).cpu().numpy()
            idx   = np.random.choice(len(possible_moves), p=probs)
            mv    = possible_moves[idx]
            ended_early   = (mv.command == MoveEnum.END_TURN
                                and any(m.command != MoveEnum.END_TURN
                                        for m in possible_moves))
            proactive_move   = mv.command in (MoveEnum.PLAY_CARD,
                                            MoveEnum.ACTIVATE_AGENT,
                                            MoveEnum.BUY_CARD,
                                            MoveEnum.CALL_PATRON)
            feats_json = {k: self.t2l(v) for k, v in feats.items()}
            self._steps.append({
                "feats": feats_json,
                "moves": [self.serialize_move(m) for m in possible_moves],
                "old_logp": logits.log_softmax(0)[idx].item(),
                "action_idx": idx,
                "stats": {"ended_early": ended_early, "proactive_move":  proactive_move},
                "tag": self._tag,
            })
            return mv
        except Exception as e:
            print("Exception in Vei.play():", e, file=sys.stderr, flush=True)
            self.crashed = True
            traceback.print_exc()
            return possible_moves[-1]

        
    def game_end(self, end_game_state, final_state):
        gamma = 0.95
        me = final_state.current_player if final_state.current_player.player_id == self.player_id else final_state.enemy_player
        opp = final_state.enemy_player if final_state.current_player.player_id == self.player_id else final_state.current_player
        prest_me = me.prestige
        prest_op = opp.prestige
        Δprest   = (prest_me - prest_op) / 40.0
        patron_bonus = 0.2 if end_game_state.reason.startswith("PATRON") else 0.0
        player_name = str(self.player_id)
        if "." in player_name:
            player_name = player_name.split(".")[-1]

        R_terminal = 1 if end_game_state.winner == player_name else -1
        R_terminal += Δprest + patron_bonus
        TURN_PEN   = -0.02
        ACTIVITY_BONUS = +0.002
        ALIVE_BONUS = +0.005

        G = R_terminal
        for step in reversed(self._steps):
            st = step["stats"]
            extra = ALIVE_BONUS
            if st["ended_early"]: extra += TURN_PEN
            if st["proactive_move"]: extra += ACTIVITY_BONUS
            G = G * gamma + extra

            step["reward"] = G
            step["done"]   = True
            del step["stats"]
        if self._traj_path and not self.crashed:
            import json, os
            with open(self._traj_path, "a", encoding="utf-8") as f:
                for step in self._steps:
                    f.write(json.dumps(step) + "\n")

    def t2l(self, t):
        return t.cpu().tolist() if torch.is_tensor(t) else t
