import csv
import os
import pathlib
import random
from typing import Dict
import numpy as np
from scripts_of_tribute.base_ai import BaseAI
from scripts_of_tribute.enums import MoveEnum, PatronId
from scripts_of_tribute.move import (
    BasicMove,
    SimpleCardMove,
    SimplePatronMove,
    MakeChoiceMoveUniqueCard,
    MakeChoiceMoveUniqueEffect
)

from scripts_of_tribute.board import GameState, CurrentPlayer, EnemyPlayer
import torch
import sys, traceback, time


from models.VeiNet import SimpleVeiNet, VeiNet
from models.card_registry import CardRegistry
from models.move_encoder import MoveEncoder
from models.state_encoder import StateEncoder

class Vei(BaseAI):

    def serialize_move(self, mv: BasicMove) -> dict:
        d = {"type": int(mv.command.value)}
        if hasattr(mv, "cardUniqueId"):
            d["cid"] = self.card_registry.uid2cid[mv.cardUniqueId]
        if hasattr(mv, "patronId"):
            d["patron"] = int(mv.patronId.value)
        if hasattr(mv, "cardsUniqueIds"):
            d["cids"] = [self.card_registry.uid2cid[u] for u in mv.cardsUniqueIds]
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

                # print(f"[Load] VeiNet tensors   ok:{len(compatible)} / {len(model_sd)}")
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
    
    def play(self, game_state: GameState, possible_moves, remaining_time):
        if self.crashed:
            return possible_moves[-1]
        if self.player_id is None:
            self.player_id = game_state.current_player.player_id
        try:
            feats = self.encoder(game_state)
            with torch.no_grad():
                logits, V = self.compute_forward(game_state, feats, possible_moves)

            probs = logits.softmax(0).cpu().numpy()
            idx   = np.random.choice(len(possible_moves), p=probs)
            mv    = possible_moves[idx]

            ended_early = (mv.command == MoveEnum.END_TURN and any(
                m.command in [MoveEnum.PLAY_CARD, MoveEnum.ACTIVATE_AGENT] for m in possible_moves))
            proactive_move = mv.command in (
                MoveEnum.PLAY_CARD, MoveEnum.ACTIVATE_AGENT,
                MoveEnum.BUY_CARD, MoveEnum.CALL_PATRON)

            feats_json = {k: self.t2l(v) for k, v in feats.items()}
            self._steps.append({
                "feats": feats_json,
                "moves": [self.serialize_move(m) for m in possible_moves],
                "old_logp": logits.log_softmax(0)[idx].item(),
                "old_value": V.detach().cpu().item(),
                "action_idx": idx,
                "stats": {
                    "ended_early": ended_early,
                    "proactive_move": proactive_move,
                    "good_moves": sum(m.command in (
                        MoveEnum.PLAY_CARD, MoveEnum.ACTIVATE_AGENT,
                        MoveEnum.BUY_CARD, MoveEnum.CALL_PATRON) for m in possible_moves),
                },
                "tag": self._tag,
            })
            return mv

        except Exception as e:
            print("Exception in Vei.play():", e, file=sys.stderr, flush=True)
            self.crashed = True
            traceback.print_exc()
            return possible_moves[-1]
        
    def game_end(self, end_game_state, final_state):
        gamma = 1.0
        me = final_state.current_player if final_state.current_player.player_id == self.player_id else final_state.enemy_player
        opp = final_state.enemy_player if final_state.current_player.player_id == self.player_id else final_state.current_player
        if isinstance(me, CurrentPlayer):
            hand_and_draw = list(me.hand) + list(me.draw_pile)
        else:
            hand_and_draw = list(me.hand_and_draw)

        all_cards = (
            list(me.played)
            + list(me.cooldown_pile)
            + [agent.representing_card for agent in me.agents]
            + hand_and_draw
        )

        total = len(all_cards)
        deck_bonus = 0.0
        if total > 0:
            counts: Dict[int, int] = {}
            for c in all_cards:
                counts[c.deck.value] = counts.get(c.deck.value, 0) + 1

            for deck_idx in (0, 1, 6, 7):
                if counts.get(deck_idx, 0) / total >= 0.4:
                    deck_bonus = 0.3
                    break

        prest_me = me.prestige
        prest_op = opp.prestige
        Δprest   = (prest_me - prest_op) / 40.0

        player_name = str(self.player_id)
        if "." in player_name:
            player_name = player_name.split(".")[-1]

        # ───── Terminal reward ───── #
        R_terminal = 1.0 if end_game_state.winner == player_name else -1.0
        patron_bonus = 0.3 if end_game_state.reason.startswith("PATRON") else 0.0
        if R_terminal < 0: patron_bonus = -patron_bonus
        R_terminal += Δprest + patron_bonus + deck_bonus

        # ───── Hyperparameters shaping ───── #
        TURN_PEN = -0.01
        ACTIVITY_BONUS = +0.004
        COMBO_COIN_BONUS = 0.05
        COMBO_POWER_BONUS = 0.05
        COMBO_PREST_BONUS = 0.05
        CARD_WWR_BONUS = 0.2  # scaled by (wwr - 0.5)

        G = R_terminal

        for i in reversed(range(len(self._steps))):
            step = self._steps[i]
            st = step["stats"]
            extra = 0.0

            # ───── Proactivity bonus / turn penalty ───── #
            if st["proactive_move"]:
                extra += ACTIVITY_BONUS
            good_moves_t = st["good_moves"]
            if st["ended_early"] and good_moves_t >= 2:
                extra += (TURN_PEN * good_moves_t)

            # ───── Combo detection ───── #
            if i < len(self._steps) - 1:
                cur = self._steps[i]["feats"]["scalars"]
                nxt = self._steps[i+1]["feats"]["scalars"]
                Δcoins = nxt[0] - cur[0]
                Δprest = nxt[3] - cur[3]
                Δpower = nxt[6] - cur[6]

                if Δcoins >= (3 / 20):  extra += COMBO_COIN_BONUS
                if Δpower >= (3 / 30):  extra += COMBO_POWER_BONUS
                if Δprest >= (2 / 80):  extra += COMBO_PREST_BONUS

            # ───── Card reward (based on WWR) ───── #
            move = step["moves"][step["action_idx"]]
            if move['type'] in [0, 1, 3]:
                cid = self.card_registry.uid2cid.get(move['cid'])
                if cid is not None:
                    wwr = self.card_registry.weighted_win_rate.get(cid, 0.5)
                    extra += (wwr - 0.5) * CARD_WWR_BONUS

            G = G * gamma + extra
            step["reward"] = G
            step["done"] = True
            del step["stats"]

        if self._traj_path and not self.crashed:
            import json
            with open(self._traj_path, "a", encoding="utf-8") as f:
                for step in self._steps:
                    f.write(json.dumps(step) + "\n")

    def t2l(self, t):
        return t.cpu().tolist() if torch.is_tensor(t) else t
    
    def compute_forward(self, game_state, feats, possible_moves):
        move_vecs = self.move_encoder(possible_moves)
        logits, V = self.net.forward_state(feats, move_vecs)
        bias = torch.zeros_like(logits)
        for i, m in enumerate(possible_moves):
            if m.command in [MoveEnum.PLAY_CARD, MoveEnum.BUY_CARD, MoveEnum.ACTIVATE_AGENT]:
                cid = self.card_registry.uid2cid.get(m.cardUniqueId)
                if cid is not None:
                    wwr = self.card_registry.weighted_win_rate.get(cid, 0.5)
                    bias[i] += (wwr - 0.5) * 0.9
        logits = logits + bias

        try:
            good_moves = sum(1 for m in possible_moves if m.command in [
                MoveEnum.PLAY_CARD, MoveEnum.ACTIVATE_AGENT])
            end_idx = next(i for i, m in enumerate(possible_moves) if m.command == MoveEnum.END_TURN)
            if end_idx < logits.size(0) and good_moves > 0:
                logits[end_idx] -= 0.1 * good_moves
        except (StopIteration, IndexError):
            pass

        return logits, V

    def choose_via_value(self, game_state, possible_moves, logits):
        topk_idx = logits.topk(min(3, len(possible_moves))).indices.tolist()
        best_val = -float('inf')
        best_move = None

        for idx in topk_idx:
            move = possible_moves[idx]
            try:
                next_state, _ = game_state.apply_move(move)
                feats_next = self.encoder(next_state)
                _, val = self.net.forward_state(feats_next)
                if val.item() > best_val:
                    best_val = val.item()
                    best_move = move
            except:
                continue

        return best_move