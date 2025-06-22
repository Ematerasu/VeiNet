from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from typing import Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_DATA = _ROOT / "data"

class CardRegistry:
    _instance_lock = threading.Lock()
    _instance: "CardRegistry | None" = None        # singleton (per process)

    def __new__(cls) -> "CardRegistry":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._bootstrap()
            return cls._instance

    name_to_cid: Dict[str, int]
    cid_to_name: List[str]
    weighted_win_rate: Dict[int, float]
    embedding: np.ndarray                         # shape (N, 65)  (64 + wwr)
    uid2cid: Dict[int, int]                       # filled during a game

    def cid_from_uid(self, uid: int, name: str | None = None) -> int:
        """
        Map *runtime* unique_id to persistent card_id.
        Provide `name` if you have it – avoids O(N) lookup.
        """
        if uid in self.uid2cid:
            return self.uid2cid[uid]

        if name is None:
            raise KeyError(f"uid {uid} unknown and no name provided")

        cid = self.name_to_cid[name]
        self.uid2cid[uid] = cid
        return cid

    def _bootstrap(self) -> None:
        self.uid2cid = {}

        with (_DATA / "cards.json").open("r", encoding="utf-8") as f:
            cards = json.load(f)

        self.name_to_cid = {c["Name"]: c["id"] for c in cards}
        max_cid = max(self.name_to_cid.values())
        self.cid_to_name = [""] * (max_cid + 1)
        for n, cid in self.name_to_cid.items():
            self.cid_to_name[cid] = n

        self.weighted_win_rate = {}
        wwr_path = _DATA / "card_weighted_winrate.csv"
        if wwr_path.exists():
            with wwr_path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    self.weighted_win_rate[int(row["card_id"])] = float(
                        row["weighted_win_rate"]
                    )

        emb64 = np.load(_DATA / "card_embeddings.npy")       # (N, 64)
        if emb64.shape[0] != max_cid + 1:
            raise ValueError(
                f"embeddings rows ({emb64.shape[0]}) "
                f"≠ max card_id+1 ({max_cid+1})"
            )

        wwr_vec = np.zeros((emb64.shape[0], 1), dtype=np.float32)
        for cid, val in self.weighted_win_rate.items():
            wwr_vec[cid, 0] = val
        self.embedding = np.concatenate([emb64, wwr_vec], axis=1)

