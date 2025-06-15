import math
import torch, torch.nn as nn, torch.nn.functional as F

class SetPool(nn.Module):
    def __init__(self, d_model=256, n_head=4):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln   = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        if x.numel() == 0:
            return self.seed.new_zeros(self.seed.size(-1))

        q = self.seed
        k = v = x.unsqueeze(0)
        out, _ = self.attn(q, k, v)
        return self.ln(out.mean(0).squeeze())  


class VeiNet(nn.Module):
    def __init__(self, d_model=256, num_scalars=11):
        super().__init__()
        self.card_proj  = nn.Linear(65, d_model)
        self.agent_proj = nn.Linear(67, d_model)

        self.hand_pool   = SetPool(d_model)
        self.played_pool = SetPool(d_model)
        self.coold_pool  = SetPool(d_model)
        self.draw_pool   = SetPool(d_model)
        self.tav_pool    = SetPool(d_model)
        self.selfa_pool  = SetPool(d_model)
        self.enem_pool   = SetPool(d_model)

        self.scalar_enc = nn.Linear(num_scalars, d_model)
        self.patron_enc = nn.Linear(10, d_model)
        self.phase_emb  = nn.Embedding(4, d_model)

        self.pre_trunk = nn.Sequential(
            nn.Linear(10 * d_model, d_model),
            nn.ReLU(),
        )
        self.trans_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=4*d_model,
                activation="gelu",
                batch_first=True),
            num_layers=2
        )
        self.post_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        self.value_head  = nn.Linear(d_model, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ——————————————————————————————————————————————
    def forward_state(self, feats):
        def card_pool(key, pool):
            x = feats[key]
            return pool(self.card_proj(x))

        def agent_pool(key, pool):
            x = feats[key]
            return pool(self.agent_proj(x))

        h0  = torch.cat([
            card_pool("hand",     self.hand_pool),
            card_pool("played",   self.played_pool),
            card_pool("cooldown", self.coold_pool),
            card_pool("draw",     self.draw_pool),
            card_pool("tavern",   self.tav_pool),

            agent_pool("agents_self",  self.selfa_pool),
            agent_pool("agents_enemy", self.enem_pool),

            self.patron_enc(feats["patrons"]),
            self.scalar_enc(feats["scalars"]),
            self.phase_emb(feats["phase"]).squeeze(0)
        ], dim=-1)
        h1 = self.pre_trunk(h0)
        tokens = h1.view(1, -1, h1.size(-1))
        h2 = self.trans_enc(tokens)
        h2 = h2.mean(1)
        trunk_out = self.post_proj(h2.squeeze(0))
        value     = self.value_head(trunk_out)
        return trunk_out, value.squeeze()
