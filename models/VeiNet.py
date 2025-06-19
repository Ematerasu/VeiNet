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
        return self.ln(out.squeeze(0).squeeze(0))  


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
        self.deck_pct_enc = nn.Linear(10, d_model)

        self.token_norm = nn.LayerNorm(d_model)
        self.post_ln = nn.LayerNorm(d_model)

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
        self.value_head  = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_state(self, feats, move_embeds):
        def card_pool(key, pool):
            x = feats[key]
            return pool(self.card_proj(x)).unsqueeze(0)

        def agent_pool(key, pool):
            x = feats[key]
            return pool(self.agent_proj(x)).unsqueeze(0)

        token_list = [
            self.token_norm(card_pool("hand",     self.hand_pool)),
            self.token_norm(card_pool("played",   self.played_pool)),
            self.token_norm(card_pool("cooldown", self.coold_pool)),
            self.token_norm(card_pool("draw",     self.draw_pool)),
            self.token_norm(card_pool("tavern",   self.tav_pool)),
            self.token_norm(agent_pool("agents_self",  self.selfa_pool)),
            self.token_norm(agent_pool("agents_enemy", self.enem_pool)),
            self.token_norm(self.patron_enc(feats["patrons"]).unsqueeze(0)),
            self.token_norm(self.scalar_enc(feats["scalars"]).unsqueeze(0)),
            self.token_norm(self.phase_emb(feats["phase"])),      # (1, d_model)
            self.token_norm(self.deck_pct_enc(feats["deck_pct"].unsqueeze(0))),
        ]
        tokens = torch.stack(token_list, dim=1)    # (1, 11, d_model)
        h = self.trans_enc(tokens).mean(1).squeeze(0)  # (d_model,)

        trunk_out = self.post_ln(self.post_proj(h))     # (d_model,)
        value     = self.value_head(trunk_out).squeeze()  # scalar

        # move_embeds: (M, d_model)
        logits = (trunk_out.unsqueeze(0) * move_embeds).sum(-1)  # (M,)

        return logits, value


class SimpleVeiNet(nn.Module):
    def __init__(self, d_model=256, num_scalars=11, num_patrons=10):
        super().__init__()
        in_dim = (
            65 + 65 + 65 + 65 + 65   # hand, played, cooldown, draw, tavern  (średnia embeddingów)
            + 67 + 67                 # agents_self, agents_enemy           (średnia agentów)
            + num_patrons            # patron one‐hot
            + num_scalars            # scalars
            + 1                      # phase (skalarnie)
            + num_patrons            # deck_pct
        )
        self.state_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        # policy/value heads
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)   # logits per move
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )
    
    def forward_state(self, feats, move_embeds):
        pools = []
        for key in ("hand","played","cooldown","draw","tavern"):
            x = feats[key]            # (N,65)
            pools.append(x.mean(0) if x.numel() else torch.zeros(65,device=x.device))
        for key in ("agents_self","agents_enemy"):
            x = feats[key]            # (N,67)
            pools.append(x.mean(0) if x.numel() else torch.zeros(67,device=x.device))

        # 2. scalars + patrons + phase + deck_pct
        pools.append(feats["patrons"].float())   # (10,)
        pools.append(feats["scalars"])           # (11,)
        pools.append(feats["phase"].float())     # (1,)
        pools.append(feats["deck_pct"])          # (10,)

        state_vec = torch.cat(pools, dim=-1)     # (in_dim,)
        H = self.state_proj(state_vec)           # (d_model,)

        # 3. policy:
        #    move_embeds: (M, d_model)
        logits = (H.unsqueeze(0) * move_embeds).sum(-1)   # (M,)

        # 4. value
        V = self.value_head(H)                          # (1,)
        return logits, V.squeeze()
