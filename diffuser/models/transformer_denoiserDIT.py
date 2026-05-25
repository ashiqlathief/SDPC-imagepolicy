import torch
import torch.nn as nn
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


def modulate(x, shift, scale):

    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):

    def __init__(self, d_model, n_heads, cond_dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        # AdaLN norms — elementwise_affine=False because AdaLN
        # learns its own scale/shift from the conditioning vector c
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention: action tokens attend to image latent z
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
            kdim=cond_dim, vdim=cond_dim,
        )

        mlp_dim = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

        # AdaLN modulation MLP:
        # takes combined (time + image_z) vector c -> 9 vectors of size d_model
        # 9 = shift_attn, scale_attn, gate_attn,
        #     shift_cross, scale_cross, gate_cross,
        #     shift_ffn,   scale_ffn,   gate_ffn
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 9 * d_model),
        )

        # Zero-init so blocks start as identity at step 0 (stable training)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c, context):

        # Unpack 9 modulation vectors from c
        (shift_attn, scale_attn, gate_attn,
         shift_cross, scale_cross, gate_cross,
         shift_ffn, scale_ffn, gate_ffn) = self.adaLN_modulation(c).chunk(9, dim=-1)

        # 1) Self-attention with AdaLN
        h = modulate(self.norm1(x), shift_attn, scale_attn)
        attn_out, _ = self.self_attn(h, h, h)
        x = x + gate_attn.unsqueeze(1) * attn_out

        # 2) Cross-attention to image z with AdaLN
        h = modulate(self.norm2(x), shift_cross, scale_cross)
        cross_out, _ = self.cross_attn(h, context, context)
        x = x + gate_cross.unsqueeze(1) * cross_out

        # 3) FFN with AdaLN
        h = modulate(self.norm3(x), shift_ffn, scale_ffn)
        x = x + gate_ffn.unsqueeze(1) * self.ffn(h)

        return x


class Transformer1DDenoisingModel(nn.Module):
    def __init__(
        self,
        horizon: int,
        action_dim: int,
        image_cond_dim: int = 256,
        d_model: int = 256,
        n_heads: int = 8,
        depth: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        returns_condition: bool = False,
        condition_dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.image_cond_dim = image_cond_dim
        self.d_model = d_model
        self.returns_condition = returns_condition

        # Project noisy actions into transformer dimension
        self.action_proj = nn.Linear(action_dim, d_model)

        # Learnable horizon positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, horizon, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Time embedding: diffusion timestep -> d_model
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

        # Image z projection: image_cond_dim -> d_model
        # So time and image_z live in the same space before combining
        self.image_proj = nn.Sequential(
            nn.Linear(image_cond_dim, d_model),
            nn.SiLU(),
        )

        # DiT blocks — each receives combined c = time_emb + image_z
        self.blocks = nn.ModuleList([
            DiTBlock(
                d_model=d_model,
                n_heads=n_heads,
                cond_dim=image_cond_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        # Final AdaLN norm before output projection
        self.norm_out = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

        # Final modulation for output norm — also conditioned on c
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),  # shift + scale for norm_out
        )
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)

        # Project back to action space
        self.action_out = nn.Linear(d_model, action_dim)

        # Zero-init output projection — stable early training (from DiT)
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

    def forward(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False):

        if not isinstance(cond, dict) or "obs_rgb" not in cond:
            raise ValueError("cond must be dict with key 'obs_rgb'")

        B, H, _ = x.shape

        # ── 1. Encode image ──────────────────────────────────────────
        # self.encoder is attached by ImageCondTransformer1DModel
        image_z = self.encoder(cond["obs_rgb"])          # (B, image_cond_dim)

        # image_z as cross-attention context token
        context = image_z.unsqueeze(1)                   # (B, 1, image_cond_dim)

        # ── 2. Build combined conditioning vector c ───────────────────
        # c carries BOTH what noise level we are at (time)
        # AND what the scene looks like (image_z)
        # Every DiTBlock uses c to modulate its LayerNorms
        t_emb = self.time_mlp(time)                      # (B, d_model)
        z_emb = self.image_proj(image_z)                 # (B, d_model)
        c = t_emb + z_emb                                # (B, d_model)

        # ── 3. Tokenize actions ───────────────────────────────────────
        tok = self.action_proj(x)                        # (B, H, d_model)
        tok = tok + self.pos_embed                       # add positional info

        # ── 4. DiT blocks ─────────────────────────────────────────────
        # Each block: self-attn + cross-attn to image + FFN
        # All three sub-layers are modulated by c via AdaLN
        for block in self.blocks:
            tok = block(tok, c, context)

        # ── 5. Output ─────────────────────────────────────────────────
        # Final AdaLN norm also conditioned on c
        shift, scale = self.final_modulation(c).chunk(2, dim=-1)
        tok = modulate(self.norm_out(tok), shift, scale) # (B, H, d_model)

        noise_pred = self.action_out(tok)                # (B, H, action_dim)
        return noise_pred