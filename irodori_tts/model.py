from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.complex(torch.cos(freqs), torch.sin(freqs))


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # x: (B, S, H, Dh), Dh must be even.
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:3], -1, 2))
    x_ = x_ * freqs_cis[None, :, None, :]
    x_ = torch.view_as_real(x_).reshape_as(x)
    return x_.type_as(x)


def get_timestep_embedding(timestep: torch.Tensor, dim: int) -> torch.Tensor:
    assert dim % 2 == 0
    half = dim // 2
    freqs = 1000.0 * torch.exp(
        -torch.log(torch.tensor(10000.0, device=timestep.device, dtype=torch.float32))
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / half
    )
    args = timestep[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(timestep.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int | tuple[int, ...], eps: float = 1e-6):
        super().__init__()
        if isinstance(dim, int):
            dim = (dim,)
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight).to(x_dtype)


class LowRankAdaLN(nn.Module):
    """
    Echo-style low-rank AdaLN that returns both modulated activations and a residual gate.
    """

    def __init__(self, model_dim: int, rank: int, eps: float):
        super().__init__()
        rank = max(1, min(int(rank), int(model_dim)))
        self.eps = eps
        self.shift_down = nn.Linear(model_dim, rank, bias=False)
        self.scale_down = nn.Linear(model_dim, rank, bias=False)
        self.gate_down = nn.Linear(model_dim, rank, bias=False)
        self.shift_up = nn.Linear(rank, model_dim, bias=True)
        self.scale_up = nn.Linear(rank, model_dim, bias=True)
        self.gate_up = nn.Linear(rank, model_dim, bias=True)
        # Match Echo/JAX AdaLN behavior: zero-init output projections.
        nn.init.zeros_(self.shift_up.weight)
        nn.init.zeros_(self.scale_up.weight)
        nn.init.zeros_(self.gate_up.weight)
        if self.shift_up.bias is not None:
            nn.init.zeros_(self.shift_up.bias)
        if self.scale_up.bias is not None:
            nn.init.zeros_(self.scale_up.bias)
        if self.gate_up.bias is not None:
            nn.init.zeros_(self.gate_up.bias)

    def forward(
        self, x: torch.Tensor, cond_embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = cond_embed.chunk(3, dim=-1)
        shift = self.shift_up(self.shift_down(F.silu(shift))) + shift
        scale = self.scale_up(self.scale_down(F.silu(scale))) + scale
        gate = self.gate_up(self.gate_down(F.silu(gate))) + gate

        x_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)
        x = x * (1.0 + scale) + shift
        gate = torch.tanh(gate)
        return x.to(x_dtype), gate


def patch_sequence_with_mask(
    seq: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Patch along sequence axis:
      seq: (B, S, D) -> (B, S//patch, D*patch)
      mask: (B, S) -> (B, S//patch) with all() over patch window.

    Note:
      For speaker conditioning in this project, `seq` is already in
      latent-patched space (D = latent_dim * latent_patch_size).
      This helper applies an additional sequence patching for
      `speaker_patch_size`.
    """
    if patch_size <= 1:
        return seq, mask
    if seq.ndim != 3 or mask.ndim != 2:
        raise ValueError(
            f"Expected seq=(B,S,D), mask=(B,S), got seq={tuple(seq.shape)} mask={tuple(mask.shape)}"
        )
    if seq.shape[0] != mask.shape[0] or seq.shape[1] != mask.shape[1]:
        raise ValueError(
            f"Sequence/mask shape mismatch: seq={tuple(seq.shape)}, mask={tuple(mask.shape)}. "
            "Expected matching (B,S)."
        )
    bsz, seq_len, dim = seq.shape
    usable = (seq_len // patch_size) * patch_size
    if usable <= 0:
        raise ValueError(
            f"Reference sequence too short for speaker_patch_size={patch_size}: seq_len={seq_len}"
        )
    seq = seq[:, :usable].reshape(bsz, usable // patch_size, dim * patch_size)
    mask = mask[:, :usable].reshape(bsz, usable // patch_size, patch_size).all(dim=-1)
    return seq, mask


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, norm_eps: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        if (dim // heads) % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)

        self.q_norm = RMSNorm((self.heads, self.head_dim), eps=norm_eps)
        self.k_norm = RMSNorm((self.heads, self.head_dim), eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        key_mask: torch.Tensor | None,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.wq(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        k = self.wk(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        v = self.wv(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        gate = self.gate(x)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rotary_emb(q, freqs_cis[:seq_len])
        k = apply_rotary_emb(k, freqs_cis[:seq_len])

        attn_mask = None
        if key_mask is not None:
            attn_mask = key_mask[:, None, None, :]

        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask,
            is_causal=False,
        ).transpose(1, 2)
        y = y.reshape(bsz, seq_len, self.dim)
        y = y * torch.sigmoid(gate)
        return self.wo(y)


class JointAttention(nn.Module):
    """
    Echo-style joint attention over latent self tokens + conditioning contexts.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        text_ctx_dim: int,
        speaker_ctx_dim: int | None,
        caption_ctx_dim: int | None,
        norm_eps: float,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        if (dim // heads) % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wk_text = nn.Linear(text_ctx_dim, dim, bias=False)
        self.wv_text = nn.Linear(text_ctx_dim, dim, bias=False)
        self.has_speaker_condition = speaker_ctx_dim is not None
        if self.has_speaker_condition:
            self.wk_speaker = nn.Linear(int(speaker_ctx_dim), dim, bias=False)
            self.wv_speaker = nn.Linear(int(speaker_ctx_dim), dim, bias=False)
        self.has_caption_condition = caption_ctx_dim is not None
        if self.has_caption_condition:
            self.wk_caption = nn.Linear(int(caption_ctx_dim), dim, bias=False)
            self.wv_caption = nn.Linear(int(caption_ctx_dim), dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.q_norm = RMSNorm((self.heads, self.head_dim), eps=norm_eps)
        self.k_norm = RMSNorm((self.heads, self.head_dim), eps=norm_eps)

    def _apply_rotary_half(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x_rot, x_passthrough = x.chunk(2, dim=-2)
        x_rot = apply_rotary_emb(x_rot, freqs_cis)
        return torch.cat([x_rot, x_passthrough], dim=-2)

    def project_context_kv(
        self,
        text_context: torch.Tensor,
        speaker_context: torch.Tensor | None,
        caption_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """
        Precompute conditioning KV projections for static conditioning.
        """
        bsz = text_context.shape[0]
        k_text = self.wk_text(text_context).reshape(
            bsz, text_context.shape[1], self.heads, self.head_dim
        )
        v_text = self.wv_text(text_context).reshape(
            bsz, text_context.shape[1], self.heads, self.head_dim
        )
        k_text = self.k_norm(k_text)
        projected: list[torch.Tensor] = [k_text, v_text]
        if self.has_speaker_condition:
            if speaker_context is None:
                raise ValueError(
                    "speaker_context is required when speaker conditioning is enabled."
                )
            if speaker_context.shape[0] != bsz:
                raise ValueError(
                    "Batch mismatch for context projection: "
                    f"text={tuple(text_context.shape)} speaker={tuple(speaker_context.shape)}"
                )
            k_speaker = self.wk_speaker(speaker_context).reshape(
                bsz, speaker_context.shape[1], self.heads, self.head_dim
            )
            v_speaker = self.wv_speaker(speaker_context).reshape(
                bsz, speaker_context.shape[1], self.heads, self.head_dim
            )
            k_speaker = self.k_norm(k_speaker)
            projected.extend([k_speaker, v_speaker])
        elif speaker_context is not None and speaker_context.shape[0] != bsz:
            raise ValueError(
                "Batch mismatch for ignored speaker context: "
                f"text={tuple(text_context.shape)} speaker={tuple(speaker_context.shape)}"
            )
        if not self.has_caption_condition:
            return tuple(projected)
        if caption_context is None:
            raise ValueError("caption_context is required when caption conditioning is enabled.")
        if caption_context.shape[0] != bsz:
            raise ValueError(
                "Batch mismatch for caption context projection: "
                f"text={tuple(text_context.shape)} caption={tuple(caption_context.shape)}"
            )
        k_caption = self.wk_caption(caption_context).reshape(
            bsz, caption_context.shape[1], self.heads, self.head_dim
        )
        v_caption = self.wv_caption(caption_context).reshape(
            bsz, caption_context.shape[1], self.heads, self.head_dim
        )
        k_caption = self.k_norm(k_caption)
        projected.extend([k_caption, v_caption])
        return tuple(projected)

    def forward(
        self,
        x: torch.Tensor,
        text_context: torch.Tensor,
        text_mask: torch.Tensor | None,
        speaker_context: torch.Tensor | None,
        speaker_mask: torch.Tensor | None,
        caption_context: torch.Tensor | None,
        caption_mask: torch.Tensor | None,
        freqs_cis: torch.Tensor,
        self_mask: torch.Tensor | None = None,
        context_kv: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.wq(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        k_self = self.wk(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        v_self = self.wv(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        if context_kv is None:
            projected = self.project_context_kv(
                text_context=text_context,
                speaker_context=speaker_context,
                caption_context=caption_context,
            )
        else:
            projected = context_kv
        if projected is None:
            raise RuntimeError("JointAttention projected context unexpectedly missing.")
        offset = 0
        k_text, v_text = projected[offset], projected[offset + 1]
        offset += 2
        k_speaker = None
        v_speaker = None
        if self.has_speaker_condition:
            k_speaker, v_speaker = projected[offset], projected[offset + 1]
            offset += 2
        k_caption = None
        v_caption = None
        if self.has_caption_condition:
            k_caption, v_caption = projected[offset], projected[offset + 1]

        q = self.q_norm(q)
        k_self = self.k_norm(k_self)
        q = self._apply_rotary_half(q, freqs_cis[:seq_len])
        k_self = self._apply_rotary_half(k_self, freqs_cis[:seq_len])

        if self_mask is None:
            self_mask = torch.ones((bsz, seq_len), dtype=torch.bool, device=x.device)
        if text_mask is None:
            text_mask = torch.ones(
                (bsz, text_context.shape[1]),
                dtype=torch.bool,
                device=x.device,
            )
        context_k = [k_self, k_text]
        context_v = [v_self, v_text]
        context_masks = [self_mask, text_mask]
        if self.has_speaker_condition:
            if speaker_context is None or k_speaker is None or v_speaker is None:
                raise ValueError(
                    "speaker_context is required when speaker conditioning is enabled."
                )
            if speaker_mask is None:
                speaker_mask = torch.ones(
                    (bsz, speaker_context.shape[1]),
                    dtype=torch.bool,
                    device=x.device,
                )
            context_k.append(k_speaker)
            context_v.append(v_speaker)
            context_masks.append(speaker_mask)
        if self.has_caption_condition:
            if caption_context is None:
                raise ValueError(
                    "caption_context is required when caption conditioning is enabled."
                )
            if caption_mask is None:
                caption_mask = torch.ones(
                    (bsz, caption_context.shape[1]),
                    dtype=torch.bool,
                    device=x.device,
                )
            if k_caption is None or v_caption is None:
                raise RuntimeError(
                    "Caption projections are missing despite enabled caption conditioning."
                )
            context_k.append(k_caption)
            context_v.append(v_caption)
            context_masks.append(caption_mask)

        k = torch.cat(context_k, dim=1)
        v = torch.cat(context_v, dim=1)
        attn_mask = torch.cat(context_masks, dim=1)
        attn_mask = attn_mask[:, None, None, :]

        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask,
            is_causal=False,
        ).transpose(1, 2)
        y = y.reshape(bsz, seq_len, self.dim)
        y = y * torch.sigmoid(self.gate(x))
        return self.wo(y)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TextBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, norm_eps: float, dropout: float):
        super().__init__()
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.attention = SelfAttention(dim, heads, norm_eps=norm_eps)
        self.mlp_norm = RMSNorm(dim, eps=norm_eps)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(
            self.attention(self.attention_norm(x), key_mask=mask, freqs_cis=freqs_cis)
        )
        x = x + self.dropout(self.mlp(self.mlp_norm(x)))
        return x


class TextEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        norm_eps: float,
        dropout: float,
    ):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            TextBlock(
                dim=dim,
                heads=heads,
                mlp_ratio=mlp_ratio,
                norm_eps=norm_eps,
                dropout=dropout,
            )
            for _ in range(layers)
        )
        self.head_dim = dim // heads
        self.register_buffer(
            "_freqs_cis_cache", torch.empty(0, 0, dtype=torch.complex64), persistent=False
        )

    def _rope_freqs(self, seq_len: int, device: torch.device) -> torch.Tensor:
        cache = self._freqs_cis_cache
        if cache.device != device or cache.shape[0] < seq_len:
            cache = precompute_freqs_cis(self.head_dim, seq_len).to(device)
            self._freqs_cis_cache = cache
        return cache[:seq_len]

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.text_embedding(input_ids)
        # Hard-mask invalid tokens so fully-masked conditioning becomes truly unconditional.
        mask_f = mask.unsqueeze(-1).to(dtype=x.dtype)
        x = x * mask_f
        freqs = self._rope_freqs(input_ids.shape[1], x.device)
        for block in self.blocks:
            x = block(x, mask=mask, freqs_cis=freqs)
            x = x * mask_f
        return x * mask_f


class ReferenceLatentEncoder(nn.Module):
    """
    Encoder for reference latents used as speaker/style conditioning.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.in_proj = nn.Linear(cfg.speaker_patched_latent_dim, cfg.speaker_dim, bias=True)
        speaker_mlp_ratio = cfg.speaker_mlp_ratio_resolved
        self.blocks = nn.ModuleList(
            TextBlock(
                dim=cfg.speaker_dim,
                heads=cfg.speaker_heads,
                mlp_ratio=speaker_mlp_ratio,
                norm_eps=cfg.norm_eps,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.speaker_layers)
        )
        self.head_dim = cfg.speaker_dim // cfg.speaker_heads
        self.register_buffer(
            "_freqs_cis_cache", torch.empty(0, 0, dtype=torch.complex64), persistent=False
        )

    def _rope_freqs(self, seq_len: int, device: torch.device) -> torch.Tensor:
        cache = self._freqs_cis_cache
        if cache.device != device or cache.shape[0] < seq_len:
            cache = precompute_freqs_cis(self.head_dim, seq_len).to(device)
            self._freqs_cis_cache = cache
        return cache[:seq_len]

    def forward(self, latent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(latent)
        x = x / 6.0
        # Keep masked reference positions strictly zero across residual/MLP paths.
        mask_f = mask.unsqueeze(-1).to(dtype=x.dtype)
        x = x * mask_f
        freqs = self._rope_freqs(x.shape[1], x.device)
        for block in self.blocks:
            x = block(x, mask=mask, freqs_cis=freqs)
            x = x * mask_f
        return x * mask_f


class DiffusionBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attention = JointAttention(
            cfg.model_dim,
            cfg.num_heads,
            cfg.text_dim,
            cfg.speaker_dim if cfg.use_speaker_condition else None,
            cfg.caption_dim_resolved if cfg.use_caption_condition else None,
            norm_eps=cfg.norm_eps,
        )
        self.mlp = SwiGLU(cfg.model_dim, int(cfg.model_dim * cfg.mlp_ratio))
        adaln_rank = max(1, min(int(cfg.adaln_rank), int(cfg.model_dim)))
        self.attention_adaln = LowRankAdaLN(
            model_dim=cfg.model_dim,
            rank=adaln_rank,
            eps=cfg.norm_eps,
        )
        self.mlp_adaln = LowRankAdaLN(
            model_dim=cfg.model_dim,
            rank=adaln_rank,
            eps=cfg.norm_eps,
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond_embed: torch.Tensor,
        text_state: torch.Tensor,
        text_mask: torch.Tensor,
        speaker_state: torch.Tensor | None,
        speaker_mask: torch.Tensor | None,
        caption_state: torch.Tensor | None,
        caption_mask: torch.Tensor | None,
        freqs_cis: torch.Tensor,
        self_mask: torch.Tensor | None = None,
        context_kv: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        h, attention_gate = self.attention_adaln(x, cond_embed)
        x = x + self.dropout(
            attention_gate
            * self.attention(
                x=h,
                text_context=text_state,
                text_mask=text_mask,
                speaker_context=speaker_state,
                speaker_mask=speaker_mask,
                caption_context=caption_state,
                caption_mask=caption_mask,
                freqs_cis=freqs_cis,
                self_mask=self_mask,
                context_kv=context_kv,
            )
        )

        h, mlp_gate = self.mlp_adaln(x, cond_embed)
        x = x + self.dropout(mlp_gate * self.mlp(h))
        return x


class TextToLatentRFDiT(nn.Module):
    """
    Text + reference-latent conditioned RF diffusion model over patched DACVAE latent sequences.

    Input x_t shape: (B, S, latent_dim * latent_patch_size)
    Output v_pred shape: same as input.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.text_encoder = TextEncoder(
            vocab_size=cfg.text_vocab_size,
            dim=cfg.text_dim,
            layers=cfg.text_layers,
            heads=cfg.text_heads,
            mlp_ratio=cfg.text_mlp_ratio_resolved,
            norm_eps=cfg.norm_eps,
            dropout=cfg.dropout,
        )
        self.caption_encoder = None
        self.caption_norm = None
        if cfg.use_caption_condition:
            self.caption_encoder = TextEncoder(
                vocab_size=cfg.caption_vocab_size_resolved,
                dim=cfg.caption_dim_resolved,
                layers=cfg.caption_layers_resolved,
                heads=cfg.caption_heads_resolved,
                mlp_ratio=cfg.caption_mlp_ratio_resolved,
                norm_eps=cfg.norm_eps,
                dropout=cfg.dropout,
            )
            self.caption_norm = RMSNorm(cfg.caption_dim_resolved, eps=cfg.norm_eps)
        self.speaker_encoder = None
        if cfg.use_speaker_condition:
            self.speaker_encoder = ReferenceLatentEncoder(cfg)
        self.text_norm = RMSNorm(cfg.text_dim, eps=cfg.norm_eps)
        self.speaker_norm = None
        if cfg.use_speaker_condition:
            self.speaker_norm = RMSNorm(cfg.speaker_dim, eps=cfg.norm_eps)

        self.cond_module = nn.Sequential(
            nn.Linear(cfg.timestep_embed_dim, cfg.model_dim, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.model_dim, cfg.model_dim, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.model_dim, cfg.model_dim * 3, bias=False),
        )

        self.in_proj = nn.Linear(cfg.patched_latent_dim, cfg.model_dim)
        self.blocks = nn.ModuleList(DiffusionBlock(cfg) for _ in range(cfg.num_layers))
        self.out_norm = RMSNorm(cfg.model_dim, eps=cfg.norm_eps)
        self.out_proj = nn.Linear(cfg.model_dim, cfg.patched_latent_dim)
        # Echo/JAX training initializes decoder out projection to zero for stable early training.
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        self.head_dim = cfg.model_dim // cfg.num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("model head_dim must be even for RoPE")
        self.register_buffer(
            "_freqs_cis_cache", torch.empty(0, 0, dtype=torch.complex64), persistent=False
        )

    def _rope_freqs(self, seq_len: int, device: torch.device) -> torch.Tensor:
        cache = self._freqs_cis_cache
        if cache.device != device or cache.shape[0] < seq_len:
            cache = precompute_freqs_cis(self.head_dim, seq_len).to(device)
            self._freqs_cis_cache = cache
        return cache[:seq_len]

    @staticmethod
    def _prepend_masked_mean_token(
        state: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepend one global summary token computed as masked mean over time.
        """
        mask_f = mask.unsqueeze(-1).to(dtype=state.dtype)
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_token = (state * mask_f).sum(dim=1, keepdim=True) / denom
        has_any = mask.any(dim=1, keepdim=True)
        state = torch.cat([mean_token, state], dim=1)
        mask = torch.cat([has_any, mask], dim=1)
        return state, mask

    def encode_conditions(
        self,
        text_input_ids: torch.Tensor,
        text_mask: torch.Tensor,
        ref_latent: torch.Tensor | None,
        ref_mask: torch.Tensor | None,
        caption_input_ids: torch.Tensor | None = None,
        caption_mask: torch.Tensor | None = None,
        text_condition_dropout: torch.Tensor | None = None,
        speaker_condition_dropout: torch.Tensor | None = None,
        caption_condition_dropout: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if text_condition_dropout is not None:
            text_mask = text_mask.clone()
            text_mask[text_condition_dropout] = False
        if self.cfg.use_speaker_condition:
            if self.speaker_encoder is None or self.speaker_norm is None:
                raise RuntimeError(
                    "Speaker conditioning is enabled but speaker modules are missing."
                )
            if ref_latent is None or ref_mask is None:
                raise ValueError(
                    "ref_latent and ref_mask are required when speaker conditioning is enabled."
                )
            if speaker_condition_dropout is not None:
                ref_mask = ref_mask.clone()
                ref_mask[speaker_condition_dropout] = False
        if self.cfg.use_caption_condition:
            if self.caption_encoder is None or self.caption_norm is None:
                raise RuntimeError(
                    "Caption conditioning is enabled but caption modules are missing."
                )
            if caption_input_ids is None or caption_mask is None:
                raise ValueError(
                    "caption_input_ids and caption_mask are required when caption conditioning is enabled."
                )
            if caption_condition_dropout is not None:
                caption_mask = caption_mask.clone()
                caption_mask[caption_condition_dropout] = False

        text_state = self.text_encoder(text_input_ids, text_mask)
        text_state = self.text_norm(text_state)
        ref_state = None
        if self.cfg.use_speaker_condition:
            ref_latent, ref_mask = patch_sequence_with_mask(
                seq=ref_latent,
                mask=ref_mask,
                patch_size=self.cfg.speaker_patch_size,
            )
            ref_state = self.speaker_encoder(ref_latent, ref_mask)
            ref_state = self.speaker_norm(ref_state)
            ref_state, ref_mask = self._prepend_masked_mean_token(ref_state, ref_mask)
        caption_state = None
        if self.cfg.use_caption_condition:
            caption_state = self.caption_encoder(caption_input_ids, caption_mask)
            caption_state = self.caption_norm(caption_state)
        return text_state, text_mask, ref_state, ref_mask, caption_state, caption_mask

    def forward_with_encoded_conditions(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_state: torch.Tensor,
        text_mask: torch.Tensor,
        speaker_state: torch.Tensor | None,
        speaker_mask: torch.Tensor | None,
        caption_state: torch.Tensor | None = None,
        caption_mask: torch.Tensor | None = None,
        latent_mask: torch.Tensor | None = None,
        context_kv_cache: list[tuple[torch.Tensor, ...]] | None = None,
    ) -> torch.Tensor:
        t_embed = get_timestep_embedding(t, self.cfg.timestep_embed_dim).to(dtype=x_t.dtype)
        cond_embed = self.cond_module(t_embed)
        cond_embed = cond_embed[:, None, :]

        x = self.in_proj(x_t)
        freqs = self._rope_freqs(x.shape[1], x.device)
        for i, block in enumerate(self.blocks):
            x = block(
                x=x,
                cond_embed=cond_embed,
                text_state=text_state,
                text_mask=text_mask,
                speaker_state=speaker_state,
                speaker_mask=speaker_mask,
                caption_state=caption_state,
                caption_mask=caption_mask,
                freqs_cis=freqs,
                self_mask=latent_mask,
                context_kv=context_kv_cache[i] if context_kv_cache is not None else None,
            )

        x = self.out_norm(x)
        x = self.out_proj(x)
        return x.to(dtype=x_t.dtype)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_mask: torch.Tensor,
        ref_latent: torch.Tensor | None,
        ref_mask: torch.Tensor | None,
        caption_input_ids: torch.Tensor | None = None,
        caption_mask: torch.Tensor | None = None,
        latent_mask: torch.Tensor | None = None,
        text_condition_dropout: torch.Tensor | None = None,
        speaker_condition_dropout: torch.Tensor | None = None,
        caption_condition_dropout: torch.Tensor | None = None,
    ) -> torch.Tensor:
        (
            text_state,
            text_mask,
            speaker_state,
            speaker_mask,
            caption_state,
            caption_mask,
        ) = self.encode_conditions(
            text_input_ids=text_input_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            caption_input_ids=caption_input_ids,
            caption_mask=caption_mask,
            text_condition_dropout=text_condition_dropout,
            speaker_condition_dropout=speaker_condition_dropout,
            caption_condition_dropout=caption_condition_dropout,
        )
        return self.forward_with_encoded_conditions(
            x_t=x_t,
            t=t,
            text_state=text_state,
            text_mask=text_mask,
            speaker_state=speaker_state,
            speaker_mask=speaker_mask,
            caption_state=caption_state,
            caption_mask=caption_mask,
            latent_mask=latent_mask,
        )

    def build_context_kv_cache(
        self,
        text_state: torch.Tensor,
        speaker_state: torch.Tensor | None,
        caption_state: torch.Tensor | None = None,
    ) -> list[tuple[torch.Tensor, ...]]:
        """
        Build per-layer projected conditioning KV tensors for faster repeated sampling steps.
        """
        return [
            block.attention.project_context_kv(
                text_context=text_state,
                speaker_context=speaker_state,
                caption_context=caption_state,
            )
            for block in self.blocks
        ]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def as_dict(self) -> dict:
        return asdict(self.cfg)
