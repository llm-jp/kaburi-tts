"""KABURI acoustic model: build / checkpoint load / CFG inference core.

cfg_scale=1.0 skips the uncond pass (numerically identical to plain inference).
cfg_scale>1.0 batches cond/uncond into one forward and blends the velocities.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch

from kaburi_tts import ASSETS_DIR, HF_ACOUSTIC_FILE, HF_REPO, REPO_ROOT
from kaburi_tts.acoustic.model import KaburiAcousticModel
from kaburi_tts.two_stream.model import load_base_with_pretrained


def _is_norm(name: str) -> bool:
    return ("norm" in name.lower()) or ("adaln" in name.lower())


def resolve_path(p: str | Path) -> Path:
    """Resolve a config path: absolute stays as is, relative is repo-root based."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def apply_lora(model, *, r, alpha, dropout, bias, target_regex, is_main,
               unfreeze_layer_norm, unfreeze_last_n_blocks, unfreeze_io):
    from peft import LoraConfig, get_peft_model
    peft_model = get_peft_model(model, LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias=bias,
        target_modules=target_regex, task_type=None, inference_mode=False))
    full_paths = ["activity_proj", "phone_proj", "phone_emb", "phone_temporal_conv",
                  "interaction_state_emb", "soft_phone_conditioner", "timing_aux_head",
                  "phone_reinject_proj"]
    re_full = re.compile(r"^base_model\.model\.(?:" + "|".join(map(re.escape, full_paths)) + r")\b")
    re_io = re.compile(r"^base_model\.model\.base\.(in_proj|out_proj|out_norm)\b") if unfreeze_io else None
    n_total = 0
    for n_, _ in peft_model.named_parameters():
        m = re.match(r"^base_model\.model\.base\.blocks\.(\d+)\.", n_)
        if m: n_total = max(n_total, int(m.group(1)) + 1)
    first_unf = max(0, n_total - unfreeze_last_n_blocks)
    pg = {"lora": [], "new": [], "norm": [], "unblk": [], "io": []}
    for name, p in peft_model.named_parameters():
        if "lora_" in name: pg["lora"].append(p); continue
        if re_full.match(name): p.requires_grad = True; pg["new"].append(p); continue
        if re_io is not None and re_io.match(name): p.requires_grad = True; pg["io"].append(p); continue
        m = re.match(r"^base_model\.model\.base\.blocks\.(\d+)\.", name)
        if m and int(m.group(1)) >= first_unf: p.requires_grad = True; pg["unblk"].append(p); continue
        if unfreeze_layer_norm and _is_norm(name): p.requires_grad = True; pg["norm"].append(p); continue
        p.requires_grad = False
    if is_main:
        tot = sum(sum(x.numel() for x in v) for v in pg.values())
        print("[lora] " + "  ".join(f"{k}={sum(x.numel() for x in v)/1e6:.2f}M" for k, v in pg.items())
              + f"  total={tot/1e6:.2f}M", flush=True)
        print(f"[unfreeze] last {unfreeze_last_n_blocks}/{n_total} blocks (= {first_unf}..{n_total-1})", flush=True)
    return peft_model, pg


def model_kwargs(batch, pct8):
    out = dict(phone_A=batch["phone_A"], phone_B=batch["phone_B"],
               activity_A=batch["activity_A"], activity_B=batch["activity_B"],
               ref_latent_A=batch["ref_latent_A"], ref_mask_A=batch["ref_mask_A"],
               ref_latent_B=batch["ref_latent_B"], ref_mask_B=batch["ref_mask_B"],
               text_A_input_ids=batch["text_A_input_ids"], text_A_mask=batch["text_A_mask"],
               text_B_input_ids=batch["text_B_input_ids"], text_B_mask=batch["text_B_mask"],
               latent_mask=batch["latent_mask"])
    if "phone_cur_A" in batch:
        keys = ("phone_cur", "phone_prev", "phone_next", "phone_pos_frac", "phone_dur_norm",
                "dist_start_norm", "dist_end_norm", "is_phone_boundary")
        for ch in ("A", "B"):
            d = {k: batch[f"{k}_{ch}"] for k in keys}
            d["phone_class"] = pct8[d["phone_cur"]] if pct8 is not None else torch.zeros_like(d["phone_cur"])
            out[f"soft_phone_{ch}"] = d
    return out


# backward-compat alias (training scripts historically used the private name)
_model_kwargs = model_kwargs


def build_model(cfg, device):
    mcfg, lcfg = cfg["model"], cfg["lora"]
    pccfg, capcfg, tacfg, kcfg = cfg.get("phone_condition", {}), cfg["capacity"], cfg["timing_aux"], cfg["kaburi"]
    base = load_base_with_pretrained(resolve_path(mcfg["base_config"]), pretrained_repo=mcfg["base_checkpoint"])
    m = KaburiAcousticModel(
        base, phone_vocab_size=int(mcfg["phone_vocab_size"]), phone_emb_dim=int(mcfg["phone_emb_dim"]),
        use_activity=bool(mcfg.get("use_activity_condition", True)),
        use_phone=bool(mcfg.get("use_phone_condition", True)),
        phone_proj_hidden_dim=mcfg.get("phone_proj_hidden_dim"),
        use_phone_temporal_conv=bool(mcfg.get("use_phone_temporal_conv", False)),
        phone_temporal_conv_kernel=int(mcfg.get("phone_temporal_conv_kernel", 5)),
        use_interaction_state=bool(mcfg.get("use_interaction_state", False)),
        interaction_state_vocab=int(mcfg.get("interaction_state_vocab", 10)),
        text_scale=float(mcfg.get("text_scale", 1.0)),
        phone_condition_mode=str(pccfg.get("mode", "hard")),
        soft_phone_class_size=int(pccfg.get("class_size", 8)),
        soft_phone_class_emb_dim=int(pccfg.get("class_emb_dim", 32)),
        soft_phone_hidden_dim=int(pccfg.get("hidden_dim", 512)),
        soft_phone_temporal_conv_kernel=int(pccfg.get("temporal_conv_kernel", 5)),
        soft_phone_zero_init_last=bool(pccfg.get("zero_init_last", True)),
        use_timing_aux_head=bool(tacfg.get("enabled", False)),
        phone_reinject_blocks=list(kcfg.get("phone_reinject_blocks", [])))
    m, _ = apply_lora(m, r=int(lcfg["r"]), alpha=int(lcfg["alpha"]), dropout=float(lcfg["dropout"]),
                      bias=str(lcfg.get("bias", "none")), target_regex=str(lcfg["target_modules_regex"]), is_main=True,
                      unfreeze_layer_norm=bool(capcfg.get("unfreeze_layer_norm", False)),
                      unfreeze_last_n_blocks=int(capcfg.get("unfreeze_last_n_blocks", 0)),
                      unfreeze_io=bool(capcfg.get("unfreeze_io", False)))
    return m.to(device)


def load_acoustic_checkpoint(model, ckpt: str | Path | None, device) -> int | str:
    """Load acoustic weights. `ckpt` may be a local .pt/.safetensors path or None
    (downloads the released checkpoint from the Hugging Face repo)."""
    if ckpt is None:
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(repo_id=HF_REPO, filename=HF_ACOUSTIC_FILE)
    ckpt = Path(ckpt)
    if ckpt.suffix == ".safetensors":
        from safetensors.torch import load_file
        state_dict, step = load_file(str(ckpt)), "?"
    else:
        st = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        state_dict, step = st["model_state_dict"], st.get("step", "?")
    miss, unexp = model.load_state_dict(state_dict, strict=False)
    print(f"[load] acoustic step={step} missing={len(miss)} unexpected={len(unexp)}", flush=True)
    return step


def default_phone_vocab_path() -> Path:
    return ASSETS_DIR / "phone_vocab.json"


@torch.no_grad()
def synth(raw, codec, batch, *, device, num_steps=32, seed=0, cfg_scale=2.5, pct8=None):
    raw.eval()
    B, T, D = batch["latent_A"].shape
    assert B == 1
    g = torch.Generator(device=device).manual_seed(seed)
    xA = torch.randn(B, T, D, device=device, generator=g)
    xB = torch.randn(B, T, D, device=device, generator=g)
    ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device) * 0.999
    mk = model_kwargs(batch, pct8)
    use_cfg = (cfg_scale != 1.0)
    if use_cfg:
        mk_c = {}
        for k, v in mk.items():
            if torch.is_tensor(v): mk_c[k] = torch.cat([v, v], 0)
            elif isinstance(v, dict): mk_c[k] = {kk: (torch.cat([vv, vv], 0) if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
            else: mk_c[k] = v
        dpm = torch.zeros(2 * B, dtype=torch.bool, device=device); dpm[B:] = True
        dsm = torch.zeros(2 * B, dtype=torch.bool, device=device)
    for i in range(num_steps):
        t = ts[i]; tn = ts[i + 1]
        if not use_cfg:
            tt = torch.full((B,), t.item(), device=device)
            vA, vB = raw(xA, xB, tt, **mk)
        else:
            tt = torch.full((2 * B,), t.item(), device=device)
            vAc, vBc = raw(torch.cat([xA, xA], 0), torch.cat([xB, xB], 0), tt, **mk_c,
                           drop_phone_mask=dpm, drop_speaker_mask=dsm)
            vA = vAc[B:].float() + cfg_scale * (vAc[:B].float() - vAc[B:].float())
            vB = vBc[B:].float() + cfg_scale * (vBc[:B].float() - vBc[B:].float())
        xA = xA + vA.float() * (tn - t); xB = xB + vB.float() * (tn - t)
    L = int(batch["latent_mask"][0].sum().item())
    wA = codec.decode_latent(xA[0, :L].to(dtype=torch.bfloat16).unsqueeze(0))[0, 0].float().cpu()
    wB = codec.decode_latent(xB[0, :L].to(dtype=torch.bfloat16).unsqueeze(0))[0, 0].float().cpu()
    n = min(wA.shape[0], wB.shape[0])
    return torch.stack([wA[:n], wB[:n]], 0)
