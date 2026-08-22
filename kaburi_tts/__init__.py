"""KABURI-TTS: 2-channel Japanese dialogue TTS (acoustic model + timing predictor)."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "assets"
CONFIGS_DIR = REPO_ROOT / "configs"

# Hugging Face repo holding the released checkpoints
HF_REPO = "llm-jp/kaburi-tts"
HF_ACOUSTIC_FILE = "acoustic/model.safetensors"
HF_PREDICTOR_FILE = "predictor/model.safetensors"
