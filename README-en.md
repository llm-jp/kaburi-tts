# KABURI-TTS

[![Japanese](https://img.shields.io/badge/README-Japanese-red.svg)](README.md)

📑 Paper (APSIPA ASC 2026, to appear) | [🤗 Model](https://huggingface.co/llm-jp/kaburi-tts) | [🖥️ Demo](https://llm-jp.github.io/kaburi-tts) | [🎧 Samples](docs/)

**KABURI-TTS** is a spoken dialogue synthesis system that generates two-speaker Japanese conversations as two simultaneous audio channels. It reproduces the timing structure characteristic of conversation — backchannels, overlaps (**kaburi** in Japanese), and pauses — from text alone.

This work was developed as part of the Dialogue Working Group of the Research and Development Center for Large Language Models (LLMC), National Institute of Informatics (NII).

> [!NOTE]
> The `paper-release-v1` tag is a frozen snapshot of the timing-predictor pipeline evaluated in the paper. The standard synthesis command directly uses the paper pipeline. See the `main` branch for the current release.

- **Acoustic model**: a rectified-flow model that adapts a pretrained single-speaker TTS ([Irodori-TTS-500M-v2](https://huggingface.co/Aratako/Irodori-TTS-500M-v2)) to the dialogue domain with LoRA + partial unfreezing. Conditioned on phone sequences, utterance activity, and speaker reference audio, it generates latents for both channels, decoded to 48 kHz stereo by the [DACVAE codec](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim). Inference uses classifier-free guidance (cfg_scale=2.5).
- **Timing predictor**: a lightweight Transformer that predicts per-phone durations, pre-utterance silences, and cross-channel gaps/overlaps from the dialogue text (phone sequences + utterance context), rendering them into two-channel phone rasters for the acoustic model.

## Setup

Python 3.10+ with a CUDA GPU (single-GPU inference, ~6 GB in bf16).

```bash
git clone https://github.com/llm-jp/kaburi-tts.git
cd kaburi-tts
uv sync   # installs Python 3.10/3.11 and all dependencies (CUDA 12.8 wheels)
```

With [uv](https://docs.astral.sh/uv/), run every command below with `uv run ...` (no venv activation needed).

Without uv (Python 3.10/3.11 + pip):

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Model weights are downloaded automatically from [llm-jp/kaburi-tts](https://huggingface.co/llm-jp/kaburi-tts) on first run (~3GB in total including the 2.3GB acoustic model; the first run takes a few extra minutes; `huggingface-cli login` is required only while the repository is private). Later runs use the cache — synthesizing one 30-second dialogue takes about 1–2 minutes on a single GPU.

## Usage

### Quick start (bundled test voices)

A reference pack of **two non-existent synthetic voices** (generated with Irodori-TTS VoiceDesign) is bundled at [assets/test_refpack/](assets/test_refpack/), so you can synthesize a dialogue text file (one utterance per line, `A: ...` / `B: ...`; see [assets/sample_dialog.txt](assets/sample_dialog.txt)) right away:

```bash
uv run scripts/kaburi_cli.py assets/sample_dialog.txt --ref-pack assets/test_refpack/ -o out.wav
uv run scripts/kaburi_cli.py assets/sample_dialog.txt --ref-pack assets/test_refpack/ --mode stat   # predictor-free statistical placement
```

The output is a stereo wav with speaker A on the left channel and speaker B on the right.

### Synthesis with your own voices

Build a reference pack from any two reference recordings (speakers A/B, 3–10 s each, with recording consent):

```bash
uv run scripts/make_ref_pack.py --ref-a voiceA.wav --ref-b voiceB.wav --out-dir refpack/
uv run scripts/kaburi_cli.py assets/sample_dialog.txt --ref-pack refpack/ -o out.wav
```

Read-style or studio-quality references can degrade quality. In that case, **bootstrap refinement** is recommended: synthesize one dialogue with KABURI and rebuild the references from its conversational output:

```bash
uv run scripts/refine_ref_pack.py --ref-pack refpack/ --out-dir refpack_refined/
uv run scripts/kaburi_cli.py assets/sample_dialog.txt --ref-pack refpack_refined/ -o out.wav
```

The bundled `assets/test_refpack/` was built this way.

> [!TIP]
> Symbols, digits, Latin characters, and proper nouns can yield unstable readings (G2P is longest-match over the MFA Japanese dictionary); plain kana/kanji text is recommended.

### Write input text in spoken style — or convert automatically

KABURI-TTS is trained on spontaneous conversations, so **the closer the input text is to real spoken Japanese, the more natural the synthesis** (short turns, backchannels, fillers, casual sentence endings; polished written-style sentences degrade quality).

Written-style dialogues can be converted automatically with the bundled **text converter** (llm-jp-3-440m fine-tuned toward the transcript style of the training corpus). It splits long sentences into short turns, inserts backchannels, and adds fillers, producing text that matches the training distribution. **As a rule of thumb, 200–250 characters of source text yield about 30 seconds of audio** (synthesis is capped at 30 seconds; anything longer is uniformly compressed into faster speech — the converter changes style, not duration):

```bash
# convert and synthesize in one go (recommended)
uv run scripts/kaburi_cli.py dialog.txt --ref-pack refpack/ --convert

# convert only, to inspect the text first
uv run scripts/convert_dialog.py dialog.txt -o dialog_spoken.txt
```

Check spontaneity with the bundled scorer:

```bash
uv run scripts/check_spontaneity.py dialog.txt
```

Without the converter, you can also pass [assets/spontaneous_rewrite_prompt.txt](assets/spontaneous_rewrite_prompt.txt) to any LLM and pick the best-scoring rewrite (`kaburi_cli.py` also warns automatically at synthesis time).

Known limitations of the converter: proper nouns and numeric expressions may occasionally be distorted, and utterances can be attributed to the wrong speaker. Inspect the converted text before synthesis when accuracy matters.

### Statistical-placement mode

A mode that places utterances statistically, based on the timing statistics of the training corpus (pause/overlap distributions), instead of using the timing predictor (the "statistical placement" condition in the paper and demo page):

```bash
uv run scripts/kaburi_cli.py assets/sample_dialog.txt --ref-pack assets/test_refpack/ --mode stat
```

## Limitations

- Output quality is bounded by the DACVAE codec (32-dim, 25 fps).
- Words missing from the G2P dictionary (inflections, digit readings, etc.) may be unclear.
- Some prosodic unnaturalness remains.
- Speaker references work best when they resemble the training recording conditions (spontaneous conversation over a meeting tool, 10 s with high speech density). `make_ref_pack.py` automatically removes silences and pads references to 10 s, but studio-quality read speech may still degrade quality.

## License of demo audio

Unlike the code, **the demo-page audio is not covered by Apache 2.0**. Demo speaker
references are derived from the [JVS corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)
(The University of Tokyo), which permits academic / non-commercial research and
personal use only (commercial use requires a separate agreement with UTokyo TLO).
The reference speakers' voices must not be used commercially or as character voices;
replace the demo audio when forking this repository for commercial purposes. See
[docs/LICENSE-audio.md](docs/LICENSE-audio.md). The bundled test reference pack
(assets/test_refpack) consists solely of non-existent synthetic voices and is free
of these restrictions.

## Terms of Use

KABURI-TTS is released for research purposes. Malicious use, including impersonation and fraud, is prohibited. Model outputs may contain biases or inaccurate or offensive content derived from the training data. The developers assume no responsibility for any damages arising from its use.

The code and model weights are released under the [Apache License 2.0](LICENSE) (demo audio below is excluded). The bundled [irodori_tts/](irodori_tts/) package remains under its upstream MIT license ([irodori_tts/LICENSE](irodori_tts/LICENSE)).

## Acknowledgements

- Base TTS / codec: [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) (MIT), [Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim), [facebookresearch/dacvae](https://github.com/facebookresearch/dacvae)
- Tokenizer: [llm-jp/llm-jp-3-150m](https://huggingface.co/llm-jp/llm-jp-3-150m)
- G2P dictionary: [MFA japanese_mfa dictionary](https://mfa-models.readthedocs.io/) (CC BY 4.0, bundled as `assets/g2p/japanese_mfa.dict`)
- Acknowledgements and funding information will be updated upon paper release.

## Citation

This work is to appear in APSIPA ASC 2026:

> Ryuichiro Higashinaka, Shinnosuke Takamichi, and Tetsuji Ogawa. "KABURI-TTS: Phoneme-Keyed Activity-conditioned Bi-channel Utterance Rendering for Interaction." In *Proceedings of the 2026 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)*, 2026. (to appear)

```bibtex
@inproceedings{kaburi-tts,
  title     = {{KABURI-TTS}: Phoneme-Keyed Activity-conditioned Bi-channel Utterance Rendering for Interaction},
  author    = {Higashinaka, Ryuichiro and Takamichi, Shinnosuke and Ogawa, Tetsuji},
  booktitle = {Proceedings of the 2026 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)},
  year      = {2026},
  note      = {to appear}
}
```
