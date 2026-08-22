<p align="center"><img src="docs/kaburi-tts-logo.png" alt="KABURI-TTS" width="440"></p>

# KABURI-TTS

[![Japanese](https://img.shields.io/badge/README-Japanese-red.svg)](README.md)

📑 Paper (APSIPA ASC 2026, to appear) | [🤗 Model](https://huggingface.co/llm-jp/kaburi-tts) | [🖥️ Demo](https://llm-jp.github.io/kaburi-tts)

**KABURI-TTS** is a spoken dialogue synthesis system that generates two-speaker Japanese conversations as two simultaneous audio channels. It reproduces the timing structure characteristic of conversation — backchannels, overlaps (**kaburi** in Japanese), and pauses — from text alone.

This work was developed as part of the Dialogue Working Group of the Research and Development Center for Large Language Models (LLMC), National Institute of Informatics (NII).

The system consists of two components:

- **Acoustic model**: a rectified-flow model that adapts a pretrained single-speaker TTS ([Irodori-TTS-500M-v2](https://huggingface.co/Aratako/Irodori-TTS-500M-v2)) to the dialogue domain with LoRA + partial unfreezing, trained on **LLM-jp-Zoom1**, a corpus of Japanese two-speaker free conversations built at NII LLMC. Conditioned on phone sequences, utterance activity, and speaker reference audio, it generates latents for both channels, decoded to 48 kHz stereo by the [DACVAE codec](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim). Inference uses classifier-free guidance (cfg_scale=2.5).
- **Raster generation**: uses a realizer (realized phones, phone durations, and intra-utterance pauses) and a gap model (gaps and overlaps), both trained on measured timing from LLM-jp-Zoom1, to turn text into two-speaker phone rasters for the acoustic model.

## Setup

Python 3.10+ with a CUDA GPU (single-GPU inference, ~6 GB in bf16). With [uv](https://docs.astral.sh/uv/), two commands set up everything, Python included:

```bash
git clone https://github.com/llm-jp/kaburi-tts.git
cd kaburi-tts
uv sync
```

Model weights are downloaded automatically from [llm-jp/kaburi-tts](https://huggingface.co/llm-jp/kaburi-tts) and its dependencies on first run (~5GB in total including the base Irodori-TTS model and codec; using `--convert` requires an additional ~1.8GB text converter download the first time). Later runs use the cache — synthesizing one 30-second dialogue takes about 1–2 minutes on a single GPU.

## Usage

### Quick start

Two reference packs of **non-existent synthetic voices** (generated with Irodori-TTS VoiceDesign) are bundled. Neither contains any real speaker (e.g. JVS), so both are free of licensing restrictions:

- [assets/test_refpack/](assets/test_refpack/) — two female voices (default)
- [assets/test_refpack_mf/](assets/test_refpack_mf/) — one male and one female voice

You can synthesize a dialogue text file (one utterance per line, `A: ...` / `B: ...`; see [assets/sample_dialogue.txt](assets/sample_dialogue.txt)) right away:

```bash
uv run scripts/kaburi_cli.py assets/sample_dialogue.txt --ref-pack assets/test_refpack/ -o out.wav
uv run scripts/kaburi_cli.py assets/sample_dialogue.txt --ref-pack assets/test_refpack/ --mode stat   # predictor-free statistical placement
```

The output is a stereo wav with speaker A on the left channel and speaker B on the right.

Feeding the bundled [assets/sample_dialogue.txt](assets/sample_dialogue.txt) (a ~30-second dialogue) produces audio like the following:

```
A: 最近さ、朝がほんと起きられなくて
B: あー、分かる
A: 目覚まし三個かけてるんだけどね
B: うんうん
A: 気づいたら全部止めちゃってるんだよね
B: えー、それもう意味ないじゃん
A: そうなんだよ、無意識に止めてるっぽくて
B: あはは
A: だからさ、最近はカーテン開けたまま寝ることにしてて
B: へえ、朝日で起きる作戦?
A: そうそう、ちょっとだけマシになった気がする
B: いいね、私もやってみようかな
```

The synthesized output (stereo, with backchannels and overlaps) is bundled as audio files: [two female voices](docs/audio/quickstart/sample_dialogue_female.mp3) and [male & female](docs/audio/quickstart/sample_dialogue_mf.mp3).

### Converting written text to spoken style (--convert)

KABURI-TTS is trained on spontaneous conversations, so **the closer the input text is to real spoken Japanese, the more natural the synthesis**.

Written-style dialogues can be converted automatically with the bundled **text converter** (llm-jp-3-440m fine-tuned toward the transcript style of LLM-jp-Zoom1). It splits long sentences into short turns, inserts backchannels, and adds fillers. **As a rule of thumb, 200–250 characters of source text yield about 30 seconds of audio** (synthesis is capped at 30 seconds; anything longer is uniformly compressed into faster speech — the converter changes style, not duration):

```bash
uv run scripts/kaburi_cli.py dialogue.txt --ref-pack assets/test_refpack/ --convert
```

To inspect the converted text first, run the conversion alone with `scripts/convert_dialogue.py dialogue.txt -o dialogue_spoken.txt`.

You can roughly check whether the input reads as spoken language with the bundled scorer. It rates the dialogue from 0 to 1 based on utterance length and the proportions of short turns, fillers, and polite endings, and prints hints when the value is low (it is an indicator of input style, not a measure of synthesis quality):

```bash
uv run scripts/check_spontaneity.py dialogue.txt
```

> [!TIP]
> Symbols, digits, Latin characters, and proper nouns can yield unstable readings (G2P is Sudachi morphological segmentation + MFA Japanese dictionary lookup; spoken-form realization is handled by the raster generation models); plain kana/kanji text is recommended.

### Raster generation

For text input, the pipeline uses the realizer and gap model below to create a two-speaker phone raster (realized pronunciation, phone durations, intra-utterance pauses, gaps, and overlaps) and passes it to the acoustic model. The required models are downloaded automatically from Hugging Face on first use, so no extra flags are needed.

- **realizer**: decides the phones that are actually pronounced (including reductions and intra-utterance pauses) and how long each one lasts.
- **gap model**: decides when each utterance starts within the dialogue (pauses and overlaps).

### Statistical-placement mode

A mode that places utterances statistically, based on the timing statistics of the training corpus (pause/overlap distributions), instead of using the timing predictor (the "statistical placement" condition in the paper and demo page):

```bash
uv run scripts/kaburi_cli.py assets/sample_dialogue.txt --ref-pack assets/test_refpack/ --mode stat
```

### Synthesis with your own voices

Build a reference pack from any two reference recordings (speakers A/B, 3–10 s each, that you have permission to use):

```bash
uv run scripts/make_ref_pack.py --ref-a voiceA.wav --ref-b voiceB.wav --out-dir refpack/
uv run scripts/kaburi_cli.py assets/sample_dialogue.txt --ref-pack refpack/ -o out.wav
```

The model is trained on spontaneous conversational speech, so references closer to that condition work best. Clean read-aloud or studio recordings fall outside the training distribution and tend to sound stiff and flat. In that case, **bootstrap refinement** helps: synthesize one dialogue with KABURI and rebuild the references from its conversational output:

```bash
uv run scripts/refine_ref_pack.py --ref-pack refpack/ --out-dir refpack_refined/
uv run scripts/kaburi_cli.py assets/sample_dialogue.txt --ref-pack refpack_refined/ -o out.wav
```

The two bundled packs were built by generating non-existent voices with [Irodori-TTS VoiceDesign](https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign) and running `make_ref_pack.py` + `refine_ref_pack.py` above.

## Limitations

- Output quality is bounded by the DACVAE codec (32-dim, 25 fps).
- Words missing from the G2P dictionary (inflections, digit readings, etc.) may be unclear.
- Some prosodic unnaturalness remains.
- Raster generation is trained on casual free conversation, so formal or read-style input may come out sounding too casual. Intra-utterance pauses may occasionally land at an unnatural position.
- Speaker references work best when they resemble the training data: natural conversation recorded over an online meeting tool, about 10 seconds long, with the reference speaker talking for most of it. `make_ref_pack.py` automatically removes silences and adjusts references to a 10 s length, but studio-recorded read speech may still degrade quality.

## License of demo audio

Unlike the code, **the demo-page audio is not covered by Apache 2.0**. The
reference voices are taken from the
[JVS corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)
(The University of Tokyo); **please review the JVS distribution page for its terms of use**.
The dialogue-demo audio is synthesized by KABURI-TTS and the comparison systems. When forking
this repository for redistribution or commercial use, comply with the JVS terms or replace the
demo audio. The bundled test reference packs (assets/test_refpack, assets/test_refpack_mf)
consist solely of non-existent synthetic voices and are free of these restrictions. See
[docs/LICENSE-audio.md](docs/LICENSE-audio.md) for details.

## Safety considerations

These are not license terms, but requests and caveats for users.

- Please do not use KABURI-TTS for impersonation, fraud, or generating misinformation. See also the usage notes of the upstream [Irodori-TTS-500M-v2](https://huggingface.co/Aratako/Irodori-TTS-500M-v2).
- Model outputs may contain biases or inaccurate or offensive content derived from the training data.
- The developers assume no responsibility for any damages arising from its use.

The code and model weights are released under the [Apache License 2.0](LICENSE) (demo audio below is excluded). The bundled [irodori_tts/](irodori_tts/) package remains under its upstream MIT license ([irodori_tts/LICENSE](irodori_tts/LICENSE)).

## Acknowledgements

- Base TTS / codec: [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) (MIT), [Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim), [facebookresearch/dacvae](https://github.com/facebookresearch/dacvae)
- Text tokenizer (acoustic model & timing predictor): [llm-jp/llm-jp-3-150m](https://huggingface.co/llm-jp/llm-jp-3-150m)
- Text-converter base: [llm-jp/llm-jp-3-440m](https://huggingface.co/llm-jp/llm-jp-3-440m)
- G2P dictionary: [MFA japanese_mfa dictionary](https://mfa-models.readthedocs.io/) v3.0.0 (McAuliffe & Sonderegger, CC BY 4.0; bundled **unmodified** as `assets/g2p/japanese_mfa.dict`; see [NOTICE](NOTICE))
- Morphological analysis: [SudachiPy](https://github.com/WorksApplications/SudachiPy) + [SudachiDict](https://github.com/WorksApplications/SudachiDict) (Apache 2.0, used for G2P segmentation and readings)

## Citation

This work will appear as the following paper (**to appear**):

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

> [!NOTE]
> **Relation to the paper**: the acoustic model is unchanged. The current release replaces the paper's single timing predictor with a realizer (realized phones, phone durations, and intra-utterance pauses) and a gap model (gaps and overlaps). The quick-start audio and the synthesized samples on the [demo page](https://llm-jp.github.io/kaburi-tts) use the current release. Use `--paper-mode` for the paper pipeline; the corresponding code snapshot is tagged `paper-release-v1`.

The decode settings used by the current inference pipeline are in [assets/raster/decode_config.json](assets/raster/decode_config.json).
