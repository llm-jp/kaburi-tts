# KABURI-TTS

[![English](https://img.shields.io/badge/README-English-red.svg)](README-en.md)

📑 Paper (APSIPA ASC 2026, to appear) | [🤗 Model](https://huggingface.co/llm-jp/kaburi-tts) | [🖥️ Demo](https://llm-jp.github.io/kaburi-tts) | [🎧 Samples](docs/)

**KABURI-TTS** は、2 話者の日本語対話音声を左右 2 チャネルで同時に生成する対話音声合成システムです。相槌・重なり（**かぶり**）・間を含む対話特有のタイミング構造を、テキストのみから再現できます。

本成果は、国立情報学研究所 大規模言語モデル研究開発センター（NII LLMC）対話WGの活動として作成されたものです。

> [!NOTE]
> この `paper-release-v1` タグは、論文で評価した timing predictor 版を再現するための固定スナップショットです。通常の合成コマンドがそのまま論文版の経路を使用します。最新版は `main` ブランチを参照してください。

- **Acoustic model**: 事前学習済み単一話者 TTS（[Irodori-TTS-500M-v2](https://huggingface.co/Aratako/Irodori-TTS-500M-v2)）を LoRA + 部分 unfreeze で対話ドメインに適応した rectified flow モデル。音素系列・発話アクティビティ・話者参照音声を条件に、2 チャネルの音声 latent を生成し、[DACVAE codec](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim) で 48 kHz ステレオ波形に復号します。推論時は classifier-free guidance（cfg_scale=2.5）を使います。
- **Timing predictor**: 対話テキスト（音素系列 + 発話文脈）から、各発話の音素継続長・発話前無音・チャネル間 gap/overlap を予測する軽量 Transformer。予測タイミングを 2 チャネルの音素ラスタに展開し、acoustic model に渡します。

## セットアップ

CUDA GPU（推論は 1 GPU で可、bf16 で ~6GB 程度）を想定しています。[uv](https://docs.astral.sh/uv/) があれば Python 本体も含めて 2 コマンドで環境が整います:

```bash
git clone https://github.com/llm-jp/kaburi-tts.git
cd kaburi-tts
uv sync   # Python 3.10/3.11 と全依存 (CUDA 12.8 wheel) を自動構築
```

以降のコマンドは `python ...` の代わりに `uv run ...` で実行できます（venv の activate 不要)。

uv を使わない場合（Python 3.10/3.11 + pip）:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

モデル重みは初回実行時に [llm-jp/kaburi-tts](https://huggingface.co/llm-jp/kaburi-tts) から自動ダウンロードされます（acoustic 2.3GB を含む計 3GB 程度。初回のみ数分かかります。リポジトリが private の間のみ `huggingface-cli login` が必要です）。2 回目以降はキャッシュが使われ、30 秒対話 1 本の合成は 1〜2 分（単 GPU）です。

## 使い方

### まず試す（同梱のテスト用参照声）

テスト用に、**実在しない合成声**（Irodori-TTS VoiceDesign で生成した 2 声）の参照パックを [assets/test_refpack/](assets/test_refpack/) に同梱しています。対話テキスト（1 行 1 発話、`A: ...` / `B: ...` 形式。例: [assets/sample_dialog.txt](assets/sample_dialog.txt)）をそのまま合成できます。

```bash
uv run scripts/kaburi_cli.py assets/sample_dialog.txt --ref-pack assets/test_refpack/ -o out.wav
```

（pip 環境なら `uv run` を `python` に読み替えてください。以下同様）

左チャネルが話者 A、右チャネルが話者 B のステレオ wav が出力されます。

### 好きな声で合成する

話者参照として任意の音声 2 本（話者 A/B、各 3〜10 秒、収録許諾のあるもの）から参照パックを作れます（初回のみ）。

```bash
uv run scripts/make_ref_pack.py --ref-a voiceA.wav --ref-b voiceB.wav --out-dir refpack/
uv run scripts/kaburi_cli.py assets/sample_dialog.txt --ref-pack refpack/ -o out.wav
```

朗読調・スタジオ品質の参照では品質が落ちることがあります。その場合は **bootstrap 精錬**（一度 KABURI で対話を合成し、その出力から対話調の参照を作り直す）を推奨します:

```bash
uv run scripts/refine_ref_pack.py --ref-pack refpack/ --out-dir refpack_refined/
uv run scripts/kaburi_cli.py assets/sample_dialog.txt --ref-pack refpack_refined/ -o out.wav
```

同梱の `assets/test_refpack/` もこの方式で作られています。

> [!TIP]
> 記号・数字・英字・固有名詞は読みが不安定になりやすいため、ひらがな・カタカナ・常用漢字中心のテキストを推奨します（G2P は MFA 日本語辞書の最長一致）。

### 入力テキストは「話し言葉」に — テキストコンバータで自動変換

KABURI-TTS は自発対話音声で学習されているため、**入力テキストが実際の話し言葉に近いほど自然に合成されます**（短いターン・相槌・フィラー・砕けた終止形。書き言葉的な整った文だと品質が落ちます）。

書き言葉的な対話は、同梱の**テキストコンバータ**（llm-jp-3-440m を学習コーパスの書き起こしスタイルへ fine-tune したもの）で話し言葉に自動変換できます。長い文の分割・相槌の挿入・フィラーの付与まで行い、出力は学習時の書き起こし分布に一致します。**分量の目安は元テキスト 200〜250 字 ≒ 30 秒**です（合成は 30 秒が上限で、超えた分は全体が早口に圧縮されます。コンバータは文体を変えるだけで尺の調整はしません）:

```bash
# 合成と一括で（推奨）
uv run scripts/kaburi_cli.py dialog.txt --ref-pack refpack/ --convert

# 変換だけ行ってテキストを確認する場合
uv run scripts/convert_dialog.py dialog.txt -o dialog_spoken.txt
```

話し言葉らしさは同梱のチェッカで確認できます:

```bash
uv run scripts/check_spontaneity.py dialog.txt
```

コンバータを使わない場合は、[assets/spontaneous_rewrite_prompt.txt](assets/spontaneous_rewrite_prompt.txt) を任意の LLM に渡して書き直し、複数案をチェッカで比較する方法もあります（`kaburi_cli.py` も合成時に自動で警告します）。

コンバータの既知の制限: 固有名詞や数値表現がまれに崩れる・発話の話者帰属が軽くずれることがあります。変換結果はテキストで確認してから合成するのが確実です。

### 統計配置モード

タイミング予測器の代わりに、学習コーパスのタイミング統計（間・かぶりの分布）に基づいて発話を統計的に配置するモードです（論文・デモページの「統計配置」条件）:

```bash
uv run scripts/kaburi_cli.py assets/sample_dialog.txt --ref-pack assets/test_refpack/ --mode stat
```

## 制限事項

- 出力品質の上限は DACVAE codec（32-dim, 25 fps）に依存します。
- G2P 辞書に無い語（活用形・数字読みなど）は読みが不明瞭になることがあります。
- 韻律がやや不自然な箇所が残ることがあります。
- 話者参照は学習時の収録条件（会議ツール経由の自発対話、10 秒・高発話密度）に近いほど品質が安定します。`make_ref_pack.py` は無音除去と 10 秒への充填を自動で行いますが、スタジオ品質の朗読音声などでは品質が低下する場合があります。

## 利用規約

KABURI-TTS は研究目的で公開しています。なりすまし・詐欺をはじめとする悪用を禁止します。モデルの出力には学習データに由来する偏り・不正確または不適切な内容が含まれる可能性があります。本モデルの利用により生じたいかなる損害についても、開発者は責任を負いません。

コードおよびモデル重みは [Apache License 2.0](LICENSE) で公開しています（下記のデモ音声を除く）。同梱の [irodori_tts/](irodori_tts/) は上流の MIT ライセンスのままです（[irodori_tts/LICENSE](irodori_tts/LICENSE)）。

### デモ音声のライセンス

コードのライセンスと異なり、**デモページの音声は Apache 2.0 の対象外**です。デモの話者参照は [JVS corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)（東京大学）から作成しており、**アカデミック・非商用研究・個人利用に限り**利用できます（商用は東大 TLO との別途契約）。参照話者の声の商用利用・キャラクターボイス的な利用はできません。本リポジトリをフォークして商用利用する場合は、デモ音声を差し替えてください。詳細は [docs/LICENSE-audio.md](docs/LICENSE-audio.md) を参照。なお同梱のテスト用参照パック（assets/test_refpack）は実在しない合成声のみで構成されており、この制約とは無関係に使用できます。

## 謝辞

- Base TTS / codec: [Irodori-TTS](https://github.com/Aratako/Irodori-TTS)（MIT）, [Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim), [facebookresearch/dacvae](https://github.com/facebookresearch/dacvae)
- Tokenizer: [llm-jp/llm-jp-3-150m](https://huggingface.co/llm-jp/llm-jp-3-150m)
- G2P 辞書: [MFA japanese_mfa dictionary](https://mfa-models.readthedocs.io/)（CC BY 4.0、`assets/g2p/japanese_mfa.dict` として同梱）
- 謝辞・研究費情報は論文公開時に更新予定です。

## 引用

本成果は下記の論文として発表予定です（**to appear**）。

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
