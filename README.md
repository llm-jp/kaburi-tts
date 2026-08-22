<p align="center"><img src="docs/kaburi-tts-logo.png" alt="KABURI-TTS" width="440"></p>

# KABURI-TTS

[![English](https://img.shields.io/badge/README-English-red.svg)](README-en.md)

📑 Paper (APSIPA ASC 2026, to appear) | [🤗 Model](https://huggingface.co/llm-jp/kaburi-tts) | [🖥️ Demo](https://llm-jp.github.io/kaburi-tts)

**KABURI-TTS** は、2 話者の日本語対話音声を左右 2 チャンネルで同時に生成する対話音声合成システムです。相槌・重なり（**かぶり**）・間を含む対話特有のタイミング構造を、テキストのみから再現できます。

本成果は、国立情報学研究所 大規模言語モデル研究開発センター（NII LLMC）対話WGの活動として作成されたものです。

システムは次の 2 つの構成要素からなります:

- **Acoustic model**: 事前学習済み単一話者 TTS（[Irodori-TTS-500M-v2](https://huggingface.co/Aratako/Irodori-TTS-500M-v2)）を、NII LLMC が構築した 2 話者日本語自由対話コーパス **LLM-jp-Zoom1**で対話ドメインに適応した rectified flow モデル（LoRA + 部分 unfreeze）。音素系列・発話アクティビティ・話者参照音声を条件に、2 チャンネルの音声 latent を生成し、[DACVAE codec](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim) で 48 kHz ステレオ波形に復号します。推論時は classifier-free guidance（cfg_scale=2.5）を使います。
- **Raster generation**: LLM-jp-Zoom1 の実測タイミングを学習した realizer（実際に発音される音素・音素長・発話内ポーズ）と gap model（間・かぶり）を用いて、テキストから 2 話者分の音素ラスタを作り、acoustic model に渡します。

## セットアップ

CUDA GPU（推論は 1 GPU で可、bf16 で ~6GB 程度）を想定しています。[uv](https://docs.astral.sh/uv/) があれば Python 本体も含めて 2 コマンドで環境が整います:

```bash
git clone https://github.com/llm-jp/kaburi-tts.git
cd kaburi-tts
uv sync
```

モデル重みは初回実行時に [llm-jp/kaburi-tts](https://huggingface.co/llm-jp/kaburi-tts) などから自動ダウンロードされます（派生元の Irodori-TTS と codec を含む計 5GB 程度。`--convert` の初回利用時は text converter 約 1.8GB が別途必要です）。2 回目以降はキャッシュが使われ、30 秒対話 1 本の合成は 1〜2 分（単 GPU）です。

## 使い方

### まず試す

テスト用に、**実在しない合成声**（Irodori-TTS VoiceDesign で生成した声）の参照パックを 2 種類同梱しています。いずれも JVS などの実在話者を含まないため、ライセンス制約なく利用できます。

- [assets/test_refpack/](assets/test_refpack/) … 女性 2 声（デフォルト）
- [assets/test_refpack_mf/](assets/test_refpack_mf/) … 男性・女性の 2 声

対話テキスト（1 行 1 発話、`A: ...` / `B: ...` 形式。例: [assets/sample_dialogue.txt](assets/sample_dialogue.txt)）をそのまま合成できます。

```bash
uv run scripts/kaburi_cli.py assets/sample_dialogue.txt --ref-pack assets/test_refpack/ -o out.wav
```

（pip 環境なら `uv run` を `python` に読み替えてください。以下同様）

左チャンネルが話者 A、右チャンネルが話者 B のステレオ wav が出力されます。

同梱の [assets/sample_dialogue.txt](assets/sample_dialogue.txt)（約 30 秒の対話）を入力すると、次のような音声が得られます:

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

この合成結果（ステレオ、相槌や重なりを含む）は [女性 2 声](docs/audio/quickstart/sample_dialogue_female.mp3) と [男性・女性](docs/audio/quickstart/sample_dialogue_mf.mp3) の音声ファイルとして同梱しています。

### 書き言葉を話し言葉に変換する（--convert）

KABURI-TTS は自発対話音声で学習されているため、**入力テキストが実際の話し言葉に近いほど自然に合成されます**。

書き言葉的な対話は、同梱の**テキストコンバータ**（llm-jp-3-440m を LLM-jp-Zoom1 の書き起こしスタイルへ fine-tune したもの）で話し言葉に自動変換できます。長い文の分割・相槌の挿入・フィラーの付与まで行います。**分量の目安は元テキスト 200〜250 字 ≒ 30 秒**です（合成は 30 秒が上限で、超えた分は全体が早口に圧縮されます。コンバータは文体を変えるだけで尺の調整はしません）:

```bash
uv run scripts/kaburi_cli.py dialogue.txt --ref-pack assets/test_refpack/ --convert
```

変換結果を先にテキストで確認したい場合は、`scripts/convert_dialogue.py dialogue.txt -o dialogue_spoken.txt` で変換だけを実行できます。

入力が話し言葉寄りかどうかは、同梱のチェッカで簡易的に確認できます。発話長・短いターン・フィラー・丁寧体の割合から 0〜1 の目安値を出し、低いときは改善のヒントを表示します（入力文体の目安であり、合成品質そのものを測る指標ではありません）:

```bash
uv run scripts/check_spontaneity.py dialogue.txt
```

> [!TIP]
> 記号・数字・英字・固有名詞は読みが不安定になりやすいため、ひらがな・カタカナ・常用漢字中心のテキストを推奨します（G2P は Sudachi 形態素分割 + MFA 日本語辞書引き。実発音への変換はラスタ生成モデルが行います）。

### 音素ラスタ生成

テキスト入力では、以下の realizer と gap model を用いて、テキストから 2 話者分の音素ラスタ（実際の発音・音素長・発話内ポーズ・間・かぶり）を作り、音響モデルに渡します。必要なモデルは初回に Hugging Face から自動ダウンロードされるため、特別な指定は不要です。

- **realizer**: 実際に発音される音素列（縮約や発話内ポーズを含む）と、各音素の長さを決めます。
- **gap model**: 各発話を対話のどのタイミングで始めるか（間・かぶり）を決めます。

### 統計配置モード

タイミング予測器の代わりに、学習コーパスのタイミング統計（間・かぶりの分布）に基づいて発話を統計的に配置するモードです（論文・デモページの「統計配置」条件）:

```bash
uv run scripts/kaburi_cli.py assets/sample_dialogue.txt --ref-pack assets/test_refpack/ --mode stat
```

### 好きな声で合成する

話者参照として任意の音声 2 本（話者 A/B、各 3〜10 秒、利用許諾のあるもの）から参照パックを作れます（初回のみ）。

```bash
uv run scripts/make_ref_pack.py --ref-a voiceA.wav --ref-b voiceB.wav --out-dir refpack/
uv run scripts/kaburi_cli.py assets/sample_dialogue.txt --ref-pack refpack/ -o out.wav
```

本モデルは自発的な対話音声で学習しているため、参照音声もそれに近いほど安定します。逆に、朗読調やスタジオ収録のように整いすぎた音声は学習データと分布が異なり、声が硬く・平坦になりやすいです。その場合は **bootstrap refinement**（一度 KABURI で対話を合成し、その対話調の出力から参照を作り直す）で改善できます:

```bash
uv run scripts/refine_ref_pack.py --ref-pack refpack/ --out-dir refpack_refined/
uv run scripts/kaburi_cli.py assets/sample_dialogue.txt --ref-pack refpack_refined/ -o out.wav
```

同梱の 2 種類の参照パックは、[Irodori-TTS VoiceDesign](https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign) で実在しない声を生成し、上記の `make_ref_pack.py` + `refine_ref_pack.py` で作成したものです。

## 制限事項

- 出力品質の上限は DACVAE codec（32-dim, 25 fps）に依存します。
- G2P 辞書に無い語（活用形・数字読みなど）は読みが不明瞭になることがあります。
- 韻律がやや不自然な箇所が残ることがあります。
- 音素ラスタ生成はくだけた自由対話で学習しているため、硬い文体や朗読調の入力では発音がくだけすぎることがあります。発話内のポーズが、まれに不自然な位置に入ることもあります。
- 話者参照は、学習データと近い条件の音声ほど品質が安定します（オンライン会議で録音した自然な会話で、10 秒程度、その中でよく話している音声）。`make_ref_pack.py` が無音の除去と 10 秒への長さ調整を自動で行いますが、スタジオ録音の朗読音声などでは品質が下がることがあります。

## 利用上の注意

以下はライセンス条件ではなく、利用にあたってのお願いと注意です。

- なりすまし・詐欺・誤情報の生成といった悪用はしないでください。派生元の [Irodori-TTS-500M-v2](https://huggingface.co/Aratako/Irodori-TTS-500M-v2) の利用上の注意も併せてご確認ください。
- モデルの出力には、学習データに由来する偏りや、不正確・不適切な内容が含まれる可能性があります。
- 本モデルの利用により生じたいかなる損害についても、開発者は責任を負いません。

コードおよびモデル重みは [Apache License 2.0](LICENSE) で公開しています（下記のデモ音声を除く）。同梱の [irodori_tts/](irodori_tts/) は上流の MIT ライセンスのままです（[irodori_tts/LICENSE](irodori_tts/LICENSE)）。

### デモ音声のライセンス

コードのライセンスと異なり、**デモページの音声は Apache 2.0 の対象外**です。話者参照には [JVS corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)（東京大学）を使用しており、**その利用条件は JVS の配布ページをご確認ください**。対話デモ音声は KABURI-TTS および比較システムによる合成です。本リポジトリをフォークして再配布・商用利用する場合は、JVS の条件に従うか、デモ音声を差し替えてください。なお同梱のテスト用参照パック（assets/test_refpack, assets/test_refpack_mf）は実在しない合成声のみで構成されており、これらの制約とは無関係に利用できます。詳細は [docs/LICENSE-audio.md](docs/LICENSE-audio.md) を参照。

## 謝辞

- Base TTS / codec: [Irodori-TTS](https://github.com/Aratako/Irodori-TTS)（MIT）, [Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim), [facebookresearch/dacvae](https://github.com/facebookresearch/dacvae)
- テキスト tokenizer（音響モデル・タイミング予測器）: [llm-jp/llm-jp-3-150m](https://huggingface.co/llm-jp/llm-jp-3-150m)
- テキストコンバータのベース: [llm-jp/llm-jp-3-440m](https://huggingface.co/llm-jp/llm-jp-3-440m)
- G2P 辞書: [MFA japanese_mfa dictionary](https://mfa-models.readthedocs.io/) v3.0.0（McAuliffe & Sonderegger, CC BY 4.0。`assets/g2p/japanese_mfa.dict` として**改変せず**同梱。詳細は [NOTICE](NOTICE)）
- 形態素解析: [SudachiPy](https://github.com/WorksApplications/SudachiPy) + [SudachiDict](https://github.com/WorksApplications/SudachiDict)（Apache 2.0、G2P の分割・読みに使用）

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

> [!NOTE]
> **論文との関係**: 音響モデルは論文と同一ですが、最新版では論文版の単一 timing predictor を realizer（実発音・音素長・発話内ポーズ）と gap model（間・かぶり）に置き換えています。本 README のクイックスタート音声と[デモページ](https://llm-jp.github.io/kaburi-tts)の合成音声は最新版で生成しています。論文版は `--paper-mode`、対応するコード一式は `paper-release-v1` タグで再現できます。

最新版の推論で使用する decode 設定は [assets/raster/decode_config.json](assets/raster/decode_config.json) にあります。
