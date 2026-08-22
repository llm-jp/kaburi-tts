# デモ音声のライセンス / License of demo audio

本リポジトリのコードは Apache License 2.0 ですが、**デモページの音声はその対象外**です。

## 直接掲載する JVS 参照音声（audio/voices/）

- 「話者参照に使用した音声」欄の **8 ファイル**は、**JVS corpus**（東京大学 猿渡研究室）の
  各話者 1 発話の単一クリップ（parallel100、4〜8 秒）です。
- **これらの音声の利用条件は JVS の配布ページに従います。ご利用の前に必ず配布元の規約をご確認ください:**
  https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus
- 本リポジトリでは **JVS コーパス本体の再配布は行わず**、各話者 1 発話のみを掲載しています。
  本リポジトリをフォークして再配布・商用利用する場合は、JVS の条件に従うか、これらの音声を差し替えてください。

## 合成音声（audio/chunk_input/, audio/text_input/）

- 対話デモの音声は **KABURI-TTS および比較システム（MOSS-TTSD / FireRedTTS-2）による合成出力**です。
  話者性は上記 JVS 話者に由来するため、これらの音声についても JVS の配布条件が適用されます。

## 合成に用いた話者参照の構成方法

- 合成の話者参照は、JVS parallel100 の単一クリップを起点に、無音除去・10 秒への充填（`prepare_jvs_refs`）を行い、
  さらに会話ドメインへ精錬（`refine_ref_pack`: 一度対話を合成し会話調クリップを抽出、計 12〜18 秒）した参照です。
  **この精錬済み参照そのものは公開していません**（合成の内部入力）。voices 欄に掲載しているのは起点の JVS 単一発話クリップです。

## 引用 / Citation

> S. Takamichi, K. Mitsui, Y. Saito, T. Koriyama, N. Tanji, and H. Saruwatari,
> "JVS corpus: free Japanese multi-speaker voice corpus," arXiv:1908.06248, 2019.

---

The code in this repository is licensed under Apache License 2.0, but **the demo audio is NOT covered by it**.

- **Directly published JVS reference audio (audio/voices/):** the 8 files in the "Reference voices" section are
  **single-utterance clips (one per speaker, 4–8 s from parallel100)** from the JVS corpus (The University of Tokyo).
  **The terms of use for these clips follow the JVS distribution page — please review the original terms before use:**
  https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus
  This repository **does not redistribute the corpus itself** and publishes only one utterance per speaker. When forking
  this repository for redistribution or commercial use, comply with the JVS terms or replace these audio files.
- **Synthesized audio (audio/chunk_input/, audio/text_input/):** produced by **KABURI-TTS and the comparison systems
  (MOSS-TTSD / FireRedTTS-2)**. Since the speaker identity derives from the JVS speakers above, the JVS distribution
  terms also apply to these clips.
- **How the synthesis references were built:** speaker references used for synthesis start from a single JVS
  parallel100 clip, are silence-trimmed and padded to 10 s (`prepare_jvs_refs`), then refined toward the
  conversational domain (`refine_ref_pack`, ~12–18 s). These refined references are **not published** (internal
  synthesis input); the voices section shows only the originating single-utterance JVS clips.
