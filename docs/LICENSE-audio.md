# デモ音声のライセンス / License of demo audio

本リポジトリのコードは Apache License 2.0 ですが、**デモページの音声はその対象外**です。

## 参照音声（audio/voices/ および各デモ音声の話者参照）

- デモの話者参照は **JVS corpus**（東京大学 猿渡研究室)の音声から作成しています。
- JVS の音声データは**アカデミック研究・非商用研究・個人利用に限り**利用可能です。
  商用利用には東京大学 TLO との別途契約が必要です。詳細は配布ページをご覧ください:
  https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus
- 本ページで公開する JVS 由来の音声ファイルは、JVS 規約が認める「一部公開」の範囲
  （10 ファイル程度以内）に収めています。JVS 本体の再配布は行いません。
- **参照話者の声を商用利用したり、キャラクターボイス等として利用することはできません。**
  本リポジトリをフォークして商用利用する場合は、これらの音声を差し替えてください。

## 合成音声（audio/chunk_input/, audio/text_input/）

- 各デモ音声は本システムによる合成出力です。話者性は上記 JVS 参照に由来するため、
  同様に非商用・研究デモの範囲でのみ公開しています。

## 引用 / Citation

> S. Takamichi, K. Mitsui, Y. Saito, T. Koriyama, N. Tanji, and H. Saruwatari,
> "JVS corpus: free Japanese multi-speaker voice corpus," arXiv:1908.06248, 2019.

---

The code in this repository is licensed under Apache License 2.0, but **the demo audio
is NOT covered by it**. Speaker references are derived from the JVS corpus (The
University of Tokyo), which permits academic / non-commercial research and personal
use only; commercial use requires a separate agreement with UTokyo TLO. JVS-derived
audio published here is kept within the "partial release" allowance (~10 files), and
the corpus itself is never redistributed. The reference speakers' voices must not be
used commercially or as character voices; replace these audio assets when forking
this repository for commercial purposes.
