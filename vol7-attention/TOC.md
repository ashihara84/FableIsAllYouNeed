# 第7巻: Attention Is All You Need 精読(目次)

> ステータス: ドラフト v1(2026-06-10)
> ゴール: 論文をセクション順に1行ずつ読み、各コンポーネントを単体実装 + テストする。「論文の式番号 ↔ 自分のコードの行」が1対1対応する状態。
> 役割: ラスボス戦本番。この巻に新しい数学は登場しない(すべて1〜6巻で装備済み)。

## ラスボス(巻頭掲示する論文箇所)

- **論文全体**。特に Section 3 Model Architecture と図1。

巻頭で宣言: これまでの巻は「論文の一部」を掲げてきた。この巻のラスボスは論文そのもの。ただし戦い方が変わる — **もう装備は全部ある**。確認しながら進む巻である。

---

## 序章 装備の点検

- 0.1 論文読解マップの総点検: 1〜6巻で何がどこまで読めるようになったかを一覧で確認(読めない箇所が Section 3.2 の組み立てと 3.5 だけであることを見せる)
- 0.2 精読の作法: 各章の構成は「原文 → 逐行読解 → 単体実装 → テスト → 式番号↔コード行の対応表」で固定
- 0.3 実装の約束: NumPy + 第5巻 autograd。各部品は独立にテスト可能に(第8巻で組み立てるため、入出力 shape を厳密に決める)

## 巻頭付録 論文読解マップ(完成直前版)

---

## 第1章 Abstract・Introduction・Background を読む

- 1.1 Abstract 逐行: "dispensing with recurrence and convolutions entirely" — 第6巻の問いへの宣戦布告として読む
- 1.2 Introduction / Background: 第6巻の体感と突き合わせる高速再読
- 1.3 関連研究の固有名詞(ByteNet, ConvS2S 等)の扱い: 深追いしない宣言(最短測地線)
- 演習: Abstract を自分の言葉で和訳する(巻末にもう一度やる)

## 第2章 3.1 Encoder and Decoder Stacks — 全体の見取り図

- 2.1 図1を地図として読む: encoder 側・decoder 側、N=6 の積み重ね
- 2.2 sub-layer の規約: LayerNorm(x + Sublayer(x))(第5巻6章の回収)、全部 d_model = 512 で揃える理由(residual のため)
- 2.3 decoder の3つ目の入力: encoder の出力がどこに刺さるか(詳細は5章へ)
- 2.4 **[コード]** 中身が恒等写像のダミー Sublayer で、stack の骨組みと shape の流れだけ先に実装(外側から作る方針)
- 演習: 図1を白紙に再現する

## 第3章 3.2.1 Scaled Dot-Product Attention — 式(1)の実装

- 3.1 原文逐行: query / key / value(第6巻7章の言葉)、"compatibility function" の意味
- 3.2 式(1)を shape で読む: (n, d_k) @ (d_k, m) → (n, m) → softmax → @ (m, d_v) → (n, d_v)。第1巻終章の再演 + 今や全記号の意味がわかる
- 3.3 additive attention との比較の段落: なぜ dot-product を選ぶか(速い = 第1巻4章のベンチマーク、√d_k で欠点を消す = 第4巻7章)
- 3.4 **[コード]** `attention(Q, K, V, mask)` の単体実装
- 3.5 masking: softmax の前に −∞ を足すという仕掛け
  - padding mask(バッチ処理の都合 — 第3巻5章のバッチが伏線)
  - causal mask(decoder の自己回帰性 — 第6巻6.4が伏線)
- 3.6 **[コード]** テスト: shape、重みの和が1、mask した位置の重みが0、手計算の小例と一致
- 演習: 2×2 の小さな Q, K, V で式(1)を手計算 → コードと照合

## 第4章 3.2.2 Multi-Head Attention — 視点を増やす

- 4.1 原文逐行: なぜ1回の attention では足りないか("averaging inhibits this")
- 4.2 仕掛け: d_model を h=8 個の頭に分割し、それぞれ別の射影 W_i^Q, W_i^K, W_i^V で別の「見方」を学ぶ(第1巻5.5の伏線回収)
- 4.3 テンソル整形の実務: (seq, d_model) → (h, seq, d_k) の reshape/transpose(batch 軸は先頭に1本足すだけで同じコードが動く。第1巻6.4「行列の束」がついに本番)
- 4.4 **[コード]** multi-head attention の単体実装 + テスト(h=1 で 3章と一致することも確認)
- 4.5 計算コストの確認: 分割しても総コストはほぼ同じ、という原文の主張を式で
- 演習: head ごとの attention マップを可視化して「見方の違い」を観察

## 第5章 3.2.3 — attention の3つの使い方

- 5.1 原文逐行: encoder self-attention / decoder masked self-attention / encoder-decoder(cross)attention
- 5.2 self-attention とは Q=K=V=自分自身: 「文が文自身を見る」(第6巻終章の問いの答え)
- 5.3 cross-attention は第6巻7章の attention 付き seq2seq と同型であることの確認(Q だけ decoder から)
- 5.4 **[コード]** 3つの使い方を呼び分けるテスト(mask と入力の組合せが正しいか)
- 演習: 3つの attention の Q, K, V がそれぞれどこから来るか、図1に書き込む

## 第6章 3.3 FFN・3.4 Embeddings and Softmax

- 6.1 式(2)の逐行: 第5巻終章で読了済み — "position-wise"(各位置に同じMLPを独立適用)だけが新情報。**[コード]** 実装 + テスト
- 6.2 3.4 逐行: 入力埋め込み(第6巻3章)、出力側の linear + softmax(第4巻6章)
- 6.3 weight sharing: 入力埋め込み・出力射影で重みを共有する話と √d_model 倍の補正
- 6.4 **[コード]** embedding 層と出力 head の実装 + テスト
- 演習: パラメータ数を数えて論文の 65M と突き合わせる(base model)

## 第7章 3.5 Positional Encoding — 語順を取り戻す

- 7.1 問題の確認: attention は集合演算 — 並べ替えても結果が同じ(**[コード]** 実験で示す)。第6巻終章の問い1(語順)
- 7.2 原文逐行: sin / cos の式(3)。波長が幾何級数で並ぶ設計
- 7.3 なぜこの形か: 相対位置が線形変換で表せるという原文の主張を、小さな例と図で確かめる(厳密証明はしない)
- 7.4 **[コード]** positional encoding の実装 + ヒートマップ可視化(縞模様の図)。埋め込みに「足す」という選択
- 7.5 learned positional embedding との比較の段落: ほぼ同じ性能、外挿への期待で sin/cos を選んだ
- 演習: PE ベクトル同士の内積を位置差ごとにプロットする

## 第8章 Section 4 Why Self-Attention — 比較表を読む

- 8.1 Table 1 を読む: 層あたり計算量・逐次操作数・最大経路長の3指標
- 8.2 各行を検算: self-attention は O(n²·d) だが逐次 O(1)・経路長 O(1)、RNN は逐次 O(n)(第6巻の痛みが表の数字になっている)
- 8.3 n² 問題: 系列が長いと自乗で効く(現代の長文脈研究への入り口として一言。深追いしない)
- 演習: 自分の実装で n を変えて実行時間を測り、表の傾向を確認

## 終章 完走確認 — 式番号 ↔ コード行 対応表の完成

- 9.1 Section 3 を通しで再読: 止まらずに読めることを確認する儀式
- 9.2 成果物: 「論文の式・主張 ↔ 自分のコードのファイル・行」完全対応表(本シリーズの卒業証書の半分)
- 9.3 部品チェックリスト最終版: 全部品が単体テスト済みであることの確認
- 9.4 まだやっていないこと: **組み立てて訓練する**(Section 5・6)。部品が動くことと、全体が学習することは別問題 — 次巻へ

---

## 論文セクション ↔ 本巻の章 対応(この巻は構成自体が対応表)

| 論文 | 章 |
|---|---|
| Abstract, 1 Introduction, 2 Background | 1章 |
| 3.1 Encoder and Decoder Stacks | 2章 |
| 3.2.1 Scaled Dot-Product Attention(式1) | 3章 |
| 3.2.2 Multi-Head Attention | 4章 |
| 3.2.3 Applications of Attention | 5章 |
| 3.3 FFN(式2)・3.4 Embeddings and Softmax | 6章 |
| 3.5 Positional Encoding(式3) | 7章 |
| 4 Why Self-Attention(Table 1) | 8章 |
| 5 Training・6 Results | (第8巻へ) |

## コードで示す箇所の方針

- 全章「単体実装 + テスト」がノルマ(この巻だけは例外的にコード必須。精読の検証手段がコードであるため)
- ただし 7.3(相対位置の線形性)や 4.1(averaging の害)など、原文の「主張の理由」は図と小例が主役で、コードは確認に徹する
