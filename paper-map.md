# 論文読解マップ — "Attention Is All You Need" ↔ 本シリーズ対応表

> ステータス: 初版 v1(2026-06-10)。全巻共通の巻頭付録(設計原則5)。
> 使い方: 各巻の巻頭に同じ表を載せ、**その巻が担当する行をハイライト**して掲載する。読者はいつでも「論文のどこまで読めるようになったか」を確認できる。

## この表の読み方

- **読める化**: その箇所が「読める」状態になる巻・章。複数あるものは「部品 → 完成」の順
- 論文の式番号は原文準拠: 式(1) = attention、式(2) = FFN。Positional Encoding の sin/cos は原文では無番号だが、本シリーズでは便宜上「式(3)」と呼ぶ

---

## 第1部: 論文のセクション順 対応表

| 論文の箇所 | 内容・キーワード | 読める化(巻・章) |
|---|---|---|
| Abstract | "dispensing with recurrence and convolutions entirely" | 6巻で痛みを体感 → **7巻1章**で逐行 |
| 1 Introduction | "sequential nature precludes parallelization" | **6巻5章**(並列化不能を実測)→ 7巻1章 |
| 2 Background | "operations ... grows in the distance between positions" | **6巻5・7章**(長距離依存)→ 7巻1章 |
| 3 Model Architecture 冒頭 | "auto-regressive" | **6巻6章**(seq2seqと生成) |
| 3 図1(アーキテクチャ図) | 全体の見取り図 | 部品: 1・4・5巻 → 全体: **7巻2章** |
| 3.1 Encoder and Decoder Stacks | N=6、d_model=512 | **7巻2章** |
| 3.1 | LayerNorm(x + Sublayer(x)) | **5巻6章**(residual + layer norm)→ 7巻2章 |
| 3.2.1 式(1) | Q, K, V(行列として) | **1巻終章**(shape)→ 6巻7章(言葉の意味)→ 7巻3章 |
| 3.2.1 式(1) | QK^T(転置・行列積・内積=類似度) | **1巻2〜4章・終章** |
| 3.2.1 式(1) | softmax | **4巻6章** |
| 3.2.1 式(1) | √d_k(scaled の根拠) | **4巻7章** |
| 3.2.1 脚注4 | 内積の平均と分散("mean 0 and variance d_k") | **4巻2章**(道具)→ **4巻7章**(再現) |
| 3.2.1 | masking(−∞ を足す) | **7巻3.5**(伏線: 3巻5章 padding、6巻6.4 自己回帰) |
| 3.2.1 | additive vs dot-product の比較 | **7巻3.3**(伏線: 1巻4章ベンチマーク) |
| 3.2.2 | Multi-Head、W_i^Q / W_i^K / W_i^V / W^O、h=8 | **1巻5.5**(shapeだけ)→ **7巻4章** |
| 3.2.2 | テンソル整形(batch, h, seq, d_k) | **1巻6.4**(行列の束)→ **7巻4.3** |
| 3.2.3 | attention の3つの使い方(self / masked self / cross) | **6巻7章**(crossの原型)→ **7巻5章** |
| 3.3 式(2) | FFN(x) = max(0, xW_1+b_1)W_2+b_2、d_ff=2048 | **5巻1・2章・終章** → 7巻6章("position-wise") |
| 3.4 | learned embeddings | **6巻3章** |
| 3.4 | 出力側 linear + softmax、weight sharing、√d_model 倍 | 4巻6章(softmax)→ **7巻6章** |
| 3.5 式(3) | Positional Encoding(sin/cos) | **7巻7章**(伏線: 6巻終章「語順はどこへ?」) |
| 4 Why Self-Attention / Table 1 | 計算量・逐次操作数・最大経路長 | **7巻8章**(伏線: 6巻の3つの痛み) |
| 5 Training 冒頭 | "training regime" という営みそのもの | **3巻全体** |
| 5.1 | training data、"batched together by approximate sequence length" | **3巻5章**(batch)→ **8巻2章**(実装) |
| 5.1 | byte-pair encoding | **6巻2章** |
| 5.2 | Hardware and Schedule | **8巻6章**(training cost と合わせて概観) |
| 5.3 | Adam(β1=0.9, β2=0.98, ε) | **8巻4章** |
| 5.3 | lrate の式、warmup_steps | **2巻3・6章・終章**(直観)→ **8巻4.6**(実装) |
| 5.4 | dropout(P_drop=0.1) | **5巻6.4** |
| 5.4 | label smoothing(ε_ls=0.1)、"This hurts perplexity" | 4巻5章(cross-entropy)・4巻終章(予告)→ **8巻3.2** |
| 6.1 / Table 2 | BLEU、training cost(FLOPs)、beam search | **8巻5・6章** |
| 6.2 / Table 3 | アブレーション(h、d_k、dropout、学習PE) | **8巻6.4**(各行を1〜7巻の議論に接続) |
| 6.3 | 構文解析への一般化 | **8巻6.5**(概観のみ) |
| 7 Conclusion | まとめと展望 | **8巻終章**(全文通読の一部として) |

---

## 第2部: 記号・用語インデックス(アルファベット・五十音順)

| 記号・用語 | 読める化(巻・章) |
|---|---|
| Adam(β1, β2, ε) | 8巻4章 |
| auto-regressive | 6巻6章 |
| batch / batching | 3巻5章 |
| BLEU | 8巻6.1 |
| BPE(byte-pair encoding) | 6巻2章 |
| cross-entropy | 4巻5章 |
| d_model, d_k, d_v, d_ff(次元の記号) | 1巻(shapeとして)→ 各部品の章 |
| dropout(P_drop) | 5巻6.4 |
| embedding | 6巻3章 |
| FFN / ReLU(max(0,·)) | 5巻1・2章 |
| KL divergence | 4巻5.4 |
| label smoothing(ε_ls) | 8巻3.2(前提: 4巻5章) |
| layer norm | 5巻6.3(前提: 4巻2章の平均・分散) |
| learning rate / lrate / warmup | 2巻3・6章 → 8巻4.6 |
| loss(損失関数) | 3巻3章 |
| masking(−∞) | 7巻3.5 |
| multi-head / h | 7巻4章 |
| perplexity | 4巻7.5 → 6巻1.4 |
| positional encoding(sin/cos) | 7巻7章 |
| Q, K, V | 1巻終章(行列)→ 6巻7.2(言葉)→ 7巻3章(完全) |
| QK^T | 1巻4章・終章 |
| residual connection(x + Sublayer(x)) | 5巻6.2 |
| softmax | 4巻6章 |
| √d_k | 4巻7章 |
| 転置(K^T の T) | 1巻3.3 |
| 内積(dot-product) | 1巻2章 |
| 尤度 / 最尤推定 | 4巻3章 |

---

## 第3部: 巻ごとの「読める化」サマリ(難易度カーブの確認用)

| 巻 | この巻で新たに読めるようになるもの |
|---|---|
| 1巻 | 式(1)の**行列構造**(Q・K・V のshape、QK^T、転置)、3.2.2の射影のshape |
| 2巻 | lrate・warmup の**意図**(5.3の式の形) |
| 3巻 | "training"・"batching"・loss という**営み全体**(Section 5 の語彙) |
| 4巻 | softmax・√d_k(脚注4)→ **式(1)が完全に読める**。perplexity |
| 5巻 | 式(2)FFN、LayerNorm(x+Sublayer(x))、dropout → **図1の attention 以外の箱** |
| 6巻 | **Abstract・Introduction・Background 全文**、BPE、embeddings、Q/K/V の言葉 |
| 7巻 | **Section 3・4 の全文 + 全部品の実装**(式番号↔コード行 対応) |
| 8巻 | **Section 5・6・7** → 論文全文通読(卒業) |

---

## メンテナンス規約

- 各巻の TOC(`volN-*/TOC.md`)末尾の対応表が一次情報。本ファイルはその統合ビュー。**TOC を変更したら本ファイルも更新する**
- 原稿執筆が進んだら「巻・章」を「巻・章・節 + コードのファイル・行」まで細分化する(7巻終章・8巻終章の成果物と合流させる)
