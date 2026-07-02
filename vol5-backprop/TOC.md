# 第5巻: ニューラルネットと誤差逆伝播 — 自動微分を自作する(目次)

> ステータス: ドラフト v1(2026-06-10)
> ゴール: ミニ autograd(micrograd 相当)を自作し、MLP を学習させる。論文アーキテクチャ図の部品の大半(FFN、residual、layer norm、dropout)が揃う。

## ラスボス(巻頭掲示する論文箇所)

- Section 3.3 Position-wise Feed-Forward Networks 式(2):
  `FFN(x) = max(0, xW_1 + b_1)W_2 + b_2`
- Section 3.1: "...a residual connection around each of the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x))"
- Section 5.4 Regularization: "We apply dropout to the output of each sub-layer..."

巻頭で宣言: この巻を終えると、**アーキテクチャ図の attention 以外の箱(Feed Forward、Add & Norm、Dropout)が全部読める**。

---

## [序章 ラスボスとの対面](manuscript/00-prologue.md)

- 0.1 ここまでの到達点: 式(1)は読めた。だが論文の図1には他にも箱がたくさんある
- 0.2 論文原文の掲示(上記ラスボス): max(0, ・)? LayerNorm? dropout?
- 0.3 この巻のもう一つの主役: ここまで勾配は毎回手で導出してきた。**もう限界**。微分を自動化する
- 0.4 この巻で**扱わないこと**: CNN、RNN(第6巻で必要になってから)、初期化理論の一般論(Xavier等は使う分だけ)

## 巻頭付録 論文読解マップ(第5巻の現在地ハイライト)

---

## [第1章 線形の限界と活性化関数](manuscript/01-activations.md)

- 1.1 **[コード]** 線形モデルで解けないデータ(XOR的な配置)を見せる — 直線では割れない
- 1.2 層を重ねても無駄: 線形 ∘ 線形 = 線形(第1巻5.3「合成 = 行列積」の帰結)
- 1.3 間に非線形を挟む: 活性化関数。ReLU = max(0, x)、sigmoid(第3巻ぶりの再会)、tanh
- 1.4 ReLU の気持ち: 折れ線で何でも近似する(万能近似は直観+図のみ。証明はしない)
- 演習: ReLU 2個で「山」を作る

## [第2章 MLP — 線形層を重ねる](manuscript/02-mlp.md)

- 2.1 MLP の定義: `linear → 活性化 → linear → …`(第1巻6章の `linear(X, W, b)` がついに本来の役割で登場)
- 2.2 forward pass を shape で読む: 隠れ層の幅という設計変数
- 2.3 **[コード]** 2層MLPの forward を NumPy で実装、XOR的データの決定境界が曲がることを確認(パラメータはまだ乱数 or 手調整)
- 2.4 さて学習しよう → ∂L/∂W_1 の手導出をやってみる → **地獄**(やらせてから救う構成)
- 演習: 手導出を1回だけ最後までやる(これが最後の手導出になる)

## [第3章 誤差逆伝播 — 連鎖律を逆向きに流す](manuscript/03-backprop.md)

- 3.1 計算グラフの再登場(第2巻5章): forward で値を、backward で勾配を流す
- 3.2 鍵となる見方: 各ノードは「自分の局所勾配 × 上流から来た勾配」を下流に渡すだけ
- 3.3 2層MLPの backprop を図 + 行列で導出: δ の伝播、∂L/∂W = 入力^T @ δ(第1巻の転置がこんな所で効く)
- 3.4 なぜ「逆向き」なのか: forward-mode との比較を直観レベルで(パラメータが多く出力が1個だから逆が得)
- 演習: 3.3 の導出を数値微分(第2巻1章)で検算

## [第4章 ミニ autograd を自作する(クライマックス)](manuscript/04-autograd.md)

- 4.1 設計方針: 値と勾配と「親と演算」を覚える Value クラス(micrograd 相当、100行強)
- 4.2 **[コード]** 演算の実装: 加算・乗算・ReLU・exp・log… 各演算に backward を1個ずつ生やす
- 4.3 **[コード]** トポロジカルソートと backward(): グラフを逆順にたどる
- 4.4 **[コード]** テスト: すべての演算を数値微分と照合する(第2巻からの習慣の集大成)
- 4.5 これが PyTorch の `loss.backward()` の正体(構造は同じ、規模が違うだけ)
- 演習: 新しい演算(tanh等)を backward 付きで追加する

## [第5章 autograd で学習する — もう手で微分しない](manuscript/05-training-with-autograd.md)

- 5.1 **[コード]** Value ベースの linear 層・MLP を組み、訓練ループ(第3巻4章の4拍子そのまま)を回す
- 5.2 **[コード]** 第3巻の回帰、第4巻のロジスティック回帰・softmax 分類を autograd で再実装 — 過去2巻の手導出が全部自動で出ることを確認する(感動ポイント)
- 5.3 NumPy 行列版 autograd への拡張(スカラー Value のままでは遅い): テンソル対応の最小限
- 演習: 隠れ層の幅・深さを変えて決定境界を観察

## [第6章 深くするための道具たち — 論文の部品を先取りする](manuscript/06-deep-toolbox.md)

- 6.1 深くすると壊れる: 勾配消失を**観測**する(深いMLPで各層の勾配ノルムをプロット)
- 6.2 residual connection: x + f(x)。「迂回路」で勾配が素通りできる(backprop で +1 が効く、を式で)
- 6.3 layer norm: 各サンプルの活性を平均0・分散1に整える(第4巻2章の平均・分散がここで実装に化ける)+ スケール γ とシフト β
- 6.4 dropout: 訓練時だけランダムに消す。なぜ過学習に効くか(第3巻6章の続き)、推論時の扱い
- 6.5 **[コード]** residual / layer norm / dropout を実装し、深いネットで効果を比較実験
- 6.6 重みの初期化: 乱数なら何でもいいわけではない(分散の議論 — 第4巻7章と同じ計算が再登場)
- 演習: residual の有無で勾配ノルムの図を比較

## [終章 ラスボス再戦 — アーキテクチャ図の箱が読める](manuscript/07-boss-rematch.md)

- 7.1 式(2)を再読: `max(0, xW_1 + b_1)W_2 + b_2` = ReLU を挟んだ2層 linear。**5章で書いたコードそのもの**
- 7.2 `LayerNorm(x + Sublayer(x))` を再読: 6章の residual + layer norm の合成。dropout の一文も読める
- 7.3 図1の部品チェックリスト: 読める箱(Feed Forward、Add & Norm、Dropout、Softmax、Linear)/ まだの箱(Multi-Head Attention、Positional Encoding、Embedding)
- 7.4 次巻予告: 残りの箱の前に、そもそも**言語をどうやって数値にするのか**(第6巻)

---

## 論文の記号 ↔ 本巻の章 対応

| 論文の記号・概念 | 章 |
|---|---|
| FFN(x) = max(0, xW_1+b_1)W_2+b_2(式2) | 1・2章 |
| ReLU(max(0,・)) | 1章 |
| LayerNorm(x + Sublayer(x))(3.1) | 6章 |
| dropout(5.4, P_drop=0.1) | 6章 |
| d_ff = 2048, d_model = 512(FFNの幅) | 2章 |

## コードで示す箇所の方針

- コードを入れる: 本巻はシリーズ最多。XORの限界(1章)、MLP forward(2章)、autograd 本体とテスト(4章)、過去巻の再実装(5章)、勾配消失と部品の効果実験(6章)
- コードを入れない: backprop の行列導出(3章)は紙と図が主役(コードは4章で書くので、3章は理解に専念)
