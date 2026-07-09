# 第6章 3.3 FFN・3.4 Embeddings and Softmax

> [目次](../TOC.md) ・ [← 前の章](05-three-attentions.md) ・ [次の章 →](07-positional-encoding.md)

図1の本体で残る箱は1つ、各ブロックの2つ目の sub-layer "Feed Forward" です。本体の外には、入口の "Input/Output Embedding" と出口の "Linear + Softmax" が残っています。

この章で読む Section 3.3 と 3.4 に、初めて見るものはほとんどありません。式(2)は第5巻終章で読了済み、埋め込みは第6巻3章、出力側の softmax は第4巻6章でやりました。未読は2か所だけ——3.3 の見出しの "Position-wise" という1語、そして 3.4 後半の**重みの共有(weight sharing)**と **$\sqrt{d_{model}}$ 倍の補正**です。論文は短く、ほとんど説明しません。この章はその行間を、これまでの巻の道具で埋めます。

## 6.1 式(2)の逐行: 第5巻終章で読了済み — "position-wise"(各位置に同じMLPを独立適用)だけが新情報。[コード] 実装 + テスト

Section 3.3 は、たったこれだけです。

> *"In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between."*
> $$\mathrm{FFN}(x) = \max(0,\ xW_1 + b_1)W_2 + b_2 \tag{2}$$
> — Vaswani et al., "Attention Is All You Need", Section 3.3
>
> 訳: 「attention sub-layer に加えて、encoder と decoder の各層は全結合の feed-forward ネットワークを含む。これは**各位置に対して、別々に、かつ同一に適用される**。中身は、間に ReLU を挟んだ2つの線形変換である。」

式(2)は第5巻終章で1記号ずつ読みました。要点だけ再掲します。$x$ `(1, 512)` を $W_1$ `(512, 2048)` で4倍の幅に広げ、ReLU で折り曲げ、$W_2$ `(2048, 512)` で元の幅に戻します。**linear → ReLU → linear、隠れ層が1つのただの MLP**(第5巻2章)です。入口と出口が同じ $d_{model} = 512$ なのは、residual で $x$ と足すため(第2章の sub-layer の規約)でした。

第5巻終章で残した宿題が "Position-wise"——「位置ごとに」です。あのときは「位置」が何かを言えませんでした。いまは言えます。この sub-layer に流れてくる $X$ `(seq, 512)` の**各行は、文の中の1つの位置(トークン)**です(第6巻)。原文の "applied to each position separately and identically" はこう読めます。

- **separately(別々に)**: 行ごとに独立に。位置 $i$ の出力は、位置 $i$ の入力だけから決まる。隣の行は見ない
- **identically(同一に)**: ただし全部の行に、**同じ** $W_1, b_1, W_2, b_2$ を使う

「各行に独立に、同じ MLP を適用する」に、行ごとの for ループは要りません。第1巻4章の行列積の性質——$XW_1$ の第 $i$ 行は $X$ の第 $i$ 行と $W_1$ だけから決まる——により、`(seq, 512) @ (512, 2048)` という1回の行列積が、**最初から「各行に同じ変換を独立適用」する演算**です。shape で書けば、

$$ \underbrace{X}_{(seq,\ 512)} \;\to\; \underbrace{\max(0,\ XW_1 + b_1)}_{(seq,\ 2048)} \;\to\; \underbrace{(\cdots)W_2 + b_2}_{(seq,\ 512)} $$

これは1本のベクトル用の MLP の forward とコードが一字一句同じです。"position-wise" は実装上の特別な指示ではなく、「行列積で書けば自然にそうなっている」という**性質に付けられた名前**でした。

論文がこの1語を見出しに掲げたのは、attention と対比させるためです。attention は softmax の重みで**他の位置の行を混ぜ合わせる**演算でした(第3章)。Transformer のブロックの中で行と行が混ざる場所は attention だけで、FFN は逆に**1つの行の中で**特徴を練り直す係、位置間の情報のやり取りを一切しません。ブロックは「位置を混ぜる(attention)→ 各位置で練る(FFN)」の交互構造で、separately はこの分業宣言なのです。

原文の残りも片付けます。"they use different parameters from layer to layer"——同じ層の中では全位置が $W$ を共有するが、$N = 6$ 段の層はそれぞれ自分の $W_1, W_2$ を持つ、という但し書きです。原文の「kernel size 1 の畳み込み2回とも言える」という一文は、本シリーズが通らない CNN の別名を紹介しているだけなので、読めなくて困りません。

実装します。新しい数学がないことが、そのままコードの短さに現れます。核心は式(2)と1対1対応のこの1行です。

```python
class PositionwiseFFN:
    def __init__(self, d_model, d_ff, rng):
        # 初期化スケールは第5巻6.6どおり: ReLU の前は He(√(2/入力次元))
        self.W1 = Tensor(rng.standard_normal((d_model, d_ff)) * np.sqrt(2.0 / d_model))
        self.b1 = Tensor(np.zeros(d_ff))
        self.W2 = Tensor(rng.standard_normal((d_ff, d_model)) * np.sqrt(1.0 / d_ff))
        self.b2 = Tensor(np.zeros(d_model))

    def __call__(self, X):
        # 式(2)と1対1対応のこの1行が、実装のすべて
        return (X @ self.W1 + self.b1).relu() @ self.W2 + self.b2
```

全文と動作確認は `code/ch06/position_wise_ffn.py` です(`python3` で全 assert 通過)。`d_ff = 4 × d_model`(論文の 512 → 2048 と同じ比率)で初期化し、テストは原文の2つの副詞を assert に翻訳しています。(2) は「行を1本ずつ流しても結果が変わらない」= separately。(3) は「行を並べ替えても、出力が同じ並べ替えになるだけ」= identically(どの行にも同じ変換だから、入れ替えても区別がつかない)。実行すると、

```
ok: position-wise FFN(式2)のテストにすべて通りました
```

と表示されます。なお (3) の「並べ替えても結果が並べ替わるだけ」という性質は、attention にも(mask がなければ)成り立ってしまう、Transformer 全体の急所でもあります。次章 3.5 Positional Encoding は、まさにこの急所の話です。

## 6.2 3.4 逐行: 入力埋め込み(第6巻3章)、出力側の linear + softmax(第4巻6章)

Section 3.4 も短い節です。前半の2文です。

> *"Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension d_model. We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities."*
> — 同論文, Section 3.4
>
> 訳: 「他の系列変換モデルと同様に、入力トークンと出力トークンを $d_{model}$ 次元のベクトルに変換するために、**学習される埋め込み**を使う。また decoder の出力を**次トークンの予測確率**に変換するために、通常の学習される線形変換と softmax 関数を使う。」

1文ずつ、手持ちの装備と突き合わせます。

**"learned embeddings to convert the input tokens ... to vectors of dimension d_model"** — 第6巻3章そのものです。埋め込み行列 $E$ `(vocab, d_model)` を用意し、トークン番号 $t$ の埋め込みは「$E$ の第 $t$ 行」です。one-hot ベクトルとの行列積 `onehot @ E` が行の取り出しに一致すること、そして $E$ が**与えるものではなく学習されるパラメータ**であることも第6巻3章で確かめました。"learned" の1語にそれが詰まっています。図1の入口、"Input Embedding" と "Output Embedding" の2つの箱がこれです(decoder 側にも入口があり、1つ右にずれたターゲット文を食べるのは第5章で見たとおり)。

**"linear transformation and softmax function to convert the decoder output to predicted next-token probabilities"** — 出口です。decoder の最終出力 $X$ `(seq, 512)` を linear で語彙数の幅に広げ、各行を softmax にかけます。

$$ \underbrace{X}_{(seq,\ 512)} \ @ \underbrace{W}_{(512,\ vocab)} \to \underbrace{\text{logits}}_{(seq,\ vocab)} \xrightarrow{\ \text{行ごとに softmax}\ } \underbrace{P(\text{次のトークン})}_{(seq,\ vocab)} $$

各行は「語彙全体の上の確率分布」になります。これは第4巻6章の softmax 回帰の出力層と同じ形で、分布の中身は第6巻1章の言語モデルの条件付き確率 $P(\text{次のトークン} \mid \text{これまで})$ です。図1の一番上、"Linear" と "Softmax" の2つの箱が読めました。

ここまで新情報はゼロ、回収だけで読み切れました。問題は残りの2文です。

## 6.3 weight sharing: 入力埋め込み・出力射影で重みを共有する話と √d_model 倍の補正

3.4 の後半、この章の主役です。

> *"In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30]. In the embedding layers, we multiply those weights by √d_model."*
> — 同論文, Section 3.4
>
> 訳: 「我々のモデルでは、**2つの埋め込み層と、softmax 前の線形変換とで、同じ重み行列を共有する**。埋め込み層では、その重みを $\sqrt{d_{model}}$ 倍する。」

登場する行列は3枚です。(a) 入力埋め込み(ソース言語、たとえば英語)、(b) 出力埋め込み(ターゲット言語、たとえばドイツ語)、(c) 出口の pre-softmax linear です。素直なら3枚必要なこれを、論文は**1枚で済ませる**と言っています。疑問が2つ湧きます。

**疑問1: 英語とドイツ語で同じ埋め込みが使えるのでしょうか?** 使えます。論文 5.1 に種明かしがあり、語彙は両言語**共通の** BPE(第6巻2章)で作られています("shared source-target vocabulary of about 37000 tokens")。番号表が1冊なら、番号→ベクトルの辞書も1冊でかまいません。

**疑問2: 埋め込み(入口)と出力射影(出口)は、向きが逆なのに共有できるのでしょうか?** shape で確かめます。$E$ `(vocab, d_model)` に対して、

- **入口**: トークン $t$ → $E$ の**第 $t$ 行** $\mathbf{e}_t$ `(d_model,)` を取り出す。番号からベクトルへの**順引き**
- **出口**: 必要なのは $W$ `(d_model, vocab)`。$W = E^T$ と置けば shape がぴったり合う

$W = E^T$ と置いたときの logits の中身を見ると、共有が単なる節約以上だとわかります。位置 $i$ の隠れ状態を $\mathbf{x}_i$ とすると、

$$ \text{logits}[i, t] = (X E^T)[i, t] = \mathbf{x}_i \cdot \mathbf{e}_t $$

**トークン $t$ のスコア = いまの隠れ状態と、トークン $t$ の埋め込みベクトルとの内積**です。第1巻2章以来の「内積 = 類似度」です。出口の仕事は「いま頭の中にあるベクトルに、一番似た埋め込みを持つトークンを探す」、ベクトルから番号への**逆引き**でした。順引きと逆引きで辞書が同じ1冊なのは自然な設計です。$E$ の第 $t$ **行**が、転置されて $E^T$ の第 $t$ **列**として内積の相手役に回ります。入口と出口は1枚の行列の行と列の関係でした。

残るは「埋め込み層では $\sqrt{d_{model}}$ 倍する」です。出口ではそのまま使い、**入口で取り出した行だけ** $\sqrt{512} \approx 22.6$ 倍します。論文は理由を書いていませんが、1枚の行列に2役を演じさせたつじつま合わせとして読めます。

2つの役は、求めるスケールが食い違います。**出口の役**から考えると、$E$ の成分は小さめであってほしいのです。logits は $d_{model}$ 項の和の内積なので、成分が大きいと logits が育ちすぎて softmax が尖り、勾配が死にます(第4巻7章で $\sqrt{d_k}$ を導いたのと同じ病気)。成分の分散を約 $1/d_{model}$ にしておくと(第5巻6.6の初期化)、行ベクトル $\mathbf{e}_t$ のノルムは約1に収まります。ところが**入口の役**にはノルム1は小さすぎます。次章 3.5 で埋め込みには positional encoding——成分が $\pm 1$ 程度の波——が**足され**、ノルム1のベクトルにそんな波を足したらトークンの意味がかき消されかねません。そこで入口だけ $\sqrt{d_{model}}$ 倍してノルムを約 $\sqrt{d_{model}}$ に引き上げ、足し算の桁を釣り合わせる——これが標準的な読み筋です。証明はしません。次章で positional encoding の正体を見るとき、この補正を思い出してください。

共有がどれだけパラメータを節約しているかは演習で数えます。先に言うと、$E$ 1枚は base model 全体の**約3割**を占める大物です。

## 6.4 [コード] embedding 層と出力 head の実装 + テスト

6.2 と 6.3 を1つのファイルにします。埋め込み(`Embedding`)と、共有行列で logits を作る出力 head(`output_logits`)です。核心の2か所を抜粋します。

```python
class Embedding:
    """論文 3.4 の learned embeddings。E (vocab, d_model) を1枚持つ。"""
    def __call__(self, ids):
        """ids: トークン番号の整数配列 (seq,) → Tensor (seq, d_model)。"""
        ids = np.asarray(ids, dtype=int)
        E = self.E
        scale = np.sqrt(self.d_model)  # 論文 3.4: "we multiply those weights by √d_model"
        out = Tensor(E.data[ids] * scale, (E,))

        def _backward():
            # 同じトークンが複数位置に現れたら勾配は全位置ぶんの合計(第2巻5章: 道が複数なら足す)。
            # np.add.at は重複 index でも全部足す(E.grad[ids] += ... は1回しか足さない)
            np.add.at(E.grad, ids, out.grad * scale)
        out._backward = _backward
        return out


def output_logits(X, E):
    """出力 head(pre-softmax linear)。logits = X @ E^T。weight sharing で同じ E を渡す。"""
    out = Tensor(X.data @ E.data.T, (X, E))
    def _backward():
        X.grad += out.grad @ E.data    # ∂L/∂X = δ @ E
        E.grad += out.grad.T @ X.data  # ∂L/∂E = δ^T @ X(第5巻3章の転置版)
    out._backward = _backward
    return out
```

全文と動作確認は `code/ch06/embedding.py` です(`python3` で全 assert 通過)。コードだけでは読めない箇所を2つ補います。

**`np.add.at` を使う理由。** `ids = [3, 7, 3, 0, 9]` のように同じトークンが文中に2回現れることは普通にあります。このとき $E$ の第3行は計算グラフの2か所で使われ、backward では2か所ぶんの勾配を**足す**必要があります(第2巻5章の連鎖律——道が複数なら足す)。素朴な `E.grad[ids] += ...` は NumPy の仕様で重複 index に1回しか足さないため、全部足す `np.add.at` を使います。autograd を自作した者だけが踏む渋い落とし穴です。

**`output_logits` を関数にした理由。** 出力 head は自前のパラメータを1つも持ちません。持たないことこそが weight sharing なので、クラスにせず「$X$ と、よそ様の $E$ を借りて logits を返す」だけの関数にしました。`emb.params()` が `[E]` の1枚きりであることが共有の証拠です。

仕上げに、この章の部品を直列につないだミニ・パイプラインで backward を検証します。全文は `code/ch06/test_ffn_embedding.py` です。心臓部はこれです。

```python
def forward_loss(emb, ffn, ids, targets):
    """ids (seq,) → 埋め込み → FFN → 共有 E で logits → 平均 cross-entropy。"""
    X = emb(ids)                       # (seq, d_model)
    H = ffn(X)                         # (seq, d_model)
    logits = output_logits(H, emb.E)   # (seq, vocab)
    return softmax_cross_entropy(logits, targets)
```

入口から出口まで、この章と第5巻の部品だけで「トークン列を食べて次トークンの損失を返す」流れが組めています(attention がまだ挟まっていないだけで、第8巻でやる組み立ての構造はもうこれです)。テストは2段構えです。

1. **数値微分との照合**: 全パラメータについて、autograd の勾配を中心差分 $(L(\theta+\varepsilon) - L(\theta-\varepsilon))/2\varepsilon$(第2巻1章)と突き合わせます。$E$ は入口と出口の**2か所**で使われますが、数値微分は「$E$ を少し動かすと損失がどう動くか」を測るだけなので2役ぶんを勝手に合算します。autograd 側が同じ値を出せば、共有の backward(lookup 側の `np.add.at` と射影側の `δ^T @ X` の合流)が正しい証拠になります
2. **共有をほどく実験**: 値だけ同じで**別ノード**の $E$ を2枚作り、入口と出口に1枚ずつ使って backward します。forward は共有版と完全に一致し、勾配は「入口役 + 出口役 = 共有1枚」に分解されることを確認します

実行結果です。

```
ok: grad_E が数値微分と一致(最大誤差 5.24e-10)
ok: grad_W1 が数値微分と一致(最大誤差 4.71e-10)
ok: grad_b1 が数値微分と一致(最大誤差 3.82e-10)
ok: grad_W2 が数値微分と一致(最大誤差 4.71e-10)
ok: grad_b2 が数値微分と一致(最大誤差 1.48e-10)
ok: 共有1枚の grad_E = 入口役 + 出口役 の和に一致
all tests passed
```

誤差は $10^{-10}$ のオーダーです。Section 3.3 と 3.4 の全部品が、検算済みで手元に揃いました。

## まとめ

- 式(2)の FFN は **linear → ReLU → linear のただの MLP**(第5巻で読了済み)。新情報は "position-wise" の1語だけで、その実体は「`(seq, d_model)` の**各行に同じ MLP を独立適用**」。行列積は最初から行ごとに働くので、実装は1本ベクトル用の MLP と同じコードになる
- attention は**行を混ぜる**係、FFN は**各行の中で練る**係。Transformer ブロックはこの2つの分業の交互
- 入口の埋め込みは $E$ `(vocab, d_model)` の行の取り出し(第6巻3章)、出口は linear + softmax で次トークンの分布(第4巻6章 + 第6巻1章)。3.4 の前半は回収のみで読める
- **weight sharing**: 入力埋め込み・出力埋め込み・pre-softmax linear の3役を $E$ 1枚が演じる。語彙が共有 BPE だから両言語で共有でき、出口の logits が $\mathbf{x} \cdot \mathbf{e}_t$(内積 = 類似度)になるから入口と出口でも共有できる。$E$ の行(順引き)と $E^T$ の列(逆引き)は同じベクトル
- **$\sqrt{d_{model}}$ 倍**は入口だけの補正。出口役には小さい成分が都合よく、入口役では次章の positional encoding との足し算に負けない大きさが要る——2役のスケール調停として読む

**ラスボスとの距離**: Section 3 で未読の節は、残り1つ——3.5 Positional Encoding だけになりました。

## 演習

**問1(TOC 指定: base model のパラメータ数)** 論文 Table 3 は base model のパラメータ数を $65 \times 10^6$ と記しています。本文と論文の設定($N=6$、$d_{model}=512$、$d_{ff}=2048$、$h=8$、語彙約37000、weight sharing あり、attention の射影にバイアスなし)から、この数を自分で概算して突き合わせてください。ヒント: 数えるのは「埋め込み」「attention 1個」「FFN 1個」「layer norm 1個」の4種類だけです。layer norm のパラメータは $\gamma, \beta$ の $2 d_{model}$ 個です(第5巻6.3)。

<details><summary>略解</summary>

部品ごとに数えます(検算スクリプトは `code/ch06/ex_param_count.py`)。

- **埋め込み + 出力 head**: 共有で $E$ 1枚 = $37000 \times 512 = 18{,}944{,}000 \approx 18.9$M
- **attention 1個**: $W^Q, W^K, W^V, W^O$ 各 `(512, 512)` で $4 \times 512^2 = 1{,}048{,}576 \approx 1.05$M。$h=8$ に分割しても総数は変わりません(第4章4.5)
- **FFN 1個**: $512 \times 2048 + 2048 + 2048 \times 512 + 512 = 2{,}099{,}712 \approx 2.1$M。FFN は attention の**約2倍重い**部品です
- **layer norm 1個**: $2 \times 512 = 1024$
- **encoder 1層** = attention + FFN + LN×2 $\approx 3.15$M、6層で $\approx 18.9$M
- **decoder 1層** = attention×2(masked self + cross)+ FFN + LN×3 $\approx 4.2$M、6層で $\approx 25.2$M

合計 $18.9 + 18.9 + 25.2 = 63.0$M。論文の 65M とは 2M ほどずれますが、語彙サイズが「約37000」という丸めである以上、この概算で 60M 台前半に乗れば突き合わせ成功です(先頭の桁とオーダーが合っており、ずれは語彙の端数で説明できる範囲)。スクリプトの assert もこの基準(60M〜70M、丸めて 63M)で通ります。

</details>

**問2(weight sharing の節約額)** 問1の内訳を使って、もし共有を**しなかった**ら(入力埋め込み・出力埋め込み・pre-softmax linear を別々の3枚にしたら)パラメータ数はいくつになるか計算してください。共有は全体の何割を節約していますか。

<details><summary>略解</summary>

3枚にすると埋め込み系は $3 \times 18.9 = 56.8$M になり、合計は $63.0 + 2 \times 18.9 = 100.9$M。共有による節約は $37.9$M で、非共有版の**約4割弱**にあたります。逆に言えば、共有してもなお $E$ 1枚(18.9M)はモデル全体の約3割を占める最大の単品パーツです。`ex_param_count.py` の `share_embeddings=False` で検算できます。

</details>

**問3(position-wise でなかったら)** FFN が position-wise である(行を混ぜない)ことに不満を持った人が、「位置をまたいで全部混ぜる FFN」を作ろうとしたとします。$X$ `(seq, 512)` を1本の長いベクトル `(seq × 512,)` に伸ばして、ふつうの linear `(seq×512, seq×512)` をかける設計です。$seq = 100$ のとき、この $W$ 1枚のパラメータ数を概算してください。また、パラメータ数のほかに、この設計が translation モデルとして致命的に困る点を1つ挙げてください。

<details><summary>略解</summary>

$W$ は $(100 \times 512)^2 = 51200^2 \approx 26$**億**個。base model 全体(63M)の40倍以上が、たった1層に必要になります。さらに致命的なのは、$W$ の shape が $seq$ に固定されてしまうこと——文の長さが変わるたびに使えなくなり、可変長の入力を扱う翻訳では成立しません。「位置をまたいで混ぜる」仕事は attention が引き受けています。attention の重み(softmax の出力)はパラメータではなく**入力から計算される**ので、どんな長さの文にも同じ $4 d_{model}^2$ 個の射影だけで対応できる——この対比こそが、FFN が安心して position-wise でいられる理由です。

</details>

## 式番号 ↔ コード対応表

| 論文の箇所 | 原文・式 | コード(`vol7-attention/code/ch06/`) |
|---|---|---|
| 3.3 式(2) | $\mathrm{FFN}(x) = \max(0, xW_1+b_1)W_2+b_2$ | `position_wise_ffn.py` — `(X @ self.W1 + self.b1).relu() @ self.W2 + self.b2` |
| 3.3 | "applied to each position **separately** and **identically**" | 同ファイルのテスト (2)(行ごと一致)と (3)(並べ替え同変) |
| 3.3 | $d_{model}=512$, $d_{ff}=2048$(4倍の比率) | `PositionwiseFFN.__init__` の `(d_model, d_ff)` / `(d_ff, d_model)` |
| 3.4 | "learned embeddings ... of dimension d_model" | `embedding.py` — `Embedding.__call__` の `E.data[ids]`(one-hot 等式は `__main__` で検証) |
| 3.4 | "linear transformation and softmax ... next-token probabilities" | `embedding.py` — `output_logits`(softmax は第5巻 `softmax_cross_entropy` が担当) |
| 3.4 | "share the same weight matrix"(weight sharing) | `output_logits(X, emb.E)` に同じ `E` を渡す。`test_ffn_embedding.py` で勾配の2役合算を検証 |
| 3.4 | "multiply those weights by √d_model" | `embedding.py` — `scale = np.sqrt(self.d_model)` の行 |
| Table 3 | base model = $65 \times 10^6$ params | `ex_param_count.py`(概算 63M で突き合わせ) |

---

> [目次](../TOC.md) ・ [← 前の章](05-three-attentions.md) ・ [次の章 →](07-positional-encoding.md)
