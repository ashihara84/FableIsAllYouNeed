# 第1章 組み立て — 部品からTransformerへ(自作スタックの卒業制作)

> [目次](../TOC.md) ・ [← 前の章](00-prologue.md) ・ [次の章 →](02-data.md)

第7巻の終章で、私たちは2つの「やっていないこと」を確認しました。**組み立てていない。訓練していない。** 図1のすべての箱に単体テスト済みの自作実装が対応しているのに、それらはまだバラバラの部品箱の中にあります。

この章で、1つ目の溝を埋めます。第7巻の部品を1行も書き直さずに import し、EncoderLayer / DecoderLayer → Encoder / Decoder → Transformer の順に組み上げて、図1を**1つの動くプログラム**にします。組み上がったら結合テストで配管を検査し、小さなバッチを実際に学習させて「全体として学習できる」ことまで確かめます。最後に、この自作スタックで論文と同じ規模の訓練をしたら何日かかるかを**実測から見積もり**ます。その数字が、次章から PyTorch を解禁する理由になります(序章0.3の方針——作ったから使う資格があり、必要になったから使う)。

**この章まで、PyTorch は登場しません。** 道具は NumPy と、第5巻で自作した autograd だけです。自作の部品だけで Transformer が組めて、動いて、学習する——それを確認してからでなければ、「PyTorch に乗り換えても中で起きていることは全部知っている」とは言えません。卒業制作です。

## 1.1 第7巻の部品の在庫確認と、組み立ての設計図

まず部品箱を開けて、在庫を数えます。第7巻で作った部品と、第5巻から引き継ぐ道具の一覧です。

| 図1の箱 | 部品(ファイル) | 論文の箇所 | 計算の流儀 |
|---|---|---|---|
| Scaled Dot-Product Attention | 第7巻 `code/ch03/attention.py` | 3.2.1 式(1) | NumPy(forward のみ) |
| Multi-Head Attention | 第7巻 `code/ch04/multi_head.py` | 3.2.2 | NumPy(forward のみ) |
| Add & Norm と stack の配管 | 第7巻 `code/ch02/stack_skeleton.py` | 3.1 | NumPy(forward のみ) |
| Feed Forward | 第7巻 `code/ch06/position_wise_ffn.py` | 3.3 式(2) | 第5巻 Tensor(backward 込み) |
| Embedding と出力 head | 第7巻 `code/ch06/embedding.py` | 3.4 | 第5巻 Tensor(backward 込み) |
| Positional Encoding | 第7巻 `code/ch07/positional_encoding.py` | 3.5 式(3) | NumPy(学習しない定数) |
| 損失(softmax + cross-entropy) | 第5巻 `code/ch05/tensor_autograd.py` | — | 第5巻 Tensor |

在庫を数えると、1つ問題が見つかります。**部品の「流儀」が2系統に分かれている**のです。attention まわりと stack の配管は NumPy の関数で、forward しかできません(第7巻はそれで十分でした——精読の検証に勾配は不要だからです)。一方、FFN と embedding は第5巻の `Tensor` で書かれていて、backward まで通ります。

第7巻のファイルを書き直すのは禁じ手にします。単体テストで保証された部品に手を入れたら、その瞬間に保証が切れるからです。代わりに、こう設計します。

**同じ重みに対して、forward を2系統持ちます。**

- **NumPy 版 forward**: 第7巻の部品をそのまま呼んで組む。役割は**正しさの基準器**。各部品は論文と突き合わせて単体テスト済みなので、この forward の出力は「論文どおりの計算」の基準になる
- **Tensor 版 forward**: 同じ重みを第5巻の `Tensor` で流す。役割は**学習可能性の証明**。`backward()` が引けるので、訓練ループが回る

2系統は同じ重み(同じ `np.ndarray`)を共有します。`Tensor` は中に `data` として ndarray を1枚持っているだけなので(第5巻5.3)、NumPy 版は `.data` を読み、Tensor 版は `Tensor` のまま使えば、重みは自動的に1つです。そして結合テスト(1.3)で、2系統の出力が `allclose` で一致することを確認します。一致すれば、NumPy 版が持つ「論文どおり」という信用が、そのまま Tensor 版に乗り移ります。**forward の正しさは第7巻の部品で保証し、学習可能性は自作 autograd で示す**——この分業が、この章の設計の背骨です。

組み立ての順序は、図1を**実装の依存関係図**として読み直すと決まります。第7巻2章で図1を「地図」として読みましたが、今回は「どの箱がどの箱を部品として使うか」の矢印で読みます。

```
attention(式1)                     ← 最下層。誰にも依存しない
   └─ multi_head_attention(3.2.2)  ← attention を使う
        └─ EncoderLayer / DecoderLayer(3.1)
             │   = multi-head + FFN(式2)を Add & Norm の配管に差す
             └─ Encoder / Decoder = 同じ層を N 回重ねる
                  └─ Transformer
                      = embedding(3.4)+ PE(3.5)→ Encoder → Decoder
                        → 出力 head(3.4、E を共有)→ logits
```

下から上へです。各箱は1つ下の箱だけを見て組めます。第7巻が下3段を作り終えているので、この章の仕事は上3段——層を作り、重ね、入口と出口を付ける——だけです。

## 1.2 EncoderLayer / DecoderLayer → Encoder / Decoder → Transformer の組み上げ

### 足りない演算を先に補う

Tensor 版 forward を書くには、attention の計算を第5巻の `Tensor` で表現できる必要があります。ところが `tensor_autograd.py` の演算は、加減乗・2次元の行列積・relu・exp・log・sum だけです。転置がなく、multi-head の「頭に裂く」列の切り出しも、Concat も、softmax も、layer norm もありません。

第5巻のファイルも変更禁止です(同じ理由——テスト済みの autograd 本体に手を入れない)。足りない演算は、この章のファイルで**外から**補います。第5巻4章でやったとおり、forward を計算して出力ノードを作り、「親への勾配の配り方」を `_backward` に1つ書きます。それだけです。第7巻の `embedding.py` がすでに同じ手口を使っていました(`np.add.at` の backward を `Tensor` の外から差していた)。今回はそれを5回繰り返します。

5つの演算のうち、要となる2つの forward の心臓部を抜き出します。まず multi-head の Concat にあたる列方向連結と、masked softmax です。

```python
def t_masked_softmax(Z, mask=None):
    """mask(True = 見てよい)を掛けてから行ごとに softmax。
    forward は第7巻3章 attention.py と同じ式(softmax と NEG_INF を共用)。
    backward は第4巻6章の手導出: dZ = A ⊙ (dA − Σ_k dA_k A_k)。
    mask された位置は A = 0 なので、勾配も自動的に 0 になる(別処理は不要)。"""
    z = Z.data if mask is None else np.where(mask, Z.data, NEG_INF)
    A = softmax(z, axis=-1)
    out = Tensor(A, (Z,))

    def _backward():
        dA = out.grad
        Z.grad += A * (dA - (dA * A).sum(axis=-1, keepdims=True))

    out._backward = _backward
    return out
```

layer norm の Tensor 版も、backward は第5巻6.3で数値微分と照合済みの式をそのまま使います(`mu` と `var` も `x` の関数なので、`x` への勾配には補正項が2つ付く)。

```python
def t_layer_norm(X, gamma, beta, eps=1e-5):
    mu = X.data.mean(axis=-1, keepdims=True)
    var = X.data.var(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (X.data - mu) * inv_std
    out = Tensor(gamma.data * x_hat + beta.data, (X, gamma, beta))

    def _backward():
        dx_hat = out.grad * gamma.data
        X.grad += inv_std * (dx_hat
                             - dx_hat.mean(axis=-1, keepdims=True)
                             - x_hat * (dx_hat * x_hat).mean(axis=-1, keepdims=True))
        gamma.grad += (out.grad * x_hat).sum(axis=0)
        beta.grad += out.grad.sum(axis=0)

    out._backward = _backward
    return out
```

残りの3つ(`t_transpose`、列の切り出し `t_cols`、その逆の `t_concat_cols`)も同じ型です。全文と、5演算すべてを数値微分(第2巻1章の中心差分)で検算する `__main__` は `code/ch01/tensor_ops.py`(`python3` で通過)にあります。

3点だけ補足します。

第一に、`t_masked_softmax` の forward は第7巻3章の `softmax` と `NEG_INF` を import して使っています。これは仕様です——あとで NumPy 版と Tensor 版の一致を確かめるとき、forward が同じ関数なら、一致は「実装の偶然」ではなく「設計の必然」になります。

第二に、softmax の backward $dZ = A \odot (dA - \sum_k dA_k A_k)$ は第4巻6章で導出した式です。mask された位置は重み $A$ が厳密に 0 なので、この式に通すと勾配も自動的に 0 になります。「未来へは勾配も流れない」が、特別な処理なしで成立しています。

第三に、`__main__` の数値微分照合は第5巻4章で micrograd に課したのと同じ儀式です。手書きの backward は、検算が通って初めて部品になります。

### 組み上げ

部品が揃いました。組みます。新しい計算は1つもありません——import した部品を、図1の配線どおりに呼んでいるだけです。

要は、第7巻2章で作った「差し替え可能な Sublayer」の骨組みに本物の部品を差すことと、その Tensor 版を並べて書くことです。EncoderLayer の2系統 forward が、構造をいちばんよく表しています。

```python
class EncoderLayer:
    """論文 3.1 encoder の1層 = self-attention + FFN(各々 Add & Norm 付き)。"""

    def forward_numpy(self, x):
        """第7巻2章 encoder_layer の差し替え可能シグネチャに、本物の部品を差す。"""
        prm = {"gamma1": self.gamma1.data, "beta1": self.beta1.data,
               "gamma2": self.gamma2.data, "beta2": self.beta2.data}

        def self_attn(x_):
            out, _ = multi_head_attention(x_, x_, self.W_q.data, self.W_k.data,
                                          self.W_v.data, self.W_o.data, self.h)
            return out

        def ffn_f(x_):
            return self.ffn(Tensor(x_)).data   # Tensor 部品を forward 専用で使う

        return encoder_layer(x, prm, self_attn, ffn_f)

    def forward_tensor(self, X):
        """同じ重み・同じ配管を Tensor で。LayerNorm(x + Sublayer(x)) が2回。"""
        a = mha_tensor(X, X, self.W_q, self.W_k, self.W_v, self.W_o, self.h)
        X = t_layer_norm(X + a, self.gamma1, self.beta1)
        X = t_layer_norm(X + self.ffn(X), self.gamma2, self.beta2)
        return X
```

multi-head attention の Tensor 版 `mha_tensor` は、第7巻4章の式そのままで、「頭に裂く」を reshape の代わりに列の切り出し(`t_cols`)+ for ループで書いたものです(`Tensor` の行列積が2次元限定のため)。head $i$ の中身は $X W$ の列ブロックそのものなので、2つの書き方は同じ計算です。心臓部はループ本体だけです。

```python
    for i in range(h):
        Qi = t_cols(Q, i * d_k, (i + 1) * d_k)        # (n, d_k) head i の担当列
        Ki = t_cols(K, i * d_k, (i + 1) * d_k)        # (m, d_k)
        Vi = t_cols(V, i * d_k, (i + 1) * d_k)        # (m, d_v)
        S = (Qi @ t_transpose(Ki)) * (1.0 / np.sqrt(d_k))  # QK^T/√d_k : (n, m)
        A = t_masked_softmax(S, mask)                 # (n, m) 行ごとの和が1
        heads.append(A @ Vi)                          # (n, d_v)
    return t_concat_cols(heads) @ W_o                 # Concat(...) W^O : (n, d_model)
```

DecoderLayer は3つの部分層(masked self / cross / FFN)になるだけで型は同じです。Encoder / Decoder は層を for で N 回重ね、Transformer は入口の embedding + PE、Encoder → Decoder、出口の出力 head(E を共有)をつなぎます。これらクラス全文と、全体 forward の2系統定義、shape を確認する `__main__` は `code/ch01/transformer.py`(`python3` で通過)にあります。Transformer の forward(NumPy 版)が入口から出口までの shape の流れです。

```python
    def forward_numpy(self, src_ids, tgt_in, causal=True):
        """src_ids: 入力文 (src_len,)、tgt_in: decoder 入力 (tgt_len,)。
        返り値: logits (tgt_len, vocab)。"""
        x = self.emb(src_ids).data + self.pe[:len(src_ids)]   # 埋め込み + PE
        memory = self.encoder.forward_numpy(x)                # (src_len, d_model)
        y = self.emb(tgt_in).data + self.pe[:len(tgt_in)]
        y = self.decoder.forward_numpy(y, memory, causal=causal)
        return output_logits(Tensor(y), self.emb.E).data      # (tgt_len, vocab)
```

読みどころを順に挙げます。

**差し替えの瞬間。** 第7巻2章の `encoder_layer(x, prm, self_attn, ffn)` は、部分層を「`(seq_len, d_model)` を受け取って同じ shape を返す関数」として受け取る設計でした。あのとき差してあったのは恒等写像のダミーです。今回、その同じ穴に `multi_head_attention` と `PositionwiseFFN` を差しました。第7巻2章の骨組みが、約束どおり1文字も変えずに本物を受け入れています。decoder 側も同様で、masked self に causal mask、cross に `memory` という割り当ては第7巻5章の配線表のとおりです。

**`mha_tensor` は同じ式の別表記。** ループは遅い書き方ですが、遅さはむしろ好都合です——1.4で「自作スタックの素の速度」を測るのですから。

**重みの共有は `.data` 経由で自動。** `forward_numpy` は各 `Tensor` の `.data` を読むだけなので、2系統の forward は常に同一の重みを見ています。「NumPy 版の重みを Tensor 版にコピーする」手順そのものが存在しないため、コピー忘れというバグも存在できません。

**乱数と環境の注意。** 重みの初期化は `np.random.default_rng(42)` で固定します。また `transformer.py` 冒頭の `warnings.filterwarnings` は macOS の Accelerate BLAS が出す**偽の**浮動小数点警告(計算結果は正しいのに matmul が divide by zero を報告する既知の不具合)への対処で、それが偽であることは次節のテストの matmul / einsum 照合で確かめます。

実行します。

```
$ python3 transformer.py
Transformer 組み上げ OK: src(7,) + tgt(5,) -> logits (5, 12)
パラメータ総数: 42,368(検算は param_count.py、結合テストは integration_test.py)
```

入力文(長さ7)と decoder 入力(長さ5)が、embedding から N=2 層の encoder・decoder を通り、`(5, 12)` の logits になって出てきました。図1が、初めて1つのプログラムとして動いた瞬間です。

ただし shape が通っただけです。配管が正しいか、学習できるかは、まだ何も保証されていません。テストに進みます。

## 1.3 結合テスト: 全体 forward の shape、causal mask が末端まで効いているか

単体テストは部品の保証、結合テストは配線の保証です。部品が全部正しくても、配線を1本間違えれば全体は壊れます。そして Transformer の配線ミスには、たちの悪い性質があります——**shape が通ってしまう**ことが多いのです。mask を渡し忘れても、cross-attention に渡す行列を取り違えても、出てくる logits の形は `(tgt_len, vocab)` のままです。エラーは出ず、ただ「学習しても性能が出ない」という形で数日後に祟ります。

だから、shape の先を検査します。検査は4段です。

1. **shape**: 入口から出口まで通ること(最低限の関門)
2. **causal mask が末端まで効くこと**: 第7巻5章で attention 単体に課した検査——未来のトークンを改変しても過去の出力が変わらない——を、今度は**embedding から logits までの全経路**に課します。途中のどの1層が mask を取りこぼしても、この検査は落ちます
3. **NumPy 版と Tensor 版の一致**: 1.1 で設計した分業の要。さらに Tensor 版の勾配を数値微分でスポット照合します
4. **丸暗記テスト**: 1バッチを過学習できること

4つ目は、覚えてほしい**デバッグの定石**です。新しくモデルと訓練の仕組みを組んだら、本番データを流す前に、ごく小さな1バッチを「丸暗記できるか」試します。数百万パラメータのモデルにとって数十トークンの暗記は造作もない仕事のはずで、それなのに loss がゼロ近くまで落ちないなら、データの並べ方・mask・損失・勾配のどこかが確実に壊れています。落ちても正しさの証明にはなりませんが、「壊れてはいない」という安心は得られます。安くて感度の高い煙探知機です。

暗記させるデータには、**規則のないでたらめな対応**をわざと使います。入力列も正解列もただの乱数です。規則がないので「賢く一般化して解いた」可能性が消え、成功の解釈が「このバッチを記憶する勾配が、出口の損失から入口の embedding まで生きて流れた」の一通りに定まります。

この4段を実行する `code/ch01/integration_test.py`(seed 42、数十秒で全 assert が通る)の核心は、(2) causal mask の検査です。「通る」だけでなく「故意に壊すと落ちる」までをワンセットにしています。

```python
# ---- (2) causal mask が末端まで効く ----
# 未来(最後の位置)のトークンを改変しても、それより前の位置の logits は1bitも動かない
k = len(tgt_in) - 1
tgt_in2 = tgt_in.copy()
tgt_in2[k] = (tgt_in[k] + 3) % vocab
logits2 = model.forward_numpy(src, tgt_in2)
assert np.array_equal(logits[:k], logits2[:k])     # 過去は完全不変(allclose ですらなく等値)
assert not np.allclose(logits[k], logits2[k])      # 改変した当の位置だけは変わる

# わざと壊す: mask を外すと同じ検査が落ちる(このテストがバグを検出できる証拠)
bad1 = model.forward_numpy(src, tgt_in, causal=False)
bad2 = model.forward_numpy(src, tgt_in2, causal=False)
assert not np.allclose(bad1[:k], bad2[:k])         # 未来が過去に漏れる
```

(3) では2系統の logits 一致を `diff < 1e-9` で確かめ、入口・中間・出口から4点の勾配を数値微分でスポット照合します。(4) では規則のない src→tgt のペア2本を、4拍子(forward → loss → backward → update)の素朴な勾配降下で300ステップ暗記させ、`history[-1] < 0.05` と argmax の完全再生を assert します。全文は `code/ch01/integration_test.py`(`python3` で通過)にあります。

実行結果です(手元のマシンで約1秒)。

```
$ python3 integration_test.py
(1) shape OK: src (7,) + tgt_in (5,) -> logits (5, 12)
(2) causal mask OK: 未来の改変は過去に漏れない(mask を外すと漏れる)
(3) NumPy/Tensor 一致 OK: logits の最大差 = 1.78e-15
    勾配照合 OK: embedding E        autograd -0.21895691 / 数値微分 -0.21895691
    勾配照合 OK: encoder層0 W_q      autograd +0.00054076 / 数値微分 +0.00054076
    勾配照合 OK: decoder層1 gamma3   autograd +0.01385448 / 数値微分 +0.01385448
    勾配照合 OK: decoder層0 cross U_o autograd -0.09500174 / 数値微分 -0.09500174
    step    0: loss = 3.012276
    step   50: loss = 0.389369
    step  100: loss = 0.007177
    step  299: loss = 0.000954
(4) 丸暗記 OK: loss 3.012 -> 0.0010、2系列とも完全再生
integration_test: すべての assert を通過しました
```

結果を1段ずつ読みます。

**(2) の検査は「故意に壊す」までがワンセットです。** `causal=False` で mask を外したら同じ検査が**落ちる**ことも確認しています。バグを入れたら落ちる——その感度を見せて初めて、「通った」に意味が出ます。過去の位置の比較に `allclose` ではなく `array_equal` を使えるのは、mask が「未来の重みを小さくする」のではなく「厳密に 0 にする」仕掛けだからです(第7巻3章の $-\infty$ の効能がここまで届いています)。

**(3) の一致が、この章の蝶番です。** 最大差 `1.78e-15`——浮動小数点の丸めの粒まで、2系統は同じ計算でした。第7巻の部品が持つ「論文どおり」という信用が、これで Tensor 版に乗り移りました。続く勾配照合は、`tensor_ops.py` の単体検算では届かない「全部つないだときの backward」への抜き取り検査です。入口(embedding)、中間(encoder の $W^Q$、cross-attention の $U^O$)、出口近く(decoder 最終層の $\gamma$)の4点で、autograd の勾配と実測の傾きが8桁一致しています。

**(4) で、Transformer が初めて学習しました。** このシリーズで組んだ最大のモデル(といっても4万パラメータですが)が、loss 3.01——ほぼ当てずっぽう($\ln 12 \approx 2.48$ よりやや上)——から 0.001 まで降りて、2系列を完全に暗記しました。使った訓練ループは forward → loss → backward → update です。第3巻4章で線形回帰に使った**4拍子と、1拍も違いません**。モデルがどれだけ育っても学習の骨格は変わらない——シリーズを貫いてきたこの主張の、これが最終確認です。

これで言えるようになりました。**部品は正しく配線され、勾配は末端まで生きていて、全体は学習できる。** 図1は完成品です。

ならば、このまま本物のコーパスで訓練すればよいのではないでしょうか。PyTorch など要らないのではないでしょうか。

——測ってみましょう。

## 1.4 自作スタックの限界を実測する

論文の Section 5 から、訓練の「量」を定めている2か所を読みます。

> *"Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens."*
> — Vaswani et al., "Attention Is All You Need", Section 5.1
>
> 訳: 文対はおおよその系列長でまとめてバッチ化した。各訓練バッチは、およそ25000のソーストークンと25000のターゲットトークンを含む文対の集合からなる。

> *"We trained our models on one machine with 8 NVIDIA P100 GPUs. [...] Each training step took about 0.4 seconds. We trained the base models for a total of 100,000 steps or 12 hours."*
> — 同論文, Section 5.2
>
> 訳: モデルは NVIDIA P100 GPU を8基積んだ1台のマシンで訓練した。(中略)1訓練ステップはおよそ0.4秒であった。base モデルは合計 100,000 ステップ、すなわち12時間訓練した。

論文の訓練は「1ステップで約25000ターゲットトークンを処理し、それを10万回」です。私たちの自作スタックは、1ステップで何トークン処理でき、何秒かかるのでしょうか。**base model と同じ構成(N=6, d_model=512, d_ff=2048, h=8, 語彙37000)を実際に組んで**測ります。

計測の心臓部は、4拍子1回 = 訓練1ステップを `time.perf_counter` で挟み、論文の訓練量(25000トークン/step × 100,000 step)に線形外挿する部分です。

```python
def one_step():
    """4拍子1回 = 訓練1ステップ(forward → loss → backward → update)。"""
    loss = softmax_cross_entropy(model.forward_tensor(src, tgt_in), tgt)
    for p in params:
        p.grad = np.zeros_like(p.data)
    loss.backward()
    for p in params:
        p.data -= 1e-4 * p.grad            # 更新も計測に含める(本番は毎step行うので)
    return loss.data
```

自作スタックは1ステップで 32 ターゲットトークンしか処理しないので、25000 トークンを流すには `25000/32 ≈ 781` 倍の時間がかかります(線形外挿。実際は系列が長いほど attention が $O(n^2)$ で重くなるので、これでも甘めの見積もり)。構成定義・組み立て・見積もり出力の全文は `code/ch01/limit_check.py`(`python3` で通過、1分弱・メモリ 1GB 強)にあります。

実行結果です(時間は手元のマシンの実測例。あなたの環境では数字が変わりますが、結論は変わりません)。

```
$ python3 limit_check.py
base model(N=6, d_model=512, d_ff=2048, h=8, 語彙37,000)を組み立て中...
  組み立て 0.2 秒、パラメータ 63,045,632(≈ 63M — 第7巻6章の検算と同じ規模)
  訓練1ステップ(32トークンの文対1本): 0.11 秒

  論文の訓練量: 25,000 トークン/step × 100,000 step
  必要時間 = 0.11 秒 × 781 × 100,000 = 8.61e+06 秒
  = 約 100 日 = 約 0.3 年(このマシン1台、自作スタックで)
  論文の実測: 8 × P100 GPU で 12 時間(5.2)

limit_check: すべての assert を通過しました — 物理的に終わらない。卒業の時です
```

数字を表に整理します。

| | 論文(8 × P100 GPU) | 自作スタック(CPU 1台・実測例) |
|---|---|---|
| 1ステップの処理量 | 約 25,000 トークン | 32 トークン |
| 1ステップの時間 | 約 0.4 秒 | 0.11 秒 |
| スループット | 約 62,500 トークン/秒 | 約 290 トークン/秒 |
| base の訓練(100,000 step) | **12 時間** | **約 100 日** |

まず認めるべきことを認めます。63M パラメータの本物サイズの Transformer を、自作スタックは**ちゃんと持ち上げました**。組めて、forward が通り、backward まで回ります。1ステップ 0.11 秒は、第5巻5.3でスカラー `Value` の遅さに絶望した身からすれば健闘です(行列演算の中身は NumPy、つまり最適化された BLAS が走っているからです)。

それでも、論文の訓練量を前にすると桁が足りません。昼夜止めずに回して**約100日**。論文の big model は 300,000 ステップですから、その再現なら**1年弱**。さらに論文の Table 3 は数十通りの構成を訓練して比べたアブレーションです。1回の訓練に100日かかる道具では、研究として成立しません。しかもこの100日は、いくつもの甘い仮定(トークン数に線形で外挿、Adam も dropout もないぶん軽い——演習3)の上に立った**楽観値**です。

足りないものは2つあります。第一に **GPU**——行列積を数千コアで並列に行うハードウェアと、それを呼び出す仕組み。第二に、バッチ・メモリ管理・最適化された backward を備えた**実戦用のスタック**。どちらも原理は全部この手で作りました。しかし原理の理解と、100日を12時間にする工学は別の仕事です。その仕事を引き受けてくれるのが PyTorch です。

第2巻で勾配降下を学んだとき、第5巻で autograd を作ったとき、PyTorch を使う「需要」はまだありませんでした——自作で間に合っていたからです。いま、初めて間に合わなくなりました。**需要が発生したので、道具を解禁します。** これが序章0.3で予告した、シリーズでただ一度の乗り換えです。

## 1.5 卒業と橋渡し: 「自作API ↔ PyTorch API 対応表」

乗り換えにあたって、橋を1枚架けておきます。下の表は「これから出会う PyTorch の道具」と「あなたがすでに作った道具」の対応表です。次章から PyTorch の API が登場するたびに、本書は脚注で「自作版のどれにあたるか」をこの表の行で示します。PyTorch を魔法の箱としてではなく、「自分が作ったあれの、速くて頑丈な版」として使うためです。

| 自作(作った場所) | PyTorch | 備考 |
|---|---|---|
| `Value`(第5巻4章)/ `Tensor`(第5巻5章) | `torch.Tensor`(`requires_grad=True`) | 値と勾配と計算グラフを1つのオブジェクトが持つ、という設計まで同じ |
| `loss.backward()`(トポロジカル順に `_backward` を呼ぶ) | `loss.backward()` | 名前まで同じ。中身も第5巻4.3の原理と同じ |
| `p.grad = np.zeros_like(p.data)` | `optimizer.zero_grad()` | 勾配は累積する(第2巻5章「道が複数なら足す」)ので、毎ステップ手動でゼロに戻す事情も同じ |
| `for p in params: p.data -= lr * p.grad` | `torch.optim.SGD(...).step()` | 4拍子の4拍目。Adam 版は第4章で自作と並走照合する |
| `X @ W + b`(第1巻6章の linear) | `nn.Linear(d_in, d_out)` | `W` と `b` を抱えて `X @ W + b` を計算する箱 |
| `PositionwiseFFN`(第7巻6章) | `nn.Linear` 2枚 + `F.relu` | 完成品の FFN モジュールは使わず、自作と同じ粒度で並べる |
| `Embedding`(第7巻6章) | `nn.Embedding` | 「行の取り出し」と「重複 index の勾配合算」(`np.add.at` で書いた部分)を内蔵 |
| `softmax_cross_entropy`(第5巻5章) | `F.cross_entropy` | softmax 込み・数値安定化込み、まで同じ設計 |
| `causal_mask`(第7巻3章) | `torch.tril(...)` + `masked_fill(-inf)` | 「softmax の前に $-\infty$」の仕掛けも同じ |
| 数値微分との照合(第2巻1章、本章1.3) | `torch.autograd.gradcheck` | 「backward を疑ったら実測の傾きと比べる」が公式APIになっている |
| `rng = np.random.default_rng(42)` | `torch.manual_seed(42)` | 再現性の規律はどちらの世界でも同じ |
| 1系列ずつ for で流す(本章) | 先頭にバッチ軸を持つテンソル `(batch, seq, d_model)` | 第1巻6.4「行列の束」。第2章のバッチングで本格化 |

この表に**載っていない**ものにも注意してください。`nn.Transformer` と `nn.MultiheadAttention`——PyTorch には Transformer の完成品が入っていますが、本書では最後まで使いません。使った瞬間、第7巻でやったことが「車輪の再発明」に格下げされてしまうからです。第2章以降のモデルは、素のテンソル演算とこの表にある最小限の部品だけで、もう一度自分の手で書きます(設計はこの章で済んでいるので、書き直しは半日仕事です)。

表にはまだ空行があります。たとえば layer norm の行がありません。これは演習2であなたが埋めます——卒業証書には自分の署名欄があるものです。

## 1.6 パラメータ数の検算: base 相当の縮小版で数える

卒業制作の最後の仕上げは、検収です。組み上げた Transformer が「余計な重みを持っていないか、必要な重みを欠いていないか」を、第7巻6章の演習で作った**紙の上の数え上げ**と突き合わせます。式と実物が1個までぴったり合えば、組み立ては過不足なしです。

`code/ch01/param_count.py`(`python3` で通過)は、第7巻6章 演習1の数え上げ式 `transformer_base_params` を import し、base 相当の縮小版(N=6, h=8, d_ff = 4×d_model の比率はそのまま、幅と語彙だけ 1/8)で組んだ実物の `n_params()` と完全一致することを assert します。検算の心臓部は3つの assert です。

```python
# 紙の上の式 = 組み上げた実物。1個の過不足もなく一致する
assert actual == expected

# h を変えても総数は変わらない(d_k = d_model / h に裂いているだけ — 第7巻4章)
model_h4 = Transformer(vocab, d_model, d_ff, 4, N, max_len=32, rng=rng)
assert model_h4.n_params() == actual

# 本家 base の 65M(第7巻6章の検算)も、同じ式の引数を変えるだけで再現できる
_, total_base = transformer_base_params()   # vocab=37000, d_model=512, d_ff=2048, N=6
assert round(total_base / 1e6) == 63        # この数え方では 63M(論文 Table 3 は 65M)
```

実行結果です。

```
$ python3 param_count.py
縮小版(N=6, d_model=64, d_ff=256, h=8, 語彙1,000)の内訳(式による計算):
  embedding(+出力head)         64,000
  encoder(6層)               298,368
  decoder(6層)               397,440
  合計(式)                     759,808
  合計(実物)                    759,808
param_count: すべての assert を通過しました — 式と実物が 759,808 個で完全一致
```

759,808 個、ぴったり一致です。第7巻6章で「論文の 65M を数えられる」ようになった式が、今度は**自分の実物の検収**に使えました。ついでに2つの事実も確認しています。head 数 $h$ を変えてもパラメータ総数は変わらないこと($d_{model}$ を $h$ 等分に裂いているだけだからです——第7巻4章4.5)。そして 1.4 で持ち上げた本家サイズの 63M も、同じ式の引数を変えるだけで出てくること。

これで検収完了です。組み上がった Transformer には、図1に描かれていない重みは1個もなく、描かれている重みは1個も欠けていません。

## まとめ

- 第7巻の部品を**1行も書き直さず**、import と関数の差し込みだけで、図1が1つの動くプログラムになった。第7巻2章の骨組み(差し替え可能な Sublayer)に本物の部品を差すだけで組み上がったのは、shape の規約を最初から揃えてきたから
- 同じ重みに **forward を2系統**持たせた。NumPy 版(第7巻部品)が正しさの基準器、Tensor 版(第5巻 autograd)が学習可能性の証明。両者は最大差 $10^{-15}$ で一致し、勾配は数値微分と一致した
- 結合テストは4段: shape、**causal mask が末端まで効くこと**(故意に壊して感度も確認)、2系統の一致、そして**丸暗記テスト**(1バッチの過学習はデバッグの定石。loss 3.01 → 0.001)
- 訓練ループは第3巻4章と同じ**4拍子**のまま。しかし論文の訓練量に外挿すると約**100日**——自作スタックは原理の証明には十分で、本番の訓練には物理的に足りない。ここで初めて PyTorch への需要が発生した
- 卒業の橋は**自作API ↔ PyTorch API 対応表**。以後、PyTorch の各 API には自作版を脚注で示す。完成品の `nn.Transformer` は最後まで使わない
- パラメータ数は紙の式と実物が **759,808 個で完全一致**。検収済み

**ラスボスとの距離**: Section 5.2 の "Each training step took about 0.4 seconds" と "12 hours" が、自分の実測(0.11秒/32トークン、約100日)と並べて読める数字になりました。Section 5 の残り——Adam の β、warmup、label smoothing——が、この巻の後半戦です。

## 演習

**問1(TOC 指定: 自分の構成でパラメータ数を見積もる)** あなたが第5章で訓練するつもりの構成を1つ決め(たとえば $N=4$、$d_{model}=256$、$d_{ff}=1024$、$h=8$、語彙8000)、パラメータ総数を紙の上で見積もってください。それから `param_count.py` の構成をその値に書き換えて実行し、実物と一致することを確認してください。GPU 1枚で数十分〜数時間という第2章の設計目標に対して、その大きさは妥当そうですか。

<details><summary>略解</summary>

例の構成($N=4$, $d_{model}=256$, $d_{ff}=1024$, $h=8$, 語彙8000)で数えます。

- embedding(共有 $E$ 1枚): $8000 \times 256 = 2{,}048{,}000$
- attention 1個: $4 \times 256^2 = 262{,}144$
- FFN 1個: $256 \times 1024 + 1024 + 1024 \times 256 + 256 = 525{,}568$
- layer norm 1個: $2 \times 256 = 512$
- encoder 1層 $= 262{,}144 + 525{,}568 + 2 \times 512 = 788{,}736$、4層で $3{,}154{,}944$
- decoder 1層 $= 2 \times 262{,}144 + 525{,}568 + 3 \times 512 = 1{,}051{,}392$、4層で $4{,}205{,}568$
- **合計 $9{,}408{,}512 \approx 9.4$M**

`Transformer(8000, 256, 1024, 8, 4, ...)` を組んで `n_params()` を呼ぶと、ちょうど 9,408,512 が返ります。base の約 1/7。「数百万〜数千万パラメータ」という第2章の設計範囲(2.1)に収まる、手頃な大きさです。$h$ は総数に影響しないので、見積もりに $h$ が出てこなくても間違いではありません。
</details>

**問2(TOC 指定: 対応表に1行を自分で追加する)** 1.5 の対応表には layer norm の行がありません。本章の `t_layer_norm`(と第5巻6.3の `layer_norm`)に対応する PyTorch API を調べて、表の書式で1行追加してください。「備考」欄には、自作版との対応で気づいたことを1つ書いてください。余力があれば dropout(第5巻6章)の行も足してみてください。

<details><summary>略解</summary>

| 自作(作った場所) | PyTorch | 備考 |
|---|---|---|
| `t_layer_norm`(本章)/ `layer_norm`(第5巻6.3) | `nn.LayerNorm(d_model)` | $\gamma, \beta$(各 $d_{model}$ 個)を抱える点、`eps`(既定 1e-5)でゼロ割りを防ぐ点まで自作版と同じ。$\gamma=1, \beta=0$ 初期化も同じ |

dropout の行は `dropout`(第5巻6章)↔ `nn.Dropout(p)`。備考に書くべきは「訓練時だけ落として推論時は素通しする、という訓練/推論の切り替え(PyTorch では `model.train()` / `model.eval()`)を自分で管理する必要がある」ことです。第3章で実際に踏みます。
</details>

**問3(見積もりの粗を探す)** 1.4 の「約100日」という見積もりには、自作スタックに**有利な**(=実際はもっとかかる方向の)仮定が少なくとも2つ含まれています。`limit_check.py` のコメントも手がかりに、それを挙げてください。逆に、自作スタックに不利な仮定はあるでしょうか。

<details><summary>略解</summary>

有利な仮定の例:

1. **トークン数への線形外挿。** 25000トークンを実際に1ステップで流すには、長い系列や大きなバッチが要ります。attention の計算量は系列長の2乗(第7巻8章の $O(n^2 \cdot d)$)なので、系列が長くなるぶん実際はもっと遅くなります。メモリも足りなくなるでしょう
2. **訓練の道具を省いている。** 本番の1ステップには Adam(モーメントの管理が全パラメータ×2枚——第4章)、dropout、label smoothing が乗ります。計測した1ステップは素の勾配降下なので、そのぶん軽い

不利な仮定もあります。たとえば 1系列ずつの Python ループや頭ごとの for ループには、行列を束ねれば消せるオーバーヘッドが含まれています。しかし、それを丁寧に潰しても数倍の改善がせいぜいで、「100日 vs 12時間」という3桁の差は埋まりません。見積もりの粗を全部直しても結論が動かない——だからこの見積もりは結論を支えるのに十分、というのが工学的な読み方です。
</details>

---

> [目次](../TOC.md) ・ [← 前の章](00-prologue.md) ・ [次の章 →](02-data.md)
