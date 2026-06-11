# 第1章 組み立て — 部品からTransformerへ(自作スタックの卒業制作)

第7巻の終章で、私たちは2つの「やっていないこと」を確認しました。**組み立てていない。訓練していない。** 図1のすべての箱に単体テスト済みの自作実装が対応しているのに、それらはまだバラバラの部品箱の中にあります。

この章で、1つ目の溝を埋めます。第7巻の部品を1行も書き直さずに import し、EncoderLayer / DecoderLayer → Encoder / Decoder → Transformer の順に組み上げて、図1を**1つの動くプログラム**にします。組み上がったら結合テストで配管を検査し、小さなバッチを実際に学習させて「全体として学習できる」ことまで確かめます。そして最後に、この自作スタックで論文と同じ規模の訓練をしたら何日かかるかを**実測から見積もり**ます。その数字が、次章から PyTorch を解禁する理由になります(序章0.3の方針です——作ったから、使う資格がある。そして、必要になったから、使う)。

宣言しておきます。**この章まで、PyTorch は登場しません。** 道具は NumPy と、第5巻で自作した autograd だけです。これは意地ではなく検証です。自作の部品だけで Transformer が組めて、動いて、学習する——それを確認してからでなければ、「PyTorch に乗り換えても中で起きていることは全部知っている」と言えないからです。卒業制作だと思ってください。

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

在庫を数えてみると、1つ問題が見つかります。**部品の「流儀」が2系統に分かれている**のです。attention まわりと stack の配管は NumPy の関数で、forward しかできません(第7巻はそれで十分でした——精読の検証に勾配は不要だからです)。一方、FFN と embedding は第5巻の `Tensor` で書かれていて、backward まで通ります。

このまま無理に1系統に直すこともできますが、第7巻のファイルを書き直すのは禁じ手にします。単体テストで保証された部品に手を入れたら、その瞬間に保証が切れるからです。代わりに、こう設計します。

**同じ重みに対して、forward を2系統持つ。**

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

下から上へ。各箱は1つ下の箱だけを見て組めます。第7巻が下3段を作り終えているので、この章の仕事は上3段——層を作り、重ね、入口と出口を付ける——だけです。

## 1.2 EncoderLayer / DecoderLayer → Encoder / Decoder → Transformer の組み上げ

### 足りない演算を先に補う

組み始める前に、もう1つだけ在庫の欠品を埋めます。Tensor 版 forward を書くには、attention の計算を第5巻の `Tensor` で表現できる必要があります。ところが `tensor_autograd.py` の演算は、加減乗・2次元の行列積・relu・exp・log・sum だけ。転置がありません。multi-head の「頭に裂く」列の切り出しも、Concat も、softmax も、layer norm もありません。

第5巻のファイルも変更禁止です(同じ理由——テスト済みの autograd 本体に手を入れない)。足りない演算は、この章のファイルで**外から**補います。やり方は第5巻4章で散々やったとおりです。forward を計算して出力ノードを作り、「親への勾配の配り方」を `_backward` に1つ書く。それだけです。実は第7巻の `embedding.py` がすでに同じ手口を使っています(`np.add.at` の backward を `Tensor` の外から差していました)。今回はそれを5回繰り返します。

`code/ch01/tensor_ops.py` の全文です。

```python
# 第8巻 第1章 1.2: 第5巻 Tensor に足りない演算の補完部品
# 第5巻・第7巻のファイルは変更しない(import のみ)という規律のため、
# 組み立てに必要だが tensor_autograd.py に無い演算はここで補う。
# backward の流儀は第5巻と同じ: 出力ノードに「親への勾配の配り方」を1つずつ書く。
# 実行: python3 tensor_ops.py — 全演算を数値微分(第2巻1章)で検算する。
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "vol5-backprop", "code", "ch05")))
sys.path.insert(0, os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "vol7-attention", "code", "ch03")))
from tensor_autograd import Tensor  # noqa: E402(第5巻5章)
from attention import softmax, NEG_INF  # noqa: E402(第7巻3章。forward を共用する)


def t_transpose(X):
    """転置。forward が転置なら、backward も勾配を転置して返すだけ。"""
    out = Tensor(X.data.T, (X,))

    def _backward():
        X.grad += out.grad.T

    out._backward = _backward
    return out


def t_cols(X, j0, j1):
    """列の切り出し X[:, j0:j1](multi-head の「頭に裂く」用 — 第7巻4章)。
    backward は切り出した場所にだけ勾配を戻す(他の列の勾配は別の頭が運ぶ)。"""
    out = Tensor(X.data[:, j0:j1], (X,))

    def _backward():
        X.grad[:, j0:j1] += out.grad

    out._backward = _backward
    return out


def t_concat_cols(parts):
    """列方向の連結 = 論文の Concat(head_1, ..., head_h)。t_cols の逆操作。"""
    out = Tensor(np.concatenate([p.data for p in parts], axis=1), tuple(parts))

    def _backward():
        j = 0
        for p in parts:
            w = p.data.shape[1]
            p.grad += out.grad[:, j:j + w]
            j += w

    out._backward = _backward
    return out


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


def t_layer_norm(X, gamma, beta, eps=1e-5):
    """layer norm の Tensor 版。forward は第5巻6.3(= 第7巻2章 stack_skeleton)と同じ式、
    backward も第5巻6.3 で数値微分と照合済みの式をそのまま使う。"""
    mu = X.data.mean(axis=-1, keepdims=True)
    var = X.data.var(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (X.data - mu) * inv_std
    out = Tensor(gamma.data * x_hat + beta.data, (X, gamma, beta))

    def _backward():
        # mu と var も x の関数なので、x への勾配には補正項が2つ付く(第5巻6.3)
        dx_hat = out.grad * gamma.data
        X.grad += inv_std * (dx_hat
                             - dx_hat.mean(axis=-1, keepdims=True)
                             - x_hat * (dx_hat * x_hat).mean(axis=-1, keepdims=True))
        gamma.grad += (out.grad * x_hat).sum(axis=0)
        beta.grad += out.grad.sum(axis=0)

    out._backward = _backward
    return out


if __name__ == "__main__":
    # 全演算を数値微分で検算する(第5巻4章で micrograd にやったのと同じ儀式)
    rng = np.random.default_rng(42)

    def numerical_grad(param, loss_fn, eps=1e-6):
        g = np.zeros_like(param)
        flat, gf = param.reshape(-1), g.reshape(-1)
        for i in range(flat.size):
            old = flat[i]
            flat[i] = old + eps
            fp = loss_fn()
            flat[i] = old - eps
            fm = loss_fn()
            flat[i] = old
            gf[i] = (fp - fm) / (2 * eps)
        return g

    def check(name, tensors, build_loss):
        """build_loss() が組む損失について、全 tensors の autograd 勾配を数値微分と照合"""
        loss = build_loss()
        for t in tensors:
            t.grad = np.zeros_like(t.data)
        loss.backward()
        for t in tensors:
            num = numerical_grad(t.data, lambda: build_loss().data)
            assert np.allclose(t.grad, num, atol=1e-5), name
        print("  ok: " + name)

    n, d, h = 4, 8, 2
    R1 = rng.standard_normal((d, n))   # 損失 L = Σ(out ⊙ R) の係数(検算の定石)
    R2 = rng.standard_normal((n, 3))
    R3 = rng.standard_normal((n, n))
    R4 = rng.standard_normal((n, d))

    X = Tensor(rng.standard_normal((n, d)))
    check("t_transpose", [X], lambda: (t_transpose(X) * R1).sum())
    check("t_cols", [X], lambda: (t_cols(X, 2, 5) * R2).sum())

    A_, B_ = Tensor(rng.standard_normal((n, 3))), Tensor(rng.standard_normal((n, 5)))
    R5 = rng.standard_normal((n, 8))
    check("t_concat_cols", [A_, B_], lambda: (t_concat_cols([A_, B_]) * R5).sum())

    Z = Tensor(rng.standard_normal((n, n)))
    mask = np.tril(np.ones((n, n), dtype=bool))   # causal mask で検算
    check("t_masked_softmax", [Z], lambda: (t_masked_softmax(Z, mask) * R3).sum())
    # mask された位置(未来)の勾配は厳密に 0
    assert np.all(Z.grad[np.triu_indices(n, k=1)] == 0.0)

    gamma, beta = Tensor(rng.standard_normal(d)), Tensor(rng.standard_normal(d))
    check("t_layer_norm", [X, gamma, beta],
          lambda: (t_layer_norm(X, gamma, beta) * R4).sum())

    print("tensor_ops: すべての assert を通過しました")
```

3点だけ補足します。

第一に、`t_masked_softmax` の forward は第7巻3章の `softmax` と `NEG_INF` を import して使っています。自分で書き直さないのは手抜きではなく仕様です——あとで NumPy 版と Tensor 版の一致を確かめるとき、forward が同じ関数なら、一致は「実装の偶然」ではなく「設計の必然」になります。

第二に、softmax の backward $dZ = A \odot (dA - \sum_k dA_k A_k)$ は第4巻6章で導出した式です。mask された位置は重み $A$ が厳密に 0 なので、この式に通すと勾配も自動的に 0 になります。「未来へは勾配も流れない」が、特別な処理なしで成立しているわけです。

第三に、`__main__` では5つの演算すべてを数値微分(第2巻1章の中心差分)と照合しています。backward を手書きしたら数値微分で検算する——第5巻4章で micrograd に課したのと同じ儀式です。手書きの backward は、検算が通って初めて部品になります。

### 組み上げ

部品が揃いました。組みます。`code/ch01/transformer.py` の全文です。長く見えますが、新しい計算は1つもありません。import した部品を、図1の配線どおりに呼んでいるだけです。

```python
# 第8巻 第1章 1.2: 組み立て — 第7巻の部品から Transformer へ
# 部品は1つも作り直さない。第7巻の各章の実装をそのまま import して、
# EncoderLayer / DecoderLayer → Encoder / Decoder → Transformer の順に箱を重ねる。
# forward は2系統:
#   forward_numpy : 第7巻の部品(NumPy)による forward。正しさの基準器
#   forward_tensor: 同じ重みを第5巻 Tensor で流す forward。学習可能(backward が通る)
# 実行: python3 transformer.py — 組み上がった全体の shape を確認する。
import os
import sys
import warnings

import numpy as np

# 環境固有の補足: macOS の Accelerate BLAS には、行列が一定サイズ以上になると
# matmul が偽の浮動小数点警告(divide by zero 等)を出す既知の不具合がある。
# 計算結果は正しい(下の __main__ と integration_test.py が isfinite と
# einsum 照合で別途確認している)ので、この偽警告だけを黙らせる。
warnings.filterwarnings("ignore", message=".*encountered in matmul",
                        category=RuntimeWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))
_V7 = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "vol7-attention", "code"))
_V5 = os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "vol5-backprop", "code", "ch05"))
for _p in ["ch02", "ch03", "ch04", "ch06", "ch07"]:
    sys.path.insert(0, os.path.join(_V7, _p))
sys.path.insert(0, _V5)

from stack_skeleton import encoder_layer, decoder_layer  # noqa: E402(第7巻2章)
from attention import causal_mask                        # noqa: E402(第7巻3章)
from multi_head import multi_head_attention              # noqa: E402(第7巻4章)
from position_wise_ffn import PositionwiseFFN            # noqa: E402(第7巻6章)
from embedding import Embedding, output_logits           # noqa: E402(第7巻6章)
from positional_encoding import positional_encoding      # noqa: E402(第7巻7章)
from tensor_autograd import Tensor                       # noqa: E402(第5巻5章)
from tensor_ops import (t_cols, t_concat_cols, t_transpose,  # noqa: E402(本章 1.2)
                        t_masked_softmax, t_layer_norm)


def mha_tensor(X_q, X_kv, W_q, W_k, W_v, W_o, h, mask=None):
    """multi-head attention の Tensor 版。式は第7巻4章 multi_head.py と同一で、
    「頭に裂く」を reshape の代わりに列の切り出しで書いたもの(中身は同じ列ブロック
    — 第7巻4章 split_heads の約束)。第5巻 Tensor の行列積は2次元限定なので、
    h 個の頭は for ループで回す。"""
    Q = X_q @ W_q                          # (n, d_model)
    K = X_kv @ W_k                         # (m, d_model)
    V = X_kv @ W_v                         # (m, d_model)
    d_model = Q.data.shape[1]
    d_k = d_model // h
    heads = []
    for i in range(h):
        Qi = t_cols(Q, i * d_k, (i + 1) * d_k)        # (n, d_k) head i の担当列
        Ki = t_cols(K, i * d_k, (i + 1) * d_k)        # (m, d_k)
        Vi = t_cols(V, i * d_k, (i + 1) * d_k)        # (m, d_v)
        S = (Qi @ t_transpose(Ki)) * (1.0 / np.sqrt(d_k))  # QK^T/√d_k : (n, m)
        A = t_masked_softmax(S, mask)                 # (n, m) 行ごとの和が1
        heads.append(A @ Vi)                          # (n, d_v)
    return t_concat_cols(heads) @ W_o                 # Concat(...) W^O : (n, d_model)


def _attn_weights(d_model, rng):
    """W^Q, W^K, W^V, W^O の4枚。初期スケールは 1/√d_model(第5巻6.6 Xavier)。"""
    s = 1.0 / np.sqrt(d_model)
    return [Tensor(rng.standard_normal((d_model, d_model)) * s) for _ in range(4)]


def _ln_params(d_model):
    """layer norm の γ=1, β=0(第5巻6.3 と同じ初期値)。"""
    return Tensor(np.ones(d_model)), Tensor(np.zeros(d_model))


class EncoderLayer:
    """論文 3.1 encoder の1層 = self-attention + FFN(各々 Add & Norm 付き)。"""

    def __init__(self, d_model, d_ff, h, rng):
        self.h = h
        self.W_q, self.W_k, self.W_v, self.W_o = _attn_weights(d_model, rng)
        self.ffn = PositionwiseFFN(d_model, d_ff, rng)        # 第7巻6章の部品
        self.gamma1, self.beta1 = _ln_params(d_model)
        self.gamma2, self.beta2 = _ln_params(d_model)

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

    def params(self):
        return ([self.W_q, self.W_k, self.W_v, self.W_o] + self.ffn.params()
                + [self.gamma1, self.beta1, self.gamma2, self.beta2])


class DecoderLayer:
    """論文 3.1 decoder の1層 = masked self-attention + cross-attention + FFN。
    cross-attention 側の重みは U と書いて self 側の W と区別する(論文は両方 W)。"""

    def __init__(self, d_model, d_ff, h, rng):
        self.h = h
        self.W_q, self.W_k, self.W_v, self.W_o = _attn_weights(d_model, rng)
        self.U_q, self.U_k, self.U_v, self.U_o = _attn_weights(d_model, rng)
        self.ffn = PositionwiseFFN(d_model, d_ff, rng)
        self.gamma1, self.beta1 = _ln_params(d_model)
        self.gamma2, self.beta2 = _ln_params(d_model)
        self.gamma3, self.beta3 = _ln_params(d_model)

    def forward_numpy(self, x, memory, causal=True):
        """第7巻2章 decoder_layer に、第5章の3つの使い方どおりの部品を差す。
        causal=False は「わざと mask を外して結合テストを壊す」実験用(1.3)。"""
        mask = causal_mask(x.shape[0]) if causal else None
        prm = {"gamma1": self.gamma1.data, "beta1": self.beta1.data,
               "gamma2": self.gamma2.data, "beta2": self.beta2.data,
               "gamma3": self.gamma3.data, "beta3": self.beta3.data}

        def self_attn(x_):
            out, _ = multi_head_attention(x_, x_, self.W_q.data, self.W_k.data,
                                          self.W_v.data, self.W_o.data, self.h,
                                          mask=mask)
            return out

        def cross_attn(x_, m_):
            out, _ = multi_head_attention(x_, m_, self.U_q.data, self.U_k.data,
                                          self.U_v.data, self.U_o.data, self.h)
            return out

        def ffn_f(x_):
            return self.ffn(Tensor(x_)).data

        return decoder_layer(x, memory, prm, self_attn, cross_attn, ffn_f)

    def forward_tensor(self, X, memory, causal=True):
        mask = causal_mask(X.data.shape[0]) if causal else None
        a = mha_tensor(X, X, self.W_q, self.W_k, self.W_v, self.W_o, self.h, mask)
        X = t_layer_norm(X + a, self.gamma1, self.beta1)
        a = mha_tensor(X, memory, self.U_q, self.U_k, self.U_v, self.U_o, self.h)
        X = t_layer_norm(X + a, self.gamma2, self.beta2)
        X = t_layer_norm(X + self.ffn(X), self.gamma3, self.beta3)
        return X

    def params(self):
        return ([self.W_q, self.W_k, self.W_v, self.W_o,
                 self.U_q, self.U_k, self.U_v, self.U_o] + self.ffn.params()
                + [self.gamma1, self.beta1, self.gamma2, self.beta2,
                   self.gamma3, self.beta3])


class Encoder:
    """"a stack of N identical layers"(論文 3.1)。層を for で重ねるだけ。"""

    def __init__(self, N, d_model, d_ff, h, rng):
        self.layers = [EncoderLayer(d_model, d_ff, h, rng) for _ in range(N)]

    def forward_numpy(self, x):
        for layer in self.layers:
            x = layer.forward_numpy(x)
        return x

    def forward_tensor(self, X):
        for layer in self.layers:
            X = layer.forward_tensor(X)
        return X

    def params(self):
        return [p for layer in self.layers for p in layer.params()]


class Decoder:
    """N 層の decoder。memory(encoder 出力)は全層に同じものが配られる。"""

    def __init__(self, N, d_model, d_ff, h, rng):
        self.layers = [DecoderLayer(d_model, d_ff, h, rng) for _ in range(N)]

    def forward_numpy(self, x, memory, causal=True):
        for layer in self.layers:
            x = layer.forward_numpy(x, memory, causal=causal)
        return x

    def forward_tensor(self, X, memory, causal=True):
        for layer in self.layers:
            X = layer.forward_tensor(X, memory, causal=causal)
        return X

    def params(self):
        return [p for layer in self.layers for p in layer.params()]


class Transformer:
    """論文 図1 の全体。入口の embedding と出口の出力 head は1枚の E を共有し(3.4)、
    embedding 直後に positional encoding を足す(3.5)。dropout は訓練の道具なので
    第3章(PyTorch 側)で戻す。バッチは第5巻 Tensor の行列積が2次元限定のため
    1系列ずつ流す(束ねるのは第2章の仕事)。"""

    def __init__(self, vocab, d_model, d_ff, h, N, max_len, rng):
        self.emb = Embedding(vocab, d_model, rng)             # 第7巻6章(√d_model 倍込み)
        self.pe = positional_encoding(max_len, d_model)       # 第7巻7章(定数行列)
        self.encoder = Encoder(N, d_model, d_ff, h, rng)
        self.decoder = Decoder(N, d_model, d_ff, h, rng)

    def forward_numpy(self, src_ids, tgt_in, causal=True):
        """src_ids: 入力文 (src_len,)、tgt_in: decoder 入力 (tgt_len,)。
        返り値: logits (tgt_len, vocab)。"""
        x = self.emb(src_ids).data + self.pe[:len(src_ids)]   # 埋め込み + PE
        memory = self.encoder.forward_numpy(x)                # (src_len, d_model)
        y = self.emb(tgt_in).data + self.pe[:len(tgt_in)]
        y = self.decoder.forward_numpy(y, memory, causal=causal)
        return output_logits(Tensor(y), self.emb.E).data      # (tgt_len, vocab)

    def forward_tensor(self, src_ids, tgt_in, causal=True):
        """同じ計算の Tensor 版。返り値の Tensor から backward() が引ける。"""
        x = self.emb(src_ids) + Tensor(self.pe[:len(src_ids)])
        memory = self.encoder.forward_tensor(x)
        y = self.emb(tgt_in) + Tensor(self.pe[:len(tgt_in)])
        y = self.decoder.forward_tensor(y, memory, causal=causal)
        return output_logits(y, self.emb.E)

    def params(self):
        return self.emb.params() + self.encoder.params() + self.decoder.params()

    def n_params(self):
        return sum(p.data.size for p in self.params())


if __name__ == "__main__":
    # 組み上げの確認(1.2): 縮小版を1台組み、入口から出口まで shape が通ること
    rng = np.random.default_rng(42)
    vocab, d_model, d_ff, h, N = 12, 32, 64, 4, 2
    model = Transformer(vocab, d_model, d_ff, h, N, max_len=16, rng=rng)

    src = np.array([3, 1, 4, 1, 5, 9, 2])   # 長さ7の入力文(のつもり)
    tgt_in = np.array([0, 2, 7, 1, 8])      # 長さ5の decoder 入力。長さ違いで混線を検出

    logits = model.forward_numpy(src, tgt_in)
    assert logits.shape == (5, vocab)        # 出力の行数は decoder 側で決まる
    assert np.all(np.isfinite(logits))       # N 層通しても発散していない

    logits_t = model.forward_tensor(src, tgt_in)
    assert logits_t.data.shape == (5, vocab)

    print("Transformer 組み上げ OK: src(7,) + tgt(5,) -> logits {}".format(logits.shape))
    print("パラメータ総数: {:,}(検算は param_count.py、結合テストは integration_test.py)"
          .format(model.n_params()))
```

読みどころを順に挙げます。

**差し替えの瞬間。** `EncoderLayer.forward_numpy` を見てください。第7巻2章の `encoder_layer(x, prm, self_attn, ffn)` は、部分層を「`(seq_len, d_model)` を受け取って同じ shape を返す関数」として受け取る設計でした。あのとき差してあったのは恒等写像のダミーです。今回、その同じ穴に `multi_head_attention` と `PositionwiseFFN` を差しました。第7巻2章で「外側から作る」と言った骨組みが、約束どおり、1文字も変えずに本物を受け入れています。decoder 側も同様で、3つの部分層への部品の割り当て——masked self に causal mask、cross に `memory`——は第7巻5章の配線表のとおりです。

**`mha_tensor` は同じ式の別表記。** 第7巻4章の `split_heads` は reshape で頭に裂きましたが、`Tensor` の行列積は2次元限定なので、Tensor 版は列の切り出し(`t_cols`)+ for ループで裂いています。第7巻4章で確認したとおり、head $i$ の中身は $X W$ の列ブロックそのものなので、2つの書き方は同じ計算です(この同一性は次節の `allclose` が裏書きします)。ループは遅い書き方ですが、遅さはむしろ好都合です——1.4で「自作スタックの素の速度」を測るのですから。

**重みの共有は `.data` 経由で自動。** `forward_numpy` は各 `Tensor` の `.data` を読むだけなので、2系統の forward は常に同一の重みを見ています。「NumPy 版の重みを Tensor 版にコピーする」という手順そのものが存在しないため、コピー忘れというバグも存在できません。

**乱数と環境の注意を2つ。** 重みの初期化は `np.random.default_rng(42)` で固定します(シリーズの規律)。また、冒頭の `warnings.filterwarnings` は macOS の Accelerate BLAS が出す**偽の**浮動小数点警告(計算結果は正しいのに matmul が divide by zero を報告する既知の不具合)への対処です。「警告を黙らせる前に、それが偽であることを確かめる」のが筋なので、次節のテストに matmul と einsum の照合を1行入れてあります。

実行します。

```
$ python3 transformer.py
Transformer 組み上げ OK: src(7,) + tgt(5,) -> logits (5, 12)
パラメータ総数: 42,368(検算は param_count.py、結合テストは integration_test.py)
```

入力文(長さ7)と decoder 入力(長さ5)が、embedding から N=2 層の encoder・decoder を通り、`(5, 12)` の logits になって出てきました。図1が、初めて1つのプログラムとして動いた瞬間です。

ただし、shape が通っただけです。配管が正しいか、学習できるかは、まだ何も保証されていません。テストに進みます。

## 1.3 結合テスト: 全体 forward の shape、causal mask が末端まで効いているか

単体テストは部品の保証、結合テストは配線の保証です。部品が全部正しくても、配線を1本間違えれば全体は壊れます。そして Transformer の配線ミスには、たちの悪い性質があります——**shape が通ってしまう**ことが多いのです。mask を渡し忘れても、cross-attention に渡す行列を取り違えても、出てくる logits の形は `(tgt_len, vocab)` のまま。エラーは出ず、ただ「学習しても性能が出ない」という形で数日後に祟ります。

だから、shape の先を検査します。検査は4段です。

1. **shape**: 入口から出口まで通ること(最低限の関門)
2. **causal mask が末端まで効くこと**: 第7巻5章で attention 単体に課した検査——未来のトークンを改変しても過去の出力が変わらない——を、今度は**embedding から logits までの全経路**に課します。途中のどの1層が mask を取りこぼしても、この検査は落ちます
3. **NumPy 版と Tensor 版の一致**: 1.1 で設計した分業の要。さらに Tensor 版の勾配を数値微分でスポット照合します
4. **丸暗記テスト**: 1バッチを過学習できること

4つ目は、この機会に覚えてほしい**デバッグの定石**です。新しくモデルと訓練の仕組みを組んだら、本番データを流す前に、ごく小さな1バッチを用意して「これを丸暗記できるか」を試します。数百万パラメータのモデルにとって、数十トークンの暗記は造作もない仕事のはずです。それなのに loss がゼロ近くまで落ちないなら、データの並べ方・mask・損失・勾配のどこかが確実に壊れています。逆に落ちても正しさの証明にはなりませんが、「壊れてはいない」という安心は得られます。安くて感度の高い煙探知機です。

暗記させるデータには、**規則のないでたらめな対応**をわざと使います。入力列も正解列もただの乱数です。規則がないので「賢く一般化して解いた」可能性が消え、成功の解釈が「このバッチを記憶する勾配が、出口の損失から入口の embedding まで生きて流れた」の一通りに定まります。

`code/ch01/integration_test.py` の全文です。

```python
# 第8巻 第1章 1.3: 結合テスト — 組み上がった Transformer 全体を検査する
# 検査は4段:
#   (1) shape: 入口から出口まで形が通る
#   (2) causal mask が末端まで効く: 未来のトークンを改変しても過去の logits は不変
#   (3) NumPy 版(第7巻部品)と Tensor 版(第5巻 autograd)の forward が一致
#       + Tensor 版の勾配を数値微分でスポット照合
#   (4) 丸暗記テスト: 1バッチを過学習できる(デバッグの定石)
# 実行: python3 integration_test.py(seed 42。数十秒で全 assert が通る)
import numpy as np

from transformer import Transformer, Tensor
from tensor_autograd import softmax_cross_entropy  # 第5巻5章(path は transformer が通す)

rng = np.random.default_rng(42)
vocab, d_model, d_ff, h, N = 12, 32, 64, 4, 2      # base の縮小版(比率は同じ d_ff = 4d)
model = Transformer(vocab, d_model, d_ff, h, N, max_len=16, rng=rng)

BOS = 0                                            # 文頭トークン(規約の本番は第2章)
src = rng.integers(1, vocab, size=7)               # 入力文(長さ7)
tgt = rng.integers(1, vocab, size=5)               # 出力文(長さ5)。長さ違いで混線を検出
tgt_in = np.concatenate([[BOS], tgt[:-1]])         # teacher forcing の1トークンずらし(第6巻6.4)

# ---- (0) 環境の自己点検: BLAS の matmul を einsum と照合(transformer.py の偽警告の件) ----
A0 = rng.standard_normal((7, d_model))
B0 = rng.standard_normal((d_model, d_model))
assert np.array_equal(A0 @ B0, np.einsum("ij,jk->ik", A0, B0))

# ---- (1) shape: 全体 forward が通り、出力が (tgt_len, vocab) ----
logits = model.forward_numpy(src, tgt_in)
assert logits.shape == (5, vocab)
assert np.all(np.isfinite(logits))
print("(1) shape OK: src (7,) + tgt_in (5,) -> logits {}".format(logits.shape))

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

# 入力文の改変は cross-attention 経由で「全位置」に届く(これは漏れではなく仕様)
src2 = src.copy()
src2[0] = (src[0] + 1) % vocab
logits3 = model.forward_numpy(src2, tgt_in)
assert not np.allclose(logits3[0], logits[0])
print("(2) causal mask OK: 未来の改変は過去に漏れない(mask を外すと漏れる)")

# ---- (3) NumPy 版と Tensor 版の forward 一致 + 勾配のスポット照合 ----
logits_t = model.forward_tensor(src, tgt_in)
diff = np.abs(logits_t.data - logits).max()
assert diff < 1e-9
print("(3) NumPy/Tensor 一致 OK: logits の最大差 = {:.2e}".format(diff))

params = model.params()
assert len(set(id(p) for p in params)) == len(params)   # 二重カウントなし(1.6 の前提)


def batch_loss():
    """損失 = 正解トークン列 tgt に対する cross-entropy(第4巻5章)。"""
    return softmax_cross_entropy(model.forward_tensor(src, tgt_in), tgt)


loss = batch_loss()
for p in params:
    p.grad = np.zeros_like(p.data)
loss.backward()

# 数値微分(第2巻1章の中心差分)でスポット照合: 入口・中間・出口から1枚ずつ
spots = [("embedding E", model.emb.E, (int(src[0]), 0)),
         ("encoder層0 W_q", model.encoder.layers[0].W_q, (3, 7)),
         ("decoder層1 gamma3", model.decoder.layers[1].gamma3, (2,)),
         ("decoder層0 cross U_o", model.decoder.layers[0].U_o, (0, 5))]
eps = 1e-5
for name, p, idx in spots:
    old = p.data[idx]
    p.data[idx] = old + eps
    fp = batch_loss().data
    p.data[idx] = old - eps
    fm = batch_loss().data
    p.data[idx] = old
    num = (fp - fm) / (2 * eps)
    assert np.isclose(p.grad[idx], num, rtol=1e-4, atol=1e-7), name
    print("    勾配照合 OK: {:<18} autograd {:+.8f} / 数値微分 {:+.8f}"
          .format(name, p.grad[idx], num))

# ---- (4) 丸暗記テスト: 1バッチ(2系列)を過学習できるか ----
# 対応関係に規則のない src -> tgt のペアを2本固定し、丸暗記させる。
# 規則がないので「学習できた」= 「このバッチを記憶する配管と勾配が末端まで生きている」。
pairs = []
for _ in range(2):
    s = rng.integers(1, vocab, size=6)
    t = rng.integers(1, vocab, size=6)
    pairs.append((s, np.concatenate([[BOS], t[:-1]]), t))

lr, num_steps = 0.5, 300
history = []
for step in range(num_steps):
    total = None
    for s, t_in, t_out in pairs:                       # 1. forward + 2. loss
        l_one = softmax_cross_entropy(model.forward_tensor(s, t_in), t_out)
        total = l_one if total is None else total + l_one
    loss = total * (1.0 / len(pairs))
    for p in params:                                   # 3. gradient
        p.grad = np.zeros_like(p.data)
    loss.backward()
    for p in params:                                   # 4. update(素朴な勾配降下)
        p.data -= lr * p.grad
    history.append(loss.data)
    if step in (0, 50, 100, num_steps - 1):
        print("    step {:>4}: loss = {:.6f}".format(step, float(loss.data)))

assert history[0] > 1.5                # 初期はほぼ当てずっぽう(ln 12 ≈ 2.48 付近)
assert history[-1] < 0.05              # 丸暗記完了
assert history[-1] < history[0] / 20   # 大きく下がった

# 丸暗記の確認: teacher forcing 下で argmax が正解列を完全再生する
for s, t_in, t_out in pairs:
    pred = np.argmax(model.forward_numpy(s, t_in), axis=-1)
    assert np.array_equal(pred, t_out)
print("(4) 丸暗記 OK: loss {:.3f} -> {:.4f}、2系列とも完全再生"
      .format(history[0], history[-1]))

print("integration_test: すべての assert を通過しました")
```

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

**(2) の検査は「故意に壊す」までがワンセットです。** 未来の改変が過去に漏れないことを確認するだけでなく、`causal=False` で mask を外したら同じ検査が**落ちる**ことも確認しています。テストは「通る」だけでは信用できません。バグを入れたら落ちる——その感度を見せて初めて、「通った」に意味が出ます。なお過去の位置の比較に `allclose` ではなく `array_equal` を使えるのは、mask が「未来の重みを小さくする」のではなく「厳密に 0 にする」仕掛けだからです(第7巻3章の $-\infty$ の効能がここまで届いています)。

**(3) の一致が、この章の蝶番です。** 最大差 `1.78e-15`——浮動小数点の丸めの粒まで、2系統は同じ計算でした。第7巻の部品が持つ「論文どおり」という信用が、これで Tensor 版に乗り移りました。続く勾配照合は、`tensor_ops.py` の単体検算では届かない「全部つないだときの backward」への抜き取り検査です。入口(embedding)、中間(encoder の $W^Q$、cross-attention の $U^O$)、出口近く(decoder 最終層の $\gamma$)の4点で、autograd の勾配と実測の傾きが8桁一致しています。

**(4) で、Transformer が初めて学習しました。** このシリーズで組んだ最大のモデル(といっても4万パラメータですが)が、loss 3.01——ほぼ当てずっぽう($\ln 12 \approx 2.48$ よりやや上)——から 0.001 まで降りて、2系列を完全に暗記しました。使った訓練ループを見てください。forward → loss → backward → update。第3巻4章で線形回帰に使った**4拍子と、1拍も違いません**。モデルがどれだけ育っても、学習の骨格は変わらない——シリーズを貫いてきたこの主張の、これが最終確認です。

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

つまり論文の訓練は「1ステップで約25000ターゲットトークンを処理し、それを10万回」です。私たちの自作スタックは、1ステップで何トークン処理でき、1ステップに何秒かかるのか。**base model と同じ構成を実際に組んで**測ります。

`code/ch01/limit_check.py` の全文です。

```python
# 第8巻 第1章 1.4: 自作スタックの限界を実測する
# 論文 base model と同じ構成(N=6, d_model=512, d_ff=2048, h=8, 語彙37000)を
# 自作スタックで1台組み、訓練1ステップの所要時間を実測する。
# そこから「論文と同じ訓練(25000トークン/step × 100,000 step)」の総時間を見積もり、
# 物理的に終わらないことを数字で確認する。PyTorch への需要はここで初めて発生する。
# 実行: python3 limit_check.py(1分弱。メモリを 1GB 強使う)
import time

import numpy as np

from transformer import Transformer
from tensor_autograd import softmax_cross_entropy  # 第5巻5章(path は transformer が通す)

rng = np.random.default_rng(42)

# 論文 3.1 Table 3 base / 5.1 の構成そのまま
vocab, d_model, d_ff, h, N = 37000, 512, 2048, 8, 6
src_len = tgt_len = 32                     # 1ステップに流す系列(バッチはこの1対だけ)

print("base model(N=6, d_model=512, d_ff=2048, h=8, 語彙{:,})を組み立て中...".format(vocab))
t0 = time.perf_counter()
model = Transformer(vocab, d_model, d_ff, h, N, max_len=64, rng=rng)
params = model.params()
n_params = model.n_params()
print("  組み立て {:.1f} 秒、パラメータ {:,}(≈ {:.0f}M — 第7巻6章の検算と同じ規模)"
      .format(time.perf_counter() - t0, n_params, n_params / 1e6))
assert n_params > 60e6                     # 確かに「論文サイズ」を持ち上げている

src = rng.integers(1, vocab, size=src_len)
tgt = rng.integers(1, vocab, size=tgt_len)
tgt_in = np.concatenate([[0], tgt[:-1]])


def one_step():
    """4拍子1回 = 訓練1ステップ(forward → loss → backward → update)。"""
    loss = softmax_cross_entropy(model.forward_tensor(src, tgt_in), tgt)
    for p in params:
        p.grad = np.zeros_like(p.data)
    loss.backward()
    for p in params:
        p.data -= 1e-4 * p.grad            # 更新も計測に含める(本番は毎step行うので)
    return loss.data


one_step()                                 # ウォームアップ(初回はキャッシュ等で遅い)
n_trials = 3
t0 = time.perf_counter()
for _ in range(n_trials):
    one_step()
t_step = (time.perf_counter() - t0) / n_trials
print("  訓練1ステップ(32トークンの文対1本): {:.2f} 秒".format(t_step))
assert t_step > 0.0

# --- 見積もり: 論文 5.1・5.3 の訓練量に外挿する ---
# 5.1: 1バッチ ≈ 25000 ターゲットトークン。5.3: base は 100,000 ステップ。
# 自作スタックは1ステップで 32 ターゲットトークンしか処理していないので、
# 同じトークン量を流すには 25000/32 ≈ 781 倍の時間がかかる(線形外挿。
# 実際は系列が長いほど attention が O(n^2) で重くなるから、これでも甘めの見積もり)
tokens_paper = 25000
steps_paper = 100000
scale = tokens_paper / tgt_len
total_sec = t_step * scale * steps_paper
total_days = total_sec / 86400
total_years = total_days / 365.0

print()
print("  論文の訓練量: {:,} トークン/step × {:,} step".format(tokens_paper, steps_paper))
print("  必要時間 = {:.2f} 秒 × {:.0f} × {:,} = {:.2e} 秒".format(
    t_step, scale, steps_paper, total_sec))
print("  = 約 {:,.0f} 日 = 約 {:.1f} 年(このマシン1台、自作スタックで)".format(
    total_days, total_years))
print("  論文の実測: 8 × P100 GPU で 12 時間(5.2)")

assert total_days > 30                     # どんなに速いマシンでも「月」では終わらない
print()
print("limit_check: すべての assert を通過しました — 物理的に終わらない。卒業の時です")
```

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

まず認めるべきことを認めます。63M パラメータの本物サイズの Transformer を、自作スタックは**ちゃんと持ち上げました**。組めて、forward が通り、backward まで回る。1ステップ 0.11 秒という数字は、第5巻5.3でスカラー `Value` の遅さに絶望した身からすれば、むしろ健闘です(行列演算の中身は NumPy、つまり最適化された BLAS が走っているからです)。

それでも、論文の訓練量を前にすると桁が足りません。昼夜止めずに回して**約100日**。論文の big model は 300,000 ステップですから、その再現なら**1年弱**。さらに言えば、論文の Table 3 は数十通りの構成を訓練して比べたアブレーションです。1回の訓練に100日かかる道具で、それは研究として成立しません。しかもこの100日という数字は、いくつもの甘い仮定(トークン数に線形で外挿、Adam も dropout もないぶん軽い、など——演習3)の上に立った**楽観値**です。

足りないものは2つあります。第一に **GPU**。行列積を数千コアで並列に行うハードウェアと、それを呼び出す仕組み。第二に、バッチ・メモリ管理・最適化された backward を備えた**実戦用のスタック**。どちらも、原理は全部この手で作りました。しかし原理の理解と、100日を12時間にする工学は、別の仕事です。その仕事を引き受けてくれるのが PyTorch です。

第2巻で勾配降下を学んだとき、第5巻で autograd を作ったとき、PyTorch を使う「需要」はまだありませんでした——自作で間に合っていたからです。いま、初めて間に合わなくなりました。**需要が発生したので、道具を解禁します。** これが序章0.3で予告した、シリーズでただ一度の乗り換えです。

## 1.5 卒業と橋渡し: 「自作API ↔ PyTorch API 対応表」

乗り換えにあたって、橋を1枚架けておきます。下の表は「これから出会う PyTorch の道具」と「あなたがすでに作った道具」の対応表です。次章から PyTorch の API が登場するたびに、本書は脚注で「自作版のどれにあたるか」をこの表の行で示します。PyTorch を魔法の箱として使うのではなく、「自分が作ったあれの、速くて頑丈な版」として使うためです。

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

この表に**載っていない**ものにも注意してください。`nn.Transformer` と `nn.MultiheadAttention`——PyTorch には Transformer の完成品が入っていますが、本書では最後まで使いません。使った瞬間、第7巻でやったことが「車輪の再発明」に格下げされてしまうからです。第2章以降のモデルは、素のテンソル演算とこの表にある最小限の部品だけで、もう一度自分の手で書きます(といっても、設計はこの章で済んでいます。書き直しは半日仕事です)。

表にはまだ空行があります。たとえば layer norm の行がありません。これは演習2であなたが埋めます——卒業証書には自分の署名欄があるものです。

## 1.6 パラメータ数の検算: base 相当の縮小版で数える

卒業制作の最後の仕上げは、検収です。組み上げた Transformer が「余計な重みを持っていないか、必要な重みを欠いていないか」を、第7巻6章の演習で作った**紙の上の数え上げ**と突き合わせて確かめます。式と実物が1個までぴったり合えば、組み立ては過不足なしです。

`code/ch01/param_count.py` の全文です。

```python
# 第8巻 第1章 1.6: パラメータ数の検算 — 紙の上の式と、組み上げた実物を突き合わせる
# 第7巻6章の演習で作った数え上げの式(transformer_base_params)を import し、
# 1.2 で組み立てた Transformer の「実際に持っている数」と完全一致することを assert する。
# 式と実物が1個までぴったり合えば、組み立てに余計な重みも欠けた重みもない。
# 実行: python3 param_count.py
import os
import sys

import numpy as np

from transformer import Transformer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "vol7-attention", "code", "ch06")))
from ex_param_count import transformer_base_params  # noqa: E402(第7巻6章 演習1)

rng = np.random.default_rng(42)

# base 相当の縮小版: N=6, h=8, d_ff = 4 × d_model の比率はそのまま、幅と語彙だけ 1/8
vocab, d_model, d_ff, h, N = 1000, 64, 256, 8, 6
model = Transformer(vocab, d_model, d_ff, h, N, max_len=32, rng=rng)

params = model.params()
assert len(set(id(p) for p in params)) == len(params)  # 同じ行列の二重カウントなし
actual = model.n_params()

breakdown, expected = transformer_base_params(vocab=vocab, d_model=d_model,
                                              d_ff=d_ff, N=N)
print("縮小版(N={}, d_model={}, d_ff={}, h={}, 語彙{:,})の内訳(式による計算):"
      .format(N, d_model, d_ff, h, vocab))
for name, n in breakdown.items():
    print("  {:<22} {:>10,}".format(name, n))
print("  {:<22} {:>10,}".format("合計(式)", expected))
print("  {:<22} {:>10,}".format("合計(実物)", actual))

# 紙の上の式 = 組み上げた実物。1個の過不足もなく一致する
assert actual == expected

# h を変えても総数は変わらない(d_k = d_model / h に裂いているだけ — 第7巻4章)
model_h4 = Transformer(vocab, d_model, d_ff, 4, N, max_len=32, rng=rng)
assert model_h4.n_params() == actual

# 本家 base の 65M(第7巻6章の検算)も、同じ式の引数を変えるだけで再現できる
_, total_base = transformer_base_params()   # vocab=37000, d_model=512, d_ff=2048, N=6
assert round(total_base / 1e6) == 63        # この数え方では 63M(論文 Table 3 は 65M)

print("param_count: すべての assert を通過しました — 式と実物が {:,} 個で完全一致"
      .format(actual))
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

759,808 個、ぴったり一致です。第7巻6章で「論文の 65M を数えられる」ようになった式が、今度は**自分の実物の検収**に使えました。ついでに2つの事実も assert で確認しています。head 数 $h$ を変えてもパラメータ総数は変わらないこと($d_{model}$ を $h$ 等分に裂いているだけだからです——第7巻4章4.5)。そして 1.4 で持ち上げた本家サイズの 63M も、同じ式の引数を変えるだけで出てくること。

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

**問2(TOC 指定: 対応表に1行を自分で追加する)** 1.5 の対応表には layer norm の行がありません。本章の `t_layer_norm`(と第5巻6.3の `layer_norm`)に対応する PyTorch API を調べて、表の書式で1行追加してください。「備考」欄には、自作版との対応で気づいたことを1つ書くこと。余力があれば dropout(第5巻6章)の行も足してみてください。

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
