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
