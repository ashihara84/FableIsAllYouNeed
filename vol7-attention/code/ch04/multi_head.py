# 第7巻 第4章 4.4: Multi-Head Attention(論文 Section 3.2.2)の単体実装
#   MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
#   head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
# 第8巻はこのファイルを import して Transformer を組み立てる(基盤モジュール)。
# 実行: python3 multi_head.py で自己点検。テスト本体は test_multi_head.py。
import os
import sys

import numpy as np

# 第3章で実装した attention(式(1))を、部品としてそのまま使う
_CH03_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ch03")
sys.path.insert(0, _CH03_DIR)
from attention import attention  # noqa: E402(パス追加後の import)


def split_heads(X, h):
    """(seq, d) -> (h, seq, d_k): 1枚の行列を h 冊の「行列の束」に分ける(第1巻6.4)。

    head i の中身は、X の列ブロック X[:, i*d_k:(i+1)*d_k] と同じ。
    軸を後ろから数えて整形するので、先頭にバッチ軸が付いた
    (batch, seq, d) -> (batch, h, seq, d_k) もこのまま動く(序章0.3の規格)。
    """
    d = X.shape[-1]
    assert d % h == 0, "d_model は h で割り切れること(論文: 512 = 8 x 64)"
    d_k = d // h
    X = X.reshape(X.shape[:-1] + (h, d_k))  # (..., seq, d) -> (..., seq, h, d_k)
    return np.swapaxes(X, -3, -2)           # (..., seq, h, d_k) -> (..., h, seq, d_k)


def combine_heads(Y):
    """(h, seq, d_v) -> (seq, h*d_v): 束を1枚に戻す。論文の Concat(head_1, ..., head_h)。

    split_heads の逆操作。同じく (batch, h, seq, d_v) -> (batch, seq, h*d_v) も動く。
    """
    h, d_v = Y.shape[-3], Y.shape[-1]
    Y = np.swapaxes(Y, -3, -2)              # (..., h, seq, d_v) -> (..., seq, h, d_v)
    return Y.reshape(Y.shape[:-2] + (h * d_v,))  # (..., seq, h, d_v) -> (..., seq, h*d_v)


def multi_head_attention(X_q, X_kv, W_q, W_k, W_v, W_o, h, mask=None):
    """Multi-Head Attention(論文 Section 3.2.2)。

    X_q : (n, d_model)  query 側の入力(self-attention では X_kv と同じ配列を渡す)
    X_kv: (m, d_model)  key / value 側の入力
    W_q, W_k, W_v: (d_model, d_model)  h 個の射影 W_i^Q, W_i^K, W_i^V を横に並べたもの
    W_o : (d_model, d_model)  出力射影 W^O
    h   : head の個数(論文では 8)
    mask: (n, m) に broadcast できる bool 配列。True = 見てよい位置。全 head 共通
    返り値: (出力 (n, d_model), attention 重み (h, n, m))
    入力の先頭にバッチ軸が付けば、出力は (batch, n, d_model)、重みは (batch, h, n, m)。
    """
    Q = split_heads(X_q @ W_q, h)   # (n, d_model) -> (h, n, d_k)
    K = split_heads(X_kv @ W_k, h)  # (m, d_model) -> (h, m, d_k)
    V = split_heads(X_kv @ W_v, h)  # (m, d_model) -> (h, m, d_v)
    heads, weights = attention(Q, K, V, mask)  # (h, n, d_v), (h, n, m)  h 冊ぶん一気に
    Y = combine_heads(heads) @ W_o  # (n, h*d_v) @ (h*d_v, d_model) -> (n, d_model)
    return Y, weights


if __name__ == "__main__":
    # 自己点検(テスト本体は test_multi_head.py)
    rng = np.random.default_rng(42)
    n, d_model, h = 5, 16, 4
    X = rng.standard_normal((n, d_model))
    W_q, W_k, W_v, W_o = (rng.standard_normal((d_model, d_model)) for _ in range(4))
    Y, A = multi_head_attention(X, X, W_q, W_k, W_v, W_o, h)  # self-attention
    assert Y.shape == (n, d_model)          # 出口は入口と同じ形(積み重ね可能)
    assert A.shape == (h, n, n)             # 重みは head ごとに1枚、計 h 枚
    assert np.allclose(A.sum(axis=-1), 1.0)
    print("ok: multi_head.py 自己点検を通過(attention は第3章の attention.py を import)")
