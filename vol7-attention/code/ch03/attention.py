# 第7巻 第3章 3.4: Scaled Dot-Product Attention — 論文 式(1) の単体実装
#   Attention(Q, K, V) = softmax(QK^T / √d_k) V
# 第8巻はこのファイルを import して Transformer を組み立てる(基盤モジュール)。
import numpy as np

# 「−∞ を足す」の実装上の代用値。softmax を通すと exp(-1e9) は完全に 0 と
# みなされるので、重みは厳密に 0 になる(本物の -np.inf を使うと、行が全部
# mask されたときに nan が出て事故の原因になるため、有限の大きな負数を使う)
NEG_INF = -1e9


def softmax(z, axis=-1):
    """数値安定版 softmax(第4巻6.2)。axis 方向の和が 1 になる。"""
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=axis, keepdims=True)  # 最大値シフト: exp(正の大数) を起こさない
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q, K, V, mask=None):
    """Scaled Dot-Product Attention(論文 Section 3.2.1, 式(1))。

    Q: (..., n, d_k), K: (..., m, d_k), V: (..., m, d_v)。
    mask: (..., n, m) にブロードキャスト可能な bool 配列。True = 見てよい位置。
    返り値: (output (..., n, d_v), weights (..., n, m))。weights は行ごとに和が1。

    先頭の ... はバッチ次元(第1巻6.4「行列の束」)。@ は最後の2軸だけを行列積に
    使うので、(n, d_k) 単体でも (batch, h, n, d_k) でも同じ式で動く(第4章で使う)。
    """
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)  # QK^T / √d_k : (..., n, m)
    if mask is not None:
        scores = np.where(mask, scores, NEG_INF)        # softmax の前に −∞(相当)を置く
    weights = softmax(scores, axis=-1)                  # 行ごとに和が 1 : (..., n, m)
    return weights @ V, weights                         # 重み付き和    : (..., n, d_v)


def causal_mask(n):
    """decoder 用の causal mask (n, n)。下三角が True = 自分と過去だけ見てよい。
    第6巻6.4の自己回帰性をテーブルにしたもの。"""
    return np.tril(np.ones((n, n), dtype=bool))


def padding_mask(is_real):
    """padding mask。is_real: (m,) の bool(True = 本物のトークン、False = 埋め草)。
    (1, m) に整形して返す(ブロードキャストで全 query に同じ禁止が掛かる)。"""
    return np.asarray(is_real, dtype=bool)[np.newaxis, :]


if __name__ == "__main__":
    # 動作確認: 式(1)の配管 (n, d_k) → (n, m) → (n, d_v) が通ること
    rng = np.random.default_rng(42)
    n, m, d_k, d_v = 4, 6, 64, 64
    Q = rng.standard_normal((n, d_k))
    K = rng.standard_normal((m, d_k))
    V = rng.standard_normal((m, d_v))

    output, weights = attention(Q, K, V)
    assert output.shape == (n, d_v)
    assert weights.shape == (n, m)
    assert np.allclose(weights.sum(axis=-1), 1.0)
    print("attention の配管 OK: ({}, {}) @ ({}, {})^T -> 重み ({}, {}) -> 出力 ({}, {})".format(
        n, d_k, m, d_k, n, m, n, d_v))
    print("テスト本体は test_attention.py へ")
