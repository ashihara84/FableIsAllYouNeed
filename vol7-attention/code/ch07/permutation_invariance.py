# 第7巻 第7章 7.1: attention は集合演算であることの実験
# 入力の行を並べ替えても、出力は「同じ並べ替え」を受けるだけで中身が変わらないことを確かめる
import os
import sys

import numpy as np

# 第3章 3.4 で実装した attention(式(1))をそのまま使う
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "ch03"))
from attention import attention

rng = np.random.default_rng(42)


def self_attention(X, W_Q, W_K, W_V):
    """self-attention: Q, K, V がすべて同じ X から作られる(第5章)"""
    out, _ = attention(X @ W_Q, X @ W_K, X @ W_V)   # (n, d_model)
    return out


# 6 トークンの文のつもり。各行が 1 トークンの埋め込みベクトル
n, d_model = 6, 16
X = rng.normal(0, 1, size=(n, d_model))                              # (n, d_model)
W_Q = rng.normal(0, 1.0 / np.sqrt(d_model), size=(d_model, d_model))
W_K = rng.normal(0, 1.0 / np.sqrt(d_model), size=(d_model, d_model))
W_V = rng.normal(0, 1.0 / np.sqrt(d_model), size=(d_model, d_model))

out = self_attention(X, W_Q, W_K, W_V)                               # (n, d_model)

# 語順をめちゃくちゃにする: 行を 3, 0, 5, 1, 4, 2 の順に並べ替え
perm = np.array([3, 0, 5, 1, 4, 2])
out_shuffled = self_attention(X[perm], W_Q, W_K, W_V)

# 出力は「同じ並べ替えを受けた元の出力」と完全に一致する
assert np.allclose(out_shuffled, out[perm], atol=1e-12)

# 1 トークンずつ見ても同じ: 各トークンが受け取る表現は語順に 1 ビットも依存しない
for new_pos, old_pos in enumerate(perm):
    assert np.allclose(out_shuffled[new_pos], out[old_pos], atol=1e-12)

print("並べ替えの前後で、各トークンの出力ベクトルは完全に一致しました")
print("すべての assert を通過しました")
