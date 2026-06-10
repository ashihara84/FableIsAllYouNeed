# 第7巻 第7章 7.4: positional_encoding のテスト
# python3 test_positional_encoding.py で実行。第8巻はこのテストが通ることを前提に import する
import os
import sys

import numpy as np

from positional_encoding import positional_encoding

# 第3章 3.4 で実装した attention(式(1))をテスト6で使う
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "ch03"))
from attention import attention

rng = np.random.default_rng(42)

# --- テスト1: shape と決定性(乱数を使わないので、何度作っても同じ行列) ---
max_len, d_model = 40, 32
pe = positional_encoding(max_len, d_model)
assert pe.shape == (max_len, d_model)
assert np.array_equal(pe, positional_encoding(max_len, d_model))

# --- テスト2: 式(3)との全数一致(定義どおりの素朴な二重ループで検算) ---
pe_naive = np.zeros((max_len, d_model))
for pos in range(max_len):
    for idx in range(d_model):
        i = idx // 2
        angle = pos / 10000.0 ** (2.0 * i / d_model)
        pe_naive[pos, idx] = np.sin(angle) if idx % 2 == 0 else np.cos(angle)
assert np.allclose(pe, pe_naive, atol=1e-12)

# --- テスト3: 値域。位置がどれだけ先でも全成分が [-1, 1] に収まる(7.5 の外挿の根拠) ---
assert np.all(np.abs(positional_encoding(10000, d_model)) <= 1.0)

# --- テスト4: 相対位置の線形性(7.3 の原文の主張)。
#     どの位置 pos でも PE[pos+k] = M_k @ PE[pos]。M_k は k だけから作れて pos に依らない ---


def offset_matrix(k, d_model):
    """位置を k だけ進める線形変換 (d_model, d_model)。
    sin/cos のペアごとに 2×2 の回転をブロック対角に並べたもの(加法定理そのまま)"""
    M = np.zeros((d_model, d_model))
    for i in range(d_model // 2):
        omega = 1.0 / 10000.0 ** (2.0 * i / d_model)
        c, s = np.cos(k * omega), np.sin(k * omega)
        M[2 * i:2 * i + 2, 2 * i:2 * i + 2] = np.array([[c, s],
                                                        [-s, c]])
    return M


for k in [1, 3, 10]:
    M_k = offset_matrix(k, d_model)          # k から一度だけ作る(pos を知らない)
    # 全位置にまとめて M_k を適用(行ごとの M_k @ pe[pos] と同じ。
    # ブロック対角行列の @ は環境によって誤警告を出すため dot で書く)
    shifted = pe[:max_len - k].dot(M_k.T)    # (max_len-k, d_model)
    assert np.allclose(pe[k:], shifted, atol=1e-9)   # その同じ M_k が全位置で通用する

# --- テスト5: 内積が位置差だけで決まる: PE[p]・PE[q] = Σ_i cos((p−q)ω_i)(演習2の根拠) ---
for delta in [1, 5, 12]:
    dots = np.array([pe[p] @ pe[p + delta] for p in range(max_len - delta)])
    assert np.allclose(dots, dots[0], atol=1e-9)        # どの p でも同じ値
omega = 1.0 / 10000.0 ** (2.0 * np.arange(d_model // 2) / d_model)
assert np.allclose(pe[7] @ pe[12], np.sum(np.cos(5 * omega)), atol=1e-9)

# --- テスト6: PE を足すと attention の並べ替え不変性が壊れる(7.1 の問題の解決の検収) ---
n = 6
X = rng.normal(0, 1, size=(n, d_model))
W_Q = rng.normal(0, 1.0 / np.sqrt(d_model), size=(d_model, d_model))
W_K = rng.normal(0, 1.0 / np.sqrt(d_model), size=(d_model, d_model))
W_V = rng.normal(0, 1.0 / np.sqrt(d_model), size=(d_model, d_model))
perm = np.array([3, 0, 5, 1, 4, 2])


def self_attention(X, W_Q, W_K, W_V):
    out, _ = attention(X @ W_Q, X @ W_K, X @ W_V)
    return out


# PE なし: 並べ替えと出力が交換する(7.1 の再現)
out = self_attention(X, W_Q, W_K, W_V)
assert np.allclose(self_attention(X[perm], W_Q, W_K, W_V), out[perm], atol=1e-12)

# PE あり: トークンを並べ替えても PE は位置 0, 1, 2, ... の順のまま足される
pe_n = positional_encoding(n, d_model)                   # (n, d_model)
out_pe = self_attention(X + pe_n, W_Q, W_K, W_V)
out_pe_shuffled = self_attention(X[perm] + pe_n, W_Q, W_K, W_V)
assert not np.allclose(out_pe_shuffled, out_pe[perm], atol=1e-6)

# --- テスト7: 第8巻との契約。d_model が奇数なら明示的に拒否する ---
try:
    positional_encoding(10, 7)
    raise AssertionError("奇数の d_model が通ってしまった")
except ValueError:
    pass

print("すべての assert を通過しました")
