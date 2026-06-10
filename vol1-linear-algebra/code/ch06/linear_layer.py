# 第1巻 第6章 6.5: 全結合層を「ただの行列演算」として実装する
# 学習はしない。W は乱数のまま。この linear が第5巻でそのまま再登場する。
import numpy as np


def linear(X, W, b):
    """全結合層: (バッチ, d_in) @ (d_in, d_out) + (d_out,) -> (バッチ, d_out)"""
    return X @ W + b


rng = np.random.default_rng(42)

# --- shape の確認 ---
N, d_in, d_out = 4, 3, 5            # バッチ4件、3次元 -> 5次元
X = rng.standard_normal((N, d_in))
W = rng.standard_normal((d_in, d_out))
b = rng.standard_normal(d_out)

Y = linear(X, W, b)
assert Y.shape == (N, d_out)        # (4, 3) @ (3, 5) + (5,) -> (4, 5)

# --- 6.1 の確認: バッチの各行は独立に変換されている ---
for i in range(N):
    assert np.allclose(Y[i], X[i] @ W + b)

# --- 6.2 の確認: + b は「各行に足す」と同じ ---
XW = X @ W
Y_manual = np.empty_like(XW)
for i in range(N):
    Y_manual[i] = XW[i] + b         # ブロードキャスティングを使わず1行ずつ足す
assert np.allclose(Y, Y_manual)

# --- 6.3 の確認: 多層に重ねる。論文 3.3 の数字 512 -> 2048 -> 512 ---
n, d_model, d_ff = 10, 512, 2048    # 単語10個の文
X2 = rng.standard_normal((n, d_model))
W1 = rng.standard_normal((d_model, d_ff))
b1 = rng.standard_normal(d_ff)
W2 = rng.standard_normal((d_ff, d_model))
b2 = rng.standard_normal(d_model)

H = linear(X2, W1, b1)
assert H.shape == (n, d_ff)         # (10, 512) -> (10, 2048)
Y2 = linear(H, W2, b2)
assert Y2.shape == (n, d_model)     # (10, 2048) -> (10, 512) 入力と同じ形に戻る

print("ok: すべての assert を通過しました")
