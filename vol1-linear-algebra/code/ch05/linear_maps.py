# 第1巻 第5章: 線形変換 — 本文の数値例の検算
# (本章は図が主役のため、このファイルは本文に掲載した手計算の検算のみを行う)
import numpy as np

# --- 5.1 行列は関数である: 入れると出てくる + 線形性 ---
A = np.array([[1.0, 2.0],
              [0.0, 1.0],
              [1.0, 0.0]])          # (3, 2): 2次元を入れると3次元が出てくる関数
x = np.array([1.0, 1.0])            # (2,)
y = np.array([2.0, -1.0])           # (2,)
c = 3.0

out = A @ x
assert out.shape == (3,)
assert np.allclose(out, np.array([3.0, 1.0, 1.0]))

# 線形性: 「足してから変換」=「変換してから足す」、「伸ばしてから変換」=「変換してから伸ばす」
assert np.allclose(A @ (x + y), A @ x + A @ y)
assert np.allclose(A @ (c * x), c * (A @ x))


# 各成分を2乗する関数は線形ではない(反例)
def square(v):
    return v ** 2


assert not np.allclose(square(x + y), square(x) + square(y))

# --- 5.2 回転・拡大縮小・射影 ---
R = np.array([[0.0, -1.0],
              [1.0, 0.0]])          # 90度回転
S = np.array([[2.0, 0.0],
              [0.0, 0.5]])          # x方向2倍・y方向半分
P = np.array([[1.0, 0.0],
              [0.0, 0.0]])          # x軸への射影

p = np.array([2.0, 1.0])
assert np.allclose(R @ p, np.array([-1.0, 2.0]))
assert np.allclose(S @ p, np.array([4.0, 0.5]))
assert np.allclose(P @ p, np.array([2.0, 0.0]))

# 行列の列 = 軸方向の単位ベクトルの行き先
e1 = np.array([1.0, 0.0])
e2 = np.array([0.0, 1.0])
assert np.allclose(R @ e1, R[:, 0])
assert np.allclose(R @ e2, R[:, 1])

# 一般の回転行列: (1, 0) は (cosθ, sinθ) へ。回転は長さを保つ
theta = np.pi / 3
R_theta = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
assert np.allclose(R_theta @ e1, np.array([np.cos(theta), np.sin(theta)]))
assert np.allclose(np.linalg.norm(R_theta @ p), np.linalg.norm(p))

# 射影は情報を失う: 違う点が同じ点に潰れる。2回掛けても1回と同じ
q = np.array([2.0, 5.0])
assert np.allclose(P @ p, P @ q)
assert np.allclose(P @ (P @ p), P @ p)

# --- 5.3 合成 = 行列積(順序で結果が変わる) ---
v = np.array([1.0, 0.0])
S2 = np.array([[2.0, 0.0],
               [0.0, 1.0]])         # x方向にだけ2倍

# 「先に回転、次に拡大」は1つの行列 S2 @ R に畳める
assert np.allclose(S2 @ (R @ v), (S2 @ R) @ v)
assert np.allclose(S2 @ (R @ v), np.array([0.0, 1.0]))
# 逆順は別の変換
assert np.allclose(R @ (S2 @ v), np.array([0.0, 2.0]))
assert not np.allclose(S2 @ (R @ v), R @ (S2 @ v))

# --- 5.4 行ベクトルの流儀: 合成が左から右に読める ---
rng = np.random.default_rng(42)
x_row = rng.standard_normal(4)            # (4,)
W1 = rng.standard_normal((4, 5))          # (4, 5)
W2 = rng.standard_normal((5, 3))          # (5, 3)
assert (x_row @ W1).shape == (5,)
assert np.allclose((x_row @ W1) @ W2, x_row @ (W1 @ W2))

# --- 5.5 同じ入力を3つの役割に変換する: shape を読む ---
n, d_model, d_k, d_v = 3, 8, 4, 4
X = rng.standard_normal((n, d_model))     # (n, d_model)
W_Q = rng.standard_normal((d_model, d_k))
W_K = rng.standard_normal((d_model, d_k))
W_V = rng.standard_normal((d_model, d_v))

Q = X @ W_Q
K = X @ W_K
V = X @ W_V
assert Q.shape == (n, d_k)
assert K.shape == (n, d_k)
assert V.shape == (n, d_v)
# Q と K は列数が同じなので、QK^T が掛けられる(読むのは終章)
assert (Q @ K.T).shape == (n, n)

print("ok: すべての assert を通過しました")
