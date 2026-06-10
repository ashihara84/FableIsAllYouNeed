import numpy as np

# 第5巻 第2章: 2層MLP の forward
# linear は第1巻6章(code/ch06/linear_layer.py)のものを1文字も変えずに使う(伏線回収)


def linear(X, W, b):
    """全結合層: (バッチ, d_in) @ (d_in, d_out) + (d_out,) -> (バッチ, d_out)"""
    return X @ W + b


def relu(Z):
    """ReLU: 要素ごとの max(0, z)。shape を変えない(第1章)"""
    return np.maximum(Z, 0.0)


def mlp_forward(X, W1, b1, W2, b2):
    """2層MLP: linear -> ReLU -> linear"""
    H = relu(linear(X, W1, b1))
    return linear(H, W2, b2)


# --- XOR的データ: 第1章で直線では割れなかった4点 ---
X = np.array([[0.0, 0.0],
              [0.0, 1.0],
              [1.0, 0.0],
              [1.0, 1.0]])
y = np.array([0.0, 1.0, 1.0, 0.0])

# --- 2.3a: 乱数の重みで、まず骨格(shape)を確認 ---
rng = np.random.default_rng(42)
h = 4                                    # 隠れ層の幅(設計変数)
W1 = rng.standard_normal((2, h))
b1 = rng.standard_normal(h)
W2 = rng.standard_normal((h, 1))
b2 = rng.standard_normal(1)

out = mlp_forward(X, W1, b1, W2, b2)
assert out.shape == (4, 1)               # (4,2) -> (4,4) -> (4,1)

H = relu(linear(X, W1, b1))
assert H.shape == (4, h)
assert np.allclose(out, linear(H, W2, b2))   # mlp_forward は受け渡しの合成にすぎない

# --- 2.3b: 手調整の重みで XOR を解く ---
W1 = np.array([[1.0, 1.0],
               [1.0, 1.0]])              # (2, 2)
b1 = np.array([0.0, -1.0])               # (2,)
W2 = np.array([[1.0],
               [-2.0]])                  # (2, 1)
b2 = np.array([0.0])                     # (1,)

H = relu(linear(X, W1, b1))
assert np.allclose(H, [[0, 0], [1, 0], [1, 0], [2, 1]])   # h1 = OR的、h2 = AND的

y_hat = mlp_forward(X, W1, b1, W2, b2).ravel()
assert np.allclose(y_hat, y)             # ぴったり [0, 1, 1, 0]

# --- 2.3c: 決定境界が「1本の直線」ではないことを格子で確認 ---
# この重みでの出力は s = x1 + x2 だけで決まり、y_hat = ReLU(s) - 2 ReLU(s - 1)。
# 「1」と判定される領域(y_hat > 0.5)は、帯 0.5 < s < 1.5 になるはず。
xs = np.linspace(-0.5, 1.5, 81)
grid = np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)   # (81*81, 2)
pred = mlp_forward(grid, W1, b1, W2, b2).ravel() > 0.5
s = grid[:, 0] + grid[:, 1]
band = (s > 0.5) & (s < 1.5)
edge = (np.abs(s - 0.5) < 1e-9) | (np.abs(s - 1.5) < 1e-9)     # 帯の縁ぴったりは丸め誤差で揺れるので除外
assert np.array_equal(pred[~edge], band[~edge])

# 1本の直線では作れないことの証人: (0,0) と (1,1) は帯の外、その中点 (0.5, 0.5) は帯の中
probe = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
p = mlp_forward(probe, W1, b1, W2, b2).ravel()
assert p[0] < 0.5 and p[1] < 0.5 and p[2] > 0.5

# --- 2.2 の確認: 論文の FFN は 2層MLP そのもの(512 -> 2048 -> 512)---
n, d_model, d_ff = 10, 512, 2048
Xp = rng.standard_normal((n, d_model))
Wf1 = rng.standard_normal((d_model, d_ff))
bf1 = rng.standard_normal(d_ff)
Wf2 = rng.standard_normal((d_ff, d_model))
bf2 = rng.standard_normal(d_model)
assert mlp_forward(Xp, Wf1, bf1, Wf2, bf2).shape == (n, d_model)   # 入力と同じ形に戻る

print("ok: すべての assert を通過しました")
