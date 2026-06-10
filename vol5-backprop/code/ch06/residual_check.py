# 第5巻 第6章 6.2: residual connection の backward で「+1」が効くことの数値検算
import numpy as np

rng = np.random.default_rng(42)

# --- (1) スカラーの鎖: 局所勾配 0.25 を10回掛けると消える。1 + 0.25 なら消えない ---
g = 0.25                  # sigmoid の微分の最大値
plain = g ** 10           # 素通しの鎖
resid = (1 + g) ** 10     # residual の鎖: 各層の係数が 1 + g になる
print("plain: {:.2e}   residual: {:.2f}".format(plain, resid))
assert plain < 1e-5
assert resid > 1.0

# --- (2) ベクトル版: y = x + f(x), f(x) = tanh(xW) の ∂L/∂x を解析と数値で照合 ---
n, d = 4, 8
x = rng.normal(0, 1, size=(n, d))
W = rng.normal(0, 0.3, size=(d, d))
R = rng.normal(0, 1, size=(n, d))   # L = Σ(y ⊙ R): 一般の上流勾配 R を作る仕掛け


def loss_fn(x):
    y = x + np.tanh(x @ W)
    return np.sum(y * R)


# 解析勾配: ∂L/∂x = R(素通り分)+ (R ⊙ tanh') W^T(f 経由分)
f = np.tanh(x @ W)
dx = R + (R * (1.0 - f ** 2)) @ W.T

# 数値勾配(第2巻1章の中心差分)で検算
eps = 1e-5
dx_num = np.zeros_like(x)
for i in range(n):
    for j in range(d):
        old = x[i, j]
        x[i, j] = old + eps
        fp = loss_fn(x)
        x[i, j] = old - eps
        fm = loss_fn(x)
        x[i, j] = old
        dx_num[i, j] = (fp - fm) / (2 * eps)

assert np.allclose(dx, dx_num, atol=1e-6)

# --- (3) f が完全に死んでいても(W = 0)、素通り分 R がそのまま届く ---
W = np.zeros((d, d))      # f(x) = tanh(0) = 0: 迂回路だけが残った状態
f = np.tanh(x @ W)
dx = R + (R * (1.0 - f ** 2)) @ W.T
assert np.allclose(dx, R)

print("すべての assert を通過しました")
