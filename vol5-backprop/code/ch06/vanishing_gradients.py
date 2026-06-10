# 第5巻 第6章 6.1: 深いMLPで勾配消失を観測する
# 10層のMLPを1回 forward / backward し、各層の重み勾配のノルムを表にする
import numpy as np

rng = np.random.default_rng(42)

n, d = 64, 64       # バッチサイズ、隠れ層の幅
depth = 10          # 隠れ層の数
sigma_w = 0.1       # 「小さめの乱数」という一見無難な初期化(6.6で再考する)

X = rng.normal(0, 1, size=(n, d))
y = rng.normal(0, 1, size=(n, 1))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# --- パラメータ: 隠れ depth 層 + 出力1層 ---
Ws = [rng.normal(0, sigma_w, size=(d, d)) for _ in range(depth)]
bs = [np.zeros(d) for _ in range(depth)]
W_out = rng.normal(0, sigma_w, size=(d, 1))
b_out = np.zeros(1)

# --- forward(backward で使う中間値を保存しながら)---
hs = [X]                                  # hs[l] = 第 l+1 層への入力
for W, b in zip(Ws, bs):
    hs.append(sigmoid(hs[-1] @ W + b))
y_pred = hs[-1] @ W_out + b_out           # (n, 1)
loss = np.mean((y_pred - y) ** 2)

# --- backward(第3章の手導出と同じ手順)---
delta = 2.0 * (y_pred - y) / n            # (n, 1)
grad_W_out = hs[-1].T @ delta
delta = delta @ W_out.T                   # (n, d) 最終隠れ層の出力への勾配
grad_Ws = [None] * depth
for l in range(depth - 1, -1, -1):
    delta = delta * hs[l + 1] * (1.0 - hs[l + 1])   # sigmoid の微分 = s(1-s)
    grad_Ws[l] = hs[l].T @ delta                    # ∂L/∂W_l = 入力^T @ δ
    delta = delta @ Ws[l].T

# --- 各層の勾配ノルムを表にする ---
norms = np.array([np.linalg.norm(g) for g in grad_Ws])
print("層   ||grad_W_l||")
for l in range(depth):
    print("{:2d}   {:.3e}".format(l + 1, norms[l]))
print("loss = {:.4f}".format(loss))
print("第1層と第10層の勾配ノルム比: {:.2e}".format(norms[0] / norms[-1]))

# --- 検証: 本節の主張をデータで確認する ---
# (1) 入口の層の勾配は出口の層より桁違いに小さい(勾配消失)
assert norms[0] < 1e-4 * norms[-1], "勾配消失が観測できていません"

# (2) 減衰は一方向: 出口から入口へ向かって単調に小さくなる
assert np.all(np.diff(norms) > 0)

# (3) forward は死んでいない: 出力は有限で、損失も普通の値
assert np.all(np.isfinite(y_pred)) and 0.1 < loss < 10.0

print("すべての assert を通過しました")
