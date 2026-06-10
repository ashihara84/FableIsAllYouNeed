# 第5巻 第3章 演習問3: 3.3 で導出した 2層MLP の backprop を数値微分(第2巻1章)で検算する
# 「grad_W = 入力^T @ δ、grad_b = δ の列和、下流へは δ @ W^T」の6行が、
# グラフのことを何も知らない中心差分と一致するかを確かめる。
import numpy as np


def numerical_grad(loss_fn, P, h=1e-5):
    """中心差分による勾配(第2巻1章の数値微分を、配列の全成分に1つずつ適用した版)"""
    grad = np.zeros_like(P)
    for idx in np.ndindex(P.shape):
        orig = P[idx]
        P[idx] = orig + h
        loss_plus = loss_fn()
        P[idx] = orig - h
        loss_minus = loss_fn()
        P[idx] = orig                        # 動かした成分は必ず元に戻す
        grad[idx] = (loss_plus - loss_minus) / (2 * h)
    return grad


rng = np.random.default_rng(42)
N, d_in, d_h, d_out = 8, 3, 4, 2
X = rng.standard_normal((N, d_in))
Y = rng.standard_normal((N, d_out))
W1 = rng.standard_normal((d_in, d_h))
b1 = rng.standard_normal(d_h)
W2 = rng.standard_normal((d_h, d_out))
b2 = rng.standard_normal(d_out)


def forward():
    """2層MLP + MSE。backward が局所勾配の計算に使う中間値も返す(第2巻5章と同じ2段構え)"""
    Z1 = X @ W1 + b1                         # (N, d_h)
    H = np.maximum(Z1, 0.0)                  # ReLU (N, d_h)
    Y_hat = H @ W2 + b2                      # (N, d_out)
    loss = np.mean((Y_hat - Y) ** 2)         # スカラー
    return Z1, H, Y_hat, loss


def backward(Z1, H, Y_hat):
    """3.3 の6行。各 linear 層のレシピは同一: grad_W = 入力.T @ δ、grad_b = δ の列和、下流へ δ @ W.T"""
    delta2 = 2 * (Y_hat - Y) / Y.size        # ∂L/∂Ŷ: MSE ノードの局所勾配 (N, d_out)
    grad_W2 = H.T @ delta2                   # (d_h, d_out)
    grad_b2 = delta2.sum(axis=0)             # (d_out,)
    delta1 = (delta2 @ W2.T) * (Z1 > 0)      # ReLU のゲート: 0/1 のマスク (N, d_h)
    grad_W1 = X.T @ delta1                   # (d_in, d_h)
    grad_b1 = delta1.sum(axis=0)             # (d_h,)
    return grad_W1, grad_b1, grad_W2, grad_b2


def loss_fn():
    return forward()[3]


Z1, H, Y_hat, loss = forward()
grad_W1, grad_b1, grad_W2, grad_b2 = backward(Z1, H, Y_hat)

# 勾配は必ずパラメータ自身と同じ shape(更新 W ← W − lr * grad_W に使うため)
assert grad_W1.shape == W1.shape and grad_b1.shape == b1.shape
assert grad_W2.shape == W2.shape and grad_b2.shape == b2.shape

# 数値微分との照合: 4つのパラメータすべて、全成分が一致する
assert np.allclose(grad_W1, numerical_grad(loss_fn, W1))
assert np.allclose(grad_b1, numerical_grad(loss_fn, b1))
assert np.allclose(grad_W2, numerical_grad(loss_fn, W2))
assert np.allclose(grad_b2, numerical_grad(loss_fn, b2))

print("ok: すべての assert を通過しました(backprop と数値微分が一致)")
