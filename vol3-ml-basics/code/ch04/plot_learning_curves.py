# 第3巻 第4章 4.3 と演習: 学習曲線を描く(本文掲載用。実行には matplotlib が必要)
# learning rate と初期値を変えて、学習曲線の形を見比べる。
import matplotlib.pyplot as plt
import numpy as np


def make_data():
    """第2章と同一の合成データ(50点)。"""
    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, 2.0, size=50)                  # (50,)
    y = 2.0 * X + 1.0 + rng.normal(0.0, 0.5, size=50)   # (50,)
    return X, y


def train(X, y, w=0.0, b=0.0, lr=0.1, num_steps=1000):
    """4.2 の4拍子。loss の推移(学習曲線)を返す。"""
    history = []
    for _ in range(num_steps):
        y_hat = w * X + b                        # forward
        loss = np.mean((y_hat - y) ** 2)         # loss
        grad_w = 2.0 * np.mean((y_hat - y) * X)  # gradient
        grad_b = 2.0 * np.mean(y_hat - y)
        w = w - lr * grad_w                      # update
        b = b - lr * grad_b
        history.append(loss)
    return w, b, history


X, y = make_data()

# --- 図4.1: 学習曲線(lr=0.1)。左: そのまま、右: 縦軸を対数に ---
_, _, history = train(X, y, lr=0.1, num_steps=300)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(history)
axes[0].set_xlabel("step")
axes[0].set_ylabel("loss (MSE)")
axes[0].set_title("学習曲線(線形スケール)")
axes[1].plot(history)
axes[1].set_yscale("log")  # 平坦に見える部分の続きが見える
axes[1].set_xlabel("step")
axes[1].set_ylabel("loss (MSE)")
axes[1].set_title("学習曲線(縦軸を対数に)")
fig.tight_layout()

# --- 図4.2: learning rate を変える(演習1)---
plt.figure(figsize=(6, 4))
for lr in [0.01, 0.1, 0.5]:
    _, _, history = train(X, y, lr=lr, num_steps=100)
    plt.plot(history, label="lr = {}".format(lr))
plt.yscale("log")
plt.xlabel("step")
plt.ylabel("loss (MSE)")
plt.legend()
plt.title("learning rate と学習曲線")

# --- 図4.3: 初期値を変える(演習2)---
plt.figure(figsize=(6, 4))
for w0, b0 in [(0.0, 0.0), (10.0, -5.0), (-3.0, 8.0)]:
    w, b, history = train(X, y, w=w0, b=b0, lr=0.1, num_steps=300)
    plt.plot(history, label="init w={}, b={} -> ({:.3f}, {:.3f})".format(w0, b0, w, b))
plt.yscale("log")
plt.xlabel("step")
plt.ylabel("loss (MSE)")
plt.legend()
plt.title("初期値と学習曲線(着地点はすべて同じ)")

plt.show()
