# 第4巻 第4章 4.4 図4.1 と演習1 図4.2: 学習曲線の比較と決定境界(本文掲載用。実行には matplotlib が必要)
# データ・実験条件は logistic_regression.py(= 第3巻エピローグ)と同一。
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)

# --- データ: 第3巻 E.1 と同一 ---
n = 100
X_normal = rng.normal(loc=[-2.0, -2.0], scale=1.0, size=(n, 2))  # (100, 2)
X_spam = rng.normal(loc=[2.0, 2.0], scale=1.0, size=(n, 2))      # (100, 2)
X = np.vstack([X_normal, X_spam])                                # (200, 2)
y = np.concatenate([np.zeros(n), np.ones(n)])                    # (200,)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def predict(X, w, b):
    return sigmoid(X @ w + b)


def log_loss(X, y, w, b):
    p = np.clip(predict(X, w, b), 1e-12, 1.0 - 1e-12)
    return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def grad_log_loss(X, y, w, b):
    p = predict(X, w, b)
    return X.T @ (p - y) / len(y), np.mean(p - y)


def mse_loss(X, y, w, b):
    return np.mean((predict(X, w, b) - y) ** 2)


def numerical_grad(loss_fn, X, y, w, b, h=1e-5):
    grad_w = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w)
        e[i] = h
        grad_w[i] = (loss_fn(X, y, w + e, b) - loss_fn(X, y, w - e, b)) / (2 * h)
    grad_b = (loss_fn(X, y, w, b + h) - loss_fn(X, y, w, b - h)) / (2 * h)
    return grad_w, grad_b


def train(loss_name, w0, b0, lr=0.5, steps=2000):
    """凍結条件(第3巻 E.3 と同一の lr・steps)で、損失だけを切り替えて学習する"""
    w = np.array(w0, dtype=float)
    b = float(b0)
    loss_fn = log_loss if loss_name == "logloss" else mse_loss
    history = [loss_fn(X, y, w, b)]
    for _ in range(steps):
        if loss_name == "logloss":
            grad_w, grad_b = grad_log_loss(X, y, w, b)      # 4.3 の解析形
        else:
            grad_w, grad_b = numerical_grad(mse_loss, X, y, w, b)  # 第3巻の再現
        w -= lr * grad_w
        b -= lr * grad_b
        history.append(loss_fn(X, y, w, b))
    return w, b, history


w0_bad = [-8.0, -8.0]   # 第3巻の凍結条件
w_B, b_B, hist_B = train("logloss", w0=w0_bad, b0=0.0)   # 本文 実験2と同じ名前
w_C, b_C, hist_C = train("mse", w0=w0_bad, b0=0.0)       # 本文 実験3と同じ名前

# --- 図4.1: 同じ凍結条件からの学習曲線。左: MSE(第3巻の再現)、右: log loss ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(hist_C)
ax1.set_xlabel("step"); ax1.set_ylabel("MSE loss"); ax1.set_title("MSE: frozen (vol.3)")
ax1.set_ylim(0, 1.05)
ax2.plot(hist_B)
ax2.set_xlabel("step"); ax2.set_ylabel("log loss"); ax2.set_title("log loss: learning")
plt.tight_layout(); plt.show()

# --- 図4.2(演習1): 決定境界と「確率の等高線」 ---
# 境界は w1 x1 + w2 x2 + b = logit(p) の直線。p = 0.5 で logit = 0(実線)、
# p = 0.1, 0.9 で logit = ±log(9)(破線)。
x1_line = np.linspace(-5.0, 5.0, 100)
plt.scatter(X_normal[:, 0], X_normal[:, 1], marker="o", label="normal (y=0)")
plt.scatter(X_spam[:, 0], X_spam[:, 1], marker="x", label="spam (y=1)")
for p_level, style in [(0.1, "--"), (0.5, "-"), (0.9, "--")]:
    logit = np.log(p_level / (1.0 - p_level))
    x2_line = (logit - w_B[0] * x1_line - b_B) / w_B[1]
    plt.plot(x1_line, x2_line, style, label=f"p = {p_level}")
plt.xlabel("x1"); plt.ylabel("x2"); plt.legend(); plt.show()
