# 第3巻 図版一括生成スクリプト
# 使い方: python3 generate_figures.py  (このディレクトリで実行。全PNGを上書き生成する)
#
# 規約:
#   - 図中の文字は英語と数式記号のみ(日本語フォント非依存)
#   - データ生成は各章の code/ と同一(np.random.default_rng(42)、同じ呼び出し順)
#     → 本文に載せた数値(谷底 (6.7, 22.0)、検証loss最小=次数3 など)と必ず一致する
#   - dpi=160, facecolor="white", bbox_inches="tight"
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (projection="3d" の登録)

# --- 共通スタイル ---
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.framealpha": 0.9,
    "legend.fontsize": 8.5,
})

# 色覚多様性に安全な固定パレット(Okabe-Ito)。系列には常にこの順で割り当てる
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_VERMILLION = "#D55E00"
C_GRAY = "#666666"

SAVE_KW = dict(dpi=160, facecolor="white", bbox_inches="tight")


def save(fig, name):
    fig.savefig(name, **SAVE_KW)
    plt.close(fig)
    print("wrote", name)


# =====================================================================
# 第2章のデータ: 勉強時間と点数(20人)。第2〜4章で共通
# =====================================================================
def make_data():
    rng = np.random.default_rng(42)
    n = 20
    X = rng.uniform(0, 9, size=(n, 1))          # 勉強時間 (20, 1)
    noise = rng.normal(0, 6.0, size=(n, 1))
    y = 7.0 * X + 20.0 + noise                  # 点数 (20, 1)  真の規則: w=7, b=20
    return X, y


X, y = make_data()
assert np.allclose(X[:3].ravel(), [6.96560444, 3.94990596, 7.72738128])


# =====================================================================
# 図2.1: 散布図(ch02-scatter.png)
# =====================================================================
fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.scatter(X, y, s=28, color=C_BLUE, zorder=3)
ax.set_xlabel("study hours  $x$")
ax.set_ylabel("score  $y$")
ax.set_xlim(0, 9.5)
save(fig, "ch02-scatter.png")


# =====================================================================
# 図2.2: 散布図 + 候補直線A〜D(ch02-candidate-lines.png)
# =====================================================================
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(X, y, s=28, color=C_GRAY, zorder=3, label="data (20 students)")
xs = np.array([0.0, 9.3])
for (name, w_c, b_c), color, ls in [
    (("A", 10.0, 0.0), C_BLUE, "--"),
    (("B", 4.0, 40.0), C_ORANGE, "--"),
    (("C", 7.0, 20.0), C_GREEN, "-"),
    (("D", 6.5, 25.0), C_VERMILLION, "-"),
]:
    ax.plot(xs, w_c * xs + b_c, color=color, ls=ls, lw=1.8,
            label=f"{name}:  $w={w_c:g},\\ b={b_c:g}$")
ax.set_xlabel("study hours  $x$")
ax.set_ylabel("score  $y$")
ax.set_xlim(0, 9.3)
ax.set_ylim(0, 100)
ax.legend(loc="upper left")
save(fig, "ch02-candidate-lines.png")


# =====================================================================
# 第3章: (w, b) 平面上の MSE の地形(code/ch03/mse_landscape.py と同一)
# =====================================================================
def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def L(w, b):
    return mse(w * X + b, y)


ws = np.linspace(3.0, 11.0, 81)
bs = np.linspace(0.0, 40.0, 81)
landscape = np.empty((len(bs), len(ws)))
for i, b in enumerate(bs):
    for j, w in enumerate(ws):
        landscape[i, j] = L(w, b)

i_min, j_min = np.unravel_index(landscape.argmin(), landscape.shape)
w_best, b_best = ws[j_min], bs[i_min]
assert (round(w_best, 1), round(b_best, 1)) == (6.7, 22.0)  # 本文の「谷底 (6.7, 22.0)」

# --- 図3.1: 等高線図(ch03-mse-contour.png) ---
fig, ax = plt.subplots(figsize=(6, 4.2))
W, B = np.meshgrid(ws, bs)
levels = [25, 50, 100, 200, 400, 800, 1600, 3200]
cs = ax.contour(W, B, landscape, levels=levels, cmap="viridis_r", linewidths=1.2)
ax.clabel(cs, fmt="%d", fontsize=7.5)
ax.plot(w_best, b_best, "x", color=C_VERMILLION, ms=9, mew=2.2, zorder=5)
ax.annotate(f"minimum on grid\n$(w, b) = ({w_best:.1f},\\ {b_best:.1f})$",
            xy=(w_best, b_best), xytext=(7.6, 6.5),
            arrowprops=dict(arrowstyle="->", lw=0.9, color=C_GRAY), fontsize=8.5)
ax.set_xlabel("$w$")
ax.set_ylabel("$b$")
ax.set_title("MSE loss landscape  $L(w, b)$  (contours)")
save(fig, "ch03-mse-contour.png")

# --- 図3.2: 3Dサーフェス(ch03-mse-surface.png) ---
fig = plt.figure(figsize=(6.4, 4.5))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(W, B, landscape, cmap="viridis", rcount=81, ccount=81,
                linewidth=0, antialiased=True)
ax.scatter([w_best], [b_best], [landscape.min()], color=C_VERMILLION,
           s=40, depthshade=False, zorder=10)  # 谷底 (6.7, 22.0)。座標は図3.1に明記
ax.set_xlabel("$w$")
ax.set_ylabel("$b$")
# mplot3d の zlabel は bbox_inches="tight" で切れやすいので text2D で置く
ax.text2D(0.93, 0.82, "$L(w, b)$", transform=ax.transAxes,
          ha="left", va="bottom", fontsize=9.5)
ax.set_zlim(0, landscape.max())
ax.view_init(elev=28, azim=-60)
ax.set_title("MSE loss landscape  $L(w, b)$  (3D surface)", pad=0)
save(fig, "ch03-mse-surface.png")


# =====================================================================
# 図4.1: 学習曲線 lr=0.01、線形/対数(ch04-learning-curves.png)
# code/ch04/plot_learning_curves.py と同一の訓練
# =====================================================================
def train_lin(Xf, yf, w=0.0, b=0.0, lr=0.01, num_steps=3000):
    history = []
    for _ in range(num_steps):
        y_hat = w * Xf + b
        history.append(np.mean((y_hat - yf) ** 2))
        grad_w = 2.0 * np.mean((y_hat - yf) * Xf)
        grad_b = 2.0 * np.mean(y_hat - yf)
        w, b = w - lr * grad_w, b - lr * grad_b
    return w, b, history


_, _, history = train_lin(X.ravel(), y.ravel())
assert round(history[0], 2) == 3529.87                       # 本文 step 0 の loss

fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
for ax, yscale, title in [(axes[0], "linear", "linear scale"),
                          (axes[1], "log", "log scale")]:
    ax.plot(history, color=C_BLUE, lw=1.6)
    ax.set_yscale(yscale)
    ax.set_xlabel("step")
    ax.set_ylabel("loss (MSE)")
    ax.set_title(title)
fig.tight_layout()
save(fig, "ch04-learning-curves.png")


# =====================================================================
# 図5.1: バッチサイズ別の学習曲線(ch05-batchsize-curves.png)
# code/ch05/minibatch_sgd.py と同一の設定(N=256, lr=0.01, 150エポック)
# =====================================================================
rng5 = np.random.default_rng(42)
N5 = 256
X5 = rng5.uniform(0, 9, size=(N5, 1))
y5 = 7.0 * X5 + 20.0 + rng5.normal(0, 6.0, size=(N5, 1))


def train_minibatch(batch_size, n_epochs=150, lr=0.01, seed=0):
    rng_shuffle = np.random.default_rng(seed)
    w = np.zeros((1, 1))
    b = 0.0
    history = []
    for _ in range(n_epochs):
        idx = rng_shuffle.permutation(N5)
        for start in range(0, N5, batch_size):
            batch = idx[start:start + batch_size]
            err = X5[batch] @ w + b - y5[batch]
            nb = len(batch)
            w = w - lr * (2.0 / nb) * (X5[batch].T @ err)
            b = b - lr * (2.0 / nb) * np.sum(err)
        history.append(np.mean((X5 @ w + b - y5) ** 2))
    return history


fig, ax = plt.subplots(figsize=(6.4, 4))
for batch_size, color, label in [(256, C_BLUE, "full batch ($B=256$)"),
                                 (32, C_ORANGE, "mini-batch ($B=32$)"),
                                 (1, C_GREEN, "SGD ($B=1$)")]:
    hist = train_minibatch(batch_size)
    ax.plot(range(1, len(hist) + 1), hist, color=color, lw=1.4, label=label)
ax.axhline(36.0, color=C_GRAY, lw=1.0, ls=":", zorder=1)
ax.set_ylim(28, 900)
ax.text(149, 34.3, "noise floor $\\approx 36\\ (=6^2)$", ha="right", va="top",
        fontsize=8, color=C_GRAY)
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("loss (MSE, full data)")
ax.legend()
save(fig, "ch05-batchsize-curves.png")


# =====================================================================
# 図6.1: 過学習のU字カーブ(ch06-overfitting-ucurve.png)
# code/ch06/polynomial_overfitting.py と同一の実験(実行に20秒ほどかかる)
# =====================================================================
rng6 = np.random.default_rng(42)


def true_f(x):
    return 0.5 + 1.0 * x - 2.0 * x**2 + 3.0 * x**3


n_train, n_val = 20, 100
x_train = rng6.uniform(-1, 1, size=n_train)
y_train = true_f(x_train) + rng6.normal(0, 0.4, size=n_train)
x_val = rng6.uniform(-1, 1, size=n_val)
y_val = true_f(x_val) + rng6.normal(0, 0.4, size=n_val)


def poly_features(x, degree):
    if degree == 0:
        return np.zeros((len(x), 0))
    return np.stack([x**k for k in range(1, degree + 1)], axis=1)


def fit_gd(Phi, yf, lr=0.1, steps=1000000):
    n, d = Phi.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(steps):
        err = Phi @ w + b - yf
        w -= lr * (2.0 / n) * (Phi.T @ err)
        b -= lr * (2.0 / n) * err.sum()
    return w, b


degrees = list(range(10))
train_losses, val_losses = [], []
for degree in degrees:
    Phi_train = poly_features(x_train, degree)
    Phi_val = poly_features(x_val, degree)
    if degree > 0:
        mu, sigma = Phi_train.mean(axis=0), Phi_train.std(axis=0)
        Phi_train = (Phi_train - mu) / sigma
        Phi_val = (Phi_val - mu) / sigma
    w6, b6 = fit_gd(Phi_train, y_train)
    train_losses.append(mse(Phi_train @ w6 + b6, y_train))
    val_losses.append(mse(Phi_val @ w6 + b6, y_val))

best_degree = int(np.argmin(val_losses))
assert best_degree == 3                                       # 本文の表: 最小は次数3
assert round(val_losses[3], 4) == 0.2210                      # 本文の表の数値と一致

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(degrees, train_losses, "o-", color=C_BLUE, lw=1.6, ms=5,
        label="training loss")
ax.plot(degrees, val_losses, "s-", color=C_VERMILLION, lw=1.6, ms=5,
        label="validation loss")
ax.plot(best_degree, val_losses[best_degree], "s", color=C_VERMILLION,
        ms=11, mfc="none", mew=1.6)
ax.annotate(f"min at degree {best_degree}",
            xy=(best_degree, val_losses[best_degree]), xytext=(4.1, 0.14),
            arrowprops=dict(arrowstyle="->", lw=0.9, color=C_GRAY), fontsize=8.5)
ax.set_yscale("log")
ax.set_xlabel("polynomial degree")
ax.set_ylabel("loss (MSE, log scale)")
ax.set_xticks(degrees)
ax.legend()
save(fig, "ch06-overfitting-ucurve.png")


# =====================================================================
# エピローグ: スパム判定(code/ch08/dead_gradient.py と同一)
# =====================================================================
rng8 = np.random.default_rng(42)
n8 = 100
X_normal = rng8.normal(loc=[-2.0, -2.0], scale=1.0, size=(n8, 2))
X_spam = rng8.normal(loc=[2.0, 2.0], scale=1.0, size=(n8, 2))
X8 = np.vstack([X_normal, X_spam])
y8 = np.concatenate([np.zeros(n8), np.ones(n8)])

# --- 図E.1: 散布図(ch08-spam-scatter.png) ---
fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.scatter(X_normal[:, 0], X_normal[:, 1], s=26, color=C_BLUE, marker="o",
           label="normal  ($y=0$)")
ax.scatter(X_spam[:, 0], X_spam[:, 1], s=30, color=C_VERMILLION, marker="^",
           label="spam  ($y=1$)")
ax.set_xlabel("$x_1$: suspicious words")
ax.set_ylabel("$x_2$: exclamation marks")
ax.legend(loc="upper left")
ax.set_aspect("equal")
save(fig, "ch08-spam-scatter.png")


# --- 図E.2: 実験1と実験2の学習曲線(ch08-learning-curves.png) ---
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def mse_loss(Xf, yf, w, b):
    return np.mean((sigmoid(Xf @ w + b) - yf) ** 2)


def numerical_grad(loss_fn, Xf, yf, w, b, h=1e-5):
    grad_w = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w)
        e[i] = h
        grad_w[i] = (loss_fn(Xf, yf, w + e, b) - loss_fn(Xf, yf, w - e, b)) / (2 * h)
    grad_b = (loss_fn(Xf, yf, w, b + h) - loss_fn(Xf, yf, w, b - h)) / (2 * h)
    return grad_w, grad_b


def train_sigmoid(w0, b0, lr=0.5, steps=2000):
    w = np.array(w0, dtype=float)
    b = float(b0)
    history = [mse_loss(X8, y8, w, b)]
    for _ in range(steps):
        grad_w, grad_b = numerical_grad(mse_loss, X8, y8, w, b)
        w -= lr * grad_w
        b -= lr * grad_b
        history.append(mse_loss(X8, y8, w, b))
    return history


hist_A = train_sigmoid([0.0, 0.0], 0.0)
hist_B = train_sigmoid([-8.0, -8.0], 0.0)
assert round(hist_A[0], 4) == 0.2500 and round(hist_A[-1], 4) == 0.0037  # 本文の実験1
assert round(hist_B[0], 6) == 0.994979                                   # 本文の実験2

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(hist_B, color=C_VERMILLION, lw=1.8,
        label="Exp. 2:  init $\\mathbf{w}=(-8, -8)$  (stuck)")
ax.plot(hist_A, color=C_BLUE, lw=1.8,
        label="Exp. 1:  init $\\mathbf{w}=(0, 0)$")
ax.set_xlabel("step")
ax.set_ylabel("loss (MSE)")
ax.set_ylim(-0.03, 1.08)
ax.legend(loc="center right")
save(fig, "ch08-learning-curves.png")

print("done: all figures generated")
