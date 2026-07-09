# 第4巻 図版の一括生成スクリプト
# 使い方: python3 generate_figures.py  (このディレクトリに chNN-*.png を出力)
#
# 方針:
# - 図中の文字は英語と数式記号のみ(日本語フォント非依存)
# - 本文・既存スクリプト(code/chNN/*.py)と数値・乱数の種を一致させる
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_KW = dict(dpi=160, facecolor="white", bbox_inches="tight")


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    print("saved:", name)


# =====================================================================
# 図2.1: 標本平均の推移と期待値 1.9(ch02 sampling_lln.py と同じ分布・種)
# =====================================================================
def fig_ch02_running_mean():
    rng = np.random.default_rng(42)
    lengths = np.array([2.0, 1.0, 3.0])  # 晴れ / 雨 / くもり の文字数
    probs = np.array([0.5, 0.3, 0.2])
    E_X = np.sum(lengths * probs)  # 1.9

    # sampling_lln.py と同じ順で乱数を消費し、本文の数表と同じ標本を得る
    samples = None
    for n in [10, 1_000, 100_000]:
        samples = rng.choice(lengths, size=n, p=probs)
    assert abs(samples.mean() - 1.8961) < 5e-5  # 本文: 10万回で 1.896

    n = len(samples)
    running_mean = np.cumsum(samples) / np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(np.arange(1, n + 1), running_mean, color="tab:blue", lw=1.2,
            label="running mean of samples")
    ax.axhline(E_X, color="gray", ls="--", lw=1.2, label=r"$\mathbb{E}[X] = 1.9$")
    ax.set_xscale("log")
    ax.set_xlim(1, n)
    ax.set_ylim(0.9, 3.1)
    ax.set_xlabel("number of samples $n$ (log scale)")
    ax.set_ylabel("running mean")
    ax.legend(loc="upper right")
    save(fig, "ch02-running-mean.png")


# =====================================================================
# 図3.1: 尤度 L(θ) = θ^7 (1−θ)^3 の山と、対数尤度 ℓ(θ)。頂上はどちらも 0.7
# =====================================================================
def fig_ch03_likelihood_peak():
    theta = np.linspace(0.001, 0.999, 999)
    L = theta**7 * (1 - theta) ** 3
    ell = np.log(L)

    # 本文 3.2 の表の数値と一致することを確認
    assert abs(0.7**7 * 0.3**3 - 0.00222) < 5e-5   # L(0.7) = 0.00222
    assert abs(np.log(0.7**7 * 0.3**3) - (-6.11)) < 5e-3  # ℓ(0.7) ≈ −6.11
    assert np.isclose(theta[np.argmax(L)], 0.7)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

    ax1.plot(theta, L, color="tab:blue")
    table_pts = np.array([0.3, 0.5, 0.7, 0.9])
    ax1.plot(table_pts, table_pts**7 * (1 - table_pts) ** 3, "o",
             color="tab:orange", zorder=3, label="values in the table")
    ax1.axvline(0.7, color="gray", ls="--", lw=1)
    ax1.annotate(r"$L(0.7) = 0.00222$", xy=(0.7, 0.00222),
                 xytext=(0.12, 0.0019), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="gray"))
    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel(r"$L(\theta) = \theta^7 (1-\theta)^3$")
    ax1.set_title("likelihood")
    ax1.legend(loc="upper left", fontsize=8)

    ax2.plot(theta, ell, color="tab:blue")
    ax2.axvline(0.7, color="gray", ls="--", lw=1)
    ax2.annotate(r"$\ell(0.7) \approx -6.11$", xy=(0.7, np.log(0.00222)),
                 xytext=(0.1, -9.5), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="gray"))
    ax2.set_ylim(-25, 0)
    ax2.set_xlabel(r"$\theta$")
    ax2.set_ylabel(r"$\ell(\theta) = \log L(\theta)$")
    ax2.set_title("log-likelihood (same peak)")

    fig.tight_layout()
    save(fig, "ch03-likelihood-peak.png")


# =====================================================================
# ch04 共通: 第3巻エピローグと同一のデータ・モデル(plot_curves_and_boundary.py の再現)
# =====================================================================
def _ch04_setup():
    rng = np.random.default_rng(42)
    n = 100
    X_normal = rng.normal(loc=[-2.0, -2.0], scale=1.0, size=(n, 2))
    X_spam = rng.normal(loc=[2.0, 2.0], scale=1.0, size=(n, 2))
    X = np.vstack([X_normal, X_spam])
    y = np.concatenate([np.zeros(n), np.ones(n)])

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
        w = np.array(w0, dtype=float)
        b = float(b0)
        loss_fn = log_loss if loss_name == "logloss" else mse_loss
        history = [loss_fn(X, y, w, b)]
        for _ in range(steps):
            if loss_name == "logloss":
                grad_w, grad_b = grad_log_loss(X, y, w, b)
            else:
                grad_w, grad_b = numerical_grad(mse_loss, X, y, w, b)
            w -= lr * grad_w
            b -= lr * grad_b
            history.append(loss_fn(X, y, w, b))
        return w, b, history

    w0_bad = [-8.0, -8.0]
    w_B, b_B, hist_B = train("logloss", w0=w0_bad, b0=0.0)  # 本文 実験2
    w_C, b_C, hist_C = train("mse", w0=w0_bad, b0=0.0)      # 本文 実験3

    # 本文の数値と一致することを確認
    assert abs(hist_B[0] - 25.1514) < 5e-4 and abs(hist_B[-1] - 0.0109) < 5e-5
    assert abs(hist_C[0] - 0.994979) < 5e-6 and hist_C[0] - hist_C[-1] < 1e-5

    return dict(X_normal=X_normal, X_spam=X_spam,
                w_B=w_B, b_B=b_B, hist_B=hist_B, hist_C=hist_C)


def fig_ch04_learning_curves(env):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
    ax1.plot(env["hist_C"], color="tab:blue")
    ax1.set_xlabel("step")
    ax1.set_ylabel("MSE loss")
    ax1.set_title("MSE: frozen (vol. 3)")
    ax1.set_ylim(0, 1.05)
    ax2.plot(env["hist_B"], color="tab:orange")
    ax2.set_xlabel("step")
    ax2.set_ylabel("log loss")
    ax2.set_title("log loss: learning")
    fig.tight_layout()
    save(fig, "ch04-learning-curves.png")


def fig_ch04_decision_boundary(env):
    w_B, b_B = env["w_B"], env["b_B"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(env["X_normal"][:, 0], env["X_normal"][:, 1], marker="o",
               s=18, alpha=0.7, label="normal (t = 0)")
    ax.scatter(env["X_spam"][:, 0], env["X_spam"][:, 1], marker="x",
               s=22, alpha=0.7, label="spam (t = 1)")
    x1_line = np.linspace(-5.0, 5.0, 100)
    for p_level, style in [(0.1, "--"), (0.5, "-"), (0.9, "--")]:
        logit = np.log(p_level / (1.0 - p_level))
        x2_line = (logit - w_B[0] * x1_line - b_B) / w_B[1]
        ax.plot(x1_line, x2_line, style, label=f"$y = {p_level}$")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "ch04-decision-boundary.png")


# =====================================================================
# 図4.3: シグモイド関数 σ(z) と、正解 t=1 のときの1件分の損失(−log y と MSE)
# =====================================================================
def fig_ch04_sigmoid_logloss():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

    z = np.linspace(-8, 8, 400)
    sig = 1.0 / (1.0 + np.exp(-z))
    ax1.plot(z, sig, color="tab:blue")
    ax1.axhline(0.0, color="gray", ls=":", lw=1)
    ax1.axhline(1.0, color="gray", ls=":", lw=1)
    ax1.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax1.axvline(0.0, color="gray", ls="--", lw=0.8)
    ax1.plot([0.0], [0.5], "o", color="tab:orange", zorder=3)
    ax1.annotate(r"$\sigma(0) = 0.5$", xy=(0, 0.5), xytext=(1.5, 0.42), fontsize=9)
    ax1.set_xlabel("$z$")
    ax1.set_ylabel(r"$\sigma(z) = 1 / (1 + e^{-z})$")
    ax1.set_title("sigmoid")
    ax1.set_ylim(-0.05, 1.05)

    y = np.linspace(0.001, 1.0, 500)
    ax2.plot(y, -np.log(y), color="tab:orange", label=r"log loss: $-\log y$")
    ax2.plot(y, (y - 1.0) ** 2, color="tab:blue", ls="--",
             label=r"MSE: $(y - 1)^2$")
    ax2.axhline(1.0, color="gray", ls=":", lw=1)
    ax2.set_xlabel(r"model output $y$")
    ax2.set_ylabel("loss for one example")
    ax2.set_title("loss when the answer is $t = 1$")
    ax2.set_ylim(0, 7)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    save(fig, "ch04-sigmoid-logloss.png")


# =====================================================================
# 図5.1: 2値分布のエントロピー H(p)(底2)。p = 0.5 で最大 1 ビット
# =====================================================================
def fig_ch05_binary_entropy():
    p = np.linspace(1e-9, 1 - 1e-9, 1001)
    H = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(p, H, color="tab:blue")
    ax.axvline(0.5, color="gray", ls="--", lw=1)
    ax.plot([0.5], [1.0], "o", color="tab:orange", zorder=3)
    ax.annotate("max: 1 bit at $p = 1/2$", xy=(0.5, 1.0),
                xytext=(0.56, 0.88), fontsize=9)
    ax.annotate("certain:\n$H = 0$", xy=(0.0, 0.0), xytext=(0.03, 0.15), fontsize=9)
    ax.set_xlabel("$p$ (probability of one of the two outcomes)")
    ax.set_ylabel(r"$H(p) = -p \log_2 p - (1-p)\log_2(1-p)$  [bits]")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    save(fig, "ch05-binary-entropy.png")


# =====================================================================
# 図6.1: 温度による softmax の尖り方(スコア (2,1,0)、6.5 の表と同じ数値)
# =====================================================================
def fig_ch06_softmax_temperature():
    def softmax(z):
        e = np.exp(z - np.max(z))
        return e / e.sum()

    z = np.array([2.0, 1.0, 0.0])
    taus = [0.1, 0.5, 1.0, 2.0, 10.0]
    # 本文 6.5 の表の数値と一致することを確認
    assert np.allclose(np.round(softmax(z / 0.5), 3), [0.867, 0.117, 0.016])
    assert np.allclose(np.round(softmax(z / 1.0), 3), [0.665, 0.245, 0.090])
    assert np.allclose(np.round(softmax(z / 10.0), 3), [0.367, 0.332, 0.301])

    fig, axes = plt.subplots(1, len(taus), figsize=(7, 3.5), sharey=True)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for ax, tau in zip(axes, taus):
        p = softmax(z / tau)
        ax.bar([0, 1, 2], p, color=colors, width=0.7)
        ax.set_title(f"$T = {tau}$", fontsize=10)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["$z_1$", "$z_2$", "$z_3$"], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.text(1, 0.94, f"$H = {-(p * np.log(p)).sum():.2f}$",
                ha="center", fontsize=8, color="dimgray")
    axes[0].set_ylabel(r"$\mathrm{softmax}(\mathbf{z}/T)$")
    fig.suptitle(r"softmax of the same scores $\mathbf{z} = (2, 1, 0)$"
                 r" at different temperatures", fontsize=10)
    fig.tight_layout()
    save(fig, "ch06-softmax-temperature.png")


# =====================================================================
# 図7.1: 内積のヒストグラム(d_k = 4, 64, 512)。√d_k で割ると幅が揃う
# =====================================================================
def fig_ch07_dot_product_histograms():
    rng = np.random.default_rng(42)
    n_keys, n_trials = 64, 1000
    d_ks = [4, 64, 512]
    raw = {}
    for d_k in d_ks:
        scores = []
        for _ in range(n_trials):
            q = rng.normal(size=d_k)
            K = rng.normal(size=(n_keys, d_k))
            scores.append((K * q).sum(axis=1))
        raw[d_k] = np.concatenate(scores)
        # 本文 7.4 の観測1: 標準偏差は √d_k とほぼ一致
        assert abs(raw[d_k].std() / np.sqrt(d_k) - 1.0) < 0.05

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
    colors = {4: "tab:blue", 64: "tab:orange", 512: "tab:green"}
    for d_k in d_ks:
        ax1.hist(raw[d_k], bins=120, density=True, alpha=0.55,
                 color=colors[d_k], label=f"$d_k = {d_k}$")
        ax2.hist(raw[d_k] / np.sqrt(d_k), bins=60, density=True, alpha=0.55,
                 color=colors[d_k], label=f"$d_k = {d_k}$")
    ax1.set_xlim(-75, 75)
    ax1.set_xlabel(r"$\mathbf{q} \cdot \mathbf{k}$")
    ax1.set_ylabel("density")
    ax1.set_title("raw dot products (spread grows)")
    ax1.legend(fontsize=8)
    ax2.set_xlim(-4.5, 4.5)
    ax2.set_xlabel(r"$\mathbf{q} \cdot \mathbf{k} \, / \, \sqrt{d_k}$")
    ax2.set_title(r"divided by $\sqrt{d_k}$ (all match)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "ch07-dot-product-histograms.png")


if __name__ == "__main__":
    fig_ch02_running_mean()
    fig_ch03_likelihood_peak()
    env = _ch04_setup()
    fig_ch04_learning_curves(env)
    fig_ch04_decision_boundary(env)
    fig_ch04_sigmoid_logloss()
    fig_ch05_binary_entropy()
    fig_ch06_softmax_temperature()
    fig_ch07_dot_product_histograms()
    print("ok: all figures generated")
