"""第2巻(微分・最適化)の図版を一括生成するスクリプト。

実行:  python3 generate_figures.py
出力:  このディレクトリに chNN-*.png を生成する(dpi=160, 白背景)。

規約:
- 図中の文字は英語と数式記号のみ(日本語説明は本文キャプションが担う)
- figsize は (5, 3.5) 〜 (7, 4.5)
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

rng = np.random.default_rng(42)

FIGDIR = os.path.dirname(os.path.abspath(__file__))

GRAY = "0.45"       # 主役の曲線・等高線
BLUE = "#1f77b4"    # 勾配ベクトル(図4.1 キャプションの「青い矢印」)
RED = "crimson"     # 降下の軌跡(図4.2 キャプションの「赤い点」)


def save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {name}")


# ----------------------------------------------------------------------
# 図1.1  割線 → 接線(f(x) = x^2, x = 3)
# ----------------------------------------------------------------------
def ch01_secant_to_tangent():
    def f(x):
        return x ** 2

    x0 = 3.0
    xs = np.linspace(-0.2, 6.2, 300)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, f(xs), color=GRAY, lw=2, label="$f(x) = x^2$")

    xline = np.linspace(0.8, 5.9, 2)
    for h, c in [(2.0, "#2ca02c"), (1.0, "#9467bd"), (0.5, "#ff7f0e")]:
        slope = (f(x0 + h) - f(x0)) / h
        ax.plot(xline, f(x0) + slope * (xline - x0), color=c, lw=1.4,
                label=f"secant $h = {h:g}$  (slope {slope:g})")
        ax.plot([x0 + h], [f(x0 + h)], "o", color=c, ms=5)

    ax.plot(xline, f(x0) + 6.0 * (xline - x0), "--", color="red", lw=2,
            label="tangent  (slope 6)")
    ax.plot([x0], [f(x0)], "ko", ms=6, zorder=5)
    ax.annotate("$(3,\\ 9)$", xy=(x0, f(x0)), xytext=(3.35, 7.6), fontsize=10)

    ax.set_xlim(-0.2, 6.2)
    ax.set_ylim(-4, 40)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "ch01-secant-to-tangent.png")


# ----------------------------------------------------------------------
# 図1.2  h と誤差の関係(前進差分 vs 中心差分, 両対数)
# ----------------------------------------------------------------------
def ch01_error_vs_h():
    def f(x):
        return x ** 3 - 2 * x

    x0, true = 2.0, 10.0  # f'(x) = 3x^2 - 2, f'(2) = 10
    hs = np.logspace(-13, -1, 25)
    fwd = np.abs((f(x0 + hs) - f(x0)) / hs - true)
    ctr = np.abs((f(x0 + hs) - f(x0 - hs)) / (2 * hs) - true)
    tiny = 1e-17  # log 軸のためのガード
    fwd = np.maximum(fwd, tiny)
    ctr = np.maximum(ctr, tiny)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(hs, fwd, "o-", ms=4, lw=1.3, label="forward difference")
    ax.loglog(hs, ctr, "s-", ms=4, lw=1.3, label="central difference")
    ax.set_xlabel("$h$")
    ax.set_ylabel("absolute error")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=9)
    save(fig, "ch01-error-vs-h.png")


# ----------------------------------------------------------------------
# 図2.1  f(x) = (x-2)^2 + 1 と各点での傾き
# ----------------------------------------------------------------------
def ch02_slope_signs():
    def f(x):
        return (x - 2) ** 2 + 1

    xs = np.linspace(-0.5, 4.5, 300)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, f(xs), color=GRAY, lw=2)

    labels = {
        0: dict(xy=(0.25, 5.15), ha="left"),
        1: dict(xy=(1.2, 2.35), ha="left"),
        2: dict(xy=(2.0, 0.35), ha="center"),
        3: dict(xy=(2.8, 2.35), ha="right"),
        4: dict(xy=(3.75, 5.15), ha="right"),
    }
    for xp in [0, 1, 2, 3, 4]:
        s = 2 * (xp - 2)
        xt = np.linspace(xp - 0.55, xp + 0.55, 2)
        ax.plot(xt, f(xp) + s * (xt - xp), color="red", lw=1.6)
        ax.plot([xp], [f(xp)], "ko", ms=5, zorder=5)
        text = f"slope ${s:+d}$" if s != 0 else "slope $0$"
        ax.annotate(text, xy=labels[xp]["xy"], ha=labels[xp]["ha"],
                    fontsize=10, color="red")

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.4, 8)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_title("$f(x) = (x-2)^2 + 1$", fontsize=11)
    save(fig, "ch02-slope-signs.png")


# ----------------------------------------------------------------------
# 図2.2  f(x) = x^4 - 2x^2 の停留点(谷底2つ + 山頂1つ)
# ----------------------------------------------------------------------
def ch02_stationary_points():
    def f(x):
        return x ** 4 - 2 * x ** 2

    xs = np.linspace(-1.62, 1.62, 400)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, f(xs), color=GRAY, lw=2)

    for xp, name, dy in [(-1.0, "min", -0.28), (0.0, "max", 0.22),
                         (1.0, "min", -0.28)]:
        ax.plot([xp - 0.35, xp + 0.35], [f(xp), f(xp)], "--",
                color="red", lw=1.4)
        ax.plot([xp], [f(xp)], "o", color="red", ms=7, zorder=5)
        ax.annotate(name, xy=(xp, f(xp) + dy), ha="center",
                    fontsize=10, color="red")

    ax.annotate("$f'(x) = 0$ at 3 points", xy=(0, 1.55), ha="center",
                fontsize=10)
    ax.set_xlim(-1.62, 1.62)
    ax.set_ylim(-1.55, 2.1)
    ax.set_xticks([-1, 0, 1])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_title("$f(x) = x^4 - 2x^2$", fontsize=11)
    save(fig, "ch02-stationary-points.png")


# ----------------------------------------------------------------------
# 図3.1  放物線の上の勾配降下: η による収束・振動・発散(4パネル)
# ----------------------------------------------------------------------
def ch03_gd_learning_rates():
    def f(x):
        return x ** 2

    def descend(x0, lr, n_steps):
        xs = [x0]
        x = x0
        for _ in range(n_steps):
            x = x - lr * 2 * x
            xs.append(x)
        return np.array(xs)

    cases = [
        (0.05, 10, "converge"),
        (0.8, 10, "zigzag converge"),
        (1.0, 10, "oscillate"),
        (1.05, 6, "diverge"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7, 4.5))
    for i, (ax, (lr, n, word)) in enumerate(zip(axes.flat, cases)):
        xs = descend(10.0, lr, n)
        m = max(12.0, 1.12 * np.max(np.abs(xs)))
        xc = np.linspace(-m, m, 400)
        ax.plot(xc, f(xc), color="0.6", lw=1.5)
        ax.plot(xs, f(xs), "o-", color=RED, ms=4, lw=1.1)
        ax.plot([xs[0]], [f(xs[0])], "o", color="k", ms=5, zorder=5)
        if i == 0:
            ax.annotate("start", xy=(xs[0], f(xs[0])),
                        xytext=(xs[0] - 1.2, f(xs[0]) + 12),
                        ha="right", fontsize=9)
        ax.set_title(f"$\\eta = {lr:g}$ : {word}", fontsize=10)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    save(fig, "ch03-gd-learning-rates.png")


# ----------------------------------------------------------------------
# 図4.1  f(x,y) = x^2 + 3y^2 の等高線と勾配ベクトル(5点)
# ----------------------------------------------------------------------
def ch04_contour_gradient():
    def f(x, y):
        return x ** 2 + 3 * y ** 2

    xg = np.linspace(-4.6, 4.6, 400)
    yg = np.linspace(-2.7, 2.7, 400)
    X, Y = np.meshgrid(xg, yg)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    cs = ax.contour(X, Y, f(X, Y), levels=[1, 3, 6, 12, 20],
                    colors="0.6", linewidths=1.0)
    ax.clabel(cs, fmt="%g", fontsize=8, colors="0.4")

    scale = 0.13
    for px, py in [(3, 1), (2, 0), (0, -1), (-2, 1), (-1, -1)]:
        gx, gy = 2 * px, 6 * py
        ax.annotate("", xy=(px + scale * gx, py + scale * gy),
                    xytext=(px, py),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.8))
        ax.plot([px], [py], "o", color=BLUE, ms=4, zorder=5)

    ax.annotate("$\\nabla f(3,1) = (6,\\ 6)$", xy=(3.0, 0.62), ha="center",
                fontsize=9, color=BLUE)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("$f(x, y) = x^2 + 3y^2$", fontsize=11)
    save(fig, "ch04-contour-gradient.png")


# ----------------------------------------------------------------------
# 図4.2  等高線 + (3.0, 1.5) からの勾配降下の軌跡(η = 0.1, 30歩)
# ----------------------------------------------------------------------
def ch04_gd_trajectory():
    def f(x, y):
        return x ** 2 + 3 * y ** 2

    # 更新則 x←0.8x, y←0.4y の閉じた式(本文 4.4 の予言と同じ)
    k = np.arange(31)
    tx = 3.0 * 0.8 ** k
    ty = 1.5 * 0.4 ** k

    xg = np.linspace(-0.9, 4.1, 400)
    yg = np.linspace(-0.7, 2.1, 400)
    X, Y = np.meshgrid(xg, yg)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.contour(X, Y, f(X, Y), levels=[0.2, 1, 3, 6, 10, 15.75],
               colors="0.6", linewidths=1.0)

    ax.plot(tx, ty, "o-", color=RED, ms=4, lw=1.1, zorder=5)
    ax.plot([tx[0]], [ty[0]], "o", color="k", ms=6, zorder=6)
    ax.annotate("start $(3.0,\\ 1.5)$", xy=(tx[0], ty[0]),
                xytext=(2.75, 1.75), fontsize=9)
    ax.plot([0], [0], "*", color="k", ms=10, zorder=6)
    ax.annotate("minimum $(0,\\ 0)$", xy=(0, 0), xytext=(-0.75, -0.5),
                fontsize=9)

    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("$f(x, y) = x^2 + 3y^2, \\quad \\eta = 0.1$", fontsize=11)
    save(fig, "ch04-gd-trajectory.png")


# ----------------------------------------------------------------------
# 図6.1  フィッティングの前後(データ点 + 当てはめた直線)
# ----------------------------------------------------------------------
def ch06_fit_before_after():
    X = np.arange(5)
    y = 2 * X - 1
    xs = np.array([-0.35, 4.35])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, 0 * xs, "--", color="0.5", lw=1.6,
            label="before: $w = 0,\\ b = 0$")
    ax.plot(xs, 2 * xs - 1, color=BLUE, lw=2,
            label="after: $w = 2,\\ b = -1$")
    ax.plot(X, y, "o", color="k", ms=7, zorder=5, label="data (5 points)")

    ax.set_xlim(-0.5, 4.5)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "ch06-fit-before-after.png")


# ----------------------------------------------------------------------
# 図6.2  warmup + decay の学習率スケジュール
# ----------------------------------------------------------------------
def ch06_lr_schedule():
    steps = np.arange(1, 401)
    warmup = 30
    lr = 0.04 * np.minimum(steps / warmup, np.sqrt(warmup / steps))

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(steps, lr, color=BLUE, lw=2)
    ax.axvline(warmup, ls="--", color="0.6", lw=1.2)
    ax.annotate("warmup ends\n(step 30)", xy=(warmup, 0.04),
                xytext=(55, 0.0375), fontsize=9, color="0.3")
    ax.annotate("linear warmup", xy=(12, 0.016), xytext=(58, 0.009),
                fontsize=9, color="0.3",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=1.0))
    ax.annotate("$\\propto 1/\\sqrt{t}$ decay", xy=(220, 0.019),
                fontsize=9, color="0.3")

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 0.045)
    ax.set_xlabel("step $t$")
    ax.set_ylabel("learning rate $\\eta_t$")
    save(fig, "ch06-lr-schedule.png")


if __name__ == "__main__":
    ch01_secant_to_tangent()
    ch01_error_vs_h()
    ch02_slope_signs()
    ch02_stationary_points()
    ch03_gd_learning_rates()
    ch04_contour_gradient()
    ch04_gd_trajectory()
    ch06_fit_before_after()
    ch06_lr_schedule()
    print("done: 9 figures")
