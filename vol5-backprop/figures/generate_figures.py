# 第5巻 図版生成スクリプト
# 使い方: python3 generate_figures.py  (このディレクトリに chNN-*.png を出力する)
#
# - 図中の文字は英語と数式記号のみ(日本語フォント非依存)
# - 第6章の図は code/ch06/ の既存スクリプトを実行し、本文の表と同一の数値を描く
import contextlib
import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(os.path.dirname(HERE), "code")

# 配色(dataviz スキルの検証済みカテゴリカルパレット。固定順で使う)
C1_BLUE = "#2a78d6"
C2_AQUA = "#1baf7a"
C3_YELLOW = "#eda100"
C4_GREEN = "#008300"
BAND_FILL = "#cde2fb"   # sequential blue step 100(淡い塗り)
INK = "#0b0b0b"
GRID = "#dddddd"

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": "#52514e",
    "axes.labelcolor": INK,
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
})


def new_axes(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig, ax


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def run_script(relpath):
    """code/ 以下のスクリプトを(printを黙らせて)実行し、名前空間を返す。
    数値は本文の表とビット単位で同一になる。"""
    path = os.path.join(CODE, relpath)
    with open(path, encoding="utf-8") as f:
        src = f.read()
    ns = {"__name__": "__generate_figures__", "__file__": path}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, path, "exec"), ns)
    return ns


# ---------------------------------------------------------------- 図1.1
# XOR的な配置(code/ch01/xor_limit.py のデータBを、乱数の消費順まで含めて再現)
def fig_ch01_xor():
    rng = np.random.default_rng(42)
    n = 50

    def blob(cx, cy):
        return rng.normal(loc=[cx, cy], scale=0.7, size=(n, 2))

    blob(-2.0, -2.0), blob(2.0, 2.0)                       # データAが先に乱数を消費する
    X0 = np.vstack([blob(-2.0, -2.0), blob(2.0, 2.0)])     # ラベル0(左下・右上)
    X1 = np.vstack([blob(-2.0, 2.0), blob(2.0, -2.0)])     # ラベル1(左上・右下)

    fig, ax = new_axes((5.4, 4.4))
    ax.axhline(0, color="#52514e", linewidth=0.8, linestyle="--", zorder=1)
    ax.axvline(0, color="#52514e", linewidth=0.8, linestyle="--", zorder=1)
    ax.scatter(X0[:, 0], X0[:, 1], s=26, color=C1_BLUE, edgecolors="white",
               linewidths=0.4, label="label 0", zorder=3)
    ax.scatter(X1[:, 0], X1[:, 1], s=30, facecolors="white", edgecolors=C2_AQUA,
               linewidths=1.5, label="label 1", zorder=3)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    ax.legend(loc="center", framealpha=0.9, edgecolor="#dddddd",
              bbox_to_anchor=(0.5, 0.5), handletextpad=0.4)
    save(fig, "ch01-xor-checkerboard.png")


# ---------------------------------------------------------------- 図1.2
def fig_ch01_relu():
    z = np.linspace(-2.5, 2.0, 200)
    fig, ax = new_axes((5.6, 3.5))
    ax.plot(z, np.maximum(z, 0.0), color=C1_BLUE, linewidth=2)
    ax.plot([0], [0], "o", color=C1_BLUE, markersize=7)
    ax.annotate("kink at $z=0$", xy=(0, 0), xytext=(-1.6, 0.7), color=INK,
                arrowprops=dict(arrowstyle="->", color="#52514e", linewidth=0.9))
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.axvline(0, color="#52514e", linewidth=0.8, linestyle="--")
    ax.set_xlabel("$z$")
    ax.set_ylabel(r"$\mathrm{ReLU}(z)=\max(0,\ z)$")
    ax.set_xticks([-2, -1, 0, 1, 2])
    save(fig, "ch01-relu.png")


# ---------------------------------------------------------------- 図1.4
def fig_ch01_relu_step():
    x = np.linspace(-1.5, 2.5, 400)
    r = np.maximum(x, 0.0) - np.maximum(x - 1.0, 0.0)
    fig, ax = new_axes((5.6, 3.5))
    ax.plot(x, r, color=C1_BLUE, linewidth=2)
    ax.plot([0, 1], [0, 1], "o", color=C1_BLUE, markersize=7)
    ax.annotate("kink at $x=0$", xy=(0, 0), xytext=(-1.3, 0.45), color=INK,
                arrowprops=dict(arrowstyle="->", color="#52514e", linewidth=0.9))
    ax.annotate("kink at $x=1$", xy=(1, 1), xytext=(1.5, 0.55), color=INK,
                arrowprops=dict(arrowstyle="->", color="#52514e", linewidth=0.9))
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.axvline(0, color="#52514e", linewidth=0.8, linestyle="--")
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$r(x)=\mathrm{ReLU}(x)-\mathrm{ReLU}(x-1)$")
    ax.set_xticks([-1, 0, 1, 2])
    ax.set_ylim(-0.25, 1.35)
    save(fig, "ch01-relu-step.png")


# ---------------------------------------------------------------- 図1.3
def fig_ch01_sigmoid_tanh():
    z = np.linspace(-6, 6, 400)
    sig = 1.0 / (1.0 + np.exp(-z))
    fig, ax = new_axes((6.2, 3.8))
    for y in (-1.0, 0.0, 1.0):
        ax.axhline(y, color="#52514e", linewidth=0.7, linestyle="--")
    ax.plot(z, sig, color=C1_BLUE, linewidth=2)
    ax.plot(z, np.tanh(z), color=C2_AQUA, linewidth=2)
    ax.text(-4.6, 0.16, "sigmoid", color=C1_BLUE)
    ax.text(1.0, -0.55, "tanh", color=C2_AQUA)
    ax.annotate("saturates\n(slope $\\approx 0$)", xy=(4.6, 0.995), xytext=(3.4, 0.42),
                color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color="#52514e", linewidth=0.9))
    ax.annotate("saturates\n(slope $\\approx 0$)", xy=(-4.3, -0.985), xytext=(-2.6, -0.52),
                color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color="#52514e", linewidth=0.9))
    ax.set_xlabel("$z$")
    ax.set_ylabel(r"$\sigma(z)$ and $\tanh(z)$")
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    save(fig, "ch01-sigmoid-tanh.png")


# ---------------------------------------------------------------- 図2.1
# 手調整した2層MLP(code/ch02/mlp_forward.py の 2.3b と同じ重み)の決定境界
def fig_ch02_decision_boundary():
    W1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    b1 = np.array([0.0, -1.0])
    W2 = np.array([[1.0], [-2.0]])
    b2 = np.array([0.0])

    def mlp_forward(X):
        H = np.maximum(X @ W1 + b1, 0.0)
        return H @ W2 + b2

    xs = np.linspace(-0.5, 1.5, 401)
    gx, gy = np.meshgrid(xs, xs)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    pred = (mlp_forward(grid).ravel() > 0.5).reshape(gx.shape)

    fig, ax = new_axes((5.4, 4.6))
    ax.contourf(gx, gy, pred, levels=[0.5, 1.5], colors=[BAND_FILL], zorder=1)
    for c in (0.5, 1.5):    # 帯の縁 = 決定境界 x1 + x2 = 0.5, 1.5
        ax.plot(xs, c - xs, color=C1_BLUE, linewidth=2, zorder=2)
    ax.scatter([0, 1], [1, 0], marker="x", s=90, color=INK, linewidths=2.2,
               label="label 1", zorder=3)
    ax.scatter([0, 1], [0, 1], s=80, facecolors="white", edgecolors=INK,
               linewidths=1.8, label="label 0", zorder=3)
    ax.text(0.5, 0.5, "predicted 1\n$0.5 < x_1+x_2 < 1.5$",
            ha="center", va="center", color="#184f95")
    ax.text(-0.27, -0.32, "predicted 0", color="#52514e")
    ax.text(1.44, 1.1, "predicted 0", color="#52514e", ha="right")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="#dddddd")
    save(fig, "ch02-decision-boundary.png")


# ---------------------------------------------------------------- 図6.1
# 勾配消失: 本文6.1の表と同じ数値(vanishing_gradients.py の実行結果そのもの)
def fig_ch06_vanishing():
    ns = run_script("ch06/vanishing_gradients.py")
    norms = ns["norms"]
    layers = np.arange(1, len(norms) + 1)

    fig, ax = new_axes((6.2, 4.0))
    ax.semilogy(layers, norms, "-o", color=C1_BLUE, linewidth=2, markersize=6)
    ax.text(1.35, 1.35e-7, r"input: $\approx 1.6 \times 10^{-7}$",
            color=INK, va="center")
    ax.text(9.75, norms[-1], r"output: $\approx 1.4$",
            color=INK, va="center", ha="right")
    ax.text(2.4, 2e-3, r"shrinks $\sim \times\, 1/10$ per layer", color="#52514e")
    ax.set_xlabel("layer $l$  (1 = input side, 10 = output side)")
    ax.set_ylabel(r"$\|\partial L/\partial W_l\|$  (log scale)")
    ax.set_xticks(layers)
    ax.set_ylim(6e-8, 12)
    save(fig, "ch06-vanishing-gradients.png")


# ---------------------------------------------------------------- 図6.2
# 4変種の比較: 本文6.5の表と同じ数値(residual_comparison.py の実行結果そのもの)
def fig_ch06_residual_comparison():
    ns = run_script("ch06/residual_comparison.py")
    results = ns["results"]
    variants = ns["VARIANTS"]
    blocks = np.arange(1, ns["depth"] + 1)
    colors = [C1_BLUE, C2_AQUA, C3_YELLOW, C4_GREEN]
    markers = ["o", "s", "^", "D"]
    styles = ["-", "-", "--", "-"]   # +ln は +dropout とほぼ重なるため破線で見せる

    fig, ax = new_axes((7.0, 4.5))
    for v, c, m, s in zip(variants, colors, markers, styles):
        ax.semilogy(blocks, results[v], marker=m, color=c, linewidth=2,
                    linestyle=s, markersize=5.5, label=v)
    ax.text(1.25, 1.05e-8, "plain: exponential decay", color="#184f95")
    ax.text(5.2, 8.5, "residual variants stay flat", color=INK)
    ax.set_xlabel("block $l$  (1 = input side, 10 = output side)")
    ax.set_ylabel(r"$\|\partial L/\partial W_1^{(l)}\|$  (log scale)")
    ax.set_xticks(blocks)
    ax.set_ylim(3e-9, 60)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="#dddddd")
    save(fig, "ch06-residual-comparison.png")


if __name__ == "__main__":
    # macOS の Accelerate BLAS は matmul で偽の浮動小数点フラグを立てることがあり
    # (単体実行では出ない RuntimeWarning が matplotlib 併用時に出る)、ここで抑止する。
    # 数値の正しさは code/ 側スクリプト内の assert が保証している。
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    fig_ch01_xor()
    fig_ch01_relu()
    fig_ch01_relu_step()
    fig_ch01_sigmoid_tanh()
    fig_ch02_decision_boundary()
    fig_ch06_vanishing()
    fig_ch06_residual_comparison()
    print("done: 7 figures")
