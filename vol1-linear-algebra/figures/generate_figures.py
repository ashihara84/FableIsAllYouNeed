"""第1巻(線形代数)の全図版を生成するスクリプト。

実行するとこのファイルと同じディレクトリに PNG を出力する:

    python3 generate_figures.py

図中の文字は英語と数式記号のみ(日本語の説明は本文キャプション側が担う)。
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)  # 全巻共通の作法(この巻の図は乱数を使わないが種は固定)

HERE = os.path.dirname(os.path.abspath(__file__))
DPI = 160

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.edgecolor": "0.3",
        "axes.labelcolor": "0.15",
        "xtick.color": "0.3",
        "ytick.color": "0.3",
    }
)

BLUE = "tab:blue"
ORANGE = "tab:orange"
GREEN = "tab:green"
RED = "tab:red"
GRAY = "0.55"


def new_axes(figsize, xlim, ylim, xlabel="$x_1$", ylabel="$x_2$"):
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    setup(ax, xlim, ylim, xlabel, ylabel)
    return fig, ax


def setup(ax, xlim, ylim, xlabel="$x_1$", ylabel="$x_2$"):
    ax.set_facecolor("white")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.grid(True, color="0.88", lw=0.7)
    ax.axhline(0, color="0.45", lw=0.9, zorder=1)
    ax.axvline(0, color="0.45", lw=0.9, zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation=0, labelpad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def arrow(ax, start, end, color, lw=2.2, ls="-", zorder=3):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            linestyle=ls,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=16,
        ),
        zorder=zorder,
    )


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# 図1.1 ベクトルの足し算 = 矢印の継ぎ足し (a=(2,1), b=(1,3), a+b=(3,4))
# ---------------------------------------------------------------------------
def fig_ch01_vector_addition():
    fig, ax = new_axes((5, 4.5), (-0.4, 4.2), (-0.4, 4.6))
    arrow(ax, (0, 0), (2, 1), BLUE)
    arrow(ax, (2, 1), (3, 4), ORANGE)
    arrow(ax, (0, 0), (3, 4), GREEN, lw=2.6)
    ax.text(1.15, 0.28, r"$\mathbf{a} = (2,\ 1)$", color=BLUE)
    ax.text(2.72, 2.35, r"$\mathbf{b} = (1,\ 3)$", color=ORANGE)
    ax.text(0.55, 2.55, r"$\mathbf{a}+\mathbf{b} = (3,\ 4)$", color=GREEN)
    ax.set_xticks(range(0, 5))
    ax.set_yticks(range(0, 5))
    save(fig, "ch01-vector-addition.png")


# ---------------------------------------------------------------------------
# 図1.2 スカラー倍 = 矢印の伸縮(3倍・0.5倍・−1倍)
# ---------------------------------------------------------------------------
def fig_ch01_scalar_multiplication():
    fig, axes = plt.subplots(1, 3, figsize=(7, 3), facecolor="white")
    v = np.array([2.0, 1.0])
    cases = [(3.0, r"$3\mathbf{v} = (6,\ 3)$", GREEN),
             (0.5, r"$0.5\,\mathbf{v} = (1,\ 0.5)$", ORANGE),
             (-1.0, r"$-\mathbf{v} = (-2,\ -1)$", RED)]
    for ax, (c, label, color) in zip(axes, cases):
        setup(ax, (-3, 7), (-2.4, 4.0))
        cv = c * v
        arrow(ax, (0, 0), tuple(cv), color, lw=2.4)
        arrow(ax, (0, 0), tuple(v), GRAY, lw=1.4, ls="--")
        ax.text(1.5, 1.35, r"$\mathbf{v}$", color="0.35")
        ax.set_title(label, fontsize=11, color=color)
        ax.set_xticks([-2, 0, 2, 4, 6])
        ax.set_yticks([-2, 0, 2, 4])
        ax.set_ylabel("")
    axes[0].set_ylabel("$x_2$", rotation=0, labelpad=8)
    fig.tight_layout()
    save(fig, "ch01-scalar-multiplication.png")


# ---------------------------------------------------------------------------
# 図2.1 内積の幾何: a=(1,1), b=(2,0), θ=45°
# ---------------------------------------------------------------------------
def fig_ch02_dot_product_angle():
    fig, ax = new_axes((5.5, 3.3), (-0.25, 2.5), (-0.25, 1.65))
    arrow(ax, (0, 0), (1, 1), BLUE)
    arrow(ax, (0, 0), (2, 0), ORANGE)
    theta = np.linspace(0, np.pi / 4, 40)
    r = 0.45
    ax.plot(r * np.cos(theta), r * np.sin(theta), color="0.35", lw=1.2)
    ax.text(0.52, 0.14, r"$\theta = 45°$", color="0.25")
    ax.text(0.62, 1.02, r"$\mathbf{a} = (1,\ 1),\ \ \|\mathbf{a}\| = \sqrt{2}$", color=BLUE)
    ax.text(1.42, 0.12, r"$\mathbf{b} = (2,\ 0),\ \ \|\mathbf{b}\| = 2$", color=ORANGE)
    ax.text(
        0.05,
        1.45,
        r"$\mathbf{a}\cdot\mathbf{b} \;=\; \|\mathbf{a}\|\,\|\mathbf{b}\|\cos\theta"
        r" \;=\; \sqrt{2}\times 2 \times \frac{1}{\sqrt{2}} \;=\; 2$",
        color="0.15",
    )
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1])
    save(fig, "ch02-dot-product-angle.png")


# ---------------------------------------------------------------------------
# 図5.1 90度回転: x=(2,1) → Rx=(-1,2)
# ---------------------------------------------------------------------------
def fig_ch05_rotation():
    fig, ax = new_axes((5.8, 3.6), (-2.4, 3.2), (-0.5, 2.9), xlabel="$x$", ylabel="$y$")
    arrow(ax, (0, 0), (2, 1), BLUE)
    arrow(ax, (0, 0), (-1, 2), ORANGE)
    ax.plot([2], [1], "o", color=BLUE, ms=7, zorder=4)
    ax.plot([-1], [2], "o", mfc="white", mec=ORANGE, mew=1.8, ms=7, zorder=4)
    # 回転を示す円弧(半径1.35、x の角度から Rx の角度まで)
    a0 = np.arctan2(1, 2)
    a1 = np.arctan2(2, -1)
    rad = 1.35
    p0 = (rad * np.cos(a0 + 0.12), rad * np.sin(a0 + 0.12))
    p1 = (rad * np.cos(a1 - 0.12), rad * np.sin(a1 - 0.12))
    ax.annotate(
        "",
        xy=p1,
        xytext=p0,
        arrowprops=dict(
            arrowstyle="-|>",
            color="0.4",
            lw=1.3,
            connectionstyle="arc3,rad=0.42",
            mutation_scale=13,
        ),
        zorder=2,
    )
    amid = (a0 + a1) / 2
    ax.text(1.62 * np.cos(amid) - 0.18, 1.62 * np.sin(amid), r"$90°$", color="0.3")
    ax.text(2.02, 0.62, r"$\mathbf{x} = (2,\ 1)$", color=BLUE)
    ax.text(-2.25, 2.25, r"$R\mathbf{x} = (-1,\ 2)$", color=ORANGE)
    ax.set_xticks(range(-2, 4))
    ax.set_yticks(range(0, 3))
    save(fig, "ch05-rotation.png")


# ---------------------------------------------------------------------------
# 図5.2 拡大縮小: (2,1) → (4,0.5)、円 → 横長の楕円
# ---------------------------------------------------------------------------
def fig_ch05_scaling():
    fig, ax = new_axes((6.5, 3.6), (-5.2, 5.6), (-2.9, 3.0), xlabel="$x$", ylabel="$y$")
    t = np.linspace(0, 2 * np.pi, 200)
    r = np.sqrt(5)  # (2,1) を通る円
    ax.plot(r * np.cos(t), r * np.sin(t), ls="--", color="0.6", lw=1.2, zorder=2)
    ax.plot(2 * r * np.cos(t), 0.5 * r * np.sin(t), ls="--", color=ORANGE, lw=1.2, zorder=2)
    ax.plot([2], [1], "o", color=BLUE, ms=7, zorder=4)
    ax.plot([4], [0.5], "o", mfc="white", mec=ORANGE, mew=1.8, ms=7, zorder=4)
    ax.annotate(
        "",
        xy=(3.85, 0.56),
        xytext=(2.18, 0.95),
        arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.3, ls="--", mutation_scale=13),
        zorder=3,
    )
    ax.text(1.0, 1.45, r"$(2,\ 1)$", color=BLUE)
    ax.text(3.7, 1.0, r"$(4,\ 0.5)$", color=ORANGE)
    ax.text(-4.9, 2.45, r"$S:\ x \times 2,\ \ y \times 0.5$", color="0.25")
    ax.text(-2.5, -2.15, "circle", color="0.5", fontsize=10)
    ax.text(2.9, -1.55, "ellipse", color=ORANGE, fontsize=10)
    ax.set_xticks(range(-4, 6, 2))
    ax.set_yticks(range(-2, 3))
    save(fig, "ch05-scaling.png")


# ---------------------------------------------------------------------------
# 図5.3 x軸への射影: (2,5) と (2,1) がどちらも (2,0) に落ちる
# ---------------------------------------------------------------------------
def fig_ch05_projection():
    fig, ax = new_axes((5, 4.5), (-0.8, 5.4), (-0.7, 5.6), xlabel="$x$", ylabel="$y$")
    ax.plot([2, 2], [0.16, 5], ls="--", color="0.55", lw=1.2, zorder=2)
    ax.annotate(
        "",
        xy=(2, 0.12),
        xytext=(2, 0.9),
        arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.3, mutation_scale=13),
        zorder=3,
    )
    ax.plot([2], [5], "o", color=BLUE, ms=7, zorder=4)
    ax.plot([2], [1], "o", color=BLUE, ms=7, zorder=4)
    ax.plot([2], [0], "o", mfc="white", mec=ORANGE, mew=1.8, ms=8, zorder=4)
    ax.text(2.25, 4.9, r"$(2,\ 5)$", color=BLUE)
    ax.text(2.25, 1.05, r"$(2,\ 1)$", color=BLUE)
    ax.text(2.3, 0.28, r"$(2,\ 0)$", color=ORANGE)
    ax.text(3.15, 2.6, "$P$: project\nonto the $x$-axis", color="0.3", fontsize=10)
    ax.set_xticks(range(0, 6))
    ax.set_yticks(range(0, 6))
    save(fig, "ch05-projection.png")


if __name__ == "__main__":
    fig_ch01_vector_addition()
    fig_ch01_scalar_multiplication()
    fig_ch02_dot_product_angle()
    fig_ch05_rotation()
    fig_ch05_scaling()
    fig_ch05_projection()
    print("all figures generated")
