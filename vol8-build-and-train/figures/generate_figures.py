# -*- coding: utf-8 -*-
"""第8巻の図版を一括生成するスクリプト。

- 依存は numpy + matplotlib のみ(torch 不要。訓練は実行しない)
- プロットに使う数値はすべて本文の数表・実行ログに実在するもののハードコード
- 図中の文字は英語と数式記号のみ(日本語フォント非依存)

実行: python3 generate_figures.py
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 共通スタイル -----------------------------------------------------------
INK = "#0b0b0b"        # 主要テキスト
INK_2 = "#52514e"      # 補助テキスト
GRID = "#e4e3e0"       # 目盛り線(控えめに)
BLUE = "#2a78d6"       # series 1
AQUA = "#1baf7a"       # series 2
RED = "#e34948"        # 「壊れた」側の系列(状態色)

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "axes.titlesize": 9.5,
    "axes.titlecolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "legend.frameon": False,
})


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


# ---- 図1.1 パラメータ数の内訳(第1章 1.6 param_count.py の出力) -------------
def fig_ch01_param_breakdown():
    labels = ["Embedding\n(+ output head)", "Encoder\n(6 layers)", "Decoder\n(6 layers)"]
    values = [64_000, 298_368, 397_440]
    total = sum(values)
    assert total == 759_808  # 本文 1.6 の合計と一致すること

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.58, color=BLUE, zorder=3)
    for yi, v in zip(y, values):
        ax.annotate(f"{v:,}  ({v / total * 100:.1f}%)",
                    (v, yi), xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8.5, color=INK)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 480_000)
    ax.set_xlabel("parameters")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("N=6, d_model=64, d_ff=256, h=8, vocab=1,000   —   total 759,808",
                 loc="left", color=INK_2)
    save(fig, "ch01-param-breakdown.png")


# ---- 図4.1 学習率スケジュール(第4章 4.6、論文 5.3 の式) --------------------
def lrate(step, d_model, warmup):
    step = np.asarray(step, dtype=float)
    return d_model ** -0.5 * np.minimum(step ** -0.5, step * warmup ** -1.5)


def lrate_no_warmup(step, d_model):
    step = np.asarray(step, dtype=float)
    return d_model ** -0.5 * step ** -0.5


def fig_ch04_lr_schedule():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))

    # 左: 論文スケール(d_model=512, warmup=4000)。山型が見える線形軸
    steps = np.arange(1, 20001)
    lr = lrate(steps, 512, 4000)
    peak = lrate(4000, 512, 4000)
    ax1.plot(steps, lr, color=BLUE, lw=1.8)
    ax1.axvline(4000, color=INK_2, lw=0.8, ls=":")
    ax1.annotate(f"peak {peak:.1e}\nat step 4000", (4000, peak),
                 xytext=(6, -2), textcoords="offset points",
                 fontsize=8, color=INK_2, va="top")
    ax1.set_xlabel("step")
    ax1.set_ylabel("lrate")
    ax1.set_ylim(0, peak * 1.15)
    ax1.set_title("paper 5.3:  d_model=512,  warmup=4000", loc="left")
    ax1.yaxis.grid(True)
    ax1.set_axisbelow(True)

    # 右: 4.6 の実験スケール(d_model=64, warmup=100)。warmup あり/なしの比較
    steps = np.arange(1, 401)
    with_w = lrate(steps, 64, 100)
    without = lrate_no_warmup(steps, 64)
    ax2.plot(steps, without, color=RED, lw=1.8, ls="--")
    ax2.plot(steps, with_w, color=BLUE, lw=1.8)
    ax2.set_yscale("log")
    ax2.annotate("no warmup\n(starts at 0.125)", (10, 0.06),
                 fontsize=8, color=RED)
    ax2.annotate("warmup\n(starts at 0.000125)", (135, 0.0006),
                 fontsize=8, color=BLUE)
    ax2.annotate("", xy=(1.6, 0.125), xytext=(1.6, 0.000125),
                 arrowprops=dict(arrowstyle="<->", color=INK_2, lw=0.8))
    ax2.annotate("$\\times 1000$", (2.0, 0.004), fontsize=8, color=INK_2)
    ax2.set_xlabel("step")
    ax2.set_ylabel("lrate (log scale)")
    ax2.set_title("experiment 4.6:  d_model=64,  warmup=100", loc="left")
    ax2.yaxis.grid(True)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    save(fig, "ch04-lr-schedule.png")


# ---- 図4.2 warmup あり/なしの loss(第4章 4.6 の実行ログ) -------------------
def fig_ch04_warmup_loss():
    steps = [1, 100, 400]
    loss_warmup = [3.510, 0.001, 0.000]   # 本文 4.6 実行ログ
    loss_none = [3.510, 3.486, 2.481]     # 同上
    chance = np.log(32)                   # 当てずっぽう(32種一様)= ln 32 ≈ 3.47

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ax.axhline(chance, color=INK_2, lw=0.9, ls=":")
    ax.annotate("uniform guess  ln 32 $\\approx$ 3.47", (400, chance),
                xytext=(0, 4), textcoords="offset points",
                ha="right", fontsize=8, color=INK_2)
    ax.plot(steps, loss_none, color=RED, lw=1.8, ls="--", marker="s", ms=5,
            label="no warmup")
    ax.plot(steps, loss_warmup, color=BLUE, lw=1.8, marker="o", ms=5,
            label="warmup (paper 5.3)")
    for x, y in zip(steps, loss_none):
        ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=8, color=RED)
    # step 1 は両者同値(3.510)なので、blue 側の値ラベルは 100/400 のみ
    for x, y in zip(steps[1:], loss_warmup[1:]):
        ax.annotate(f"{y:.3f}", (x, y), xytext=(0, -13), textcoords="offset points",
                    ha="center", fontsize=8, color=BLUE)
    ax.set_xlabel("step")
    ax.set_ylabel("loss (copy task)")
    ax.set_xlim(-14, 420)
    ax.set_ylim(-0.55, 4.1)
    ax.set_xticks(steps)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="center right", fontsize=8.5)
    save(fig, "ch04-warmup-loss.png")


# ---- 図5.2 訓練 loss の推移(第5章 5.1 train.py の数表) ---------------------
def fig_ch05_loss_curve():
    steps = [8, 96, 192, 288, 384, 480, 576, 672, 768, 864, 960, 1000]
    loss_train = [7.726, 1.504, 0.982, 0.927, 0.989, 1.029,
                  0.956, 0.940, 0.913, 0.937, 1.115, 0.943]
    loss_unseen = [7.728, 1.501, 1.101, 1.077, 1.151, 1.180,
                   1.136, 1.122, 1.103, 1.117, 1.475, 1.145]
    chance = np.log(275)   # 当てずっぽう(語彙275一様)≈ 5.6
    floor = 0.9            # label smoothing の床(本文 5.1、約0.9)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axhline(chance, color=INK_2, lw=0.9, ls=":")
    ax.annotate("uniform guess  ln 275 $\\approx$ 5.6", (1000, chance),
                xytext=(0, 4), textcoords="offset points",
                ha="right", fontsize=8, color=INK_2)
    ax.axhline(floor, color=INK_2, lw=0.9, ls=":")
    ax.annotate("label-smoothing floor $\\approx$ 0.9", (1000, floor),
                xytext=(0, -11), textcoords="offset points",
                ha="right", fontsize=8, color=INK_2)
    ax.plot(steps, loss_unseen, color=AQUA, lw=1.8, ls="--", marker="s", ms=4.5,
            label="loss (unseen, 25 pairs)")
    ax.plot(steps, loss_train, color=BLUE, lw=1.8, marker="o", ms=4.5,
            label="loss (train, 225 pairs)")
    ax.annotate("7.73 — worse than guessing", (8, 7.726),
                xytext=(10, 0), textcoords="offset points",
                va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("step")
    ax.set_ylabel("loss (label smoothing $\\epsilon_{ls}$=0.1)")
    ax.set_xlim(-20, 1030)
    ax.set_ylim(0, 8.3)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88), fontsize=8.5)
    save(fig, "ch05-loss-curve.png")


# ---- 図5.1 cross-attention マップ(第5章 5.3 attention_map.py の数値表) -----
def fig_ch05_cross_attention():
    # 本文 5.3 の数値表そのまま。トークンはローマ字表記(かれ→kare 等)
    A = np.array([
        [1.00, 0.00, 0.00, 0.00],   # かれ
        [0.52, 0.25, 0.23, 0.00],   # は
        [0.03, 0.00, 0.75, 0.22],   # さかな
        [0.25, 0.75, 0.00, 0.00],   # を
        [0.00, 1.00, 0.00, 0.00],   # たべます
        [0.35, 0.01, 0.25, 0.39],   # <eos>
    ])
    src_toks = ["he</w>", "eats</w>", "fish</w>", "<eos>"]
    out_toks = ["kare</w>", "wa</w>", "sakana</w>", "wo</w>",
                "tabemasu</w>", "<eos>"]

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    im = ax.imshow(A, cmap="Greys", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(src_toks)), src_toks, rotation=20, ha="right")
    ax.set_yticks(range(len(out_toks)), out_toks)
    ax.set_xlabel("input position (key)")
    ax.set_ylabel("output step (query)")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = A[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if v > 0.55 else INK)
    ax.set_title('src: "he eats fish"   (last decoder layer, mean of 4 heads)',
                 loc="left", color=INK_2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("attention weight", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)
    save(fig, "ch05-cross-attention.png")


# ---- 図6.1 modified n-gram precision(第6章 6.1 rug の例) -------------------
def fig_ch06_bleu_ngram():
    labels = ["$p_1$\n(1-gram)", "$p_2$\n(2-gram)", "$p_3$\n(3-gram)", "$p_4$\n(4-gram)"]
    fracs = [(5, 6), (4, 5), (3, 4), (2, 3)]   # 本文 6.1 の表
    values = [a / b for a, b in fracs]
    geo = float(np.prod(values)) ** 0.25       # (1/3)^{1/4} ≈ 0.760

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    x = np.arange(len(labels))
    ax.bar(x, values, width=0.56, color=BLUE, zorder=3)
    for xi, (a, b), v in zip(x, fracs, values):
        ax.annotate(f"{a}/{b}", (xi, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    ax.axhline(geo, color=INK_2, lw=0.9, ls=":")
    ax.annotate("geometric mean (dotted)\n$(1/3)^{1/4} \\approx 0.760$",
                (3.42, 0.93), ha="right", va="top", fontsize=8, color=INK_2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("modified n-gram precision")
    ax.set_title('ref: "the cat sat on the mat"  /  cand: "... on the rug"',
                 loc="left", color=INK_2)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    save(fig, "ch06-bleu-ngram.png")


if __name__ == "__main__":
    fig_ch01_param_breakdown()
    fig_ch04_lr_schedule()
    fig_ch04_warmup_loss()
    fig_ch05_loss_curve()
    fig_ch05_cross_attention()
    fig_ch06_bleu_ngram()
    print("ok: all figures generated")
