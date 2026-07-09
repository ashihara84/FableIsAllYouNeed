# 第6巻 図版一括生成スクリプト
# 実行: python3 generate_figures.py  (このディレクトリで実行し、PNG を同じ場所に出力)
# 方針:
#   - 図中の文字は英語と数式記号のみ(日本語フォントに依存しない)
#   - 数値は本文の表・実行ログと一致させる(ここにハードコード)
#   - dpi=160, facecolor="white", bbox_inches="tight"
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)  # 乱数を使う図はないが、規約として固定

# ---- 共通スタイル(dataviz 規約: 単一系列は青1色、罫線・軸は控えめなグレー) ----
BLUE = "#2a78d6"      # series-1
INK = "#0b0b0b"       # primary ink
MUTED = "#898781"     # axis / labels
GRID = "#e1e0d9"      # hairline grid
BASE = "#c3c2b7"      # axis line
RED = "#e34948"       # 強調点(ch03 の合成ベクトルの着地点)

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def save(fig, name):
    fig.savefig(name)
    plt.close(fig)
    print("saved:", name)


# ============================================================
# 図3.1  埋め込み空間の「意味の算術」(2次元の概念図)
#   king - man + woman ≈ queen の平行四辺形
# ============================================================
def fig_ch03_analogy():
    pts = {
        "man": (0.15, 0.10),
        "woman": (0.15, 0.90),
        "king": (0.85, 0.15),
        "queen": (0.88, 0.92),
    }
    target = (0.85, 0.95)  # king + (woman - man) の着地点(queen のすぐそば)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.grid(False)

    # 「男 → 女」の差ベクトル2本(ほぼ平行)
    for a, b in [("man", "woman"), ("king", "queen")]:
        (x0, y0), (x1, y1) = pts[a], pts[b]
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2,
                            shrinkA=8, shrinkB=8),
        )
    ax.text(0.09, 0.5, "woman $-$ man", color=BLUE, rotation=90,
            ha="center", va="center", fontsize=10)
    ax.text(0.945, 0.53, "queen $-$ king", color=BLUE, rotation=87,
            ha="center", va="center", fontsize=10)

    # 合成 king - man + woman の経路(破線)と着地点(×)
    ax.annotate(
        "", xy=target, xytext=pts["king"],
        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5,
                        linestyle="--", shrinkA=8, shrinkB=6),
    )
    ax.plot(*target, marker="x", color=RED, markersize=11, markeredgewidth=2.5,
            zorder=5)
    ax.text(target[0] - 0.03, target[1] + 0.055,
            "king $-$ man $+$ woman", color=RED, fontsize=10,
            ha="right", va="center")

    # 4単語の点とラベル
    for w, (x, y) in pts.items():
        ax.plot(x, y, "o", color=INK, markersize=7, zorder=4)
        dy = -0.06 if w in ("man", "king") else 0.0
        dx = 0.035
        ax.text(x + dx, y + dy, w, fontsize=12, va="center", color=INK)

    ax.set_xlim(0, 1.12)
    ax.set_ylim(-0.02, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("royalty direction", color=MUTED)
    ax.set_ylabel("gender direction (male $\\to$ female)", color=MUTED)
    save(fig, "ch03-analogy-parallelogram.png")


# ============================================================
# 図5.3  痛み1: 総仕事量を固定しても訓練時間は L に比例(表5.2)
# ============================================================
def fig_ch05_seqlen_time():
    L = [8, 16, 32, 64, 128]           # B×L = 512 を固定
    ms = [1.03, 1.53, 2.80, 5.32, 9.10]
    matmul_ms = 0.46                   # 同じ512トークンを1発の行列積で

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(L, ms, marker="o", color=BLUE, lw=2, markersize=7, zorder=3,
            label="RNN, one training step")
    ax.axhline(matmul_ms, color=MUTED, lw=1.5, ls="--", zorder=2,
               label="single matmul, same 512 tokens (0.46 ms)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(L)
    ax.set_xticklabels([str(x) for x in L])
    ax.minorticks_off()
    ax.set_xlabel("sequence length $L$  (total work fixed: $B \\times L = 512$)")
    ax.set_ylabel("time per training step [ms]")
    ax.set_ylim(0, 10)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.annotate("$\\times 8.8$", xy=(128, 9.10), xytext=(80, 8.6),
                color=INK, fontsize=10, ha="right", va="center")
    save(fig, "ch05-seqlen-time.png")


# ============================================================
# 図5.4  痛み2: 勾配は距離とともに指数的に減衰(表5.3、縦軸対数)
# ============================================================
def fig_ch05_gradient_decay():
    d = [1, 2, 4, 8, 16, 32, 64]
    norm = [1.98e-2, 1.99e-2, 1.64e-2, 9.89e-3, 4.91e-3, 1.08e-3, 9.93e-5]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(d, norm, marker="o", color=BLUE, lw=2, markersize=7, zorder=3)
    ax.set_xlabel("distance $d$ from the loss (tokens)")
    ax.set_ylabel("$\\|\\partial\\,\\mathrm{loss}\\,/\\,\\partial\\,\\mathbf{x}\\|$")
    ax.set_xticks([1, 8, 16, 32, 64])  # 1,2,4 はラベルが重なるので間引く
    ax.annotate("0.5% of $d=1$", xy=(64, 9.93e-5), xytext=(44, 3.2e-4),
                color=INK, fontsize=10, ha="right", va="bottom",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    save(fig, "ch05-gradient-decay.png")


# ============================================================
# 図6.1  痛み3: 入力長 vs 系列一致率(6.3 の表と同じ値)
# ============================================================
def fig_ch06_length_accuracy():
    lengths = [2, 4, 6, 8, 10, 12]
    acc = [1.00, 1.00, 0.81, 0.26, 0.04, 0.00]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lengths, acc, marker="o", color=BLUE, lw=2, markersize=7, zorder=3)
    for x, y in zip(lengths, acc):
        ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 9),
                    textcoords="offset points", ha="center",
                    fontsize=9, color=INK)
    ax.set_xlabel("input length")
    ax.set_ylabel("sequence accuracy")
    ax.set_xticks(lengths)
    ax.set_ylim(-0.05, 1.12)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    save(fig, "ch06-length-accuracy.png")


# ============================================================
# 図7.1  attention 重みのヒートマップ(本文の ASCII 行列と同一の値)
#   入力 deadbeefcafe → 出力 efacfeebdaed、逆対角パターン
# ============================================================
def fig_ch07_attention_heatmap():
    src = list("deadbeefcafe")   # 入力位置(key)
    out = list("efacfeebdaed")   # 出力ステップ(query)
    A = np.array([
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.98],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.00, 0.00, 0.01, 0.94, 0.01],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.98, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.99, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.99, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.03, 0.00, 0.01, 0.01, 0.92, 0.01, 0.00, 0.01, 0.00, 0.00],
        [0.01, 0.01, 0.00, 0.01, 0.04, 0.87, 0.04, 0.00, 0.01, 0.00, 0.00, 0.00],
        [0.01, 0.00, 0.00, 0.02, 0.96, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.02, 0.90, 0.06, 0.00, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.02, 0.93, 0.02, 0.00, 0.01, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.01, 0.96, 0.01, 0.00, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.58, 0.08, 0.00, 0.05, 0.05, 0.01, 0.00, 0.00, 0.09, 0.00, 0.03, 0.10],
    ])
    assert A.shape == (len(out), len(src))
    assert np.all(A.sum(axis=1) > 0.97)  # 各行は softmax の出力(丸めで ±0.03)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.grid(False)
    im = ax.imshow(A, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(src)))
    ax.set_xticklabels(src)
    ax.set_yticks(range(len(out)))
    ax.set_yticklabels(out)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("input position $i$ (key)", fontsize=10)
    ax.set_ylabel("output step $t$ (query)", fontsize=10)
    ax.tick_params(length=0)

    # マスの境界にごく薄い白罫線(セルの区切りを見やすく)
    ax.set_xticks(np.arange(-0.5, len(src)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(out)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("attention weight $a_{t,i}$", fontsize=10)
    cbar.outline.set_edgecolor(BASE)
    save(fig, "ch07-attention-heatmap.png")


if __name__ == "__main__":
    fig_ch03_analogy()
    fig_ch05_seqlen_time()
    fig_ch05_gradient_decay()
    fig_ch06_length_accuracy()
    fig_ch07_attention_heatmap()
    print("all figures generated.")
