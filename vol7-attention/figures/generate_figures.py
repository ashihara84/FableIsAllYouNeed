# 第7巻 図版の一括生成スクリプト
#   python3 generate_figures.py で全 PNG をこのディレクトリに生成する。
#   図中の文字は英語と数式記号のみ(日本語フォント非依存)。
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code" / "ch03"))
sys.path.insert(0, str(HERE.parent / "code" / "ch07"))
from attention import softmax  # noqa: E402  (第3章の数値安定版 softmax)
from positional_encoding import positional_encoding  # noqa: E402

rng = np.random.default_rng(42)

SAVE_KW = dict(dpi=160, facecolor="white", bbox_inches="tight")


def save(fig, name):
    path = HERE / name
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    print(f"wrote {path.name}")


# ---------------------------------------------------------------------------
# 図3.1: softmax の重み分布 — √d_k で割る前と後(第3章 3.3)
# ---------------------------------------------------------------------------
def fig_ch03_softmax_scaling():
    d_k, m = 512, 10
    q = rng.standard_normal(d_k)
    K = rng.standard_normal((m, d_k))
    # macOS Accelerate は大きな行列積で偽の警告を出すことがある(結果は正常。
    # code/ch08/attention_scaling.py と同じ対処)。有限性は assert で確認する
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = K @ q                   # (m,) 各成分の標準偏差は ≈ √d_k
    assert np.isfinite(scores).all()
    w_raw = softmax(scores, axis=-1)
    w_scaled = softmax(scores / np.sqrt(d_k), axis=-1)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), sharey=True)
    for ax, w, title in [
        (axes[0], w_raw, r"softmax$(QK^T)$  —  no scaling"),
        (axes[1], w_scaled, r"softmax$(QK^T/\sqrt{d_k})$  —  scaled"),
    ]:
        ax.bar(np.arange(m), w, color="#4878cf", edgecolor="none")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("key position $j$")
        ax.set_xticks(np.arange(m))
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("attention weight")
    axes[0].annotate("nearly one-hot", xy=(int(np.argmax(w_raw)), 1.0),
                     xytext=(5.1, 0.85), fontsize=9,
                     arrowprops=dict(arrowstyle="->", lw=0.8))
    axes[1].annotate("weights spread out", xy=(int(np.argmax(w_scaled)),
                                               float(w_scaled.max())),
                     xytext=(3.6, 0.6), fontsize=9,
                     arrowprops=dict(arrowstyle="->", lw=0.8))
    fig.suptitle(r"$d_k = 512$: one query row, $m = 10$ keys", fontsize=10, y=1.02)
    save(fig, "ch03-softmax-scaling.png")


# ---------------------------------------------------------------------------
# 図3.2: padding mask と causal mask のパターン(第3章 3.5)
# ---------------------------------------------------------------------------
def fig_ch03_masks():
    n = 8
    m_real = 6
    pad = np.broadcast_to(np.arange(n) < m_real, (n, n))
    causal = np.tril(np.ones((n, n), dtype=bool))

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
    cmap = matplotlib.colors.ListedColormap(["#3b3b3b", "#dce6f5"])
    for ax, mask, title in [
        (axes[0], pad, f"padding mask  (last {n - m_real} keys = padding)"),
        (axes[1], causal, "causal mask  (upper triangle = future)"),
    ]:
        ax.imshow(mask, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("key position $j$")
        ax.set_ylabel("query position $i$")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticks(np.arange(-0.5, n), minor=True)
        ax.set_yticks(np.arange(-0.5, n), minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="minor", length=0)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, "T" if mask[i, j] else "F",
                        ha="center", va="center", fontsize=7,
                        color="#1f3a63" if mask[i, j] else "#cccccc")
    fig.suptitle("mask convention: True (light) = may attend,  "
                 "False (dark) = score set to $-\\infty$", fontsize=10, y=1.0)
    fig.tight_layout()
    save(fig, "ch03-masks.png")


# ---------------------------------------------------------------------------
# 図7.1: ペア i=0 の (cos, sin) — 単位円上の等間隔な回転(第7章 7.3)
# ---------------------------------------------------------------------------
def fig_ch07_rotation_circle():
    positions = np.arange(7)            # omega_0 = 1 → 1 rad per token
    x, y = np.cos(positions), np.sin(positions)

    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="#bbbbbb", lw=1)
    ax.scatter(x, y, s=45, color="#4878cf", zorder=3)
    for p in positions:
        ax.annotate(f"pos {p}", (x[p], y[p]),
                    xytext=(x[p] * 1.16 - 0.06, y[p] * 1.16 - 0.02), fontsize=9)

    # 「k = 2 進める」= どこから出発しても同じ回転(0→2 と 3→5 で同じ弧)
    for start, color in [(0, "#d1495b"), (3, "#d1495b")]:
        ax.annotate("",
                    xy=(x[start + 2], y[start + 2]), xytext=(x[start], y[start]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.6,
                                    connectionstyle="arc3,rad=0.35"))
    ax.text(0.12, 1.28, "advance by $k=2$\n= same rotation", fontsize=9,
            color="#d1495b", ha="center")

    ax.set_xlabel(r"$\cos(pos\,\omega_0)$")
    ax.set_ylabel(r"$\sin(pos\,\omega_0)$")
    ax.set_title(r"PE pair $i=0$  ($\omega_0 = 1$): 1 rad per position", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.45, 1.65)
    ax.grid(alpha=0.25)
    save(fig, "ch07-pe-rotation-circle.png")


# ---------------------------------------------------------------------------
# 図7.2: positional encoding 行列のヒートマップ(第7章 7.4)
# ---------------------------------------------------------------------------
def fig_ch07_pe_heatmap():
    max_len, d_model = 100, 128
    pe = positional_encoding(max_len, d_model)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    im = ax.imshow(pe, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("dimension")
    ax.set_ylabel(r"position $pos$")
    ax.set_title(f"positional_encoding({max_len}, {d_model})", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("PE value")
    save(fig, "ch07-pe-heatmap.png")


# ---------------------------------------------------------------------------
# 図8.1: self-attention 実行時間の n^2 スケーリング(第8章 8.2)
#   本文掲載の実測値(code/ch08/attention_scaling.py の筆者環境での出力)を
#   そのままプロットする。本文の数値表と矛盾させないため再計測はしない。
# ---------------------------------------------------------------------------
def fig_ch08_timing():
    ns = np.array([256, 512, 1024, 2048, 4096])
    ms = np.array([0.21, 0.76, 3.30, 14.32, 54.39])   # 本文 8.2 の実測表と同じ

    fig, ax = plt.subplots(figsize=(6, 4))
    # 参照線は最初の実測点に揃える
    ref2 = ms[0] * (ns / ns[0]) ** 2
    ref1 = ms[0] * (ns / ns[0])
    ax.loglog(ns, ref2, "--", color="#999999", lw=1.2,
              label=r"slope 2  ($\propto n^2$)")
    ax.loglog(ns, ref1, ":", color="#999999", lw=1.2,
              label=r"slope 1  ($\propto n$)")
    ax.loglog(ns, ms, "o-", color="#4878cf", lw=1.8, ms=6,
              label="measured (min of 5 runs)")
    ax.set_xlabel("sequence length $n$")
    ax.set_ylabel("time (ms)")
    ax.set_title(r"attention$(Q, K, V)$,  $d_k = 64$ fixed", fontsize=10)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.annotate("log-log slope $\\approx$ 2.02", xy=(ns[3], ms[3]),
                xytext=(300, 20), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9, loc="lower right")
    save(fig, "ch08-attention-timing.png")


if __name__ == "__main__":
    fig_ch03_softmax_scaling()
    fig_ch03_masks()
    fig_ch07_rotation_circle()
    fig_ch07_pe_heatmap()
    fig_ch08_timing()
    print("done")
