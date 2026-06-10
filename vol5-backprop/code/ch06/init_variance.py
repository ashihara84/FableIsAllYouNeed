# 第5巻 第6章 6.6: 重みの初期化 — 内積の分散の計算(第4巻7章)が再登場する
import numpy as np

rng = np.random.default_rng(42)

n, d = 1000, 256

# --- (1) 1層通すと標準偏差は √d・σ_w 倍になる(第4巻7章と同じ計算)---
# y_i = Σ_j x_j W_ji は d 項の内積。成分が独立・平均0なら Var(y_i) = d・σ_w²・Var(x)
X = rng.normal(0, 1, size=(n, d))     # 入力の標準偏差は 1
for sigma_w in [0.01, 0.0625, 0.1]:
    y_out = X @ rng.normal(0, sigma_w, size=(d, d))
    predicted = np.sqrt(d) * sigma_w
    print("σ_w = {:6.4f}: 実測 std = {:.4f}  理論 √d・σ_w = {:.4f}".format(
        sigma_w, y_out.std(), predicted))
    assert np.allclose(y_out.std(), predicted, rtol=0.05)

# --- (2) 10層重ねると √d・σ_w の10乗が効く: 消滅 / 維持 / 爆発 ---
depth = 10
sigmas = {
    "小さすぎ (0.01)": 0.01,                      # √d・σ_w = 0.16
    "Xavier (1/√d)": 1.0 / np.sqrt(d),            # √d・σ_w = 1
    "大きすぎ (4/√d)": 4.0 / np.sqrt(d),          # √d・σ_w = 4
}
print()
print("線形10層を通したあとの活性の標準偏差:")
finals = {}
for name, sigma_w in sigmas.items():
    h = X
    for _ in range(depth):
        h = h @ rng.normal(0, sigma_w, size=(d, d))
    finals[name] = h.std()
    print("  {:16s}: {:.3e}".format(name, finals[name]))

assert finals["小さすぎ (0.01)"] < 1e-6          # 消滅
assert 0.5 < finals["Xavier (1/√d)"] < 2.0       # 維持
assert finals["大きすぎ (4/√d)"] > 1e4           # 爆発

# --- (3) tanh を挟むと、大きすぎる初期値は爆発の代わりに「飽和」する ---
print()
print("tanh 10層を通したあとの活性:")
for name, sigma_w in [("Xavier (1/√d)", 1.0 / np.sqrt(d)),
                      ("大きすぎ (8/√d)", 8.0 / np.sqrt(d))]:
    h = X
    for _ in range(depth):
        h = np.tanh(h @ rng.normal(0, sigma_w, size=(d, d)))
    print("  {:16s}: std = {:.3f}  平均|h| = {:.3f}".format(name, h.std(), np.abs(h).mean()))
    if "大きすぎ" in name:
        assert np.abs(h).mean() > 0.9    # ほぼ全員が ±1 に張り付く(勾配は死ぬ)
    else:
        assert np.abs(h).mean() < 0.5    # 適正帯域に収まる

print("すべての assert を通過しました")
