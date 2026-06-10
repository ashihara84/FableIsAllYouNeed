# 第2巻 第4章 4.1〜4.3: 偏微分の数値検算 → 「最も急な方向」を総当たりで探す実験
import numpy as np


def f(x, y):
    """この章の実験台: x方向に間延びしたお椀"""
    return x ** 2 + 3 * y ** 2


def grad_f(x, y):
    """手計算した勾配: ∇f = (2x, 6y)"""
    return np.array([2 * x, 6 * y])


# --- 4.1 偏微分を数値微分で検算する(第1章の習慣の2変数版) ---
h = 1e-5
x0, y0 = 3.0, 1.0

df_dx = (f(x0 + h, y0) - f(x0 - h, y0)) / (2 * h)  # y を止めて x だけ動かす
df_dy = (f(x0, y0 + h) - f(x0, y0 - h)) / (2 * h)  # x を止めて y だけ動かす

assert np.allclose(df_dx, 2 * x0)  # 手計算: ∂f/∂x = 2x
assert np.allclose(df_dy, 6 * y0)  # 手計算: ∂f/∂y = 6y
assert np.allclose(grad_f(x0, y0), np.array([6.0, 6.0]))

# --- 4.3 いろんな方向に微小に動いて、f の増分を比べる ---
step = 1e-4
results = []
for deg in range(0, 360, 15):
    theta = np.deg2rad(deg)
    d = np.array([np.cos(theta), np.sin(theta)])  # 長さ1の方向ベクトル
    delta = f(x0 + step * d[0], y0 + step * d[1]) - f(x0, y0)
    results.append((deg, delta))
    print(f"{deg:3d}度: Δf = {delta:+.3e}")

best_deg, best_delta = max(results, key=lambda t: t[1])
worst_deg, worst_delta = min(results, key=lambda t: t[1])
print("増分が最大の方向:", best_deg, "度 / 最小の方向:", worst_deg, "度")

# 勾配 (6, 6) の向きは45度。最大は45度、最小はその正反対の225度のはず
assert best_deg == 45
assert worst_deg == 225

# 増分は内積で予言できる: Δf ≈ step × (∇f・d)
g = grad_f(x0, y0)
for deg, delta in results:
    theta = np.deg2rad(deg)
    d = np.array([np.cos(theta), np.sin(theta)])
    assert np.allclose(delta, step * np.dot(g, d), atol=1e-6)

print("ok: すべての assert を通過しました")
