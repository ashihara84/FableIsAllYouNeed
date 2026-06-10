# 第3巻 第3章 3.4: (w, b) 平面上の MSE の地形 — 椀型であることを数値で確かめる
# 等高線図そのものは matplotlib で描く(本文に掲載)。ここでは図に見えるはずの性質を assert で検証する。
import numpy as np


def mse(y_pred, y):
    """平均二乗誤差: (N, 1), (N, 1) -> スカラー"""
    return np.mean((y_pred - y) ** 2)


# --- 第2章と同じ合成データ(再掲) ---
rng = np.random.default_rng(42)
N = 50
X = rng.uniform(0.0, 10.0, size=(N, 1))     # (50, 1)
noise = rng.normal(0.0, 1.0, size=(N, 1))   # (50, 1)
y = 2.0 * X + 1.0 + noise                   # (50, 1)  正解の直線: w=2.0, b=1.0

# --- 3.1 残差は打ち消し合う: 平均残差 0 でもひどい予測はありうる ---
y_pred_flat = np.full_like(y, y.mean())     # 全予測を y の平均で固定した「水平線」
residual = y_pred_flat - y                  # (50, 1)
assert np.isclose(residual.mean(), 0.0)     # 残差の平均は厳密に 0(符号が打ち消し合う)
assert mse(y_pred_flat, y) > 25.0           # それでも MSE は大きい(悪い予測は悪いと言える)

# --- 3.2 二乗の性質: 完全な予測で 0、外れるほど急に増える ---
assert mse(y, y) == 0.0                     # ズレゼロなら損失ゼロ
r1 = np.array([[1.0], [1.0], [1.0], [1.0]])  # 全員が 1 ずつ外す
r2 = np.array([[0.0], [0.0], [0.0], [4.0]])  # 1人だけ 4 外す(絶対値の合計は同じ 4)
assert np.isclose(np.abs(r1).mean(), np.abs(r2).mean())  # 平均絶対誤差では同点
assert np.mean(r2**2) == 4 * np.mean(r1**2)              # MSE は大外れを 4 倍重く罰する

# --- 3.3 損失はパラメータの関数: L(w, b) を定義する ---
def L(w, b):
    """データ (X, y) を固定し、(w, b) を入力とする損失関数。スカラー -> スカラー"""
    return mse(w * X + b, y)

assert L(2.0, 1.0) < L(0.0, 0.0) < L(-2.0, 5.0)  # 正解の直線に近いほど損失は小さい

# --- 3.4 (w, b) 平面に MSE の地形を計算する ---
ws = np.linspace(0.0, 4.0, 81)              # (81,) 刻み幅 0.05
bs = np.linspace(-3.0, 5.0, 81)             # (81,) 刻み幅 0.1
landscape = np.empty((len(bs), len(ws)))    # (81, 81)  行が b、列が w
for i, b in enumerate(bs):
    for j, w in enumerate(ws):
        landscape[i, j] = L(w, b)

# 性質1: 地形のどこを見ても損失は正(ノイズがあるので 0 にはならない)
assert landscape.min() > 0.0

# 性質2: 谷底は正解の直線 (w, b) = (2.0, 1.0) のすぐそば
i_min, j_min = np.unravel_index(landscape.argmin(), landscape.shape)
w_best, b_best = ws[j_min], bs[i_min]
assert abs(w_best - 2.0) <= 0.1
assert abs(b_best - 1.0) <= 0.5

# 性質3: 椀型 — どの行(b 固定)も、どの列(w 固定)も「下って、上る」一回きり。
# 途中に別のくぼみ(局所的な谷)は一つもない。
def is_valley(values):
    """1次元配列が「単調に下って、単調に上る」形かを判定する"""
    k = values.argmin()
    return np.all(np.diff(values[: k + 1]) < 0) and np.all(np.diff(values[k:]) > 0)

for i in range(len(bs)):
    assert is_valley(landscape[i, :])       # 横方向に切っても谷は1つ
for j in range(len(ws)):
    assert is_valley(landscape[:, j])       # 縦方向に切っても谷は1つ

# --- 演習: 第2章の手動フィットを地形の上に置く ---
w_hand, b_hand = 1.8, 2.5                   # 第2章で記録した値に置き換えてよい
assert L(w_hand, b_hand) > landscape.min()  # 手動フィットは谷底ではない(どれだけ惜しいかは図で)

print("ok: すべての assert を通過しました")
print(f"  谷底(グリッド上): w = {w_best:.2f}, b = {b_best:.2f}, L = {landscape.min():.3f}")
print(f"  手動フィット      : w = {w_hand:.2f}, b = {b_hand:.2f}, L = {L(w_hand, b_hand):.3f}")
