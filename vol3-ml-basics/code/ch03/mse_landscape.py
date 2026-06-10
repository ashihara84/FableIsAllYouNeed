# 第3巻 第3章 3.4: (w, b) 平面上の MSE の地形 — 椀型であることを数値で確かめる
# 等高線図そのものは matplotlib で描く(本文に掲載)。ここでは図に見えるはずの性質を assert で検証する。
import numpy as np


def mse(y_pred, y):
    """平均二乗誤差: (N, 1), (N, 1) -> スカラー"""
    return np.mean((y_pred - y) ** 2)


# --- 第2章と同じ合成データ(再掲。20人ぶんの勉強時間と点数) ---
rng = np.random.default_rng(42)
n = 20
X = rng.uniform(0, 9, size=(n, 1))          # 勉強時間 (20, 1)
noise = rng.normal(0, 6.0, size=(n, 1))
y = 7.0 * X + 20.0 + noise                  # 点数 (20, 1)  真の規則: w=7, b=20
assert np.allclose(X[:3].ravel(), [6.96560444, 3.94990596, 7.72738128])  # 第2章と同じ20人

# --- 3.1 残差は打ち消し合う: 平均残差 0 でもひどい予測はありうる ---
y_pred_flat = np.full_like(y, y.mean())     # 全予測を y の平均で固定した「水平線」
residual = y_pred_flat - y                  # (20, 1)
assert np.isclose(residual.mean(), 0.0)     # 残差の平均は厳密に 0(符号が打ち消し合う)
assert mse(y_pred_flat, y) > 250.0          # それでも MSE は大きい(悪い予測は悪いと言える)

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

assert L(7.0, 20.0) < L(0.0, 50.0) < L(0.0, 0.0)  # 真の規則に近いほど損失は小さい

# --- 3.4 (w, b) 平面に MSE の地形を計算する ---
ws = np.linspace(3.0, 11.0, 81)             # (81,) 刻み幅 0.1
bs = np.linspace(0.0, 40.0, 81)             # (81,) 刻み幅 0.5
landscape = np.empty((len(bs), len(ws)))    # (81, 81)  行が b、列が w
for i, b in enumerate(bs):
    for j, w in enumerate(ws):
        landscape[i, j] = L(w, b)

# 性質1: 地形のどこを見ても損失は正(ばらつきがあるので 0 にはならない)
assert landscape.min() > 0.0

# 性質2: 谷底は真の規則 (w, b) = (7.0, 20.0) のすぐそば
i_min, j_min = np.unravel_index(landscape.argmin(), landscape.shape)
w_best, b_best = ws[j_min], bs[i_min]
assert abs(w_best - 7.0) <= 0.5
assert abs(b_best - 20.0) <= 3.0

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

# --- 演習: 第2章の候補A〜Dと手動フィットを地形の上に置く ---
for w_c, b_c in [(10.0, 0.0), (4.0, 40.0), (7.0, 20.0), (6.5, 25.0)]:
    assert L(w_c, b_c) > landscape.min()    # どの候補も谷底ではない
assert L(7.0, 20.0) < L(6.5, 25.0)          # 前章の論争の決着: 候補Cの勝ち(約22.0 < 約24.5)
w_hand, b_hand = 6.7, 23.0                  # 第2章の演習3で筆者が記録した値に置き換えてよい
assert L(w_hand, b_hand) > landscape.min()  # 手動フィットも谷底ではない(どれだけ惜しいかは図で)

print("ok: すべての assert を通過しました")
print(f"  谷底(グリッド上): w = {w_best:.2f}, b = {b_best:.2f}, L = {landscape.min():.3f}")
print(f"  候補C (7.0, 20.0) : L = {L(7.0, 20.0):.3f}")
print(f"  候補D (6.5, 25.0) : L = {L(6.5, 25.0):.3f}")
print(f"  手動フィット      : w = {w_hand:.2f}, b = {b_hand:.2f}, L = {L(w_hand, b_hand):.3f}")
