# 第2巻 第4章 4.4: 2変数の勾配降下 — 軌跡を記録し、手計算の予言と照合する
import numpy as np


def f(x, y):
    return x ** 2 + 3 * y ** 2


def grad_f(x, y):
    return np.array([2 * x, 6 * y])


def gradient_descent_2d(start, lr, n_steps):
    """点 (x, y) を勾配の逆向きに動かし続け、通った点をすべて記録する"""
    p = np.array(start, dtype=float)  # 現在地 (2,)
    trajectory = [p.copy()]
    for _ in range(n_steps):
        p = p - lr * grad_f(p[0], p[1])  # 更新はこの1行。第3章と同じ形
        trajectory.append(p.copy())
    return np.array(trajectory)  # (n_steps + 1, 2)


start = (3.0, 1.5)
lr = 0.1
n_steps = 30
traj = gradient_descent_2d(start, lr, n_steps)

print("出発点 :", traj[0])
print(" 3歩目 :", traj[3])
print("10歩目 :", traj[10])
print("30歩目 :", traj[30])

# --- 検証1: f の値は1歩ごとに必ず減っている ---
values = np.array([f(px, py) for px, py in traj])
assert np.all(np.diff(values) < 0)

# --- 検証2: 軌跡は手計算の予言と一致する(x は1歩ごとに0.8倍、y は0.4倍) ---
k = np.arange(n_steps + 1)
assert np.allclose(traj[:, 0], 3.0 * 0.8 ** k)
assert np.allclose(traj[:, 1], 1.5 * 0.4 ** k)

# --- 検証3: 最小値 (0, 0) のすぐそばまで来ていて、そこでの勾配はほぼ零ベクトル ---
assert np.allclose(traj[-1], np.zeros(2), atol=1e-2)
assert np.linalg.norm(grad_f(traj[-1][0], traj[-1][1])) < 0.1

print("ok: すべての assert を通過しました")
