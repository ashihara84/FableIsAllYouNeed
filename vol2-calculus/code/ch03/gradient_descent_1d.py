# 第2巻 第3章 3.3: 1変数の勾配降下を実装し、η を変えて収束・振動・発散を観察
import numpy as np


def f(x):
    """目標の関数。最小値は x = 0 で f(0) = 0。"""
    return x ** 2


def grad_f(x):
    """f の導関数 f'(x) = 2x(第1章のルールで手計算)。"""
    return 2.0 * x


def gradient_descent(grad, x0, lr, n_steps):
    """勾配降下法。x ← x − lr・grad(x) を n_steps 回繰り返し、軌跡を返す。"""
    xs = [x0]
    x = x0
    for _ in range(n_steps):
        x = x - lr * grad(x)  # ← アルゴリズムの本体はこの1行
        xs.append(x)
    return np.array(xs)


x0 = 10.0
n_steps = 50

# --- η を変えて 50 ステップ走らせ、最後の x を表にする ---
print("出発点 x0 = {}, f(x) = x^2, {} ステップ".format(x0, n_steps))
print("{:>8} {:>14}".format("lr", "x_50"))
for lr in [0.001, 0.01, 0.1, 0.45, 0.8, 1.0, 1.1]:
    xs = gradient_descent(grad_f, x0, lr, n_steps)
    print("{:>8} {:>14.6g}".format(lr, xs[-1]))

# --- 運命1: 収束(小さすぎる η は遅い) ---
xs_crawl = gradient_descent(grad_f, x0, 0.001, n_steps)
assert 9.0 < xs_crawl[-1] < 9.1  # 50ステップでほぼ動けていない

xs_slow = gradient_descent(grad_f, x0, 0.01, n_steps)
assert 3.6 < xs_slow[-1] < 3.7  # 向かってはいるが、まだ遠い

xs_conv = gradient_descent(grad_f, x0, 0.1, n_steps)
assert abs(xs_conv[-1]) < 1e-3                # 最小値 x = 0 に収束
assert np.all(np.diff(np.abs(xs_conv)) < 0)   # |x| は毎ステップ単調減少

# 3.2 の閉じた式 x_t = (1 − 2η)^t · x0 と完全一致することの検算
t = np.arange(n_steps + 1)
assert np.allclose(xs_conv, x0 * (1.0 - 2.0 * 0.1) ** t)

# --- 運命1': 収束はするが、谷を飛び越えながら(0.5 < η < 1) ---
xs_zigzag = gradient_descent(grad_f, x0, 0.8, n_steps)
assert abs(xs_zigzag[-1]) < 1e-3                       # 収束はする
assert np.all(xs_zigzag[:-1] * xs_zigzag[1:] < 0)      # ただし符号が毎回反転

# --- 運命2: 振動(η = 1.0)。±10 を永遠に往復し、近づきも遠ざかりもしない ---
xs_osc = gradient_descent(grad_f, x0, 1.0, n_steps)
assert np.allclose(np.abs(xs_osc), x0)                 # |x| = 10 のまま
assert np.all(xs_osc[:-1] * xs_osc[1:] < 0)            # 符号は毎回反転

# --- 運命3: 発散(η = 1.1)。毎ステップ谷底から遠ざかる ---
xs_div = gradient_descent(grad_f, x0, 1.1, n_steps)
assert abs(xs_div[-1]) > 1e4                           # 9万超まで爆発
assert np.all(np.diff(np.abs(xs_div)) > 0)             # |x| は毎ステップ単調増加

print("すべての assert を通過: 収束・振動・発散を数値で確認できました")
