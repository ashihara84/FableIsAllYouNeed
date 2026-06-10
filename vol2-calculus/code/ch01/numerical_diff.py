# 第2巻 第1章 1.4: 数値微分(中心差分)を実装し、手計算の微分(解析解)と突き合わせる
import numpy as np


def numerical_diff(f, x, h=1e-5):
    """中心差分による数値微分: (f(x+h) - f(x-h)) / 2h"""
    return (f(x + h) - f(x - h)) / (2 * h)


def f(x):
    return x ** 3 - 2 * x


def f_prime(x):
    return 3 * x ** 2 - 2  # 1.3 のルールで手計算した微分


# --- 突き合わせ: 数値微分 ≈ 解析解 になっているか ---
for x in [-1.0, 0.0, 0.5, 2.0]:
    approx = numerical_diff(f, x)
    exact = f_prime(x)
    print(f"x = {x:5.2f}   数値微分 = {approx:.10f}   解析解 = {exact:.10f}")
    assert np.allclose(approx, exact)

print("ok: 4点すべてで数値微分と解析解が一致しました")


# --- h を変えて誤差を観察: 前進差分 vs 中心差分 ---
def forward_diff(f, x, h):
    """前進差分: (f(x+h) - f(x)) / h"""
    return (f(x + h) - f(x)) / h


x = 2.0
exact = f_prime(x)  # 10.0

hs = [1e-1, 1e-2, 1e-3, 1e-5, 1e-8, 1e-11, 1e-13]
errors_central = {}

print()
print("      h     前進差分の誤差   中心差分の誤差")
for h in hs:
    err_fwd = abs(forward_diff(f, x, h) - exact)
    err_ctr = abs(numerical_diff(f, x, h) - exact)
    errors_central[h] = err_ctr
    print(f"  {h:.0e}      {err_fwd:.2e}       {err_ctr:.2e}")

# h = 1e-5 のとき中心差分はほぼ厳密(誤差 1e-8 未満)
assert errors_central[1e-5] < 1e-8
# h を小さくしすぎると、丸め誤差でかえって悪化する
assert errors_central[1e-13] > errors_central[1e-5]

print()
print("ok: すべての assert を通過しました")
