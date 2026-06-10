# 第1巻 第2章 2.5: forループで内積を手実装 → np.dot と一致確認 → 速度比較
import time

import numpy as np


def dot_loop(a, b):
    """forループによる内積。定義「掛けて、足す」をそのまま書いたもの。"""
    assert a.shape == b.shape  # 次元が違うベクトル同士の内積は定義されない
    total = 0.0
    for i in range(a.shape[0]):
        total += a[i] * b[i]
    return total


# --- 一致確認(小さなベクトル) ---
a = np.array([2.0, 1.0, 3.0])
b = np.array([1.0, 5.0, -2.0])

result_loop = dot_loop(a, b)
result_np = np.dot(a, b)

print("dot_loop :", result_loop)   # 1.0
print("np.dot   :", result_np)     # 1.0
assert np.allclose(result_loop, result_np)

# --- 一致確認(大きな乱数ベクトル) ---
rng = np.random.default_rng(42)
n = 1_000_000
x = rng.standard_normal(n)
y = rng.standard_normal(n)

assert np.allclose(dot_loop(x, y), np.dot(x, y))

# --- 速度比較 ---
t0 = time.perf_counter()
dot_loop(x, y)
t1 = time.perf_counter()
np.dot(x, y)
t2 = time.perf_counter()

time_loop = t1 - t0
time_np = t2 - t1
print(f"forループ: {time_loop * 1000:.1f} ms")
print(f"np.dot  : {time_np * 1000:.3f} ms")
print(f"速度比  : 約 {time_loop / time_np:.0f} 倍")

print("ok: すべての assert を通過しました")
