# 第1巻 第4章 4.5: 3重ループで行列積を手実装 → `@` と一致確認 → ベンチマーク
import time

import numpy as np


def matmul_by_hand(A, B):
    """3重ループによる行列積。C[i, j] = (Aの行i) と (Bの列j) の内積。"""
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, "内側のshapeが一致しません: ({}, {}) @ ({}, {})".format(m, n, n2, p)
    C = np.zeros((m, p))
    for i in range(m):          # 結果の行を選ぶ
        for j in range(p):      # 結果の列を選ぶ
            s = 0.0
            for k in range(n):  # 内積: 掛けて、足す
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C


rng = np.random.default_rng(42)

# --- 一致確認: 小さな行列で `@` と比べる ---
A = rng.normal(size=(4, 3))
B = rng.normal(size=(3, 5))
C_hand = matmul_by_hand(A, B)
C_numpy = A @ B
assert C_hand.shape == (4, 5)
assert np.allclose(C_hand, C_numpy)
print("一致確認 OK: (4, 3) @ (3, 5) -> (4, 5)")

# --- ベンチマーク: (200, 200) @ (200, 200) ---
N = 200
A = rng.normal(size=(N, N))
B = rng.normal(size=(N, N))

_ = A @ B  # ウォームアップ(初回呼び出しの準備コストを計測から外す)

t0 = time.perf_counter()
C_hand = matmul_by_hand(A, B)
t1 = time.perf_counter()
sec_hand = t1 - t0

t0 = time.perf_counter()
C_numpy = A @ B
t1 = time.perf_counter()
sec_numpy = t1 - t0

assert np.allclose(C_hand, C_numpy)
print("3重ループ: {:.3f} 秒".format(sec_hand))
print("@ 演算子 : {:.6f} 秒".format(sec_numpy))
print("速度比   : 約 {:,} 倍".format(int(sec_hand / sec_numpy)))
