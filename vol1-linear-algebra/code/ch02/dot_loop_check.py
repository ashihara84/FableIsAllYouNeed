# 第1巻 第2章 2.5: forループで内積を手実装し、np.dot と一致確認する
import numpy as np


def dot_loop(a, b):
    """forループによる内積。定義「掛けて、足す」をそのまま書いたもの。"""
    assert a.shape == b.shape  # 次元が違うベクトル同士の内積は定義されない
    total = 0.0
    for i in range(a.shape[0]):
        total += a[i] * b[i]
    return total


a = np.array([2.0, 1.0, 3.0])
b = np.array([1.0, 5.0, -2.0])

print("dot_loop :", dot_loop(a, b))   # 1.0
print("np.dot   :", np.dot(a, b))     # 1.0
assert np.allclose(dot_loop(a, b), np.dot(a, b))
print("ok: 一致しました")
