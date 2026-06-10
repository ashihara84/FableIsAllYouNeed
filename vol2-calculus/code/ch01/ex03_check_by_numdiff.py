# 第2巻 第1章 演習3 略解: 手で微分した結果を数値微分(中心差分)で検算する
import numpy as np


def numerical_diff(f, x, h=1e-5):
    """中心差分による数値微分: (f(x+h) - f(x-h)) / 2h"""
    return (f(x + h) - f(x - h)) / (2 * h)


# (手で求めた関数, その導関数) のペア。演習1の3問
cases = [
    ("(a) x^4 - 3x^2 + 5", lambda x: x ** 4 - 3 * x ** 2 + 5,
     lambda x: 4 * x ** 3 - 6 * x),
    ("(b) (x^2 + 1)(x - 2)", lambda x: (x ** 2 + 1) * (x - 2),
     lambda x: 3 * x ** 2 - 4 * x + 1),
    ("(c) (3w + 1)^2", lambda w: (3 * w + 1) ** 2,
     lambda w: 18 * w + 6),
]

for name, f, f_prime in cases:
    for x in [-2.0, -0.5, 0.0, 1.0, 3.0]:
        assert np.allclose(numerical_diff(f, x), f_prime(x))
    print(f"{name}: 5点すべてで一致")

print("ok: 手の微分はすべて正しい")
