# 第1巻 第3章 3.2: shape注釈の見本 — すべての配列に shape をコメントで添える
import numpy as np

X = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])   # (2, 3)
v = np.array([7.0, 8.0, 9.0])     # (3,)

assert X.shape == (2, 3)
assert v.shape == (3,)

print("ok: すべての assert を通過しました")
