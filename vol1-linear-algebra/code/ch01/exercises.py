# code/ch01/exercises.py — 第1章 演習の略解(問2・問3・問4)
import numpy as np

# --- 問2: 問1の手計算を NumPy で検算 ---
a = np.array([1.0, 2.0])
b = np.array([3.0, -1.0])
assert np.allclose(a + b, [4.0, 1.0])
assert np.allclose(2 * a, [2.0, 4.0])
assert np.allclose(a - 2 * b, [-5.0, 4.0])

# --- 問3: list と ndarray の振る舞いの違い ---
assert [1, 2, 3] + [4, 5, 6] == [1, 2, 3, 4, 5, 6]                       # 連結
assert np.array_equal(np.array([1, 2, 3]) + np.array([4, 5, 6]), [5, 7, 9])  # 足し算
assert [1, 2] * 3 == [1, 2, 1, 2, 1, 2]                                  # 繰り返し
assert np.array_equal(np.array([1, 2]) * 3, [3, 6])                      # スカラー倍

# --- 問4: 意味の算術(1.5 の採点表) ---
# 成分の順: (王族度, 男性度, 女性度, 人間度)
king  = np.array([0.9, 0.9, 0.1, 1.0])
man   = np.array([0.1, 0.9, 0.1, 1.0])
woman = np.array([0.1, 0.1, 0.9, 1.0])
queen = np.array([0.9, 0.1, 0.9, 1.0])
assert np.allclose(king - man + woman, queen)
# 人間度は king − man の時点で差分 0(王と男に「人間であること」の違いはない)

print("第1章 演習: すべての assert を通過しました")
