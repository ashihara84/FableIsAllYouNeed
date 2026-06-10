# code/ch01/numpy_intro.py
import numpy as np

# --- list は「数学のベクトル」ではない ---
x_list = [1.0, 2.0, 3.0]
y_list = [10.0, 20.0, 30.0]

assert x_list + y_list == [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]  # 連結される
assert x_list * 2 == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]          # 繰り返される

# --- ndarray は成分ごとに演算する ---
x = np.array([1.0, 2.0, 3.0])
y = np.array([10.0, 20.0, 30.0])

assert np.allclose(x + y, [11.0, 22.0, 33.0])    # 足し算(1.3)
assert np.allclose(2 * x, [2.0, 4.0, 6.0])       # スカラー倍(1.3)
assert np.allclose(-1 * x, [-1.0, -2.0, -3.0])   # 反転
assert np.allclose(x - y, x + (-1) * y)          # 引き算 = (-1)倍の足し算

# 足す順番は入れ替えてよい(第4章の行列積では成り立たない)
assert np.allclose(x + y, y + x)

# --- shape: 配列の形 ---
assert x.shape == (3,)            # 3次元ベクトル

v = np.zeros(512)                 # 全成分 0 の 512次元ベクトル
assert v.shape == (512,)          # 論文の単語ベクトルと同じ次元

# --- 1.3 の手計算例の検算 ---
a = np.array([2.0, 1.0])
b = np.array([1.0, 3.0])
assert np.allclose(a + b, [3.0, 4.0])
assert np.allclose(3 * a, [6.0, 3.0])
assert np.allclose((a + b) - b, a)   # b を足して b を引けば元に戻る

# --- 要素ごとの積(内積ではない。内積は第2章) ---
assert np.allclose(x * y, [10.0, 40.0, 90.0])

print("第1章: すべての assert を通過しました")
