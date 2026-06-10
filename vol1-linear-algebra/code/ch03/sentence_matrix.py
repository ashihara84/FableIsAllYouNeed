# 第1巻 第3章 3.5: 文章 = 単語ベクトルを行に積んだ行列
import numpy as np

# おもちゃの4次元「意味ベクトル」
# 次元の意味: [生き物らしさ, 大きさ, 速さ, 水との縁](値は説明用に手で決めたもの)
words = ["猫", "犬", "魚", "走る", "泳ぐ"]
cat  = np.array([0.9, 0.3, 0.7, 0.1])   # (4,)
dog  = np.array([0.8, 0.4, 0.7, 0.2])   # (4,)
fish = np.array([0.8, 0.2, 0.4, 0.9])   # (4,)
run  = np.array([0.1, 0.0, 0.9, 0.0])   # (4,)
swim = np.array([0.1, 0.0, 0.6, 0.9])   # (4,)

# --- ベクトルを行に積む: 単語の行列 X ---
X = np.array([cat, dog, fish, run, swim])   # (5, 4) = (単語数, 次元)
assert X.shape == (5, 4)

# 行を1本取り出すと、1単語のベクトルに戻る
assert X[2].shape == (4,)            # X[2]: (4,) — 3番めの単語「魚」
assert np.allclose(X[2], fish)

# --- 転置: 行の束 → 列の束 ---
Xt = X.T                             # (4, 5)
assert Xt.shape == (4, 5)
assert np.allclose(Xt.T, X)          # 2回転置すると元に戻る
assert np.allclose(Xt[0], X[:, 0])   # X^T の1行めは、X の1列め(全単語の第1成分)

# 注意: 1次元配列 (n,) に .T を付けても何も起きない
v = np.array([1.0, 2.0, 3.0])        # (3,)
assert v.T.shape == (3,)             # (3,) のまま。行と列の区別がそもそもない

# --- 行列 × ベクトル: 「各行との内積を縦に並べる」 ---
q = cat                              # (4,) 問い合わせ役: 「猫」に似た単語は?
scores = X @ q                       # (5, 4) @ (4,) -> (5,)
assert scores.shape == (5,)

# 定義どおりの手実装(行ごとに内積)と一致するか検算
scores_loop = np.array([np.dot(X[i], q) for i in range(X.shape[0])])  # (5,)
assert np.allclose(scores, scores_loop)

for w, s in zip(words, scores):
    print(f"{w}\t{s:.2f}")

print("ok: すべての assert を通過しました")
