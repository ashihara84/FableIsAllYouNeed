# 第1巻 第4章 4.6: 単語ベクトルの行列 X に対して X @ X.T を計算してみる
import numpy as np

# 第3章と同じく、単語ベクトルを「行に積んだ」行列をつくる
# 成分の意味(この章のための手作りの特徴): [動物らしさ, 乗り物らしさ, 大きさ]
words = ["猫", "犬", "魚", "車", "バス"]
X = np.array([
    [1.0, 0.0, 0.2],   # 猫
    [1.0, 0.0, 0.3],   # 犬
    [0.8, 0.0, 0.1],   # 魚
    [0.0, 1.0, 0.7],   # 車
    [0.0, 1.0, 0.9],   # バス
])
assert X.shape == (5, 3)  # (単語数, 次元)

# ラスボスに先に触る: 行列とその転置の積
S = X @ X.T
assert S.shape == (5, 5)  # (単語数, 単語数) の正方行列

# 各マスの正体: S[i, j] = (単語iのベクトル) と (単語jのベクトル) の内積
for i in range(5):
    for j in range(5):
        assert np.isclose(S[i, j], np.dot(X[i], X[j]))

# 内積は順序を入れ替えても同じなので、この表は対称になる
assert np.allclose(S, S.T)

# 表として眺める
print("      " + "".join("{:>6}".format(w) for w in words))
for i, w in enumerate(words):
    print("{:>6}".format(w) + "".join("{:6.2f}".format(v) for v in S[i]))

# 「猫」に最も似ている単語(自分自身を除く)は「犬」のはず
i_cat = words.index("猫")
scores = S[i_cat].copy()
scores[i_cat] = -np.inf  # 自分自身は除外
assert words[int(np.argmax(scores))] == "犬"
print("「猫」に最も似ている単語:", words[int(np.argmax(scores))])
