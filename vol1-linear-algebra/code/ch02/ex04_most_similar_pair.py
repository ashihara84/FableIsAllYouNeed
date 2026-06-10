# 第1巻 第2章 演習4 略解: 最も「似ている」ベクトル対を内積で探す
import numpy as np

vectors = {
    "猫": np.array([4.0, 4.0, 1.0]),
    "犬": np.array([4.0, 3.0, 2.0]),
    "車": np.array([0.0, 1.0, 5.0]),
    "バス": np.array([0.0, 0.5, 4.0]),
}

names = list(vectors.keys())
best_pair = None
best_score = -np.inf

for i in range(len(names)):
    for j in range(i + 1, len(names)):  # 同じ対を2回数えない・自分自身は除く
        score = np.dot(vectors[names[i]], vectors[names[j]])
        print(f"{names[i]} ・ {names[j]} = {score}")
        if score > best_score:
            best_score = score
            best_pair = (names[i], names[j])

print("最も似ている対:", best_pair, "内積 =", best_score)
assert best_pair == ("猫", "犬")
assert np.allclose(best_score, 30.0)
