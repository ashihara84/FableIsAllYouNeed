# 第6巻 第3章 3.4: 小コーパスで埋め込みを学習し、king − man + woman を検算する
# 第1巻1.5で予告した「意味の算術」の伏線回収。
# 第5巻5章の自作 autograd(tensor_autograd.py)を import して使う。
import os
import sys
import warnings

import numpy as np

# 一部の macOS 環境(Accelerate BLAS + NumPy 2.0系)では、正しい行列積でも
# 誤った RuntimeWarning が出ることが知られている(計算結果は正しい)。本筋ではないので非表示にする
warnings.filterwarnings("ignore", message=".*encountered in matmul")

# --- 第5巻の autograd を借りてくる(vol5 のファイルは変更しない) ---
_VOL5 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "..", "vol5-backprop", "code", "ch05")
sys.path.insert(0, _VOL5)
from tensor_autograd import Tensor, softmax_cross_entropy

rng = np.random.default_rng(42)

# --- 小コーパス(訓練データ) ---
# 性別と王族らしさの2軸が文脈に現れるように手で設計してある(本文3.4参照)
corpus = [
    "the king rules the palace",
    "the queen rules the palace",
    "the prince guards the palace",
    "the princess guards the palace",
    "the man works in the village",
    "the woman works in the village",
    "the boy plays in the village",
    "the girl plays in the village",
    "the king is a man",
    "the queen is a woman",
    "the prince is a boy",
    "the princess is a girl",
    "he is the king",
    "she is the queen",
    "he is the prince",
    "she is the princess",
    "he is the man",
    "she is the woman",
    "he is the boy",
    "she is the girl",
    "the king wears the crown",
    "the queen wears the crown",
]

# --- 語彙とトークンID(第2章なら BPE の出番だが、ここでは空白区切りで足りる) ---
sentences = [s.split() for s in corpus]
vocab = sorted(set(w for s in sentences for w in s))
V = len(vocab)                                # 語彙サイズ V = 22
word2id = {w: i for i, w in enumerate(vocab)}

# --- 訓練ペアの作成(skip-gram: 中心語から文脈語を当てる) ---
window = 3
centers, contexts = [], []
for s in sentences:
    for i in range(len(s)):
        for j in range(max(0, i - window), min(len(s), i + window + 1)):
            if j != i:
                centers.append(word2id[s[i]])
                contexts.append(word2id[s[j]])
centers = np.array(centers)
contexts = np.array(contexts)
n = len(centers)                              # n = 372 ペア

# --- one-hot 行列 X (n, V): 第 i 行は centers[i] の位置だけ 1 ---
X_onehot = np.zeros((n, V))
X_onehot[np.arange(n), centers] = 1.0

# 3.1 の確認: 異なる単語の one-hot どうしの内積は必ず 0(すべての単語が等しく無関係)
assert np.dot(X_onehot[0], X_onehot[1]) == 0.0 or centers[0] == centers[1]

# --- パラメータ: 埋め込み行列 E (V, d) と出力層 W_out (d, V) ---
d = 8                                          # 埋め込みの次元(論文なら d_model = 512)
E = Tensor(rng.normal(0.0, 0.1, (V, d)))
W_out = Tensor(rng.normal(0.0, 0.1, (d, V)))
X = Tensor(X_onehot)

# 3.2 の確認: one-hot @ E は「E から行を取り出す」のと同じ(lookup はただの行列積)
assert np.allclose((X @ E).data, E.data[centers])

# --- 訓練ループ(第3巻4章の4拍子: forward → loss → backward → 更新) ---
lr = 5.0
losses = []
for epoch in range(300):
    E.grad = np.zeros_like(E.data)            # 勾配は += で溜まるので毎回ゼロに戻す
    W_out.grad = np.zeros_like(W_out.data)
    logits = (X @ E) @ W_out                  # (n, V): 文脈語の当てっこのスコア
    loss = softmax_cross_entropy(logits, contexts)
    loss.backward()
    E.data -= lr * E.grad
    W_out.data -= lr * W_out.grad
    losses.append(float(loss.data))

# 開始時の損失は「V 択を一様に当てずっぽう」の log V ≈ 3.09 付近。そこから下がったか
assert abs(losses[0] - np.log(V)) < 0.1
assert losses[-1] < losses[0] - 0.5
print("loss: %.3f -> %.3f (log V = %.3f)" % (losses[0], losses[-1], np.log(V)))

# --- 学習した埋め込みで意味の算術を検算する ---
Emb = E.data                                   # (V, d) 学習済み埋め込み


def cosine(a, b):
    """コサイン類似度(第1巻2.3)。正規化してから内積"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def nearest(vec, exclude=()):
    """vec にコサイン類似度が高い順に語彙を並べる(exclude の単語は除く)"""
    scores = [(cosine(vec, Emb[word2id[w]]), w) for w in vocab if w not in exclude]
    return sorted(scores, reverse=True)


def analogy(a, b, c):
    """a − b + c に最も近い単語を返す(word2vec の流儀で a, b, c 自身は候補から除く)"""
    v = Emb[word2id[a]] - Emb[word2id[b]] + Emb[word2id[c]]
    return nearest(v, exclude=(a, b, c))


# 第1巻1.5の伏線回収: king − man + woman ≈ queen
top = analogy("king", "man", "woman")
print("king - man + woman ->", [(w, round(float(s), 3)) for s, w in top[:3]])
assert top[0][1] == "queen"
assert top[0][0] > 0.8

# 同じ算術がもう1組でも成り立つか: prince − boy + girl ≈ princess
top2 = analogy("prince", "boy", "girl")
print("prince - boy + girl ->", [(w, round(float(s), 3)) for s, w in top2[:3]])
assert top2[0][1] == "princess"

# 平行性の確認: 「男 → 女」の差ベクトルが、ペアによらずほぼ同じ向きを向いている
d_kq = Emb[word2id["king"]] - Emb[word2id["queen"]]
d_mw = Emb[word2id["man"]] - Emb[word2id["woman"]]
d_bg = Emb[word2id["boy"]] - Emb[word2id["girl"]]
print("cos(king-queen, man-woman) = %.3f" % cosine(d_kq, d_mw))
print("cos(boy-girl,   man-woman) = %.3f" % cosine(d_bg, d_mw))
assert cosine(d_kq, d_mw) > 0.7
assert cosine(d_bg, d_mw) > 0.7

# --- 演習用: 埋め込み空間の近傍語を観察する ---
for w in ["king", "queen", "man", "village"]:
    print("%-8s の近傍:" % w, [(u, round(float(s), 3)) for s, u in nearest(Emb[word2id[w]], exclude=(w,))[:3]])

print("第3章: すべての assert を通過しました")
