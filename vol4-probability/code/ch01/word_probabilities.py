# 第4巻 第1章 演習3 略解: 小さなコーパスで単語の出現確率を数える
# 本文 1.3(ユニグラム)と 1.4(「the」の次の条件付き分布)の手計算を、
# コードで数え直して assert で一致確認する。確率の2つのルール(0以上・全部足すと1)が
# 「数えて割る」から自動的に満たされることも確認する。
from collections import Counter

import numpy as np

# --- コーパス: 本文 1.3 と同じ4文 ---
corpus = [
    "the cat sat on the mat",
    "the cat ate the fish",
    "the dog sat on the mat",
    "the dog chased the cat",
]

# --- 1.3 ユニグラム: P(W = w) = (w の出現回数) / (総単語数) ---
words = []
for sentence in corpus:
    words.extend(sentence.split())

total = len(words)
counts = Counter(words)
P = {w: c / total for w, c in counts.items()}

print(f"総単語数: {total}語, 語彙サイズ: {len(counts)}語")
print("単語の出現確率(ユニグラム):")
for w, p in sorted(P.items(), key=lambda kv: -kv[1]):
    print(f"  P(W = {w:6s}) = {counts[w]:2d}/{total} = {p:.3f}")

# 本文 1.3 の表と一致するか
assert total == 22
assert counts["the"] == 8 and np.isclose(P["the"], 8 / 22)
assert counts["cat"] == 3 and np.isclose(P["cat"], 3 / 22)
assert counts["mat"] == 2 and np.isclose(P["mat"], 2 / 22)
# 確率のルール: どの値も0以上、全部足すと1
assert all(p >= 0 for p in P.values())
assert np.isclose(sum(P.values()), 1.0)

# --- 1.4 隣り合う2語のペアを数える(文をまたぐペアは作らない) ---
pairs = []
for sentence in corpus:
    ws = sentence.split()
    for i in range(len(ws) - 1):
        pairs.append((ws[i], ws[i + 1]))

n_pairs = len(pairs)
pair_counts = Counter(pairs)
prev_counts = Counter(prev for prev, nxt in pairs)
assert n_pairs == 18


def next_word_distribution(context):
    """条件付き分布 P(W | C = context) を「絞って、数えて、割る」で作る"""
    followers = Counter(nxt for prev, nxt in pairs if prev == context)
    return {nxt: c / prev_counts[context] for nxt, c in followers.items()}


P_next = next_word_distribution("the")
print('\n「the」の次に来る単語の条件付き分布:')
for w, p in sorted(P_next.items(), key=lambda kv: -kv[1]):
    print(f"  P(W = {w:4s} | C = the) = {p:.3f}")

# 本文 1.4 の表と一致するか
assert np.isclose(P_next["cat"], 3 / 8)
assert np.isclose(P_next["dog"], 1 / 4)
assert np.isclose(P_next["mat"], 1 / 4)
assert np.isclose(P_next["fish"], 1 / 8)
# 条件付き分布も、それ自体が1つの確率分布(0以上・全部足すと1)
assert all(p >= 0 for p in P_next.values())
assert np.isclose(sum(P_next.values()), 1.0)
# どの文脈で絞っても、同じ2つのルールが成り立つ
for context in prev_counts:
    dist = next_word_distribution(context)
    assert all(p >= 0 for p in dist.values())
    assert np.isclose(sum(dist.values()), 1.0)

# --- 1.4 同時確率と掛け算の規則: P(C, W) = P(C) × P(W | C) ---
P_joint = pair_counts[("the", "cat")] / n_pairs  # 「前が the かつ 次が cat」のペアの割合
P_prev_the = prev_counts["the"] / n_pairs        # 「前が the」のペアの割合
print("\n掛け算の規則の検算:")
print(f"  P(C=the, W=cat)          = 3/18 = {P_joint:.4f}")
print(f"  P(C=the) × P(W=cat|C=the) = {P_prev_the:.4f} × {P_next['cat']:.4f}"
      f" = {P_prev_the * P_next['cat']:.4f}")

assert np.isclose(P_joint, 3 / 18)
assert np.isclose(P_joint, P_prev_the * P_next["cat"])

print("\nok: すべての assert を通過しました")
