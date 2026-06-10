"""第6巻 第4章: n-gram 言語モデル — 数えて、生成して、perplexity で測る。

本文 4.1〜4.3 のコードを1本にまとめたもの。
- 4.1 カウント: 条件付き確率を頻度で推定(第4巻第3章の最尤推定 θ̂ = k/n の言語版)
- 4.2 生成: n を変えて文章を生成する(初めて機械が文を書く)
- 4.3 壁: 組合せ爆発とゼロ頻度を実測し、perplexity の表にする
"""
from collections import Counter, defaultdict

import numpy as np

rng = np.random.default_rng(42)

BOS, EOS = "<s>", "</s>"  # 文頭・文末の印(語彙には EOS だけが加わる)

# --- コーパス: 訓練50文。最初の4文は第4巻第1章と同じもの ---
TRAIN_TEXT = """
the cat sat on the mat
the cat ate the fish
the dog sat on the mat
the dog chased the cat
the cat chased the mouse
the mouse ran to the house
the dog ate the bone
the bird sat on the house
the bird saw the cat
the cat saw the bird
the old cat slept in the garden
the small dog ran to the river
the man saw the dog
the man liked the old dog
the woman liked the small cat
the woman saw the bird in the garden
the cat slept on the mat
the dog slept in the house
the fish swam in the river
the bird flew to the garden
the hungry cat ate the fish
the hungry dog ate the bone
the black cat sat on the house
the white dog sat in the garden
the mouse ate the cheese
the cat liked the warm sun
the dog ran to the man
the man ran to the river
the woman sat in the garden
the old man slept in the house
the small bird flew to the river
the black dog chased the white cat
the white cat chased the small mouse
the mouse slept in the small house
the fish saw the bird
the bird ate the small fish
the cat was hungry
the dog was happy
the old man was happy
the small mouse was hungry
the sun was warm
the garden was quiet
the man liked the quiet garden
the woman liked the warm sun
the cat saw the fish in the river
the dog saw the mouse in the garden
the hungry bird ate the cheese
the black cat slept on the warm mat
the man chased the black dog
the woman chased the small bird
"""

# --- テスト用の10文: モデルには一度も見せない(第3巻の訓練/テスト分割と同じ規律) ---
TEST_TEXT = """
the white cat sat on the mat
the hungry dog ran to the house
the old man liked the black cat
the dog chased the bird in the garden
the small fish swam in the river
the small mouse ate the cheese
the white dog slept on the mat
the man saw the white cat
the old cat slept in the warm sun
the woman saw the fish in the river
"""

train_sents = [line.split() for line in TRAIN_TEXT.strip().split("\n")]
test_sents = [line.split() for line in TEST_TEXT.strip().split("\n")]
assert len(train_sents) == 50 and len(test_sents) == 10


# === 4.1 カウント: 条件付き確率を頻度で推定 ===

def ngram_counts(sents, n):
    """文のリストから n-gram の出現を数える。

    返り値は「文脈(直前 n−1 トークンのタプル)→ 次トークンの Counter」の辞書。
    文頭には BOS を n−1 個敷き、文末には EOS を1個置く(文の終わりも予測対象)。
    """
    counts = defaultdict(Counter)
    for ws in sents:
        tokens = [BOS] * (n - 1) + ws + [EOS]
        for i in range(n - 1, len(tokens)):
            context = tuple(tokens[i - n + 1:i])
            counts[context][tokens[i]] += 1
    return counts


def prob(counts, context, w):
    """最尤推定 P̂(w | context) = count(context, w) / count(context, ・)"""
    c = counts[context]
    total = sum(c.values())
    if total == 0:
        return 0.0  # 文脈そのものが未観測
    return c[w] / total


# 第4巻第1章の4文コーパスで検算: P̂(cat | the) = 3/8(あの表と同じ数)
counts_v4 = ngram_counts(train_sents[:4], 2)
assert counts_v4[("the",)]["cat"] == 3          # k: 「the cat」の回数
assert sum(counts_v4[("the",)].values()) == 8   # n: 「the ○」の回数
assert np.isclose(prob(counts_v4, ("the",), "cat"), 3 / 8)

# 50文で数え直しても、どの文脈の条件付き分布も「0以上・足すと1」を自動で満たす
counts2 = ngram_counts(train_sents, 2)
for context, c in counts2.items():
    total = sum(c.values())
    assert all(v > 0 for v in c.values())
    assert np.isclose(sum(v / total for v in c.values()), 1.0)


# === 4.2 生成: 条件付き分布からサンプリングして文を書かせる ===

def generate(counts, n, max_len=20):
    """BOS だけの文脈から始め、P̂(・| 文脈) を引いては1語ずつ進める。"""
    context = tuple([BOS] * (n - 1))
    out = []
    while len(out) < max_len:
        dist = counts[context]
        words = sorted(dist)                                # 順序を固定(再現性)
        p = np.array([dist[w] for w in words], dtype=float)
        p = p / p.sum()
        w = str(rng.choice(words, p=p))
        if w == EOS:
            break
        out.append(w)
        if n > 1:
            context = context[1:] + (w,)
    return " ".join(out)


train_set = {" ".join(ws) for ws in train_sents}

print("=== 生成(各 n で10文ずつ。[copy] は訓練コーパスの丸写し) ===")
copied = {}
for n in [1, 2, 3, 4]:
    counts_n = ngram_counts(train_sents, n)
    gens = [generate(counts_n, n) for _ in range(10)]
    copied[n] = sum(g in train_set for g in gens)
    print(f"--- n = {n}(丸写し {copied[n]}/10) ---")
    for g in gens:
        print("  " + g + ("   [copy]" if g in train_set else ""))

# n を上げると文は正しくなるが、それは「上手くなった」のではなく「暗記に近づいた」
assert copied[1] == 0 and copied[2] == 0
assert copied[4] >= 6


# === 4.3 壁(その1): 組合せ爆発 — V^n と「実際に観測された異なり n-gram 数」 ===

vocab = sorted({w for ws in train_sents for w in ws})
V = len(vocab) + 1  # +1 は EOS(BOS は予測対象でないため数えない)
n_tokens = sum(len(ws) + 1 for ws in train_sents)  # 訓練中の n-gram の延べ個数(EOS 込み)
assert V == 37 and n_tokens == 348

print("\n=== 組合せ爆発: 可能な n-gram の数 vs 観測された n-gram の数 ===")
print(f"語彙サイズ V = {V}(EOS 込み), 訓練トークン数 = {n_tokens}")
seen = {}
for n in [1, 2, 3, 4, 5]:
    counts_n = ngram_counts(train_sents, n)
    seen[n] = sum(len(c) for c in counts_n.values())  # 異なり n-gram の個数
    print(f"  n = {n}: V^n = {V ** n:>12,} 通り | 観測 {seen[n]:>3} 通り"
          f"(カバー率 {seen[n] / V ** n:.2%})")

# 観測できる異なり n-gram は最大でも延べ個数 n_tokens を超えられない(必然の頭打ち)
assert all(seen[n] <= n_tokens for n in [1, 2, 3, 4, 5])
assert seen[5] / V ** 5 < 1e-5  # n = 5 では可能な組合せの 0.001% も見えていない


# === 4.3 壁(その2): ゼロ頻度 — テスト文の n-gram のうち、訓練で一度も見ていない割合 ===

print("\n=== ゼロ頻度: テスト10文のうち訓練で確率 0 になる n-gram の割合 ===")
zero_rate = {}
for n in [1, 2, 3, 4]:
    counts_n = ngram_counts(train_sents, n)
    zeros, total = 0, 0
    for ws in test_sents:
        tokens = [BOS] * (n - 1) + ws + [EOS]
        for i in range(n - 1, len(tokens)):
            p = prob(counts_n, tuple(tokens[i - n + 1:i]), tokens[i])
            zeros += (p == 0.0)
            total += 1
    zero_rate[n] = zeros / total
    print(f"  n = {n}: {zeros:>2}/{total} = {zero_rate[n]:.1%}")

assert zero_rate[1] == 0.0 and zero_rate[2] == 0.0  # 1・2-gram は全部見たことがある
assert zero_rate[3] > 0.0                           # 3-gram で初めて「見たことがない」が出る
assert zero_rate[4] > zero_rate[3]                  # n を増やすほど悪化する


# === 4.3 壁(その3): perplexity — 訓練では下がり続け、テストでは無限大に発散 ===

def perplexity(counts, n, sents):
    """PP = exp(−(1/N) Σ log P̂)。第4巻第7章の「平均分岐数」をコーパスで測る。"""
    log_sum, N = 0.0, 0
    for ws in sents:
        tokens = [BOS] * (n - 1) + ws + [EOS]
        for i in range(n - 1, len(tokens)):
            p = prob(counts, tuple(tokens[i - n + 1:i]), tokens[i])
            if p == 0.0:
                return float("inf")  # log 0 = −∞。一発で perplexity 全体が無限大
            log_sum += np.log(p)
            N += 1
    return float(np.exp(-log_sum / N))


print("\n=== perplexity(平均分岐数): 訓練 vs テスト ===")
pp_train, pp_test = {}, {}
for n in [1, 2, 3, 4]:
    counts_n = ngram_counts(train_sents, n)
    pp_train[n] = perplexity(counts_n, n, train_sents)
    pp_test[n] = perplexity(counts_n, n, test_sents)
    print(f"  n = {n}: 訓練 {pp_train[n]:7.2f} | テスト {pp_test[n]:7.2f}")

# 訓練データ上では、n を増やすほど perplexity は単調に下がる(丸暗記に向かう)
assert pp_train[1] > pp_train[2] > pp_train[3] > pp_train[4]
# テストでは 2-gram が最良で、3-gram 以降はゼロ頻度を踏んで無限大
assert pp_test[2] < pp_test[1]
assert pp_test[3] == float("inf") and pp_test[4] == float("inf")

print("\nok: すべての assert を通過しました")
