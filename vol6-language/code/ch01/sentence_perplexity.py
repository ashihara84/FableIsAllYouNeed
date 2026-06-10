# 第6巻 第1章 演習3 略解: 文の確率(連鎖分解)と perplexity を小コーパスで計算する
# 本文 1.2 の表と 1.4 の数値(数えるモデル ≈ 1.26、一様モデル = 9)を assert で検算する。
# モデルの作り方は第4巻1章と同じ「絞って、数えて、割る」。
import math

import numpy as np

# --- コーパス: 第4巻1章から引き継いだ4文 ---
corpus = [
    "the cat sat on the mat",
    "the cat ate the fish",
    "the dog sat on the mat",
    "the dog chased the cat",
]
sentences = [s.split() for s in corpus]
vocab = sorted(set(w for ws in sentences for w in ws))
V = len(vocab)
assert V == 9


def p_next(prefix):
    """P(w_t = ・ | 文頭からの単語列 = prefix)。ゲームは「4文から1文を無作為に選ぶ」"""
    t = len(prefix)
    matched = [ws for ws in sentences if ws[:t] == prefix]   # 絞って
    followers = [ws[t] for ws in matched if len(ws) > t]     # 数えて
    return {w: followers.count(w) / len(matched) for w in set(followers)}  # 割る


def sentence_log_prob(words, next_dist_fn):
    """log P(w_1, ..., w_T) を連鎖分解(本文 1.2 の式)で計算する(自然対数)。
    モデルが確率 0 を返す続き(コーパスにない続き)に出会ったら -inf を返す"""
    log_p = 0.0
    for t in range(len(words)):
        dist = next_dist_fn(words[:t])
        if words[t] not in dist:
            return -math.inf
        log_p += math.log(dist[words[t]])
    return log_p


# --- 1.2 の表: 各ステップの条件付き確率と、連鎖分解の積 = 1/4 ---
target = "the cat sat on the mat".split()
steps = [p_next(target[:t])[target[t]] for t in range(len(target))]
assert np.allclose(steps, [1.0, 1 / 2, 1 / 2, 1.0, 1.0, 1.0])

P_sentence = math.exp(sentence_log_prob(target, p_next))
assert np.isclose(P_sentence, np.prod(steps))
assert np.isclose(P_sentence, 1 / 4)   # 直接数えた「4文中1文」と一致

# どの文も「4文中1文」。コーパスにない文には確率 0(丸暗記の正体)
for ws in sentences:
    assert np.isclose(math.exp(sentence_log_prob(ws, p_next)), 1 / 4)
assert sentence_log_prob("the cat sat on the fish".split(), p_next) == -math.inf


# --- 1.4: 1語あたり cross-entropy と perplexity ---
def cross_entropy_per_word(words, next_dist_fn):
    """H = -(1/T) Σ_t log q(w_t | w_<t)。1語あたりの平均の驚き(ナット)"""
    return -sentence_log_prob(words, next_dist_fn) / len(words)


def uniform_next(prefix):
    """一様モデル: どんな文脈でも語彙9語に 1/9 ずつ配る(何も学んでいないモデル)"""
    return {w: 1.0 / V for w in vocab}


H = cross_entropy_per_word(target, p_next)
ppl = math.exp(H)
H_u = cross_entropy_per_word(target, uniform_next)
ppl_u = math.exp(H_u)
print(f"数えるモデル: H = {H:.3f} ナット/語, perplexity = {ppl:.3f}")
print(f"一様モデル  : H = {H_u:.3f} ナット/語, perplexity = {ppl_u:.3f}")
assert np.isclose(ppl, 4 ** (1 / 6))                       # ≈ 1.26
assert np.isclose(ppl, P_sentence ** (-1 / len(target)))   # = P(文)^(-1/T)
assert np.isclose(ppl_u, float(V))   # 何も知らないモデルの平均分岐数 = 語彙サイズ 9
assert ppl < ppl_u

# 訓練に使っていない文では、数えるモデルの perplexity は無限大(本文 1.4 の罠2)
assert cross_entropy_per_word("the cat sat on the fish".split(), p_next) == math.inf

print("ok: 第1章の数値はすべて検算が通りました")
