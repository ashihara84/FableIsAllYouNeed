"""第8巻 第6章 6.2: BLEU をフルスクラッチ実装する。

BLEU (Papineni et al., 2002) = modified n-gram precision の幾何平均 × brevity penalty
- modified precision: 候補文の n-gram の一致を、参照訳での出現回数を上限(clip)に数える
- brevity penalty: 短すぎる候補文への罰
- corpus-level: 分子・分母をコーパス全体で合算してから割る(文ごとの BLEU の平均ではない)

第6巻4章の n-gram(数えるだけの言語モデル)が、評価指標として再登場している。
実行: python3 bleu.py で手計算例との一致を自己点検。テスト本体は test_bleu.py。
"""
import math
from collections import Counter


def ngram_counts(tokens, n):
    """トークン列に含まれる n-gram を数える(第6巻4章と同じ「数えるだけ」)。"""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def modified_precision_counts(candidate, references, n):
    """modified n-gram precision の(分子, 分母)を返す。

    あえて割らずに返す: corpus BLEU は文ごとの比を平均するのではなく、
    分子・分母をコーパス全体で合算してから1回だけ割る(ここが急所)。
    """
    cand = ngram_counts(candidate, n)
    if not cand:
        return 0, 0  # 候補が n トークン未満なら、この n の n-gram は存在しない
    # clip の上限: 同じ n-gram について、参照訳ごとの出現回数の最大値
    max_ref = Counter()
    for ref in references:
        for g, c in ngram_counts(ref, n).items():
            max_ref[g] = max(max_ref[g], c)
    clipped = sum(min(c, max_ref[g]) for g, c in cand.items())
    return clipped, sum(cand.values())


def closest_ref_length(candidate, references):
    """候補文と長さが最も近い参照訳の長さ(同差なら短い方を採る)。"""
    c = len(candidate)
    return min((abs(len(r) - c), len(r)) for r in references)[1]


def corpus_bleu(candidates, references_list, max_n=4):
    """corpus-level BLEU(0〜1)。candidates[i] の参照訳の集合が references_list[i]。"""
    assert len(candidates) == len(references_list)
    num = [0] * max_n  # n ごとの分子(clip 済み一致数)のコーパス合算
    den = [0] * max_n  # n ごとの分母(候補の n-gram 総数)のコーパス合算
    c_total, r_total = 0, 0
    for cand, refs in zip(candidates, references_list):
        for n in range(1, max_n + 1):
            a, b = modified_precision_counts(cand, refs, n)
            num[n - 1] += a
            den[n - 1] += b
        c_total += len(cand)
        r_total += closest_ref_length(cand, refs)
    if min(den) == 0 or min(num) == 0:
        return 0.0  # どこかの n で一致が1つもなければ log 0 = -inf(幾何平均は0)
    log_p = sum(math.log(a / b) for a, b in zip(num, den)) / max_n
    bp = 1.0 if c_total > r_total else math.exp(1.0 - r_total / c_total)
    return bp * math.exp(log_p)


def sentence_bleu(candidate, references, max_n=4):
    """1文だけの corpus BLEU。観察・デバッグ用(報告には corpus_bleu を使うこと)。"""
    return corpus_bleu([candidate], [references], max_n)


if __name__ == "__main__":
    # --- 自己点検: 本文 6.1 の手計算例と1つずつ照合する ---

    ref = "the cat sat on the mat".split()

    # 手計算例1: rug 版。p1..p4 = 5/6, 4/5, 3/4, 2/3、BP = 1、BLEU = (1/3)^(1/4)
    cand = "the cat sat on the rug".split()
    assert modified_precision_counts(cand, [ref], 1) == (5, 6)
    assert modified_precision_counts(cand, [ref], 2) == (4, 5)
    assert modified_precision_counts(cand, [ref], 3) == (3, 4)
    assert modified_precision_counts(cand, [ref], 4) == (2, 3)
    assert math.isclose(sentence_bleu(cand, [ref]), (1 / 3) ** 0.25)

    # 手計算例2: clip。「the」だけ7連発 → 分子は参照訳の上限 2 で頭打ち(2/7)
    cand = "the the the the the the the".split()
    assert modified_precision_counts(cand, [ref], 1) == (2, 7)

    # 手計算例3: brevity penalty。完全な前置詞句止まり → 全 precision 1 でも罰が残る
    cand = "the cat sat on the".split()
    for n in [1, 2, 3, 4]:
        a, b = modified_precision_counts(cand, [ref], n)
        assert a == b  # precision はすべて 1
    assert math.isclose(sentence_bleu(cand, [ref]), math.exp(1 - 6 / 5))

    # 手計算例4: 語順の入れ替え。意味は同じでも 4-gram 一致が 0 → BLEU = 0
    cand = "on the mat sat the cat".split()
    assert sentence_bleu(cand, [ref]) == 0.0

    # 完全一致なら 1.0(満点)
    assert math.isclose(sentence_bleu(ref, [ref]), 1.0)

    print("ok: bleu.py の手計算例をすべて通過しました")
