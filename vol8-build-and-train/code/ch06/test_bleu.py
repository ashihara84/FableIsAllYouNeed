"""第8巻 第6章: bleu.py のテスト。python3 test_bleu.py で実行。

本文 6.1 の手計算例(bleu.py の自己点検と同じもの)に加えて、
BLEU の設計が「2つのズル」を本当に防ぐか、corpus-level の合算が
文ごとの平均とどう違うか、を性質テストとして固定する。
"""
import math

from bleu import (closest_ref_length, corpus_bleu, modified_precision_counts,
                  ngram_counts, sentence_bleu)

REF = "the cat sat on the mat".split()

# === テスト1: 手計算例との全数一致(本文 6.1) =================================

# rug 版: p_n = 5/6, 4/5, 3/4, 2/3 → 幾何平均 (1/3)^(1/4)、BP = 1
cand = "the cat sat on the rug".split()
expected = [(5, 6), (4, 5), (3, 4), (2, 3)]
for n in [1, 2, 3, 4]:
    assert modified_precision_counts(cand, [REF], n) == expected[n - 1]
assert math.isclose(sentence_bleu(cand, [REF]), (1 / 3) ** 0.25)

# === テスト2: ズルその1(繰り返し)は clip が潰す =============================

cand = "the the the the the the the".split()
assert ngram_counts(cand, 1) == {("the",): 7}
assert modified_precision_counts(cand, [REF], 1) == (2, 7)  # clip なしなら 7/7 だった

# === テスト3: ズルその2(短文)は brevity penalty が潰す ======================

cand = "the cat sat on the".split()
for n in [1, 2, 3, 4]:
    a, b = modified_precision_counts(cand, [REF], n)
    assert a == b  # precision は全部 1.0(語尾を捨てただけなので)
assert math.isclose(sentence_bleu(cand, [REF]), math.exp(1 - 6 / 5))  # それでも約 0.819

# === テスト4: 語順の入れ替え — 意味が同じでも 4-gram が全滅して BLEU = 0 =======

scrambled = "on the mat sat the cat".split()
assert modified_precision_counts(scrambled, [REF], 1) == (6, 6)  # 単語は全部合っている
assert modified_precision_counts(scrambled, [REF], 4) == (0, 3)  # 4連続の一致は皆無
assert sentence_bleu(scrambled, [REF]) == 0.0

# === テスト5: 満点と零点の境界条件 ============================================

assert math.isclose(sentence_bleu(REF, [REF]), 1.0)        # 完全一致は 1.0
assert sentence_bleu("a b c".split(), [REF]) == 0.0        # 1語も合わなければ 0
assert sentence_bleu([], [REF]) == 0.0                     # 空出力でも壊れず 0

# === テスト6: 複数参照訳 — どれか1つに完全一致すれば満点 ======================

ref2 = "there is a cat on the mat".split()
assert math.isclose(sentence_bleu(ref2, [REF, ref2]), 1.0)
# clip の上限は「参照訳ごとの出現回数の最大値」: the は REF 側の 2 が上限になる
cand = "the the there".split()
assert modified_precision_counts(cand, [REF, ref2], 1) == (3, 3)
# 参照訳の長さは「候補に最も近いもの」を採用する
assert closest_ref_length("a b c d e f g".split(), [REF, ref2]) == 7

# === テスト7: corpus-level は文ごとの平均ではない =============================

# 文B 単独では 4-gram 全滅で BLEU = 0。しかしコーパスに混ぜると、
# 分子・分母の合算なので全体は 0 にならない(第6巻4章のゼロ頻度の教訓と同じ顔)
cands = [REF, scrambled]            # 文A: 完全一致, 文B: 語順入れ替え
refs_list = [[REF], [REF]]
assert sentence_bleu(scrambled, [REF]) == 0.0
assert corpus_bleu(cands, refs_list) > 0.5
# 検算: 合算は n=4 で (3+0)/(3+3)、n=3 で (4+1)/(4+4)、n=2 で (5+3)/(5+5)、
#       n=1 で (6+6)/(6+6)。BP = 1(c = r = 12)
expected = (1.0 * (8 / 10) * (5 / 8) * (3 / 6)) ** 0.25
assert math.isclose(corpus_bleu(cands, refs_list), expected)

print("ok: test_bleu.py のすべての assert を通過しました")
