"""第8巻 第6章 6.2: 自作モデルの perplexity / BLEU を測る — 評価パイプラインの最小形。

本デモは候補文を直書きし、評価側(この章の担当分)を単体で検証する。測る側のコードは、
測られる側が何であっても変わらない——実モデルで測るには、第5章のチェックポイント
(code/ch05/ch05_checkpoint.pt)を読み込み、CANDIDATES を生成文に差し替えればよい(演習)。

実行: python3 evaluate_demo.py(CPU で数秒)
"""
import math

from bleu import corpus_bleu, modified_precision_counts, sentence_bleu

# === perplexity: 検証 loss(平均 cross-entropy)の指数 =========================


def perplexity(nll_per_token):
    """1トークンあたりの負の対数尤度(自然対数)の列から perplexity を計算する。

    検証 loss が「トークンあたり平均 cross-entropy」なら math.exp(loss) と同じこと。
    第4巻7.5の「平均分岐数」、第6巻4章で n-gram を測った、あの物差し。
    """
    return math.exp(sum(nll_per_token) / len(nll_per_token))


# 検算1: 全トークンで語彙 37 の一様分布なら、平均分岐数はちょうど 37
V = 37
assert math.isclose(perplexity([math.log(V)] * 100), V)

# 検算2: 半分のトークンで2択(-log 1/2)、半分で1択(-log 1)なら √2 ≈ 1.41
assert math.isclose(perplexity([math.log(2), 0.0] * 50), math.sqrt(2))

# 検算3: 訓練ループの検証 loss からは exp(loss) の1行で出る
val_loss = 2.3  # 第5章の訓練ループが報告する数値の例
assert math.isclose(perplexity([val_loss] * 10), math.exp(val_loss))

# === BLEU: 検証セットの参照訳 vs モデルの出力 =================================

# 参照訳(検証セット)。語彙の世界観は第6巻4章のおもちゃのコーパスと同じ
REFERENCES = [
    "the cat sat on the mat",
    "the dog chased the black cat",
    "the small bird flew to the river",
    "the hungry mouse ate the cheese",
]

# モデルの出力のつもりの候補文(第5章のチェックポイントで生成した文に差し替え可能)
CANDIDATES = [
    "the cat sat on the mat",            # 完全一致
    "the dog chased the cat",            # 1語の脱落
    "the bird flew to the small river",  # 修飾語の位置ずれ
    "the mouse was hungry",              # 短すぎ + 大胆な書き換え
]

cands = [c.split() for c in CANDIDATES]
refs_list = [[r.split()] for r in REFERENCES]

print("=== 文ごとの観察(報告には使わない。corpus との差を見るため)===")
for c, rl in zip(cands, refs_list):
    print(f"  BLEU {100 * sentence_bleu(c, rl):5.1f} | {' '.join(c)}")

print("\n=== corpus-level の内訳 ===")
num, den = [0] * 4, [0] * 4
for c, rl in zip(cands, refs_list):
    for n in range(1, 5):
        a, b = modified_precision_counts(c, rl, n)
        num[n - 1] += a
        den[n - 1] += b
for n in range(1, 5):
    print(f"  p_{n} = {num[n - 1]:2d}/{den[n - 1]:2d} = {num[n - 1] / den[n - 1]:.3f}")

score = corpus_bleu(cands, refs_list)
print(f"\ncorpus BLEU = {100 * score:.1f}")

# --- 検証: 本節の主張をデータで確認する ---
# (1) 文ごとでは、完全一致は 100、4-gram が全滅する文は 0 に張り付く。
#     修飾語ずれの文は「bird flew to the」の 4-gram が1本生き残るので 0 にならない
assert math.isclose(sentence_bleu(cands[0], refs_list[0]), 1.0)
assert 0.0 < sentence_bleu(cands[2], refs_list[2]) < sentence_bleu(cands[1], refs_list[1])
assert sentence_bleu(cands[3], refs_list[3]) == 0.0
# (2) corpus-level なら、ゼロの文が混ざっても全体は 0 にならない(合算の効能)
assert 0.0 < score < 1.0
# (3) この実行の値の固定(回帰テスト): 手元の合算 p_n と BP からの再計算と一致
c_total = sum(len(c) for c in cands)
r_total = sum(len(rl[0]) for rl in refs_list)
bp = 1.0 if c_total > r_total else math.exp(1.0 - r_total / c_total)
expected = bp * math.exp(sum(math.log(a / b) for a, b in zip(num, den)) / 4)
assert math.isclose(score, expected)

print("\nok: evaluate_demo.py のすべての assert を通過しました")
