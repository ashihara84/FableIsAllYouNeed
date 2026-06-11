# 第8巻 第1章 1.6: パラメータ数の検算 — 紙の上の式と、組み上げた実物を突き合わせる
# 第7巻6章の演習で作った数え上げの式(transformer_base_params)を import し、
# 1.2 で組み立てた Transformer の「実際に持っている数」と完全一致することを assert する。
# 式と実物が1個までぴったり合えば、組み立てに余計な重みも欠けた重みもない。
# 実行: python3 param_count.py
import os
import sys

import numpy as np

from transformer import Transformer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "vol7-attention", "code", "ch06")))
from ex_param_count import transformer_base_params  # noqa: E402(第7巻6章 演習1)

rng = np.random.default_rng(42)

# base 相当の縮小版: N=6, h=8, d_ff = 4 × d_model の比率はそのまま、幅と語彙だけ 1/8
vocab, d_model, d_ff, h, N = 1000, 64, 256, 8, 6
model = Transformer(vocab, d_model, d_ff, h, N, max_len=32, rng=rng)

params = model.params()
assert len(set(id(p) for p in params)) == len(params)  # 同じ行列の二重カウントなし
actual = model.n_params()

breakdown, expected = transformer_base_params(vocab=vocab, d_model=d_model,
                                              d_ff=d_ff, N=N)
print("縮小版(N={}, d_model={}, d_ff={}, h={}, 語彙{:,})の内訳(式による計算):"
      .format(N, d_model, d_ff, h, vocab))
for name, n in breakdown.items():
    print("  {:<22} {:>10,}".format(name, n))
print("  {:<22} {:>10,}".format("合計(式)", expected))
print("  {:<22} {:>10,}".format("合計(実物)", actual))

# 紙の上の式 = 組み上げた実物。1個の過不足もなく一致する
assert actual == expected

# h を変えても総数は変わらない(d_k = d_model / h に裂いているだけ — 第7巻4章)
model_h4 = Transformer(vocab, d_model, d_ff, 4, N, max_len=32, rng=rng)
assert model_h4.n_params() == actual

# 本家 base の 65M(第7巻6章の検算)も、同じ式の引数を変えるだけで再現できる
_, total_base = transformer_base_params()   # vocab=37000, d_model=512, d_ff=2048, N=6
assert round(total_base / 1e6) == 63        # この数え方では 63M(論文 Table 3 は 65M)

print("param_count: すべての assert を通過しました — 式と実物が {:,} 個で完全一致"
      .format(actual))
