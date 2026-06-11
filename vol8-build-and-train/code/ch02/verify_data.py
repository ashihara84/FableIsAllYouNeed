# 第8巻 第2章: data.py の契約検証スクリプト
#
# data.py は第3・5・6章が import する共有モジュールなので、他章が前提に
# している契約(特殊トークンの番号、関数の入出力の形)をここで assert する。
# 仕上げに第7巻3章の attention と接続し、pad mask が末端まで効くこと
# (PAD への注意の重みが厳密に 0 になること)を確認する。

import os
import sys

import numpy as np

import data

# --- 第7巻3章の attention を import する(後の巻が前の巻を使う一方向の依存)--
_VOL7_CH03 = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "vol7-attention", "code", "ch03"))
if _VOL7_CH03 not in sys.path:
    sys.path.insert(0, _VOL7_CH03)
from attention import attention  # noqa: E402

# --- 契約1: 特殊トークンの番号 ------------------------------------------------
assert data.PAD == 0 and data.BOS == 1 and data.EOS == 2
assert data.itos[data.PAD] == "<pad>"
assert data.itos[data.BOS] == "<bos>"
assert data.itos[data.EOS] == "<eos>"

# --- 契約2: コーパスは決定的で、規定の規模 ------------------------------------
corpus = data.make_corpus()
assert corpus == data.make_corpus() == data.CORPUS
assert 150 <= len(corpus) <= 300
assert len(set(corpus)) == len(corpus)          # 重複ペアなし
print("コーパス: %d ペア(重複なし・決定的)" % len(corpus))
print("語彙: %d(特殊 3 + BPE %d)" % (data.vocab_size, data.vocab_size - 3))

# --- 契約3: encode_pair / decode ----------------------------------------------
encoded = [data.encode_pair(s, t) for s, t in corpus]
for (s, t), (s_ids, t_ids) in zip(corpus, encoded):
    assert s_ids[-1] == data.EOS                       # src = [..., EOS]
    assert t_ids[0] == data.BOS and t_ids[-1] == data.EOS  # tgt = [BOS, ..., EOS]
    assert all(0 <= i < data.vocab_size for i in s_ids + t_ids)
    assert data.PAD not in s_ids and data.PAD not in t_ids  # PAD は文には現れない
    assert data.decode(s_ids) == s and data.decode(t_ids) == t  # 往復一致
print("encode_pair / decode: 全 %d ペアで往復一致" % len(corpus))

# --- 契約4: make_batches ------------------------------------------------------
rng = np.random.default_rng(42)
_, rate_rand = data.make_batches(encoded, batch_size=32, rng=rng, by_length=False)
rng = np.random.default_rng(42)
batches, rate_len = data.make_batches(encoded, batch_size=32, rng=rng)

assert sum(src.shape[0] for src, tgt in batches) == len(corpus)
for src, tgt in batches:
    assert isinstance(src, np.ndarray) and isinstance(tgt, np.ndarray)
    assert src.dtype == np.int64 and tgt.dtype == np.int64
    assert src.ndim == 2 and tgt.ndim == 2 and src.shape[0] == tgt.shape[0]
    assert (tgt[:, 0] == data.BOS).all()

# 2.3節の数字(rng=42 で決定的): 長さ順バッチは padding の無駄を 1/3 にする
assert round(rate_rand, 3) == 0.264
assert round(rate_len, 3) == 0.087
assert rate_len < rate_rand / 2
print("padding率: ランダム %.1f%% / 長さ順 %.1f%%" % (100 * rate_rand, 100 * rate_len))

# --- 契約5: pad mask が第7巻の attention の末端まで効く -----------------------
src, _ = batches[0]                       # (B, m)
B, m = src.shape
mask = data.make_pad_mask(src)            # (B, m)  True = 見てよい
assert mask.shape == (B, m) and mask.dtype == bool

rng = np.random.default_rng(0)
n, d_k = 4, 8
Q = rng.standard_normal((B, n, d_k))
K = rng.standard_normal((B, m, d_k))
V = rng.standard_normal((B, m, d_k))
_, w_masked = attention(Q, K, V, mask=mask[:, None, :])   # (B, n, m)
_, w_open = attention(Q, K, V)

pad_pos = np.broadcast_to(~mask[:, None, :], (B, n, m))   # PAD の位置(全 query 行)
assert (w_masked[pad_pos] == 0.0).all()                   # mask あり: 重みは厳密に 0
assert np.allclose(w_masked.sum(axis=-1), 1.0)            # 本物だけで和が 1
assert (w_open[pad_pos] > 0).all()                        # mask なし: PAD に漏れる
print("pad mask: PAD への注意の重みは厳密に 0(mask なしでは漏れる)")

print()
print("すべての assert を通過しました。")
