# 第8巻 第1章 1.3: 結合テスト — 組み上がった Transformer 全体を検査する
# 検査は4段:
#   (1) shape: 入口から出口まで形が通る
#   (2) causal mask が末端まで効く: 未来のトークンを改変しても過去の logits は不変
#   (3) NumPy 版(第7巻部品)と Tensor 版(第5巻 autograd)の forward が一致
#       + Tensor 版の勾配を数値微分でスポット照合
#   (4) 丸暗記テスト: 1バッチを過学習できる(デバッグの定石)
# 実行: python3 integration_test.py(seed 42。数十秒で全 assert が通る)
import numpy as np

from transformer import Transformer, Tensor
from tensor_autograd import softmax_cross_entropy  # 第5巻5章(path は transformer が通す)

rng = np.random.default_rng(42)
vocab, d_model, d_ff, h, N = 12, 32, 64, 4, 2      # base の縮小版(比率は同じ d_ff = 4d)
model = Transformer(vocab, d_model, d_ff, h, N, max_len=16, rng=rng)

BOS = 0                                            # 文頭トークン(規約の本番は第2章)
src = rng.integers(1, vocab, size=7)               # 入力文(長さ7)
tgt = rng.integers(1, vocab, size=5)               # 出力文(長さ5)。長さ違いで混線を検出
tgt_in = np.concatenate([[BOS], tgt[:-1]])         # teacher forcing の1トークンずらし(第6巻6.4)

# ---- (0) 環境の自己点検: BLAS の matmul を einsum と照合(transformer.py の偽警告の件) ----
A0 = rng.standard_normal((7, d_model))
B0 = rng.standard_normal((d_model, d_model))
assert np.array_equal(A0 @ B0, np.einsum("ij,jk->ik", A0, B0))

# ---- (1) shape: 全体 forward が通り、出力が (tgt_len, vocab) ----
logits = model.forward_numpy(src, tgt_in)
assert logits.shape == (5, vocab)
assert np.all(np.isfinite(logits))
print("(1) shape OK: src (7,) + tgt_in (5,) -> logits {}".format(logits.shape))

# ---- (2) causal mask が末端まで効く ----
# 未来(最後の位置)のトークンを改変しても、それより前の位置の logits は1bitも動かない
k = len(tgt_in) - 1
tgt_in2 = tgt_in.copy()
tgt_in2[k] = (tgt_in[k] + 3) % vocab
logits2 = model.forward_numpy(src, tgt_in2)
assert np.array_equal(logits[:k], logits2[:k])     # 過去は完全不変(allclose ですらなく等値)
assert not np.allclose(logits[k], logits2[k])      # 改変した当の位置だけは変わる

# わざと壊す: mask を外すと同じ検査が落ちる(このテストがバグを検出できる証拠)
bad1 = model.forward_numpy(src, tgt_in, causal=False)
bad2 = model.forward_numpy(src, tgt_in2, causal=False)
assert not np.allclose(bad1[:k], bad2[:k])         # 未来が過去に漏れる

# 入力文の改変は cross-attention 経由で「全位置」に届く(これは漏れではなく仕様)
src2 = src.copy()
src2[0] = (src[0] + 1) % vocab
logits3 = model.forward_numpy(src2, tgt_in)
assert not np.allclose(logits3[0], logits[0])
print("(2) causal mask OK: 未来の改変は過去に漏れない(mask を外すと漏れる)")

# ---- (3) NumPy 版と Tensor 版の forward 一致 + 勾配のスポット照合 ----
logits_t = model.forward_tensor(src, tgt_in)
diff = np.abs(logits_t.data - logits).max()
assert diff < 1e-9
print("(3) NumPy/Tensor 一致 OK: logits の最大差 = {:.2e}".format(diff))

params = model.params()
assert len(set(id(p) for p in params)) == len(params)   # 二重カウントなし(1.6 の前提)


def batch_loss():
    """損失 = 正解トークン列 tgt に対する cross-entropy(第4巻5章)。"""
    return softmax_cross_entropy(model.forward_tensor(src, tgt_in), tgt)


loss = batch_loss()
for p in params:
    p.grad = np.zeros_like(p.data)
loss.backward()

# 数値微分(第2巻1章の中心差分)でスポット照合: 入口・中間・出口から1枚ずつ
spots = [("embedding E", model.emb.E, (int(src[0]), 0)),
         ("encoder層0 W_q", model.encoder.layers[0].W_q, (3, 7)),
         ("decoder層1 gamma3", model.decoder.layers[1].gamma3, (2,)),
         ("decoder層0 cross U_o", model.decoder.layers[0].U_o, (0, 5))]
eps = 1e-5
for name, p, idx in spots:
    old = p.data[idx]
    p.data[idx] = old + eps
    fp = batch_loss().data
    p.data[idx] = old - eps
    fm = batch_loss().data
    p.data[idx] = old
    num = (fp - fm) / (2 * eps)
    assert np.isclose(p.grad[idx], num, rtol=1e-4, atol=1e-7), name
    print("    勾配照合 OK: {:<18} autograd {:+.8f} / 数値微分 {:+.8f}"
          .format(name, p.grad[idx], num))

# ---- (4) 丸暗記テスト: 1バッチ(2系列)を過学習できるか ----
# 対応関係に規則のない src -> tgt のペアを2本固定し、丸暗記させる。
# 規則がないので「学習できた」= 「このバッチを記憶する配管と勾配が末端まで生きている」。
pairs = []
for _ in range(2):
    s = rng.integers(1, vocab, size=6)
    t = rng.integers(1, vocab, size=6)
    pairs.append((s, np.concatenate([[BOS], t[:-1]]), t))

lr, num_steps = 0.5, 300
history = []
for step in range(num_steps):
    total = None
    for s, t_in, t_out in pairs:                       # 1. forward + 2. loss
        l_one = softmax_cross_entropy(model.forward_tensor(s, t_in), t_out)
        total = l_one if total is None else total + l_one
    loss = total * (1.0 / len(pairs))
    for p in params:                                   # 3. gradient
        p.grad = np.zeros_like(p.data)
    loss.backward()
    for p in params:                                   # 4. update(素朴な勾配降下)
        p.data -= lr * p.grad
    history.append(loss.data)
    if step in (0, 50, 100, num_steps - 1):
        print("    step {:>4}: loss = {:.6f}".format(step, float(loss.data)))

assert history[0] > 1.5                # 初期はほぼ当てずっぽう(ln 12 ≈ 2.48 付近)
assert history[-1] < 0.05              # 丸暗記完了
assert history[-1] < history[0] / 20   # 大きく下がった

# 丸暗記の確認: teacher forcing 下で argmax が正解列を完全再生する
for s, t_in, t_out in pairs:
    pred = np.argmax(model.forward_numpy(s, t_in), axis=-1)
    assert np.array_equal(pred, t_out)
print("(4) 丸暗記 OK: loss {:.3f} -> {:.4f}、2系列とも完全再生"
      .format(history[0], history[-1]))

print("integration_test: すべての assert を通過しました")
