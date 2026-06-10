# 第7巻 第6章 6.1: Position-wise Feed-Forward Network(論文 3.3 式(2))
# FFN(x) = max(0, x W1 + b1) W2 + b2 — ReLU を挟んだ2層 linear(第5巻2章の MLP)。
# "position-wise" の実体: (seq, d_model) の各行に、同じ W1, b1, W2, b2 を独立に適用する。
# 2次元の行列積は最初から「行ごとに同じ変換」なので、特別な実装は何も要らない。
# 第8巻がこのファイルを import して組み立てに使う。
import os
import sys

import numpy as np

_VOL5 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "..", "vol5-backprop", "code", "ch05")
sys.path.insert(0, os.path.normpath(_VOL5))
from tensor_autograd import Tensor  # noqa: E402(第5巻5章の autograd)


class PositionwiseFFN:
    """論文 3.3 式(2)。入力 X (seq, d_model) → 出力 (seq, d_model)。

    第5巻の Tensor の行列積は2次元どうし限定なので、入力は2次元に限る。
    バッチを流すときは (batch*seq, d_model) に平らにしてから渡す
    (position-wise なので、行をどう束ねても結果は行ごとに同じ)。
    """

    def __init__(self, d_model, d_ff, rng):
        # 初期化のスケールは第5巻6.6の議論どおり: ReLU の前は He(√(2/入力次元))
        self.W1 = Tensor(rng.standard_normal((d_model, d_ff)) * np.sqrt(2.0 / d_model))
        self.b1 = Tensor(np.zeros(d_ff))
        self.W2 = Tensor(rng.standard_normal((d_ff, d_model)) * np.sqrt(1.0 / d_ff))
        self.b2 = Tensor(np.zeros(d_model))

    def __call__(self, X):
        # 式(2)と1対1対応のこの1行が、実装のすべて
        return (X @ self.W1 + self.b1).relu() @ self.W2 + self.b2

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    d_model, d_ff, seq = 6, 24, 5  # d_ff = 4 × d_model(論文の 512 → 2048 と同じ比率)
    ffn = PositionwiseFFN(d_model, d_ff, rng)
    X = rng.standard_normal((seq, d_model))
    Y = ffn(Tensor(X))

    # (1) shape: 入口と出口は同じ (seq, d_model)(residual で足すための条件 — 第2章)
    assert Y.data.shape == (seq, d_model)

    # (2) "separately": 各行を1本ずつ流しても、まとめて流しても同じ結果
    for i in range(seq):
        y_i = ffn(Tensor(X[i:i + 1]))
        assert np.allclose(y_i.data, Y.data[i:i + 1])

    # (3) "identically": 行を並べ替えて流すと、出力も同じ並べ替えになるだけ
    perm = rng.permutation(seq)
    Y_perm = ffn(Tensor(X[perm]))
    assert np.allclose(Y_perm.data, Y.data[perm])

    # (4) backward が4つのパラメータ全部に流れる(第8巻で学習させるための最低条件)
    loss = (Y * Y).sum()
    loss.backward()
    for p in ffn.params():
        assert p.grad.shape == p.data.shape
        assert not np.allclose(p.grad, 0.0)

    print("ok: position-wise FFN(式2)のテストにすべて通りました")
