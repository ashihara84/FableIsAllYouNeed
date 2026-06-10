# 第7巻 第6章 6.4: 埋め込み層と出力 head(論文 3.4 Embeddings and Softmax)
# - Embedding: トークン番号の列 (seq,) → (seq, d_model)。lookup は E の行の取り出し
#   (第6巻3章)。論文の指定どおり、取り出した行を √d_model 倍して返す。
# - output_logits: decoder 出力 (seq, d_model) → 語彙スコア (seq, vocab)。
#   重み行列は埋め込みの E をそのまま使う(weight sharing): logits = X @ E^T。
# 第8巻がこのファイルを import して組み立てに使う。
import os
import sys

import numpy as np

_VOL5 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "..", "vol5-backprop", "code", "ch05")
sys.path.insert(0, os.path.normpath(_VOL5))
from tensor_autograd import Tensor  # noqa: E402(第5巻5章の autograd)


class Embedding:
    """論文 3.4 の learned embeddings。E (vocab, d_model) を1枚持つ。"""

    def __init__(self, vocab_size, d_model, rng):
        # 成分の分散はおよそ 1/d_model(行ベクトルのノルムが約1になるスケール)
        self.E = Tensor(rng.standard_normal((vocab_size, d_model)) / np.sqrt(d_model))
        self.d_model = d_model

    def __call__(self, ids):
        """ids: トークン番号の整数配列 (seq,) → Tensor (seq, d_model)。"""
        ids = np.asarray(ids, dtype=int)
        assert ids.ndim == 1  # 下流の行列積が2次元限定。バッチは (batch*seq,) に平らに
        E = self.E
        scale = np.sqrt(self.d_model)  # 論文 3.4: "we multiply those weights by √d_model"
        out = Tensor(E.data[ids] * scale, (E,))

        def _backward():
            # 同じトークンが複数の位置に現れたら、その行の勾配は全位置ぶんの合計
            # (第2巻5章: 道が複数なら足す)。np.add.at は重複 index でも全部足す
            # (E.grad[ids] += ... は重複分を1回しか足さないので使えない)
            np.add.at(E.grad, ids, out.grad * scale)

        out._backward = _backward
        return out

    def params(self):
        return [self.E]


def output_logits(X, E):
    """出力 head(pre-softmax linear)。logits = X @ E^T。

    X: Tensor (seq, d_model) — decoder の出力。E: Tensor (vocab, d_model) — 埋め込み行列。
    戻り値 (seq, vocab)。位置 i・トークン t の成分は「位置 i の隠れ状態と
    トークン t の埋め込みベクトルの内積」(第1巻2章: 内積 = 類似度)。
    確率にするには各行を softmax に通す(訓練では第5巻の softmax_cross_entropy が一括処理)。
    """
    out = Tensor(X.data @ E.data.T, (X, E))

    def _backward():
        X.grad += out.grad @ E.data    # ∂L/∂X = δ @ (E^T)^T = δ @ E
        E.grad += out.grad.T @ X.data  # ∂L/∂E = δ^T @ X(第5巻3章の式の転置版)

    out._backward = _backward
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    vocab, d_model, seq = 11, 8, 5
    emb = Embedding(vocab, d_model, rng)
    ids = np.array([3, 7, 3, 0, 9])  # トークン3をわざと2回使う

    # --- 埋め込み側 ---
    X = emb(ids)
    assert X.data.shape == (seq, d_model)

    # lookup = one-hot @ E(第6巻3章の等式)。√d_model 倍も込みで一致を確認
    onehot = np.zeros((seq, vocab))
    onehot[np.arange(seq), ids] = 1.0
    assert np.allclose(X.data, onehot @ emb.E.data * np.sqrt(d_model))

    # --- 出力 head 側(weight sharing: 同じ E を渡す)---
    logits = output_logits(X, emb.E)
    assert logits.data.shape == (seq, vocab)

    # 共有の確認: 入力側と出力側のパラメータを合わせても、行列は E の1枚だけ
    assert emb.params() == [emb.E]

    # --- backward: E は「埋め込み」と「出力射影」の2役ぶんの勾配を受け取る ---
    loss = (logits * logits).sum()
    loss.backward()
    assert emb.E.grad.shape == (vocab, d_model)
    # ids に登場したトークン(0,3,7,9)の行は、lookup 側の経路からも勾配が来る
    assert not np.allclose(emb.E.grad[3], 0.0)
    # ids に登場しないトークン(例: 5)の行も、出力射影側の経路から勾配が来る
    assert not np.allclose(emb.E.grad[5], 0.0)

    print("ok: 埋め込みと出力 head(論文3.4)のテストにすべて通りました")
