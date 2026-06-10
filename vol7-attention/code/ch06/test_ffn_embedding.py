# 第7巻 第6章 6.4: FFN と埋め込み・出力 head の照合テスト
# 「埋め込み → FFN → 出力 head → cross-entropy」のミニ・パイプラインを組み、
# autograd の勾配を数値微分(第2巻)と突き合わせる。
# E は weight sharing で2か所(入口と出口)から使われる — 数値微分は
# 「E を少し動かすと損失がどう動くか」を測るだけなので、2役ぶんの効果を自動で合算する。
# autograd 側も同じ値を出せば、共有の backward は正しい。
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding import Embedding, output_logits  # noqa: E402
from position_wise_ffn import PositionwiseFFN, Tensor  # noqa: E402
from tensor_autograd import softmax_cross_entropy  # noqa: E402(第5巻5章)

assert Tensor is not None  # import 経路の確認(vol5 → vol7 の一方向依存)


def forward_loss(emb, ffn, ids, targets):
    """ids (seq,) → 埋め込み → FFN → 共有 E で logits → 平均 cross-entropy。"""
    X = emb(ids)                       # (seq, d_model)
    H = ffn(X)                         # (seq, d_model)
    logits = output_logits(H, emb.E)   # (seq, vocab)
    return softmax_cross_entropy(logits, targets)


def numerical_grad(param, compute_loss, eps=1e-6):
    """中心差分 (L(θ+ε) − L(θ−ε)) / 2ε を全成分で計算する(第2巻1章)。"""
    grad = np.zeros_like(param.data)
    it = np.nditer(param.data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = param.data[idx]
        param.data[idx] = orig + eps
        loss_plus = compute_loss()
        param.data[idx] = orig - eps
        loss_minus = compute_loss()
        param.data[idx] = orig
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()
    return grad


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    vocab, d_model, d_ff, seq = 11, 6, 12, 5
    emb = Embedding(vocab, d_model, rng)
    ffn = PositionwiseFFN(d_model, d_ff, rng)
    ids = np.array([3, 7, 3, 0, 9])        # 同じトークンの重複もテストに含める
    targets = np.array([7, 3, 0, 9, 1])    # 「次のトークン」の正解(値は何でもよい)

    # --- autograd の勾配 ---
    loss = forward_loss(emb, ffn, ids, targets)
    loss.backward()
    assert loss.data.shape == ()  # 損失はスカラー

    # --- 数値微分との照合: E(2役ぶん)と FFN の全パラメータ ---
    compute = lambda: forward_loss(emb, ffn, ids, targets).data  # noqa: E731
    for param, name in [(emb.E, "E"), (ffn.W1, "W1"), (ffn.b1, "b1"),
                        (ffn.W2, "W2"), (ffn.b2, "b2")]:
        num = numerical_grad(param, compute)
        assert np.allclose(param.grad, num, atol=1e-7), name
        print("ok: grad_%s が数値微分と一致(最大誤差 %.2e)"
              % (name, np.abs(param.grad - num).max()))

    # --- weight sharing の検証: 共有をほどくと、勾配は「2役の和」に分解される ---
    shared_grad = emb.E.grad.copy()  # 共有1枚の E が受け取った勾配(上で計算済み)
    emb_in = Embedding(vocab, d_model, rng)   # 入口専用の E(値は共有版と同じ)
    emb_out = Embedding(vocab, d_model, rng)  # 出口専用の E(同上)
    emb_in.E.data[:] = emb.E.data
    emb_out.E.data[:] = emb.E.data
    H = ffn(emb_in(ids))
    loss2 = softmax_cross_entropy(output_logits(H, emb_out.E), targets)
    loss2.backward()
    assert np.allclose(loss2.data, loss.data)  # forward は共有版と完全に同じ
    # 共有版の grad_E = 入口役の勾配 + 出口役の勾配(第2巻5章: 道が複数なら足す)
    assert np.allclose(emb_in.E.grad + emb_out.E.grad, shared_grad)
    print("ok: 共有1枚の grad_E = 入口役 + 出口役 の和に一致")

    print("all tests passed")
