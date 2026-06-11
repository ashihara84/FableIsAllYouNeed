# 第8巻 第1章 1.2: 第5巻 Tensor に足りない演算の補完部品
# 第5巻・第7巻のファイルは変更しない(import のみ)という規律のため、
# 組み立てに必要だが tensor_autograd.py に無い演算はここで補う。
# backward の流儀は第5巻と同じ: 出力ノードに「親への勾配の配り方」を1つずつ書く。
# 実行: python3 tensor_ops.py — 全演算を数値微分(第2巻1章)で検算する。
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "vol5-backprop", "code", "ch05")))
sys.path.insert(0, os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "vol7-attention", "code", "ch03")))
from tensor_autograd import Tensor  # noqa: E402(第5巻5章)
from attention import softmax, NEG_INF  # noqa: E402(第7巻3章。forward を共用する)


def t_transpose(X):
    """転置。forward が転置なら、backward も勾配を転置して返すだけ。"""
    out = Tensor(X.data.T, (X,))

    def _backward():
        X.grad += out.grad.T

    out._backward = _backward
    return out


def t_cols(X, j0, j1):
    """列の切り出し X[:, j0:j1](multi-head の「頭に裂く」用 — 第7巻4章)。
    backward は切り出した場所にだけ勾配を戻す(他の列の勾配は別の頭が運ぶ)。"""
    out = Tensor(X.data[:, j0:j1], (X,))

    def _backward():
        X.grad[:, j0:j1] += out.grad

    out._backward = _backward
    return out


def t_concat_cols(parts):
    """列方向の連結 = 論文の Concat(head_1, ..., head_h)。t_cols の逆操作。"""
    out = Tensor(np.concatenate([p.data for p in parts], axis=1), tuple(parts))

    def _backward():
        j = 0
        for p in parts:
            w = p.data.shape[1]
            p.grad += out.grad[:, j:j + w]
            j += w

    out._backward = _backward
    return out


def t_masked_softmax(Z, mask=None):
    """mask(True = 見てよい)を掛けてから行ごとに softmax。
    forward は第7巻3章 attention.py と同じ式(softmax と NEG_INF を共用)。
    backward は第4巻6章の手導出: dZ = A ⊙ (dA − Σ_k dA_k A_k)。
    mask された位置は A = 0 なので、勾配も自動的に 0 になる(別処理は不要)。"""
    z = Z.data if mask is None else np.where(mask, Z.data, NEG_INF)
    A = softmax(z, axis=-1)
    out = Tensor(A, (Z,))

    def _backward():
        dA = out.grad
        Z.grad += A * (dA - (dA * A).sum(axis=-1, keepdims=True))

    out._backward = _backward
    return out


def t_layer_norm(X, gamma, beta, eps=1e-5):
    """layer norm の Tensor 版。forward は第5巻6.3(= 第7巻2章 stack_skeleton)と同じ式、
    backward も第5巻6.3 で数値微分と照合済みの式をそのまま使う。"""
    mu = X.data.mean(axis=-1, keepdims=True)
    var = X.data.var(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (X.data - mu) * inv_std
    out = Tensor(gamma.data * x_hat + beta.data, (X, gamma, beta))

    def _backward():
        # mu と var も x の関数なので、x への勾配には補正項が2つ付く(第5巻6.3)
        dx_hat = out.grad * gamma.data
        X.grad += inv_std * (dx_hat
                             - dx_hat.mean(axis=-1, keepdims=True)
                             - x_hat * (dx_hat * x_hat).mean(axis=-1, keepdims=True))
        gamma.grad += (out.grad * x_hat).sum(axis=0)
        beta.grad += out.grad.sum(axis=0)

    out._backward = _backward
    return out


if __name__ == "__main__":
    # 全演算を数値微分で検算する(第5巻4章で micrograd にやったのと同じ儀式)
    rng = np.random.default_rng(42)

    def numerical_grad(param, loss_fn, eps=1e-6):
        g = np.zeros_like(param)
        flat, gf = param.reshape(-1), g.reshape(-1)
        for i in range(flat.size):
            old = flat[i]
            flat[i] = old + eps
            fp = loss_fn()
            flat[i] = old - eps
            fm = loss_fn()
            flat[i] = old
            gf[i] = (fp - fm) / (2 * eps)
        return g

    def check(name, tensors, build_loss):
        """build_loss() が組む損失について、全 tensors の autograd 勾配を数値微分と照合"""
        loss = build_loss()
        for t in tensors:
            t.grad = np.zeros_like(t.data)
        loss.backward()
        for t in tensors:
            num = numerical_grad(t.data, lambda: build_loss().data)
            assert np.allclose(t.grad, num, atol=1e-5), name
        print("  ok: " + name)

    n, d, h = 4, 8, 2
    R1 = rng.standard_normal((d, n))   # 損失 L = Σ(out ⊙ R) の係数(検算の定石)
    R2 = rng.standard_normal((n, 3))
    R3 = rng.standard_normal((n, n))
    R4 = rng.standard_normal((n, d))

    X = Tensor(rng.standard_normal((n, d)))
    check("t_transpose", [X], lambda: (t_transpose(X) * R1).sum())
    check("t_cols", [X], lambda: (t_cols(X, 2, 5) * R2).sum())

    A_, B_ = Tensor(rng.standard_normal((n, 3))), Tensor(rng.standard_normal((n, 5)))
    R5 = rng.standard_normal((n, 8))
    check("t_concat_cols", [A_, B_], lambda: (t_concat_cols([A_, B_]) * R5).sum())

    Z = Tensor(rng.standard_normal((n, n)))
    mask = np.tril(np.ones((n, n), dtype=bool))   # causal mask で検算
    check("t_masked_softmax", [Z], lambda: (t_masked_softmax(Z, mask) * R3).sum())
    # mask された位置(未来)の勾配は厳密に 0
    assert np.all(Z.grad[np.triu_indices(n, k=1)] == 0.0)

    gamma, beta = Tensor(rng.standard_normal(d)), Tensor(rng.standard_normal(d))
    check("t_layer_norm", [X, gamma, beta],
          lambda: (t_layer_norm(X, gamma, beta) * R4).sum())

    print("tensor_ops: すべての assert を通過しました")
