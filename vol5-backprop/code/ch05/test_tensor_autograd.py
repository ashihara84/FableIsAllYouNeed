# 第5巻 第5章 5.3: tensor_autograd.py の数値微分照合テスト
# 第4章4.4と同じ流儀: すべての演算について、autograd の勾配を中心差分と突き合わせる。
# 実行: python3 test_tensor_autograd.py
import numpy as np

from tensor_autograd import Tensor, softmax_cross_entropy

rng = np.random.default_rng(42)


def numerical_grad(f, arrays, idx, h=1e-6):
    """arrays[idx] の各要素を ±h 動かして f(arrays) の変化を測る(第2巻1章の中心差分)。"""
    a = arrays[idx]
    grad = np.zeros_like(a)
    flat = a.reshape(-1)
    grad_flat = grad.reshape(-1)
    for i in range(flat.size):
        orig = flat[i]
        flat[i] = orig + h
        f_plus = f(arrays)
        flat[i] = orig - h
        f_minus = f(arrays)
        flat[i] = orig
        grad_flat[i] = (f_plus - f_minus) / (2 * h)
    return grad


def check(name, f_tensor, f_numpy, shapes, positive=False):
    """f_tensor: Tensor のリスト -> スカラー Tensor。f_numpy: ndarray のリスト -> float。
    全入力について autograd の勾配と数値微分を照合する。"""
    arrays = [rng.standard_normal(s) for s in shapes]
    if positive:  # log 用: 入力を正に寄せる
        arrays = [np.abs(a) + 0.5 for a in arrays]
    tensors = [Tensor(a.copy()) for a in arrays]
    out = f_tensor(tensors)
    out.backward()
    for idx, t in enumerate(tensors):
        num = numerical_grad(f_numpy, [a.copy() for a in arrays], idx)
        assert np.allclose(t.grad, num, rtol=1e-4, atol=1e-7), (
            "{} の入力{}: autograd と数値微分が不一致\n{}\n{}".format(name, idx, t.grad, num))
    print("ok: {:<28} autograd = 数値微分".format(name))


# --- 各演算を1つずつ照合する ---
check("add(同形)",
      lambda t: (t[0] + t[1]).sum(),
      lambda a: (a[0] + a[1]).sum(), [(3, 4), (3, 4)])

check("add(ブロードキャスト)",  # (3,4) + (4,): バイアス加算の形
      lambda t: ((t[0] + t[1]) * t[0]).sum(),
      lambda a: ((a[0] + a[1]) * a[0]).sum(), [(3, 4), (4,)])

check("add(サイズ1の軸)",
      lambda t: ((t[0] + t[1]) * t[1]).sum(),
      lambda a: ((a[0] + a[1]) * a[1]).sum(), [(3, 1), (3, 4)])

check("mul(要素ごと)",
      lambda t: (t[0] * t[1]).sum(),
      lambda a: (a[0] * a[1]).sum(), [(3, 4), (3, 4)])

check("mul(ブロードキャスト)",
      lambda t: (t[0] * t[1]).sum(),
      lambda a: (a[0] * a[1]).sum(), [(3, 4), (4,)])

check("matmul",
      lambda t: (t[0] @ t[1]).sum(),
      lambda a: (a[0] @ a[1]).sum(), [(3, 4), (4, 5)])

check("matmul(2乗を絡める)",  # 勾配が定数にならないようにした版
      lambda t: ((t[0] @ t[1]) * (t[0] @ t[1])).sum(),
      lambda a: ((a[0] @ a[1]) ** 2).sum(), [(3, 4), (4, 5)])

check("relu",
      lambda t: (t[0].relu() * t[1]).sum(),
      lambda a: (np.maximum(a[0], 0.0) * a[1]).sum(), [(3, 4), (3, 4)])

check("log", lambda t: t[0].log().sum(),
      lambda a: np.log(a[0]).sum(), [(3, 4)], positive=True)

check("exp", lambda t: t[0].exp().sum(),
      lambda a: np.exp(a[0]).sum(), [(3, 4)])

check("sum", lambda t: (t[0].sum() * t[0].sum()),
      lambda a: float(a[0].sum() ** 2), [(3, 4)])

check("mean", lambda t: (t[0].mean() * t[0].mean()),
      lambda a: float(a[0].mean() ** 2), [(3, 4)])

check("同じ Tensor を2回使う",  # 勾配の累積(+=)の確認
      lambda t: (t[0] * t[0] + t[0]).sum(),
      lambda a: (a[0] * a[0] + a[0]).sum(), [(3, 4)])

check("2層MLP(全部入り)",
      lambda t: ((t[0] @ t[1] + t[2]).relu() @ t[3] + t[4]).exp().mean(),
      lambda a: np.exp(np.maximum(a[0] @ a[1] + a[2], 0.0) @ a[3] + a[4]).mean(),
      [(5, 3), (3, 4), (4,), (4, 2), (2,)])

# --- softmax_cross_entropy の照合 ---
targets = np.array([0, 2, 1, 2, 0])
check("softmax_cross_entropy",
      lambda t: softmax_cross_entropy(t[0], targets),
      lambda a: -np.mean((a[0] - np.log(np.exp(a[0]).sum(axis=1, keepdims=True)))
                         [np.arange(5), targets]),
      [(5, 3)])

# 数値安定性: 巨大な logits でも nan / inf を出さない
big = Tensor(np.array([[1000.0, 0.0, -1000.0], [500.0, 500.0, 500.0]]))
loss_big = softmax_cross_entropy(big, np.array([0, 1]))
loss_big.backward()
assert np.isfinite(loss_big.data) and np.all(np.isfinite(big.grad))
print("ok: softmax_cross_entropy        logits=±1000 でも有限のまま(数値安定)")

print("ok: すべての assert を通過しました")
