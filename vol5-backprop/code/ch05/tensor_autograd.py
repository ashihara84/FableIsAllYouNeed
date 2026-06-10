# 第5巻 第5章 5.3: NumPy 行列版の最小 autograd
# ノードが持つ値を「数1個」(第4章の Value)から「行列1枚」(np.ndarray)に置き換えたもの。
# 設計は第4章と同一: 値と勾配と「親と演算」を覚え、backward() がグラフを逆順にたどる。
# 第6〜8巻はこのファイルを import して使う。
import numpy as np


def _unbroadcast(grad, shape):
    """ブロードキャストで膨らんだ勾配を、元の shape へ足し戻す。
    forward で (n, d) + (d,) のように複製された値は、backward では
    「複製された先の勾配の合計」を受け取る(第2巻5章: 道が複数なら足す)。"""
    # 次元の数が増えていたら、先頭の軸を合計して落とす
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # サイズ1の軸に沿って複製されていたら、その軸を合計して 1 に戻す
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """np.ndarray を1枚持つ計算グラフのノード。API は第4章の Value と同じ。"""

    def __init__(self, data, _children=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = tuple(_children)

    # --- 加算(ブロードキャスト対応。バイアス (n,d) + (d,) はここを通る) ---
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    __radd__ = __add__

    # --- 要素ごとの乗算(ブロードキャスト対応) ---
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __neg__(self):
        return self * (-1.0)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        return Tensor(other) + (-self)

    # --- 行列積(2次元どうしのみ。第3章で手導出した backward の式がそのまま入る) ---
    def __matmul__(self, other):
        assert self.data.ndim == 2 and other.data.ndim == 2  # 最小限: 2次元のみ対応
        out = Tensor(self.data @ other.data, (self, other))

        def _backward():
            self.grad += out.grad @ other.data.T   # ∂L/∂X = δ @ W^T
            other.grad += self.data.T @ out.grad   # ∂L/∂W = X^T @ δ(第3章3.3の式)

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), (self,))

        def _backward():
            self.grad += (self.data > 0.0) * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,))

        def _backward():
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,))

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,))

        def _backward():
            self.grad += out.grad  # スカラーの勾配が全要素にブロードキャストで配られる

        out._backward = _backward
        return out

    def mean(self):
        return self.sum() * (1.0 / self.data.size)

    def backward(self):
        """トポロジカル順にグラフをたどり、全ノードの .grad を埋める(第4章4.3と同じ)。"""
        topo = []
        visited = set()
        stack = [(self, False)]
        while stack:  # 再帰だと深いグラフで上限に当たるため、スタックで書いてある
            node, children_done = stack.pop()
            if children_done:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for child in node._prev:
                if id(child) not in visited:
                    stack.append((child, False))
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()


def softmax_cross_entropy(logits, targets):
    """softmax + cross-entropy(第4巻6章)を1つのノードに融合した数値安定版。
    logits: Tensor (n, K)、targets: 正解クラス番号の整数配列 (n,)。
    平均損失のスカラー Tensor を返す。backward は第4巻6章の手導出 (p - t) / n。"""
    targets = np.asarray(targets, dtype=int)
    n = logits.data.shape[0]
    z = logits.data - logits.data.max(axis=1, keepdims=True)  # 最大値シフト(第4巻6.2)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    out = Tensor(-log_probs[np.arange(n), targets].mean(), (logits,))

    def _backward():
        probs = np.exp(log_probs)
        probs[np.arange(n), targets] -= 1.0  # p - onehot(t)
        logits.grad += probs / n * out.grad

    out._backward = _backward
    return out


if __name__ == "__main__":
    # 動作確認の最小版(本格的な数値微分照合は test_tensor_autograd.py)
    rng = np.random.default_rng(42)
    X = Tensor(rng.standard_normal((4, 3)))
    W = Tensor(rng.standard_normal((3, 5)))
    b = Tensor(np.zeros(5))
    loss = softmax_cross_entropy((X @ W + b).relu() @ Tensor(rng.standard_normal((5, 2))),
                                 np.array([0, 1, 0, 1]))
    loss.backward()
    assert W.grad.shape == W.data.shape and not np.allclose(W.grad, 0.0)
    assert b.grad.shape == b.data.shape
    print("ok: forward/backward が一通り動きました(照合テストは test_tensor_autograd.py)")
