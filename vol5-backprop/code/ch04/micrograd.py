# 第5巻 第4章: ミニ autograd — Value クラス(シリーズの基盤モジュール)
# 第5巻5章・第6巻・第7巻・第8巻1章がこのファイルを import する。
# 依存は標準ライブラリのみ。本文(4.1〜4.3節)のコードを順につなげたもの
# + 演習問1の tanh(後の巻で使うため同梱)。
import math


class Value:
    """スカラー1個の値 .data と勾配 .grad を持ち、計算グラフを記録するノード"""

    def __init__(self, data, _parents=(), _op=""):
        self.data = float(data)
        self.grad = 0.0                  # まだ何も流れてきていない
        self._backward = lambda: None    # 葉ノード(入力・パラメータ)は何もしない
        self._parents = set(_parents)    # この値を作るのに使われたノード
        self._op = _op                   # この値を作った演算(デバッグ用の名札)

    def __repr__(self):
        return "Value(data={}, grad={})".format(self.data, self.grad)

    # --- 基本演算: forward で値を計算し、backward(局所勾配の伝え方)を1個ずつ生やす ---

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad        # 加算の局所勾配はどちらの入力にも 1
            other.grad += out.grad       # += なのは「パスが複数あれば足す」ため
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad   # 乗算の局所勾配は「相方の値」
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        # 指数は定数のみ。Value 同士の累乗は exp/log で書ける(演習問3)
        assert isinstance(k, (int, float)), "指数は int か float の定数のみ"
        out = Value(self.data ** k, (self,), "**{}".format(k))

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad   # (x^k)' = k x^(k-1)
        out._backward = _backward
        return out

    # --- 派生演算: 上の3つの組み合わせ。backward は書かなくても自動で正しくなる ---

    def __neg__(self):                   # -x
        return self * -1

    def __sub__(self, other):            # x - y
        return self + (-other)

    def __truediv__(self, other):        # x / y
        return self * other ** -1

    # --- 右側演算子: 2.0 + x のように左辺が普通の数のとき Python が呼ぶ ---

    def __radd__(self, other):           # 定数 + x
        return self + other

    def __rmul__(self, other):           # 定数 * x
        return self * other

    def __rsub__(self, other):           # 定数 - x
        return (-self) + other

    def __rtruediv__(self, other):       # 定数 / x
        return self ** -1 * other

    # --- 非線形関数 ---

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad     # (e^x)' = e^x。forward の結果を再利用
        out._backward = _backward
        return out

    def log(self):
        # 自然対数。定義域は data > 0(損失関数で使うときは中身を正に保つこと)
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad   # (log x)' = 1/x
        out._backward = _backward
        return out

    def tanh(self):
        # 演習問1の略解。tanh' = 1 - tanh^2 なので forward の結果 t だけで書ける
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad
        out._backward = _backward
        return out

    # --- backward: グラフをトポロジカルソートし、出力から逆順に勾配を流す ---

    def backward(self):
        topo = []                        # 「親が必ず自分より前に来る」順のノード列
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v._parents:
                    build(p)             # 親を先に積む
                topo.append(v)           # 親が全員積まれてから自分を積む
        build(self)

        self.grad = 1.0                  # dL/dL = 1 から伝播を始める
        for v in reversed(topo):         # 出力側から葉に向かって逆順に
            v._backward()
