# 第5巻 第4章 4.4: micrograd の全演算を数値微分と照合するテスト
# 第2巻1章で作った検算機(中心差分)との突き合わせ — シリーズの習慣の集大成。
# micrograd.py と同じく標準ライブラリのみで動かす(基盤モジュールのテストは
# numpy のない環境でも回せるように。数値比較は np.allclose の代わりに math.isclose)。
import math

from micrograd import Value


def numerical_diff(f, x, h=1e-5):
    """中心差分による数値微分(第2巻1章): (f(x+h) - f(x-h)) / 2h"""
    return (f(x + h) - f(x - h)) / (2 * h)


def isclose(a, b):
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


def check_unary(f, xs, name):
    """f は Value 1個を受けて Value 1個を返す式。
    backward() の勾配と、forward だけを使った数値微分が、全点で一致するか"""
    for x in xs:
        v = Value(x)
        f(v).backward()
        num = numerical_diff(lambda t: f(Value(t)).data, x)
        assert isclose(v.grad, num), (name, x, v.grad, num)
    print("ok:", name)


pts = [-2.0, -0.7, 0.5, 1.3, 3.0]        # 標準の評価点(relu の折れ目 0 は外す)
pos = [0.5, 1.3, 3.0]                    # log と小数指数の pow 用(正の点のみ)
nonzero = [-2.0, -0.7, 0.5, 3.0]         # 除算・負の指数用(0 を外す)

# --- 演算ごとの照合 ---
check_unary(lambda v: v + 2.5, pts, "__add__ (Value + 定数)")
check_unary(lambda v: 2.5 + v, pts, "__radd__ (定数 + Value)")
check_unary(lambda v: v + v, pts, "__add__ (同じ変数を2回使う → 勾配が累積して 2)")
check_unary(lambda v: v * 3.0, pts, "__mul__ (Value × 定数)")
check_unary(lambda v: -1.5 * v, pts, "__rmul__ (定数 × Value)")
check_unary(lambda v: v * v * v, pts, "__mul__ (x*x*x: パス3本の合流 → 3x^2)")
check_unary(lambda v: -v, pts, "__neg__")
check_unary(lambda v: v - 4.0, pts, "__sub__")
check_unary(lambda v: 4.0 - v, pts, "__rsub__")
check_unary(lambda v: v / 2.0, pts, "__truediv__")
check_unary(lambda v: 2.0 / v, nonzero, "__rtruediv__")
check_unary(lambda v: v ** 3, pts, "__pow__ (整数指数)")
check_unary(lambda v: v ** -2, nonzero, "__pow__ (負の指数)")
check_unary(lambda v: v ** 0.5, pos, "__pow__ (小数指数)")
check_unary(lambda v: v.relu(), pts, "relu")
check_unary(lambda v: v.exp(), pts, "exp")
check_unary(lambda v: v.log(), pos, "log")
check_unary(lambda v: v.tanh(), pts, "tanh")


# --- 全演算を混ぜた合成式でも一致するか(グラフが深く・枝分かれしても壊れない) ---
def f_composite(v):
    a = (3.0 * v + 1.0).relu() + (-v).relu()
    b = (v ** 2 + 1.0).log()                  # 中身は常に正なので log が安全
    return (v / 2.0).tanh() * (1.0 - v) + (b - a) / (v.exp() + 2.0)


check_unary(f_composite, pts, "合成式(全演算入り)")


# --- 多変数: backward 1回で全偏微分が同時に出る(第2巻5章のひし形グラフ) ---
def g(x, y):
    return (x + y) * x                        # 展開すると x^2 + xy


for xv, yv in [(2.0, 3.0), (-1.0, 0.5), (1.5, -2.0)]:
    x, y = Value(xv), Value(yv)
    g(x, y).backward()
    assert isclose(x.grad, 2 * xv + yv)       # 解析解 ∂z/∂x = 2x + y
    assert isclose(y.grad, xv)                # 解析解 ∂z/∂y = x
    assert isclose(x.grad, numerical_diff(lambda t: g(Value(t), Value(yv)).data, xv))
    assert isclose(y.grad, numerical_diff(lambda t: g(Value(xv), Value(t)).data, yv))
print("ok: 2変数(ひし形グラフ。解析解・数値微分の両方と一致)")


# --- 集大成: 小さな2層MLP(2-2-1, tanh)の損失を、全9パラメータについて照合 ---
def mlp_loss(params, x1, x2, target):
    w11, w12, w21, w22, b1, b2, u1, u2, c = params
    h1 = (x1 * w11 + x2 * w12 + b1).tanh()    # 隠れ層 1 ユニット目
    h2 = (x1 * w21 + x2 * w22 + b2).tanh()    # 隠れ層 2 ユニット目
    out = h1 * u1 + h2 * u2 + c               # 出力層(線形)
    return (out - target) ** 2                # 二乗誤差


raw = [0.5, -0.3, 0.8, -0.5, 0.1, -0.2, 0.7, 0.9, -0.1]
params = [Value(p) for p in raw]
mlp_loss(params, 1.0, -2.0, 0.5).backward()   # backward は1回だけ

for i, p in enumerate(params):
    def f_i(t, i=i):
        ps = [Value(r) for r in raw]
        ps[i] = Value(t)                      # i 番目のパラメータだけ動かす
        return mlp_loss(ps, 1.0, -2.0, 0.5).data
    num = numerical_diff(f_i, raw[i])
    assert isclose(p.grad, num), (i, p.grad, num)
print("ok: 2層MLPの全9パラメータの勾配が数値微分と一致")

print()
print("ok: すべての assert を通過しました")
