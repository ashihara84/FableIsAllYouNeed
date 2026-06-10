# 第2巻 第5章: 連鎖律 — 小さな計算グラフの forward と微分の伝播
# 「微分は辺をたどる掛け算(分岐は足し算)」を、手で組んだグラフで確かめる。
# 検算には第1章の数値微分を使う(この習慣は第5巻の autograd テストに引き継がれる)。
import numpy as np


def numerical_diff(f, x, h=1e-5):
    """中心差分による数値微分(第1章で作った検算機)"""
    return (f(x + h) - f(x - h)) / (2 * h)


# --- 例1: まっすぐな鎖 y = (3x + 1)^2 ---
# グラフ: x → [×3] → u1 → [+1] → u2 → [2乗] → y

def forward_chain(x):
    u1 = 3 * x        # 辺の局所微分: du1/dx = 3
    u2 = u1 + 1       # 辺の局所微分: du2/du1 = 1
    y = u2 ** 2       # 辺の局所微分: dy/du2 = 2*u2
    return u1, u2, y  # 中間値も返す。backward が局所微分の計算に使うため


def backward_chain(x):
    u1, u2, y = forward_chain(x)   # 先に前向きに値を流す
    dy_du2 = 2 * u2                # 局所微分は forward の値から決まる
    du2_du1 = 1.0
    du1_dx = 3.0
    # 出力側から、辺をたどって掛けていく
    dy_du1 = dy_du2 * du2_du1
    dy_dx = dy_du1 * du1_dx
    return dy_dx


x = 1.0
u1, u2, y = forward_chain(x)
assert np.allclose((u1, u2, y), (3.0, 4.0, 16.0))   # (3*1+1)^2 = 16

grad_x = backward_chain(x)
assert np.allclose(grad_x, 24.0)   # 辺の上の 3 × 1 × 8 = 24

# x を変えても、展開して求めた式 18x + 6 とも、数値微分とも常に一致する
for x in [-2.0, 0.0, 0.5, 1.0, 3.0]:
    assert np.allclose(backward_chain(x), 18 * x + 6)
    assert np.allclose(backward_chain(x),
                       numerical_diff(lambda t: forward_chain(t)[2], x))


# --- 例2: 枝分かれのあるグラフ z = (x + y)・x ---
# x は「[+] 経由」と「[×] へ直接」の2本のパスで z に届く。合流は足し算。

def forward_diamond(x, y):
    a = x + y         # 局所微分: ∂a/∂x = 1, ∂a/∂y = 1
    z = a * x         # 局所微分: ∂z/∂a = x, ∂z/∂x(直接の辺)= a
    return a, z


def backward_diamond(x, y):
    a, z = forward_diamond(x, y)   # 先に前向きに値を流す
    dz_da = x                      # 掛け算ノードの局所微分は「相方の値」
    dz_dx_direct = a
    da_dx = 1.0
    da_dy = 1.0
    grad_x = dz_da * da_dx + dz_dx_direct   # パス2本: 掛けて、足す
    grad_y = dz_da * da_dy                  # パス1本
    return grad_x, grad_y


x, y = 2.0, 3.0
a, z = forward_diamond(x, y)
assert np.allclose((a, z), (5.0, 10.0))

grad_x, grad_y = backward_diamond(x, y)
assert np.allclose(grad_x, 7.0)   # パス1: 1×2、パス2: 5。足して 7
assert np.allclose(grad_y, 2.0)

# 展開した z = x^2 + xy の偏微分(∂z/∂x = 2x+y, ∂z/∂y = x)とも、数値微分とも一致
for x, y in [(2.0, 3.0), (-1.0, 0.5), (0.0, 4.0), (1.5, -2.0)]:
    grad_x, grad_y = backward_diamond(x, y)
    assert np.allclose(grad_x, 2 * x + y)
    assert np.allclose(grad_y, x)
    assert np.allclose(grad_x,
                       numerical_diff(lambda t: forward_diamond(t, y)[1], x))
    assert np.allclose(grad_y,
                       numerical_diff(lambda t: forward_diamond(x, t)[1], y))

# backward 1回で ∂z/∂x と ∂z/∂y が同時に出た = 勾配 ∇z(第4章)が一度に手に入る

print("ok: すべての assert を通過しました")
