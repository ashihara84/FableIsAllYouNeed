# 第2巻 第3章 演習3(略解): 凸でない関数で局所解にハマる体験
def f(x):
    """凸でない関数。谷が2つある。"""
    return x ** 4 - 2.0 * x ** 2 + 0.5 * x


def grad_f(x):
    """f'(x) = 4x^3 − 4x + 0.5(手計算)。"""
    return 4.0 * x ** 3 - 4.0 * x + 0.5


def gradient_descent(grad, x0, lr, n_steps):
    x = x0
    for _ in range(n_steps):
        x = x - lr * grad(x)
    return x


lr = 0.02
n_steps = 200

x_left = gradient_descent(grad_f, x0=-1.5, lr=lr, n_steps=n_steps)
x_right = gradient_descent(grad_f, x0=+1.5, lr=lr, n_steps=n_steps)

print("x0 = -1.5 からの着地点: x = {:.4f}, f(x) = {:.4f}".format(x_left, f(x_left)))
print("x0 = +1.5 からの着地点: x = {:.4f}, f(x) = {:.4f}".format(x_right, f(x_right)))

# どちらの着地点も「傾きほぼ 0」— アルゴリズムはどちらも"谷底"と報告する
assert abs(grad_f(x_left)) < 1e-6
assert abs(grad_f(x_right)) < 1e-6

# だが着いた谷は別物: 左の谷(約 -1.06)と右の谷(約 0.93)
assert -1.1 < x_left < -1.0
assert 0.9 < x_right < 1.0

# 右の谷は局所解: 左の谷より f の値が高い(差は約 1.0)
assert f(x_right) - f(x_left) > 0.9

print("assert を通過: 出発点の違いだけで、別の谷に着地しました")
