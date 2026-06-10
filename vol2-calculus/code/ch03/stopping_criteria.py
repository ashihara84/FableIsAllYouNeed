# 第2巻 第3章 3.4: 止めどき — 収束判定とステップ数上限
def grad_f(x):
    """f(x) = x^2 の導関数。"""
    return 2.0 * x


def gradient_descent_until(grad, x0, lr, tol, max_steps):
    """|f'(x)| < tol になるまで降下する。max_steps は保険(必ず付ける)。

    返り値: (止まったときの x, 実行した更新回数)
    """
    x = x0
    for step in range(max_steps):
        g = grad(x)
        if abs(g) < tol:
            return x, step  # 谷底とみなして停止
        x = x - lr * g
    return x, max_steps     # 保険発動: 判定を満たさないまま打ち切り


x0 = 10.0
lr = 0.1

# --- tol を1桁ずつ厳しくすると、ステップ数はどう増えるか ---
print("f(x) = x^2, x0 = {}, lr = {}".format(x0, lr))
print("{:>8} {:>10} {:>14}".format("tol", "steps", "x"))
for tol in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
    x, steps = gradient_descent_until(grad_f, x0, lr, tol, max_steps=10000)
    print("{:>8.0e} {:>10} {:>14.6g}".format(tol, steps, x))

# tol = 1e-6 なら 76 ステップで止まり、|f'(x)| が確かに tol を下回っている
x, steps = gradient_descent_until(grad_f, x0, lr, tol=1e-6, max_steps=10000)
assert steps == 76
assert abs(grad_f(x)) < 1e-6

# 1桁厳しくするごとに、増えるステップ数はほぼ一定(等比収束なので)
_, s2 = gradient_descent_until(grad_f, x0, lr, tol=1e-2, max_steps=10000)
_, s6 = gradient_descent_until(grad_f, x0, lr, tol=1e-6, max_steps=10000)
_, s10 = gradient_descent_until(grad_f, x0, lr, tol=1e-10, max_steps=10000)
assert (s6 - s2) == (s10 - s6) == 41  # 4桁ぶんで41ステップずつ

# --- 保険が要る理由: η = 1.0(振動)では収束判定が永遠に満たされない ---
x, steps = gradient_descent_until(grad_f, x0, lr=1.0, tol=1e-6, max_steps=500)
assert steps == 500          # max_steps で打ち切られた
assert abs(x) == x0          # x は ±10 を往復したまま

print("すべての assert を通過: 収束判定とステップ数上限を確認できました")
