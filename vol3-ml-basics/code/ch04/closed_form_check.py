# 第3巻 第4章 4.4: 解析解との答え合わせ
# 1変数の線形回帰に限り、解析解は逆行列なしの式で書ける。
# 勾配降下(4.2)が同じ答えに着いていることを確認する。
import numpy as np


def make_data():
    """第2章と同一の合成データ(20人)。"""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 9, size=(20, 1))       # 勉強時間 (20, 1)
    noise = rng.normal(0, 6.0, size=(20, 1))
    y = 7.0 * X + 20.0 + noise                # 点数 (20, 1)
    return X.ravel(), y.ravel()               # 本章はスカラー w で進めるので (20,) に潰す


def train(X, y, w=0.0, b=0.0, lr=0.01, num_steps=5000):
    """4.2 の4拍子を関数に畳んだだけ。中身は一字も変わっていない。"""
    for _ in range(num_steps):
        y_hat = w * X + b                        # forward
        grad_w = 2.0 * np.mean((y_hat - y) * X)  # gradient(loss は更新に不要)
        grad_b = 2.0 * np.mean(y_hat - y)
        w = w - lr * grad_w                      # update
        b = b - lr * grad_b
    return w, b


X, y = make_data()

# 勾配降下の答え
w_gd, b_gd = train(X, y)

# 解析解(1変数の特例。多変数では逆行列が必要になる)
x_mean, y_mean = X.mean(), y.mean()
w_star = np.sum((X - x_mean) * (y - y_mean)) / np.sum((X - x_mean) ** 2)
b_star = y_mean - w_star * x_mean

print("勾配降下: w = {:.6f}, b = {:.6f}".format(w_gd, b_gd))
print("解析解  : w = {:.6f}, b = {:.6f}".format(w_star, b_star))
assert np.allclose([w_gd, b_gd], [w_star, b_star], atol=1e-6)
print("ok: 勾配降下は解析解と一致しました")
