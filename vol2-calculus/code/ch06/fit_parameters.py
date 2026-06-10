# 第2巻 第6章 6.2: パラメータ2個の関数をデータに合わせて勾配降下で調整する
# 入力 X と正解 y は固定。動かすのはパラメータ w, b だけ。
# (この章のパラメータはスカラーなので、第1巻の規律に従い小文字 w で書く)
import numpy as np

# --- データ(5件)。学習の間、この値は一切変更しない ---
X = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
y = np.array([-1.0, 1.0, 3.0, 5.0, 7.0])   # 実は y = 2x - 1 から作った(未知のつもりで扱う)


def predict(X, w, b):
    """パラメータ w, b による予測: y_hat = w * x + b(全データ一斉)"""
    return w * X + b


def error_total(X, y, w, b):
    """データとのズレの合計: 各点のズレの2乗を全件足したもの"""
    return np.sum((predict(X, w, b) - y) ** 2)


def gradients(X, y, w, b):
    """E の w, b それぞれによる偏微分(第5章の連鎖律で手で導出した式)"""
    diff = predict(X, w, b) - y          # 中の関数の値
    grad_w = np.sum(2 * diff * X)        # 外の微分 2*diff × 中の微分 x_i
    grad_b = np.sum(2 * diff)            # 外の微分 2*diff × 中の微分 1
    return grad_w, grad_b


# --- 検算: 解析勾配を数値微分と突き合わせる(第1章の習慣) ---
def numerical_grad(f, v, h=1e-6):
    return (f(v + h) - f(v - h)) / (2 * h)


w0, b0 = 0.5, 0.0
grad_w, grad_b = gradients(X, y, w0, b0)
assert np.allclose(grad_w, numerical_grad(lambda v: error_total(X, y, v, b0), w0))
assert np.allclose(grad_b, numerical_grad(lambda v: error_total(X, y, w0, v), b0))

# --- 勾配降下でパラメータを調整する。第3章のアルゴリズムそのまま、動かす対象が違うだけ ---
w, b = 0.0, 0.0
lr = 0.01
E_history = [error_total(X, y, w, b)]
for step in range(1000):
    grad_w, grad_b = gradients(X, y, w, b)
    w -= lr * grad_w
    b -= lr * grad_b
    E_history.append(error_total(X, y, w, b))

# E は最初の10ステップで単調に減っている
for t in range(10):
    assert E_history[t + 1] < E_history[t]

# データを作った規則 w=2, b=-1 をほぼ言い当てている
assert np.allclose([w, b], [2.0, -1.0], atol=1e-3)
assert error_total(X, y, w, b) < 1e-6

print("ok: すべての assert を通過しました")
print(f"  推定: w = {w:.4f}, b = {b:.4f}  (正解: w = 2, b = -1)")
print(f"  ズレの合計: {E_history[0]:.1f} -> {E_history[-1]:.2e}")
