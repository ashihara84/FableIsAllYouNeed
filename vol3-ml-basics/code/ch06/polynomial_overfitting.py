# 第3巻 第6章 6.3: 多項式回帰で過学習を起こす
# 次数を上げると訓練lossは下がり続けるが、検証lossは途中で反転する
import numpy as np

rng = np.random.default_rng(42)


# --- データ生成: 正解は3次多項式 + ノイズ(神の視点) ---
def true_f(x):
    return 0.5 + 1.0 * x - 2.0 * x**2 + 3.0 * x**3


n_train, n_val = 20, 100
noise = 0.4
x_train = rng.uniform(-1, 1, size=n_train)
y_train = true_f(x_train) + rng.normal(0, noise, size=n_train)
x_val = rng.uniform(-1, 1, size=n_val)
y_val = true_f(x_val) + rng.normal(0, noise, size=n_val)


# --- 特徴量: x を [x^1, ..., x^degree] に広げる(degree=0 なら空) ---
def poly_features(x, degree):
    if degree == 0:
        return np.zeros((len(x), 0))  # 列ゼロ: モデルは b だけになる
    return np.stack([x**k for k in range(1, degree + 1)], axis=1)


def mse(y_pred, y):
    return np.mean((y_pred - y) ** 2)


# --- 訓練: 第4章と同じ4拍子(forward → loss → gradient → update) ---
def fit_gd(X, y, lr=0.1, steps=1000000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(steps):
        err = X @ w + b - y                  # forward と残差
        grad_w = 2.0 / n * (X.T @ err)
        grad_b = 2.0 / n * err.sum()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


# --- 実験: 次数 0〜9 で訓練し、訓練lossと検証lossを記録する ---
degrees = list(range(10))
train_losses = []
val_losses = []

print("次数  訓練loss  検証loss")
for degree in degrees:
    Phi_train = poly_features(x_train, degree)  # (n_train, degree)
    Phi_val = poly_features(x_val, degree)      # (n_val, degree)

    if degree > 0:
        # 列ごとの標準化。統計量は訓練データだけから計算する(6.2の規律)
        mu = Phi_train.mean(axis=0)
        sigma = Phi_train.std(axis=0)
        Phi_train = (Phi_train - mu) / sigma
        Phi_val = (Phi_val - mu) / sigma

    w, b = fit_gd(Phi_train, y_train)
    train_loss = mse(Phi_train @ w + b, y_train)
    val_loss = mse(Phi_val @ w + b, y_val)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print("{:4d}  {:8.4f}  {:8.4f}".format(degree, train_loss, val_loss))

train_losses = np.array(train_losses)
val_losses = np.array(val_losses)

# --- 検証: 本章の主張をデータで確認する ---
# (1) 訓練lossは次数を上げても増えない(下がり続ける)
assert np.all(np.diff(train_losses) <= 1e-6), "訓練lossが増えた次数があります"

# (2) 検証lossは途中で反転する: 最小は端(次数9)ではなく途中にある
best_degree = int(np.argmin(val_losses))
assert 0 < best_degree < 9, "検証lossの最小が端にあります"

# (3) この実験では、検証lossはデータの正体(3次)を言い当てる
assert best_degree == 3

# (4) 反転は明確: 次数9の検証lossは最小値の2倍を超えて悪化している
assert val_losses[9] > 2 * val_losses[best_degree]

# (5) その間も訓練lossは下がっている(訓練lossでは過学習を検出できない)
assert train_losses[9] < train_losses[best_degree]

print("検証lossが最小になる次数:", best_degree)
print("すべての assert を通過しました")
