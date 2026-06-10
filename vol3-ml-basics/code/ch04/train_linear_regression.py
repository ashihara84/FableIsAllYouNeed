# 第3巻 第4章 4.2: 線形回帰のフルスクラッチ訓練(NumPyのみ)
# forward -> loss -> gradient -> update の「4拍子」。
# この4つの拍は、第8巻の Transformer の訓練ループまで一切変わらない。
import numpy as np


def make_data():
    """第2章と同一の合成データ(20人)。真の規則 y = 7x + 20 + ばらつき(標準偏差6)。"""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 9, size=(20, 1))       # 勉強時間 (20, 1)
    noise = rng.normal(0, 6.0, size=(20, 1))
    y = 7.0 * X + 20.0 + noise                # 点数 (20, 1)
    return X.ravel(), y.ravel()               # 本章はスカラー w で進めるので (20,) に潰す


def mse(y_hat, y):
    """第3章で定義した平均二乗誤差。"""
    return np.mean((y_hat - y) ** 2)


X, y = make_data()
assert np.allclose(X[:3], [6.96560444, 3.94990596, 7.72738128])  # 第2章と同じ20人

# --- 4.1 の検算: 手で導いた勾配を数値微分と突き合わせる(第2巻1章の習慣) ---
w0, b0 = 0.0, 0.0
r = (w0 * X + b0) - y               # 残差 (20,)
grad_w0 = 2.0 * np.mean(r * X)      # 手導出: ∂L/∂w = (2/n) Σ r_i x_i
grad_b0 = 2.0 * np.mean(r)          # 手導出: ∂L/∂b = (2/n) Σ r_i

h = 1e-6                            # 数値微分: (L(θ+h) - L(θ-h)) / 2h
num_w = (mse((w0 + h) * X + b0, y) - mse((w0 - h) * X + b0, y)) / (2 * h)
num_b = (mse(w0 * X + (b0 + h), y) - mse(w0 * X + (b0 - h), y)) / (2 * h)
assert np.allclose(grad_w0, num_w, atol=1e-4)
assert np.allclose(grad_b0, num_b, atol=1e-4)
print("勾配の検算 OK: 手導出 ({:.4f}, {:.4f}) = 数値微分 ({:.4f}, {:.4f})".format(
    grad_w0, grad_b0, num_w, num_b))

# --- 訓練ループ: 4拍子 ---
w, b = 0.0, 0.0     # 初期値
lr = 0.01           # learning rate(第2巻3章)。0.03 を超えると発散する(演習1)
num_steps = 5000
history = []        # 学習曲線用に loss を記録(4.3)

for step in range(num_steps):
    y_hat = w * X + b                        # 1. forward : 現在の (w, b) で予測
    loss = mse(y_hat, y)                     # 2. loss    : 悪さを1つの数にする
    grad_w = 2.0 * np.mean((y_hat - y) * X)  # 3. gradient: 4.1 で手導出した式
    grad_b = 2.0 * np.mean(y_hat - y)
    w = w - lr * grad_w                      # 4. update  : 坂を下る(第2巻3章)
    b = b - lr * grad_b
    history.append(loss)
    if step in (0, 1, 2, 3, 10, 100, 500, 1000, 2000):
        print("step {:>4}: loss = {:.6f}".format(step, loss))

print("step {:>4}: loss = {:.6f}".format(num_steps - 1, history[-1]))
print("学習結果: w = {:.4f}, b = {:.4f}".format(w, b))

# --- 検算 1: loss は一度も増えずに下がり続けたか ---
assert history[-1] < history[0]
for i in range(len(history) - 1):
    assert history[i + 1] <= history[i] + 1e-12
# --- 検算 2: 収束先(この値の種明かしは 4.4) ---
assert np.allclose([w, b], [6.7221, 22.0922], atol=1e-3)

# --- 第2章の演習の答え合わせ: 手動フィットと勝負する ---
w_hand, b_hand = 6.7, 23.0  # 第2章の演習3で筆者が記録した値
loss_hand = mse(w_hand * X + b_hand, y)
loss_gd = mse(w * X + b, y)
assert loss_gd < loss_hand  # 勾配降下の勝ち
print("手動フィット (w={}, b={}): loss = {:.6f}".format(w_hand, b_hand, loss_hand))
print("勾配降下                 : loss = {:.6f}".format(loss_gd))
print("ok: すべての assert を通過しました")
