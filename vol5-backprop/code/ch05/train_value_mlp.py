# 第5巻 第5章 5.1〜5.3: Value ベースの MLP 訓練と、過去巻の再実装
# - 5.1: linear 層と MLP を Value で組み、4拍子(第3巻4章)の訓練ループを回す
# - 5.2: 第3巻4章の線形回帰・第3巻エピローグ/第4巻4章のロジスティック回帰・
#        第4巻6章の softmax 分類を autograd で再実装し、手導出・数値微分と照合する
# - 5.3: スカラー Value の遅さを実測し、行列版 tensor_autograd.py と比較する
# 実行: python3 train_value_mlp.py
import os
import sys
import time

import numpy as np

# 第4章で自作した micrograd(Value)を import する。autograd 本体はもう書かない
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ch04"))
from micrograd import Value  # noqa: E402


# ============================================================
# 5.1 Value ベースの linear 層と MLP
# ============================================================

class LinearValue:
    """第1巻6章の linear(X, W, b) = X @ W + b を、Value 1個ずつで組んだもの。"""

    def __init__(self, rng, d_in, d_out):
        # 初期値はひとまず小さな乱数(初期化の理論は第6章6.6)
        self.W = [[Value(rng.uniform(-1.0, 1.0)) for _ in range(d_out)]
                  for _ in range(d_in)]
        self.b = [Value(0.0) for _ in range(d_out)]
        self.d_in, self.d_out = d_in, d_out

    def __call__(self, x):
        out = []
        for j in range(self.d_out):
            acc = self.b[j]
            for i in range(self.d_in):
                acc = acc + x[i] * self.W[i][j]
            out.append(acc)
        return out

    def parameters(self):
        return [w for row in self.W for w in row] + self.b


class MLPValue:
    """linear → ReLU → linear → …(第2章の MLP)。最後の層は活性化なし。"""

    def __init__(self, rng, sizes):
        self.layers = [LinearValue(rng, d_in, d_out)
                       for d_in, d_out in zip(sizes[:-1], sizes[1:])]

    def __call__(self, x):
        for k, layer in enumerate(self.layers):
            x = layer(x)
            if k < len(self.layers) - 1:
                x = [h.relu() for h in x]
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def sigmoid_value(z):
    """第3巻エピローグの sigmoid の Value 版。"""
    return 1.0 / (1.0 + (-z).exp())


def log_loss_value(p, y):
    """第4巻4章の log loss(1サンプル分)。y は 0.0 か 1.0 の float。"""
    return -(y * p.log() + (1.0 - y) * (1.0 - p).log())


def make_xor_data():
    """第1章1.1のデータB(4塊の市松模様、200点)を1点も違わず再現する。
    乱数列を合わせるため、第1章と同じ順でデータA(対照用の2塊)も先に引いて捨てる。"""
    rng = np.random.default_rng(42)
    n = 50

    def blob(cx, cy):
        return rng.normal(loc=[cx, cy], scale=0.7, size=(n, 2))

    blob(-2.0, -2.0), blob(2.0, 2.0)                          # データAの分を消費
    X = np.vstack([blob(-2.0, -2.0), blob(2.0, 2.0),          # ラベル0(左下・右上)
                   blob(-2.0, 2.0), blob(2.0, -2.0)])         # ラベル1(左上・右下)
    y = np.concatenate([np.zeros(2 * n), np.ones(2 * n)])
    return X, y


def train_mlp_value(X, y, hidden=8, lr=0.5, num_steps=120, verbose=True):
    """4拍子(第3巻4章)で Value ベースの MLP を訓練する。所要時間も返す。"""
    rng = np.random.default_rng(0)
    model = MLPValue(rng, [2, hidden, 1])
    n = len(X)
    history = []
    t0 = time.perf_counter()
    for step in range(num_steps):
        # 1. forward: 全サンプルの予測(計算グラフがここで組み上がる)
        probs = [sigmoid_value(model([Value(X[i, 0]), Value(X[i, 1])])[0])
                 for i in range(n)]
        # 2. loss: log loss の平均(第4巻4章)
        loss = sum((log_loss_value(p, y[i]) for i, p in enumerate(probs)),
                   Value(0.0)) * (1.0 / n)
        # 3. gradient: 手導出の代わりに backward() 一発。grad は累積するので先にゼロへ
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()
        # 4. update: 坂を下る(第2巻3章)
        for p in model.parameters():
            p.data -= lr * p.grad
        history.append(loss.data)
        if verbose and step in (0, 10, 40, num_steps - 1):
            print("  step {:>4}: loss = {:.6f}".format(step, loss.data))
    elapsed = time.perf_counter() - t0
    acc = np.mean([(sigmoid_value(model([Value(x1), Value(x2)])[0]).data >= 0.5)
                   == (lab == 1.0) for (x1, x2), lab in zip(X, y)])
    return model, history, acc, elapsed


print("=== 5.1 Value ベースの MLP を XOR データで訓練する ===")
X_xor, y_xor = make_xor_data()

# --- 訓練の前に: autograd の勾配を、第2章2.4・第3章3.3の手導出(行列の式)と照合する ---
rng_check = np.random.default_rng(0)
model0 = MLPValue(rng_check, [2, 8, 1])
n_xor = len(X_xor)
probs0 = [sigmoid_value(model0([Value(X_xor[i, 0]), Value(X_xor[i, 1])])[0])
          for i in range(n_xor)]
loss0 = sum((log_loss_value(p, y_xor[i]) for i, p in enumerate(probs0)),
            Value(0.0)) * (1.0 / n_xor)
for p in model0.parameters():
    p.grad = 0.0
loss0.backward()

# 同じ初期パラメータで、第3章3.3の行列 backprop を NumPy で手実行する
W1_np = np.array([[v.data for v in row] for row in model0.layers[0].W])  # (2, 8)
b1_np = np.array([v.data for v in model0.layers[0].b])                   # (8,)
W2_np = np.array([[v.data for v in row] for row in model0.layers[1].W])  # (8, 1)
b2_np = np.array([v.data for v in model0.layers[1].b])                   # (1,)
Z = X_xor @ W1_np + b1_np                                # (200, 8)
H = np.maximum(Z, 0.0)                                   # (200, 8)
p_hat = 1.0 / (1.0 + np.exp(-(H @ W2_np + b2_np)))       # (200, 1)
delta_s = (p_hat - y_xor.reshape(-1, 1)) / n_xor         # (200, 1) 第4巻4.3の貯金
delta_Z = (delta_s @ W2_np.T) * (Z > 0.0)                # (200, 8) δ の逆伝播(第3章3.3)
grad_W1_hand = X_xor.T @ delta_Z                         # ∂L/∂W = 入力^T @ δ
grad_W1_auto = np.array([[v.grad for v in row] for row in model0.layers[0].W])
assert np.allclose(grad_W1_auto, grad_W1_hand, atol=1e-12)
assert np.allclose([v.grad for v in model0.layers[0].b], delta_Z.sum(axis=0), atol=1e-12)
assert np.allclose(np.array([[v.grad for v in row] for row in model0.layers[1].W]),
                   H.T @ delta_s, atol=1e-12)
print("  検算 OK: 第2章2.4で地獄だった ∂L/∂W_1 が、backward() と一致")

model, hist_mlp, acc_mlp, sec_value = train_mlp_value(X_xor, y_xor)
print("  正解率: {:.1%}   所要時間: {:.2f} 秒".format(acc_mlp, sec_value))

assert hist_mlp[-1] < 0.05          # 損失は下がりきった
assert acc_mlp >= 0.99              # 第1章1.1で50%台に張りついたデータBが、ほぼ全件正解


# ============================================================
# 5.2 過去2巻の再実装 — 手導出が全部自動で出る
# ============================================================

print("=== 5.2a 第3巻4章の線形回帰を autograd で ===")


def make_regression_data():
    """第3巻2章と同一の合成データ(勉強時間→点数、20人)。真の規則 y = 7x + 20 + ばらつき(標準偏差6)。"""
    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, 9.0, size=20)
    y = 7.0 * X + 20.0 + rng.normal(0.0, 6.0, size=20)
    return X, y


X_reg, y_reg = make_regression_data()
n_reg = len(X_reg)
assert np.allclose(X_reg[:3], [6.96560444, 3.94990596, 7.72738128])  # 第3巻と同じ20人


def mse_loss_value(w, b, X, y):
    """MSE を Value で組む。微分の式はどこにも書かない。"""
    total = Value(0.0)
    for i in range(len(X)):
        r = w * X[i] + b - y[i]
        total = total + r * r
    return total * (1.0 / len(X))


# --- 照合: (0, 0) 地点の勾配を「手導出 vs 数値微分 vs autograd」で三つ巴に ---
w, b = Value(0.0), Value(0.0)
loss = mse_loss_value(w, b, X_reg, y_reg)
loss.backward()

r0 = (0.0 * X_reg + 0.0) - y_reg
grad_w_hand = 2.0 * np.mean(r0 * X_reg)     # 第3巻4.1の手導出: (2/n) Σ r_i x_i
grad_b_hand = 2.0 * np.mean(r0)

h = 1e-6                                    # 第2巻1章の中心差分
mse_np = lambda w_, b_: np.mean((w_ * X_reg + b_ - y_reg) ** 2)
grad_w_num = (mse_np(h, 0.0) - mse_np(-h, 0.0)) / (2 * h)
grad_b_num = (mse_np(0.0, h) - mse_np(0.0, -h)) / (2 * h)

print("  手導出   : grad_w = {:.6f}, grad_b = {:.6f}".format(grad_w_hand, grad_b_hand))
print("  数値微分 : grad_w = {:.6f}, grad_b = {:.6f}".format(grad_w_num, grad_b_num))
print("  autograd : grad_w = {:.6f}, grad_b = {:.6f}".format(w.grad, b.grad))
assert np.allclose(w.grad, grad_w_hand, atol=1e-9)   # 自動微分 = 第3巻の手導出
assert np.allclose(b.grad, grad_b_hand, atol=1e-9)
assert np.allclose(w.grad, grad_w_num, atol=1e-6)    # 自動微分 = 実測の傾き
assert np.allclose(b.grad, grad_b_num, atol=1e-6)

# --- 訓練: 第3巻4.2と同じ 4拍子・同じ lr(0.01。0.03 を超えると発散するのも同じ) ---
w, b = Value(0.0), Value(0.0)
lr = 0.01
for step in range(20000):
    loss = mse_loss_value(w, b, X_reg, y_reg)   # 1. forward + 2. loss
    w.grad, b.grad = 0.0, 0.0                   # 3. gradient
    loss.backward()
    w.data -= lr * w.grad                       # 4. update
    b.data -= lr * b.grad

print("  学習結果: w = {:.6f}, b = {:.6f}(第3巻4.4の解析解と一致するか?)".format(w.data, b.data))
assert np.allclose([w.data, b.data], [6.7221, 22.0922], atol=1e-3)  # 解析解(第3巻4.4)
print("  ok: 勾配の式を1行も書かずに、第3巻と同じ答えに着いた")

print("=== 5.2b 第3巻エピローグの分類を、悪い初期値から log loss で ===")


def make_spam_data():
    """第3巻エピローグ E.1 と同一の合成データ(2クラス各100件)。"""
    rng = np.random.default_rng(42)
    n = 100
    X_normal = rng.normal(loc=[-2.0, -2.0], scale=1.0, size=(n, 2))
    X_spam = rng.normal(loc=[2.0, 2.0], scale=1.0, size=(n, 2))
    return np.vstack([X_normal, X_spam]), np.concatenate([np.zeros(n), np.ones(n)])


X_cls, y_cls = make_spam_data()
n_cls = len(X_cls)


def logistic_loss_value(w1, w2, b, X, y):
    """sigmoid + log loss(第4巻4章)を Value で組む。これも微分の式はなし。
    p を経由せず z から直接組むのは数値の都合: 初期値 (-8, -8) では sigmoid が
    float の精度で 1.0 に張り付き、log(1 - p) = log(0) が事故になるため。
    -log σ(z) = log(1 + e^{-z}), -log(1 - σ(z)) = log(1 + e^{z}) を使う。"""
    total = Value(0.0)
    for i in range(len(X)):
        z = w1 * X[i, 0] + w2 * X[i, 1] + b
        loss_i = y[i] * (1.0 + (-z).exp()).log() + (1.0 - y[i]) * (1.0 + z.exp()).log()
        total = total + loss_i
    return total * (1.0 / len(X))


# --- 照合: 第4巻4.3の手導出 (p - y)・x が自動で出てくるか ---
w1, w2, b = Value(-8.0), Value(-8.0), Value(0.0)   # エピローグで学習が死んだ初期値
loss = logistic_loss_value(w1, w2, b, X_cls, y_cls)
loss.backward()

p_np = 1.0 / (1.0 + np.exp(-(X_cls @ np.array([-8.0, -8.0]) + 0.0)))
grad_hand = (p_np - y_cls) @ X_cls / n_cls          # 第4巻4.3: (1/n) Σ (p_i - y_i) x_i
grad_b_hand = np.mean(p_np - y_cls)


def log_loss_np(w_, b_):
    """数値微分用の NumPy 版 log loss(logaddexp は log(1 + e^z) の安定実装)。"""
    z = X_cls @ w_ + b_
    return np.mean(y_cls * np.logaddexp(0.0, -z) + (1.0 - y_cls) * np.logaddexp(0.0, z))


h = 1e-6
w_bad = np.array([-8.0, -8.0])
grad_num = np.array([
    (log_loss_np(w_bad + [h, 0], 0.0) - log_loss_np(w_bad - [h, 0], 0.0)) / (2 * h),
    (log_loss_np(w_bad + [0, h], 0.0) - log_loss_np(w_bad - [0, h], 0.0)) / (2 * h)])
grad_b_num = (log_loss_np(w_bad, h) - log_loss_np(w_bad, -h)) / (2 * h)

print("  手導出   : grad_w = [{:.6f}, {:.6f}], grad_b = {:.6f}".format(
    grad_hand[0], grad_hand[1], grad_b_hand))
print("  数値微分 : grad_w = [{:.6f}, {:.6f}], grad_b = {:.6f}".format(
    grad_num[0], grad_num[1], grad_b_num))
print("  autograd : grad_w = [{:.6f}, {:.6f}], grad_b = {:.6f}".format(
    w1.grad, w2.grad, b.grad))
assert np.allclose([w1.grad, w2.grad, b.grad],
                   [grad_hand[0], grad_hand[1], grad_b_hand], atol=1e-9)
assert np.allclose([w1.grad, w2.grad, b.grad],
                   [grad_num[0], grad_num[1], grad_b_num], atol=1e-5)
# MSE のときは 1e-5 程度だった勾配(第3巻エピローグの実測)が、log loss では桁違いに大きい
assert np.abs(w1.grad) > 1.0

# --- 訓練: エピローグの実験2と同じ初期値 w = (-8, -8) から ---
params = [w1, w2, b]
lr = 0.5
for step in range(300):
    loss = logistic_loss_value(w1, w2, b, X_cls, y_cls)
    for p in params:
        p.grad = 0.0
    loss.backward()
    for p in params:
        p.data -= lr * p.grad

p_final = 1.0 / (1.0 + np.exp(-(X_cls @ np.array([w1.data, w2.data]) + b.data)))
acc_cls = np.mean((p_final >= 0.5) == (y_cls == 1))
print("  300ステップ後: loss = {:.6f}, 正解率 = {:.1%}".format(loss.data, acc_cls))
assert acc_cls >= 0.99   # エピローグでは 0.5% のまま死んでいた初期値から、ほぼ全件正解へ
print("  ok: 勾配が死んだ初期値 (-8, -8) からでも学習が進んだ")

print("=== 5.2c 第4巻6章の softmax 分類の勾配も自動で出る ===")
# 小さな3クラス問題で、autograd の勾配が手導出 (p - t) と一致することだけ確認する
# (本格的な訓練はスカラー Value では遅すぎるので、5.3 の行列版で行う)
rng3 = np.random.default_rng(0)
X3 = rng3.standard_normal((12, 2))
t3 = np.array([0, 1, 2] * 4)
W3 = [[Value(rng3.uniform(-1.0, 1.0)) for _ in range(3)] for _ in range(2)]
b3 = [Value(0.0) for _ in range(3)]

total = Value(0.0)
for i in range(12):
    z = [X3[i, 0] * W3[0][k] + X3[i, 1] * W3[1][k] + b3[k] for k in range(3)]
    exps = [zk.exp() for zk in z]
    denom = exps[0] + exps[1] + exps[2]
    p_target = exps[t3[i]] / denom            # softmax(第4巻6.2)
    total = total + -(p_target.log())         # cross-entropy(第4巻5章)
loss3 = total * (1.0 / 12)
for row in W3:
    for v in row:
        v.grad = 0.0
loss3.backward()

# 手導出(第4巻6.3): ∂L/∂W = (1/n) X^T @ (P - T)
W3_np = np.array([[v.data for v in row] for row in W3])
Z = X3 @ W3_np
P = np.exp(Z - Z.max(axis=1, keepdims=True))
P = P / P.sum(axis=1, keepdims=True)
T = np.zeros_like(P)
T[np.arange(12), t3] = 1.0
grad_W3_hand = X3.T @ (P - T) / 12
grad_W3_auto = np.array([[v.grad for v in row] for row in W3])
assert np.allclose(grad_W3_auto, grad_W3_hand, atol=1e-9)
print("  ok: softmax + cross-entropy の勾配 (p - t) も自動で出た")


# ============================================================
# 5.3 スカラー Value は遅い — 行列版 tensor_autograd との実測比較
# ============================================================

print("=== 5.3 実測: スカラー Value vs NumPy 行列版 Tensor ===")
from tensor_autograd import Tensor, softmax_cross_entropy  # noqa: E402(本文の流れ通り、ここで import)


def train_mlp_tensor(X, y, hidden=8, lr=0.5, num_steps=120):
    """5.1 と同じ課題・同じ4拍子を、行列版 Tensor で。2クラス softmax で表す。"""
    rng = np.random.default_rng(0)
    W1 = Tensor(rng.uniform(-1.0, 1.0, size=(2, hidden)))
    b1 = Tensor(np.zeros(hidden))
    W2 = Tensor(rng.uniform(-1.0, 1.0, size=(hidden, 2)))
    b2 = Tensor(np.zeros(2))
    params = [W1, b1, W2, b2]
    X_t = Tensor(X)
    targets = y.astype(int)
    t0 = time.perf_counter()
    for step in range(num_steps):
        logits = (X_t @ W1 + b1).relu() @ W2 + b2          # 1. forward(バッチ丸ごと)
        loss = softmax_cross_entropy(logits, targets)      # 2. loss
        for p in params:                                   # 3. gradient
            p.grad = np.zeros_like(p.data)
        loss.backward()
        for p in params:                                   # 4. update
            p.data -= lr * p.grad
    elapsed = time.perf_counter() - t0
    pred = np.argmax(((X_t @ W1 + b1).relu() @ W2 + b2).data, axis=1)
    return np.mean(pred == targets), elapsed, loss.data


acc_tensor, sec_tensor, loss_tensor = train_mlp_tensor(X_xor, y_xor)
print("  スカラー Value: {:.2f} 秒(5.1 の実測)".format(sec_value))
print("  行列版 Tensor : {:.4f} 秒   正解率: {:.1%}".format(sec_tensor, acc_tensor))
print("  速度比: 約 {:.0f} 倍".format(sec_value / sec_tensor))
assert acc_tensor >= 0.99       # 精度はスカラー版と同等(塊の境目の数点だけが残る)
assert sec_tensor < sec_value   # 同じ課題・同じステップ数で、行列版が速い

print("ok: すべての assert を通過しました")
