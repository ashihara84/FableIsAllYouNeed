# 第3巻 エピローグ E.3: シグモイド + MSE で勾配がほぼゼロになることの観測実験
# このファイルがやるのは「観測」だけ。なぜこうなるのかは、この巻では説明しない(第4巻で種明かし)。
# 勾配はすべて数値微分(第2巻第1章の中心差分)で測る。式の導出を使わないので、
# 「導出ミスで勾配が小さく見えている」という疑いの余地がない。
import numpy as np

rng = np.random.default_rng(42)

# --- E.1 のデータ: スパム判定の合成データ(2クラス各100件) ---
# 特徴量は2つ(標準化済みのつもりの値): x1 = 怪しい単語の出現量、x2 = 感嘆符の多さ
n = 100
X_normal = rng.normal(loc=[-2.0, -2.0], scale=1.0, size=(n, 2))  # 通常メール (100, 2)
X_spam = rng.normal(loc=[2.0, 2.0], scale=1.0, size=(n, 2))      # スパム     (100, 2)
X = np.vstack([X_normal, X_spam])                                # (200, 2)
y = np.concatenate([np.zeros(n), np.ones(n)])                    # (200,) ラベル: 0=通常, 1=スパム


# --- E.2 の応急処置: シグモイドで出力を 0〜1 に潰す ---
def sigmoid(z):
    """どんな実数も 0〜1 の間に潰す(この形の理由は次巻)"""
    return 1.0 / (1.0 + np.exp(-z))


# 入力をどれだけ大きく・小さくしても、出力は 0〜1 からはみ出さない
print("シグモイドの値:")
for z in [-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0]:
    print(f"  sigmoid({z:6.1f}) = {sigmoid(z):.6f}")
assert np.isclose(sigmoid(0.0), 0.5)


def predict(X, w, b):
    """モデル: 線形(第1巻)の出力をシグモイドで 0〜1 に潰したもの。(n,) を返す"""
    return sigmoid(X @ w + b)


def mse_loss(X, y, w, b):
    """損失は第3章の MSE をそのまま流用"""
    return np.mean((predict(X, w, b) - y) ** 2)


def accuracy(X, y, w, b):
    """0.5 以上をスパムと判定したときの正解率"""
    return np.mean((predict(X, w, b) >= 0.5) == (y == 1))


# シグモイドの出力は必ず 0〜1 に収まっている
assert np.all((predict(X, np.array([1.0, 1.0]), 0.0) > 0.0)
              & (predict(X, np.array([1.0, 1.0]), 0.0) < 1.0))


# --- 勾配は数値微分で測る(第2巻第1章の中心差分。式を使わず、実際に少し動かして傾きを測る) ---
def numerical_grad(loss_fn, X, y, w, b, h=1e-5):
    grad_w = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w)
        e[i] = h
        grad_w[i] = (loss_fn(X, y, w + e, b) - loss_fn(X, y, w - e, b)) / (2 * h)
    grad_b = (loss_fn(X, y, w, b + h) - loss_fn(X, y, w, b - h)) / (2 * h)
    return grad_w, grad_b


def train(w0, b0, lr=0.5, steps=2000):
    """第4章と同じ4拍子: forward → loss → gradient → update"""
    w = np.array(w0, dtype=float)
    b = float(b0)
    history = [mse_loss(X, y, w, b)]
    for _ in range(steps):
        grad_w, grad_b = numerical_grad(mse_loss, X, y, w, b)
        w -= lr * grad_w
        b -= lr * grad_b
        history.append(mse_loss(X, y, w, b))
    return w, b, history


# === 実験1: 初期値 w = (0, 0), b = 0 から学習する → うまくいく ===
w_A, b_A, hist_A = train(w0=[0.0, 0.0], b0=0.0)
print("実験1(初期値ゼロ):")
print(f"  loss: {hist_A[0]:.4f} -> {hist_A[-1]:.4f}   正解率: {accuracy(X, y, w_A, b_A):.1%}")

assert hist_A[-1] < 0.01          # 損失はほぼゼロまで下がった
assert accuracy(X, y, w_A, b_A) >= 0.99  # ほぼ全件正解。この問題自体は解ける

# === 実験2: 初期値 w = (-8, -8), b = 0 から学習する → ほぼ全件間違えたまま止まる ===
w0_bad = np.array([-8.0, -8.0])
grad_w0, grad_b0 = numerical_grad(mse_loss, X, y, w0_bad, 0.0)
w_B, b_B, hist_B = train(w0=w0_bad, b0=0.0)
print("実験2(初期値 w = (-8, -8)):")
print(f"  開始時の loss: {hist_B[0]:.6f}(ほぼ最悪)  正解率: {accuracy(X, y, w0_bad, 0.0):.1%}")
print(f"  開始時の勾配: grad_w = [{grad_w0[0]:.1e}, {grad_w0[1]:.1e}], grad_b = {grad_b0:.1e}")
print(f"  2000ステップ後の loss: {hist_B[-1]:.6f}   正解率: {accuracy(X, y, w_B, b_B):.1%}")

assert hist_B[0] > 0.9                       # 損失はほぼ最悪の値なのに、
assert np.max(np.abs(grad_w0)) < 1e-4        # 勾配はほぼゼロ。
assert np.abs(grad_b0) < 1e-4                # (足元の坂が、実測でほぼ真っ平ら)
assert hist_B[0] - hist_B[-1] < 1e-5         # 2000ステップ回しても損失はほぼ動かず、
assert accuracy(X, y, w_B, b_B) <= 0.05      # ほぼ全件間違えたまま

# === 実験3: 顕微鏡 — 「自信を持って間違えている」1点での勾配 ===
# スパムの典型のような1件を選ぶ(クラスタ中心 (2, 2) にいちばん近い点)
i = int(np.argmin(np.abs(X[n:] - [2.0, 2.0]).sum(axis=1))) + n
x_one = X[i:i + 1]   # (1, 2)
y_one = y[i:i + 1]   # (1,)  正解ラベルは 1(スパム)

w_wrong = np.array([-3.0, -3.0])  # このパラメータは x_one の出力を 0(非スパム)の端へ振り切らせる
b_wrong = 0.0
pred_one = predict(x_one, w_wrong, b_wrong)[0]
grad_w1, grad_b1 = numerical_grad(mse_loss, x_one, y_one, w_wrong, b_wrong)
print("実験3(自信を持って間違えている1点):")
print(f"  正解ラベル: {y_one[0]:.0f}(スパム)   モデルの出力: {pred_one:.6f}(ほぼ0 = 非スパムと確信)")
print(f"  この1点での勾配: grad_w = [{grad_w1[0]:.1e}, {grad_w1[1]:.1e}], grad_b = {grad_b1:.1e}")

assert pred_one < 0.001                      # 間違いの大きさはほぼ最大なのに、
assert np.max(np.abs(grad_w1)) < 1e-4        # 勾配の絶対値は極小
assert np.abs(grad_b1) < 1e-4


# --- 比較対照: シグモイドを外した素の線形回帰 + MSE なら、同じ点で勾配は大きい ---
def linear_mse_loss(X, y, w, b):
    """第4章の線形回帰の損失(シグモイドなし)"""
    return np.mean((X @ w + b - y) ** 2)


grad_w2, grad_b2 = numerical_grad(linear_mse_loss, x_one, y_one, w_wrong, b_wrong)
print(f"  (比較)シグモイドを外すと: grad_w = [{grad_w2[0]:.1f}, {grad_w2[1]:.1f}], grad_b = {grad_b2:.1f}")

assert np.max(np.abs(grad_w2)) > 10.0  # 第4章の世界では「大きく間違えるほど勾配も大きい」が生きている

print()
print("ok: すべての assert を通過しました")
print("  - 同じデータ・同じ損失・同じアルゴリズムで、初期値だけで学習が止まった")
print("  - 自信を持って間違えている点ほど、勾配が小さい(回帰のときと逆)")
print("  - なぜこうなるのか、正しい損失は何か → 第4巻『確率と情報量』へ")
