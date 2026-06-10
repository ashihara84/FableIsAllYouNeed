# 第4巻 第4章 4.3〜4.5: ロジスティック回帰のフルスクラッチ実装 — 第3巻の宿題の回収
# データ・初期値・学習率・ステップ数は第3巻エピローグ(vol3-ml-basics/code/ch08/dead_gradient.py)
# と完全に同一。変えるのは損失関数だけ: MSE → log loss。
# それだけで、凍りついていた学習が動き出すことを assert で確認する。
import numpy as np

rng = np.random.default_rng(42)

# --- データ: 第3巻 E.1 と同一(乱数の種も生成順も同じ。1文字も変えていない) ---
n = 100
X_normal = rng.normal(loc=[-2.0, -2.0], scale=1.0, size=(n, 2))  # 通常メール (100, 2)
X_spam = rng.normal(loc=[2.0, 2.0], scale=1.0, size=(n, 2))      # スパム     (100, 2)
X = np.vstack([X_normal, X_spam])                                # (200, 2)
y = np.concatenate([np.zeros(n), np.ones(n)])                    # (200,) ラベル: 0=通常, 1=スパム


# --- モデル: 第3巻 E.2 と同一。変えない ---
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def predict(X, w, b):
    """モデルの出力 = 「スパムである確率」(4.1 でこの解釈が正当化された)。(n,) を返す"""
    return sigmoid(X @ w + b)


def accuracy(X, y, w, b):
    """0.5 以上をスパムと判定したときの正解率(第3巻 E.2 と同一)"""
    return np.mean((predict(X, w, b) >= 0.5) == (y == 1))


# --- 損失: ここだけが第3巻との違い。MSE → log loss(4.2 で導出した負の対数尤度) ---
def log_loss(X, y, w, b):
    p = np.clip(predict(X, w, b), 1e-12, 1.0 - 1e-12)  # p がちょうど 0/1 に丸まると log が無限大に飛ぶため
    return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def grad_log_loss(X, y, w, b):
    """4.3 で手導出した解析形: (出力 − 正解) を入力で重みづけて平均。数式の (y − t) はここの (p − y)"""
    p = predict(X, w, b)
    grad_w = X.T @ (p - y) / len(y)   # (2,)
    grad_b = np.mean(p - y)
    return grad_w, grad_b


def mse_loss(X, y, w, b):
    """比較対照: 第3巻 E.2 の損失(そのまま)"""
    return np.mean((predict(X, w, b) - y) ** 2)


# --- 数値微分(第3巻 E.3 と同一の中心差分)。手導出の検算と、MSE 版の再現に使う ---
def numerical_grad(loss_fn, X, y, w, b, h=1e-5):
    grad_w = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w)
        e[i] = h
        grad_w[i] = (loss_fn(X, y, w + e, b) - loss_fn(X, y, w - e, b)) / (2 * h)
    grad_b = (loss_fn(X, y, w, b + h) - loss_fn(X, y, w, b - h)) / (2 * h)
    return grad_w, grad_b


# --- 4.3 の検算: 手で導いた (y − t)・x を数値微分と突き合わせる(第2巻1章以来の習慣) ---
w_chk = np.array([0.7, -1.3])
b_chk = 0.4
gw_hand, gb_hand = grad_log_loss(X, y, w_chk, b_chk)
gw_num, gb_num = numerical_grad(log_loss, X, y, w_chk, b_chk)
assert np.allclose(gw_hand, gw_num, atol=1e-6)
assert np.allclose(gb_hand, gb_num, atol=1e-6)
print("検算: 手導出の勾配 (y − t)・x は数値微分と一致しました")


def train_logloss(w0, b0, lr=0.5, steps=2000):
    """第3巻 E.3 の train と同じ4拍子・同じ lr・同じ steps。勾配だけ解析形(4.3)を使う"""
    w = np.array(w0, dtype=float)
    b = float(b0)
    history = [log_loss(X, y, w, b)]
    for _ in range(steps):
        grad_w, grad_b = grad_log_loss(X, y, w, b)
        w -= lr * grad_w
        b -= lr * grad_b
        history.append(log_loss(X, y, w, b))
    return w, b, history


def train_mse(w0, b0, lr=0.5, steps=2000):
    """第3巻 E.3 の train の忠実な再現(勾配は数値微分のまま)。比較対照"""
    w = np.array(w0, dtype=float)
    b = float(b0)
    history = [mse_loss(X, y, w, b)]
    for _ in range(steps):
        grad_w, grad_b = numerical_grad(mse_loss, X, y, w, b)
        w -= lr * grad_w
        b -= lr * grad_b
        history.append(mse_loss(X, y, w, b))
    return w, b, history


# === 実験1: 初期値ゼロ + log loss → うまくいく(MSE でも解けた条件。まずは同点を確認) ===
w_A, b_A, hist_A = train_logloss(w0=[0.0, 0.0], b0=0.0)
print("実験1(初期値ゼロ, log loss):")
print(f"  loss: {hist_A[0]:.4f} -> {hist_A[-1]:.4f}   正解率: {accuracy(X, y, w_A, b_A):.1%}")

assert hist_A[-1] < 0.02
assert accuracy(X, y, w_A, b_A) >= 0.99

# === 実験2(本丸): 第3巻の凍結条件 w = (-8, -8), b = 0 + log loss → 学習が動き出す ===
w0_bad = np.array([-8.0, -8.0])
grad_w0_log, grad_b0_log = grad_log_loss(X, y, w0_bad, 0.0)
w_B, b_B, hist_B = train_logloss(w0=w0_bad, b0=0.0)
print("実験2(第3巻の凍結条件 w = (-8, -8), log loss):")
print(f"  開始時の正解率: {accuracy(X, y, w0_bad, 0.0):.1%}(第3巻と同じ、ほぼ全件誤り)")
print(f"  開始時の勾配: grad_w = [{grad_w0_log[0]:.2f}, {grad_w0_log[1]:.2f}], "
      f"grad_b = {grad_b0_log:.2f}")
print(f"  loss: {hist_B[0]:.4f} -> {hist_B[-1]:.4f}   正解率: {accuracy(X, y, w_B, b_B):.1%}")

assert accuracy(X, y, w0_bad, 0.0) <= 0.05            # 出発点は第3巻と同じ「ほぼ最悪」なのに、
assert np.max(np.abs(grad_w0_log)) > 0.5              # 勾配は死んでいない(MSE では 1e-4 未満だった)
assert hist_B[0] - hist_B[-1] > 10.0                  # 損失は大きく下がり、
assert accuracy(X, y, w_B, b_B) >= 0.99               # ほぼ全件正解に到達。宿題回収

# === 実験3: 同じ凍結条件 + MSE(第3巻の再現)→ やはり凍ったまま ===
grad_w0_mse, grad_b0_mse = numerical_grad(mse_loss, X, y, w0_bad, 0.0)
w_C, b_C, hist_C = train_mse(w0=w0_bad, b0=0.0)
print("実験3(同じ凍結条件, MSE — 第3巻の再現):")
print(f"  開始時の勾配: grad_w = [{grad_w0_mse[0]:.1e}, {grad_w0_mse[1]:.1e}], "
      f"grad_b = {grad_b0_mse:.1e}")
print(f"  loss: {hist_C[0]:.6f} -> {hist_C[-1]:.6f}   正解率: {accuracy(X, y, w_C, b_C):.1%}")

assert np.max(np.abs(grad_w0_mse)) < 1e-4             # 第3巻 E.3 の assert がそのまま通る
assert hist_C[0] - hist_C[-1] < 1e-5
assert accuracy(X, y, w_C, b_C) <= 0.05

# === 実験4: 顕微鏡の再訪 — 「自信を持って間違えている」1点(第3巻 実験3と同じ点・同じパラメータ) ===
i = int(np.argmin(np.abs(X[n:] - [2.0, 2.0]).sum(axis=1))) + n
x_one = X[i:i + 1]   # (1, 2)
y_one = y[i:i + 1]   # (1,)  正解ラベルは 1(スパム)

w_wrong = np.array([-3.0, -3.0])
b_wrong = 0.0
pred_one = predict(x_one, w_wrong, b_wrong)[0]
gw_log_1, gb_log_1 = grad_log_loss(x_one, y_one, w_wrong, b_wrong)
gw_mse_1, gb_mse_1 = numerical_grad(mse_loss, x_one, y_one, w_wrong, b_wrong)
print("実験4(自信を持って間違えている1点、第3巻 実験3と同一):")
print(f"  モデルの出力: {pred_one:.6f}(ほぼ0 = 非スパムと確信。正解は 1)")
print(f"  log loss の勾配: grad_w = [{gw_log_1[0]:.2f}, {gw_log_1[1]:.2f}], grad_b = {gb_log_1:.2f}")
print(f"  MSE の勾配:      grad_w = [{gw_mse_1[0]:.1e}, {gw_mse_1[1]:.1e}], grad_b = {gb_mse_1:.1e}")

assert pred_one < 0.001                               # 間違いの大きさはほぼ最大。このとき
assert np.max(np.abs(gw_log_1)) > 1.0                 # log loss は大声で訂正を要求し(勾配 ≈ −x)、
assert np.max(np.abs(gw_mse_1)) < 1e-4                # MSE は黙り込む(第3巻で観測した現象)

# === 4.5 の種明かし: MSE + sigmoid の勾配の解析形(第3巻では数値微分の裏に隠れていた式) ===
def grad_mse_sigmoid(X, y, w, b):
    """∂L/∂z = 2 (y − t) · y(1 − y) を入力で重みづけて平均。y(1 − y) が犯人"""
    p = predict(X, w, b)
    dz = 2.0 * (p - y) * p * (1.0 - p)   # (n,)
    grad_w = X.T @ dz / len(y)
    grad_b = np.mean(dz)
    return grad_w, grad_b


# 解析形が第3巻の数値微分と一致することを、穏当な点と飽和した点の両方で確認
for w_t, b_t in [(np.array([0.7, -1.3]), 0.4), (w0_bad, 0.0)]:
    gw_a, gb_a = grad_mse_sigmoid(X, y, w_t, b_t)
    gw_n, gb_n = numerical_grad(mse_loss, X, y, w_t, b_t)
    assert np.allclose(gw_a, gw_n, atol=1e-6)
    assert np.allclose(gb_a, gb_n, atol=1e-6)

factor = pred_one * (1.0 - pred_one)   # 顕微鏡の1点での y(1 − y)
print(f"  顕微鏡の1点での y(1 − y) = {factor:.1e}  ← 勾配をこの倍率で押し潰していた犯人")
assert factor < 1e-5

# === 演習1(数値部分): 決定境界 — predict がちょうど 0.5 になる直線 ===
# 境界は w1 x1 + w2 x2 + b = 0、すなわち x2 = -(w1 x1 + b) / w2
x1_line = np.linspace(-5.0, 5.0, 11)                  # (11,)
x2_line = -(w_B[0] * x1_line + b_B) / w_B[1]          # (11,)
on_line = np.stack([x1_line, x2_line], axis=1)        # (11, 2) 境界上の点たち
assert np.allclose(predict(on_line, w_B, b_B), 0.5)   # 境界上では確率がちょうど 1/2

# 2つのクラスタ中心は境界の反対側に落ちる
p_centers = predict(np.array([[-2.0, -2.0], [2.0, 2.0]]), w_B, b_B)
assert p_centers[0] < 0.5 < p_centers[1]

print()
print("ok: すべての assert を通過しました")
print("  - 第3巻と同じデータ・同じ初期値 w = (-8, -8)・同じ lr/steps で、損失を log loss に")
print("    替えただけで学習が進み、ほぼ全件正解に到達した(MSE 版は再現でも凍ったまま)")
print("  - 自信を持って間違えている点で、log loss の勾配は大きく、MSE は y(1−y) に潰されていた")
