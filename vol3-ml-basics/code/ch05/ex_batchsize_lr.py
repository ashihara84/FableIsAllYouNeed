# 第3巻 第5章 演習: バッチサイズと learning rate の関係を実験する(略解のコード)
# minibatch_sgd.py と同じデータ・同じ訓練ループで、(batch_size, lr) の組を総当たりする。
import numpy as np

rng = np.random.default_rng(42)

# --- 第2章と同じ生成規則の合成データ(minibatch_sgd.py と同一) ---
N = 256
X = rng.uniform(0, 9, size=(N, 1))             # 勉強時間 (N, 1)
y = 7.0 * X + 20.0 + rng.normal(0, 6.0, size=(N, 1))  # 点数 (N, 1)


def predict(X, w, b):
    return X @ w + b


def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def gradients(X, y_true, w, b):
    n = X.shape[0]
    err = predict(X, w, b) - y_true
    grad_w = (2.0 / n) * (X.T @ err)
    grad_b = (2.0 / n) * np.sum(err)
    return grad_w, grad_b


def train(batch_size, n_epochs, lr, seed=0):
    rng_shuffle = np.random.default_rng(seed)
    w = np.zeros((1, 1))
    b = 0.0
    for epoch in range(n_epochs):
        idx = rng_shuffle.permutation(N)
        for start in range(0, N, batch_size):
            batch = idx[start:start + batch_size]
            grad_w, grad_b = gradients(X[batch], y[batch], w, b)
            w = w - lr * grad_w
            b = b - lr * grad_b
    return mse(predict(X, w, b), y)


# --- (batch_size, lr) の総当たり。各組 30 エポック ---
np.seterr(all="ignore")                        # 発散組の overflow 警告を黙らせる
batch_sizes = [1, 32, 256]
lrs = [0.001, 0.005, 0.02, 0.03]

print("30エポック後の loss(行: lr、列: batch_size)")
print(f"{'lr':>8} " + "".join(f"{f'bs={bs}':>12}" for bs in batch_sizes))
table = {}
for lr in lrs:
    cells = []
    for bs in batch_sizes:
        loss = train(batch_size=bs, n_epochs=30, lr=lr)
        table[(bs, lr)] = loss
        cells.append("  発散" if not loss < 1e6 else f"{loss:12.3f}")
    print(f"{lr:>8} " + "".join(f"{c:>12}" for c in cells))

# --- 観察した関係を assert で固定する ---
# (1) 同じ lr=0.03 でも、バッチが小さいと発散し、大きいと収束する
assert not table[(1, 0.03)] < 1e6              # batch_size=1 は発散(inf/nan に飛ぶ)
assert table[(256, 0.03)] < 80.0               # batch_size=256 は下り続けている。実測 約72

# (2) 同じ lr=0.001 では立場が逆転: 小バッチは床(およそ36)に到達、大バッチはまだ遠い
assert table[(1, 0.001)] < 38.0                # 更新回数が多いので進む。実測 約36.1
assert table[(256, 0.001)] > 200.0             # 30回しか更新していないので進まない。実測 約220

# (3) 「ちょうどいい lr」はバッチサイズに依存する:
#     batch_size=1 のベストは小さい lr 側、batch_size=256 のベストは大きい lr 側
best_lr_bs1 = min(lrs, key=lambda lr: table[(1, lr)])
best_lr_bs256 = min(lrs, key=lambda lr: table[(256, lr)])
assert best_lr_bs1 <= 0.005                    # 実測 0.001
assert best_lr_bs256 >= 0.02                   # 実測 0.03
assert best_lr_bs1 < best_lr_bs256

print()
print(f"batch_size=1   のベスト lr: {best_lr_bs1}")
print(f"batch_size=256 のベスト lr: {best_lr_bs256}")
print()
print("ok: すべての assert を通過しました")
