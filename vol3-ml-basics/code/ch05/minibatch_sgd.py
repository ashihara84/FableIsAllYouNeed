# 第3巻 第5章: ミニバッチ — データを少しずつ食べる
# 5.1〜5.4 の実験を1ファイルにまとめたもの。
# 全データ / 1件ずつ / ミニバッチ の3流儀を同じ訓練ループで切り替えて比較する。
import numpy as np

rng = np.random.default_rng(42)

# --- 第2章と同じ生成規則の合成データ(模試で人数が 256 人に増えた) ---
N = 256
X = rng.uniform(0, 9, size=(N, 1))             # 勉強時間 (N, 1)
y = 7.0 * X + 20.0 + rng.normal(0, 6.0, size=(N, 1))  # 点数 (N, 1)  真の規則: w=7, b=20


def predict(X, w, b):
    """線形回帰の予測: (n, 1) @ (1, 1) + スカラー -> (n, 1)"""
    return X @ w + b


def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def gradients(X, y_true, w, b):
    """第4章で手導出した MSE の勾配。X は (n, 1)、n はバッチの件数"""
    n = X.shape[0]
    err = predict(X, w, b) - y_true            # (n, 1)
    grad_w = (2.0 / n) * (X.T @ err)           # (1, 1)
    grad_b = (2.0 / n) * np.sum(err)           # スカラー
    return grad_w, grad_b


def train(batch_size, n_epochs, lr, shuffle=True, seed=0):
    """ミニバッチSGD。batch_size=N なら全データ、=1 なら1件ずつと同じ。
    history は「各エポック終了時の全データでの loss」のリスト。"""
    rng_shuffle = np.random.default_rng(seed)  # シャッフル専用の乱数(データとは別)
    w = np.zeros((1, 1))                       # (1, 1)
    b = 0.0
    history = []
    n_steps = 0
    for epoch in range(n_epochs):
        idx = np.arange(N)
        if shuffle:
            idx = rng_shuffle.permutation(N)   # エポックごとに順番を混ぜる
        for start in range(0, N, batch_size):
            batch = idx[start:start + batch_size]  # 端数が出たら小さいバッチになる
            grad_w, grad_b = gradients(X[batch], y[batch], w, b)
            w = w - lr * grad_w
            b = b - lr * grad_b
            n_steps += 1
        history.append(mse(predict(X, w, b), y))
    return w, b, history, n_steps


# === 5.1 の確認: 全データの勾配は「1件ずつの勾配の平均」である ===
w0 = np.zeros((1, 1))
b0 = 0.0
grad_w_full, grad_b_full = gradients(X, y, w0, b0)

per_sample = [gradients(X[i:i + 1], y[i:i + 1], w0, b0) for i in range(N)]
assert np.allclose(grad_w_full, np.mean([gw for gw, gb in per_sample], axis=0))
assert np.allclose(grad_b_full, np.mean([gb for gw, gb in per_sample]))

# ミニバッチの勾配も同じ作り(バッチ内の平均)なので、全データ勾配の「推定値」になる
batch = np.arange(32)
grad_w_mb, grad_b_mb = gradients(X[batch], y[batch], w0, b0)
assert grad_w_mb.shape == grad_w_full.shape    # 形は同じ。中身は近いが一致はしない
assert not np.allclose(grad_w_mb, grad_w_full)

# === 5.3 の確認: epoch / step / shuffle の会計 ===
# N=256, batch_size=32 -> 1エポック = 256/32 = 8 ステップ。5エポックで 40 ステップ
_, _, _, n_steps = train(batch_size=32, n_epochs=5, lr=0.01)
assert n_steps == 40

# シャッフルは「順番を混ぜる」だけで、1エポックに全データをちょうど1回ずつ使う
idx = np.random.default_rng(0).permutation(N)
assert sorted(idx) == list(range(N))

# === 5.4 の実験: バッチサイズを変えて、同じ1エポック(データ1周)を比較 ===
results_1epoch = {}
for batch_size in [256, 32, 1]:
    _, _, history, n_steps = train(batch_size=batch_size, n_epochs=1, lr=0.01)
    results_1epoch[batch_size] = history[-1]
    print(f"batch_size={batch_size:>3}  steps={n_steps:>3}  loss={history[-1]:.3f}")

# 同じ「データ1周」でも、更新回数が多いほど loss が下がっている
assert results_1epoch[1] < results_1epoch[32] < results_1epoch[256]
assert results_1epoch[256] > 500.0             # 全データ(1更新)はまだ遠い。実測 約718
assert results_1epoch[32] < 150.0              # ミニバッチ(8更新)。実測 約117
assert results_1epoch[1] < 50.0                # 1件ずつ(256更新)。実測 約44.5(床の近く)

# === 5.2 の確認: 長く走らせれば3流儀とも収束するが、到達の速さが違う ===
final = {}
for batch_size in [256, 32, 1]:
    w, b, history, _ = train(batch_size=batch_size, n_epochs=150, lr=0.01)
    final[batch_size] = (w, b, history)

# ミニバッチと1件ずつは、150エポックでばらつきの床(およそ36 = 6^2)の近くまで到達
for batch_size in [32, 1]:
    w, b, history = final[batch_size]
    assert history[-1] < 38.0                  # 実測: 32 -> 35.92, 1 -> 36.16
    assert np.allclose(w, 7.0, atol=0.2)       # 正解 w=7 のすぐそば
    assert abs(b - 20.0) < 0.6                 # 正解 b=20 のすぐそば

# 全データは同じ150エポック(=たった150更新)ではまだ途中(実測 約56.5)
assert final[256][2][-1] > 50.0

print("ok: すべての assert を通過しました")
