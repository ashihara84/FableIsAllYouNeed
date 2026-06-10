# 第4巻 第6章 6.4: softmax 回帰(多クラス分類)のフルスクラッチ実装
import numpy as np


def softmax_rows(Z):
    """(n, K) の各行に数値安定化版 softmax(6.2)を適用する。"""
    Z_shift = Z - Z.max(axis=1, keepdims=True)  # 行ごとの最大値シフト
    E = np.exp(Z_shift)
    return E / E.sum(axis=1, keepdims=True)


def one_hot(y, K):
    """整数ラベル (n,) を one-hot 行列 (n, K) にする。"""
    T = np.zeros((len(y), K))
    T[np.arange(len(y)), y] = 1.0
    return T


def forward(X, W, b):
    """モデル: スコア Z = X @ W + b を softmax で確率に。X:(n,d) W:(d,K) b:(K,) → (n,K)"""
    return softmax_rows(X @ W + b)


def cross_entropy(Y, T):
    """平均 cross-entropy(第5章)。Y:(n,K) 予測確率、T:(n,K) one-hot 正解。"""
    return float(-(T * np.log(Y)).sum(axis=1).mean())


def gradients(X, Y, T):
    """6.3 の結果: スコアへの勾配は (Y - T) / n。あとは第2巻の連鎖律で W と b へ。"""
    n = X.shape[0]
    dZ = (Y - T) / n         # (n, K)
    grad_W = X.T @ dZ        # (d, n) @ (n, K) → (d, K)
    grad_b = dZ.sum(axis=0)  # (K,)
    return grad_W, grad_b


rng = np.random.default_rng(42)

# --- データ: メール3分類(通常 / スパム / メルマガ)。第3巻エピローグの2クラスを3クラスに拡張 ---
n_per_class = 100
centers = np.array([
    [-2.0, -2.0],  # クラス0: 通常メール
    [2.0, 2.0],    # クラス1: スパム
    [-2.0, 2.0],   # クラス2: メルマガ
])
X = np.vstack([rng.normal(loc=c, scale=1.0, size=(n_per_class, 2)) for c in centers])
y = np.repeat(np.arange(3), n_per_class)
assert X.shape == (300, 2) and y.shape == (300,)

K = 3
T = one_hot(y, K)
assert T.shape == (300, 3)
assert np.allclose(T.sum(axis=1), 1.0)     # 各行は確率分布(ちょうど1つだけ1)
assert np.all(T.argmax(axis=1) == y)       # 1の位置 = 正解クラス

# --- 勾配チェック: 解析勾配 (Y - T) を数値微分(第2巻)と突き合わせる ---
W = rng.normal(scale=0.1, size=(2, K))
b = np.zeros(K)
grad_W, grad_b = gradients(X, forward(X, W, b), T)
eps = 1e-6
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        W_pos = W.copy(); W_pos[i, j] += eps
        W_neg = W.copy(); W_neg[i, j] -= eps
        num = (cross_entropy(forward(X, W_pos, b), T)
               - cross_entropy(forward(X, W_neg, b), T)) / (2 * eps)
        assert np.isclose(grad_W[i, j], num, rtol=1e-4, atol=1e-8)
print("勾配チェック OK: 解析勾配 (Y - T) は数値微分と一致")

# --- 学習ループ: forward → loss → gradient → update(第3巻と同じ4拍子) ---
lr = 0.5
losses = []
for step in range(300):
    Y = forward(X, W, b)
    loss = cross_entropy(Y, T)
    losses.append(loss)
    grad_W, grad_b = gradients(X, Y, T)
    W -= lr * grad_W
    b -= lr * grad_b

acc = float((forward(X, W, b).argmax(axis=1) == y).mean())
print("loss: {:.4f} → {:.4f}, accuracy: {:.3f}".format(losses[0], losses[-1], acc))
assert losses[-1] < losses[0] * 0.1  # 損失は1桁以上下がる
assert acc >= 0.95                   # 300件中ほぼ全部当たる
print("すべての assert を通過")
