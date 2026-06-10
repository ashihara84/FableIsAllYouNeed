# 第5巻 第6章 6.3: layer norm の実装(forward と手動 backward、数値微分で検算)
import numpy as np

rng = np.random.default_rng(42)


def layer_norm(x, gamma, beta, eps=1e-5):
    """各行(各サンプル)を平均0・分散1に整え、gamma で伸縮、beta で平行移動する。
    x: (n, d), gamma: (d,), beta: (d,)"""
    mu = x.mean(axis=-1, keepdims=True)        # (n, 1) 各行の平均(第4巻2章)
    var = x.var(axis=-1, keepdims=True)        # (n, 1) 各行の分散(第4巻2章)
    inv_std = 1.0 / np.sqrt(var + eps)         # eps はゼロ割り防止
    x_hat = (x - mu) * inv_std                 # 平均0・分散1に標準化
    out = gamma * x_hat + beta                 # 学習可能な伸縮とシフト
    cache = (x_hat, inv_std, gamma)
    return out, cache


def layer_norm_backward(dout, cache):
    """mu と var も x の関数なので、x への勾配には補正項が2つ付く"""
    x_hat, inv_std, gamma = cache
    dgamma = (dout * x_hat).sum(axis=0)
    dbeta = dout.sum(axis=0)
    dx_hat = dout * gamma
    dx = inv_std * (dx_hat
                    - dx_hat.mean(axis=-1, keepdims=True)
                    - x_hat * (dx_hat * x_hat).mean(axis=-1, keepdims=True))
    return dx, dgamma, dbeta


# --- 検証1: 出力は本当に各行が平均0・分散1か(gamma=1, beta=0 のとき) ---
n, d = 8, 16
x = rng.normal(3.0, 2.0, size=(n, d))      # わざと平均3・標準偏差2のずれたデータ
gamma = np.ones(d)
beta = np.zeros(d)
out, cache = layer_norm(x, gamma, beta)
assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-12)
assert np.allclose(out.std(axis=-1), 1.0, atol=1e-3)   # eps の分だけ僅かに 1 未満

# --- 検証2: 手動 backward を数値微分(第2巻1章)で検算する ---
R = rng.normal(0, 1, size=(n, d))          # L = Σ(out ⊙ R)


def loss_fn():
    out, _ = layer_norm(x, gamma, beta)
    return np.sum(out * R)


dx, dgamma, dbeta = layer_norm_backward(R, cache)


def numerical_grad(param):
    g = np.zeros_like(param)
    flat = param.reshape(-1)
    gf = g.reshape(-1)
    eps = 1e-5
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + eps
        fp = loss_fn()
        flat[i] = old - eps
        fm = loss_fn()
        flat[i] = old
        gf[i] = (fp - fm) / (2 * eps)
    return g


assert np.allclose(dx, numerical_grad(x), atol=1e-6)
assert np.allclose(dgamma, numerical_grad(gamma), atol=1e-6)
assert np.allclose(dbeta, numerical_grad(beta), atol=1e-6)

print("すべての assert を通過しました")
