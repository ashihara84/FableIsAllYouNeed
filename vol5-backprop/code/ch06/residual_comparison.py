# 第5巻 第6章 6.5: residual / layer norm / dropout の有無で各層の勾配ノルムを比較する
# 4変種とも同一の重み・同一のデータ。違いはブロックのつなぎ方だけ
import numpy as np

rng = np.random.default_rng(42)

n, d = 64, 64
depth = 10
sigma_w = 0.1      # 6.1 と同じ「一見無難な」初期化(条件を揃える)
p_drop = 0.1       # 論文 5.4 の P_drop = 0.1


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# --- 6.3 の layer_norm / layer_norm_backward をそのまま再掲 ---
def layer_norm(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mu) * inv_std
    out = gamma * x_hat + beta
    cache = (x_hat, inv_std, gamma)
    return out, cache


def layer_norm_backward(dout, cache):
    x_hat, inv_std, gamma = cache
    dgamma = (dout * x_hat).sum(axis=0)
    dbeta = dout.sum(axis=0)
    dx_hat = dout * gamma
    dx = inv_std * (dx_hat
                    - dx_hat.mean(axis=-1, keepdims=True)
                    - x_hat * (dx_hat * x_hat).mean(axis=-1, keepdims=True))
    return dx, dgamma, dbeta


# --- データと重み(全変種で共通)---
X = rng.normal(0, 1, size=(n, d))
y = rng.normal(0, 1, size=(n, 1))
params = []
for _ in range(depth):
    params.append({
        "W1": rng.normal(0, sigma_w, size=(d, d)), "b1": np.zeros(d),
        "W2": rng.normal(0, sigma_w, size=(d, d)), "b2": np.zeros(d),
        "gamma": np.ones(d), "beta": np.zeros(d),
    })
W_out = rng.normal(0, sigma_w, size=(d, 1))

VARIANTS = ["plain", "residual", "residual+ln", "residual+ln+dropout"]


def forward_backward(variant, mask_rng):
    """各ブロックの ||grad_W1|| の配列と、各ブロック出力の標準偏差の配列を返す"""
    caches = []
    h_stds = []
    h = X
    for prm in params:
        x_in = h
        z1 = x_in @ prm["W1"] + prm["b1"]
        a = sigmoid(z1)
        f = a @ prm["W2"] + prm["b2"]          # 論文の FFN と同じ形(第2章)
        mask = None
        if variant == "residual+ln+dropout":
            mask = (mask_rng.random(f.shape) >= p_drop) / (1.0 - p_drop)
            f = f * mask                       # LayerNorm(x + Dropout(Sublayer(x)))
        if variant == "plain":
            h, ln_cache = f, None
        elif variant == "residual":
            h, ln_cache = x_in + f, None
        else:
            h, ln_cache = layer_norm(x_in + f, prm["gamma"], prm["beta"])
        caches.append((x_in, a, mask, ln_cache))
        h_stds.append(h.std())
    y_pred = h @ W_out
    loss_grad = 2.0 * (y_pred - y) / n         # (n, 1)

    delta = loss_grad @ W_out.T                # (n, d)
    norms = [0.0] * depth
    for l in range(depth - 1, -1, -1):
        x_in, a, mask, ln_cache = caches[l]
        prm = params[l]
        if ln_cache is not None:
            delta, _, _ = layer_norm_backward(delta, ln_cache)
        df = delta                             # 「和」の backward: f 側にも素通り側にも delta
        if mask is not None:
            df = df * mask
        da = df @ prm["W2"].T
        dz1 = da * a * (1.0 - a)
        norms[l] = np.linalg.norm(x_in.T @ dz1)    # ||grad_W1||
        dx_f = dz1 @ prm["W1"].T
        delta = dx_f if variant == "plain" else delta + dx_f
    return np.array(norms), np.array(h_stds)


results = {}
stds = {}
for v in VARIANTS:
    results[v], stds[v] = forward_backward(v, mask_rng=np.random.default_rng(7))

# --- 表: 各ブロックの ||grad_W1|| ---
header = "ブロック  " + "  ".join("{:>11s}".format(v) for v in VARIANTS)
print(header)
for l in range(depth):
    row = "{:6d}    ".format(l + 1)
    row += "  ".join("{:11.3e}".format(results[v][l]) for v in VARIANTS)
    print(row)

# --- 検証: 本節の主張をデータで確認する ---
ratio = {v: results[v][0] / results[v][-1] for v in VARIANTS}
for v in VARIANTS:
    print("{:>20s}: 第1/第10ブロックの勾配ノルム比 = {:.2e}".format(v, ratio[v]))

# (1) plain では勾配が消失する(6.1 の再現): 入口は出口の 1e-5 倍未満
assert ratio["plain"] < 1e-5

# (2) residual を入れると、入口と出口が同じ桁に並ぶ
for v in ["residual", "residual+ln", "residual+ln+dropout"]:
    assert ratio[v] > 0.1, v
    # 全ブロックが2桁以内に収まる(どの層も学習に参加できる)
    assert results[v].max() / results[v].min() < 100.0, v

# (3) plain に対して、residual は入口の層の勾配を桁違いに回復させる
assert results["residual"][0] > 1e4 * results["plain"][0]

# (4) residual だけでは活性のスケールが膨らむ(和の分散は足し算 — 第4巻2章)
assert stds["residual"][-1] > 1.3 * stds["residual"][0]

# (5) layer norm を入れると、何段重ねてもスケールは 1 のまま
assert np.allclose(stds["residual+ln"], 1.0, atol=0.01)

print("residual の出力 std: 1ブロック目 {:.2f} → 10ブロック目 {:.2f}".format(
    stds["residual"][0], stds["residual"][-1]))
print("residual+ln の出力 std: 全ブロックで {:.3f}〜{:.3f}".format(
    stds["residual+ln"].min(), stds["residual+ln"].max()))
print("すべての assert を通過しました")
