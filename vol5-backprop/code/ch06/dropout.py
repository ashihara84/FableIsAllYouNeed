# 第5巻 第6章 6.4: dropout(訓練時のみ消す・逆スケーリングで期待値を保つ)
import numpy as np

rng = np.random.default_rng(42)


def dropout(x, p_drop, rng, training=True):
    """訓練時: 各要素を確率 p_drop で 0 にし、生き残りを 1/(1-p_drop) 倍する。
    推論時: 何もしない(逆スケーリングのおかげで、これで期待値が揃う)。"""
    if not training:
        return x, None
    mask = (rng.random(x.shape) >= p_drop) / (1.0 - p_drop)
    return x * mask, mask


# backward は mask を掛けるだけ: dropout は「定数 mask との要素ごとの積」なので
# d(x * mask)/dx = mask。消えた要素には勾配も流れない。

p_drop = 0.1   # 論文 5.4 の P_drop = 0.1
x = rng.normal(0, 1, size=(4, 8))

# --- 検証1: 推論時は完全に素通し ---
out_eval, _ = dropout(x, p_drop, rng, training=False)
assert out_eval is x

# --- 検証2: 訓練時、各要素は「0」か「x の 1/(1-p) 倍」のどちらかになる ---
out_train, mask = dropout(x, p_drop, rng, training=True)
dropped = (mask == 0.0)
assert np.all(out_train[dropped] == 0.0)
assert np.allclose(out_train[~dropped], x[~dropped] / (1.0 - p_drop))

# --- 検証3: 落ちる割合はおよそ p_drop ---
big = np.ones((1000, 1000))
_, mask_big = dropout(big, p_drop, rng)
drop_rate = (mask_big == 0.0).mean()
assert abs(drop_rate - p_drop) < 0.001
print("落ちた割合: {:.4f} (p_drop = {})".format(drop_rate, p_drop))

# --- 検証4: 逆スケーリングのおかげで、訓練時の出力の期待値は x に一致する ---
trials = 20000
acc = np.zeros_like(x)
for _ in range(trials):
    out, _ = dropout(x, p_drop, rng)
    acc += out
mean_out = acc / trials
assert np.allclose(mean_out, x, atol=0.02)
print("2万回の平均と x の最大ずれ: {:.4f}".format(np.abs(mean_out - x).max()))

print("すべての assert を通過しました")
