# 第4巻 第6章 6.2 & 6.5: 素朴な softmax のオーバーフローを観測 → 最大値シフトで直す → 温度
import numpy as np


def softmax_naive(z):
    """定義をそのまま書いた softmax。あとで壊れるところを観測するための版。"""
    e = np.exp(z)
    return e / e.sum()


def softmax(z):
    """数値安定化版 softmax。最大値を引いてから exp する(答えは naive 版と同じ)。"""
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())  # 最大成分が 0 になる → exp は高々 1 → オーバーフロー不可能
    return e / e.sum()


def sigmoid(z):
    """第3巻エピローグ以来のシグモイド。"""
    return 1.0 / (1.0 + np.exp(-z))


# --- 小さなスコアなら naive 版で何も問題ない ---
z = np.array([2.0, 1.0, 0.0])
p = softmax_naive(z)
assert np.all(p > 0) and np.isclose(p.sum(), 1.0)
print("softmax([2, 1, 0]) =", np.round(p, 4))

# --- 大きなスコアで壊す: exp(1000) は float64 に収まらない(上限はおよそ exp(709)) ---
z_big = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over="ignore", invalid="ignore"):  # 警告表示を抑えて nan を観測する
    p_naive = softmax_naive(z_big)
assert np.isnan(p_naive).any()  # inf / inf = nan
print("naive 版に [1000, 1001, 1002] を入れると:", p_naive)

# --- 最大値シフトで直す ---
p_stable = softmax(z_big)
assert np.all(p_stable > 0) and np.isclose(p_stable.sum(), 1.0)
print("安定版に同じものを入れると      :", np.round(p_stable, 4))

# シフト不変性: [1000, 1001, 1002] の softmax は [-2, -1, 0] の softmax と同じ
assert np.allclose(p_stable, softmax(np.array([-2.0, -1.0, 0.0])))

# 小さなスコアでは naive 版と完全に一致する(直したのは計算手順だけで、関数は同じ)
assert np.allclose(softmax(z), softmax_naive(z))

# --- K=2 の softmax はシグモイドそのもの(片方のスコアを 0 に固定した形) ---
for s in [-3.0, -0.5, 0.0, 1.2, 4.0]:
    p2 = softmax(np.array([s, 0.0]))
    assert np.isclose(p2[0], sigmoid(s))
print("K=2 の softmax はシグモイドと一致")


# --- 6.5: 温度 ---
def softmax_with_temperature(z, tau):
    """温度 tau 付き softmax。tau < 1 で尖り、tau > 1 でなだらかになる。"""
    return softmax(np.asarray(z, dtype=float) / tau)


def entropy(p):
    """第5章のエントロピー(驚きの期待値)。分布の不確かさの量。"""
    return float(-(p * np.log(p)).sum())


z = np.array([2.0, 1.0, 0.0])
assert np.allclose(softmax_with_temperature(z, 1.0), softmax(z))  # tau=1 は素の softmax

prev_h = -1.0
for tau in [0.1, 0.5, 1.0, 2.0, 10.0]:
    p = softmax_with_temperature(z, tau)
    h = entropy(p)
    assert h > prev_h  # 温度を上げるほどエントロピー(不確かさ)が単調に増える
    prev_h = h
    print("tau={:>4}: {}  entropy={:.3f}".format(tau, np.round(p, 3), h))

# 極端な温度: tau→0 で one-hot(hardmax)に、tau→∞ で一様分布に近づく
assert softmax_with_temperature(z, 0.01)[0] > 0.999
assert np.allclose(softmax_with_temperature(z, 1000.0), np.ones(3) / 3, atol=1e-3)

# 温度は順位を変えない: argmax はどの温度でも同じ
for tau in [0.1, 0.5, 1.0, 2.0, 10.0]:
    assert softmax_with_temperature(z, tau).argmax() == z.argmax()

print("すべての assert を通過")
