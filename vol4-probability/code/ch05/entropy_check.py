import numpy as np


def entropy(p, base=2.0):
    """エントロピー H(p)。p は確率分布 (K,)。0 log 0 = 0 の慣習に従う"""
    p = np.asarray(p, dtype=float)
    mask = p > 0
    return -np.sum(p[mask] * np.log(p[mask])) / np.log(base)


def cross_entropy(p, q, base=2.0):
    """cross-entropy H(p, q)。p, q は確率分布 (K,)"""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    return -np.sum(p[mask] * np.log(q[mask])) / np.log(base)


def kl(p, q, base=2.0):
    """KL divergence = cross-entropy - entropy"""
    return cross_entropy(p, q, base) - entropy(p, base)


# --- 5.2 エントロピー ---
assert np.allclose(entropy([0.5, 0.5]), 1.0)          # コイン = 1 ビット
assert np.allclose(entropy([1.0, 0.0, 0.0]), 0.0)     # 確実 = 0 ビット
assert np.allclose(entropy([1/3, 1/3, 1/3]), np.log2(3))
assert np.allclose(entropy([0.5, 0.25, 0.25]), 1.5)   # 本文の例3

# --- 5.3 / 5.4 cross-entropy と KL ---
p = np.array([0.5, 0.25, 0.25])
q = np.array([0.25, 0.5, 0.25])
assert np.allclose(cross_entropy(p, q), 1.75)         # 本文の数値例
assert np.allclose(kl(p, q), 0.25)                    # 余分な損 0.25 ビット
assert np.allclose(cross_entropy(p, p), entropy(p))   # q = p なら損ゼロ
assert np.allclose(kl(p, p), 0.0)

# KL >= 0(したがって H(p,q) >= H(p))をランダムな分布 1000 組で確認
rng = np.random.default_rng(42)
for _ in range(1000):
    a = rng.random(5); a /= a.sum()
    b = rng.random(5); b /= b.sum()
    assert kl(a, b) >= 0.0

# 非対称性: KL(p||q) と KL(q||p) は一般に一致しない(演習3の分布)
p2 = np.array([0.5, 0.5])
q2 = np.array([0.25, 0.75])
assert not np.allclose(kl(p2, q2), kl(q2, p2))

# --- 5.5 log loss = 2クラスの cross-entropy(式 5.4)---
for t in (0, 1):
    for y in (0.1, 0.5, 0.8, 0.99):
        log_loss = -(t * np.log(y) + (1 - t) * np.log(1 - y))
        ce = cross_entropy([t, 1 - t], [y, 1 - y], base=np.e)
        assert np.allclose(log_loss, ce)

print("ok: 第5章の数値はすべて検算が通りました")
