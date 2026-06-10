"""第4巻 第3章: 尤度関数の形と「対数が積を和に変える」ことの数値確認。

本文 3.2〜3.4 のコード片を1本にまとめたもの(本章は数式が主役のため、コードはこの最小限のみ)。
"""
import numpy as np

# --- 3.2 尤度関数 L(θ) = θ^7 (1-θ)^3 の頂上は θ = 0.7 ---
theta = np.linspace(0.001, 0.999, 999)  # (999,) θ の候補を 0.001 刻みで敷き詰める
L = theta**7 * (1 - theta)**3           # (999,) 各候補の尤度

assert np.isclose(theta[np.argmax(L)], 0.7)  # 山の頂上は θ = 0.7

# --- 3.3 困りごと1: 1000個の積はアンダーフローで消える ---
rng = np.random.default_rng(42)
p = rng.uniform(0.1, 0.9, size=1000)  # (1000,) 確率のつもりの数を1000個

assert np.prod(p) == 0.0              # 積は 0.0 ぴったり: 情報が完全に消えた

# --- 3.3 対数は頂上の場所を動かさない(単調性) ---
assert np.argmax(np.log(L)) == np.argmax(L)

# --- 3.3 対数の和なら生き残る ---
log_lik = np.sum(np.log(p))   # 積の対数 = 対数の和

assert np.isfinite(log_lik)   # 今度は何事もなく計算できる
print("sum of log p:", log_lik)  # -838.98...

# --- 3.4 (log x)' = 1/x の数値検算(第2巻第1章の中心差分) ---
h = 1e-5
for x in [0.5, 2.0, 4.0]:
    slope = (np.log(x + h) - np.log(x - h)) / (2 * h)
    assert np.isclose(slope, 1.0 / x)

print("all assertions passed")
