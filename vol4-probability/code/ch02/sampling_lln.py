# 第4巻 第2章 2.4: サンプリングで期待値・分散を体感する — 大数の法則の数値版
import numpy as np

rng = np.random.default_rng(42)

# 「明日の天気は」に続く単語の分布(2.1節の例)
# X = 次に来る単語の文字数
lengths = np.array([2.0, 1.0, 3.0])  # 晴れ / 雨 / くもり の文字数
probs = np.array([0.5, 0.3, 0.2])    # それぞれの確率

assert np.allclose(probs.sum(), 1.0)  # 確率は全部足すと1(第1章のルール)

# --- 手計算の再現: 定義どおりに期待値と分散を出す ---
E_X = np.sum(lengths * probs)                 # 期待値: 値 × 確率 の総和
Var_X = np.sum((lengths - E_X) ** 2 * probs)  # 分散: 偏差の2乗の期待値

print(f"期待値(手計算): {E_X:.2f}")    # 1.90
print(f"分散  (手計算): {Var_X:.2f}")  # 0.49

assert np.allclose(E_X, 1.9)
assert np.allclose(Var_X, 0.49)

# --- サンプリング: この分布から実際に単語を引いてみる ---
for n in [10, 1_000, 100_000]:
    samples = rng.choice(lengths, size=n, p=probs)  # (n,)
    print(f"n={n:7d}  標本平均 {samples.mean():.4f}  標本分散 {samples.var():.4f}")

# n を大きくすれば、手計算の値にいくらでも近づく(大数の法則)
samples = rng.choice(lengths, size=1_000_000, p=probs)  # (1000000,)
assert np.allclose(samples.mean(), E_X, atol=0.005)
assert np.allclose(samples.var(), Var_X, atol=0.005)

print("ok: 標本平均 → 期待値、標本分散 → 分散 への接近を確認しました")
