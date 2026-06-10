# 第4巻 第2章 演習2 略解: 演習1の手計算をサンプリングで一致確認する
import numpy as np

rng = np.random.default_rng(42)

# 「明日の天気は」の改訂版: 雨 / 晴れ / かみなり
lengths = np.array([1.0, 2.0, 4.0])   # 単語の文字数
probs = np.array([0.5, 0.25, 0.25])   # それぞれの確率

# 手計算(演習1)の値
E_X = np.sum(lengths * probs)
Var_X = np.sum((lengths - E_X) ** 2 * probs)
assert np.allclose(E_X, 2.0)
assert np.allclose(Var_X, 1.5)

# サンプリングで一致確認
samples = rng.choice(lengths, size=1_000_000, p=probs)  # (1000000,)
print(f"標本平均 {samples.mean():.4f}  (手計算 {E_X:.1f})")
print(f"標本分散 {samples.var():.4f}  (手計算 {Var_X:.1f})")

assert np.allclose(samples.mean(), E_X, atol=0.01)
assert np.allclose(samples.var(), Var_X, atol=0.01)
print("ok: 手計算とシミュレーションが一致しました")
