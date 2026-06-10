# 第4巻 第2章 2.4: 独立な成分を d 個足すと分散は d 倍に育つ — √d_k の伏線の実験
import numpy as np

rng = np.random.default_rng(42)

n = 100_000  # 1つの d につき、和を10万回作ってばらつきを測る

print(" d     平均      分散  (理論値)  標準偏差  (√d)")
for d in [1, 4, 16, 64]:
    parts = rng.choice([-1.0, 1.0], size=(n, d))  # 平均0・分散1の独立な成分 (n, d)
    S = parts.sum(axis=1)                         # d 個の和を n 回分 (n,)

    print(f"{d:3d}  {S.mean():+.4f}  {S.var():8.3f}  ({d:3d})    {S.std():7.3f}  ({np.sqrt(d):.3f})")

    assert abs(S.mean()) < 0.05 * np.sqrt(d)   # 平均は 0 のまま動かない
    assert np.allclose(S.var(), d, rtol=0.05)  # 分散は d 倍に育つ

    scaled = S / np.sqrt(d)                            # √d で割ると……
    assert np.allclose(scaled.var(), 1.0, atol=0.05)   # 分散1に戻る

print("ok: 和の分散は d 倍、√d で割れば分散1に戻ることを確認しました")
