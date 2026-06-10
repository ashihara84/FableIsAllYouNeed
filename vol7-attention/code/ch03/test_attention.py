# 第7巻 第3章 3.6: Scaled Dot-Product Attention のテスト
# python3 test_attention.py で全 assert が通ること。
# この単体テスト群が、第8巻で全部品を組み立てるときの結合テストの前提になる。
import numpy as np

from attention import attention, causal_mask, padding_mask, softmax

rng = np.random.default_rng(42)


# --- テスト1: shape — 式(1)の配管 (n,d_k) @ (d_k,m) → (n,m) → @ (m,d_v) → (n,d_v) ---
n, m, d_k, d_v = 4, 6, 8, 5          # n本の query が m 本の key/value を見る。d_k ≠ d_v でも通る
Q = rng.standard_normal((n, d_k))
K = rng.standard_normal((m, d_k))
V = rng.standard_normal((m, d_v))

output, weights = attention(Q, K, V)
assert weights.shape == (n, m)       # 類似度の総当たり表(第1巻終章)と同じ shape
assert output.shape == (n, d_v)      # 出力の行数は query の本数、列数は value の次元
print("ok: shape — (n, d_k)=({}, {}) と (m, d_v)=({}, {}) から出力 ({}, {})".format(
    n, d_k, m, d_v, n, d_v))


# --- テスト2: 各行の重みは確率分布(全成分が正で、和が1) ---
assert np.all(weights > 0)
assert np.allclose(weights.sum(axis=-1), 1.0)
print("ok: 重みの各行は和が1の確率分布")


# --- テスト3: 出力の各行は value ベクトルの凸結合 → 成分ごとの min/max に挟まれる ---
assert np.all(output >= V.min(axis=0) - 1e-12)
assert np.all(output <= V.max(axis=0) + 1e-12)
print("ok: 出力は value の重み付き平均(min と max の間)に収まる")


# --- テスト4: 数値安定性 — スコアが大きくても nan / inf を出さない(第4巻6.2) ---
big, _ = attention(1000.0 * Q, 1000.0 * K, V)
assert np.all(np.isfinite(big))
print("ok: 大きなスコアでもオーバーフローしない(数値安定版 softmax)")


# --- テスト5: padding mask — 埋め草の位置の重みは0、本物だけの計算と完全一致 ---
m_real = 4                                       # 6本の key のうち、最後の2本は埋め草とする
is_real = np.array([True] * m_real + [False] * (m - m_real))
mask_pad = padding_mask(is_real)                 # (1, m) — 全 query に同じ禁止が掛かる

out_pad, w_pad = attention(Q, K, V, mask=mask_pad)
assert np.allclose(w_pad[:, m_real:], 0.0)       # 埋め草の位置の重みは厳密に0
assert np.allclose(w_pad.sum(axis=-1), 1.0)      # 残りの位置だけで再び和が1

# 最強の検証: 「埋め草を mask した結果」=「埋め草が最初から存在しない結果」
out_trim, _ = attention(Q, K[:m_real], V[:m_real])
assert np.allclose(out_pad, out_trim)
print("ok: padding mask — 埋め草の重みが0になり、埋め草なしの計算と一致")


# --- テスト6: causal mask — 未来(上三角)の重みは0、過去だけの計算と一致 ---
Qs = rng.standard_normal((n, d_k))               # self-attention の設定: n = m
Ks = rng.standard_normal((n, d_k))
Vs = rng.standard_normal((n, d_v))

out_c, w_c = attention(Qs, Ks, Vs, mask=causal_mask(n))
assert np.allclose(w_c[np.triu_indices(n, k=1)], 0.0)   # 未来の重みはすべて0
assert np.allclose(out_c[0], Vs[0])              # 位置0は1択 → 重み1 → 出力は V の0行目そのもの

# 各位置 i の出力は「key/value を位置 i までに切り詰めた計算」と一致(未来は存在しないのと同じ)
for i in range(n):
    out_i, _ = attention(Qs[i:i + 1], Ks[:i + 1], Vs[:i + 1])
    assert np.allclose(out_c[i], out_i[0])
print("ok: causal mask — 未来の重みが0になり、過去だけの計算と一致")


# --- テスト7: 2×2 の手計算例と一致(演習の答え合わせ。数値は本文の手計算と同じ) ---
Q2 = np.array([[1.0, 0.0],
               [0.0, 1.0]])
K2 = np.array([[2.0, 0.0],
               [0.0, 2.0]])
V2 = np.array([[10.0, 0.0],
               [0.0, 20.0]])

out2, w2 = attention(Q2, K2, V2)

# 手計算: QK^T = [[2,0],[0,2]], √d_k = √2 で割って [[√2, 0], [0, √2]]
# 1行目の softmax: (e^√2, e^0) / (e^√2 + 1) = (0.80444, 0.19556)
w_big = np.exp(np.sqrt(2)) / (np.exp(np.sqrt(2)) + 1.0)  # ≈ 0.80444
assert np.allclose(w2, [[w_big, 1 - w_big],
                        [1 - w_big, w_big]])
assert np.allclose(w2, [[0.80444, 0.19556],
                        [0.19556, 0.80444]], atol=1e-5)
# 出力 = 重み付き和: 1行目 = 0.80444*[10,0] + 0.19556*[0,20] = [8.0444, 3.9112]
assert np.allclose(out2, [[8.0444, 3.9112],
                          [1.9556, 16.0888]], atol=1e-3)
print("ok: 2×2 の手計算例と一致(重みも出力も)")


# --- テスト8: mask なし = 全部 True の mask(「mask は禁止の追加」であることの確認) ---
out_all, w_all = attention(Q, K, V, mask=np.ones((n, m), dtype=bool))
assert np.allclose(out_all, output) and np.allclose(w_all, weights)
print("ok: 全位置 True の mask は mask なしと同じ")


# --- テスト9: バッチ次元(第1巻6.4「行列の束」)— 束で計算しても1枚ずつと同じ ---
B = 3
Qb = rng.standard_normal((B, n, d_k))
Kb = rng.standard_normal((B, m, d_k))
Vb = rng.standard_normal((B, m, d_v))
out_b, w_b = attention(Qb, Kb, Vb)
assert out_b.shape == (B, n, d_v) and w_b.shape == (B, n, m)
for b in range(B):
    out_1, w_1 = attention(Qb[b], Kb[b], Vb[b])
    assert np.allclose(out_b[b], out_1) and np.allclose(w_b[b], w_1)
print("ok: バッチ入力 (B, n, d_k) でも1枚ずつの計算と一致(第4章への布石)")


# --- テスト10: softmax 単体 — 安定版とシフト不変性(第4巻6.2の再確認) ---
z = np.array([[1000.0, 1001.0, 1002.0]])
p = softmax(z)
assert np.all(np.isfinite(p)) and np.allclose(p.sum(axis=-1), 1.0)
assert np.allclose(p, softmax(z - 1000.0))       # シフト不変性
print("ok: softmax — 数値安定・シフト不変")

print()
print("すべての assert を通過しました: 式(1)の実装は論文の主張どおりに動いています")
