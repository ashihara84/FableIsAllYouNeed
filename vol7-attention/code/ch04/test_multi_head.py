# 第7巻 第4章: multi_head.py のテスト
# 実行: python3 test_multi_head.py(すべての assert を通過すれば ok が出る)
import numpy as np

from multi_head import attention, combine_heads, multi_head_attention, split_heads

rng = np.random.default_rng(42)

# === テスト0: split_heads と combine_heads は互いに逆操作 =====================
X = rng.standard_normal((6, 8))         # 単語6個、d_model=8
B = split_heads(X, 4)                   # (6, 8) -> (4, 6, 2): (6, 2) 行列が4冊
assert B.shape == (4, 6, 2)
assert np.allclose(combine_heads(B), X)  # 分けて戻すと元どおり

# split は「列のブロック分け」: head i の中身は X の列 [i*d_k : (i+1)*d_k]
for i in range(4):
    assert np.allclose(B[i], X[:, i * 2:(i + 1) * 2])

# === テスト1: shape ===========================================================
n, m, d_model, h = 5, 7, 16, 4          # cross-attention を想定して n != m
X_q = rng.standard_normal((n, d_model))
X_kv = rng.standard_normal((m, d_model))
W_q, W_k, W_v, W_o = (rng.standard_normal((d_model, d_model)) for _ in range(4))

Y, A = multi_head_attention(X_q, X_kv, W_q, W_k, W_v, W_o, h)
assert Y.shape == (n, d_model)          # 出口は入口 X_q と同じ形(積み重ね可能)
assert A.shape == (h, n, m)             # 重みは head ごとに1枚、計 h 枚

# === テスト2: 各 head の重みは確率分布(非負・各行の和が1)===================
assert np.all(A >= 0)
assert np.allclose(A.sum(axis=-1), 1.0)

# === テスト3: mask した位置の重みは全 head で 0 ===============================
mask = np.ones((n, m), dtype=bool)
mask[:, -2:] = False                    # 後ろ2列は見てはいけない
Y_m, A_m = multi_head_attention(X_q, X_kv, W_q, W_k, W_v, W_o, h, mask=mask)
assert np.allclose(A_m[:, :, -2:], 0.0)         # 禁止位置の重みは全 head で 0
assert np.allclose(A_m.sum(axis=-1), 1.0)       # 残りの位置で再び和が 1
assert Y_m.shape == (n, d_model)

# === テスト4: h=1 のとき第3章の単独 attention と一致(TOC 指定)==============
Y1, A1 = multi_head_attention(X_q, X_kv, W_q, W_k, W_v, W_o, h=1)
out_single, w_single = attention(X_q @ W_q, X_kv @ W_k, X_kv @ W_v)
assert np.allclose(Y1, out_single @ W_o)        # 出力射影 W^O を除けば単独 attention
assert np.allclose(A1[0], w_single)             # 重みは完全一致

# mask 込みでも一致
Y1m, A1m = multi_head_attention(X_q, X_kv, W_q, W_k, W_v, W_o, h=1, mask=mask)
out_s_m, w_s_m = attention(X_q @ W_q, X_kv @ W_k, X_kv @ W_v, mask=mask)
assert np.allclose(Y1m, out_s_m @ W_o)
assert np.allclose(A1m[0], w_s_m)

# === テスト5: 束ね計算 = head ごとのループ計算(4.3 の整形の裏取り)===========
# 論文の head_i = Attention(X W_i^Q, X W_i^K, X W_i^V) を、W の列ブロックを
# 切り出して 1 head ずつ素直に計算し、束ね版と一致することを確かめる。
d_k = d_model // h
outs = []
for i in range(h):
    sl = slice(i * d_k, (i + 1) * d_k)  # W_i^Q は W_q の列ブロック i
    head_i, w_i = attention(X_q @ W_q[:, sl], X_kv @ W_k[:, sl], X_kv @ W_v[:, sl])
    assert np.allclose(w_i, A[i])       # head i の重みが束ね版の i 冊目と一致
    outs.append(head_i)
Y_loop = np.concatenate(outs, axis=1) @ W_o     # Concat(head_1, ..., head_h) W^O
assert np.allclose(Y_loop, Y)

# === テスト6: 論文の数字で通す(d_model=512, h=8, d_k=64)=====================
n8, m8 = 10, 12
Xq8 = rng.standard_normal((n8, 512))
Xkv8 = rng.standard_normal((m8, 512))
Wq8, Wk8, Wv8, Wo8 = (rng.standard_normal((512, 512)) * 0.03 for _ in range(4))
with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
    # macOS の Accelerate BLAS は、有限の値どうしの大きな行列積でも誤った
    # RuntimeWarning を出すことがある(NumPy の既知の問題)。結果は正しいので
    # 警告だけ抑え、直後に有限性を assert で確認する。
    Y8, A8 = multi_head_attention(Xq8, Xkv8, Wq8, Wk8, Wv8, Wo8, h=8)
assert np.all(np.isfinite(Y8)) and np.all(np.isfinite(A8))
assert Y8.shape == (n8, 512)
assert A8.shape == (8, n8, m8)
assert np.allclose(A8.sum(axis=-1), 1.0)

# パラメータ数: W^Q, W^K, W^V, W^O の4枚で 4 x 512 x 512 = 1,048,576
n_params = sum(W.size for W in (Wq8, Wk8, Wv8, Wo8))
assert n_params == 4 * 512 * 512 == 1048576

# === テスト7: バッチ軸を付けても動く(序章0.3の規格、組み立ては第8巻)========
batch = 3
Xb = rng.standard_normal((batch, n, d_model))
Yb, Ab = multi_head_attention(Xb, Xb, W_q, W_k, W_v, W_o, h)  # batch 付き self-attention
assert Yb.shape == (batch, n, d_model)
assert Ab.shape == (batch, h, n, n)             # 第1巻6.4で予告した4階テンソル
for b in range(batch):                          # 各系列を1本ずつ処理した結果と一致
    Y_b, A_b = multi_head_attention(Xb[b], Xb[b], W_q, W_k, W_v, W_o, h)
    assert np.allclose(Yb[b], Y_b) and np.allclose(Ab[b], A_b)

print("ok: すべての assert を通過しました")
print("  - split/combine は逆操作で、split は W の列ブロック分けと同じ")
print("  - 重みは head ごとに確率分布、mask した位置は全 head で 0")
print("  - h=1 は第3章の単独 attention と一致(出力射影 W^O を1枚かませた形)")
print("  - 束ねた一括計算は head ごとのループ計算と一致")
print("  - バッチ軸 (batch, seq, d_model) を付けてもそのまま動く")
