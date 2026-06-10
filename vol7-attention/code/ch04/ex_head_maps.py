# 第7巻 第4章 演習3: head ごとの attention マップを数値表で観察する
# 同じ入力・同じ文でも、head ごとに重みの分布(=どこを見るか)が違うことを見る。
# 実行: python3 ex_head_maps.py
import numpy as np

from multi_head import multi_head_attention

rng = np.random.default_rng(42)

n, d_model, h = 4, 8, 2                 # 単語4個の文、d_model=8 を 2 head に分割
X = rng.standard_normal((n, d_model))   # 文(乱数の埋め込みで代用)
W_q, W_k, W_v, W_o = (rng.standard_normal((d_model, d_model)) for _ in range(4))

Y, A = multi_head_attention(X, X, W_q, W_k, W_v, W_o, h)  # self-attention
assert A.shape == (h, n, n)

print("head ごとの attention マップ(行 = query 位置、列 = どこを見たか)")
for i in range(h):
    print()
    print("--- head " + str(i) + " ---")
    print(np.round(A[i], 2))

# 観察の言語化を assert で固定する:
# (1) どの head も各行は和が 1 の確率分布
assert np.allclose(A.sum(axis=-1), 1.0)
# (2) head 0 と head 1 の重みは一致しない(= 同じ文を「別の見方」で見ている)
#     W は乱数のままでも、射影が違えば見る場所はすでに違う。
#     学習はこの「違い」を意味のある違いに育てる(第8巻)。
assert not np.allclose(A[0], A[1], atol=0.1)
# (3) 各 head で「最も強く見ている位置」の並びも同一ではない
assert not np.array_equal(A[0].argmax(axis=-1), A[1].argmax(axis=-1))

print()
print("ok: head ごとに重みの分布が異なることを確認しました")
