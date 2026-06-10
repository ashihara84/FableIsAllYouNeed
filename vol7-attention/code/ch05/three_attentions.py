"""第7巻 第5章: 3.2.3 — attention の3つの使い方の呼び分けテスト。

論文 Section 3.2.3 の3つの使い方
  (a) encoder self-attention        : Q = K = V = encoder の前層出力。mask なし
  (b) decoder masked self-attention : Q = K = V = decoder の前層出力 + causal mask
  (c) encoder-decoder (cross) attention : Q = decoder 側、K = V = encoder 出力。mask なし
について、入力と mask の組合せが正しいことを assert で確認する。

python3 three_attentions.py で全 assert が通る。
"""
import os
import sys

import numpy as np

# 第3章 attention.py / 第4章 multi_head.py(並列執筆中)があれば import する
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "ch03"))
sys.path.insert(0, os.path.join(_HERE, "..", "ch04"))

try:
    from attention import attention  # 第3章 3.4 の単体実装
except ImportError:
    # 第3章 attention.py が無い単体配布時のための予備実装(通常は上の import が使われる)
    def softmax(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - x_max)
        return e / np.sum(e, axis=axis, keepdims=True)

    def attention(Q, K, V, mask=None):
        """式(1): softmax(QK^T / sqrt(d_k)) V。

        Q: (n_q, d_k), K: (n_kv, d_k), V: (n_kv, d_v)
        mask: bool (n_q, n_kv)。True = 見てよい位置。False の位置は softmax の
              手前で -inf に置き換える(論文 3.2.3 "setting to -inf")
        返り値: (出力 (n_q, d_v), 重み (n_q, n_kv))
        """
        d_k = Q.shape[-1]
        scores = Q @ K.T / np.sqrt(d_k)              # (n_q, n_kv)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        weights = softmax(scores, axis=-1)           # 行ごとの和が1
        return weights @ V, weights

try:
    from multi_head import multi_head_attention  # 第4章 4.4 の単体実装
except ImportError:
    # 第4章 multi_head.py が無い単体配布時のための予備実装(通常は上の import が使われる)
    def multi_head_attention(X_q, X_kv, W_Q, W_K, W_V, W_O, h, mask=None):
        """multi-head attention の最小 forward(batch なし・1系列)。

        X_q : (n_q, d_model)  — Q の出どころ
        X_kv: (n_kv, d_model) — K と V の出どころ(self なら X_q と同じ配列)
        返り値: (出力 (n_q, d_model), 重み (h, n_q, n_kv))
        """
        n_q, d_model = X_q.shape
        d_k = d_model // h
        Q = X_q @ W_Q                                # (n_q, d_model)
        K = X_kv @ W_K                               # (n_kv, d_model)
        V = X_kv @ W_V                               # (n_kv, d_model)
        heads, all_w = [], []
        for i in range(h):
            sl = slice(i * d_k, (i + 1) * d_k)       # head i の担当次元
            out_i, w_i = attention(Q[:, sl], K[:, sl], V[:, sl], mask=mask)
            heads.append(out_i)
            all_w.append(w_i)
        concat = np.concatenate(heads, axis=-1)      # (n_q, d_model)
        return concat @ W_O, np.stack(all_w)         # 重みは (h, n_q, n_kv)


# ---- 3つの使い方は「同じ関数、違う引数」。違いはこの3つの wrapper がすべて ----

def causal_mask(n):
    """位置 i から見てよいのは j <= i(下三角が True)。第3章 3.5 の causal mask"""
    return np.tril(np.ones((n, n), dtype=bool))


def encoder_self_attention(enc_x):
    """(a) Q = K = V = encoder の前層出力。mask なし(全位置が全位置を見る)"""
    return attention(enc_x, enc_x, enc_x, mask=None)


def decoder_masked_self_attention(dec_x):
    """(b) Q = K = V = decoder の前層出力。causal mask で未来(j > i)を遮断"""
    m = dec_x.shape[0]
    return attention(dec_x, dec_x, dec_x, mask=causal_mask(m))


def cross_attention(dec_x, enc_out):
    """(c) Q だけ decoder 側。K = V = encoder スタックの最終出力。mask なし"""
    return attention(dec_x, enc_out, enc_out, mask=None)


# ---- テスト ----

rng = np.random.default_rng(42)
d_model = 8
n_enc, n_dec = 5, 3   # encoder 5トークン、decoder 3トークン。長さを変えて出どころを判別する
enc_x = rng.normal(size=(n_enc, d_model))    # encoder のある層への入力
dec_x = rng.normal(size=(n_dec, d_model))    # decoder のある層への入力
enc_out = rng.normal(size=(n_enc, d_model))  # encoder スタックの最終出力(のつもり)

# --- (a) encoder self-attention: Q = K = V の出どころが同じ ---
out_a, w_a = encoder_self_attention(enc_x)
assert out_a.shape == (n_enc, d_model)
assert w_a.shape == (n_enc, n_enc)                 # 自分 × 自分なので正方形
assert np.allclose(w_a.sum(axis=-1), 1.0)
assert np.all(w_a > 0)                             # mask なし: どの位置も全位置を見る
# 同じ enc_x を Q, K, V に3回渡した結果と完全一致(= self の定義そのもの)
out_ref, w_ref = attention(enc_x, enc_x, enc_x)
assert np.allclose(out_a, out_ref) and np.allclose(w_a, w_ref)

# --- (b) decoder masked self-attention: 未来の重みが厳密に 0 ---
out_b, w_b = decoder_masked_self_attention(dec_x)
assert w_b.shape == (n_dec, n_dec)
assert np.allclose(w_b.sum(axis=-1), 1.0)          # mask があっても行和は1
assert np.all(w_b[np.triu_indices(n_dec, k=1)] == 0.0)  # 上三角(未来)が完全に0
assert np.isclose(w_b[0, 0], 1.0)                  # 先頭の位置は自分しか見られない
# 自己回帰性: 位置 t の出力は、t より後の入力をどう変えても変わらない
dec_x2 = dec_x.copy()
dec_x2[-1] += 100.0                                # 最後のトークンだけ大きく変える
out_b2, _ = decoder_masked_self_attention(dec_x2)
assert np.allclose(out_b[:-1], out_b2[:-1])        # 過去の位置の出力は不変
assert not np.allclose(out_b[-1], out_b2[-1])      # 当の位置だけ変わる

# --- (c) cross-attention: Q は decoder 側、K/V は encoder 出力 ---
out_c, w_c = cross_attention(dec_x, enc_out)
assert out_c.shape == (n_dec, d_model)             # 行数は decoder 側の長さ
assert w_c.shape == (n_dec, n_enc)                 # (decoder長, encoder長)の長方形
assert np.allclose(w_c.sum(axis=-1), 1.0)
assert np.allclose(out_c, w_c @ enc_out)           # 出力は encoder 出力(V)の重み付き和
# decoder 側を変えると「どこを見るか」(Q 経由で重み)が変わり、
_, w_c2 = cross_attention(dec_x2, enc_out)
assert not np.allclose(w_c2[-1], w_c[-1])
# encoder 出力を変えると「見られる中身」(K, V)が変わる
out_c3, _ = cross_attention(dec_x, enc_out + 1.0)
assert not np.allclose(out_c3, out_c)

# --- multi-head でも組合せは同じ(第4章の部品で呼び分けても通ることの確認)---
h = 2
W_Q, W_K, W_V, W_O = (rng.normal(size=(d_model, d_model)) * 0.5 for _ in range(4))
# (b') decoder masked self: X_q = X_kv = dec_x、causal mask
_, w_mh_self = multi_head_attention(dec_x, dec_x, W_Q, W_K, W_V, W_O, h,
                                    mask=causal_mask(n_dec))
assert w_mh_self.shape == (h, n_dec, n_dec)
for i in range(h):                                 # 全 head で未来の重みが0
    assert np.all(w_mh_self[i][np.triu_indices(n_dec, k=1)] == 0.0)
# (c') cross: X_q = dec_x、X_kv = enc_out、mask なし
out_mh, w_mh_cross = multi_head_attention(dec_x, enc_out, W_Q, W_K, W_V, W_O, h)
assert out_mh.shape == (n_dec, d_model)
assert w_mh_cross.shape == (h, n_dec, n_enc)
assert np.allclose(w_mh_cross.sum(axis=-1), 1.0)

print("OK: 3つの使い方(self / masked self / cross)の入力と mask の組合せ、すべての assert を通過")
