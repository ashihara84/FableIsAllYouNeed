# 第7巻 第2章 2.4: encoder / decoder stack の骨組み
# 部分層(Sublayer)の中身はすべて恒等写像のダミー。
# residual + layer norm の配置と shape の流れだけを先に確定させる(外側から作る)。
# ダミーは第3〜6章で attention / FFN に差し替わり、第8巻が組み立てに使う。
import numpy as np

rng = np.random.default_rng(42)


def layer_norm(x, gamma, beta, eps=1e-5):
    """各行(各位置)を平均0・分散1に整え、gamma で伸縮、beta で平行移動する。
    第5巻6章と同じ定義(勾配は第8巻の領分なので forward のみ)。
    x: (seq_len, d_model), gamma: (d_model,), beta: (d_model,)"""
    mu = x.mean(axis=-1, keepdims=True)        # (seq_len, 1) 各行の平均
    var = x.var(axis=-1, keepdims=True)        # (seq_len, 1) 各行の分散
    x_hat = (x - mu) / np.sqrt(var + eps)      # 平均0・分散1に標準化
    return gamma * x_hat + beta


def dummy_sublayer(x):
    """恒等写像のダミー部分層。第3〜4章で attention に、第6章で FFN に差し替わる。
    契約: (seq_len, d_model) を受け取り (seq_len, d_model) を返す(residual のため)"""
    return x


def dummy_cross_sublayer(x, memory):
    """decoder の3つ目の部分層のダミー。第5章で cross-attention に差し替わる。
    memory(encoder stack の出力)を受け取るが、ダミーなのでまだ使わない"""
    assert memory.shape[-1] == x.shape[-1]     # 列数 d_model だけは揃っている契約
    return x


def encoder_layer(x, prm, self_attn, ffn):
    """encoder の1層 = 部分層2つ。各部分層は LayerNorm(x + Sublayer(x)) で包む。
    x: (seq_len, d_model) -> (seq_len, d_model)"""
    x = layer_norm(x + self_attn(x), prm["gamma1"], prm["beta1"])   # Add & Norm その1
    x = layer_norm(x + ffn(x), prm["gamma2"], prm["beta2"])         # Add & Norm その2
    return x


def decoder_layer(x, memory, prm, self_attn, cross_attn, ffn):
    """decoder の1層 = 部分層3つ。3つ目の入力 memory は2番目の部分層に刺さる。
    x: (tgt_len, d_model), memory: (src_len, d_model) -> (tgt_len, d_model)"""
    x = layer_norm(x + self_attn(x), prm["gamma1"], prm["beta1"])           # masked self-attention(第3章)
    x = layer_norm(x + cross_attn(x, memory), prm["gamma2"], prm["beta2"])  # cross-attention(第5章)
    x = layer_norm(x + ffn(x), prm["gamma3"], prm["beta3"])                 # FFN(第6章)
    return x


def encoder_stack(x, params, self_attn, ffn):
    """N 層の encoder。x: (src_len, d_model) -> (src_len, d_model)"""
    for prm in params:                         # "a stack of N = 6 identical layers"
        x = encoder_layer(x, prm, self_attn, ffn)
    return x


def decoder_stack(x, memory, params, self_attn, cross_attn, ffn):
    """N 層の decoder。memory は全層に同じものが配られる。
    x: (tgt_len, d_model), memory: (src_len, d_model) -> (tgt_len, d_model)"""
    for prm in params:
        x = decoder_layer(x, memory, prm, self_attn, cross_attn, ffn)
    return x


def init_addnorm_params(d_model, n_sublayers):
    """1層分の Add & Norm のパラメータ。gamma=1, beta=0 は第5巻6.3 と同じ初期値"""
    prm = {}
    for i in range(1, n_sublayers + 1):
        prm["gamma" + str(i)] = np.ones(d_model)
        prm["beta" + str(i)] = np.zeros(d_model)
    return prm


if __name__ == "__main__":
    N, d_model = 6, 512                        # 論文 3.1 の base model の数字
    src_len, tgt_len = 5, 7                    # わざと違う長さにして混線を検出する
    x_src = rng.normal(0, 1, size=(src_len, d_model))
    x_tgt = rng.normal(0, 1, size=(tgt_len, d_model))

    enc_params = [init_addnorm_params(d_model, n_sublayers=2) for _ in range(N)]
    dec_params = [init_addnorm_params(d_model, n_sublayers=3) for _ in range(N)]

    # --- 検証1: encoder は (src_len, d_model) を保ったまま N=6 層を通す ---
    memory = encoder_stack(x_src, enc_params, dummy_sublayer, dummy_sublayer)
    assert memory.shape == (src_len, d_model)

    # --- 検証2: decoder の出力 shape は tgt 側で決まり、memory の長さに依存しない ---
    out = decoder_stack(x_tgt, memory, dec_params,
                        dummy_sublayer, dummy_cross_sublayer, dummy_sublayer)
    assert out.shape == (tgt_len, d_model)

    # --- 検証3: 出力の各行は平均0・分散1(最後の部分層も Add & Norm で終わるため)---
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-12)
    assert np.allclose(out.std(axis=-1), 1.0, atol=1e-3)

    # --- 検証4: ダミーの骨組みは「正しく何もしない」。
    # Sublayer(x) = x なら LayerNorm(x + x) = LayerNorm(2x) = LayerNorm(x)
    # (layer norm は定数倍に不変)。よって6層全体が layer_norm 1回分に潰れる ---
    gamma1, beta1 = np.ones(d_model), np.zeros(d_model)
    assert np.allclose(memory, layer_norm(x_src, gamma1, beta1), atol=1e-4)

    # --- 検証5: d_model を守らない部分層は residual の足し算で即死する ---
    def bad_sublayer(x):
        return x[:, :256]                      # (seq_len, 256) — 契約違反

    try:
        encoder_layer(x_src, enc_params[0], bad_sublayer, dummy_sublayer)
        raise AssertionError("shape が合わない足し算が通ってしまった")
    except ValueError:
        pass                                   # (5,512) + (5,256) は定義できない(第1巻6.2)

    # --- 検証6: Add & Norm の箱の総数 = 2N + 3N = 30(演習3の検算)---
    n_addnorm = sum(len(p) // 2 for p in enc_params + dec_params)
    assert n_addnorm == 2 * N + 3 * N == 30

    print("すべての assert を通過しました")
