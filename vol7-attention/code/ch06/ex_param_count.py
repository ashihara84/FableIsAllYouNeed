# 第7巻 第6章 演習1: Transformer base model のパラメータ数を数えて、
# 論文 Table 3 の「65 × 10^6」と突き合わせる。
# 設定は論文 3.1〜3.4 と Table 3 base 行: N=6, d_model=512, d_ff=2048, h=8,
# 語彙は 5.1 の "shared source-target vocabulary of about 37000 tokens"。
import numpy as np


def transformer_base_params(vocab=37000, d_model=512, d_ff=2048, N=6,
                            share_embeddings=True):
    """部品ごとのパラメータ数の内訳(dict)と合計を返す。"""
    # --- 埋め込みと出力 head(3.4)---
    # 共有すれば E (vocab, d_model) の1枚。共有しなければ
    # 入力埋め込み・出力埋め込み・pre-softmax linear の3枚(6.3参照)
    n_emb_matrices = 1 if share_embeddings else 3
    embedding = n_emb_matrices * vocab * d_model

    # --- attention 1個ぶん(3.2.2)---
    # W^Q, W^K, W^V: 各 (d_model, d_model)(h 個の頭に切る前の全体。第4章)
    # W^O: (d_model, d_model)。バイアスなし。h で分割しても総数は変わらない
    attn = 4 * d_model * d_model

    # --- FFN 1個ぶん(3.3 式(2))---
    ffn = d_model * d_ff + d_ff + d_ff * d_model + d_model  # W1, b1, W2, b2

    # --- layer norm 1個ぶん(γ と β。第5巻6.3)---
    ln = 2 * d_model

    # --- 層の構成(3.1)---
    enc_layer = attn + ffn + 2 * ln              # self-attn + FFN、各 sub-layer に LN
    dec_layer = 2 * attn + ffn + 3 * ln          # masked self-attn + cross-attn + FFN

    breakdown = {
        "embedding(+出力head)": embedding,
        "encoder(%d層)" % N: N * enc_layer,
        "decoder(%d層)" % N: N * dec_layer,
    }
    return breakdown, sum(breakdown.values())


if __name__ == "__main__":
    breakdown, total = transformer_base_params()
    for name, n in breakdown.items():
        print("%-22s %12s" % (name, format(n, ",")))
    print("%-22s %12s (= %.1fM)" % ("合計", format(total, ","), total / 1e6))

    # 論文 Table 3: base model は 65 × 10^6。
    # 語彙サイズが「約37000」という丸めなので、概算が同じオーダー・同じ先頭桁
    # (60M台)に収まれば「一致」とみなす
    assert 60e6 < total < 70e6
    assert round(total / 1e6) == 63  # この数え方では 63M

    # 1層あたりの感覚: attention 1個 ≈ 1.05M、FFN 1個 ≈ 2.10M
    # (FFN は attention の約2倍重い、というのは意外と知られていない)
    attn = 4 * 512 * 512
    ffn = 512 * 2048 + 2048 + 2048 * 512 + 512
    assert np.isclose(attn / 1e6, 1.05, atol=0.01)
    assert np.isclose(ffn / 1e6, 2.10, atol=0.01)
    assert ffn > 2 * attn - 5000  # FFN ≈ attention の2倍

    # weight sharing の節約額(演習2): 共有しないと3枚で +37.9M
    _, total_unshared = transformer_base_params(share_embeddings=False)
    saving = total_unshared - total
    assert saving == 2 * 37000 * 512  # E が 3枚 → 1枚 で 2枚ぶんの節約
    print("共有しない場合: %.1fM(節約額 %.1fM)" % (total_unshared / 1e6, saving / 1e6))

    print("ok: base model ≈ %.0fM — 論文 Table 3 の 65M と同じオーダーで一致" % (total / 1e6))
