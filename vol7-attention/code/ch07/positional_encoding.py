# 第7巻 第7章 7.4: Positional Encoding(論文 3.5、式(3))
# 第8巻がこのファイルを import して組み立てに使う。
# 変更したら必ず test_positional_encoding.py を実行すること。
import numpy as np


def positional_encoding(max_len, d_model):
    """論文 3.5 式(3) の positional encoding 行列を返す。

    PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
    PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))

    返り値: (max_len, d_model)。学習パラメータを持たない決め打ちの定数行列。
    """
    if d_model % 2 != 0:
        raise ValueError("d_model は偶数を仮定する(sin/cos を列のペアで使うため)")
    pos = np.arange(max_len, dtype=np.float64)[:, np.newaxis]      # (max_len, 1)
    i = np.arange(d_model // 2, dtype=np.float64)[np.newaxis, :]   # (1, d_model/2)
    angle = pos / 10000.0 ** (2.0 * i / d_model)                   # (max_len, d_model/2)
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(angle)   # 偶数列 2i
    pe[:, 1::2] = np.cos(angle)   # 奇数列 2i+1
    return pe


if __name__ == "__main__":
    pe = positional_encoding(50, 512)
    assert pe.shape == (50, 512)

    # 位置 0 では角度がすべて 0: sin の列は 0、cos の列は 1
    assert np.allclose(pe[0, 0::2], 0.0)
    assert np.allclose(pe[0, 1::2], 1.0)

    # 定義式そのままのスポットチェック(全数検査は test_positional_encoding.py)
    assert np.isclose(pe[3, 0], np.sin(3.0))
    assert np.isclose(pe[3, 1], np.cos(3.0))
    assert np.isclose(pe[7, 10], np.sin(7.0 / 10000.0 ** (10.0 / 512.0)))
    assert np.isclose(pe[7, 11], np.cos(7.0 / 10000.0 ** (10.0 / 512.0)))

    # 全成分が [-1, 1]: 位置がどれだけ先でも値が暴れない
    assert np.all(np.abs(pe) <= 1.0)

    print("positional_encoding: すべての assert を通過しました")
