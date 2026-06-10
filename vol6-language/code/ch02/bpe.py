"""BPE(byte-pair encoding)のフルスクラッチ実装(第6巻2章)。

train_bpe / encode / decode を提供する。第8巻2章がこのモジュールを
import して再利用するため、関数として独立させてある。
"""

import collections

END = "</w>"  # 単語末マーカー。これがないと decode で空白を復元できない


def get_word_counts(text):
    """テキストを空白で単語に切り、(記号タプル → 出現回数) の辞書にする。"""
    counts = collections.Counter(text.split())
    return {tuple(word) + (END,): n for word, n in counts.items()}


def count_pairs(word_counts):
    """隣り合う記号ペアの出現回数を数える。"""
    pairs = collections.Counter()
    for symbols, n in word_counts.items():
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += n
    return pairs


def merge_pair(symbols, pair):
    """記号列 symbols の中のすべての pair を連結して1記号にする。"""
    merged = pair[0] + pair[1]
    out = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def train_bpe(text, num_merges):
    """text からマージ規則を最大 num_merges 個学習する。

    返り値: merges(学習した順のペアのリスト)、vocab(語彙 = 文字 + マージ産物)
    """
    word_counts = get_word_counts(text)
    vocab = sorted({s for symbols in word_counts for s in symbols})
    merges = []
    for _ in range(num_merges):
        pairs = count_pairs(word_counts)
        if not pairs:
            break  # 全単語が1記号になり、もうマージできない
        # 最頻ペアを選ぶ。同数のときは辞書順で先のものを取り、結果を決定的にする
        best = min(pairs, key=lambda p: (-pairs[p], p))
        word_counts = {merge_pair(symbols, best): n
                       for symbols, n in word_counts.items()}
        merges.append(best)
        vocab.append(best[0] + best[1])
    return merges, vocab


def encode(text, merges):
    """text をトークン列(文字列のリスト)に変換する。"""
    tokens = []
    for word in text.split():
        symbols = tuple(word) + (END,)
        for pair in merges:  # 学習したのと同じ順でマージを適用する
            symbols = merge_pair(symbols, pair)
        tokens.extend(symbols)
    return tokens


def decode(tokens):
    """トークン列を文章に戻す。"""
    return "".join(tokens).replace(END, " ").rstrip()


if __name__ == "__main__":
    # 2.3節で手で追ったのと同じ小さなコーパス
    corpus = " ".join(["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3)

    # --- 語彙が育つ過程の観察 -------------------------------------------
    merges, vocab = train_bpe(corpus, num_merges=10)
    n_chars = len(vocab) - len(merges)
    print("基礎文字: %d 種類, マージ後の語彙: %d 種類" % (n_chars, len(vocab)))
    for k, pair in enumerate(merges, 1):
        print("マージ%2d: %r + %r -> %r" % (k, pair[0], pair[1], pair[0] + pair[1]))

    # 2.3節の手作業トレースと一致するか(最初の5マージ)
    assert merges[:5] == [("e", "s"), ("es", "t"), ("est", END),
                          ("l", "o"), ("lo", "w")]

    # --- encode → decode の往復一致 -------------------------------------
    assert decode(encode(corpus, merges)) == corpus

    # コーパスにない単語も、部品に割れるだけで未知語にはならない
    assert encode("lowest", merges) == ["low", "est</w>"]
    assert decode(encode("lowest", merges)) == "lowest"

    # マージ0個(文字単位)でも往復は成立する
    merges0, _ = train_bpe(corpus, num_merges=0)
    assert decode(encode("low lower", merges0)) == "low lower"

    # --- 語彙サイズとトークン列の長さ(演習2の計測)----------------------
    print()
    print("マージ回数  語彙サイズ  コーパスのトークン数")
    for m in [0, 2, 4, 6, 8, 10]:
        ms, vs = train_bpe(corpus, num_merges=m)
        n_tok = len(encode(corpus, ms))
        print("%8d  %8d  %8d" % (m, len(vs), n_tok))

    print()
    print("すべての assert を通過しました。")
