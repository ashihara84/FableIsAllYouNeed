# 第8巻 第2章: 共有データモジュール — コーパスからバッチへ(論文 5.1 の縮小再現)
#
# 第3章(訓練ループ)・第5章(訓練の実際)・第6章(評価)がこのファイルを
# import する。以下は他章との契約なので、変更しないこと。
#
#   PAD = 0, BOS = 1, EOS = 2                  … 特殊トークンの番号
#   make_corpus()  -> [(英文, 和文), ...]       … 決定的なトイ対訳(乱数なし)
#   encode_pair(src, tgt) -> (src_ids, tgt_ids) … src は [..., EOS]、
#                                                  tgt は [BOS, ..., EOS]
#   decode(ids)    -> 文字列                    … PAD/BOS/EOS は読み飛ばす
#   make_batches(pairs, batch_size, rng, by_length=True)
#                  -> (バッチ列, padding率)      … バッチ列の各要素は
#                     (src (B, L_s), tgt (B, L_t)) の np.int64 配列(PAD 埋め)
#   make_pad_mask(batch) -> bool 配列 (B, L)    … True = 本物のトークン
#                     (第7巻3章の mask と同じ「True = 見てよい」規約)
#
# BPE は第6巻2章 bpe.py をそのまま import する(再実装しない)。

import os
import sys

import numpy as np

# --- 第6巻2章の BPE を import する(vol6 側のファイルは変更しない)----------
_VOL6_CH02 = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "vol6-language", "code", "ch02"))
if _VOL6_CH02 not in sys.path:
    sys.path.insert(0, _VOL6_CH02)
import bpe  # noqa: E402  (sys.path を整えてから import する)

# --- 特殊トークン(契約: 第3・5・6章がこの番号を前提にする)------------------
PAD = 0   # 埋め草。バッチを長方形にするための詰め物(2.3節・2.4節)
BOS = 1   # 生成開始の合図。decoder の最初の入力(第6巻6章)
EOS = 2   # 生成終了の宣言。decoder が最後に出すべき記号(第6巻6章)
SPECIALS = ["<pad>", "<bos>", "<eos>"]

NUM_MERGES = 200  # BPE のマージ回数 = 語彙サイズのつまみ(第6巻2章)


def make_corpus():
    """英日のトイ対訳コーパスを返す。完全に決定的(乱数なし)。

    文はすべて本書のための自作で、著作権フリー。日本語は BPE(空白で単語に
    切る)に合わせて分かち書きにしてある。意味の自然さより形の規則性を
    優先した人工対訳である。
    """
    pairs = [
        # --- 挨拶・定型文 ---------------------------------------------------
        ("hello", "こんにちは"),
        ("good morning", "おはよう ございます"),
        ("good evening", "こんばんは"),
        ("good night", "おやすみ なさい"),
        ("thank you", "ありがとう ございます"),
        ("you are welcome", "どう いたしまして"),
        ("excuse me", "すみません"),
        ("i am sorry", "ごめん なさい"),
        ("goodbye", "さようなら"),
        ("see you tomorrow", "また あした"),
        ("yes", "はい"),
        ("no", "いいえ"),
        ("please", "おねがい します"),
        ("nice to meet you", "はじめまして"),
        ("how are you", "おげんき です か"),
        ("i am fine", "げんき です"),
        # --- 数 --------------------------------------------------------------
        ("one", "いち"), ("two", "に"), ("three", "さん"),
        ("four", "よん"), ("five", "ご"), ("six", "ろく"),
        ("seven", "なな"), ("eight", "はち"), ("nine", "きゅう"),
        ("ten", "じゅう"),
    ]

    # --- 日常文(主語 × 動詞 × 目的語のテンプレート)-------------------------
    subjects = [("i", "わたし"), ("you", "あなた"), ("he", "かれ"),
                ("she", "かのじょ"), ("we", "わたしたち"), ("they", "かれら")]
    # (基本形, 三単現, 日本語の述語の型)。{} に目的語が入る
    verbs = [("like", "likes", "{} が すき です"),
             ("eat",  "eats",  "{} を たべます"),
             ("see",  "sees",  "{} を みます"),
             ("want", "wants", "{} が ほしい です")]
    objects = [("apples", "りんご"), ("oranges", "みかん"),
               ("books", "ほん"), ("cats", "ねこ"),
               ("dogs", "いぬ"), ("fish", "さかな"),
               ("tea", "おちゃ"), ("eggs", "たまご")]

    for s_en, s_ja in subjects:
        for v_base, v_3rd, v_ja in verbs:
            v_en = v_3rd if s_en in ("he", "she") else v_base
            for o_en, o_ja in objects:
                pairs.append(("%s %s %s" % (s_en, v_en, o_en),
                              "%s は %s" % (s_ja, v_ja.format(o_ja))))

    # --- 長めの文(目的語2つ。長さの分布に幅を持たせる — 2.3節で効く)--------
    object_pairs = [(objects[0], objects[1]), (objects[2], objects[3]),
                    (objects[4], objects[5]), (objects[6], objects[7])]
    for s_en, s_ja in subjects[:2]:                       # i, you のみ
        for v_base, _, v_ja in verbs:
            for (o1_en, o1_ja), (o2_en, o2_ja) in object_pairs:
                pairs.append(
                    ("%s %s %s and %s" % (s_en, v_base, o1_en, o2_en),
                     "%s は %s" % (s_ja, v_ja.format(o1_ja + " と " + o2_ja))))

    return pairs


def _build_vocab():
    """コーパス全文(英 + 日を合算)で BPE を1回だけ訓練し、語彙表を作る。

    英日を混ぜて訓練するのは、論文 5.1 の "shared source-target vocabulary"
    の縮小再現(2.2節)。語彙番号 0, 1, 2 は特殊トークンに予約する。
    """
    corpus = make_corpus()
    text = " ".join(s + " " + t for s, t in corpus)
    merges, vocab = bpe.train_bpe(text, NUM_MERGES)
    # マージの産物は経路違いで重複しうるので、順序を保って一意化してから番号を振る
    seen, tokens = set(), []
    for t in vocab:
        if t not in seen:
            seen.add(t)
            tokens.append(t)
    itos = SPECIALS + tokens            # id -> トークン文字列
    stoi = {t: i for i, t in enumerate(itos)}   # トークン文字列 -> id
    return corpus, merges, stoi, itos


CORPUS, MERGES, stoi, itos = _build_vocab()   # import 時に一度だけ走る(数十ms)
vocab_size = len(itos)


def encode_pair(src, tgt):
    """対訳ペア(文字列2つ)を ID 列2つにする。

    規約(2.4節): src_ids = [本文..., EOS]、tgt_ids = [BOS, 本文..., EOS]。
    decoder 入力と正解への「1トークンずらし」は第3章が tgt_ids から作る。
    """
    src_ids = [stoi[t] for t in bpe.encode(src, MERGES)] + [EOS]
    tgt_ids = [BOS] + [stoi[t] for t in bpe.encode(tgt, MERGES)] + [EOS]
    return src_ids, tgt_ids


def decode(ids):
    """ID 列を文字列に戻す。特殊トークン(PAD/BOS/EOS)は読み飛ばす。"""
    tokens = [itos[int(i)] for i in ids if int(i) >= len(SPECIALS)]
    return bpe.decode(tokens)


def pad_block(seqs):
    """長さの違う ID 列のリストを、PAD で埋めた長方形 (B, L_max) にする。"""
    L = max(len(s) for s in seqs)
    out = np.full((len(seqs), L), PAD, dtype=np.int64)  # PAD=0 なので zeros と同じ
    for k, s in enumerate(seqs):
        out[k, :len(s)] = s
    return out


def make_batches(pairs, batch_size, rng, by_length=True):
    """encode_pair 済みのペア列をバッチに切る。返り値: (バッチ列, padding率)。

    by_length=True のとき、論文 5.1 の "batched together by approximate
    sequence length" を再現する: シャッフル → 長さで安定ソート(同じ長さの
    中の順序はエポックごとに変わる)→ batch_size ごとに切る → バッチの
    並び順をシャッフル(短い文ばかりの時間帯を作らないため)。
    padding率 = 全バッチの全セルに占める PAD の割合(src と tgt の合算)。
    """
    order = list(rng.permutation(len(pairs)))
    if by_length:
        order.sort(key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
    chunks = [order[k:k + batch_size] for k in range(0, len(order), batch_size)]
    chunks = [chunks[k] for k in rng.permutation(len(chunks))]

    batches, n_pad, n_cells = [], 0, 0
    for idx in chunks:
        src = pad_block([pairs[i][0] for i in idx])
        tgt = pad_block([pairs[i][1] for i in idx])
        batches.append((src, tgt))
        n_pad += int((src == PAD).sum() + (tgt == PAD).sum())
        n_cells += src.size + tgt.size
    return batches, n_pad / n_cells


def make_pad_mask(batch):
    """padding mask (B, L)。True = 本物のトークン、False = PAD。

    第7巻3章の「True = 見てよい」規約に合わせてある。attention の scores
    (B, n, m) に掛けるときは mask[:, None, :] と整形してブロードキャストする
    (全 query 行に同じ禁止が掛かる — 第7巻3章 padding_mask と同じ形)。
    """
    return batch != PAD


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # --- コーパスと語彙の規模 -------------------------------------------------
    print("対訳ペア数: %d" % len(CORPUS))
    print("語彙サイズ: %d(特殊トークン %d + BPE %d)"
          % (vocab_size, len(SPECIALS), vocab_size - len(SPECIALS)))
    assert 150 <= len(CORPUS) <= 300
    assert make_corpus() == make_corpus()   # 決定的(乱数なし)

    # --- encode_pair / decode の往復一致(全ペア)-----------------------------
    encoded = [encode_pair(s, t) for s, t in CORPUS]
    for (s, t), (s_ids, t_ids) in zip(CORPUS, encoded):
        assert s_ids[-1] == EOS
        assert t_ids[0] == BOS and t_ids[-1] == EOS
        assert decode(s_ids) == s and decode(t_ids) == t
    print("往復一致 OK: 全 %d ペアで decode(encode_pair(...)) == 元の文" % len(CORPUS))

    # コーパスにない文も、部品に割れるだけで未知語にはならない(第6巻2章)
    s_ids, t_ids = encode_pair("ten cats", "じゅう ねこ")
    assert decode(s_ids) == "ten cats" and decode(t_ids) == "じゅう ねこ"

    # --- 2.3節の実験: ランダムバッチ vs 長さ順バッチの padding 率 --------------
    rng = np.random.default_rng(42)
    _, rate_rand = make_batches(encoded, batch_size=32, rng=rng, by_length=False)
    rng = np.random.default_rng(42)
    batches, rate_len = make_batches(encoded, batch_size=32, rng=rng)
    print("padding率(batch_size=32): ランダム %.1f%% / 長さ順 %.1f%%"
          % (100 * rate_rand, 100 * rate_len))
    assert rate_len < rate_rand / 2       # 長さ順は無駄を半分以下にする
    assert round(rate_rand, 3) == 0.264   # 本文の数字(rng=42 で固定)
    assert round(rate_len, 3) == 0.087

    # --- バッチの中身の検査 ----------------------------------------------------
    n_total = sum(src.shape[0] for src, tgt in batches)
    assert n_total == len(CORPUS)         # 全ペアがちょうど1回ずつ入っている
    src, tgt = batches[0]
    assert src.dtype == np.int64 and tgt.dtype == np.int64
    assert src.ndim == 2 and tgt.ndim == 2 and src.shape[0] == tgt.shape[0]
    assert (tgt[:, 0] == BOS).all()       # tgt の先頭は全行 BOS
    for row_s, row_t in zip(src, tgt):    # PAD は必ず末尾に固まっている
        real = row_s[row_s != PAD]
        assert real[-1] == EOS and (row_s[len(real):] == PAD).all()
        real = row_t[row_t != PAD]
        assert real[-1] == EOS and (row_t[len(real):] == PAD).all()
    print("バッチ検査 OK: 先頭バッチ src %s, tgt %s" % (src.shape, tgt.shape))

    # --- pad mask -------------------------------------------------------------
    mask = make_pad_mask(src)
    assert mask.shape == src.shape and mask.dtype == bool
    assert mask.sum() == (src != PAD).sum()
    assert mask[:, None, :].shape == (src.shape[0], 1, src.shape[1])
    print("pad mask OK: True(見てよい)= %d / %d セル" % (mask.sum(), mask.size))

    print()
    print("すべての assert を通過しました。")
