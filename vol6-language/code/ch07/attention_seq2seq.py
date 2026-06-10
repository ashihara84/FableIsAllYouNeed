# 第6巻 第7章: attention 付き seq2seq — 「全部見ればいい」
# 第6章と同じタスク(文字列反転、長さ可変、seed 42)・同じ条件で、
# ボトルネック型 seq2seq(第6章)と attention 付き seq2seq を並べて訓練し、
# 「長い入力に強くなる」ことを assert で固定する。
# Tensor は第5巻5章の自作 autograd をそのまま import する(vol5 側は変更しない)。
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..",
                                "vol5-backprop", "code", "ch05"))
from tensor_autograd import Tensor, softmax_cross_entropy  # noqa: E402

# ---------------------------------------------------------------
# タスクとデータ(第6章と共通)
# ---------------------------------------------------------------
ALPHABET = "abcdefgh"          # 8 文字の小さな語彙
V = len(ALPHABET)              # 出力語彙サイズ
BOS = V                        # decoder の開始記号 <bos>(decoder 側だけが使う)
stoi = {ch: i for i, ch in enumerate(ALPHABET)}


def _make_data_local(n_pairs, min_len, max_len, rng):
    """文字列反転タスクのデータ生成(第6章と同一仕様: 長さ可変、文字は一様ランダム)。
    (入力文字列, 反転した文字列) のペアを n_pairs 個返す。"""
    pairs = []
    for _ in range(n_pairs):
        L = int(rng.integers(min_len, max_len + 1))
        s = "".join(rng.choice(list(ALPHABET), size=L))
        pairs.append((s, s[::-1]))
    return pairs


def _find_ch06_make_data():
    """第6章の code/ch06/ に make_data があればそれを使う(並列執筆のための保険)。
    見つからない・仕様が合わない場合は None を返し、上の自前定義に切り替える。"""
    ch06 = os.path.join(_HERE, "..", "ch06")
    if not os.path.isdir(ch06):
        return None
    sys.path.insert(0, ch06)
    import importlib
    for fname in sorted(os.listdir(ch06)):
        if not fname.endswith(".py"):
            continue
        try:
            mod = importlib.import_module(fname[:-3])
        except Exception:
            continue
        fn = getattr(mod, "make_data", None)
        if fn is None:
            continue
        try:  # 仕様の突き合わせ: 反転タスクで同じ引数で呼べるか
            probe = fn(4, 3, 5, np.random.default_rng(0))
            src, tgt = probe[0]
            if isinstance(src, str) and tgt == src[::-1]:
                return fn
        except Exception:
            continue
    return None


_ch06_fn = _find_ch06_make_data()
USING_CH06_DATA = _ch06_fn is not None
make_data = _ch06_fn if USING_CH06_DATA else _make_data_local


def encode_batch(pairs):
    """同じ長さのペアのリストを (B, L) の整数配列 X, Y にする。"""
    X = np.array([[stoi[c] for c in src] for src, _ in pairs])
    Y = np.array([[stoi[c] for c in tgt] for _, tgt in pairs])
    return X, Y


def one_hot(idx, n):
    m = np.zeros((len(idx), n))
    m[np.arange(len(idx)), idx] = 1.0
    return m


# ---------------------------------------------------------------
# Tensor に足りない演算を自章で補う(vol5 のファイルは変更しない)
# ---------------------------------------------------------------
def tanh_t(x):
    """tanh(第5巻4章 演習問1の行列版)。"""
    out = Tensor(np.tanh(x.data), (x,))

    def _backward():
        x.grad += (1.0 - out.data ** 2) * out.grad

    out._backward = _backward
    return out


def row_sum(x):
    """(B, H) を行ごとに合計して (B, 1) に。query と key の内積をバッチで取るために使う。"""
    out = Tensor(x.data.sum(axis=1, keepdims=True), (x,))

    def _backward():
        x.grad += out.grad  # (B,1) が (B,H) へブロードキャストで配られる

    out._backward = _backward
    return out


def concat_cols(tensors):
    """(B, w_i) たちを横に並べて (B, Σw_i) に。backward は各自の列を切り戻すだけ。"""
    out = Tensor(np.concatenate([t.data for t in tensors], axis=1), tuple(tensors))

    def _backward():
        lo = 0
        for t in tensors:
            hi = lo + t.data.shape[1]
            t.grad += out.grad[:, lo:hi]
            lo = hi

    out._backward = _backward
    return out


def col(x, i):
    """(B, T) の第 i 列を (B, 1) で取り出す。"""
    out = Tensor(x.data[:, i:i + 1], (x,))

    def _backward():
        x.grad[:, i:i + 1] += out.grad

    out._backward = _backward
    return out


def softmax_rows(x):
    """行ごとの softmax(第4巻6章)。attention の重みを作る。
    backward は第4巻6章の導出と同じ p * (δ - Σ p δ)。"""
    z = x.data - x.data.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    out = Tensor(p, (x,))

    def _backward():
        inner = (out.grad * p).sum(axis=1, keepdims=True)
        x.grad += p * (out.grad - inner)

    out._backward = _backward
    return out


# ---------------------------------------------------------------
# モデル: RNN encoder-decoder(with_attention で第6章型/第7章型を切替)
# ---------------------------------------------------------------
D = 16   # 埋め込みの次元
H = 24   # 隠れ状態の次元


def init_model(rng, with_attention):
    def mat(a, b):
        return Tensor(rng.standard_normal((a, b)) / np.sqrt(a))  # 第5巻6.2の分散維持init

    p = {
        "E_enc": mat(V, D),         # (V, D)  encoder の埋め込み
        "Wxh": mat(D, H), "Whh": mat(H, H), "bh": Tensor(np.zeros(H)),
        "E_dec": mat(V + 1, D),     # (V+1, D) decoder の埋め込み(<bos> の分 +1)
        "Wxd": mat(D, H), "Whd": mat(H, H), "bd": Tensor(np.zeros(H)),
        # attention ありは [s, c] の連結 (B, 2H) を、なしは s (B, H) を読む
        "Wout": mat(2 * H if with_attention else H, V),
        "bout": Tensor(np.zeros(V)),
    }
    p["_attn"] = with_attention
    return p


def params_of(p):
    return [v for k, v in p.items() if isinstance(v, Tensor)]


def encode(p, X):
    """X: (B, L) → encoder の隠れ状態のリスト [h_1, ..., h_L](各 (B, H))。"""
    B, L = X.shape
    h = Tensor(np.zeros((B, H)))
    hs = []
    for t in range(L):
        x = Tensor(one_hot(X[:, t], V)) @ p["E_enc"]            # (B, D)
        h = tanh_t(x @ p["Wxh"] + h @ p["Whh"] + p["bh"])       # (B, H)
        hs.append(h)
    return hs


def decode_step(p, y_prev, s, hs):
    """1ステップ分の decoder。y_prev: (B,) 直前の出力(教師 or 自分の予測)。
    戻り値: (新しい状態 s, 語彙の logits, attention 重み or None)"""
    x = Tensor(one_hot(y_prev, V + 1)) @ p["E_dec"]             # (B, D)
    s = tanh_t(x @ p["Wxd"] + s @ p["Whd"] + p["bd"])           # (B, H) ← query
    if p["_attn"]:
        # (1) 内積: 今の状態 s(query)と各位置 h_i(key)の類似度
        scores = [row_sum(s * h_i) for h_i in hs]               # 各 (B, 1)
        # (2) softmax: 類似度を「合計1の注目度」に
        A = softmax_rows(concat_cols(scores))                   # (B, T)
        # (3) 重み付き和: 注目度で各位置の中身 h_i(value)を混ぜる
        c = col(A, 0) * hs[0]                                   # (B, H)
        for i in range(1, len(hs)):
            c = c + col(A, i) * hs[i]
        feat = concat_cols([s, c])                              # (B, 2H)
    else:
        A = None
        feat = s                                                # (B, H) ボトルネックのみ
    logits = feat @ p["Wout"] + p["bout"]                       # (B, V)
    return s, logits, A


def seq_loss(p, X, Y):
    """teacher forcing(第6章6.4)での平均 cross-entropy。X, Y: (B, L)。"""
    B, L = X.shape
    hs = encode(p, X)
    s = hs[-1]                                  # 最後の隠れ状態から decoder を開始
    y_prev = np.full(B, BOS)
    total = None
    for t in range(L):
        s, logits, _ = decode_step(p, y_prev, s, hs)
        loss_t = softmax_cross_entropy(logits, Y[:, t])
        total = loss_t if total is None else total + loss_t
        y_prev = Y[:, t]                        # 教師の正解を次の入力に
    return total * (1.0 / L)


def greedy_decode(p, X):
    """自己回帰生成(第6章6.4)。自分の予測を次の入力に回す。
    戻り値: 予測 (B, L) と、attention 重みの記録 (L, B, T) or None。"""
    B, L = X.shape
    hs = encode(p, X)
    s = hs[-1]
    y_prev = np.full(B, BOS)
    preds, maps = [], []
    for t in range(L):
        s, logits, A = decode_step(p, y_prev, s, hs)
        y_prev = logits.data.argmax(axis=1)
        preds.append(y_prev)
        if A is not None:
            maps.append(A.data)
    preds = np.stack(preds, axis=1)
    return preds, (np.stack(maps) if maps else None)


def token_accuracy(p, pairs):
    """文字単位の正解率。長さごとにまとめてバッチ評価する。"""
    by_len = {}
    for pair in pairs:
        by_len.setdefault(len(pair[0]), []).append(pair)
    correct = total = 0
    for group in by_len.values():
        X, Y = encode_batch(group)
        preds, _ = greedy_decode(p, X)
        correct += (preds == Y).sum()
        total += Y.size
    return correct / total


def sgd_step(params, lr):
    for w in params:
        w.data -= lr * w.grad
        w.grad = np.zeros_like(w.data)


# ---------------------------------------------------------------
# 実験: 第6章型と第7章型を同一条件で訓練し、長い入力で比較する
# ---------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # データ: 訓練は長さ 2〜12 を混在、評価は「短い 2〜5」と「長い 10〜12」に分ける
    train_pairs = make_data(4096, 2, 12, rng)
    test_short = make_data(256, 2, 5, rng)
    test_long = make_data(256, 10, 12, rng)
    print("ch06 の make_data を import:", "成功" if USING_CH06_DATA else
          "見つからず(自前定義を使用)")

    # 長さごとのバケツ(ミニバッチは同じ長さで揃える)
    buckets = {}
    for pair in train_pairs:
        buckets.setdefault(len(pair[0]), []).append(pair)
    lengths = sorted(buckets.keys())

    baseline = init_model(rng, with_attention=False)   # 第6章のボトルネック型
    attn = init_model(rng, with_attention=True)        # 本章の attention 型

    n_steps, batch_size, lr = 3500, 32, 0.1
    t0 = time.time()
    for step in range(n_steps):
        L = lengths[rng.integers(len(lengths))]
        bucket = buckets[L]
        idx = rng.integers(len(bucket), size=batch_size)
        X, Y = encode_batch([bucket[i] for i in idx])
        for model in (baseline, attn):                 # 同じバッチで両方を1歩ずつ
            loss = seq_loss(model, X, Y)
            loss.backward()
            sgd_step(params_of(model), lr)
        if (step + 1) % 500 == 0:
            print("step %4d  loss(ボトルネック)=%.3f  loss(attention)=%.3f" %
                  (step + 1, seq_loss(baseline, X, Y).data,
                   seq_loss(attn, X, Y).data))
    print("訓練時間: %.1f 秒" % (time.time() - t0))

    # ---- 比較: 文字単位の正解率 ----
    acc = {(name, tag): token_accuracy(m, pairs)
           for name, m in [("ボトルネック", baseline), ("attention", attn)]
           for tag, pairs in [("短い(2-5)", test_short), ("長い(10-12)", test_long)]}
    for (name, tag), a in acc.items():
        print("%-8s  %-10s  正解率 %.3f" % (name, tag, a))

    # 第6章と同じ痛み: ボトルネック型は長い入力で崩れる。attention は崩れない
    assert acc[("attention", "長い(10-12)")] >= 0.90, "attention が長い入力で崩れている"
    assert acc[("attention", "長い(10-12)")] >= acc[("ボトルネック", "長い(10-12)")] + 0.15, \
        "「長い入力に強くなる」差が出ていない"

    # ---- ハイライト: attention 重みの行列(どこを見て変換したか)----
    src = "deadbeefcafe"                               # 長さ12(長い側)の例
    X, Y = encode_batch([(src, src[::-1])])
    preds, maps = greedy_decode(attn, X)
    pred_str = "".join(ALPHABET[i] for i in preds[0])
    A = maps[:, 0, :]                                  # (出力ステップ, 入力位置)
    print("\n入力:", src, " → 出力:", pred_str)
    print("attention 重み(行=出力ステップ, 列=入力位置):")
    print("      " + "  ".join("%4s" % c for c in src))
    for t in range(len(src)):
        print("%4s  " % pred_str[t] + "  ".join("%.2f" % w for w in A[t]))

    # 反転タスクの正しい注目先は「逆対角線」: t 文字目を書くとき位置 L-1-t を見る
    L = len(src)
    anti_diag = A[np.arange(L), L - 1 - np.arange(L)]
    assert pred_str == src[::-1], "例文の反転に失敗している"
    assert anti_diag.mean() >= 0.5, "attention が逆対角線を向いていない"
    print("\nok: attention は長い入力に強く、重みは逆対角線を向いています")
