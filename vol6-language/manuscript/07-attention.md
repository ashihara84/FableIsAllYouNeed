
# 第7章 attention — 「全部見ればいい」

前の章で、私たちは encoder-decoder という分業を手に入れ、文字列反転という小さな「翻訳」タスクを解きました。そして3つめの痛みに出会いました。**固定長ボトルネック**です。encoder は入力文全体を読み、その内容を1本の隠れ状態ベクトルに詰め込む。decoder はその1本だけを頼りに出力文を書く。入力が長くなるほど、1本のベクトルに収まりきらない情報が増え、精度は目に見えて崩れていきました。

第5章の痛みと合わせると、手元の不満は3つ揃っています。RNN は並列化できない(痛み1)。遠くの情報が薄まる(痛み2)。そして文全体を1本に圧縮する無理(痛み3)。この章では、このうち痛み3を——そして実は痛み2も——一気に解消するアイデアを実装します。アイデアの名前は **attention(注意機構)**。論文のタイトルに掲げられた、あの attention です。

この章は本巻のクライマックスです。章の終わりに、あなたは "Attention Is All You Need" というタイトルが何を主張しているのか、論文を1ページも読まずに、自分の実験結果から言い当てることになります。

## 7.1 発想の転換: 1本に圧縮せず、decoder の各ステップで入力の全位置を見直す

ボトルネックの正体を、もう一度正確に言語化しておきましょう。

第6章の encoder は、入力の各位置で隠れ状態 $\mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_L$ を順に計算していました。つまり、**位置ごとの読み取り結果は、すでに全部手元にある**のです。それなのに私たちは、最後の1本 $\mathbf{h}_L$ だけを decoder に渡し、残りの $L-1$ 本を捨てていました。ボトルネックは、モデルの能力不足というより、**設計による自業自得**です。

人間の翻訳者にたとえてみます。第6章のモデルは、原文を一度だけ通読し、原文を伏せて、記憶だけで訳文を書く翻訳者です。短い文ならそれでも書けますが、長い文では細部から忘れていきます。普通の翻訳者はそうしません。訳文を1語書くたびに、**原文の該当箇所に目を戻します**。原文は机の上にあるのですから、見ればいいのです。

これをモデルの言葉に直すと、こうなります。

- encoder の隠れ状態 $\mathbf{h}_1, \ldots, \mathbf{h}_L$ を捨てずに全部取っておく
- decoder は1文字書くたびに、この $L$ 本を**全部見直し**、いま必要な位置の情報を取り出して使う

「全部見ればいい」。これが attention の発想のすべてです。圧縮という難題を解いたのではなく、圧縮そのものをやめたのです。

ただし、すぐに次の問題が立ちます。decoder のあるステップで「いま必要な位置」とはどこでしょうか。文字列反転なら、3文字目を書くときに見るべきは入力の後ろから3番目です。しかしそれはタスクを知っている私たちの答えであって、モデルは自分で見つけなければなりません。しかも「3文字目のときは位置 $L-3$」のような if 文をハードコードするわけにはいきません。**どこを見るかも、学習で獲得できる形**——つまりパラメータ付きの微分可能な計算——にする必要があります。

ここで、シリーズの最初から温めてきた道具が出番を迎えます。

## 7.2 どこを見るかは内積で決める: query と key の類似度 → softmax で重み → 重み付き和

第1巻2章を思い出してください。あの章の冒頭で、私たちはこう書きました——文章の中のある単語が、他のどの単語に注意を向けるべきか。それを決めるには「関連の強さ」を測る必要があり、**2つのベクトルから1つの数を作る**道具として内積(dot product)を導入したのでした。あのときの予告を、いまから回収します。

### 直観: 「探している側」と「探される側」の相性を測る

decoder のいまの隠れ状態を $\mathbf{s}$ とします。$\mathbf{s}$ は「いま何文字目を、どんな文脈で書こうとしているか」の要約ですから、いわば**探し物の内容を表すベクトル**です。一方、encoder の各隠れ状態 $\mathbf{h}_i$ は「入力の位置 $i$ に何があったか」の要約、つまり**探される側のベクトル**です。

「いまの探し物 $\mathbf{s}$ に、位置 $i$ はどれくらい関係があるか」——これは2つのベクトルの類似度の問題です。類似度なら内積で測れます。

$$e_i = \mathbf{s} \cdot \mathbf{h}_i \qquad (i = 1, \ldots, L)$$

これで各位置に「関連の強さ」の点数 $e_1, \ldots, e_L$ が付きました。次に、この点数をどう使うか。「一番点数の高い位置だけを見る」(argmax)としたいところですが、argmax は微分できず、勾配が流れません。学習で獲得させるには、**全位置を少しずつ、点数に応じた配合で見る**方が都合がよいのです。点数の列を「合計1の配合比」に変える関数——第4巻6章で作った **softmax** が、ここで合流します。

$$a_i = \frac{e^{e_i}}{\sum_{j=1}^{L} e^{e_j}}$$

小さな数値例で確かめましょう。位置が3つあり、点数が $e = (2.0,\ 0.5,\ -1.0)$ だったとします。softmax を通すと

$$a = (0.79,\ 0.18,\ 0.04)$$

となります(合計は1)。点数1位の位置を約8割見つつ、2位にも2割弱の目配りを残す。argmax の「1位だけ」を、なめらかにした形です。しかも第4巻6.5節で見たとおり、点数の差が開けば softmax はいくらでも argmax に近づけます。**どこをどれだけ見るかを、モデルは点数の付け方(=パラメータ)ごと学習できる**ことになります。

最後に、この配合比で encoder の隠れ状態を混ぜ合わせます。

$$\mathbf{c} = \sum_{i=1}^{L} a_i \mathbf{h}_i$$

$\mathbf{c}$ は**文脈ベクトル(context vector)**と呼ばれます。decoder はこの $\mathbf{c}$ を(自分の状態 $\mathbf{s}$ と並べて)使い、次の1文字を予測します。1本に固定圧縮されたベクトルの代わりに、**ステップごとに配合を変えて作り直される**1本。これが attention です。

### 形式化: query・key・value

いま登場した3つの役割に、論文と同じ名前を付けます。ここがシリーズ全体の急所なので、ゆっくり進みます。

- **query(クエリ)**: 探し物の内容を表すベクトル。上の式の $\mathbf{s}$。「いま何が欲しいか」
- **key(キー)**: 探される側が掲げる「見出し」のベクトル。上の式の $\mathbf{h}_i$。query との内積で照合される
- **value(バリュー)**: 照合に通ったときに実際に取り出される「中身」のベクトル。上の式ではこれも $\mathbf{h}_i$

手順は3拍子です。shape も添えて1枚にまとめます(バッチサイズ $B$、隠れ次元 $H$、入力長 $L$)。

| 手順 | 式 | 使う道具 | shape |
|---|---|---|---|
| (1) 照合 | $e_i = \mathbf{s} \cdot \mathbf{h}_i$ | 内積 = 類似度(第1巻2章) | $(B, H) \times (B, H) \to (B, 1)$ |
| (2) 配合比 | $a = \mathrm{softmax}(e_1, \ldots, e_L)$ | softmax(第4巻6章) | $(B, L) \to (B, L)$ |
| (3) 取り出し | $\mathbf{c} = \sum_i a_i \mathbf{h}_i$ | 重み付き和 | $(B, L), (B, H) \to (B, H)$ |

query で問い、key と照合し、value を取り出す。辞書の引き方そのものです。ただし普通の辞書と違って、完全一致した1項目だけを返すのではなく、**全項目を「一致度」の重みで混ぜて返す**。微分可能にするための、たったそれだけの変更です。

頭文字を取って **Q・K・V**。論文の式(1)に並んでいた $Q, K, V$ という3つの大文字は、この query・key・value のことです。第1巻の序章で「何ひとつ読めない」と確認したあの式の、最後まで正体不明だった3文字に、いま初めて意味が入りました。

2つ、正直に断っておきます。第一に、今回の実装では key と value が同じベクトル $\mathbf{h}_i$ を兼ねています。論文では $W^Q, W^K, W^V$ という行列を掛けて、1つのベクトルから query 用・key 用・value 用の3つの顔を**作り分け**ます。その作り分けがなぜ要るのかは、第7巻で self-attention を組むときに必要に迫られて理解します。第二に、論文の式(1)には $\sqrt{d_k}$ で割る操作が入っていますが、今回は割りません。なぜ割るのか——その答えは第4巻7章で仕込んだ「内積の分散」の議論が第7巻で回収します。今日のところは、**割らなくても動く規模**で動かします。

## 7.3 [コード] attention 付き seq2seq: 長い入力に強くなることの確認と、attention 重みの可視化

実装に移ります。タスク・データ・訓練条件は第6章と完全に揃えます。揃えなければ「attention のおかげで良くなった」と言えないからです。比較のため、第6章型(ボトルネックのみ)と本章型(attention 付き)の**両方をこのファイルの中で同じ乱数・同じバッチで訓練**します。

コードは `code/ch07/attention_seq2seq.py` にまとめてあり、本文のコードを順につなげたものがそのままこのファイルです。4つの部分に分けて読んでいきます。

### (1) 準備: タスクとデータ(第6章と共通)

```python
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
```

第6章の `make_data`(文字列反転、長さ可変)をそのまま使います。`_find_ch06_make_data` は第6章のコードが手元にあればそれを import し、なければ同一仕様の `_make_data_local` に切り替える保険です。どちらが使われたかは実行時に表示されます。

### (2) Tensor に足りない演算を補う

本章のモデルは第5巻5章の `Tensor`(自作 autograd)で組みます。ただし `Tensor` には、attention に必要な演算がいくつか足りません。**第5巻のファイルには手を入れず**、足りない分をこの章のファイルで補います。第5巻4章でやったのと同じ作業——forward を書き、局所勾配を `_backward` に書く——の繰り返しです。

```python
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
```

5つとも、構造は第5巻4章の雛形どおりです。新顔は `softmax_rows` だけに見えますが、これも forward は第4巻6.2節の最大値シフト付き softmax そのまま、backward は第4巻6章の勾配導出をそのまま式にしたものです。**この章で新しい数学は1つも増えていない**——道具はすべて、過去の巻からの持ち込みです。

### (3) モデル: 第6章型と第7章型をフラグ1つで切り替える

```python
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
```

読みどころは `decode_step` の `if p["_attn"]:` ブロック、たった7行です。7.2節の表の3拍子——内積・softmax・重み付き和——が、そのまま3つの段落になっています。`row_sum(s * h_i)` が内積(要素ごとの積を行方向に合計したもの。第1巻2章の定義そのまま)、`softmax_rows` が配合比、`col(A, i) * hs[i]` の合計が文脈ベクトル $\mathbf{c}$ です。第6章型との違いは、この7行と、出力層が $(B, H)$ の代わりに $[\mathbf{s}, \mathbf{c}]$ を連結した $(B, 2H)$ を読むことだけ。**encoder も decoder の再帰も損失も、第6章と1行も変わっていません。**

もう1つ、見逃せないことがあります。attention の中には、学習で初めて値が決まる**専用のパラメータが1つもない**のです(`Wout` の入力幅が広がっただけ)。内積も softmax も重み付き和も、固定の演算です。それでも「どこを見るか」が学習できるのは、query($\mathbf{s}$)と key($\mathbf{h}_i$)を作る RNN のパラメータに、attention 経由で勾配が流れ込むからです。「位置 $t$ を書くときに位置 $L-t+1$ と内積が大きくなるような表現を作れ」という圧力が、loss から逆向きに encoder まで届く。autograd(第5巻)を作っておいたおかげで、私たちはこの複雑な経路の backward を1行も書かずに済んでいます。

### (4) 実験: 同一条件で訓練し、長い入力で比較する

```python
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
```

実行します(手元のマシンでは訓練に十数秒。長くても90秒以内に収まる規模です)。

```
ch06 の make_data を import: 見つからず(自前定義を使用)
step  500  loss(ボトルネック)=0.875  loss(attention)=0.867
step 1000  loss(ボトルネック)=0.225  loss(attention)=0.087
step 1500  loss(ボトルネック)=0.725  loss(attention)=0.430
step 2000  loss(ボトルネック)=0.642  loss(attention)=0.160
step 2500  loss(ボトルネック)=0.081  loss(attention)=0.013
step 3000  loss(ボトルネック)=0.294  loss(attention)=0.030
step 3500  loss(ボトルネック)=0.813  loss(attention)=0.014
訓練時間: 11.5 秒
ボトルネック    短い(2-5)     正解率 0.989
ボトルネック    長い(10-12)   正解率 0.508
attention  短い(2-5)     正解率 1.000
attention  長い(10-12)   正解率 0.992
```

数字を表に整理します。これが本章の主結果です。

| モデル | 短い入力(2〜5文字) | 長い入力(10〜12文字) |
|---|---|---|
| ボトルネック型(第6章) | 0.989 | **0.508** |
| attention 付き(本章) | 1.000 | **0.992** |

短い入力では両者とも、ほぼ満点です。差が出るのは長い入力で、ボトルネック型は正解率 0.508——12文字の半分を間違える水準まで崩れます(第6章で観察した痛み3の再現です)。attention 付きは 0.992。**崩れません。** 入力が長くなっても、decoder は毎ステップ原文の必要な位置を見に行けるのですから、「覚えきれない」という failure mode 自体が消えたのです。コードの assert は、この差(0.15以上)と attention 側の水準(0.90以上)を固定しています。

### どこを見て変換したか: attention 重みの行列

そして、本巻のハイライトです。実行結果の続きには、入力 `deadbeefcafe`(12文字)を反転させたときの attention 重み $a$ が、**出力ステップ × 入力位置**の行列として表示されます。各行は softmax の出力なので合計1。値が大きいほど「その文字を書くとき、その入力位置を強く見ていた」ことを意味します。

```
入力: deadbeefcafe  → 出力: efacfeebdaed
attention 重み(行=出力ステップ, 列=入力位置):
         d     e     a     d     b     e     e     f     c     a     f     e
   e  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.01  0.98
   f  0.00  0.00  0.00  0.00  0.00  0.00  0.02  0.00  0.00  0.01  0.94  0.01
   a  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.01  0.98  0.00  0.00
   c  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.01  0.99  0.00  0.00  0.00
   f  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.99  0.00  0.00  0.00  0.00
   e  0.00  0.00  0.03  0.00  0.01  0.01  0.92  0.01  0.00  0.01  0.00  0.00
   e  0.01  0.01  0.00  0.01  0.04  0.87  0.04  0.00  0.01  0.00  0.00  0.00
   b  0.01  0.00  0.00  0.02  0.96  0.01  0.00  0.00  0.00  0.00  0.00  0.00
   d  0.00  0.00  0.02  0.90  0.06  0.00  0.00  0.01  0.00  0.00  0.00  0.00
   a  0.00  0.02  0.93  0.02  0.00  0.01  0.02  0.00  0.00  0.00  0.00  0.00
   e  0.01  0.96  0.01  0.00  0.00  0.01  0.00  0.00  0.00  0.00  0.00  0.00
   d  0.58  0.08  0.00  0.05  0.05  0.01  0.00  0.00  0.09  0.00  0.03  0.10
```

この表を、しばらく眺めてください。1行目: 出力の1文字目 `e` を書くとき、モデルは入力の**最後**の位置(`e`)を重み 0.98 で見ています。2行目: 2文字目を書くときは後ろから2番目を 0.94 で。以下、注目の山が1行ごとに1つずつ左へずれていき、右上から左下への**逆対角線**を描きます。これは文字列反転の正解の手順——$t$ 文字目を書くときは入力の後ろから $t$ 番目を写す——そのものです。

誰もこの手順を教えていません。与えたのは入出力のペアと cross-entropy だけです。それでも勾配降下は、「正しい位置と内積が大きくなる表現」を encoder と decoder の双方に作り上げ、結果としてアルゴリズムが**重み行列の模様として浮かび上がった**のです。コードの最後の assert は、この逆対角線上の重みの平均が 0.5 以上であることまで検証しています。モデルが何を考えているかは普通ブラックボックスですが、attention の重みは例外的に「どこを見たか」をそのまま見せてくれます。これが図にしたときのいわゆる attention map です。

```python
# 図7.1 の描画コード(掲載のみ): A は上の (L, L) 行列
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(A, cmap="Greys", vmin=0.0, vmax=1.0)
ax.set_xticks(range(len(src)), list(src))
ax.set_yticks(range(len(src)), list(pred_str))
ax.set_xlabel("input position (key)")
ax.set_ylabel("output step (query)")
plt.show()
```

図7.1: 入力 `deadbeefcafe` を反転するときの attention 重み。横軸が入力位置(key)、縦軸が出力ステップ(query)、マスの濃さが重み $a_{t,i}$。右上から左下へ濃いマスが一直線に並ぶ——「後ろから順に1文字ずつ写す」という反転の手順が、誰に教わるでもなく模様として現れる。

## 7.4 振り返ると: attention が本体で、RNN は足枷では?

実験は成功しました。立ち止まって、いま動いているモデルの全体を眺め直してみましょう。部品は3つあります。

1. **encoder の RNN**: 入力を1文字ずつ読んで $\mathbf{h}_1, \ldots, \mathbf{h}_L$ を作る
2. **attention**: decoder の各ステップで、$\mathbf{h}_1, \ldots, \mathbf{h}_L$ から必要な情報を取り出す
3. **decoder の RNN**: 出力を1文字ずつ書き、状態 $\mathbf{s}$(= query)を更新する

この章の性能向上は、どの部品の手柄だったでしょうか。明らかに attention です。RNN は第6章から1行も変えていないのですから。長い入力で 0.508 → 0.992 という改善は、まるごと attention が持ち込んだものです。

では、この3部品を「第5章で測った痛み」の観点で採点し直してみます。

**痛み2(長距離依存)を解いたのは誰か。** attention です。RNN では、位置 $i$ の情報が出力ステップ $t$ に届くまでに、隠れ状態の更新を距離に比例した回数くぐり抜ける必要がありました。遠いほど薄まり、勾配は消えていく(第5章5.4)。attention では、どんなに離れた位置でも $\mathbf{s} \cdot \mathbf{h}_i$ の**内積1発**で直結します。距離という概念そのものが、計算経路から消えているのです。

**痛み1(並列化できない)を抱えたままなのは誰か。** RNN です。encoder は相変わらず $\mathbf{h}_t$ を計算するのに $\mathbf{h}_{t-1}$ を待ちます。入力が2倍になれば、待ち時間も2倍。第1巻4章のベンチマークで見たとおり、現代のハードウェアは「大きな行列積を一度に」が得意で、「小さな計算を順番に」が苦手です。一方 attention を見てください。内積・softmax・重み付き和——すべて行列演算で、**全位置をまとめて一度に**計算できる形をしています(私たちの実装は教育用に for ループで書きましたが、計算の依存関係としては待ちがありません)。

採点表にすると、こうなります。

| 部品 | 長距離依存 | 並列化 | この章での貢献 |
|---|---|---|---|
| RNN(encoder / decoder) | 弱い(薄まる) | 不可(逐次) | 第6章から変更なし |
| attention | 強い(内積で直結) | 可能(行列演算) | 0.508 → 0.992 の改善 |

並べてみて、奇妙なことに気づかないでしょうか。**性能を支えているのは attention で、速度と性能の足を引っ張っているのは RNN です。** 私たちは「RNN を attention で補強する」つもりでこの章を書き始めました。しかし出来上がったものを正直に評価すると、補強材のほうが本体より強い。むしろこう問うべきに見えてきます——

**attention が本体で、RNN は足枷なのではないか。RNN を、取り除けないか。**

2017年、Google の研究者たちがまさにこの問いに踏み切りました。RNN を完全に取り除き、attention だけで系列を処理するアーキテクチャを提案したのです。その論文のタイトルが、

> *"Attention Is All You Need"*
> — Vaswani et al., 2017
>
> 訳: 注意こそが、あなたに必要なすべて

——「attention だけでいい」。

第1巻の序章でこのタイトルを初めて見たとき、それは外国語の呪文でした。いまのあなたには、これが**挑発**として読めるはずです。当時の常識では、系列を扱うモデルの本体は RNN(あるいはその改良版)であり、attention は便利な付属品でした。タイトルはその常識への正面からの反語です——付属品だと思っていたそれが、すべてだ。本体だと思っていた RNN は、いらない。

ただし、勢いで RNN を捨てる前に、RNN が黙って担っていた仕事を見落とすわけにはいきません。少なくとも2つあります。第一に、**語順**。attention の3拍子(内積→softmax→重み付き和)をよく見ると、$\mathbf{h}_1, \ldots, \mathbf{h}_L$ を並べ替えても結果が変わりません。重み付きの「和」は順序を忘れるのです。いまのモデルで $\mathbf{h}_i$ が位置の情報を含んでいられるのは、RNN が**順番に読んでいる**からでした。RNN を消したら、「dead」と「daed」の区別はどこへ行くのでしょうか。第二に、query を作っていたのは decoder の RNN でした。RNN なしで、query はどこから来るのでしょうか。入力の各位置が、自分以外の位置に attention するとしたら——自分自身の文への attention とは、何を意味するのでしょうか。

この2つの問いに答えるのが、論文の Positional Encoding と Self-Attention です。つまり第7巻の仕事です。私たちはもう、第7巻を「論文に書いてあるから」読むのではありません。**自分の手で RNN を追い詰めた結果、その2つが必要だと自分で言えてしまった**から読むのです。需要は、揃いました。

## まとめ

- ボトルネックの解消法は圧縮の改良ではなく圧縮の放棄: encoder の全隠れ状態を取っておき、decoder の各ステップで**全部見直す**。これが attention
- どこを見るかは3拍子で決まる: **query と key の内積**で類似度を測り(第1巻2章の回収)、**softmax** で合計1の配合比にし(第4巻6章の回収)、その重みで **value の重み付き和**を取る
- 論文の $Q$・$K$・$V$ は query・key・value。今回は key と value を encoder の隠れ状態が兼ねた。$W^Q, W^K, W^V$ による作り分けと $\sqrt{d_k}$ は第7巻で
- 第6章と同一条件の比較で、長い入力の正解率が 0.508 → 0.992。attention 重みの行列には「どこを見て変換したか」が模様(反転タスクなら逆対角線)として現れる
- 採点し直すと、長距離依存に強く並列化可能なのは attention、その両方に弱いのは RNN。「attention が本体で RNN は足枷では?」——この問いがタイトル "Attention Is All You Need" の主張そのもの。ただし RNN を消すと語順と query の出どころが宙に浮く(第7巻への需要)

**ラスボスとの距離**: 論文の式(1)の $Q, K, V$ という3文字に、今日初めて意味が入りました。残るは $\sqrt{d_k}$ と、$Q, K, V$ を作り分ける $W^Q, W^K, W^V$——どちらも第7巻で討ち取ります。

## 演習

**問1**(attention マップを読んで誤変換の原因を探す — 本章のメイン演習)`attention_seq2seq.py` の `n_steps` を 3500 から 600 に減らして訓練し、わざと未熟なモデルを作ってください。`test_long` の中から反転に失敗した例を1つ見つけ、その attention 重みの行列を本文と同じ形式で表示して、**間違えた文字の行**を観察してください。正解した行と比べて、重みの分布にどんな違いがありますか。「モデルがどこを見て間違えたか」を、行列を根拠に1〜2文で説明してください。

**問2**(注目先のずれの定量化)正しく訓練したモデル(`n_steps=3500`)について、`test_long` の全例で「各出力ステップ $t$ の重みの argmax が、正解の注目先 $L-1-t$ と一致した割合」を計算してください。本文の正解率 0.992 と、この「注目先の正解率」は近い値になるでしょうか。

**問3**(配合比の尖り方 — 第4巻6.5節の温度の再演)`decode_step` の中で、softmax に入れる直前のスコアを一律に $1/4$ 倍(あるいは $4$ 倍)してから訓練し、attention 重みの行列がどう変わるかを観察してください。第4巻6.5節の温度の言葉で言うと、それぞれ温度を上げた・下げたことに相当します。重みの「尖り方」と正解率にどんな影響がありますか。

<details>
<summary>略解</summary>

**問1** 失敗例の検索は、本文の可視化コードを1例ずつのループに包むだけです。

```python
for src, tgt in test_long:
    X, Y = encode_batch([(src, tgt)])
    preds, maps = greedy_decode(attn, X)
    pred_str = "".join(ALPHABET[i] for i in preds[0])
    if pred_str != tgt:
        break   # 最初の失敗例で止める(表示は本文の可視化コードを流用)
```

未熟なモデルの失敗例では、典型的には次のどちらかが見えます。(a) 間違えた文字の行だけ重みがぼやけて、複数の位置に 0.2〜0.4 程度ずつ散っている(どこを見るか決めきれず、混ざった文脈ベクトルから多数派の文字を出した)。(b) 重みの山は1つだが、**正解の隣の列**に立っている(見る場所を1つ取り違え、その位置の文字をそのまま写した)。いずれも「出力の誤り」が「注目先の誤り」として行列に直接写っているのが観察のポイントです。誤りの原因をここまで具体的に指差せるのは、attention が「どこを見たか」を重みとして公開しているからで、第8巻で実際の翻訳モデルを debug するときにも同じ読み方をします。

**問2** 一致割合は次で計算できます。

```python
hits = total = 0
for src, tgt in test_long:
    X, Y = encode_batch([(src, tgt)])
    preds, maps = greedy_decode(attn, X)
    A = maps[:, 0, :]
    L = len(src)
    hits += (A.argmax(axis=1) == (L - 1 - np.arange(L))).sum()
    total += L
print(hits / total)
```

手元の実行では 0.98 前後で、文字の正解率 0.992 とほぼ一致します。このタスクでは「正しい場所を見ること」と「正しい文字を書くこと」がほとんど同義だからです(正しい位置さえ見れば、value にはその位置の文字情報が入っている)。

**問3** スコアを $1/4$ 倍するのは温度を上げる操作で、softmax の出力はなだらかになります(第4巻6.5節)。重みの行列は逆対角線がにじんで太くなり、文脈ベクトルに隣の位置の情報が混ざるため、訓練を同じステップ数で打ち切ると正解率は下がりやすくなります(訓練を続ければ、モデルがスコアの差を約4倍に広げるよう適応して、ある程度回復します)。$4$ 倍は温度を下げる操作で、重みは argmax に近い一点集中になります。一見良さそうですが、softmax が飽和する(出力がほぼ 0/1 に張り付く)と勾配が小さくなり、訓練初期に「どこを見るべきか」の試行錯誤がしにくくなる——という形で学習の進みに影響が出ます。スコアの**スケール**が attention の学習のしやすさを左右する、というこの観察は覚えておいてください。論文の式(1)が $\sqrt{d_k}$ で割っている理由(第7巻)に、まっすぐつながります。

</details>

---

本章のコードは `code/ch07/attention_seq2seq.py`(本文の(1)〜(4)を順につなげたもの)にまとめてあり、`python3 attention_seq2seq.py` で「長い入力で attention が勝つこと」「重みが逆対角線を向くこと」が assert により検算できます。第6章の `make_data` が `code/ch06/` にあればそれを import して使います(なければ同一仕様の自前定義に切り替わります)。
