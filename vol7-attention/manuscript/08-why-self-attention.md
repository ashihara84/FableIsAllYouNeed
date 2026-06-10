# 第8章 Section 4 Why Self-Attention — 比較表を読む

前章までで、Section 3 を読み終えました。scaled dot-product attention(第3章)、multi-head(第4章)、3つの使い方(第5章)、FFN と embedding(第6章)、positional encoding(第7章)——アーキテクチャの部品は、すべて自分の手で実装し、テストが通っています。

その次に置かれた Section 4 は、少し変わったセクションです。新しい部品が、ひとつも出てきません。数式も(ほぼ)ありません。あるのは **Table 1 という1枚の表**と、その表をめぐる数段落の文章だけです。タイトルは "Why Self-Attention"——なぜ self-attention なのか。つまりここは設計の説明ではなく、**設計の弁明**です。再帰も畳み込みも捨てて attention だけにした、その判断の根拠を数字で示す場所です。

2017年にこの弁明を向けられた相手は、論文の査読者たちでした。しかし、いまこの弁明を読む資格を最も強く持っているのは、第6巻で RNN を自分で実装し、訓練し、**痛い目に遭った**あなたです。系列が長くなると訓練が遅くなる。遠くの情報が薄れて消える。あの体感を持つ読者にとって、Section 4 は弁明ではなく**答え合わせ**です。

この章で読む原文の核を掲げます。

> *"Motivating our use of self-attention we consider three desiderata. One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required. The third is the path length between long-range dependencies in the network."*
> — Vaswani et al., "Attention Is All You Need", Section 4
>
> 訳: 我々が self-attention を採用した動機として、3つの満たしたい条件(desiderata)を考える。第一は、層あたりの総計算量である。第二は、並列化できる計算の量であり、これは必要な逐次操作の最小数で測る。第三は、ネットワーク内の長距離依存の間の経路長である。

この3つの「条件」を日本語にし、Table 1 の各マスを自分で検算し、最後に自分の attention 実装で実測する——それがこの章の仕事です。

## 8.1 Table 1 を読む: 層あたり計算量・逐次操作数・最大経路長の3指標

表を見る前に、列の見出しになっている3つの指標を、それぞれ日本語にしておきます。表は、列の意味がわかって初めて読めるものです。

**指標1: 層あたり計算量(complexity per layer)。** 1つの層が、長さ $n$ の系列全体を一度処理するのに必要な演算(掛け算・足し算)の回数です。これは「電気代」の指標です。多ければ多いほど、同じ計算機での処理は遅く、高くつきます。

**指標2: 逐次操作数(sequential operations)。** 原文の定義をよく読んでください——「並列化できる計算の量を、**必要な逐次操作の最小数**で測る」。つまり、どれだけ計算機を並べても順番待ちを強いられるステップが何回あるか、という指標です。互いに独立な計算は手分けして一斉に終わらせられますが、「前の結果がないと次が始められない」計算は、計算機を1万台並べても1列に並んで待つしかありません。第6巻5.3で実測した、**痛み1(並列化できない)の物差し**がこの列です。

**指標3: 最大経路長(maximum path length)。** これは少し説明が要ります。原文はこう続けています。

> *"One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies."*
> — 同論文, Section 4
>
> 訳: そのような(長距離の)依存関係を学習できるかどうかを左右する重要な要因のひとつは、順方向・逆方向の信号がネットワーク内で通らなければならない経路の長さである。入力と出力の任意の位置の組の間で、この経路が短いほど、長距離依存の学習は容易になる。

位置 $i$ の情報が位置 $j$ の出力に影響を与えるには、ネットワークの中の計算を何ステップか経由する必要があります。その「ステップ数」が経路長です。順方向だけの話ではありません。学習時には勾配が**同じ道を逆向きに**通ります(第5巻3章)。そして道が長いほど、信号は薄まり、勾配は消えやすい——第5巻6.1で深さ方向に観測し、第6巻5.4で系列方向に再演した、あの現象です。**痛み2(長距離依存が苦手)の物差し**が、この列です。表に載るのは、最も離れた2位置の間の経路長、つまり最悪値です。

3つの指標が出揃ったところで、もうひとつだけ道具を足します。表のマスを埋めている $O(\cdot)$ という記法です。シリーズでここまで正面から使ってこなかったので、必要になったいま導入します(需要駆動)。

$O(n^2)$ と書いたら、それは「$n$ を大きくしていったとき、量が $n^2$ のペースで増える」という意味です。注目するのは**増え方だけ**で、定数倍や増え方の遅い項は捨てます。演算回数が $2n^2 d + n^2$ 回なら、係数の2も、増え方で負ける $n^2$ も捨てて $O(n^2 \cdot d)$。乱暴に見えますが、理由があります。第1巻4章のベンチマークで見たとおり、同じ行列積でも3重ループと `@` では実行時間が数百倍違いました。定数倍は実装次第でいくらでも動くので、層の**方式**を比較する土俵では、実装に左右されない「増え方」だけを見るのです。裏を返せば、$O$ が同じでも実際の速さが同じとは限らないことには注意してください。

準備ができました。Table 1 を再掲します。

> Table 1: *"Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. $n$ is the sequence length, $d$ is the representation dimension, $k$ is the kernel size of convolutions and $r$ the size of the neighborhood in restricted self-attention."*
>
> 訳: 各種の層についての、最大経路長・層あたり計算量・逐次操作の最小数。$n$ は系列長、$d$ は表現の次元、$k$ は畳み込みのカーネル幅、$r$ は制限付き self-attention の近傍サイズ。

| 層の種類 | 層あたり計算量 | 逐次操作数 | 最大経路長 |
|---|---|---|---|
| Self-Attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k(n))$ |
| Self-Attention (restricted) | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

$n$ は系列の長さ(トークン数)、$d$ は各位置のベクトルの次元($d_{model}$ に相当)です。この2つの登場人物の感覚を、論文の base model の値で持っておきましょう——$d = 512$、$n$ は翻訳タスクの文なら数十から百程度。**当時は $n < d$ が普通だった**。この事実が、あとで効いてきます。

## 8.2 各行を検算: self-attention は O(n²·d) だが逐次 O(1)・経路長 O(1)、RNN は逐次 O(n)

表は、眺めるものではなく検算するものです。私たちは self-attention を自分で実装済み(第3章)、RNN も自分で実装済み(第6巻5章)ですから、各マスを自分のコードに照らして埋め直せるはずです。1行ずつ行きます。

### Self-Attention 行: O(n²·d)・O(1)・O(1)

**計算量 $O(n^2 \cdot d)$。** 第3章で書いた `attention` の中身は、実質3行でした。$Q, K, V$ をすべて同じ系列から作る self-attention では、shape はそれぞれ `(n, d)` です。

1. `scores = Q @ K.T / √d` : `(n, d) @ (d, n)` → `(n, n)`。マスが $n^2$ 個、1マスは長さ $d$ のベクトル同士の内積なので掛け算 $d$ 回。合計 $n^2 d$ 回
2. `weights = softmax(scores)` : $n^2$ 個の成分への適用で $n^2$ 程度。$n^2 d$ に比べれば増え方が遅いので $O$ 記法では消えます
3. `weights @ V` : `(n, n) @ (n, d)` → `(n, d)`。これも $n^2 d$ 回

合わせて約 $2n^2d + n^2$、つまり $O(n^2 \cdot d)$。表のマスと一致しました。なお第4章の multi-head では1ヘッドあたりの次元は $d_k = d/h$ に減りますが、$h$ ヘッド分の合計はほぼ $d$ に戻るのでした(第4章4.5「分割しても総コストはほぼ同じ」)。表の $d$ はその合計で読みます。

**逐次操作数 $O(1)$。** 原文はこう言い切ります。

> *"As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires O(n) sequential operations."*
> — 同論文, Section 4
>
> 訳: Table 1 に示したとおり、self-attention 層は**定数回の逐次操作**で全位置どうしを結びつける。一方、再帰層は $O(n)$ 回の逐次操作を必要とする。

根拠は、上の3行が全部、行列演算だということです。$QK^T$ は内積の総当たり表で(第1巻4章)、$n^2$ 個のマスはどれも他のマスの結果を待ちません。softmax も行ごとに独立、`weights @ V` も同じ。つまり self-attention は系列を1歩ずつ歩かず、**何歩あってもひと跨ぎ**です。$n$ をいくら伸ばしても順番待ちのステップ数は増えない——これが $O(1)$ の意味です。

**最大経路長 $O(1)$。** 位置 $i$ と位置 $j$ がどれだけ離れていても、`scores` の $(i, j)$ マスで $\mathbf{q}_i \cdot \mathbf{k}_j$ という内積を**直接**取ります。隣どうしでも、文頭と文末でも、1層の中で1ステップ。間に中継ぎがいないのです。勾配も同じ1ステップの道を逆向きに通れます。

### Recurrent 行: O(n·d²)・O(n)・O(n)

**計算量 $O(n \cdot d^2)$。** 第6巻5章で書いた RNN の1ステップは $\mathbf{h}_t = f(W_h \mathbf{h}_{t-1} + W_x \mathbf{x}_t)$ でした。主役は $W_h$ `(d, d)` と $\mathbf{h}_{t-1}$ `(d,)` の行列ベクトル積で、掛け算 $d^2$ 回。これを系列に沿って $n$ ステップ繰り返すので、合計 $O(n \cdot d^2)$ です。

**逐次操作数 $O(n)$。** $\mathbf{h}_t$ の計算は $\mathbf{h}_{t-1}$ が出るまで始められません。1ステップごとの計算自体は行列演算で並列化できますが、**ステップとステップの間**は直列で、$n$ 回の順番待ちは絶対に消せません。

**最大経路長 $O(n)$。** 位置1の情報が位置 $n$ の出力に効くには、$\mathbf{h}_1 \to \mathbf{h}_2 \to \cdots \to \mathbf{h}_n$ という隠れ状態のリレーを $n-1$ 回経由するしかありません。勾配は同じリレーを逆走し、1ステップごとに薄まっていきます。

### 数字の正体: 第6巻の痛みが、表になっている

ここで立ち止まってください。いま検算した Recurrent 行の $O(n)$ という2つの数字に、見覚えがあるはずです。

- **逐次操作数 $O(n)$** —— これは第6巻5.3の**痛み1**です。系列が長いほど訓練が遅くなり、GPU 的な並列計算と相性最悪だと、あなたは自分の訓練時間で体感しました。あの「待ち時間」を記号にしたものが、このマスです
- **最大経路長 $O(n)$** —— これは第6巻5.4の**痛み2**です。遠くの情報が薄まり、勾配が系列を遡るうちに消えていくのを、あなたは勾配ノルムの実測で見ました。あの「遠さ」を記号にしたものが、このマスです

つまり Table 1 は、新しい主張をしている表ではありません。**第6巻であなたが踏んだ痛みを、3つの指標に符号化した表**です。そして Self-Attention 行は、その2マスがどちらも $O(1)$ になっています。待たない。遠くない。第6巻7.4で立てた「attention が本体で、RNN は足枷では?」という問いへの、これが定量的な回答です(なお第6巻6.3の痛み3=固定長ボトルネックがこの表にないのは、attention の導入自体がすでに解決済みだからです——第6巻7章)。

### ただし「計算量」の列には注意書きがある

良いことずくめに見えますが、計算量の列だけは様子が違います。Self-Attention の $O(n^2 \cdot d)$ と Recurrent の $O(n \cdot d^2)$ ——どちらが大きいのでしょうか。比を取ると $\frac{n^2 d}{n d^2} = \frac{n}{d}$。つまり**大小は $n$ と $d$ の関係次第**です。論文も、勝利宣言に条件を付けています。

> *"In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length n is smaller than the representation dimensionality d, which is most often the case with sentence representations used by state-of-the-art models in machine translations, such as word-piece and byte-pair representations."*
> — 同論文, Section 4
>
> 訳: 計算量の面では、**系列長 $n$ が表現の次元 $d$ より小さいとき**、self-attention 層は再帰層より速い。機械翻訳の最先端モデルが使う文表現(word-piece や byte-pair 表現)では、ほとんどの場合これが成り立つ。

8.1 の感覚値を思い出してください。BPE でトークン化した文(第6巻2章)なら $n$ は数十から百、$d = 512$。たしかに $n < d$ で、当時の翻訳タスクでは計算量でも self-attention に分がありました。しかし、これは**条件付きの勝利**です。$n$ が $d$ を超えた途端、攻守は逆転します。この注意書きの意味は、8.3 で回収します。

### 残り2行は、足早に

Convolutional 行と Self-Attention (restricted) 行は、本シリーズが通らなかった道です(畳み込みを深追いしないことは、第1章1.3で決めた最短測地線の方針どおりです)。表を読むのに必要な分だけ確認します。カーネル幅 $k$ の畳み込み層は各位置の**近傍 $k$ 個だけ**を結びつけるので、1層では遠い位置どうしが繋がらず、層を重ねる必要があって最大経路長は(dilated convolution で)$O(\log_k n)$。計算量は $O(k \cdot n \cdot d^2)$ で再帰層の $k$ 倍です。最終行の restricted self-attention は「各位置が近傍 $r$ 個の key だけを見る」変種で、計算量は $O(r \cdot n \cdot d)$ に下がる代わりに経路長が $O(n/r)$ に伸びます。**計算量を削ると、経路が伸びる**——このトレードオフの形は、8.3 への伏線です。

Section 4 の最終段落は、おまけとして解釈可能性に触れています—— *"As side benefit, self-attention could yield more interpretable models."*(副産物として、self-attention はより解釈しやすいモデルをもたらしうる)。attention の重みを見れば「どこを見て出力したか」が観察できる、という話で、あなたは第6巻7.3の翻訳の可視化と第4章の演習(head ごとの attention マップ)で、すでにこの副産物を味わっています。

### [コード] 検算の仕上げ: 実測 — n を倍にすると、時間は4倍になるか

机上の検算は済みました。第1巻からの流儀で、実測で締めましょう。検証したいのは Self-Attention 行の計算量 $O(n^2 \cdot d)$ です。$d$ を固定して $n$ を倍々に増やしたとき、$n^2$ の項が支配的なら、実行時間は**1段ごとに約4倍**になるはずです。測る対象は、第3章で自分が書いた `attention` そのものです。コードの全文は `code/ch08/attention_scaling.py` にあります。

```python
# 第7巻 第8章: self-attention の実行時間スケーリング実測
#   Table 1 の self-attention 行「層あたり計算量 O(n²·d)」を、自分の第3章の
#   実装で確かめる。n を倍々に増やし、実行時間が n に対して「線形より明確に
#   速く」伸びること(n² 傾向)を観測する。
#
#   assert は意図的に緩い(単調増加 + 超線形)。実行時間は BLAS・CPU・他の
#   プロセスの影響でぶれるため、「4.00倍ぴったり」を固定すると環境差で落ちる。
#   検証したいのは係数ではなく「自乗で効く」という傾向そのものである。
import sys
import time
from pathlib import Path

import numpy as np

# 第3章で実装した自分の attention をそのまま使う(精読の検算は自分のコードで)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch03"))
from attention import attention

rng = np.random.default_rng(42)

D_K = 64                            # 論文 base model の1ヘッドあたりの次元
NS = [256, 512, 1024, 2048, 4096]   # 系列長 n を倍々に振る
REPEATS = 5                         # 各 n で5回測り最小値を採る(ぶれ対策)


def measure(n, d_k, repeats=REPEATS):
    """系列長 n の self-attention 1回の実行時間(秒)。repeats 回の最小値。

    self-attention なので Q, K, V はすべて同じ系列由来 → shape は全部 (n, d_k)。
    配列の生成は計測の外に置く(測りたいのは attention 本体だけ)。
    """
    Q = rng.standard_normal((n, d_k))
    K = rng.standard_normal((n, d_k))
    V = rng.standard_normal((n, d_k))
    # 一部の BLAS(macOS の Accelerate 等)は、大きな行列積で浮動小数点の
    # 状態フラグを誤って立て、結果は正常なのに警告だけを出すことがある。
    # 結果の有限性は下の assert で直接確かめた上で、偽の警告は黙らせる
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        output, _ = attention(Q, K, V)
        assert np.isfinite(output).all(), "出力に inf / nan — これは本物の異常"
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            attention(Q, K, V)
            best = min(best, time.perf_counter() - t0)
    return best


if __name__ == "__main__":
    measure(NS[0], D_K, repeats=1)  # ウォームアップ(初回の準備コストを捨てる)

    print(f"d_k = {D_K} 固定、系列長 n を倍々に(各 {REPEATS} 回計測の最小値)")
    print()
    print("     n | 時間 (ms) | 直前との比 (n²なら≈4, 線形なら≈2)")
    print("-" * 55)
    times = []
    for n in NS:
        t = measure(n, D_K)
        ratio = f"{t / times[-1]:.2f}" if times else "   -"
        times.append(t)
        print(f"{n:6d} | {t * 1000:9.2f} | {ratio}")

    # log-log の傾き: 時間 ∝ n^slope の slope を最小二乗で推定(n² なら ≈ 2)
    slope = np.polyfit(np.log(NS), np.log(times), 1)[0]
    print("-" * 55)
    print(f"log-log の傾き: {slope:.2f}(n² 傾向なら 2 前後)")

    # === assert(緩め): 係数ではなく「傾向」だけを固定する ===
    # 1. n を増やすと時間は単調に増える
    for t_prev, t_next in zip(times, times[1:]):
        assert t_next > t_prev, "n を倍にしたのに時間が増えていない"

    # 2. 線形より明確に速く伸びる: n が16倍(256→4096)のとき、線形なら時間も
    #    16倍止まり。n² なら理論上256倍。中間の16倍超なら「超線形」と言える
    assert times[-1] / times[0] > (NS[-1] / NS[0]), (
        "時間の伸びが線形以下 — n² 傾向が観測できていない"
    )

    print()
    print("ok: n を増やすと実行時間は単調に増え、線形を明確に超えて伸びる")
    print("    (Table 1 の self-attention 行 O(n²·d) と整合)")
```

計測には2つの作法を入れてあります。各 $n$ で5回測って**最小値**を採るのは、たまたま OS が裏で忙しかった回に引きずられないため。assert を「単調増加 + 線形超え」という緩い条件にしているのは、実行時間が環境(BLAS、CPU、同時に動くプロセス)に強く依存し、「ぴったり4倍」を固定すると正しい実装でも環境差で落ちるためです。固定したいのは係数ではなく、**自乗で効くという傾向**です。

筆者の環境での実行結果はこうでした。

```
d_k = 64 固定、系列長 n を倍々に(各 5 回計測の最小値)

     n | 時間 (ms) | 直前との比 (n²なら≈4, 線形なら≈2)
-------------------------------------------------------
   256 |      0.21 |    -
   512 |      0.76 | 3.55
  1024 |      3.30 | 4.35
  2048 |     14.32 | 4.34
  4096 |     54.39 | 3.80
-------------------------------------------------------
log-log の傾き: 2.02(n² 傾向なら 2 前後)

ok: n を増やすと実行時間は単調に増え、線形を明確に超えて伸びる
    (Table 1 の self-attention 行 O(n²·d) と整合)
```

「直前との比」の列を縦に読んでください。3.55、4.35、4.34、3.80——$n$ を2倍にするたびに、時間はおよそ**4倍**です。締めくくりの log-log の傾き 2.02 は、「実行時間は $n$ のほぼ2乗に比例して伸びている」という意味です。Table 1 の左上のマス $O(n^2 \cdot d)$ が、自分のコードの実測として目の前に現れました。検算、完了です。

## 8.3 n² 問題: 系列が長いと自乗で効く

最後に、いま実測したばかりの数字を、逆側から眺め直します。

直前との比「約4倍」は、8.2 までの文脈では勝利の確認でした。しかし同じ数字は、**請求書**としても読めます。系列長を2倍にすると、計算時間は4倍。10倍にすれば100倍。しかも増えるのは時間だけではありません。attention の重み行列は `(n, n)` ですから、**メモリも自乗で**膨らみます。実測では $n = 4096$ で54ミリ秒——たった1層・1ヘッド・順方向だけでこれです。2017年の翻訳タスクでは $n$ は数十から百で、$n < d$ という条件に守られてこの請求書は無視できる額でした。しかし「文」が「文書」になり、書籍1冊やコードベース全体を丸ごと文脈に入れたくなった現代では、$n$ は数十万を超え、この $n^2$ こそが Transformer の最大の請求書になっています。

実は、論文自身がこの問題を予見しています。Table 1 の最終行 Self-Attention (restricted) がそれで、原文は「非常に長い系列のためには近傍 $r$ に制限する手があり、ただし経路長は $O(n/r)$ に伸びる。今後の研究課題としたい」と書いています。実際その後、見る相手を間引く方法(sparse attention)、計算の順序を工夫してメモリの自乗を回避する方法、そして再び状態を持ち回る設計への揺り戻しまで、「$n^2$ をどうにかする」研究が現在まで続いています。長文脈(long context)と呼ばれるこの研究領域の入り口は、すべて Table 1 左上のこの1マスです——本書の測地線は原論文までなので、これ以上は立ち入りませんが、いつかその扉を開くとき、あなたはこの章の検算をそのまま持っていけます。

## まとめ

- Section 4 は新しい部品のない**弁明のセクション**。3つの指標——層あたり計算量(電気代)・逐次操作数(順番待ちの回数)・最大経路長(2位置を結ぶ信号の道のり)——で層の方式を比較します。$O(\cdot)$ は定数倍を捨てて「増え方」だけを見る記法です
- 検算の結果: self-attention は計算量 $O(n^2 \cdot d)$・逐次 $O(1)$・経路長 $O(1)$、RNN は計算量 $O(n \cdot d^2)$・逐次 $O(n)$・経路長 $O(n)$。すべて自分の実装(第3章、第6巻5章)の構造から導けました
- Recurrent 行の逐次 $O(n)$ は**第6巻5.3の痛み1(並列化不能)**、経路長 $O(n)$ は**第6巻5.4の痛み2(長距離依存)**そのもの。Table 1 は第6巻の体感を記号に符号化した表であり、self-attention はその2マスを $O(1)$ にします
- 計算量の大小($n^2 d$ 対 $n d^2$)は $n$ と $d$ の関係次第。**$n < d$ のときだけ self-attention が安い**、という原文の条件付きの勝利宣言を読みました
- 実測でも、$n$ を2倍にするごとに実行時間は約4倍(log-log の傾き約2)。この $n^2$ は現代の長文脈研究が格闘し続けている請求書でもあります

**ラスボスとの距離**: Section 4 読了。これで Section 3 と合わせ、論文の心臓部はすべて読めました。残るは終章での通し再読と、第8巻の Section 5・6(訓練と結果)だけです。

## 演習

**問1** `code/ch08/attention_scaling.py` を自分の環境で実行し、「直前との比」と log-log の傾きを記録してください。さらに、第3章の `causal_mask(n)` を付けた場合($\mathrm{attention}(Q, K, V, \mathrm{mask})$)でも測り、$n^2$ 傾向が変わらないことを確認してください。

<details><summary>略解</summary>

比はおよそ4(環境により3〜5程度のぶれは正常)、傾きは2前後になります。mask を付けると `np.where` の適用が増えますが、それも `(n, n)` の表への操作、つまり $O(n^2)$ の仕事です。$O(n^2 \cdot d)$ という支配項は変わらないので、傾向(比≈4、傾き≈2)はそのまま残ります。定数倍が少し増えるだけです。

</details>

**問2** RNN の逐次性を模擬して、時間が**線形**に伸びることも確かめましょう。$W$ `(d, d)` と $\mathbf{h}$ `(d,)` を用意し、`for _ in range(n): h = np.tanh(W @ h)` を $n = 256, 512, 1024, 2048$ で計測してください(第6巻5章の実装を使ってもかまいません)。直前との比はいくつになるはずですか。

<details><summary>略解</summary>

1ステップの仕事 $d^2$ は $n$ に依存しないので、時間は $n$ に比例し、比は**約2**になります(log-log の傾きは約1)。比だけ見れば RNN の方が「伸びが穏やか」ですが、この計測ループの1周1周が**直列の順番待ち**であることが痛み1の本体です。attention の $n^2$ は独立な仕事の山なので計算機を並べて崩せますが、RNN の $n$ は何台並べても崩せません。Table 1 の計算量の列と逐次操作数の列は、別々の病気を測っているのです。

</details>

**問3** 論文の base model の $d = 512$ について、$n^2 d$(self-attention)と $n d^2$(RNN)の演算回数を $n = 50, 512, 5000$ で具体的に計算し、大小を比べてください。攻守が入れ替わる境目はどこですか。

<details><summary>略解</summary>

比は $\frac{n^2 d}{n d^2} = \frac{n}{d}$ なので、境目は $n = d = 512$ です。$n = 50$ では $n^2 d = 1.28 \times 10^9$、$n d^2 = 1.31 \times 10^{10}$ で self-attention が約10分の1と圧勝。$n = 512$ で両者は $1.34 \times 10^{11}$ に並び、$n = 5000$ では $n^2 d = 1.28 \times 10^{13}$ 対 $n d^2 = 1.31 \times 10^{12}$ で self-attention が約10倍重くなります。原文の "when the sequence length n is smaller than the representation dimensionality d" は、この境目のことです。

</details>

## 論文の主張 ↔ 本章の対応表

| 論文の箇所・主張 | 本章での対応 |
|---|---|
| Section 4 冒頭: 3つの desiderata(計算量・逐次操作数・経路長) | 8.1(3指標の日本語化と $O(\cdot)$ 記法) |
| 経路長と長距離依存の学習("The shorter these paths ...") | 8.1 指標3(第5巻6.1・第6巻5.4と接続) |
| Table 1 全体 | 8.1 で再掲、8.2 で行ごとに検算 |
| Self-Attention 行 $O(n^2 \cdot d)$ / $O(1)$ / $O(1)$ | 8.2(第3章 `attention` の3行から導出)+ `code/ch08/attention_scaling.py`(`measure()` で実測、比≈4・傾き≈2) |
| Recurrent 行 $O(n \cdot d^2)$ / $O(n)$ / $O(n)$ | 8.2(第6巻5章の実装から導出。痛み1=逐次 $O(n)$、痛み2=経路長 $O(n)$)・演習問2 |
| "faster ... when n is smaller than d" | 8.2(比 $n/d$ の導出)・演習問3 |
| Convolutional 行・restricted 行 | 8.2(概観のみ・最短測地線) |
| "could be restricted to a neighborhood of size r ... O(n/r)" | 8.3($n^2$ 問題と長文脈研究の入り口) |
| "As side benefit ... more interpretable models" | 8.2(第6巻7.3・第4章演習の可視化との接続) |
