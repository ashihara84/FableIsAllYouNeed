# 第7章 3.5 Positional Encoding — 語順を取り戻す

前章までで、Section 3.2 の attention 一族と、3.3 の FFN、3.4 の埋め込み・出力 head を読み終えました。巻頭の論文読解マップで Section 3 に残っている「未」は、ただ1つ——3.5 Positional Encoding です。

ただし、この章は「最後の1ピースを埋める消化試合」ではありません。先に進む前に、片付けなければならない嫌な予感があります。第6巻終章に、私たちはこんな問いを残してきました。

**RNN を本当に取り除いたら、語順の情報はどうなるのか?**

RNN は、語順を「読む順序」として構造そのものに焼き込んでいました。$h_t$ は $h_{t-1}$ の後にしか計算できない——あの並列化を阻む足枷(第6巻5章の痛み1)は、裏を返せば「1番目、2番目、3番目」という順序の無料の刻印でもあったのです。Transformer はその RNN を捨てました。では、「太郎が犬を追う」と「犬が太郎を追う」を、このモデルはどう区別するのでしょうか。

この章では、まずこの不安が**現実の故障である**ことをコードで確定させます(7.1)。それから論文 3.5 の処方箋——sin と cos——を逐行で読み(7.2)、なぜその形なのかを小さな数値例で確かめ(7.3)、実装してテストし(7.4)、論文が比較検討したもう1つの選択肢を読んで締めます(7.5)。使う数学は高校の三角関数だけです。


## 7.1 問題の確認: attention は集合演算 — 並べ替えても結果が同じ

式(1)をもう一度見てください。

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

この式のどこに「位置」があるでしょうか。スコア行列 $QK^T$ の $(i, j)$ 成分は $\mathbf{q}_i \cdot \mathbf{k}_j$——行の**中身**どうしの内積です。$i$ が文頭なのか文末なのかは、式のどこにも現れません。softmax も重み付き和も同じです。各行は「他の行に何が書いてあるか」だけを見ていて、「それが何番目の行か」を知る手段を持っていません。

だとすると、恐ろしい予言が立ちます。**入力 $X$ の行を並べ替えても、出力は「同じ並べ替えを受けた元の出力」になるはず**——つまり、各トークンが受け取る表現ベクトルは、語順をどう変えても1ビットも変わらないはずです。予言はコードで確定させましょう。第3章で実装した attention をそのまま使います。

```python
# 第7巻 第7章 7.1: attention は集合演算であることの実験
# 入力の行を並べ替えても、出力は「同じ並べ替え」を受けるだけで中身が変わらないことを確かめる
import os
import sys

import numpy as np

# 第3章 3.4 で実装した attention(式(1))をそのまま使う
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "ch03"))
from attention import attention

rng = np.random.default_rng(42)


def self_attention(X, W_Q, W_K, W_V):
    """self-attention: Q, K, V がすべて同じ X から作られる(第5章)"""
    out, _ = attention(X @ W_Q, X @ W_K, X @ W_V)   # (n, d_model)
    return out


# 6 トークンの文のつもり。各行が 1 トークンの埋め込みベクトル
n, d_model = 6, 16
X = rng.normal(0, 1, size=(n, d_model))                              # (n, d_model)
W_Q = rng.normal(0, 1.0 / np.sqrt(d_model), size=(d_model, d_model))
W_K = rng.normal(0, 1.0 / np.sqrt(d_model), size=(d_model, d_model))
W_V = rng.normal(0, 1.0 / np.sqrt(d_model), size=(d_model, d_model))

out = self_attention(X, W_Q, W_K, W_V)                               # (n, d_model)

# 語順をめちゃくちゃにする: 行を 3, 0, 5, 1, 4, 2 の順に並べ替え
perm = np.array([3, 0, 5, 1, 4, 2])
out_shuffled = self_attention(X[perm], W_Q, W_K, W_V)

# 出力は「同じ並べ替えを受けた元の出力」と完全に一致する
assert np.allclose(out_shuffled, out[perm], atol=1e-12)

# 1 トークンずつ見ても同じ: 各トークンが受け取る表現は語順に 1 ビットも依存しない
for new_pos, old_pos in enumerate(perm):
    assert np.allclose(out_shuffled[new_pos], out[old_pos], atol=1e-12)

print("並べ替えの前後で、各トークンの出力ベクトルは完全に一致しました")
print("すべての assert を通過しました")
```

```
並べ替えの前後で、各トークンの出力ベクトルは完全に一致しました
すべての assert を通過しました
```

実験の中身を確認します。6トークンの文に見立てた $X$ `(6, 16)` の行を、`perm = [3, 0, 5, 1, 4, 2]` という滅茶苦茶な順に並べ替えてから self-attention に通しました。1つ目の assert は、その出力が「元の出力に同じ並べ替えを施したもの」と `atol=1e-12` で一致する——事実上ビット単位で同じ——ことを言っています。2つ目の assert はもっと生々しい確認です。元の文で3番目だったトークンは、シャッフル後の文では先頭にいますが、**受け取る表現ベクトルは完全に同一**です。「犬」というトークンが文のどこに置かれていても、attention を通った後の「犬」の中身は同じなのです。

「並べ替えても結果が同じ」の正確な意味はこうです。出力の行の**順番**は入力に追随して変わりますが、各トークンに紐づく中身は不変——行の集合として見れば、出力は完全に同一です。この性質を並べ替え同変性(permutation equivariance)と呼びますが、名前より本質が大事です。**attention にとって、入力は系列(sequence)ではなく集合(set)です。** 私たちは6個のベクトルを「文」のつもりで渡していましたが、attention が見ていたのは順序情報のないベクトルの袋でした。

しかも、これは attention だけの病気ではありません。multi-head(第4章)は同じ attention の束なので同罪です。FFN は "position-wise"(第6章)——各行に独立に同じ MLP を適用するだけで、行番号は使いません。residual も layer norm(第2章)も行単位の操作です。つまり、**ここまで組み上げてきた部品のすべてが語順を見ていません**。このまま第8巻で組み立てれば、「太郎が犬を追う」と「犬が太郎を追う」に対して、各単語はまったく同じ表現を受け取ります。

第6巻終章の問いその1「RNN を本当に取り除いたら、語順の情報はどうなるのか」——答えは「**消える**」でした。RNN が構造の副産物として無料でくれていた語順を、Transformer は自前で調達しなければなりません。構造が運んでくれないのなら、残る手段は1つ、**データに混ぜて入り口から入れる**ことです。それが論文 Section 3.5 の仕事です。

## 7.2 原文逐行: sin / cos の式(3) — 波長が幾何級数で並ぶ設計

原文を読みます。

> *"Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension $d_{model}$ as the embeddings, so that the two can be summed."*
> — Vaswani et al., "Attention Is All You Need", Section 3.5
>
> 訳: 我々のモデルは再帰も畳み込みも含まないので、モデルが系列の順序を利用できるようにするには、トークンの相対的または絶対的な位置に関する何らかの情報を注入しなければならない。そのために、encoder と decoder のスタックの最下部で、入力埋め込みに「positional encoding」を加算する。positional encoding は埋め込みと同じ次元 $d_{model}$ を持ち、それゆえ両者は足し合わせることができる。

逐行で読みます。"contains no recurrence and no convolution" は、7.1 でコードが突きつけた事実そのものです。"inject"(注入する)という動詞の選択に注目してください——RNN のように構造で順序を持つのではなく、**入力データに位置情報を混ぜ込む**という方針宣言です。"at the bottoms of the stacks" は、注入はスタックの入り口で**一度だけ**という意味です。N = 6 段(第2章)の各段で足し直すのではありません。そして "the same dimension $d_{model}$ ... so that the two can be summed"——埋め込みと**足し算**するために形を揃えた、とあります。連結ではなく加算を選んだこの一言の意味は、7.4 で実装に触れてから考えます。

では、何を注入するのか。続く式がこの章の主役です。

> $$PE_{(pos, 2i)} = \sin\!\left(pos / 10000^{2i/d_{model}}\right)$$
> $$PE_{(pos, 2i+1)} = \cos\!\left(pos / 10000^{2i/d_{model}}\right)$$
> — 同論文, Section 3.5(原文では式番号なし。本シリーズでは式(3)と呼ぶ)
>
> 訳: 位置 $pos$、次元 $2i$ 番目の成分は $\sin(pos/10000^{2i/d_{model}})$、次元 $2i+1$ 番目の成分は $\cos(pos/10000^{2i/d_{model}})$。

記号を1つずつ確定させます。$PE$ は `(max_len, d_model)` の行列で、$pos$ がその行番号(位置: 0, 1, 2, ...)、$i$ が**列のペア番号**です。$d_{model}$ 本の列は2本ずつ組になっていて、偶数列 $2i$ が sin、奇数列 $2i+1$ が cos を担当します。1つのペアは共通の角度 $pos \cdot \omega_i$ を使います。ここで

$$\omega_i = \frac{1}{10000^{2i/d_{model}}}$$

と置きました。つまりペア $i$ の中身は $(\sin(pos\,\omega_i),\ \cos(pos\,\omega_i))$ です。

この構造には、よく効く見立てがあります。**$PE$ の各ペアは、次元ごとに違う速さで回る時計の針です。** $pos$ を時刻だと思ってください。ペア $i$ は角速度 $\omega_i$ で回る針で、sin と cos はその針の先端の座標です。$i = 0$ の針は最速で、1トークン進むごとに1ラジアン、約6.3トークンで文字盤を一周します。$i$ が1つ増えるごとに針は $10000^{2/d_{model}}$ 倍(d_model = 512 なら約1.037倍)ずつ遅くなり、最後のペアは一周に約6万トークンかかります。d_model = 512 での実際の値を表にします。

| ペア $i$ | 角速度 $\omega_i$ | 波長(一周にかかるトークン数 $2\pi/\omega_i$) |
|---:|---:|---:|
| 0 | 1 | 6.28 |
| 1 | 0.9647 | 6.51 |
| 64 | 0.1 | 62.8 |
| 128 | 0.01 | 628 |
| 255 | 0.000104 | 約 60,600 |

原文は続けてこう言います。"The wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$."(波長は $2\pi$ から $10000 \cdot 2\pi$ までの等比数列をなす)。表の右列がまさにそれで、隣のペアと比べると波長が常に一定倍率で伸びていきます。$10000^{2i/d_{model}}$ という一見ぎょっとする分母は、「波長を等比で並べる」と決めた瞬間に出てくる素直な式です。

なぜ針が1本ではいけないのでしょうか。最速の針だけだと、6.3トークンごとに同じ値に戻ってしまい、位置 0 と位置 6 の区別がつきません(周期性の衝突)。逆に最遅の針だけだと、隣の位置との差が 0.0001 ラジアン——あまりに微小で、位置の違いがほとんど信号になりません。時計が時針・分針・秒針を併用するのと同じ理屈で、**速い針が近距離の分解能を、遅い針が遠距離の一意性を受け持ち、全部の針を同時に読めば広い範囲の位置が事実上一意に決まる**のです。

## 7.3 なぜこの形か: 相対位置が線形変換で表せる

それにしても、位置を表すだけなら $PE_{pos} = pos$(行番号をそのまま書く)でもよさそうなものです。なぜ三角関数なのか。論文は理由を一文で述べています。

> *"We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$."*
> — Vaswani et al., "Attention Is All You Need", Section 3.5
>
> 訳: この関数を選んだのは、固定されたずれ幅 $k$ に対して $PE_{pos+k}$ が $PE_{pos}$ の線形関数で表せるため、モデルが相対位置に基づく attention を容易に学習できるだろうと我々が仮説を立てたからである。

まず、なぜ「線形関数で表せる」と嬉しいのかを考えます。attention が位置情報を使う経路は $W^Q$, $W^K$ による射影(第4章)——つまり**線形変換**です。言語では「2つ前の単語を見る」のような相対位置の関係が重要ですが、「$k$ 個ずれた位置の $PE$」が行列1つの掛け算で作れる形になっていれば、その種の関係は線形変換しか持たないモデルにとって学びやすいはずだ——という理屈です。注意してほしいのは "hypothesized"(仮説を立てた)という語で、原文自身がこれを証明済みの定理ではなく設計仮説として提示しています。私たちも厳密な証明はしません。主張そのもの——$PE_{pos+k}$ が $PE_{pos}$ の線形関数であること——を、確かめ算で検証します。

道具は高校数学の加法定理です。

$$\sin(\alpha + \beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta, \qquad \cos(\alpha + \beta) = \cos\alpha\cos\beta - \sin\alpha\sin\beta$$

$\alpha = pos\,\omega_i$、$\beta = k\,\omega_i$ と置いてペア $i$ の2成分に当てはめると、

$$\begin{pmatrix} \sin((pos+k)\,\omega_i) \\ \cos((pos+k)\,\omega_i) \end{pmatrix} = \begin{pmatrix} \cos k\omega_i & \sin k\omega_i \\ -\sin k\omega_i & \cos k\omega_i \end{pmatrix} \begin{pmatrix} \sin(pos\,\omega_i) \\ \cos(pos\,\omega_i) \end{pmatrix}$$

右辺の $2 \times 2$ 行列は回転行列(第1巻5章)です——針の先端を固定角 $k\omega_i$ だけ回す変換にほかなりません。そして決定的なのは、**この行列の成分に $pos$ が入っていない**ことです。$k$ と $\omega_i$ だけから作れます。針の言葉で言えば当然で、「時刻を $k$ 進める」とは、いまが何時であろうと各針を同じ角度だけ回すことだからです。ペアごとのこの $2 \times 2$ を対角線上に $d_{model}/2$ 個並べたブロック対角行列を $M_k$ `(d_model, d_model)` とすれば、$PE_{pos+k} = M_k \, PE_{pos}$——線形関数で表せました。

小さな数値例で確かめ算をします。$d_{model} = 4$(ペアは2つ、$\omega_0 = 1$、$\omega_1 = 0.01$)、$pos = 3$、$k = 2$ とします。実装(7.4)で値を出すと、

$$PE_3 = (0.1411,\ -0.9900,\ 0.0300,\ 0.9996), \qquad PE_5 = (-0.9589,\ 0.2837,\ 0.0500,\ 0.9988)$$

$k = 2$ **だけ**から作った変換行列は

$$M_2 = \begin{pmatrix} -0.4161 & 0.9093 & 0 & 0 \\ -0.9093 & -0.4161 & 0 & 0 \\ 0 & 0 & 0.9998 & 0.0200 \\ 0 & 0 & -0.0200 & 0.9998 \end{pmatrix}$$

で、$M_2 \, PE_3 = (-0.9589,\ 0.2837,\ 0.0500,\ 0.9988)$。$PE_5$ と4桁すべて一致します。$M_2$ を作るとき $pos = 3$ という情報は一度も使っていないので、同じ $M_2$ が $PE_0 \to PE_2$ にも $PE_{10} \to PE_{12}$ にも通用するはずです。この「同じ行列が全位置で通用する」ことの全数チェックは、7.4 のテスト4(`offset_matrix`)が引き受けます。

図7.1 は、この回転の様子を最速ペア $i = 0$ で描いたものです(コードは掲載のみ)。

```python
import matplotlib.pyplot as plt
import numpy as np

pos = np.arange(10)
s, c = np.sin(pos * 1.0), np.cos(pos * 1.0)   # ペア i=0(ω=1)の sin, cos

fig, ax = plt.subplots(figsize=(5, 5))
theta = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(theta), np.sin(theta), alpha=0.3)   # 単位円
ax.scatter(c, s)
for p in pos:
    ax.annotate(str(p), (c[p], s[p]))
ax.set_xlabel("cos")
ax.set_ylabel("sin")
ax.set_aspect("equal")
plt.show()
```

図7.1: ペア $i=0$ の $(\cos, \sin)$ を平面に打った図。位置 0, 1, 2, ... の点が単位円の上を**等しい角度間隔**で進んでいく。「位置を $k$ 進める」がどの点から出発しても同じ回転になることが、円周上の等間隔さとして見える。

## 7.4 実装とテスト — ヒートマップ、そして「足す」という選択

実装します。式(3)は2行のブロードキャスト(第1巻6章)で書けます。このファイルは**第8巻が import して Transformer の組み立てに使う基盤部品**です。

```python
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
```

中身は3行です。`pos` `(max_len, 1)` と `i` `(1, d_model/2)` の割り算がブロードキャストで `angle` `(max_len, d_model/2)` になり、その sin を偶数列 `0::2` に、cos を奇数列 `1::2` に流し込みます。乱数も学習パラメータも登場しない、完全に決め打ちの定数行列であることに注意してください。

中身を数値でも見ておきます。`positional_encoding(6, 8)` の全成分です($d_{model} = 8$ なのでペアは4つ、$\omega$ は左から 1, 0.1, 0.01, 0.001)。

| $pos$ | sin$_0$ | cos$_0$ | sin$_1$ | cos$_1$ | sin$_2$ | cos$_2$ | sin$_3$ | cos$_3$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| 1 | 0.841 | 0.540 | 0.100 | 0.995 | 0.010 | 1.000 | 0.001 | 1.000 |
| 2 | 0.909 | −0.416 | 0.199 | 0.980 | 0.020 | 1.000 | 0.002 | 1.000 |
| 3 | 0.141 | −0.990 | 0.296 | 0.955 | 0.030 | 1.000 | 0.003 | 1.000 |
| 4 | −0.757 | −0.654 | 0.389 | 0.921 | 0.040 | 0.999 | 0.004 | 1.000 |
| 5 | −0.959 | 0.284 | 0.479 | 0.878 | 0.050 | 0.999 | 0.005 | 1.000 |

左のペアほど行ごとに激しく動き、右のペアはほぼ $(0, 1)$ から動きません。そして、**どの2つの行も互いに異なります**。各行が、その位置の指紋になっているのです。

この行列を大きなサイズで画像にしたのが、よく論文の解説記事で見かける縞模様のヒートマップです(コードは掲載のみ)。

```python
import matplotlib.pyplot as plt
from positional_encoding import positional_encoding

pe = positional_encoding(100, 512)   # (100, 512)
fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(pe, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
ax.set_xlabel("dimension")
ax.set_ylabel("position")
fig.colorbar(im, ax=ax)
plt.show()
```

図7.2: positional encoding のヒートマップ(横軸が次元、縦軸が位置、色が値)。左端の列は縦方向に細かく振動する縞、右へ進むほど縞の間隔が広がり、右端はほぼ一様な色になる。縞の太さ = 波長が左から右へ等比数列で太っていく様子が、そのまま模様として見える。上の数値表は、この図の左上の小さな切れ端を数字で読んだものです。

テストを書きます。7.2 と 7.3 で読んだ主張を、そのまま assert に翻訳したものです。

```python
# 第7巻 第7章 7.4: positional_encoding のテスト
# python3 test_positional_encoding.py で実行。第8巻はこのテストが通ることを前提に import する
import os
import sys

import numpy as np

from positional_encoding import positional_encoding

# 第3章 3.4 で実装した attention(式(1))をテスト6で使う
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "ch03"))
from attention import attention

rng = np.random.default_rng(42)

# --- テスト1: shape と決定性(乱数を使わないので、何度作っても同じ行列) ---
max_len, d_model = 40, 32
pe = positional_encoding(max_len, d_model)
assert pe.shape == (max_len, d_model)
assert np.array_equal(pe, positional_encoding(max_len, d_model))

# --- テスト2: 式(3)との全数一致(定義どおりの素朴な二重ループで検算) ---
pe_naive = np.zeros((max_len, d_model))
for pos in range(max_len):
    for idx in range(d_model):
        i = idx // 2
        angle = pos / 10000.0 ** (2.0 * i / d_model)
        pe_naive[pos, idx] = np.sin(angle) if idx % 2 == 0 else np.cos(angle)
assert np.allclose(pe, pe_naive, atol=1e-12)

# --- テスト3: 値域。位置がどれだけ先でも全成分が [-1, 1] に収まる(7.5 の外挿の根拠) ---
assert np.all(np.abs(positional_encoding(10000, d_model)) <= 1.0)

# --- テスト4: 相対位置の線形性(7.3 の原文の主張)。
#     どの位置 pos でも PE[pos+k] = M_k @ PE[pos]。M_k は k だけから作れて pos に依らない ---


def offset_matrix(k, d_model):
    """位置を k だけ進める線形変換 (d_model, d_model)。
    sin/cos のペアごとに 2×2 の回転をブロック対角に並べたもの(加法定理そのまま)"""
    M = np.zeros((d_model, d_model))
    for i in range(d_model // 2):
        omega = 1.0 / 10000.0 ** (2.0 * i / d_model)
        c, s = np.cos(k * omega), np.sin(k * omega)
        M[2 * i:2 * i + 2, 2 * i:2 * i + 2] = np.array([[c, s],
                                                        [-s, c]])
    return M


for k in [1, 3, 10]:
    M_k = offset_matrix(k, d_model)          # k から一度だけ作る(pos を知らない)
    # 全位置にまとめて M_k を適用(行ごとの M_k @ pe[pos] と同じ。
    # ブロック対角行列の @ は環境によって誤警告を出すため dot で書く)
    shifted = pe[:max_len - k].dot(M_k.T)    # (max_len-k, d_model)
    assert np.allclose(pe[k:], shifted, atol=1e-9)   # その同じ M_k が全位置で通用する

# --- テスト5: 内積が位置差だけで決まる: PE[p]・PE[q] = Σ_i cos((p−q)ω_i)(演習2の根拠) ---
for delta in [1, 5, 12]:
    dots = np.array([pe[p] @ pe[p + delta] for p in range(max_len - delta)])
    assert np.allclose(dots, dots[0], atol=1e-9)        # どの p でも同じ値
omega = 1.0 / 10000.0 ** (2.0 * np.arange(d_model // 2) / d_model)
assert np.allclose(pe[7] @ pe[12], np.sum(np.cos(5 * omega)), atol=1e-9)

# --- テスト6: PE を足すと attention の並べ替え不変性が壊れる(7.1 の問題の解決の検収) ---
n = 6
X = rng.normal(0, 1, size=(n, d_model))
W_Q = rng.normal(0, 1.0 / np.sqrt(d_model), size=(d_model, d_model))
W_K = rng.normal(0, 1.0 / np.sqrt(d_model), size=(d_model, d_model))
W_V = rng.normal(0, 1.0 / np.sqrt(d_model), size=(d_model, d_model))
perm = np.array([3, 0, 5, 1, 4, 2])


def self_attention(X, W_Q, W_K, W_V):
    out, _ = attention(X @ W_Q, X @ W_K, X @ W_V)
    return out


# PE なし: 並べ替えと出力が交換する(7.1 の再現)
out = self_attention(X, W_Q, W_K, W_V)
assert np.allclose(self_attention(X[perm], W_Q, W_K, W_V), out[perm], atol=1e-12)

# PE あり: トークンを並べ替えても PE は位置 0, 1, 2, ... の順のまま足される
pe_n = positional_encoding(n, d_model)                   # (n, d_model)
out_pe = self_attention(X + pe_n, W_Q, W_K, W_V)
out_pe_shuffled = self_attention(X[perm] + pe_n, W_Q, W_K, W_V)
assert not np.allclose(out_pe_shuffled, out_pe[perm], atol=1e-6)

# --- テスト7: 第8巻との契約。d_model が奇数なら明示的に拒否する ---
try:
    positional_encoding(10, 7)
    raise AssertionError("奇数の d_model が通ってしまった")
except ValueError:
    pass

print("すべての assert を通過しました")
```

```
すべての assert を通過しました
```

読みどころを3つ挙げます。テスト2は、ベクトル化した実装を**定義どおりの素朴な二重ループ**で全数検算しています(速い実装を遅い実装で検算する、第1巻4章以来の流儀です)。テスト4は 7.3 の確かめ算の全数版で、`offset_matrix(k, d_model)` が $pos$ を一切受け取らずに作られている——にもかかわらず全位置で通用する——ことが、forループの構造そのものに表れています。

そしてテスト6が、この章のクライマックスです。`X + pe_n`——埋め込みに PE を**足してから** self-attention に通すと、7.1 で成立していた assert が `assert not np.allclose(...)` に**反転**します。トークンを並べ替えても PE は位置 0, 1, 2, ... の順で足されるので、同じトークンでも置かれた位置によって違う入力ベクトルになり、出力が変わる。モデルが語順を受け取った瞬間です。第8巻は、このテストが通っていることを前提に `positional_encoding` を import します。

最後に、7.2 で保留した「足す」という選択を一言。位置情報を入れるなら、埋め込みの隣に**連結**(concatenation)して `(n, d_model + d_pe)` にする手もあったはずです。加算を選ぶ利点は、$d_{model}$ が変わらないことです。連結すると後段のすべての $W$ が太り、residual の「全部 $d_{model}$ で揃える」という規約(第2章)も崩れます。「内容と位置を同じ512次元に混ぜて、互いに潰し合わないのか」という不安は当然ですが、埋め込み自体が学習されるパラメータである(第6章)ことが答えの半分です——モデルは、PE と干渉しにくい置き場所に意味を配置するよう学習できます。残りの半分は「実験的にこれでうまくいった」という事実で、原文もこの点をそれ以上は論じていません。

## 7.5 learned positional embedding との比較

論文 3.5 の最終段落は、設計の比較です。

> *"We also experimented with using learned positional embeddings instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than those encountered during training."*
> — Vaswani et al., "Attention Is All You Need", Section 3.5
>
> 訳: 代わりに学習型の positional embedding を使う実験も行ったが、両者はほぼ同一の結果を生んだ(Table 3 の行 (E) を参照)。我々が正弦波版を選んだのは、訓練中に遭遇したより長い系列長へモデルが外挿できる可能性があるからである。

learned positional embedding(学習型位置埋め込み)とは、`(max_len, d_model)` の行列を**パラメータとして**持ち、位置 $pos$ の行をそのまま使う方式です。仕掛けは第6巻3章の埋め込み行列とまったく同じで、対象が「単語」から「位置番号」に変わっただけ——位置の表現を設計せず、モデル自身に考えさせます。

結果は "nearly identical"。つまり、7.2〜7.3 で読み解いた幾何級数も加法定理も、**性能のためには必須ではなかった**のです。少し拍子抜けしますが、これは誠実な報告です。そのうえで著者らが sin/cos を選んだ決め手が、外挿(extrapolation)でした。learned 方式は訓練時に確保した `max_len` 行しか持たず、それより長い入力には対応する行がそもそも**存在しません**。一方 sin/cos は任意の $pos$ に値を返し、しかもテスト3で確認したとおり、位置がどれだけ先でも全成分が $[-1, 1]$ に収まります。"may allow"(可能性がある)という控えめな助動詞も読みどころで、これは保証ではなく期待です。

なお、その後の歴史では learned 方式を採るモデルも、相対位置や回転を使うさらに別の方式も登場し、決定版は今も1つに定まっていません。私たちは深追いしません(最短測地線)。この論文の時点での結論——「どちらでもほぼ同じ。なら、外挿に期待が持てる方を」——が読めれば、Section 3.5 は完読です。

## まとめ

- **attention は集合演算**: 入力の行を並べ替えると、出力は同じ並べ替えを受けるだけで各トークンの中身は1ビットも変わらない(第3章の `attention` を使ってコードで確認)。multi-head・FFN・residual・layer norm もすべて行単位の操作で、語順はモデルのどこにも届いていなかった——第6巻終章の問いその1の答えは「語順は消える」
- 処方箋は式(3): 列をペアにして、角速度 $\omega_i = 1/10000^{2i/d_{model}}$ の sin/cos を並べる。波長は $2\pi$ から $10000 \cdot 2\pi$ までの等比数列で、速いペアが近距離の分解能、遅いペアが遠距離の一意性を担う
- 相対位置のシフトは線形変換: 加法定理により $PE_{pos+k} = M_k\,PE_{pos}$。$M_k$ は $k$ だけから作れるブロック対角の回転行列で、$pos$ に依らない(数値例 + テスト4で全数確認。厳密証明はせず、原文も「仮説」と明言)
- 実装はパラメータゼロの定数行列で、埋め込みに**連結ではなく加算**する($d_{model}$ を保ち、residual の規約とも整合)。足した瞬間、並べ替え不変性が壊れる = モデルが語順を取り戻す(テスト6)
- learned positional embedding と性能はほぼ同じ。訓練長を超える系列への**外挿**の期待で sin/cos が選ばれた

**ラスボスとの距離**: Section 3.5 完読。これで **Section 3 に読めない文は1つも残っていません**。部品も `positional_encoding` で全点揃いました。残るは Section 4 の「なぜ self-attention か」の比較表(次章)と、全部品の組み立て(第8巻)です。

## 演習

**問1**(電卓で式(3))$d_{model} = 4$ のとき、位置 $pos = 2$ の $PE$ の4成分を電卓(または `np.sin` の単発呼び出し)で計算してください。そのうえで `positional_encoding(3, 4)` の最終行と照合してください。

**問2**(内積を位置差ごとにプロットする)$d_{model} = 64$ の PE で、内積 $PE_p \cdot PE_{p+\Delta}$ を $\Delta = 0, 1, \dots, 50$ についてプロットしてください。起点を $p = 0, 20, 50$ と変えて3本の曲線を重ね描きすること。曲線の形・最大値・$p$ を変えたときの変化について、見えたことを言語化してください。

**問3**(外挿の感触)`positional_encoding(1_000_001, 64)` を作り、最終行(位置100万)の全成分が $[-1, 1]$ に収まることを assert で確かめてください。learned positional embedding(max_len = 512 で訓練したとする)に位置100万を要求すると何が起きるかも考えてください。

<details>
<summary>略解</summary>

**問1** ペアは2つで $\omega_0 = 1$、$\omega_1 = 0.01$。$PE_2 = (\sin 2,\ \cos 2,\ \sin 0.02,\ \cos 0.02) = (0.9093,\ -0.4161,\ 0.0200,\ 0.9998)$。`positional_encoding(3, 4)[2]` と4桁一致すれば正解です。位置 2 で最速の針はもう約 115 度も回っているのに、最遅の針はまだ 0.02 ラジアン——針の速さの差を手で感じられる例です。

**問2** プロットのコード(掲載のみ):

```python
import matplotlib.pyplot as plt
import numpy as np
from positional_encoding import positional_encoding

pe = positional_encoding(200, 64)
deltas = np.arange(0, 51)
fig, ax = plt.subplots(figsize=(7, 4))
for p in [0, 20, 50]:
    ax.plot(deltas, [pe[p] @ pe[p + d] for d in deltas], label="p = {}".format(p))
ax.set_xlabel("offset Δ")
ax.set_ylabel("dot product")
ax.legend()
plt.show()
```

手元の数値はこうなります($p$ によらず同一)。

| $\Delta$ | 0 | 1 | 2 | 3 | 5 | 10 | 20 | 50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $PE_p \cdot PE_{p+\Delta}$ | 32.000 | 30.917 | 28.304 | 25.587 | 23.504 | 21.052 | 18.879 | 15.674 |

見えること: (1) 3本の曲線は**完全に重なります**。加法定理から $PE_p \cdot PE_{p+\Delta} = \sum_i \cos(\Delta\,\omega_i)$ となり、$p$ が式から消えるためです(テスト5の中身)。(2) 最大値は $\Delta = 0$ の 32 で、これは $\sin^2 + \cos^2 = 1$ がペア数 $d_{model}/2$ 個ぶん積み上がった値です。(3) $\Delta$ が大きくなるほど内積はおおむね減っていきます(速い針の周期性で細かい凸凹は出ます)。第1巻2章の「内積 = 類似度」がここでも効いていて、**PE のベクトル空間では「位置が近い」が「似ている」として表現されている**——attention は内積しか測れませんが、その内積に位置の近さがちゃんと写るのです。

**問3** `pe = positional_encoding(1_000_001, 64)` に対して `assert np.all(np.abs(pe[-1]) <= 1.0)` が通ります(手元では最大絶対値 0.99999)。式(3)は $pos$ がいくつでも有界な値を返す——これが外挿の最低条件です。learned 方式では位置100万の行はそもそも存在しないため、表引きが範囲外エラーになります。動かすには行列を拡張して再訓練するしかなく、「訓練で見た長さまでしか動けない」という learned 方式の構造的な限界がここに出ます(ただし sin/cos なら長い系列で**性能が良い**ことまでは保証されません。原文の "may" の慎重さを思い出してください)。

</details>

## 式・主張 ↔ コード 対応表

| 論文 Section 3.5 の式・主張 | 本章のコード |
|---|---|
| (前提)"contains no recurrence and no convolution" — 語順が消えている | `code/ch07/permutation_invariance.py` 全体(assert 2組) |
| 式(3) $PE_{(pos,2i)} = \sin(\cdot)$, $PE_{(pos,2i+1)} = \cos(\cdot)$ | `code/ch07/positional_encoding.py` の `angle` / `pe[:, 0::2]` / `pe[:, 1::2]` の3行(全数検算はテスト2) |
| "wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$" | 同上 `angle` の分母 `10000.0 ** (2.0 * i / d_model)`(7.2 の波長表) |
| "the two can be summed" — 埋め込みへの加算 | `code/ch07/test_positional_encoding.py` テスト6 の `X + pe_n` |
| "$PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$" | 同 `offset_matrix()` とテスト4 |
| "extrapolate to sequence lengths longer than those encountered during training" | 同 テスト3(位置10000でも値域 $[-1,1]$)+ 演習3 |
| (テスト5・演習2)内積が位置差だけで決まる | 同 テスト5 |
