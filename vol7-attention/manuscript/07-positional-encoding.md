# 第7章 3.5 Positional Encoding — 語順を取り戻す

> [目次](../TOC.md) ・ [← 前の章](06-ffn-embeddings.md) ・ [次の章 →](08-why-self-attention.md)

巻頭の論文読解マップで Section 3 に残った「未」は、3.5 Positional Encoding ただ1つです。ただしこれは消化試合ではありません。第6巻終章に、私たちはこの問いを残してきました。

**RNN を本当に取り除いたら、語順の情報はどうなるのか?**

RNN は語順を「読む順序」として構造に焼き込んでいました。$h_t$ は $h_{t-1}$ の後にしか計算できない——並列化を阻む足枷(第6巻5章の痛み1)は、裏を返せば「1番目、2番目、3番目」という順序の無料の刻印でもありました。Transformer はその RNN を捨てた。では「太郎が犬を追う」と「犬が太郎を追う」をどう区別するのでしょうか。

この章では、まずこの不安が**現実の故障である**ことをコードで確定させ(7.1)、論文 3.5 の処方箋——sin と cos——を逐行で読み(7.2)、なぜその形なのかを数値例で確かめ(7.3)、実装してテストし(7.4)、もう1つの選択肢を読んで締めます(7.5)。使う数学は高校の三角関数だけです。


## 7.1 問題の確認: attention は集合演算 — 並べ替えても結果が同じ

式(1)をもう一度見ます。

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

この式のどこに「位置」があるでしょうか。スコア行列 $QK^T$ の $(i, j)$ 成分は $\mathbf{q}_i \cdot \mathbf{k}_j$——行の**中身**どうしの内積で、$i$ が文頭か文末かは式のどこにも現れません。softmax も重み付き和も同じです。各行は「他の行に何が書いてあるか」だけを見て、「それが何番目の行か」を知る手段を持ちません。

だとすると恐ろしい予言が立ちます。**入力 $X$ の行を並べ替えても、各トークンが受け取る表現ベクトルは語順をどう変えても1ビットも変わらないはず**です。コードで確定させましょう。第3章で実装した attention をそのまま使います。

```python
# 第7巻 第7章 7.1: attention は集合演算であることの実験
# 6 トークンの文に見立てた X の行を並べ替え、出力が「同じ並べ替え」を受けるだけか確かめる
n, d_model = 6, 16
X = rng.normal(0, 1, size=(n, d_model))
out = self_attention(X, W_Q, W_K, W_V)                  # (n, d_model)

perm = np.array([3, 0, 5, 1, 4, 2])                     # 語順を滅茶苦茶に
out_shuffled = self_attention(X[perm], W_Q, W_K, W_V)

# 出力は「同じ並べ替えを受けた元の出力」と完全に一致する
assert np.allclose(out_shuffled, out[perm], atol=1e-12)
# 各トークンが受け取る表現は語順に 1 ビットも依存しない
for new_pos, old_pos in enumerate(perm):
    assert np.allclose(out_shuffled[new_pos], out[old_pos], atol=1e-12)
```

(全文と動作確認は `code/ch07/permutation_invariance.py`。`python3` で全 assert 通過。`self_attention` は第3章の `attention` を Q=K=V=X で呼ぶラッパ)

1つ目の assert は、シャッフル後の出力が「元の出力に同じ並べ替えを施したもの」と `atol=1e-12`——事実上ビット単位——で一致することを言います。2つ目はもっと生々しい確認です。元の文で3番目だったトークンはシャッフル後の文では先頭にいますが、**受け取る表現ベクトルは完全に同一**。「犬」が文のどこに置かれても、attention を通った後の「犬」の中身は同じなのです。

この性質を並べ替え同変性(permutation equivariance)と呼びます。出力の行の**順番**は入力に追随しますが、各トークンに紐づく中身は不変です。要するに、**attention にとって入力は系列(sequence)ではなく集合(set)**です。私たちは6個のベクトルを「文」のつもりで渡していましたが、attention が見ていたのは順序情報のないベクトルの袋でした。

これは attention だけの病気ではありません。multi-head(第4章)は同じ attention の束、FFN は "position-wise"(第6章)で各行に独立に同じ MLP を適用するだけ、residual も layer norm(第2章)も行単位の操作です。**ここまで組み上げた部品のすべてが語順を見ていません**。このまま第8巻で組み立てれば、「太郎が犬を追う」と「犬が太郎を追う」に対して各単語はまったく同じ表現を受け取ります。

第6巻終章の問いその1の答えは「**消える**」でした。RNN が構造の副産物として無料でくれていた語順を、Transformer は自前で調達しなければなりません。構造が運ばないなら、残る手段は1つ、**データに混ぜて入り口から入れる**こと。それが Section 3.5 の仕事です。

## 7.2 原文逐行: sin / cos の式(3) — 波長が幾何級数で並ぶ設計

原文を読みます。

> *"Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension $d_{model}$ as the embeddings, so that the two can be summed."*
> — Vaswani et al., "Attention Is All You Need", Section 3.5
>
> 訳: 我々のモデルは再帰も畳み込みも含まないので、モデルが系列の順序を利用できるようにするには、トークンの相対的または絶対的な位置に関する何らかの情報を注入しなければならない。そのために、encoder と decoder のスタックの最下部で、入力埋め込みに「positional encoding」を加算する。positional encoding は埋め込みと同じ次元 $d_{model}$ を持ち、それゆえ両者は足し合わせることができる。

"contains no recurrence and no convolution" は、7.1 でコードが突きつけた事実そのものです。"inject"(注入する)——構造で順序を持つのではなく、**入力データに位置情報を混ぜ込む**という方針宣言。"at the bottoms of the stacks" は、注入はスタックの入り口で**一度だけ**(N = 6 段の各段で足し直すのではない)。"the same dimension $d_{model}$ ... so that the two can be summed"——埋め込みと**足し算**するために形を揃えた。連結ではなく加算を選んだこの一言の意味は、7.4 で考えます。

では何を注入するのか。続く式が主役です。

> $$PE_{(pos, 2i)} = \sin\!\left(pos / 10000^{2i/d_{model}}\right)$$
> $$PE_{(pos, 2i+1)} = \cos\!\left(pos / 10000^{2i/d_{model}}\right)$$
> — 同論文, Section 3.5(原文では式番号なし。本シリーズでは式(3)と呼ぶ)
>
> 訳: 位置 $pos$、次元 $2i$ 番目の成分は $\sin(pos/10000^{2i/d_{model}})$、次元 $2i+1$ 番目の成分は $\cos(pos/10000^{2i/d_{model}})$。

記号を確定します。$PE$ は `(max_len, d_model)` の行列、$pos$ が行番号(位置: 0, 1, 2, ...)、$i$ が**列のペア番号**です。$d_{model}$ 本の列は2本ずつ組になり、偶数列 $2i$ が sin、奇数列 $2i+1$ が cos を担当、1つのペアは共通の角度 $pos \cdot \omega_i$ を使います。ここで

$$\omega_i = \frac{1}{10000^{2i/d_{model}}}$$

と置けば、ペア $i$ の中身は $(\sin(pos\,\omega_i),\ \cos(pos\,\omega_i))$ です。

よく効く見立てがあります。**$PE$ の各ペアは、次元ごとに違う速さで回る時計の針**です。$pos$ を時刻と思ってください。ペア $i$ は角速度 $\omega_i$ で回る針で、sin と cos はその針の先端の座標。$i = 0$ の針は最速で、1トークン進むごとに1ラジアン、約6.3トークンで一周します。$i$ が1つ増えるごとに針は $10000^{2/d_{model}}$ 倍(d_model = 512 なら約1.037倍)ずつ遅くなり、最後のペアは一周に約6万トークンかかります。d_model = 512 での実際の値:

| ペア $i$ | 角速度 $\omega_i$ | 波長(一周にかかるトークン数 $2\pi/\omega_i$) |
|---:|---:|---:|
| 0 | 1 | 6.28 |
| 1 | 0.9647 | 6.51 |
| 64 | 0.1 | 62.8 |
| 128 | 0.01 | 628 |
| 255 | 0.000104 | 約 60,600 |

原文は続けます。"The wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$."(波長は $2\pi$ から $10000 \cdot 2\pi$ までの等比数列をなす)。表の右列がそれで、隣のペアと比べ波長が常に一定倍率で伸びます。$10000^{2i/d_{model}}$ という一見ぎょっとする分母は、「波長を等比で並べる」と決めた瞬間に出てくる素直な式です。

なぜ針が1本ではいけないのか。最速の針だけだと6.3トークンごとに同じ値に戻り、位置 0 と位置 6 の区別がつきません(周期性の衝突)。最遅の針だけだと隣の位置との差が 0.0001 ラジアン——微小すぎて信号になりません。時計が時針・分針・秒針を併用するのと同じで、**速い針が近距離の分解能を、遅い針が遠距離の一意性を受け持ち、全部の針を同時に読めば広い範囲の位置が事実上一意に決まる**のです。

## 7.3 なぜこの形か: 相対位置が線形変換で表せる

位置を表すだけなら $PE_{pos} = pos$(行番号をそのまま書く)でもよさそうです。なぜ三角関数なのか。論文は理由を一文で述べます。

> *"We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$."*
> — Vaswani et al., "Attention Is All You Need", Section 3.5
>
> 訳: この関数を選んだのは、固定されたずれ幅 $k$ に対して $PE_{pos+k}$ が $PE_{pos}$ の線形関数で表せるため、モデルが相対位置に基づく attention を容易に学習できるだろうと我々が仮説を立てたからである。

なぜ「線形関数で表せる」と嬉しいのか。attention が位置情報を使う経路は $W^Q$, $W^K$ による射影(第4章)——**線形変換**です。言語では「2つ前の単語を見る」のような相対位置の関係が重要ですが、「$k$ 個ずれた位置の $PE$」が行列1つの掛け算で作れる形なら、その種の関係は線形変換しか持たないモデルにとって学びやすい。注意したいのは "hypothesized"(仮説を立てた)で、原文自身がこれを証明済みの定理ではなく設計仮説として提示しています。私たちも厳密な証明はせず、主張そのもの——$PE_{pos+k}$ が $PE_{pos}$ の線形関数であること——を確かめ算で検証します。

道具は高校数学の加法定理です。

$$\sin(\alpha + \beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta, \qquad \cos(\alpha + \beta) = \cos\alpha\cos\beta - \sin\alpha\sin\beta$$

$\alpha = pos\,\omega_i$、$\beta = k\,\omega_i$ と置いてペア $i$ の2成分に当てはめると、

$$\begin{pmatrix} \sin((pos+k)\,\omega_i) \\ \cos((pos+k)\,\omega_i) \end{pmatrix} = \begin{pmatrix} \cos k\omega_i & \sin k\omega_i \\ -\sin k\omega_i & \cos k\omega_i \end{pmatrix} \begin{pmatrix} \sin(pos\,\omega_i) \\ \cos(pos\,\omega_i) \end{pmatrix}$$

右辺の $2 \times 2$ 行列は回転行列(第1巻5章)——針の先端を固定角 $k\omega_i$ だけ回す変換です。決定的なのは、**この行列の成分に $pos$ が入っていない**こと。$k$ と $\omega_i$ だけから作れます。針の言葉では当然で、「時刻を $k$ 進める」とは、いまが何時であろうと各針を同じ角度だけ回すことだからです。ペアごとのこの $2 \times 2$ を対角線上に $d_{model}/2$ 個並べたブロック対角行列を $M_k$ `(d_model, d_model)` とすれば、$PE_{pos+k} = M_k \, PE_{pos}$——線形関数で表せました。

数値例で確かめ算をします。$d_{model} = 4$(ペアは2つ、$\omega_0 = 1$、$\omega_1 = 0.01$)、$pos = 3$、$k = 2$。実装(7.4)で値を出すと、

$$PE_3 = (0.1411,\ -0.9900,\ 0.0300,\ 0.9996), \qquad PE_5 = (-0.9589,\ 0.2837,\ 0.0500,\ 0.9988)$$

$k = 2$ **だけ**から作った変換行列は

$$M_2 = \begin{pmatrix} -0.4161 & 0.9093 & 0 & 0 \\ -0.9093 & -0.4161 & 0 & 0 \\ 0 & 0 & 0.9998 & 0.0200 \\ 0 & 0 & -0.0200 & 0.9998 \end{pmatrix}$$

で、$M_2 \, PE_3 = (-0.9589,\ 0.2837,\ 0.0500,\ 0.9988)$。$PE_5$ と4桁すべて一致します。$M_2$ を作るとき $pos = 3$ は一度も使っていないので、同じ $M_2$ が $PE_0 \to PE_2$ にも $PE_{10} \to PE_{12}$ にも通用するはずです。この全数チェックは、7.4 のテスト4(`offset_matrix`)が引き受けます。

図7.1 は、この回転を最速ペア $i = 0$ で描いたものです。

図7.1: ペア $i=0$ の $(\cos, \sin)$ を平面に打った図。位置 0, 1, 2, ... の点が単位円の上を**等しい角度間隔**で進んでいく。「位置を $k$ 進める」がどの点から出発しても同じ回転になることが、円周上の等間隔さとして見える。(描画コードは `code/ch07` 参照)

## 7.4 実装とテスト — ヒートマップ、そして「足す」という選択

実装します。式(3)は2行のブロードキャスト(第1巻6章)で書けます。このファイルは**第8巻が import して Transformer の組み立てに使う基盤部品**です。

```python
# 第7巻 第7章 7.4: Positional Encoding(論文 3.5、式(3))
def positional_encoding(max_len, d_model):
    """PE[pos, 2i] = sin(pos / 10000^(2i/d_model)), PE[pos, 2i+1] = cos(...)
    返り値: (max_len, d_model)。学習パラメータを持たない決め打ちの定数行列。"""
    if d_model % 2 != 0:
        raise ValueError("d_model は偶数を仮定する(sin/cos を列のペアで使うため)")
    pos = np.arange(max_len, dtype=np.float64)[:, np.newaxis]      # (max_len, 1)
    i = np.arange(d_model // 2, dtype=np.float64)[np.newaxis, :]   # (1, d_model/2)
    angle = pos / 10000.0 ** (2.0 * i / d_model)                   # (max_len, d_model/2)
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(angle)   # 偶数列 2i
    pe[:, 1::2] = np.cos(angle)   # 奇数列 2i+1
    return pe
```

(全文と `__main__` のスポットチェック assert は `code/ch07/positional_encoding.py`。`python3` で全 assert 通過)

中身は3行です。`pos` `(max_len, 1)` と `i` `(1, d_model/2)` の割り算がブロードキャストで `angle` `(max_len, d_model/2)` になり、その sin を偶数列 `0::2`、cos を奇数列 `1::2` に流し込みます。乱数も学習パラメータも登場しない、完全に決め打ちの定数行列です。

数値でも見ます。`positional_encoding(6, 8)` の全成分($d_{model} = 8$ なのでペアは4つ、$\omega$ は左から 1, 0.1, 0.01, 0.001)。

| $pos$ | sin$_0$ | cos$_0$ | sin$_1$ | cos$_1$ | sin$_2$ | cos$_2$ | sin$_3$ | cos$_3$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| 1 | 0.841 | 0.540 | 0.100 | 0.995 | 0.010 | 1.000 | 0.001 | 1.000 |
| 2 | 0.909 | −0.416 | 0.199 | 0.980 | 0.020 | 1.000 | 0.002 | 1.000 |
| 3 | 0.141 | −0.990 | 0.296 | 0.955 | 0.030 | 1.000 | 0.003 | 1.000 |
| 4 | −0.757 | −0.654 | 0.389 | 0.921 | 0.040 | 0.999 | 0.004 | 1.000 |
| 5 | −0.959 | 0.284 | 0.479 | 0.878 | 0.050 | 0.999 | 0.005 | 1.000 |

左のペアほど行ごとに激しく動き、右のペアはほぼ $(0, 1)$ から動きません。そして**どの2つの行も互いに異なります**。各行が、その位置の指紋になっているのです。

この行列を大きなサイズで画像にしたのが、論文の解説記事でよく見る縞模様のヒートマップです。

図7.2: positional encoding のヒートマップ(横軸が次元、縦軸が位置、色が値)。左端の列は縦方向に細かく振動する縞、右へ進むほど縞の間隔が広がり、右端はほぼ一様な色になる。縞の太さ = 波長が左から右へ等比数列で太っていく様子が、そのまま模様として見える。上の数値表は、この図の左上の小さな切れ端を数字で読んだものです。(描画コードは `code/ch07` 参照)

テストを書きます。7.2 と 7.3 で読んだ主張を、そのまま assert に翻訳したものです。

```python
# 第7巻 第7章 7.4: positional_encoding のテスト(抜粋)
# テスト2: 式(3)との全数一致(定義どおりの素朴な二重ループで検算)
pe_naive = np.zeros((max_len, d_model))
for pos in range(max_len):
    for idx in range(d_model):
        i = idx // 2
        angle = pos / 10000.0 ** (2.0 * i / d_model)
        pe_naive[pos, idx] = np.sin(angle) if idx % 2 == 0 else np.cos(angle)
assert np.allclose(pe, pe_naive, atol=1e-12)

# テスト4: 相対位置の線形性。M_k は k だけから作れて pos に依らない
def offset_matrix(k, d_model):
    """位置を k だけ進める線形変換。sin/cos のペアごとに 2×2 回転をブロック対角に並べる"""
    M = np.zeros((d_model, d_model))
    for i in range(d_model // 2):
        omega = 1.0 / 10000.0 ** (2.0 * i / d_model)
        c, s = np.cos(k * omega), np.sin(k * omega)
        M[2*i:2*i+2, 2*i:2*i+2] = np.array([[c, s], [-s, c]])
    return M
for k in [1, 3, 10]:
    M_k = offset_matrix(k, d_model)          # k から一度だけ作る(pos を知らない)
    shifted = pe[:max_len - k].dot(M_k.T)
    assert np.allclose(pe[k:], shifted, atol=1e-9)   # 同じ M_k が全位置で通用

# テスト6: PE を足すと並べ替え不変性が壊れる(7.1 の問題の解決の検収)
out = self_attention(X, W_Q, W_K, W_V)
assert np.allclose(self_attention(X[perm], W_Q, W_K, W_V), out[perm], atol=1e-12)  # PE なし
pe_n = positional_encoding(n, d_model)
out_pe = self_attention(X + pe_n, W_Q, W_K, W_V)
out_pe_shuffled = self_attention(X[perm] + pe_n, W_Q, W_K, W_V)
assert not np.allclose(out_pe_shuffled, out_pe[perm], atol=1e-6)   # PE あり: 反転する
```

(全文——テスト1の決定性、テスト3の値域 $[-1,1]$、テスト5の内積、テスト7の奇数 $d_{model}$ 拒否を含む——は `code/ch07/test_positional_encoding.py`。`python3 test_positional_encoding.py` で全 assert 通過)

読みどころ3つ。テスト2は、ベクトル化した実装を**定義どおりの素朴な二重ループ**で全数検算します(速い実装を遅い実装で検算する、第1巻4章以来の流儀)。テスト4は 7.3 の確かめ算の全数版で、`offset_matrix(k, d_model)` が $pos$ を一切受け取らずに作られるのに全位置で通用することが、forループの構造に表れています。

そしてテスト6が、この章のクライマックスです。`X + pe_n`——埋め込みに PE を**足してから** self-attention に通すと、7.1 で成立していた assert が `assert not np.allclose(...)` に**反転**します。トークンを並べ替えても PE は位置 0, 1, 2, ... の順で足されるので、同じトークンでも置かれた位置によって違う入力ベクトルになり、出力が変わる。モデルが語順を受け取った瞬間です。第8巻は、このテストが通っていることを前提に `positional_encoding` を import します。

最後に、7.2 で保留した「足す」という選択を。位置情報を入れるなら、埋め込みの隣に**連結**(concatenation)して `(n, d_model + d_pe)` にする手もありました。加算を選ぶ利点は $d_{model}$ が変わらないこと。連結すると後段のすべての $W$ が太り、residual の「全部 $d_{model}$ で揃える」規約(第2章)も崩れます。「内容と位置を同じ512次元に混ぜて潰し合わないのか」という不安には、埋め込み自体が学習パラメータである(第6章)ことが答えの半分です——モデルは PE と干渉しにくい置き場所に意味を配置するよう学習できます。残り半分は「実験的にこれでうまくいった」という事実で、原文もこれ以上は論じていません。

## 7.5 learned positional embedding との比較

論文 3.5 の最終段落は、設計の比較です。

> *"We also experimented with using learned positional embeddings instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than those encountered during training."*
> — Vaswani et al., "Attention Is All You Need", Section 3.5
>
> 訳: 代わりに学習型の positional embedding を使う実験も行ったが、両者はほぼ同一の結果を生んだ(Table 3 の行 (E) を参照)。我々が正弦波版を選んだのは、訓練中に遭遇したより長い系列長へモデルが外挿できる可能性があるからである。

learned positional embedding(学習型位置埋め込み)とは、`(max_len, d_model)` の行列を**パラメータとして**持ち、位置 $pos$ の行をそのまま使う方式です。仕掛けは第6巻3章の埋め込み行列と同じで、対象が「単語」から「位置番号」に変わっただけ——位置の表現を設計せず、モデル自身に考えさせます。

結果は "nearly identical"。7.2〜7.3 で読み解いた幾何級数も加法定理も、**性能のためには必須ではなかった**のです。少し拍子抜けしますが誠実な報告です。そのうえで著者らが sin/cos を選んだ決め手が外挿(extrapolation)でした。learned 方式は訓練時に確保した `max_len` 行しか持たず、それより長い入力には対応する行がそもそも**存在しません**。一方 sin/cos は任意の $pos$ に値を返し、テスト3で確認したとおり位置がどれだけ先でも全成分が $[-1, 1]$ に収まります。"may allow"(可能性がある)という控えめな助動詞も読みどころで、これは保証ではなく期待です。

その後の歴史では learned 方式を採るモデルも、相対位置や回転を使う別方式も登場し、決定版は今も1つに定まっていません。私たちは深追いしません(最短測地線)。この論文の時点での結論——「どちらでもほぼ同じ。なら、外挿に期待が持てる方を」——が読めれば、Section 3.5 は完読です。

## まとめ

- **attention は集合演算**: 入力の行を並べ替えると、出力は同じ並べ替えを受けるだけで各トークンの中身は1ビットも変わらない(第3章の `attention` をコードで確認)。multi-head・FFN・residual・layer norm もすべて行単位で、語順はモデルのどこにも届いていなかった——第6巻終章の問いその1の答えは「語順は消える」
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

**問2** プロットのコード(`positional_encoding(200, 64)` を作り、$p \in \{0, 20, 50\}$ ごとに `[pe[p] @ pe[p+d] for d in deltas]` を描く)は `code/ch07` 参照。手元の数値はこうなります($p$ によらず同一)。

| $\Delta$ | 0 | 1 | 2 | 3 | 5 | 10 | 20 | 50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $PE_p \cdot PE_{p+\Delta}$ | 32.000 | 30.917 | 28.304 | 25.587 | 23.504 | 21.052 | 18.879 | 15.674 |

見えること: (1) 3本の曲線は**完全に重なります**。加法定理から $PE_p \cdot PE_{p+\Delta} = \sum_i \cos(\Delta\,\omega_i)$ となり、$p$ が式から消えるためです(テスト5の中身)。(2) 最大値は $\Delta = 0$ の 32 で、これは $\sin^2 + \cos^2 = 1$ がペア数 $d_{model}/2$ 個ぶん積み上がった値です。(3) $\Delta$ が大きくなるほど内積はおおむね減ります(速い針の周期性で細かい凸凹は出ます)。第1巻2章の「内積 = 類似度」がここでも効いていて、**PE のベクトル空間では「位置が近い」が「似ている」として表現されている**——attention は内積しか測れませんが、その内積に位置の近さがちゃんと写るのです。

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

---

> [目次](../TOC.md) ・ [← 前の章](06-ffn-embeddings.md) ・ [次の章 →](08-why-self-attention.md)
