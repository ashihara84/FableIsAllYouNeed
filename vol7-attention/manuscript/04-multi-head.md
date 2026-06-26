# 第4章 3.2.2 Multi-Head Attention — 視点を増やす

前章で論文の式(1)が完成しました。`attention(Q, K, V, mask)` は単体テストを通過し、手計算の小例とも一致しています。ところが論文の図2は2枚組です。左が前章で作った Scaled Dot-Product Attention、右が **Multi-Head Attention** で、キャプションにはこうあります。

> *"(right) Multi-Head Attention consists of several attention layers running in parallel."*
> — Vaswani et al., "Attention Is All You Need", Figure 2
>
> 訳: (右)Multi-Head Attention は、並列に走る複数の attention 層からなる。

つまり Transformer は式(1)を**1回だけでは使わない**。同じ計算を $h$ 個並べて走らせ、結果を束ねる。図1のアーキテクチャ図の橙色ブロックの正体は、式(1)そのものではなくこの「束」の方でした。

なぜ1回では足りないのか。並べると何が良く、コストは何倍になるのか。この章でセクション 3.2.2 を逐行で読み、実装し、テストします。新しい数学は登場しません。主役は、第1巻6.4で「本格的な出番は第7巻」と予告したまま温存してきた道具——**行列の束**です。

## 4.1 原文逐行 — なぜ1回の attention では足りないか("averaging inhibits this")

セクション 3.2.2 の最初の段落は、仕掛けの説明です。

> *"Instead of performing a single attention function with $d_{model}$-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values $h$ times with different, learned linear projections to $d_k$, $d_k$ and $d_v$ dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding $d_v$-dimensional output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2."*
> — Vaswani et al., "Attention Is All You Need", Section 3.2.2
>
> 訳: $d_{model}$ 次元の key・value・query で単一の attention 関数を実行する代わりに、query・key・value を、それぞれ別々に学習される線形射影で $h$ 回、$d_k$・$d_k$・$d_v$ 次元へ射影する方が有益であることを我々は見出した。射影されたそれぞれの query・key・value に対して attention 関数を並列に実行し、$d_v$ 次元の出力を得る。これらを連結(concatenate)し、もう一度射影したものが最終的な値となる。図2参照。

手順は全部で4つ。(1) 線形射影を $h$ 通り用意し、(2) それぞれの射影先で attention を並列に実行し、(3) 出力を連結して、(4) もう一度射影する。線形射影は第1巻5章、attention は前章の装備ですから、読めない単語はありません。読めないのは**動機**です。論文は次の2文で答えます。この章でいちばん重要な2文です。

> *"Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this."*
> — 同上
>
> 訳: multi-head attention により、モデルは異なる位置にある、異なる表現部分空間の情報に**同時に**注意を向けられる。単一の attention head では、平均化がこれを妨げる。

並列に走る attention の1本1本を、論文は **head**(頭)と呼びます。以降この本でも head と書きます。

"averaging inhibits this"——平均化が「これ」を妨げる。「これ」とは直前の "jointly attend"、**複数の見方を同時に持つこと**です。なぜ平均化がそれを妨げるのか。

前章で確かめたとおり、attention の出力は **value の重み付き平均**です。重みは softmax の出力なので非負で足すと1。つまり1回の attention が query 1本に作れるのは、「どこをどれだけ見るか」という**ただ1つの配分表**と、その配分による**ただ1つの平均**だけです。

小さな例で困る状況を作ります。

> 昨日、太郎が花子に駅で会った。

「会った」の位置から他の単語を見にいくとき、見たいものが少なくとも2系統あります。

- **誰が**会ったのか —— 主語を探す見方。「太郎が」を見たい
- **どこで**会ったのか —— 場所を探す見方。「駅で」を見たい

value を2次元で $\mathbf{v}_{太郎} = (1, 0)$、$\mathbf{v}_{駅} = (0, 1)$ とします(他は省略)。「主語の見方」は配分 $(1, 0)$ で出力 $(1, 0)$、「場所の見方」は配分 $(0, 1)$ で出力 $(0, 1)$。

1回の attention では配分表を1つしか持てず、折衷して $(0.5,\ 0.5)$ とするしかありません。出力は

$$0.5 \cdot (1, 0) + 0.5 \cdot (0, 1) = (0.5,\ 0.5)$$

この $(0.5, 0.5)$ が困りものです。後段の層には「主語は太郎、場所は駅」という2つの答えが**足し合わされて1つに溶けた**ものしか届かない。「value が $(0.5, 0.5)$ である別の単語を1点だけ見た」場合と出力上は区別がつきません。2つの見方は平均を取った瞬間に互いを薄め合う——これが "averaging inhibits this" です。

head が2つあれば話は変わります。head 1 は配分 $(1, 0)$ で $(1, 0)$、head 2 は配分 $(0, 1)$ で $(0, 1)$。手順(3)で2出力を**連結**すれば $(1, 0, 0, 1)$。平均ではなく**並置**なので、両方の答えが別の成分として残ります。どう混ぜるかは最後の射影(手順4)が学習で決めればよい。

「異なる表現部分空間(representation subspaces)」にも触れておきます。head ごとの射影は、$d_{model} = 512$ 次元から $d_k = 64$ 次元を切り出す線形変換(第1巻5章)——512次元のうち**どの側面を残すかを選ぶ変換**です。射影が違えば内積、つまり類似度が変わる。「文法的な働きが似ている」と「意味が似ている」は別の「似ている」であり、別の射影がそれぞれを拾う。head とは、**類似度の測り方そのものを複数持つ**仕掛けでもあるのです。

## 4.2 仕掛け — $d_{model}$ を $h = 8$ 個の頭に分割し、それぞれ別の射影 $W_i^Q, W_i^K, W_i^V$ で別の「見方」を学ぶ

動機が読めたので式に進みます。原文は手順を2行の数式にまとめています。

> $$\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h)\, W^O$$
> $$\mathrm{head}_i = \mathrm{Attention}(Q W_i^Q,\ K W_i^K,\ V W_i^V)$$
> — Vaswani et al., "Attention Is All You Need", Section 3.2.2

2行目は初めて見る式ではありません。第1巻の序章でラスボスの一部として掲げられ、第1巻5.5で「shape だけは読める」ところまで攻略した式です。あのとき私たちは $XW^Q, XW^K, XW^V$ を「**同じ入力を3つの役割に変換する**」操作として読み、$W^Q$ の shape を `(512, 64)` と確かめ、添字 $i$ は「第7巻の精読で扱う」と保留しました。

その予告を回収します。**$i$ は「何番目の見方か」、つまり head の番号です。** 第1巻5.5の「役割への変換」一式($W_i^Q, W_i^K, W_i^V$ の3枚)を、論文は $h$ 組用意する。$i$ 組目の射影で作った $Q, K, V$ に式(1)を適用したものが $\mathrm{head}_i$。役割(query / key / value)× 視点(head $1 \ldots h$)で、射影行列は計 $3h$ 枚。第1巻5.5の読みは「同じ入力を、**視点ごとに別々の**役割に変換する」へと完成します。$W^Q$ が `(512, 512)` ではなく `(512, 64)` だったのは1 head 分の射影だったから——これが6巻越しの種明かしです。

記号の罠を先回りします。この式の $Q, K, V$ は**射影前の入力**で、$QW_i^Q$ と掛けて初めて式(1)に渡ります。射影**済み**の行列を指していた式(1)の $Q, K, V$ とは立場が違う——同じ文字の二重使用です。self-attention なら3つとも同じ行列 $X$(配線は次章 3.2.3 の主題)。混同を避けるため、実装では射影前の入力を `X_q`(query の出どころ)、`X_kv`(key と value の出どころ)と呼び分けます。

shape を確定させます。原文が1文で指定しています。

> *"Where the projections are parameter matrices $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$ and $W^O \in \mathbb{R}^{h d_v \times d_{model}}$."*
> — 同上
>
> 訳: ここで射影は、パラメータ行列 $W_i^Q$ `(d_model, d_k)`、$W_i^K$ `(d_model, d_k)`、$W_i^V$ `(d_model, d_v)`、および $W^O$ `(h·d_v, d_model)` である。

具体的な数字も原文がすぐ後で与えます。

> *"In this work we employ $h = 8$ parallel attention layers, or heads. For each of these we use $d_k = d_v = d_{model}/h = 64$."*
> — 同上
>
> 訳: 本研究では $h = 8$ 本の並列な attention 層、すなわち head を用いる。それぞれについて $d_k = d_v = d_{model}/h = 64$ とする。

$d_k = d_{model} / h$。512次元を8等分して64次元ずつ——「$d_{model}$ を $h$ 個の head に**分割**する」と呼ばれる理由です。入力を $X$ `(n, 512)` として、式全体を shape で読み通します。

| 段階 | 式 | shape の流れ |
|---|---|---|
| 射影($i$ 組目) | $XW_i^Q,\ XW_i^K,\ XW_i^V$ | `(n, 512) @ (512, 64) → (n, 64)` |
| attention | $\mathrm{head}_i = \mathrm{Attention}(\cdots)$ | `(n, 64)`(式(1)は形を $d_v$ 側に揃えて返す) |
| 連結 | $\mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_8)$ | `(n, 64)` × 8 → `(n, 512)` |
| 出力射影 | $\cdots W^O$ | `(n, 512) @ (512, 512) → (n, 512)` |

入口が `(n, 512)`、出口も `(n, 512)`。序章0.3で約束した sub-layer の規格——同じ shape を受け取り同じ shape を返す、residual を足すための設計(第2章)——を、multi-head attention はこの連結と出力射影で満たしています。

$W^O$ の役割も言葉にします。連結しただけの `(n, 512)` は、head 1 の答えが第1〜64列、head 2 が第65〜128列……と**縦割りのまま並んでいるだけ**。4.1の例でいえば「主語の答え」と「場所の答え」が別区画に置いてある状態です。$W^O$ はこの512列を混ぜ合わせて1つの表現に編集する全結合(第1巻6章)で、どう組み合わせるかは学習に委ねられます。

## 4.3 テンソル整形の実務 — (batch, seq, d_model) → (batch, h, seq, d_k) の reshape / transpose(第1巻6.4「行列の束」がついに本番)

式は読めました。次は実装の段取りです。素朴に書くなら $W_1^Q$ から $W_8^Q$ まで `(512, 64)` を8枚持ち、forループで head を1本ずつ計算する。動きますが、第1巻からの問いをここでも立てます。**このループ、行列演算にまとめられないか。** まとめられます。鍵は射影の側と attention の側に1つずつ、計2つ。

なお、見出しの batch 軸は説明の間は省きます。文1本($X$ `(n, 512)`)で進め、結論を先に言えばバッチは**先頭に軸が1本増えるだけ**で、コードは1文字も変わりません。

**鍵その1: 8枚の射影は1枚の大きな射影にまとめられる。** `(512, 64)` を8枚、横に貼り合わせます。

$$W^Q = \big[\, W_1^Q \;\big|\; W_2^Q \;\big|\; \cdots \;\big|\; W_8^Q \,\big] \qquad \texttt{(512, 64)} \times 8 \to \texttt{(512, 512)}$$

この $W^Q$ で一度に射影すると $XW^Q$ は `(n, 512) @ (512, 512) → (n, 512)`。行列積は列ごとに独立でした(第1巻4章)から、結果の**第 $i$ 列ブロック**(第 $64i$ 〜 $64(i+1)-1$ 列)は $X W_i^Q$ と完全に一致します。「8回の射影」は「1回の行列積と列の切り分け」に置き換わる。$K, V$ も同様。実装でパラメータを `(512, 512)` の4枚($W^Q, W^K, W^V, W^O$)だけ持てばよいのは、この貼り合わせのおかげです。

**鍵その2: 切り分けた8冊に attention を一斉に適用できる。** `(n, 512)` を8冊の `(n, 64)` ——第1巻6.4の**行列の束** `(8, n, 64)` ——に切り分けます。NumPy では2段階の整形です。

**1段目は reshape。** `(n, 512) → (n, 8, 64)`。各行の512個を「64個ずつ、8グループ」と区切り直すだけで、数の並び順は変わりません。軸の意味は(単語, head, 次元)になります。

**2段目は transpose(軸の入れ替え)。** `(n, 8, 64) → (8, n, 64)`。1軸目と2軸目を入れ替えて(head, 単語, 次元)に。「最後の2軸が行列、残りは束ね方」という第1巻6.4の読み方に当てはめれば、`(n, 64)` の行列が8冊——head $i$ の $Q$ 行列が $i$ 冊目に入った束の完成です。

誰もが一度はやる失敗を先回りで潰します。**「目盛りの付け替えなら reshape 一発で `(8, n, 64)` にすればいいのでは?」**——これが罠です。単語2個、$d_{model} = 4$、$h = 2$ で見ます。

$$X = \begin{pmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \end{pmatrix}$$

(1行目が単語a、2行目が単語b)。正しい head 1 は「各単語の前半2次元」、列ブロック $\begin{pmatrix} 1 & 2 \\ 5 & 6 \end{pmatrix}$ であるべきです。ところが `X.reshape(2, 2, 2)` は数を並び順に先頭から詰めるので、1冊目は

$$\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$$

単語aの前半と後半が2単語のように同居し、単語bは1冊目から消えます。shape は `(2, 2, 2)` で同じなのでエラーは出ない。**黙って間違う**のが怖いところです。

reshape にできるのは「並び順を保ったまま区切り直す」ことだけで、軸の**意味の順番**を入れ替えるのは transpose の仕事。だから「reshape で head の軸を作り、transpose で前に出す」2段が要ります。逆向き(束→1枚)は transpose してから reshape——行きを逆順にたどります。これが論文の Concat の実装です。

束ができたら attention です。前章の `attention(Q, K, V, mask)` は shape を `(..., n, d_k)` のように**後ろの2軸だけで**書いてありました。`@` も softmax(axis=-1)も、先頭の軸が何本でも「束の中の行列ごと」に働きます。第1巻6.4で「`@` は束の中の行列ごとに掛け算をしてくれるが、この章では使わない」と封印したあの性質を、**ここでついに使います**。$Q, K, V$ の束 `(8, n, 64)` をそのまま渡せば、スコア `(8, n, n)`、重み `(8, n, n)`、出力 `(8, n, 64)`——8冊ぶんの式(1)が、forループなしの1呼び出しで揃います。

全体を1本につなげるとこうなります。

$$\texttt{(n, 512)} \xrightarrow{@\,W^Q} \texttt{(n, 512)} \xrightarrow{\mathrm{reshape}} \texttt{(n, 8, 64)} \xrightarrow{\mathrm{transpose}} \texttt{(8, n, 64)}$$
$$\xrightarrow{\mathrm{attention}} \texttt{(8, n, 64)} \xrightarrow{\mathrm{transpose}} \texttt{(n, 8, 64)} \xrightarrow{\mathrm{reshape}} \texttt{(n, 512)} \xrightarrow{@\,W^O} \texttt{(n, 512)}$$

入口と出口だけ見れば `(n, 512) → (n, 512)` の、ただの sub-layer。内部でだけ一時的に束へ姿を変える——序章0.3で予告した「multi-head の内部だけ4階に整形する」の中身です。

最後に batch 軸を戻します。文が $B$ 本なら入力は `(B, n, 512)`。整形を「前から何軸目」ではなく**後ろから数えて**書いておけば(コードでは `swapaxes(-3, -2)` のように負の軸番号を使う)、各 shape の先頭に $B$ が付くだけで同じコードが動きます。束は `(B, 8, n, 64)`——第1巻6.4が「multi-head attention を実装するとき、4次元のテンソルが現れます」と予告した、まさにその4階テンソルです。6巻前の伏線、回収完了です。

## 4.4 [コード] multi-head attention の単体実装 + テスト

実装します。attention は前章の `code/ch03/attention.py` を import して**そのまま**使います——部品を独立に作り後の章が前の章を積む、この巻の約束(序章0.3)どおりの構えです。核心は `multi_head_attention` 本体で、4.3の流れの図と1行ずつ対応します。

```python
def split_heads(X, h):
    """(seq, d) -> (h, seq, d_k): 1枚の行列を h 冊の「行列の束」に分ける(第1巻6.4)。"""
    d = X.shape[-1]
    assert d % h == 0, "d_model は h で割り切れること(論文: 512 = 8 x 64)"
    d_k = d // h
    X = X.reshape(X.shape[:-1] + (h, d_k))  # (..., seq, d) -> (..., seq, h, d_k)
    return np.swapaxes(X, -3, -2)           # -> (..., h, seq, d_k)

def combine_heads(Y):
    """(h, seq, d_v) -> (seq, h*d_v): 束を1枚に戻す。論文の Concat。split_heads の逆操作。"""
    h, d_v = Y.shape[-3], Y.shape[-1]
    Y = np.swapaxes(Y, -3, -2)              # -> (..., seq, h, d_v)
    return Y.reshape(Y.shape[:-2] + (h * d_v,))

def multi_head_attention(X_q, X_kv, W_q, W_k, W_v, W_o, h, mask=None):
    Q = split_heads(X_q @ W_q, h)   # (n, d_model) -> (h, n, d_k)
    K = split_heads(X_kv @ W_k, h)  # (m, d_model) -> (h, m, d_k)
    V = split_heads(X_kv @ W_v, h)  # (m, d_model) -> (h, m, d_v)
    heads, weights = attention(Q, K, V, mask)  # (h, n, d_v), (h, n, m)  h 冊ぶん一気に
    Y = combine_heads(heads) @ W_o  # (n, h*d_v) @ (h*d_v, d_model) -> (n, d_model)
    return Y, weights
```

全文と動作確認は `code/ch04/multi_head.py`(`python3` で全 assert 通過。`W_q, W_k, W_v, W_o` は h 個の射影を横に並べた `(d_model, d_model)`、第3章の `ch03/attention.py` を import)。本体は空行を除けば**6行**——射影3本、束への分割(同じ行で `split_heads`)、attention 1回、結合と出力射影。`split_heads` と `combine_heads` も「reshape して swapaxes」「swapaxes して reshape」で、4.3の2段とその逆順そのままです。

mask は引数で受け取り attention にそのまま渡します。`(n, m)` の mask は束のスコア `(h, n, m)` へブロードキャストされ、**全 head に同じ禁止**が掛かる。見る場所は head ごとに違ってよいが、見てはいけない場所は全 head 共通——causal mask の用途(次章)を考えればそうあるべきです。返り値には出力と一緒に attention 重み `(h, n, m)` も返します。head ごとの重みは観察(章末演習)と次章以降のテストに使います。

テストは `code/ch04/test_multi_head.py` に8本(`python3` で全 assert 通過)。検査項目は次のとおりです。

- **テスト0**: `split_heads` と `combine_heads` が互いに逆操作。split の結果が**列ブロック切り**($X$ の第 $i$ ブロック列 = head $i$)と一致(鍵その1の検証)
- **テスト1〜2**: 出力と重みの shape。重みが head ごとに確率分布(非負・行和1)
- **テスト3**: mask した位置の重みが**全 head で**厳密に0
- **テスト4**: $h = 1$ のとき前章の単独 attention と一致
- **テスト5**: 束ねた一括計算が head ごとのループ計算と一致(鍵その2の検証)
- **テスト6**: 論文の数字 $d_{model} = 512, h = 8$ で通る。パラメータ数の確認
- **テスト7**: バッチ軸 `(batch, n, d_model)` を付けても系列を1本ずつ処理した結果と一致

この章の核心である**テスト4とテスト5**だけ本文に再掲します。まずテスト4——multi-head が単独 attention の**一般化**であることの確認です。

```python
# === テスト4: h=1 のとき第3章の単独 attention と一致(TOC 指定)==============
Y1, A1 = multi_head_attention(X_q, X_kv, W_q, W_k, W_v, W_o, h=1)
out_single, w_single = attention(X_q @ W_q, X_kv @ W_k, X_kv @ W_v)
assert np.allclose(Y1, out_single @ W_o)        # 出力射影 W^O を除けば単独 attention
assert np.allclose(A1[0], w_single)             # 重みは完全一致
```

$h = 1$ なら「1等分」、つまり何も分割しない。$d_k = d_{model}$ のフル次元で attention が1回走るだけなので、結果は「射影してから前章の `attention` を1回呼び、$W^O$ を1枚かませたもの」と完全に一致します。multi-head は前章を置き換える別物ではなく、前章を $h = 1$ という特殊な場合として含む**一般化**である——2つの章が地続きであることの、いちばん直接的な証明です。

次にテスト5——4.3の整形のトリック全体が、論文の式の**素直な読み下し**と一致することの確認です。

```python
# === テスト5: 束ね計算 = head ごとのループ計算(4.3 の整形の裏取り)===========
d_k = d_model // h
outs = []
for i in range(h):
    sl = slice(i * d_k, (i + 1) * d_k)  # W_i^Q は W_q の列ブロック i
    head_i, w_i = attention(X_q @ W_q[:, sl], X_kv @ W_k[:, sl], X_kv @ W_v[:, sl])
    assert np.allclose(w_i, A[i])       # head i の重みが束ね版の i 冊目と一致
    outs.append(head_i)
Y_loop = np.concatenate(outs, axis=1) @ W_o     # Concat(head_1, ..., head_h) W^O
assert np.allclose(Y_loop, Y)
```

ループ版は論文の2行を上から下へそのまま書いたものです。$W_i^Q$ は $W^Q$ の列ブロック、$\mathrm{head}_i$ は単独 attention の呼び出し、$\mathrm{Concat}$ は `np.concatenate`、最後に $W^O$。これと reshape / swapaxes で束ねた `multi_head_attention` の結果が `np.allclose` で一致する。**整形は速さのための実装上の工夫であって、計算の中身は論文の式と寸分違わない**——「式番号とコードの行が1対1対応する」というこの巻のゴールが、assert の形で確認されます。

```bash
python3 multi_head.py
python3 test_multi_head.py
```

どちらも `ok` が出れば、論文 3.2.2 の全主張があなたの環境で検証されています。なお `test_multi_head.py` のテスト6には `np.errstate` で警告を抑える箇所があります。一部の実行環境(macOS の Accelerate)で、有限の値どうしの大きな行列積に BLAS が誤った警告を出すためで、結果が正しいことは直後の `np.isfinite` の assert で確認しています。

## 4.5 計算コストの確認 — 分割しても総コストはほぼ同じ、という原文の主張を式で

セクション 3.2.2 の最後の1文が残っています。

> *"Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality."*
> — Vaswani et al., "Attention Is All You Need", Section 3.2.2
>
> 訳: 各 head の次元が小さくなっているため、総計算コストはフル次元の単一 head attention と同程度である。

attention を8本に増やしたのにコストは同程度——式で検算します。コストの数え方は第1巻4章と同じ**乗算の回数**。query 側の系列長を $n$、key / value 側を $m$ とし、`(a, b) @ (b, c)` の乗算回数は $a \cdot b \cdot c$ 回と数えます。

**比較対象その1: フル次元の単独 attention。** $d_{model}$ 次元のまま $Q, K, V$ を作り($W^Q$ は `(512, 512)`)、式(1)を1回。

| 計算 | shape | 乗算回数 |
|---|---|---|
| 射影 $XW^Q, XW^K, XW^V$ | `(n, 512) @ (512, 512)` など3本 | $3\, n\, d_{model}^2$ |
| スコア $QK^T$ | `(n, 512) @ (512, m)` | $n\, m\, d_{model}$ |
| 重み付き和(重み行列 $\times\ V$) | `(n, m) @ (m, 512)` | $n\, m\, d_{model}$ |

**比較対象その2: $h = 8$ に分割した multi-head。** head 1本の射影は `(n, 512) @ (512, 64)` で $n \cdot d_{model} \cdot d_k$ 回。役割3つ × head 8本で

$$3h \cdot n\, d_{model}\, d_k = 3\, n\, d_{model} \cdot (h\, d_k) = 3\, n\, d_{model}^2$$

最後の等号で効いたのが**分割の定義 $h \cdot d_k = d_{model}$**。射影のコストはフル次元1本と8分割で**完全に同じ**。スコアも head 1本 $n \cdot m \cdot d_k$、8本で $n m \cdot (h d_k) = n m\, d_{model}$——同じ。重み付き和も同じ。つまり

$$\text{head 1本のコスト} \approx \frac{\text{フル次元のコスト}}{h}$$

を $h$ 本足すから総額が変わらない。512次元の仕事を64次元 × 8人で分担しただけで、総量は増えていない——"due to the reduced dimension of each head" の中身はこの割り算です。

「同程度(similar)」であって「同じ」ではない理由も挙げます。multi-head には出力射影 $W^O$ の $n\, d_{model}^2$ 回が追加で乗る。射影3本に対し1本の上乗せで、定数倍は $3 \to 4$、ざっくり3割増。スコア計算と合わせた全体ではそれより小さな差です。論文の "similar" は正確な筆致でした。

この検算が示すのは、**8つの視点はほぼタダで手に入っている**こと。では何も失っていないのか。失っているものはあります——各 head の内積は64次元で取られるので、1本あたりの類似度の「解像度」は512次元より粗い。「1つの精密な見方」と「8つの粗い見方」の交換です。どちらが得かは理屈では決まらず、原文の "we found it beneficial"(有益であることを**見出した**)は、実験で確かめた上での選択という宣言として読めます。

前章とのつながりをもう1つ。式(1)の $\sqrt{d_k}$ の $d_k$ は、multi-head では**head 1本ぶんの次元 64**。スコアを割るのは $\sqrt{512}$ ではなく $\sqrt{64} = 8$。第4巻7章の「内積の分散は次元に比例する」は head の中の64次元の内積に適用されます(実装では `attention` が `Q.shape[-1]` から自動で読み取る)。

## まとめ

- 1回の attention は query 1本につき**1つの重み配分と1つの平均**しか作れない。複数の見方は平均された瞬間に溶け合う——これが "averaging inhibits this"。multi-head は出力を平均ではなく**連結**で並置し、複数の見方を同時に保持する
- 仕掛けは「**役割 × 視点**」の射影: $\mathrm{head}_i = \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)$。第1巻5.5で保留した添字 $i$ は head の番号であり、$W_i^Q$ `(512, 64)` が1 head 分だったことの種明かし。連結後の $W^O$ `(512, 512)` が8つの答えを編集する
- 実装は「**1回の大きな射影 + 整形**」: 8枚の $W_i^Q$ は1枚の $W^Q$ `(512, 512)` に貼り合わせられ、head への分割は reshape → transpose の2段で**行列の束** `(h, n, d_k)` を作る操作になる(第1巻6.4の本番)。reshape 一発で済まそうとすると黙って間違う
- 前章の `attention` は束をそのまま受け取り、$h$ 冊ぶんの式(1)を一括で計算する。$h = 1$ で単独 attention と厳密に一致し、head ごとのループ計算とも厳密に一致する(どちらも assert で検証済み)
- 分割の定義 $h \cdot d_k = d_{model}$ により、射影もスコアも総乗算回数はフル次元1本と同じ。追加コストは $W^O$ の1本分のみ——**8つの視点はほぼタダ**

**ラスボスとの距離**: Section 3.2.2 を完読しました。図2の右半分と、図1の橙色ブロックの中身が、コードと1対1で読めます。Section 3.2 に残るは 3.2.3——この部品を**どこに3回配線するか**だけです。

## 対応表 — 論文 3.2.2 ↔ コード

| 論文の式・主張 | コード(`code/ch04/`) |
|---|---|
| $\mathrm{head}_i = \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)$ | `multi_head.py` — `multi_head_attention()` の射影3行 + `attention(Q, K, V, mask)`(attention 本体は第3章 `ch03/attention.py`) |
| $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$(head ごとの射影) | `W_q` の列ブロック。一致の検証は `test_multi_head.py` テスト5 |
| $\mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h)$ | `multi_head.py` — `combine_heads()` |
| $\cdots W^O$(出力射影) | `multi_head.py` — `combine_heads(heads) @ W_o` |
| $h = 8$, $d_k = d_v = d_{model}/h = 64$ | `split_heads()` の `d_k = d // h`(割り切れることは assert)。論文の数字での実行は `test_multi_head.py` テスト6 |
| "averaging inhibits this" | 4.1 の数値例(主張の理由は小例が主役、コードは確認に徹する——本巻の方針)。head ごとの重みの観察は `ex_head_maps.py`(演習3) |
| "the total computational cost is similar" | 4.5 の式($3nd_{model}^2$ の一致と $W^O$ の上乗せ)。数値での確認は演習4 |
| $h = 1$ は単独 attention(3.2.1 との整合) | `test_multi_head.py` テスト4 |

## 演習

**問1**(shape 追い)cross-attention を想定して、`X_q` `(10, 512)`、`X_kv` `(12, 512)`、$h = 8$ とします。`multi_head_attention` の内部で現れる次の各値の shape を、実行せずに答えてください。(a) `X_q @ W_q`、(b) `split_heads(X_q @ W_q, 8)`、(c) `split_heads(X_kv @ W_k, 8)`、(d) スコア $QK^T/\sqrt{d_k}$ の束、(e) attention の出力の束、(f) `combine_heads` の出力、(g) 最終出力と重み。

**問2**(整形の罠)$X = \begin{pmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \end{pmatrix}$、$h = 2$ とします。(a) `X.reshape(2, 2, 2)`(正しい1段目)と、それに `swapaxes(-3, -2)` を適用した結果(正しい束)を、数字を書いて答えてください。(b) 誤って `X.reshape(2, 2, 2)` を「そのまま束」とみなした場合の1冊目と、正しい束の1冊目の違いを述べ、なぜ transpose(swapaxes)を省けないのかを1文で説明してください。

**問3**(head ごとの attention マップを数値表で観察)単語4個・$d_{model} = 8$・$h = 2$ の self-attention を乱数の重みで実行し、head 0 と head 1 の attention 重み `(4, 4)` を `np.round(A[i], 2)` の数値表で並べて表示するコードを書いてください。そして、(a) 各行の和、(b) 2つの head の表が同じかどうか、(c) 各 head が「最も強く見ている位置」(各行の argmax)を観察し、言葉にしてください。

**問4**(コストの検算)$n = m = 10$、$d_{model} = 512$ として、(a) フル次元の単独 attention(射影3本 + スコア + 重み付き和)と、(b) $h = 8$ の multi-head($W^O$ 込み)の総乗算回数をそれぞれ計算し、比を求めてください。系列長が $n = m = 1000$ になると比はどう変わるかも計算し、理由を説明してください。

<details>
<summary>略解</summary>

**問1** (a) `(10, 512)`。(b) `(8, 10, 64)`。(c) `(8, 12, 64)`。(d) `(8, 10, 64) @ (8, 64, 12) → (8, 10, 12)`(束の中の行列ごとの積)。(e) `(8, 10, 12) @ (8, 12, 64) → (8, 10, 64)`。(f) `(10, 512)`。(g) 出力 `(10, 512)`(`X_q` と同じ形)、重み `(8, 10, 12)`。重みが正方形でない(10 × 12)ことが、$Q$ と $K, V$ の出どころが別である証拠になります(次章で効きます)。

**問2** (a) `X.reshape(2, 2, 2)` は `[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]`(軸の意味は 単語, head, 次元)。`swapaxes(-3, -2)` 後は `[[[1, 2], [5, 6]], [[3, 4], [7, 8]]]`(head, 単語, 次元)。1冊目 `[[1, 2], [5, 6]]` は「全単語の前半2次元」= 正しい head 1 です。(b) 誤った1冊目は `[[1, 2], [3, 4]]` で、単語aの前半と後半が「2単語」のふりをして同居し、単語bが含まれません。reshape は数の並び順を保ったまま区切り直すだけなので、軸の意味の順番(単語が先か head が先か)を入れ替えるには transpose が必要です。

**問3** 解答例は `code/ch04/ex_head_maps.py` にあります(`python3 ex_head_maps.py` で実行できます)。骨格は次のとおりです。

```python
import numpy as np
from multi_head import multi_head_attention

rng = np.random.default_rng(42)
n, d_model, h = 4, 8, 2
X = rng.standard_normal((n, d_model))
W_q, W_k, W_v, W_o = (rng.standard_normal((d_model, d_model)) for _ in range(4))
Y, A = multi_head_attention(X, X, W_q, W_k, W_v, W_o, h)
for i in range(h):
    print("--- head", i, "---")
    print(np.round(A[i], 2))
```

観察: (a) どの head のどの行も和は 1.00(丸め誤差を除く)——head それぞれが独立の確率分布を持ちます。(b) 2つの表は一致しません。筆者の環境(seed 42)では、head 1 が位置0をほぼ一点読みする行を持つ一方、head 0 は広く薄く配る傾向が出ます。(c) 各行の argmax の並びも head 間で異なります。$W$ はまだ乱数なのに、射影が違うだけで「どこを見るか」はすでに違う——**学習とは、この『違い』をでたらめな違いから意味のある違い(構文を見る head、近傍を見る head……)へ育てることです**(訓練は第8巻)。

**問4** (a) 射影 $3 \times 10 \times 512^2 = 7{,}864{,}320$、スコア $10 \times 10 \times 512 = 51{,}200$、重み付き和 $51{,}200$。計 $7{,}966{,}720$ 回。(b) 射影は同じ $7{,}864{,}320$、スコアと重み付き和も同じ計 $102{,}400$、$W^O$ が $10 \times 512^2 = 2{,}621{,}440$。計 $10{,}588{,}160$ 回。比は約 $1.33$——増分はちょうど射影3本に対する $W^O$ 1本分です。$n = m = 1000$ では、射影系は系列長に比例して千倍になる一方、スコア系は $n m$ で1万倍に効いて支配項が入れ替わり($1000 \times 1000 \times 512 \approx 5.1$ 億 × 2)、比は約 $1.14$ に縮みます。系列が長いほど $W^O$ の上乗せは霞み、"similar" はさらに正確になります(そしてこの $n^2$ の伸びこそ、第8章で読む Table 1 の主役です)。

</details>
