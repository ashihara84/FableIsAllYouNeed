# 第5章 3.2.3 — attention の3つの使い方

> [目次](../TOC.md) ・ [← 前の章](04-multi-head.md) ・ [次の章 →](06-ffn-embeddings.md)

部品としての attention は、第3章で式(1)の `attention(Q, K, V, mask)` を実装し、第4章で multi-head 化した時点で完成しています。残っているのは**引数に何を渡すか**だけです。$Q$ はどこから来るのか、$K$ と $V$ は、mask は付けるのでしょうか。

第2章の図1を見直してください。橙色の Multi-Head Attention ブロックは**3箇所**——encoder 側に1つ、decoder 側に2つ——にあり、置かれた場所も矢印の刺さり方も違います。論文のセクション 3.2.3 は、この3箇所それぞれに「何を渡すか」を指定する**配線表**です。

この章に新しい計算は出てきません。原文の3つの箇条書きを読み、入力と mask の組合せを確定させ、第3章・第4章の関数を3通りに**呼び分ける**こと——それだけです。この章を終えると、Section 3.2 が最初から最後まで読めます。

## 5.1 原文逐行 — encoder self-attention / decoder masked self-attention / encoder-decoder(cross)attention

セクション 3.2.3 の冒頭は、たった1文です。

> *"The Transformer uses multi-head attention in three different ways:"*
> — Vaswani et al., "Attention Is All You Need", Section 3.2.3
>
> 訳: Transformer は multi-head attention を3つの異なる方法で使う。

続く3つの箇条書きを、原文の順番のまま1つずつ読みます。

### 箇条その1 — encoder-decoder attention

> *"In 'encoder-decoder attention' layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2, 9]."*
> — 同上
>
> 訳: 「encoder-decoder attention」層では、query は直前の decoder 層から来て、記憶側の key と value は encoder の出力から来る。これにより decoder のすべての位置が、入力系列のすべての位置に注意を向けられる。これは [38, 2, 9] のような sequence-to-sequence モデルにおける典型的な encoder-decoder attention 機構を模したものである。

読み取るべきは、**$Q$ と $K, V$ の出どころが分かれている**ことです。query は decoder 側、key と value は encoder 側です。出どころが2系統に分かれるのは、3つのうちこれだけです。"memory keys and values" の "memory" が示すとおり、encoder の出力は decoder から見れば「参照しに行く記憶」、decoder が作りかけの系列が「問い」です。

最後の "This mimics..." の参照番号 [2] は Bahdanau らの attention 付き seq2seq——**第6巻7章で私たちが実装した、まさにあれ**です。確認は 5.3 でやります。

### 箇条その2 — encoder の self-attention

> *"The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder."*
> — 同上
>
> 訳: encoder は self-attention 層を含む。self-attention 層では、key、value、query のすべてが同じ場所——ここでは encoder の直前の層の出力——から来る。encoder の各位置は、直前の層のすべての位置に注意を向けられる。

定義はこの1文に尽きます。**"all of the keys, values and queries come from the same place"**——$Q$ も $K$ も $V$ も全部同じ場所、すなわち encoder の直前の層の出力、処理中の文そのものから来ます。文が、文自身に問いかけるのです。この読解は 5.2 の主役です。

mask の言及がないことにも注意してください。encoder の self-attention では各位置が**全位置**を見ます。過去も未来も隠しません(理由は次の箇条との対比でわかります)。

### 箇条その3 — decoder の masked self-attention

> *"Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections. See Figure 2."*
> — 同上
>
> 訳: 同様に、decoder の self-attention 層では、decoder の各位置が**その位置まで(その位置を含む)**のすべての位置に注意を向けられる。自己回帰性を保つため、decoder 内では左向きの情報の流れを防ぐ必要がある。これを scaled dot-product attention の内部で、softmax への入力のうち不正な接続に対応する値をすべてマスクする(−∞ にする)ことで実装する。図2を参照。

decoder 側にも self-attention があり、1つだけ制限が付きます。"up to and including that position"——見てよいのは**自分自身を含む過去だけ**です。

"prevent leftward information flow" は、系列を左から右へ書く図で読めます。右にあるのは未来のトークンです。その情報が左(過去の位置の表現)へ流れ込むのを防ぐ——**未来からのカンニングの禁止**です。理由は "to preserve the auto-regressive property" です。decoder は生成時、これまでに出したトークンだけを頼りに次の1個を出します(第6巻6.4の自己回帰生成)。ところが訓練時は teacher forcing で正解の系列が丸ごと手元にあるため、何もしなければ位置 $t$ の予測が位置 $t+1$ の正解を見てしまい、訓練と生成が食い違います。だから訓練時にも未来を**構造的に**見えなくしておくのです。

実装方法も原文が指定しています。softmax の入力を $-\infty$ にすれば $e^{-\infty} = 0$ で重みが厳密に0になる——**第3章3.5で実装した causal mask、そのもの**です。図2の "Mask (opt.)" が "opt."(optional)だった理由も確定します。3つの使い方のうち mask を使うのは**この1つだけ**だからです。

### 配線表としてまとめる

指定されていたのは「$Q, K, V$ の出どころ」と「mask の有無」の組合せだけです。表にします。

| 使い方 | 図1での場所 | $Q$ の出どころ | $K, V$ の出どころ | mask |
|---|---|---|---|---|
| encoder self-attention | encoder 内 | 直前の encoder 層の出力 | 同じ(= $Q$ と同一) | なし |
| decoder masked self-attention | decoder 内・下段 | 直前の decoder 層の出力 | 同じ(= $Q$ と同一) | causal mask |
| encoder-decoder(cross)attention | decoder 内・上段 | 直前の decoder 層の出力 | encoder スタックの最終出力 | なし |

原文の箇条書きは cross → encoder self → decoder masked self の順でしたが、以降は self(5.2)→ cross(5.3)の順で進めます。新しい概念は self-attention の方で、cross はすでに第6巻で作ったものだからです。

## 5.2 self-attention とは Q=K=V=自分自身 — 「文が文自身を見る」

第6巻の終章で残した2つの問いの、2つ目がこれです。

**「自分自身への attention とは、いったい何なのか?」**

第6巻7章で attention を初めて導入したとき、$Q$ と $K, V$ は別の場所から来ていました。query は decoder の隠れ状態、key と value は encoder の各位置です。attention とは**2つの系列の間の橋**だ、という印象のまま第6巻を終えた人が多いはずです。

self-attention は、この橋の**両端を同じ系列につなぎます**。原文の定義をそのまま式にすれば、ある層への入力を $X$ `(n, d_model)` として

$$Q = K = V = X$$

です(multi-head 化では、ここからさらに $XW_i^Q, XW_i^K, XW_i^V$ と head ごとの射影がかかりますが——第4章でやったとおり——**出どころが1つ**という点は変わりません)。問いかける側も、問いかけられる側も、答えを差し出す側も、すべて処理中の文自身です。一言でいえば、**文が文自身を見る**のです。

具体例で考えます。

> 「猫が皿に近づいた。**それ**は空腹だった。」

「それ」が指すのは猫か皿でしょうか。「空腹だった」まで読めば猫だとわかります。つまり「それ」の意味は単語単体では決まらず、**文中の他のトークンを参照して初めて**決まります。self-attention はこの参照を計算します。「それ」の位置の query が全位置の key と内積を取り、「猫」との相性が高ければ softmax の重みが「猫」に集まり、出力——「それ」の位置の新しい表現——には「猫」の value が濃く混ざります。1層通るたびに、各トークンの表現が文脈を取り込んで更新されます。

ここで第6巻5章の痛みを思い出してください。「文脈を混ぜる」仕事を、RNN は隠れ状態の持ち回りでやっていました。1トークンずつ順番に読むので並列化できず(痛み1)、遠い情報は持ち回りの間に薄まります(痛み2)。self-attention は同じ仕事を**全位置いっせいの行列演算1回**でやります。隣を見るのも100トークン先を見るのも等しく内積1回、逐次の持ち回りが消えたので並列化を妨げるものもありません(この対比を数字にしたのが Section 4 の表で、第8章で読みます)。

「RNN を取り除いたら、文脈は誰が混ぜるのか」——第6巻終章の問いその2の答えはこうです。**文自身が混ぜる。各位置が文の全位置に問いかけ、内積の重みで全位置の情報を引き寄せる。それが self-attention です。**

decoder 側の self-attention も定義は同じです。処理中の(生成途中の)系列を $X_{dec}$ `(m, d_model)` として $Q = K = V = X_{dec}$ です。違いは causal mask が挟まることだけです。重み行列 `(m, m)` の上三角——位置 $i$ から見た未来 $j > i$ ——が厳密に0になります。「文が文自身を見る。ただし自分から過去だけ」が decoder 版です。

shape の目印を1つ挙げます。self-attention の重み行列は $Q$ 側も $K$ 側も同じ系列なので**必ず正方形**です(`(n, n)` または `(m, m)`)。

## 5.3 cross-attention は第6巻7章の attention 付き seq2seq と同型 — Q だけ decoder から

残る encoder-decoder attention(以下、通称どおり **cross-attention**)は、確認だけで終わります。**もう作ったことがある**からです。

第6巻7章で実装した attention 付き seq2seq では、decoder の各ステップで、

1. 今の decoder の状態(query)と、encoder の各位置(key)の内積で相性を測り、
2. softmax で重みにして、
3. encoder の各位置の情報(value)の重み付き和を取る。

原文の箇条その1と並べると、"the queries come from the previous decoder layer" が1に、"the memory keys and values come from the output of the encoder" が1と3に、"attend over all positions in the input sequence" が2と3に対応する——**一語一語、第6巻7章の実装と対応します。** 原文自身が "This mimics the typical encoder-decoder attention mechanisms..." と種明かしをしているとおりです。

違いは1つだけです。第6巻では query が **RNN の隠れ状態**でしたが、Transformer では RNN を取り除いたので query は**直前の decoder 層の出力**(masked self-attention と residual を通ってきた表現)になります。橋の構造はそのまま、両端を支える土台が RNN から attention の積み重ねに置き換わった——「attention だけでいい」というタイトルの主張は、配線レベルではこの置き換えのことです。

つまり cross-attention の指定は、こう要約できます。

$$Q = X_{dec} \quad (m, d_{model}), \qquad K = V = Z_{enc} \quad (n, d_{model})$$

$X_{dec}$ は直前の decoder 層の出力、$Z_{enc}$ は encoder スタックの**最終**出力です。$Z_{enc}$ は一度計算したら固定で、decoder の全層(N=6 のすべて)の cross-attention に同じものが刺さります。図1で encoder の天辺から出た矢印が decoder へ横に渡るとき、矢印は2本に分かれて attention ブロックの $K$ と $V$ に刺さり、$Q$ だけは decoder 側の下から来る——第2章で保留した図1の最後の謎が解けました。

mask がないことも確認します。入力文は翻訳を始める前から丸ごと手元にあるので、decoder がその全位置を見ることは何のカンニングでもありません。隠すべき未来は decoder 自身の出力系列の中にしかなく、それは1つ手前の masked self-attention がすでに処理しています。

shape の目印: cross-attention の重み行列は `(m, n)`——decoder の長さ × encoder の長さの**長方形**です。正方形なら self、長方形なら cross です。重み行列の形だけで使い方を判別できます。

## 5.4 [コード] 3つの使い方を呼び分けるテスト

配線表をコードにします。要点は、3つの使い方が**同じ関数の呼び分けにすぎない**ことです。第3章の `attention`(と第4章の `multi_head_attention`)に、5.1の表どおりの引数を渡す wrapper を3つ書き、入力と mask の組合せが正しいことを assert で確かめます。

テストの設計は3点です。

- **(a) self は出どころが1つ**: 同じ配列を $Q, K, V$ に3回渡した結果と完全一致。重み行列が正方形で、mask なしなら全成分が正
- **(b) masked self は未来が厳密に0**: 重み行列の上三角が0。さらに**自己回帰性そのもの**——未来のトークンをどう書き換えても、過去の位置の出力が1ビットも変わらない
- **(c) cross は $Q$ だけ decoder**: encoder と decoder の系列長を変え(5トークンと3トークン)、重み行列が `(decoder長, encoder長)` の長方形になる。出力が encoder 出力($V$)の重み付き和になる

核心は3つの wrapper です。配線表がそのまま引数リストになっています。

```python
def causal_mask(n):
    """位置 i から見てよいのは j <= i(下三角が True)。第3章 3.5 の causal mask"""
    return np.tril(np.ones((n, n), dtype=bool))

def encoder_self_attention(enc_x):
    """(a) Q = K = V = encoder の前層出力。mask なし(全位置が全位置を見る)"""
    return attention(enc_x, enc_x, enc_x, mask=None)

def decoder_masked_self_attention(dec_x):
    """(b) Q = K = V = decoder の前層出力。causal mask で未来(j > i)を遮断"""
    m = dec_x.shape[0]
    return attention(dec_x, dec_x, dec_x, mask=causal_mask(m))

def cross_attention(dec_x, enc_out):
    """(c) Q だけ decoder 側。K = V = encoder スタックの最終出力。mask なし"""
    return attention(dec_x, enc_out, enc_out, mask=None)
```

テスト(b)の核心 assert は、未来のトークンを乱暴に改変しても過去の出力が完全一致することの確認です。

```python
# 自己回帰性: 位置 t の出力は、t より後の入力をどう変えても変わらない
dec_x2 = dec_x.copy()
dec_x2[-1] += 100.0                                # 最後のトークンだけ大きく変える
out_b2, _ = decoder_masked_self_attention(dec_x2)
assert np.allclose(out_b[:-1], out_b2[:-1])        # 過去の位置の出力は不変
assert not np.allclose(out_b[-1], out_b2[-1])      # 当の位置だけ変わる
```

全文(self/masked self/cross の wrapper、各テスト、multi-head 版の呼び分け確認、import フォールバックを含む)と動作確認は `code/ch05/three_attentions.py` です(`python3 three_attentions.py` で全 assert 通過)。

3つの wrapper の本体はそれぞれ実質1行です。`encoder_self_attention` は同じ配列を3回渡すだけ、`decoder_masked_self_attention` はそれに causal mask を足すだけ、`cross_attention` は第1引数と第2引数の出どころが違うだけです。**5.1の配線表が、そのまま関数の引数リストになっている**ことを確かめてください。

テスト(b)の後半が、この章でいちばん大事な assert です。最後のトークン `dec_x[-1]` に 100 を足す乱暴な改変をしても、それより前の位置の出力は**完全に**一致します。重みが0であるだけでなく、未来の情報が過去の出力に届く経路そのものが存在しない——"prevent leftward information flow" を数値で確認したことになります。

テスト(c)では encoder を5トークン、decoder を3トークンとわざと長さを変えてあります。self なら重み行列は必ず正方形になるので、`(3, 5)` という長方形が通ることが「$Q$ だけが decoder から来ている」ことの動かぬ証拠です。

multi-head 版のテストは、第4章の部品でも呼び分けの規約が同じであることの確認です。出どころを決めるのは射影**前**の入力($X_q$ と $X_{kv}$)であり、head への分割や射影は出どころを変えません。causal mask は全 head に同じものがかかります。

## まとめ

- 論文 3.2.3 は新しい計算を導入しない。第3章・第4章で作った同じ attention 関数の、**引数と mask の組合せを3通りに指定する配線表**である
- **self-attention とは $Q = K = V =$ 自分自身**。文が文自身を見て、各位置の表現が文脈を取り込んで更新される。RNN が逐次の持ち回りでやっていた「文脈を混ぜる」仕事を、全位置並列・距離によらず一様な行列演算で置き換える——第6巻終章の問いその2「自分自身への attention とは?」の答え
- decoder の self-attention には causal mask が付く。未来の重みを厳密に0にして自己回帰性を守る(訓練時の teacher forcing と生成時の条件を一致させる)。3つのうち mask を使うのはこれだけ
- cross-attention は第6巻7章の attention 付き seq2seq と同型。$Q$ だけが decoder 側、$K, V$ は encoder スタックの最終出力。query の土台が RNN の隠れ状態から attention 層の出力に置き換わった点だけが違う
- 重み行列の shape が見分けの目印: self は正方形 `(n, n)`、cross は長方形 `(m, n)`

**ラスボスとの距離**: Section 3.2 はこの章で完読です。図1の3つの attention ブロックすべてに $Q, K, V$ の出どころと mask を書き込めるようになりました。残るは 3.3(FFN)・3.4(埋め込み)・3.5(positional encoding)——次章へ進みます。

## 演習

**問1**(図1への書き込み — この章の総仕上げ)図1(第2章で白紙再現したアーキテクチャ図)を印刷するか描き写し、3つの Multi-Head Attention ブロックそれぞれについて、$Q$、$K$、$V$ の3本の矢印が**どこから**来るかを図中に書き込んでください。mask の有無も添えてください。書き込み終えたら、5.1の配線表と突き合わせて検算してください。

**問2**(mask の有無の理由)次の3つに答えてください。(1) decoder の self-attention に causal mask が必要なのはなぜか。(2) encoder の self-attention に mask が不要なのはなぜか。(3) cross-attention で decoder が encoder 出力の「未来側」(入力文の後ろの方)を見ても自己回帰性が壊れないのはなぜか。

**問3**(shape を予測してから実行)5.4のコードで `n_enc = 7, n_dec = 4` に書き換えたとき、(a) encoder self-attention、(b) decoder masked self-attention、(c) cross-attention の重み行列の shape はそれぞれどうなるか、**実行する前に**紙の上で答えてください。そのうえで書き換えて実行し、(b) で重みが0になる成分の個数も数えて照合してください。

<details>
<summary>略解</summary>

**問1** 書き込む内容は5.1の配線表のとおりです。encoder 内のブロック: $Q, K, V$ の3本とも直前の encoder 層(最下層なら埋め込み + positional encoding)から、mask なし。decoder 下段: 3本とも直前の decoder 層から、causal mask あり。decoder 上段: $K, V$ の2本は encoder スタックの天辺(最終出力)から横に渡ってきた矢印、$Q$ の1本だけ真下(masked self-attention + residual の出力)から、mask なし。encoder から渡る矢印が**2本に分かれて**刺さっているのは、$K$ と $V$ の2役を担うためです。

**問2** (1) 対象が decoder 自身の出力系列で、生成時には位置 $t$ より先がまだ存在しないからです。訓練時は teacher forcing で全正解が手元にあるため、mask がなければ位置 $t$ の予測が位置 $t+1$ 以降の正解を参照してしまい、訓練と生成で条件が食い違います。(2) encoder の対象は**入力文**で、処理開始の時点で全体が確定しています。「まだ存在しないもの」を見る危険がないので、隠す理由がありません。(3) cross-attention で見るのは入力文の表現であり、その「後ろの方」は時間的な未来ではなく最初から与えられている情報です。自己回帰性が守るべきは decoder 自身の出力系列だけで、それは1つ手前の masked self-attention がすでに遮断しています。

**問3** (a) `(7, 7)`、(b) `(4, 4)`、(c) `(4, 7)`。self は出どころが1つなので正方形、cross は (decoder長, encoder長) の長方形です。(b) で0になるのは上三角(対角線を除く)の成分なので $4 \times 3 / 2 = 6$ 個。実行して `np.sum(w_b == 0.0)` が 6 を返すことを確認できれば一致です。

</details>

---

> [目次](../TOC.md) ・ [← 前の章](04-multi-head.md) ・ [次の章 →](06-ffn-embeddings.md)
