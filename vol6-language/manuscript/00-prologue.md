# 序章 ラスボスとの対面

> [目次](../TOC.md) ・ [← 前の章](../../vol5-backprop/manuscript/07-boss-rematch.md) ・ [次の章 →](01-language-models.md)

## 0.1 ここまでの到達点 — 部品は揃った。だが、何を流す?

棚卸しから始めましょう。第1巻の `X @ W + b`、勾配降下法(第2巻)、「学習 = 損失最小化」という世界観(第3巻)、softmax と cross-entropy と $\sqrt{d_k}$(第4巻)、誤差逆伝播と自作 autograd、residual connection・layer norm・dropout(第5巻)。その第5巻の終章で、論文のアーキテクチャ図(図1)を棚卸しした結果はこうでした。**attention 以外の箱は、全部読める。** 読めずに残ったのは Multi-Head Attention、Positional Encoding、Embedding の3種類だけです。

次は本丸の attention に挑みたくなります。しかし第5巻の最後に、避けて通れない問いに突き当たりました。

ここまでのシリーズで、入力 $X$ はいつも**数値**でした。家賃を予測する物件の面積。XOR の平面上の点。数値だったからこそ `X @ W + b` に流せたし、微分も学習もできました。すべての道具は「入力が数値ベクトル $\mathbf{x}$ である」という前提の上に建っています。

ところが、この論文は翻訳のモデルです。図1の一番下、すべての矢印の出発点に書かれた "Inputs" とは、**文章**です。「私は猫が好きだ」という文は行列ではありません。足せないし、掛けられないし、微分もできません。

**言語は、どうやって $\mathbf{x}$ になるのでしょうか。**

これに答えない限り、揃えた部品には何も流せません。第6巻は、この問いから始まる巻です。

## 0.2 ラスボスの掲示

この巻のラスボスを掲げます。ただし今回は様子が違います。これまでのラスボスは数式でした。式(1)、式(2)、$\sqrt{d_k}$。今回は**数式が1本もありません**。全部、論文の地の文——英語の文章です。

> *"This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples."*
> — Vaswani et al., "Attention Is All You Need", Section 1 Introduction
>
> *"In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions."*
> — 同論文, Section 2 Background
>
> *"Sentences were encoded using byte-pair encoding, which has a shared source-target vocabulary of about 37000 tokens."*
> — 同論文, Section 5.1

訳してみます。

1つ目: 「この本質的に逐次的な性質は、訓練サンプル内での並列化を不可能にする。メモリ制約がサンプル間のバッチ化を制限するため、これは系列長が長くなるほど深刻になる。」

2つ目: 「これらのモデルでは、入力または出力の任意の2つの位置の信号を関係づけるのに必要な演算数が、位置間の距離に応じて増える——ConvS2S では線形に、ByteNet では対数的に。これにより、離れた位置どうしの依存関係を学習することが難しくなる。」

3つ目: 「文は byte-pair encoding で符号化された。これは原言語と目的言語で共有される、約37000トークンの語彙を持つ。」

訳しても、読めないはずです。

「逐次的な性質(sequential nature)」——**何が**逐次的なのか。主語 "This" が指すものを、私たちはまだ知りません。「並列化を不可能にする」——何を並列化したくて、なぜできないのか。2つ目の「位置(positions)」も謎です。何の中の、どこのことか。「離れた位置どうしの依存関係」と言われても、何と何が依存し合うのかがわかりません。3つ目に至っては、"byte-pair encoding" という見たことのない言葉が説明もなしに置かれ、「37000トークン」のトークンとは何か——単語のことなら、なぜ "words" と書かないのか。

読めなくて構いません。**今は読めない。それを確認するためにここに掲げました。**

そして約束します。この巻を終えたとき、あなたは**論文の Introduction と Background を全文読める**ようになっています。

これはこれまでの巻の約束とは質が違います。式が読めるとは「計算が追える」ことでした。Introduction と Background が読めるとは、**この論文がなぜ書かれたのかが読める**ということです。何に困っていて、それまでの手法のどこが行き詰まっていて、だから何を捨てたのか——「なぜ attention **だけ**にしたのか」という、論文の動機そのものです。

## 0.3 この巻でやること

まず「次の単語を当てるゲーム」——**言語モデル**という問題設定を立てます(第1章)。次に、文章を記号の列に切るトークン化と、ラスボス3つ目の正体である BPE(第2章)。トークンをベクトルにする埋め込み(第3章)。ここまでで、冒頭の問い「言語はどうやって $\mathbf{x}$ になるか」に答えが出ます。

後半は、そのベクトル列を処理するモデルの系譜です。数えるだけの n-gram(第4章)、記憶を持ち回る RNN(第5章)、翻訳を可能にした seq2seq(第6章)、seq2seq に付け足された attention(第7章)。

歴史の年表のように見えるかもしれません。しかし**この巻の目的は、歴史の網羅ではありません**。本当の目的は、**Transformer が解決した問題を、自分の手で踏んで痛がる**ことです。

n-gram を実装すると、組合せ爆発でデータが足りなくなる壁に突き当たります。RNN を訓練すると2つの痛みを体感します——1トークンずつしか進めないせいで遅い(並列化できない)、遠くの情報が薄まって届かない(長距離依存が苦手)。seq2seq では3つ目の痛みに出会います——文全体を1本のベクトルに圧縮する無理。

この痛みこそ、さきほどのラスボスの正体です。「逐次的な性質が並列化を不可能にする」は第5章であなたの訓練時間が教えてくれます。「位置間の距離に応じて演算数が増える」も薄まっていく記憶として体感します。Introduction と Background は、痛みを知らない人には他人事の文章ですが、痛みを踏んだ人には**自分の話**として読めるのです。

そして第7章で、attention に出会います。論文の式(1)に出てきた $Q$・$K$・$V$ という3文字に、初めて**言葉としての意味**が与えられるのもこの章です。第1巻では shape として、第4巻では softmax の入力として読んできたあの式が、「どこを見て訳すか」という仕事の記述だったとわかります。そのとき、タイトルの "All You Need" が何に対する宣言なのかが刺さります。

## 0.4 この巻で扱わないこと

NLP の教科書を開いたことがある人は、形態素解析、構文解析、品詞タグ付け、固有表現抽出——といった目次を見たことがあるかもしれません。

**この巻には、どれも出てきません。**

理由はシリーズを通して同じです。Transformer の計算に登場しないからです。Transformer は文を文法的に解析しません。主語がどれで述語がどれか、という情報を誰からも教わりません。文章をトークンに切り、ベクトルにして、あとは学習に任せる——それがこの論文の流儀であり、私たちはその最短経路だけを通ります。

word2vec という有名な手法の詳細も扱いません。「埋め込みは学習で獲得できる」という発想の源流として名前には触れますが(第3章)、学習アルゴリズムの中身には立ち入りません。RNN の改良版である LSTM も、ゲート構造を図で概観するだけで実装はしません(第5章)。動機の理解に必要な分だけ、です。

## 巻頭付録 論文読解マップについて

シリーズ共通の巻頭付録、**論文読解マップ**(`paper-map.md`)を開いてください。第6巻が担当するのは、地図の中の次の行です。

| 論文の箇所 | 内容 | この巻の章 |
|---|---|---|
| 1 Introduction | "sequential nature precludes parallelization"(並列化不能) | 第5章 |
| 2 Background | "grows in the distance between positions"(長距離依存) | 第5・7章 |
| 2 Background | "auto-regressive" | 第6章 |
| 5.1 | byte-pair encoding | 第2章 |
| 3.4 | learned embeddings, $d_{model}$ | 第3章 |
| 3.2.1 | $Q$, $K$, $V$ という言葉の意味 | 第7章 |
| (評価指標) | perplexity(第4巻7章の回収) | 第1章 |

この巻を終えると、論文の**最初の2セクションがまるごと**読める領域に変わります。式ではなく、動機が読めるようになる巻です。

それでは始めましょう。最初の問いはこうです——「彼は毎朝コーヒーを」の次に来る単語を、あなたはなぜ予想できるのでしょうか?

---

> [目次](../TOC.md) ・ [← 前の章](../../vol5-backprop/manuscript/07-boss-rematch.md) ・ [次の章 →](01-language-models.md)
