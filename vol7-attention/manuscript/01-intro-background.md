# 第1章 Abstract・Introduction・Background を読む

1〜6巻で揃えた道具はすべて手元にあり、論文読解マップで「まだ」が残るのは Section 3.2 の組み立てと Section 3.5 だけです。いよいよ論文の1ページ目を開きます。

ただしこの章には実装がありません。序章0.2で決めた精読の作法「原文 → 逐行読解 → 単体実装 → テスト → 対応表」のうち、Abstract・Introduction・Background には実装すべき部品がまだ登場しないからです。作法がフル稼働するのは第2章からです。

ではなぜ1章を割くのか。この巻で実装する Section 3 の全部品、その**設計意図のすべてが最初の2ページに予告されています**。何を捨て、なぜ捨て、代償をどこで払うのか。それを先に読み取っておけば、第2章以降の各部品が「天下り」ではなく「予告の回収」として読めます。つまりこの章の仕事は地図づくりで、最初の2ページの各文に「この巻の第何章で実装するか」という行き先を書き込んでいきます。

## 1.1 Abstract 逐行: "dispensing with recurrence and convolutions entirely" — 第6巻の問いへの宣戦布告として読む

Abstract をきちんと逐行で読むのは、シリーズを通してこれが初めてです。第6巻のラスボスは Introduction と Background の文でしたから、Abstract はずっと素通りしてきました。論文の顔である7つの文を、1文ずつ読んでいきます。

> *"The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder."*
> — Vaswani et al., "Attention Is All You Need", Abstract
>
> 訳: 支配的な系列変換モデルは、エンコーダとデコーダを含む、複雑な再帰型または畳み込み型のニューラルネットワークに基づいている。

第1文は現状報告です。「系列変換(sequence transduction)」は系列を入れて系列を出す問題——第6巻6章の翻訳の問題設定そのもの。「エンコーダとデコーダを含む」は入力を読む係と出力を書く係の分業(第6巻6.1)。「再帰型(recurrent)」は隠れ状態を持ち回って1トークンずつ読む RNN(第6巻5章)。すべて自分の手で実装したものの名前です。

1つだけ踏んでいないのが「畳み込み型(convolutional)」です。これは意図的に通らなかった道で、扱いは 1.3 で決めます。

もう1つ、さりげない単語 "complex"——複雑な、に注目してください。論文は冒頭から既存モデルを「複雑」と形容します。この形容詞は2文あとの対比の前フリです。

> *"The best performing models also connect the encoder and decoder through an attention mechanism."*
> — 同論文, Abstract
>
> 訳: 最高性能のモデルはさらに、エンコーダとデコーダを attention 機構で接続している。

第2文は、第6巻7章で実装した attention 付き seq2seq の話です。固定長ボトルネック(第6巻6.3)に苦しむ encoder-decoder に attention を付け足すと長い入力に強くなった——あの体験が、2017年時点の「最高性能のモデル」の標準装備だったと言っています。ここまでの2文で、論文は第6巻で登った階段をそのままなぞっています。

そして第3文。この論文でいちばん有名な一文です。

> *"We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."*
> — 同論文, Abstract
>
> 訳: 我々は Transformer という新しい単純なネットワークアーキテクチャを提案する。これは attention 機構のみに基づき、再帰と畳み込みを完全に排除したものである。

第6巻7.4で、私たちは「attention が本体で、RNN は足枷では?」という素朴な疑いに行き着きました。attention 付き seq2seq の中で一番遅くて一番弱いのが RNN の部分だったからです。

この第3文はその問いへの宣戦布告です。"dispensing with recurrence and convolutions **entirely**"——「RNN を改良する」でも「依存を減らす」でもなく、第6巻で3つの痛み(並列化不能・長距離依存・固定長ボトルネック)の発生源として特定した再帰を、まるごと捨てると宣言しています。第1文の "complex" に対しここでは "simple"——部品を捨てたからこそ単純になった、という対比もきれいに決まっています。

ただし威勢のいい宣言には必ず請求書が付きます。RNN を捨てると困ることを第6巻終章で2つ挙げました。1トークンずつ読む仕組みを捨てたら**語順の情報はどこへ行くのか**、そして encoder-decoder 間ではなく**自分自身への attention とは何なのか**。この請求書の支払いが、それぞれ第7章(Positional Encoding)と第5章(self-attention)です。Abstract はまだ何も答えず、宣言だけして先へ進みます。

> *"Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train."*
> — 同論文, Abstract
>
> 訳: 2つの機械翻訳タスクでの実験により、これらのモデルが品質で上回りつつ、より並列化可能で、訓練に要する時間が大幅に少ないことが示された。

第4文の売り文句は品質・並列性・訓練時間の3つ。注目は後ろの2つです。"more parallelizable" は、第6巻5.3で訓練時間として体感した痛み1(並列化不能)への返答です。$h_t$ が $h_{t-1}$ を待つ仕組みを捨てれば待ち行列が消え、並列化できれば訓練も速い。3つの売り文句のうち2つが「速さ」というのは、この論文の動機が品質だけではなかった証拠です。

> *"Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU."*
> — 同論文, Abstract
>
> 訳: 我々のモデルは WMT 2014 英独翻訳タスクで 28.4 BLEU を達成し、アンサンブルを含む既存の最良結果を 2 BLEU 以上上回った。

> *"On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature."*
> — 同論文, Abstract
>
> 訳: WMT 2014 英仏翻訳タスクでは、8基の GPU で3.5日訓練したのち、単一モデルとして史上最高の 41.8 BLEU を達成した。これは文献中の最良モデルの訓練コストのごく一部である。

第5文・第6文は実験結果の数字です。ここは保留します。BLEU が何をどう測るのか、28.4 がどれくらい偉いのか——**評価指標の詳細は第8巻**で扱います。いま読み取るべきは数字の中身ではなく文の構造です。「品質で上回った(2 BLEU 以上)」と「訓練コストはごく一部(8 GPU で3.5日)」が第4文の裏付けとして並ぶ。捨てたのに勝った、しかも安く勝った、という主張です。

> *"We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data."*
> — 同論文, Abstract
>
> 訳: 訓練データが大規模な場合と限られた場合の両方で英語の構文解析への適用に成功したことにより、Transformer が他のタスクにもよく一般化することを示す。

第7文は「翻訳以外でも動く」という一般性の主張です。構文解析(constituency parsing)の中身には立ち入りません(第8巻で概観だけします)。「他のタスクにもよく一般化する」——2017年の著者たちの想像を、その後の現実(GPT も BERT も、今日あなたが会話した AI も)は遥かに超えていきました。

以上7文。Abstract の構造はこうです。**現状(1〜2文)→ 宣言(3文)→ 売り文句(4文)→ 証拠(5〜6文)→ 一般性(7文)**。3文目の宣言は、第6巻終章の問いから見れば宣戦布告であり、同時に2枚の請求書(語順・self-attention)を切った瞬間でもありました。

## 1.2 Introduction / Background: 第6巻の体感と突き合わせる高速再読

Introduction と Background は、第6巻終章で一度全文読んでいます。だからここでは全文をなぞり直さず、**鍵になる文だけを引いて、第6巻のどの体感と対応するかを固定する**高速再読をします。初読の解説が必要な人は第6巻終章8.1へ。

Introduction の核心は、この2文です。

> *"Recurrent models typically factor computation along the symbol positions of the input and output sequences."*
> — 同論文, Section 1 Introduction
>
> 訳: 再帰型モデルは通常、入力および出力系列の記号位置に沿って計算を分解する。

> *"This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples."*
> — 同論文, Section 1 Introduction
>
> 訳: この本質的に逐次的な性質は、訓練サンプル内での並列化を不可能にする。メモリ制約がサンプル間のバッチ化を制限するため、これは系列長が長くなるほど深刻になる。

第6巻の巻頭では "This" が何を指すか分かりませんでした。いまは指せます。1つ目の文の「位置に沿って計算を分解する」、つまり $h_t$ を $h_{t-1}$ から順に作るあの forループです(第6巻5.1)。各ステップが前を待つため系列方向のループは展開できません。第1巻4章で行列積が並列計算と相性抜群なことを実測した私たちにとって、「並列化を不可能にする」がどれほど致命的な悪口かは、訓練の待ち時間(第6巻5.3)が教えてくれた通りです。**痛み1、確認済み。**

次に、Introduction は attention を持ち上げます。

> *"Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences."*
> — 同論文, Section 1 Introduction
>
> 訳: attention 機構は、さまざまなタスクにおいて強力な系列モデリング・系列変換モデルに不可欠な要素となっており、入力または出力系列内の距離に関係なく依存関係をモデル化することを可能にする。

"without regard to their distance"——距離に関係なく。第6巻7章で attention 付き seq2seq を実装したとき、decoder は入力の全位置を**等しく1ステップで**見渡せました。RNN の記憶のように遠い情報が薄まることもなく(痛み2への返答)、文全体を1本のベクトルに圧縮する必要もない(痛み3への返答)。固定長ボトルネックは、実はこの論文の前に attention がすでに解いていた——論文はそれを前提に書かれています。

> *"In all but a few cases, however, such attention mechanisms are used in conjunction with a recurrent network."*
> — 同論文, Section 1 Introduction
>
> 訳: しかしごく少数の例外を除き、こうした attention 機構は再帰型ネットワークと組み合わせて使われている。

"however"——この逆接が論文全体の蝶番です。距離に関係なく依存関係を結べる attention が、よりによって逐次性という病気を持つ RNN に同居している。第6巻7.4の私たちの違和感(「attention が本体で、RNN は足枷では?」)と一字一句同じ違和感を、著者たちはここで表明しています。だから提案はこうなります。

> *"In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output."*
> — 同論文, Section 1 Introduction
>
> 訳: 本研究では Transformer を提案する。これは再帰を避け、代わりに attention 機構だけに頼って、入力と出力の間の大域的な依存関係を取り出すモデルアーキテクチャである。

Abstract 第3文の言い換えですが、"to draw global dependencies between input and output" という目的語が付きました。attention は飾りでも補助でもなく、依存関係を取り出す**唯一の**仕組みに昇格しています。

Background に進みます。前半は畳み込み系の関連研究で、これは 1.3 に回します。ここで拾うべきは、この巻全体の主役の初登場シーンです。

> *"Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence."*
> — 同論文, Section 2 Background
>
> 訳: self-attention(intra-attention とも呼ばれる)は、系列の表現を計算するために、単一の系列内の異なる位置どうしを関係づける attention 機構である。

第6巻7章の attention は decoder から encoder へ、**2つの**系列の間に架かる橋でした。ここで定義される self-attention は "a single sequence"、**単一の**系列の中で位置どうしを関係づけると言っています。文が文自身を見るとはどういうことか。第6巻終章の問いその2が、ここで論文の用語を得ました。答え(Q・K・V がすべて自分自身になる種明かし)は第5章で実装とともに確認します。

ここまでを表に固定しておきましょう。第6巻の3つの痛みと論文の文の対応表です。

| 第6巻で踏んだ痛み | 体感した場所 | 論文の対応箇所 |
|---|---|---|
| 痛み1: 並列化不能 | 第6巻5.3(訓練時間) | Introduction: "precludes parallelization within training examples" |
| 痛み2: 長距離依存 | 第6巻5.4(薄まる記憶) | Introduction: "without regard to their distance" / Background: "grows in the distance between positions" |
| 痛み3: 固定長ボトルネック | 第6巻6.3(圧縮の無理) | Introduction の attention の段落(attention がすでに解決済み、という前提) |

Introduction と Background は、痛みを知らない人には他人事の文章で、痛みを踏んだ人には自分の話として読める——第6巻序章の予告通りでした。再読はここまでです。

## 1.3 関連研究の固有名詞(ByteNet, ConvS2S 等)の扱い: 深追いしない宣言(最短測地線)

Background の前半には、見慣れない固有名詞が並んでいます。

> *"The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU, ByteNet and ConvS2S, all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions."*
> — 同論文, Section 2 Background
>
> 訳: 逐次計算を減らすという目標は、Extended Neural GPU・ByteNet・ConvS2S の基盤でもある。これらはいずれも畳み込みニューラルネットワークを基本部品として使い、すべての入力・出力位置について隠れ表現を並列に計算する。

Extended Neural GPU、ByteNet、ConvS2S。いずれも RNN の逐次性を畳み込み(convolution)で迂回しようとした、Transformer の競争相手たちです。

先に宣言します。**この3つを深追いしません。** 畳み込みニューラルネットワークの仕組みは、第1巻序文から一貫して通ってきた最短測地線の上にないからです。Transformer は畳み込みを使いません(Abstract 第3文で "dispensing with ... convolutions entirely")。勝者が捨てた部品の内部構造は、論文を読むためにも実装するためにも必要ではありません。

ただし固有名詞は読み飛ばしても、**この段落の論理の骨格**は持ち帰る必要があります。論文が自分の強みを主張する、次の対比だからです。

> *"In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions."*
> — 同論文, Section 2 Background
>
> 訳: これらのモデルでは、入力または出力の任意の2つの位置の信号を関係づけるのに必要な演算数が、位置間の距離に応じて増える——ConvS2S では線形に、ByteNet では対数的に。これにより、離れた位置どうしの依存関係を学習することが難しくなる。

骨格はこうです。畳み込みは「窓」で近所だけをまとめて見る演算です。1層では窓の幅ぶんの距離しか結べないので、離れた2位置を関係づけるには層を重ねて信号をバケツリレーする必要があり、必要な段数は距離とともに伸びます——素朴に重ねれば距離に比例して(線形)、窓を飛び飛びに広げる工夫をしても距離の対数で(これが ByteNet 側の意味ですが工夫の中身は深追いしません)。RNN の「1トークンずつ」よりはるかにマシでも、**遠い2点を結ぶコストが距離の関数として伸びる**ことに変わりはありません。

> *"In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2."*
> — 同論文, Section 2 Background
>
> 訳: Transformer ではこれが定数回の操作にまで削減される。ただし attention で重み付けた位置を平均することで実効的な解像度が下がるという代償があり、この効果には 3.2 節で述べる Multi-Head Attention で対抗する。

これが対比の決着です。self-attention では、どの2つの位置も**1回の attention で直接つながります**。第6巻7章の可視化を思い出してください。attention の重み行列は全位置ペアの類似度を一度に計算していました。隣の単語も100語先の単語も同じ1ステップ——距離 $n$ に対して操作数 $O(1)$ です。距離に応じて伸びる($O(n)$ や $O(\log n)$)か、距離によらず一定($O(1)$)か。**この対比の骨格こそ、固有名詞の森から持ち帰るべき唯一の戦利品です。**

そしてこの文には請求書も同封されています。"albeit"(〜という代償はあるが)以下です。全位置を重み付き平均でまとめると情報が混ざってぼやける——その対策が Multi-Head Attention だと、論文はすでにここで予告しています。この予告の回収が本巻の第4章です。また $O(1)$ という主張は Section 4 の Table 1 で計算量の比較表として数字になり、私たちは第8章でそれを検算します。Background の1段落に、この巻の後半の予告が2つも埋まっていたわけです。

なお、ここで安くなったものとまだ安くなっていないものを区別しておきます。$O(1)$ になったのは「遠い2点を**結ぶ**コスト」であって計算の総量ではありません。全位置ペアを一度に見るとは、ペアの数だけ計算するということでもあります。この代金の話も第8章で精算します。

## まとめ

- Abstract は「現状 → 宣言 → 売り文句 → 証拠 → 一般性」の7文構成。第3文 "dispensing with recurrence and convolutions entirely" は、第6巻7.4の問い「attention が本体で、RNN は足枷では?」への宣戦布告として読める
- ただし宣言には請求書が2枚付いてくる: 語順の情報はどこへ(→第7章)、自分自身への attention とは(→第5章)。BLEU の数字の評価は第8巻まで保留
- Introduction / Background の鍵文は、第6巻の3つの痛み(並列化不能・長距離依存・固定長ボトルネック)と1対1で突き合わせられる。"however" の逆接(attention が RNN に同居している)が論文全体の蝶番
- ByteNet・ConvS2S・Extended Neural GPU は深追いしない(最短測地線)。持ち帰るのは骨格のみ: 遠い2点を結ぶ操作数が距離に応じて伸びる(線形/対数)畳み込み系に対し、self-attention は $O(1)$。その代償(平均によるぼやけ)への対策が Multi-Head Attention(→第4章)、計算量の精算は Table 1(→第8章)

**ラスボスとの距離**: Abstract・Section 1・Section 2 を逐行で読了。論文の残りは Section 3 以降——ここから先は、読むたびに実装する領域です。

## 演習

**演習1** Abstract の7文を、原文を見ながら**自分の言葉で**和訳してください。本文の訳をなぞるのではなく、第6巻までの自分の体験(実装した RNN、踏んだ痛み、attention の可視化)を知っている人にしか書けない訳を目指すこと。訳はノートに残しておいてください——**この巻の終章で、もう一度同じ演習をやります**。Section 3 の全部品を実装し終えたあとの自分の訳と、いまの訳を見比べるためです。

<details><summary>略解</summary>

訳例(逐語訳ではなく「自分の言葉」の一例です):

「いま主流の翻訳モデルは、エンコーダ・デコーダ式の RNN か CNN で、強いものは attention を橋渡しに使っている(ここまで第6巻でやった通り)。我々の Transformer は、その橋だけを残して RNN も CNN も全部捨てた。結果は、翻訳品質で勝ち、並列化でも勝ち、訓練時間でも勝ち。英独翻訳 28.4 BLEU でアンサンブル込みの過去最高を 2 以上更新、英仏翻訳は 8 GPU・3.5日という格安の訓練で単一モデル最高の 41.8 BLEU(BLEU の中身は第8巻で精算)。おまけに構文解析でも動いたので、翻訳専用機ではない。」

巻末の再訳で比べたいのは、たとえば "based solely on attention mechanisms" の重みです。いまは宣言として訳すしかありませんが、終章のあなたは、その attention の中身(式(1)、multi-head、3つの使い方)をすべて自分のコードで持っています。

</details>

**演習2** 第6巻で踏んだ3つの痛み(並列化不能・長距離依存・固定長ボトルネック)のそれぞれについて、Introduction または Background から対応する英文を1つずつ抜き出してください。3つのうち1つは、他の2つと「論文での扱われ方」が違います。どれが、どう違うでしょうか。

<details><summary>略解</summary>

並列化不能 → "This inherently sequential nature precludes parallelization within training examples, ..."(Introduction)。長距離依存 → "the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, ..."(Background。Introduction の "without regard to their distance" も可)。固定長ボトルネック → attention を紹介する段落 "Attention mechanisms have become an integral part ..."(Introduction)。

扱いが違うのは固定長ボトルネックです。前の2つは「Transformer がこれから解く問題」として書かれているのに対し、固定長ボトルネックは attention 付き seq2seq(第6巻7章)が**すでに解いた問題**であり、論文はそれを解決済みの前提として書いています。だからこの論文の新規性は「attention を導入したこと」ではなく「attention **以外を捨てた**こと」にあります。

</details>

**演習3** "dispensing with recurrence and convolutions entirely" という宣言によって、逆に**新しく生じる**問題が2つあります(第6巻終章で予告したものです)。それぞれ何か、そして本巻のどの章が答えるかを述べてください。

<details><summary>略解</summary>

(1) 語順の喪失。RNN は1トークンずつ読む仕組みそのものが語順を担っていましたが、attention は全位置を一度に見るため、このままでは単語を並べ替えても結果が変わりません。答えは第7章の Positional Encoding(Section 3.5)。(2) 自分自身への attention とは何か。第6巻7章の attention は encoder と decoder という2つの系列の橋でしたが、RNN を捨てた後、系列の表現そのものを作る仕事も attention が引き受ける必要があります。これが Background で定義された self-attention("relating different positions of a single sequence")で、答えは第5章(Q=K=V=自分自身)。

</details>
