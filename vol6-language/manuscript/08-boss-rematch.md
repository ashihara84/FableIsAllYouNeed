# 終章 ラスボス再戦 — Introduction が読める

> [目次](../TOC.md) ・ [← 前の章](07-attention.md) ・ [次の章 →](../../vol7-attention/manuscript/00-prologue.md)

第7章までで、この巻の登り道はすべて踏み終えました。言語モデルという問題設定(第1章)、BPE によるトークン化(第2章)、埋め込み(第3章)、n-gram の壁(第4章)、RNN の2つの痛み(第5章)、seq2seq のボトルネック(第6章)、そして attention という発想の転換(第7章)。

序章で、私たちはひとつの約束をしました。この巻を終えたとき、論文の Introduction と Background が**全文読める**——数式ではなく、「なぜ著者たちは attention だけにしたのか」という**動機そのもの**が読める——ようになっている、という約束です。いまから、それを確かめます。

やり方は毎巻同じです。**巻頭に掲げたものとまったく同じ引用を、もう一度掲げ、今度は1文ずつ読んでいきます。** ただし今回読むのは式ではなく、英語の散文です。そして読むための道具は、定義でも定理でもなく、**この巻であなたが実際に体験した、訓練の遅さと、モデルの物忘れ**です。

## 8.1 再読の儀式 — 全部、自分の痛みとして読める

序章 0.2 に掲げた3つの引用を、一言一句変えずに再掲します。

> *"This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples."*
> — Vaswani et al., "Attention Is All You Need", Section 1 Introduction
>
> *"In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions."*
> — 同論文, Section 2 Background
>
> *"Sentences were encoded using byte-pair encoding, which has a shared source-target vocabulary of about 37000 tokens."*
> — 同論文, Section 5.1

序章でこれを見たとき、"sequential nature" が何の性質なのか、"positions" が何の位置なのか、見当もつきませんでした。いまはどうでしょうか。1つ目から読んでいきます。

### 1つ目 — あなたの訓練が遅かった理由が書いてある

> *"This inherently sequential nature precludes parallelization within training examples..."*
>
> 訳: この本質的に逐次的な性質は、訓練事例の内部での並列化を不可能にする。これは、メモリ制約により事例をまたいだバッチ化が制限されるため、系列長が長くなるほど深刻になる。

"This inherently sequential nature" の "This" が指すのは、直前で論文が説明している RNN の計算規則、つまり $h_t$ は $h_{t-1}$ から作る、という規則です。第5章 5.1 で実装した、あの隠れ状態の持ち回りです。

"precludes parallelization within training examples"(訓練事例の**内部での**並列化を不可能にする)。1つの文の中では、$h_t$ は $h_{t-1}$ の完成を待たなければなりません。第5章 5.3 でコードの形で確認したとおり、時間方向の for ループは行列積1発に潰せません。第1巻4章のベンチマークで見た「行列演算なら一瞬」という武器が、系列の方向にだけは使えない——あの噛み合わなさです。

"which becomes critical at longer sequence lengths"(系列長が長くなるほど深刻になる)。第5章の演習で系列長を変えて測った訓練時間、あのグラフの右肩上がりが、この1節の中身です。

最後の "as memory constraints limit batching across examples" は、序章では意味不明だった部分です。「文の内部で並列化できないなら、せめて文を**たくさん束ねて**(バッチ化して)事例間で並列化すればいい」という逃げ道がまずあります。しかし長い系列では1事例あたりのメモリ消費が膨らみ、束ねられる本数が減ります。つまり逃げ道もメモリで頭打ちになる——「事例内で並列化できない」という欠陥が、長い系列では**ごまかしきれなくなる**、と言っているのです。これは、第5章であなたが流した訓練ログの正確な要約として読めます。

### 2つ目 — あなたのモデルが忘れた理由が書いてある

> *"...the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions..."*
>
> 訳: これらのモデルでは、入力または出力の任意の2位置の信号を関係づけるのに必要な演算数が、位置間の距離とともに増える(ConvS2S では線形に、ByteNet では対数的に)。このため、離れた位置どうしの依存関係を学習することがより難しくなる。

"relate signals from two arbitrary input or output positions"(任意の2位置の信号を関係づける)。文の先頭の単語と末尾の単語を結びつけたい、という話です。RNN では、位置 1 の情報が位置 $t$ に届くには $t$ 回の隠れ状態の更新をくぐり抜けるしかなく、距離の分だけ中継地点が増えます。

そして中継のたびに何が起きたでしょうか。第5章 5.4 で測ったとおり、遠い情報は薄まり、勾配は消えていきました。第5巻6.1で深さ方向に見た勾配消失が、系列方向で再演される——あれです。"This makes it more difficult to learn dependencies between distant positions"(離れた位置どうしの依存関係を学習することがより難しくなる)は、あの勾配ノルムのグラフの言語化です。

ConvS2S や ByteNet は、RNN とは別の路線(畳み込みベース)で同じ問題と戦っていた先行モデルです。彼らでも演算数は距離とともに「線形に」「対数的に」育つ、と論文は言います。個々の中身を知る必要はありません。読み取るべきことは1つ——**「距離とともに育つ」こと自体が敵だ**、と論文は名指ししているのです。

では、距離とともに育たない方法を私たちはもう知っているでしょうか。知っています。第7章で実装した attention は、どの位置からどの位置へも内積1回——**常に1ステップ**——で繋がりました。attention 重みの可視化で、decoder が入力のはるか遠くを直接見にいく様子を、あなたは自分の目で見ています。

### 3つ目 — あなたが実装したものの名前が書いてある

> *"Sentences were encoded using byte-pair encoding, which has a shared source-target vocabulary of about 37000 tokens."*
>
> 訳: 文は byte-pair encoding で符号化され、ソースとターゲットで共有される約37,000トークンの語彙を持つ。

これはもう解説が要りません。byte-pair encoding は第2章でフルスクラッチ実装した、「一番多いペアをくっつける、を繰り返すだけ」のあのアルゴリズムです。

"a shared source-target vocabulary"(ソースとターゲットで共有される語彙)も読めます。翻訳では入力言語と出力言語の2つがありますが、語彙表を別々に持たず、両言語のコーパスを合わせて1つの BPE 語彙を育てた、ということです。"about 37000 tokens" の意味もわかります。第2章の演習で確かめたとおり BPE の語彙サイズはマージ回数で決まる**設計者のつまみ**でした。37000 は、著者たちがそのつまみに選んだ設定値です。

3つの引用が、すべて読めました。1つ目と2つ目は**あなたが自分の手で踏んだ痛み**として、3つ目は**あなたが自分の手で作った道具**として読めました。

ここまで読めると、論文のタイトルの意味が刺さります。第7章 7.4 で私たちは問いました——「attention が本体で、RNN は足枷では?」。Introduction と Background は、まさにこの問いを論文の言葉で組み立て直したものです。逐次性という足枷(1つ目)、距離という足枷(2つ目)。だから RNN を捨てて、足枷のない attention **だけ**にする。*Attention Is All You Need*。序章ではパロディの元ネタすら見えなかったこのタイトルが、いまは**設計判断の宣言文**として読めます。

## 8.2 残る問い — RNN を取り除いた廃墟で

勝利を宣言する前に、正直になるべきことがあります。

第7章で作ったのは **attention 付きの seq2seq** でした。encoder も decoder も RNN のままで、attention はその上に乗った補助装置です。「RNN は足枷では?」と問うところまでは行きましたが、**実際に取り除いてはいません**。論文は、取り除きました。

RNN は足枷でしたが、足枷なりに**仕事**もしていました。机上で取り除いてみると、2つの穴が空きます。この2つの穴を正確に言語化することが、この巻の最後の仕事です。

**問い1: 語順の情報はどうなる?**

RNN は1トークンずつ左から右へ読みました。遅さの原因だったこの逐次性は、同時に**語順をタダで教えてくれる仕組み**でもありました。「犬」を読んでから「噛む」を読むのと、「噛む」を読んでから「犬」を読むのとでは、隠れ状態 $h$ の中身が違います。読む**順番**そのものが、順序の情報をモデルに刻んでいたのです。

一方、attention の計算(第7章 7.2)は、内積でスコアを出し、softmax で重みにし、重み付き和を取る——この3ステップのどこにも「位置」が登場しません。query と key の**中身**だけで重みが決まり、和は足す順番を選びません。つまり入力トークンの並び順をシャッフルしても、attention の出力は本質的に同じものになります。

これは致命的です。「犬が人を噛んだ」と「人が犬を噛んだ」は同じトークンの集合であり、語順を失った attention にはこの2文が**同じ単語の袋**にしか見えません。RNN を取り除くなら、語順の情報を別の方法で入力に**注入し直す**必要があります。——それは何でしょうか。

**問い2: 自分自身への attention とは何か?**

第7章の attention には明確な向きがありました。query は decoder から、key と value は encoder の出力からでした。つまり「**出力側が入力側を**見る」装置でした。

では、RNN を取り除いた encoder の側を考えます。RNN の隠れ状態 $h$ は「ここまでの要約」、つまり**入力文の中の単語どうしの関係を取り込む**仕事をしていました。「それ」が何を指すか、「銀行」が川岸か金融機関か——文脈の取り込みです。RNN がいなくなったら、この仕事は誰がやるのでしょうか。

手元に残る道具は attention 1つだけです。だとすれば答えの候補は1つ——**入力文が、入力文自身を見ることです。** query も key も value もすべて同じ系列から作る attention です。しかしそれは何を計算していることになるのでしょうか。単語が自分の仲間に問い合わせをする——その重み付き和は、何の「要約」になるのでしょうか。

この2つの問いには、それぞれ答えの名前があります。問い1の答えは **Positional Encoding**、問い2の答えは **Self-Attention** です。どちらも論文の Section 3 にあり、第7巻で精読します。

順序に注意してください。私たちは「論文に Positional Encoding が出てくるから学ぶ」のではありません。**RNN を取り除いたら語順が消える、だから位置の情報が要る**——需要が先にあり、論文の部品はその需要への回答として読まれます。この巻の全体が、第7巻のためのこの2つの需要を作るためにありました。

## 8.3 次巻予告 — いよいよ論文精読

巻頭付録の論文読解マップを開いて、第6巻の行に印を付けてください。Introduction、Background、そして 5.1 の BPE です。論文の「動機」のページに、灯りがともりました。

地図を少し引いて眺めると、気づくことがあります。第1巻で $QK^T$ という行列の配管が、第2巻と第3巻で「学習=損失最小化」が、第4巻で softmax と $\sqrt{d_k}$ が、第5巻で residual connection・layer norm・dropout・backprop が、そしてこの巻でトークン化・埋め込み・attention の発想と論文の動機が読めるようになりました。

**もう、新しい数学は出てきません。** 道具はすべて、あなたの手の中にあります。

次の第7巻は、これまでの6冊とは作りが違います。新しい山に登る巻ではなく、**論文そのものをセクション順に、1行ずつ読む**巻です。scaled dot-product attention、multi-head、そして 8.2 で言語化した2つの問いへの答え——Self-Attention と Positional Encoding です。各部品を単体実装し、テストを書き、「論文の式番号 ↔ 自分のコードの行」が1対1で対応する状態まで持っていきます。

6冊かけて、ラスボスの城の地図を作り終えました。次巻、正面から入城します。

---

> [目次](../TOC.md) ・ [← 前の章](07-attention.md) ・ [次の章 →](../../vol7-attention/manuscript/00-prologue.md)
