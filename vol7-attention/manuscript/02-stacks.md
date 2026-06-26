# 第2章 3.1 Encoder and Decoder Stacks — 全体の見取り図

ここからは Section 3 Model Architecture——「設計図」に入ります。進み方は序章 0.2 のとおり、「原文 → 逐行読解 → 単体実装 → テスト → 対応表」です。この章が受け持つ Section 3.1 に式は1本もありません。代わりにこの章は巻全体の**地図**になります。第3章以降で作る部品——attention、FFN、埋め込み——が建物のどこに取り付けられるのか、部品より先に骨組みを押さえます。

## 2.1 図1を地図として読む: encoder 側・decoder 側、N=6 の積み重ね

Section 3 は、図1のすぐ手前で、こう始まります。

> *"Most competitive neural sequence transduction models have an encoder-decoder structure. Here, the encoder maps an input sequence of symbol representations (x₁, ..., xₙ) to a sequence of continuous representations z = (z₁, ..., zₙ). Given z, the decoder then generates an output sequence (y₁, ..., yₘ) of symbols one element at a time. At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next."*
> — Vaswani et al., "Attention Is All You Need", Section 3
>
> 訳: 競争力のあるニューラル系列変換モデルの多くは、encoder-decoder 構造を持つ。encoder は記号表現の入力系列 (x₁, ..., xₙ) を連続表現の系列 z = (z₁, ..., zₙ) に写す。z が与えられると、decoder は記号の出力系列 (y₁, ..., yₘ) を1要素ずつ生成する。各ステップにおいてモデルは自己回帰的であり、次の要素を生成するとき、それまでに生成した記号を追加の入力として使う。

この段落は1文残らず既読です。encoder-decoder 構造は第6巻6章の seq2seq そのもの、「1要素ずつ・自己回帰的」は第6巻6.4。目を留めてほしいのは z が**系列**であること——入力の位置ごとに1本ずつ、n 本のベクトルです。文全体を1本に圧縮して破綻した第6巻6章の RNN encoder と違い、1本に潰さない設計が最初から宣言されています。問題設定は seq2seq から変えず、変えるのは中身です。

> *"The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively."*
> — 同論文, Section 3
>
> 訳: Transformer はこの全体構造を踏襲し、encoder と decoder の双方に、積み重ねた self-attention と、位置ごとの全結合層を用いる。それぞれ図1の左半分と右半分に示す。

RNN の文字がありません。attention と全結合層を**積むだけ**。第6巻終章の問い(RNN を本当に抜いたらどうなるのか)への回答が、ここから始まります。

論文の図1を手元に開いてください。ここからは「眺める絵」ではなく「地図」として読みます。見取り図を描いておきます。

```
                                  出力確率
                                     ↑
                             linear → softmax
                                     ↑
  ┌─[ Add&Norm ]←┐        ┌─[ Add&Norm ]←──┐
  │     FFN     ─┘        │      FFN      ─┘
  │                  z    │ [ Add&Norm ]←───┐
  │ [ Add&Norm ]←┐  ───→ │ cross-attention ─┘
  │ self-attention─┘      │ [ Add&Norm ]←───┐
  │                       │ masked self-attn─┘
  └─────────── ×6        └─────────────── ×6
        ↑                          ↑
  埋め込み + 位置符号化      埋め込み + 位置符号化
        ↑                          ↑
     入力の文             出力の文(1つ右にずらす)
```

図2.1: 図1の見取り図。左の塔が encoder、右の塔が decoder。encoder の出力 z は decoder の**全6層**の cross-attention に同じものが配られる。

要点は3つ。**塔は2本**(入力の文は左の足元から、書きかけの出力の文は右の足元から入る)。**どちらも同じ層を N = 6 回積む**(図1の「N×」)——6階建てのビルが2棟です。積めるのは、各層が `(seq_len, d_model)` を受け取り同じ shape を返すから(「形が戻るなら、同じ処理にもう一度流せる」第1巻6.3)。そして3つ目: **2本の塔は似ているが、同じではない。** この仕分けが地図の核心です。

| | encoder(左) | decoder(右) |
|---|---|---|
| 積む回数 | N = 6 | N = 6 |
| 1層の部分層の数 | **2つ**(self-attention、FFN) | **3つ**(masked self-attention、cross-attention、FFN) |
| self-attention | 全位置を見てよい | **mask 付き**(後ろを見ない) |
| 外からの入力 | 入力の文だけ | 出力の文 + **encoder の出力 z** |
| 塔の出口 | z として decoder へ | linear → softmax → 確率 |

対称なのは骨格。非対称なのは decoder の追加装備3つ——mask、cross-attention、出口の softmax(第4巻6章)——で、理由はすべて「decoder は文を1語ずつ**書く係**」だから。書く係は答えを先読みできず、原文を参照する必要があり、最後は次の語の確率を出さねばなりません。原文で確認します。

> *"Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network."*
> — 同論文, Section 3.1
>
> 訳: encoder は N = 6 個の同一の層の積み重ねで構成される。各層は2つの部分層(sub-layer)を持つ。1つ目は multi-head self-attention 機構、2つ目は単純な、位置ごと(position-wise)の全結合フィードフォワードネットワークである。

「層(layer)」がビルの1階分、「部分層(sub-layer)」がその階の設備です。multi-head self-attention は第3〜4章、position-wise FFN は第6章で作ります。今日は「設備2つ」という個数だけで十分です。

> *"Decoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack."*
> — 同論文, Section 3.1
>
> 訳: decoder もまた N = 6 個の同一の層の積み重ねで構成される。encoder の各層が持つ2つの部分層に加えて、decoder は3つ目の部分層を挿入する。この部分層は、encoder stack の出力に対する multi-head attention を実行する。

"also" と "In addition to"——対称性と非対称性は原文の文法にも書き込まれています。3つ目の部分層は 2.3 で読みます。

## 2.2 sub-layer の規約: LayerNorm(x + Sublayer(x)) と「全部 d_model = 512」の理由

encoder の段落は、こう続きます。

> *"We employ a residual connection around each of the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself."*
> — Vaswani et al., "Attention Is All You Need", Section 3.1
>
> 訳: 2つの部分層それぞれの周りに residual connection を施し、そのあとに layer normalization を適用する。すなわち各部分層の出力は LayerNorm(x + Sublayer(x)) である。ここで Sublayer(x) は部分層自身が実装する関数である。

この文は**第5巻6章で読み切った文そのもの**です。「$x\,+$」が residual connection——部分層の勾配がどれだけ痩せても素通りの経路が残る配管(第5巻6.2)。外側の LayerNorm が、residual の和で太る分散を平均0・分散1に整え直す管理人(第5巻6.3)。図1で各部分層の直後に必ず付く「Add & Norm」の箱の中身が、この式です。第5巻との違いはひとつ——あのとき Sublayer の中身は実験用の小さな MLP でしたが、この巻では本物の attention と FFN が入ります。**中身が何であれ、包み方はこの1パターン**。それが「規約」です。

この規約が成立する条件があります。$x + \mathrm{Sublayer}(x)$ という足し算は、行列同士の足し算なので shape が同じときにしか定義されません(第1巻6.2)。$x$ が `(seq_len, 512)` で $\mathrm{Sublayer}(x)$ が `(seq_len, 256)` なら、式はそこで止まる。residual を使うと決めた瞬間、**すべての部分層は「入力と同じ shape を返す」義務を負う**のです。論文はその義務を明文化しています。

> *"To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension d_model = 512."*
> — 同論文, Section 3.1
>
> 訳: これらの residual connection を成立させるため、モデル中のすべての部分層は、埋め込み層とともに、次元 d_model = 512 の出力を生成する。

"To facilitate these residual connections"——理由が文頭に書いてあります。モデル中の 512 という数字は、性能チューニングの結果である以前に、**足し算を成立させるための取り決め**です。FFN が途中で 2048 次元に膨らんでも(第1巻6.3)、部分層の出口は必ず `(seq_len, 512)`。この1点を守るから Add & Norm の包装紙は全部品で使い回せて、層は何階でも積めます。$d_{model}$ は全館共通の廊下幅の名前です。

なお論文の完全形では各部分層の出力に dropout が掛かります($\mathrm{LayerNorm}(x + \mathrm{Dropout}(\mathrm{Sublayer}(x)))$——第5巻6.4で読んだ Section 5.4 の規定)。dropout は訓練時にだけ働く道具なので、forward の骨組みを作るこの巻では省き、訓練を扱う第8巻で取り付けます。

## 2.3 decoder の3つ目の入力: encoder の出力がどこに刺さるか

decoder の段落の残りです。

> *"Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i."*
> — 同論文, Section 3.1
>
> 訳: encoder と同様に、各部分層の周りに residual connection を施し、そのあとに layer normalization を適用する。さらに decoder stack の self-attention 部分層には、各位置が後続の位置を参照できないようにする変更を加える。このマスキングは、出力埋め込みが1位置ずらされている事実と組み合わさって、位置 i の予測が、位置 i より前の既知の出力のみに依存することを保証する。

1文目は、2.2 の規約が decoder の3つの部分層にもそのまま適用されるという確認です。2文目が mask。「後ろを見てはいけない」理由は第6巻6.4で体感済みです——本番では位置 i+1 以降はまだ書かれていないのだから、訓練でそこを見たらカンニングです。「出力埋め込みを1位置ずらす」も同じ場所で出た話。ただし「参照できないようにする変更」が**どんな計算か**——答えは「softmax の前に $-\infty$ を足す」という仕掛けです——は、attention 本体ができてから第3章3.5で実装します。ここでは予告だけ。

そして節題の問い——**encoder の出力 z は、どこに刺さるのか**。図1では、左の塔のてっぺんから出た矢印が、右の塔の**中段の attention の箱**に横から入っています。本書ではこの部分層を cross-attention と呼びます(論文 Section 3.2.3 の "encoder-decoder attention")。図1から読み取れる事実を2つ確定させます。

1. **decoder の全6層に、同じ z が配られる。** encoder の i 階と decoder の i 階が個別につながるのではありません。
2. **z が刺さるのは3つの部分層のうち真ん中の1つだけ。** masked self-attention と FFN は z を見ません。

「z のどの成分を、どの重みで見るのか」は第5章の主役です。そこで、これが第6巻7章の attention 付き seq2seq の attention と**同型**であることを確認します。

## 2.4 [コード] ダミー Sublayer で stack の骨組みを実装する

実装に入ります。ただし attention も FFN もまだありません。部品を待つ代わりに、**外側から作ります**。部分層の中身をいったん全部**恒等写像**(入力をそのまま返す関数)のダミーで埋めるのです。すると、この章で読んだこと——Add & Norm の配置、d_model の規約、N 回の積み重ね、z の刺さる場所——だけが裸で残った骨組みができます。shape の流れを assert で固めておけば、第3章以降は「ダミーを本物に差し替える」作業になり、同じ assert が毎回の回帰テストになります。第8巻はこの骨組みをそのまま組み立てに使います。

shape の規約を先に決めます。**この巻では入力を `(seq_len, d_model)`——1文ぶんの行列——で通します。** バッチの軸を足して「行列の束」(第1巻6.4)にする話は、multi-head の整形と一緒に第4章4.3で。

核心は、部分層を**引数として受け取り** Add & Norm で包む2つの層関数です。

```python
def encoder_layer(x, prm, self_attn, ffn):
    """encoder の1層 = 部分層2つ。各部分層は LayerNorm(x + Sublayer(x)) で包む。"""
    x = layer_norm(x + self_attn(x), prm["gamma1"], prm["beta1"])   # Add & Norm その1
    x = layer_norm(x + ffn(x), prm["gamma2"], prm["beta2"])         # Add & Norm その2
    return x

def decoder_layer(x, memory, prm, self_attn, cross_attn, ffn):
    """decoder の1層 = 部分層3つ。3つ目の入力 memory は2番目の部分層に刺さる。"""
    x = layer_norm(x + self_attn(x), prm["gamma1"], prm["beta1"])           # masked self-attention(第3章)
    x = layer_norm(x + cross_attn(x, memory), prm["gamma2"], prm["beta2"])  # cross-attention(第5章)
    x = layer_norm(x + ffn(x), prm["gamma3"], prm["beta3"])                 # FFN(第6章)
    return x
```

部分層を引数で渡すのは差し替えのためです。第3章で attention ができたら `dummy_sublayer` の代わりに渡す——骨組み本体に手を入れずに、ダミーが本物になっていきます。stack 本体は `for prm in params:` で N=6 層を回すだけ。`layer_norm` は第5巻6章と同一の定義です(あちらは backward 用 cache も返しましたが、勾配は第8巻の領分なので forward だけ)。全文と動作確認は `code/ch02/stack_skeleton.py`(`python3` で全 assert 通過)。

テストの要点を読みます。検証1・2が shape の流れの本体で、入力文を5語、出力文を7語と**わざと違う長さ**にしてあります。同じ長さだと両側の shape を取り違えるバグが偶然通るからです。decoder の出力は `(tgt_len, d_model)`——長さは自分の側の入力で決まり、memory の長さに依存しません。この事実は第5章で cross-attention の shape を読むときに効きます。

検証4は一風変わったテストです。部分層が恒等写像なら $x + \mathrm{Sublayer}(x) = 2x$。そして layer norm は入力の定数倍に不変です(分子の $x - \mu$ も分母の $\sigma$ も同じ倍率で伸びて約分される)。よって1階目の Add & Norm を抜けた時点で各行は平均0・分散1になり、以降の階では何も起きない。**6階建てのビル全体が layer_norm 1回分に潰れます**。骨組みが「正しく何もしない」ことの確認であり、同時にいまのビルは空っぽだという宣告です。

検証5は 2.2 の規約をわざと破るテストです。出力を256次元に削る違反部分層を差し込むと、residual の足し算 `(5, 512) + (5, 256)` が定義できず ValueError。"To facilitate these residual connections" の1文が、エラーとして体感できます。

```
$ python3 stack_skeleton.py
すべての assert を通過しました
```

## 論文の記述 ↔ コードの行 対応表

| 論文 Section 3.1 の記述 | stack_skeleton.py |
|---|---|
| "a stack of N = 6 identical layers" | `encoder_stack` の for ループ(52〜53行)、`N = 6`(75行) |
| "two sub-layers" / "a third sub-layer" | `encoder_layer` の Add & Norm 2回(36〜37行)、`decoder_layer` の3回(44〜46行) |
| "attention over the output of the encoder stack" | `cross_attn(x, memory)`(45行)。中身は第5章 |
| "LayerNorm(x + Sublayer(x))" | 36〜37、44〜46行。`layer_norm` の定義は10〜17行(第5巻6章と同一) |
| "produce outputs of dimension d_model = 512" | 各関数の shape 契約(docstring)と検証1〜2。破ると検証5(102〜110行) |
| "prevent positions from attending to subsequent positions" | **まだ無い**(第3章3.5で mask として実装) |
| 自己回帰・出力埋め込みの1位置ずらし | **まだ無い**(訓練の配線として第8巻) |

「まだ無い」の2行が、地図の上の空白地帯です。空白が**どこにあるか分かっている**ことが、地図を持っているということです。

## まとめ

- Transformer は seq2seq(第6巻6章)の問題設定を引き継ぐ。encoder が入力を**系列のまま** z に写し、decoder が自己回帰で1語ずつ書く。RNN は登場しない
- どちらの塔も同一の層を N = 6 回積む。積めるのは各層が `(seq_len, d_model)` を受け取り同じ shape を返す——形が戻る(第1巻6.3)——から
- 全部分層は $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$ の規約で包まれる(第5巻6章の回収)。**d_model = 512 への統一は residual の足し算を成立させるための取り決め**(第1巻6.2: shape が合わなければ足せない)
- decoder の追加装備は mask(第3章3.5へ)、cross-attention(z の刺さる場所。第5章へ)、出口の linear → softmax(第6章へ)の3つ
- 骨組みは**外側から**作った。恒等写像のダミー部分層で shape の流れを assert で固定し、以降の章はダミーを本物に差し替える

**ラスボスとの距離**: Section 3.1 は全文読了。図1は「眺める絵」から「空白の位置まで分かっている地図」になりました。残る空白は attention の中身——式(1)、次章です。

## 演習

**問1(図1を白紙に再現する)** 論文を閉じて、白紙に図1を描いてください。描き終えたら論文と突き合わせ、抜けた箱と矢印を数えます。完成度より「どこを思い出せなかったか」の記録が大事です。

<details><summary>略解</summary>

チェックリスト: (1) 塔が2本、それぞれに ×N (2) encoder 層は2部分層、decoder 層は masked self-attention → cross-attention → FFN(この順) (3) 全部分層の直後に Add & Norm(2塔1層分で計5箱)と residual の回り込み矢印 (4) encoder 最上階から decoder **全層**の cross-attention への矢印 (5) 両塔の足元に埋め込み + 位置符号化、decoder 側の「1つ右にずらす」 (6) decoder の頭上に linear → softmax。忘れやすいのは (3) の矢印と (5) のずらしです。

</details>

**問2(規約を破る)** 検証5の `bad_sublayer` は ValueError で即死しました。では、出力を `(d_model,)`(文全体を1本に要約)にする部分層ならどうなるでしょうか。予想してから確かめてください。

<details><summary>略解</summary>

エラーになりません。`(5, 512) + (512,)` は第1巻6.2のブロードキャスティング「行列 (N, d) + ベクトル (d,)」に**合致してしまう**ため、足し算も layer_norm も、shape しか見ない検証1〜3も通ります。契約違反には即死するものと静かに通ってしまうものがある——「assert は書いた分しか守ってくれない」ことの実例です。第3章以降の部品では、出力が2次元の `(seq_len, d_model)` であることまで assert します。

</details>

**問3(Add & Norm を数える)** base model 全体で、(a) Add & Norm の箱はいくつあるか。(b) gamma と beta を合わせた layer norm のパラメータは全部で何個か。

<details><summary>略解</summary>

(a) encoder $2 \times 6 = 12$、decoder $3 \times 6 = 18$、計 **30個**(検証6の数字)。(b) 1箱あたり gamma `(512,)` + beta `(512,)` で 1024 個、全体で $30 \times 1024 = $ **30,720個**。第6章の演習で論文の 65M とパラメータ数を突き合わせるとき、layer norm の取り分がごく小さいことが見えてきます。

</details>
