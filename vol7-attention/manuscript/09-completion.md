# 終章 完走確認 — 式番号 ↔ コード行 対応表の完成

第8章までで、この巻の仕事はすべて終わりました。Section 3 の5つの小節を順に読み、出てきた部品をその場で実装し、テストを通してきました。新しい数学はひとつも登場しませんでした——序章で宣言したとおり、装備は最初から全部あったのです。

序章で、私たちはこの巻のラスボスをこう定めました。**論文そのもの。特に Section 3 Model Architecture と図1。** これまでの巻のように「論文の一部」を掲げるのではなく、本体に正面から挑む巻でした。

毎巻の巻末で繰り返してきた儀式を、今回も行います。ただし今回は1つの式の再読では終わりません。**Section 3 を、頭から終わりまで、通しで読みます。**


## 9.1 Section 3 通し再読 — 止まらずに読めることを確認する儀式

ラスボス、再戦です。序章に掲げた Section 3 の冒頭を、一言一句変えずに再掲します。

> *"The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively."*
> — Vaswani et al., "Attention Is All You Need", Section 3
>
> 訳: Transformer はこの全体アーキテクチャに従い、encoder と decoder の両方を、積み重ねた self-attention と位置ごとの全結合層で構成する。それぞれ図1の左半分と右半分に示す。

序章でこの文を見たとき、単語は全部知っているのに、文としては読めませんでした。"stacked self-attention" の stack とは何か、"point-wise" とは何の位置か。いまから、この文に続く 3.1 から 3.5 までを駆け抜けます。各小節につき確認することは1つだけ——**どこで立ち止まるか**です。立ち止まらなければ、勝ちです。

**3.1 Encoder and Decoder Stacks。** N=6 の同一レイヤーの積み重ね、$\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$ という配管の規約、すべてを $d_{model} = 512$ で揃える理由。第2章で、中身が空っぽのダミー Sublayer のまま骨組みを先に組んで読めるようになりました(配管そのものは第5巻第6章の装備です)。

**3.2.1 Scaled Dot-Product Attention。** 式(1)と、$\sqrt{d_k}$ の脚注と、softmax の前に $-\infty$ を足す masking。第3章で実装し、2×2 の手計算と一致することまで確かめました。式(1)は第1巻終章から数えて4度目の再会ですが、今回が初めての「完全装備での再会」でした。

**3.2.2 Multi-Head Attention。** "averaging inhibits this" の一文と、$d_{model}$ を h=8 個の頭に裂くテンソル整形。第4章で実装し、h=1 のとき第3章の実装と完全一致することをテストで確認しました。

**3.2.3 Applications of Attention in our Model。** attention の3つの使い方——encoder self、decoder masked self、encoder-decoder(cross)。第5章で、Q・K・V と mask の組合せを呼び分けるテストとして読めるようになりました。

**3.3 Position-wise Feed-Forward Networks。** 式(2)。第5巻終章ですでに読めていた式に、第6章で "position-wise" の一語(各位置に同じ MLP を独立適用)を足して読了しました。

**3.4 Embeddings and Softmax。** 入力埋め込み、出力側の linear + softmax、3か所の weight sharing と $\sqrt{d_{model}}$ 倍。第6章で実装し、base model のパラメータ数 65M を自分で数えて論文と突き合わせました。

**3.5 Positional Encoding。** sin / cos の式(3)。第7章で、attention が語順を見ていないことをコードの実験で確かめてから、語順を「足し算で」取り戻しました。第6巻終章の問い「語順はどこへ?」の回収です。

——以上です。3.1 から 3.5 まで、立ち止まる場所はもうありません。序章では「読めない箇所は 3.2 の組み立てと 3.5 だけ」でした。その2つが埋まり、Section 3 は端から端まで、知っている部品の組み合わせとして読めます。ついでに言えば、Section 4 の Table 1 も第8章で全行検算済みです。

## 9.2 成果物 — 論文の式・主張 ↔ 自分のコード 完全対応表

この巻のゴールは、序章でこう定義しました。「論文の式番号 ↔ 自分のコードの行」が1対1対応する状態。それを1枚の表にしたものが、この巻の成果物です。

| 論文の箇所 | 式・主張 | 自分のコード | 本巻の章 |
|---|---|---|---|
| 3.1 | N=6 の stack、$\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$ | `code/ch02/stack_skeleton.py` | 第2章 |
| 3.2.1 式(1) | $\mathrm{Attention}(Q,K,V) = \mathrm{softmax}(QK^T/\sqrt{d_k})V$ | `code/ch03/attention.py` | 第3章 |
| 3.2.1 masking | softmax の前に $-\infty$(padding / causal) | `code/ch03/attention.py` の `mask` 引数 | 第3章 |
| 3.2.2 | $\mathrm{MultiHead} = \mathrm{Concat}(\mathrm{head}_1, ..., \mathrm{head}_h)W^O$、h=8 | `code/ch04/multi_head.py` | 第4章 |
| 3.2.3 | attention の3つの使い方(self / masked self / cross) | `code/ch05/three_attentions.py` | 第5章 |
| 3.3 式(2) | $\mathrm{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$ | `code/ch06/position_wise_ffn.py` | 第6章 |
| 3.4 | embedding、出力 linear + softmax、weight sharing、$\sqrt{d_{model}}$ 倍 | `code/ch06/embedding.py` | 第6章 |
| 3.5 式(3) | $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$(cos も同様) | `code/ch07/positional_encoding.py` | 第7章 |
| 4 / Table 1 | 層あたり計算量・逐次操作数・最大経路長の検算 | `code/ch08/attention_scaling.py` | 第8章 |

この表の各行は、3つの保証を束ねています。**原文を逐行で読んだ**こと(各章の前半)、**自分の手で実装した**こと(各章の `code/`)、そして**テストが通る**こと(`python3` で実行すれば assert が全部通る)。論文の主張とあなたのコードが、行単位で握手している状態です。

これが、本シリーズの**卒業証書の半分**です。

なぜ半分なのでしょうか。この表が保証しているのは「論文の Section 3・4 が読めて、全部品が単体で正しく動く」ことまでです。論文にはまだ Section 5 Training と Section 6 Results が残っています。そして、そこに書かれていることは、この表のどの行にも対応していません。残り半分の正体は 9.4 で直視します。

巻頭付録の論文読解マップを開いて、第7巻の行に印を付けてください。第1巻終章で初めて灯した明かりは、Section 3 と 4 の全域に届きました。地図の暗がりは、Section 5 から先だけです。

## 9.3 部品チェックリスト最終版 — 「まだの箱」がすべて消えた

第5巻終章で、私たちは図1の部品チェックリストを作りました。あのとき表は2つに割れていました——「読める箱」と「まだ読めない箱」。まだ読めない箱は3種類ありました。Multi-Head Attention、Positional Encoding、Embedding です。

その3つの、現在の状態です。

| 第5巻終章の「まだの箱」 | 何が足りなかったか | どこで読める箱になったか | 単体テスト |
|---|---|---|---|
| Multi-Head Attention | attention そのもの。なぜ類似度の表で文が処理できるのか、なぜヘッドが複数要るのか | 第6巻7章(attention の動機)→ 本巻第3〜5章(式(1)・multi-head・3つの使い方) | `code/ch03/`〜`code/ch05/` |
| Positional Encoding | 語順——「位置」を数値で表す方法 | 本巻第7章(式(3)、足すという選択) | `code/ch07/` |
| Embedding(Input / Output) | そもそも単語をベクトルにする方法 | 第6巻3章(埋め込み)→ 本巻第6章($\sqrt{d_{model}}$ 倍と weight sharing まで) | `code/ch06/` |

**まだの箱は、ゼロになりました。** 第5巻ですでに読めていた箱(Feed Forward、Add & Norm、Dropout、出口の Linear と Softmax)と合わせて、図1のすべての箱が「読める箱」です。しかもこの巻の規律により、読めるだけではありません——**すべての箱に、単体テスト済みの自分の実装が対応しています**。第5巻の表にあった「どの巻で読めるようになるか」という列は、もう書く必要がありません。未来形の列が消えた、シリーズで最初のチェックリストです。

序章の言葉を回収しましょう。この巻は「確認しながら進む巻」でした。1〜6巻で配った伏線——第1巻の shape と行列の束、第3巻のバッチ、第4巻の softmax と $\sqrt{d_k}$、第5巻の配管と autograd、第6巻の埋め込みと「3つの痛み」——が、それぞれどの章で回収されたか。それがそのまま 9.1 と 9.2 の表になっています。新しい数学ゼロという序章の宣言は、最後まで守られました。

## 9.4 まだやっていないことの正直な確認 — 組み立てて、訓練する

ここまで読めるようになったものを数えてきました。最後に、毎巻の流儀どおり、**できていないこと**をごまかさずに数えます。

論文の目次で言えば、残りは Section 5 Training と Section 6 Results です。語彙としては読めるものも多いはずです——training data、Adam、learning rate、dropout。3巻・2巻・5巻で出会った言葉たちです。しかし「単語が読める」ことと「自分にできる」ことの間には、この巻で埋めなかった溝が2つあります。

**1つ目、組み立てていません。** 9.2 の表の部品は、すべて単体でテストされています。しかし、単体テストはあくまで単体の保証です。attention の出力を FFN に渡し、N=6 段積み、encoder の出力を decoder の cross-attention に刺し、入口の embedding から出口の softmax まで1本につなぐ——図1を**1つの動くプログラムにする**作業は、まだ何もしていません。各部品の入出力 shape を厳密に揃えてきたのは(序章 0.3 の約束)、この組み立てのためです。

**2つ目、訓練していません。** こちらが本命です。この巻の $W$ は、第1巻と同じく、最後まで乱数のままでした。テストが確かめたのは「配管が正しい」ことであって、「良い翻訳をする」ことではありません。乱数の $W$ を持つ Transformer は、完璧に組み上がっていても、でたらめな文を吐きます。

そして、ここに罠があります。**部品が動くことと、全体が学習することは、別問題です。** 全部品の単体テストが通っていても、数千万個のパラメータを持つ全体が安定して学習する保証はどこにもありません。論文の Section 5 が Adam のハイパーパラメータや warmup の式や label smoothing をわざわざ数値込みで指定しているのは、それが飾りではなく、**この規模のモデルを学習させるために必要な処方箋**だからです。第2巻で直観だけ作った warmup、第4巻終章で予告した label smoothing——伏線の最後の束は、訓練の現場で回収されます。

第8巻「全部組み立てて訓練する」で、この2つの溝を埋めます。部品を1つの Transformer に組み上げ、訓練ループを書き、Adam + warmup で実際に学習させ、小さなコーパスで翻訳を出し、BLEU で測る。そこまで終えたとき、Section 5・6 が読めるようになり、論文を第1ページから最終ページまで通読して、卒業証書のもう半分が手に入ります。

部品は、全部揃いました。全部、動きます。組み立てに行きましょう。
