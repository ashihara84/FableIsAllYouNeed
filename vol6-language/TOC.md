# 第6巻: 言語をベクトルにする — TransformerまでのNLP(目次)

> ステータス: ドラフト v1(2026-06-10)
> ゴール: 言語モデルという問題設定を確立し、RNN の苦しみを実装で体感して「attention だけでいい」というタイトルの意味を刺す。歴史の章ではなく**動機**の章。

## ラスボス(巻頭掲示する論文箇所)

- Section 1 Introduction: "This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths..."
- Section 2 Background: "...the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions..."
- Section 3.4 Embeddings and Softmax / Section 5.1: "...encoded using byte-pair encoding"

巻頭で宣言: この巻を終えると、**論文の Introduction と Background が全文読める**。つまり「なぜ attention だけにしたのか」という論文の動機そのものが読める。

---

## 序章 ラスボスとの対面

- 0.1 ここまでの到達点: 部品はほぼ揃った。だが入力が x という数値ベクトルである前提だった。**言語はどうやって x になる?**
- 0.2 論文原文の掲示(上記ラスボス): "sequential nature"? "positions"? 何の話か全くわからないはず
- 0.3 この巻の構成宣言: n-gram → RNN → seq2seq → attention付きseq2seq と登るが、目的は歴史の網羅ではなく「Transformer が解決した問題を、自分の手で踏んで痛がる」こと
- 0.4 この巻で**扱わないこと**: 形態素解析・構文解析等の古典NLP、word2vec の詳細(動機の理解に必要な分だけ)

## 巻頭付録 論文読解マップ(第6巻の現在地ハイライト)

---

## 第1章 言語モデル — 次の単語を当てるゲーム

- 1.1 問題設定: P(次の単語 | これまでの単語)。第4巻1章の条件付き確率がそのまま骨格になる
- 1.2 文の確率 = 条件付き確率の積(連鎖分解)
- 1.3 「次を当てられる = 言語がわかっている」という作業仮説(GPT への遠い伏線)
- 1.4 評価: cross-entropy と perplexity(第4巻7章の回収。平均分岐数として体感)
- 演習: 人力言語モデル — 文の続きを予想して perplexity を競う

## 第2章 トークン化 — 文章を記号列に切る

- 2.1 何を1単位にするか: 文字(系列が長すぎる)vs 単語(語彙が爆発、未知語)のジレンマ
- 2.2 サブワードという折衷: 頻出パターンを1トークンに
- 2.3 BPE のアルゴリズム: 一番多いペアをくっつける、を繰り返すだけ
- 2.4 **[コード]** BPE をフルスクラッチ実装し、小さなコーパスで語彙を育てる過程を観察
- 2.5 論文 5.1 を覗き見: "byte-pair encoding" がもう読める
- 演習: 語彙サイズを変えてトークン列の長さを比較

## 第3章 埋め込み — トークンをベクトルにする

- 3.1 one-hot ベクトル: まずは素朴に。内積が常に0 = 「すべての単語が等しく無関係」という欠陥
- 3.2 埋め込み行列: one-hot @ E = E の行の取り出し(第1巻3章の行列の見方が効く)。lookup はただの行列演算
- 3.3 E も学習されるパラメータである: 意味は与えるものではなく、タスクから染み込むもの
- 3.4 **[コード]** 小さなコーパスで埋め込みを学習し、king − man + woman を検算(**第1巻1.5の伏線回収**)。類似度は内積/cosine(第1巻2章の回収)
- 3.5 論文 3.4 を覗き見: "learned embeddings of dimension d_model" — 読める
- 演習: 埋め込み空間の近傍語を観察

## 第4章 n-gram — 数えるだけの言語モデル

- 4.1 直前 n−1 個だけ見る、という割り切り。条件付き確率を頻度で推定(第4巻3章の最尤推定そのもの)
- 4.2 **[コード]** n-gram モデルを実装し、文章を生成してみる(初めて「機械が文を書く」体験)
- 4.3 壁: n を増やすと組合せ爆発でデータが足りない(スパースネス)。遠くの文脈は構造的に見えない
- 4.4 教訓: 「数える」のではなく「一般化する」必要がある → ニューラルへ
- 演習: n を変えて生成文と perplexity を比較

## 第5章 RNN — 記憶を持つネットワーク

- 5.1 発想: 隠れ状態 h を持ち回り、1トークンずつ読む。h = 「ここまでの要約」
- 5.2 **[コード]** RNN 言語モデルを実装(第5巻の autograd / NumPy で)・訓練。n-gram より良い perplexity を確認
- 5.3 **痛み1: 並列化できない** — h_t は h_{t-1} を待つ。GPU的計算(第1巻4章のベンチマーク)と相性最悪なことを訓練時間で体感
- 5.4 **痛み2: 長距離依存** — 遠い情報が薄まる/勾配が消える(第5巻6.1の再演が系列方向で起きる)
- 5.5 LSTM: ゲートで記憶を守る改良(概観のみ。実装はしない — 最短測地線)。それでも2つの痛みは本質的に残る
- 演習: 系列長と訓練時間・勾配ノルムの関係を測る

## 第6章 seq2seq — 翻訳という問題設定

- 6.1 系列を入れて系列を出す: encoder-decoder という分業(論文タイトルの "Transduction" の正体)
- 6.2 **[コード]** RNN encoder-decoder を小さなタスク(文字列反転や日付変換など)で実装
- 6.3 **痛み3: 固定長ボトルネック** — 文全体を1本のベクトルに圧縮する無理。長い入力で性能が崩れるのを観察
- 6.4 teacher forcing と自己回帰生成(第8巻の訓練ループの前提知識をここで)
- 演習: 入力長と精度の関係をプロット

## 第7章 attention — 「全部見ればいい」

- 7.1 発想の転換: 1本に圧縮せず、decoder の各ステップで**入力の全位置を見直す**
- 7.2 どこを見るかは内積で決める: 今の状態(query)と各位置(key)の類似度 → softmax で重み → 重み付き和(value)。**Q・K・V という言葉をここで初めて導入**(第1巻2章の「内積=類似度」、第4巻6章の softmax が合流する)
- 7.3 **[コード]** attention 付き seq2seq を実装。6章と同じタスクで長い入力に強くなることを確認 + attention 重みの可視化(どこを見て訳したかが見える、本巻のハイライト図)
- 7.4 振り返ると: RNN 部分が一番遅くて一番弱い。「attention が本体で、RNN は足枷では?」という問い — **タイトル "Attention Is All You Need" がここで刺さる**
- 演習: attention マップを読んで誤訳の原因を探す

## 終章 ラスボス再戦 — Introduction が読める

- 8.1 Introduction / Background を全文再読: "sequential nature precludes parallelization"(5.3で体感した)、"grows in the distance between positions"(5.4で体感した)— 全部自分の痛みとして読める
- 8.2 残る問い: RNN を本当に取り除いたら、語順の情報はどうなる? 自分自身への attention とは?(第7巻の Positional Encoding / Self-Attention への正確な需要を言語化して終わる)
- 8.3 次巻予告: いよいよ論文精読。もう新しい数学は出てこない

---

## 論文の記号 ↔ 本巻の章 対応

| 論文の記号・概念 | 章 |
|---|---|
| Introduction(RNNの並列化不能・長距離依存) | 5章 |
| Background(位置間距離と計算量) | 5・7章 |
| byte-pair encoding(5.1) | 2章 |
| learned embeddings, d_model(3.4) | 3章 |
| query / key / value という言葉 | 7章 |
| auto-regressive(3節冒頭) | 6章 |

## コードで示す箇所の方針

- コードを入れる: BPE(2章)、埋め込み学習(3章)、n-gram生成(4章)、RNN LMと痛みの計測(5章)、seq2seq(6章)、attention付きseq2seqと重み可視化(7章)
- コードを入れない: 言語モデルの定式化(1章)は数式で。LSTM の内部(5.5)は図のみ(実装しない判断を明記)
