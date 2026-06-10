# 夜間ビルド計画(2026-06-10 夜 → 06-11 朝)

> 全8巻の原稿初稿を一晩で書き切る。オーケストレータ(メインセッション)用の進行表。
> コンテキスト要約後の自分への注意: このファイルと git log を見れば現在地がわかる。

## 体制

- 文体基準: STYLE.md + 第1巻序章(オーケストレータが直接執筆した見本)
- 章ごとに1エージェント(バックグラウンド並列)。各エージェントは CLAUDE.md / DESIGN.md / 担当巻 TOC.md / STYLE.md / paper-map.md / vol1 序章(見本)を読んでから執筆
- 検収: 各ウェーブ完了時にファイル存在・分量・規約(ですます調、shape注釈、ラスボス引用、演習+略解)をスポットチェック → git commit
- コード: numpy / torch は実行検証する。matplotlib は掲載のみ
- 環境メモ: Python 3.9.6 / numpy 2.0.2 / torch 2.8.0 / matplotlib なし

## ウェーブ進行表

| # | 内容 | ファイル数 | 状態 |
|---|---|---|---|
| 0 | git init, STYLE.md, BUILD-PLAN.md, vol1 序章(見本), 初回コミット | - | 完了 |
| 1 | 第1巻 1〜6章 + 終章 | 7 | 完了(全コードPASS、検収済) |
| 2 | 第2巻(序章〜終章 8)+ 第3巻(序章〜エピローグ 9) | 17 | 進行中 |
| 3 | 第4巻(9)+ 第5巻(8) | 17 | 未 |
| 4 | 第6巻(9)+ 第7巻(10) | 19 | 未 |
| 5 | 第8巻(10) | 10 | 未 |
| 6 | 通し検収: 伏線整合(クリフハンガー連鎖・回収)、paper-map との突き合わせ、最終コミット | - | 未 |

ウェーブ5を4と並行launchしてもよい(8巻は7巻原稿に依存しない。TOCにのみ依存)。

## ファイル名規約(確定)

- vol1: 00-prologue / 01-vectors / 02-dot-product / 03-matrices / 04-matmul / 05-linear-maps / 06-batch-and-linear / 07-boss-rematch
- vol2: 00-prologue / 01-derivative / 02-minima / 03-gradient-descent-1d / 04-partial-and-gradient / 05-chain-rule / 06-fitting-parameters / 07-boss-rematch
- vol3: 00-prologue / 01-what-is-learning / 02-linear-regression / 03-loss / 04-training / 05-minibatch / 06-generalization / 07-boss-rematch / 08-epilogue-cliffhanger
- vol4: 00-prologue / 01-probability / 02-expectation-variance / 03-likelihood / 04-logistic-regression / 05-entropy / 06-softmax / 07-variance-of-dot-products / 08-boss-rematch
- vol5: 00-prologue / 01-activations / 02-mlp / 03-backprop / 04-autograd / 05-training-with-autograd / 06-deep-toolbox / 07-boss-rematch
- vol6: 00-prologue / 01-language-models / 02-tokenization / 03-embeddings / 04-ngram / 05-rnn / 06-seq2seq / 07-attention / 08-boss-rematch
- vol7: 00-prologue / 01-intro-background / 02-stacks / 03-scaled-dot-product / 04-multi-head / 05-three-attentions / 06-ffn-embeddings / 07-positional-encoding / 08-why-self-attention / 09-completion
- vol8: 00-prologue / 01-assembly / 02-data / 03-training-loop / 04-adam-warmup / 05-training-in-practice / 06-evaluation / 07-graduation / 08-epilogue-gpt-bert

## 検収チェックリスト(各ウェーブ)

- [ ] 全ファイル存在・分量(本文章 8,000字以上、序章・終章 4,000字以上)
- [ ] ですます調 / shape注釈 / ラスボス引用(序章・終章で同一)/ 演習+略解
- [ ] [コード] 指定章: code/chNN/ に .py があり `python3` で assert が通る(torch は8巻のみ)
- [ ] TOC にある伏線文言(「第N巻で回収」等)が原稿に存在する
- [ ] commit: `vol N: 初稿(章リスト)`
