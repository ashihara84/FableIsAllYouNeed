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
| 2 | 第2巻(序章〜終章 8)+ 第3巻(序章〜エピローグ 9) | 17 | 完了(コード全PASS※)|
| 2b | 第3巻データ整合パス(2章のデータ仕様 w=7,b=20,勉強時間→点数 を正に 3〜6章を統一) | - | 完了(5章の本文と5巻5章コードはオーケストレータが手動修正) |
| 3 | 第4巻(9)+ 第5巻(8) | 17 | 完了(コード全PASS、5巻5章原稿はオーケストレータが直接執筆) |
| 4 | 第6巻(9)+ 第7巻(10) | 19 | **中断: 月間利用上限**(claude.ai/settings/usage で引き上げ後に再開) |
| 5 | 第8巻(10) | 10 | 未(同上) |
| 6 | 通し検収: 伏線整合(クリフハンガー連鎖・回収)、paper-map との突き合わせ、最終コミット | - | 未 |

## 再開手順(上限引き上げ後)

1. ウェーブ4: 第6巻9章+第7巻10章を章ごとに並列起動。プロンプトの型はウェーブ1〜3と同じ(必読ファイル列挙 → 担当章 → TOC準拠 → 序章・終章ペアに同一引用を明記 → assert検証)
   - 第6巻の序章・終章引用: Introduction の "This inherently sequential nature precludes parallelization..." + Background の "...grows in the distance between positions..." + 5.1 の "byte-pair encoding"
   - 第7巻: 巻全体がラスボス。各章 = 論文セクション(TOC参照)。**第5巻の tensor_autograd.py / micrograd.py を import する契約**を全章に明記
   - 第6巻5章(RNN)と7章(attention)は tensor_autograd を使う。実装が重い章なので分量・実行時間に注意
2. ウェーブ5: 第8巻10章。1章は第7巻のコードに依存するため、第7巻完了後に起動が安全(序章・2〜8章は並行可)
3. ウェーブ6: 通し検収(クリフハンガー連鎖、序章終章引用の全巻一致、paper-map突き合わせ、コード全実行)

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
