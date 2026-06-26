# 第3章 訓練ループ — 4拍子の最終形

第1章で第7巻の部品を Transformer に組み上げ、自作スタックでは訓練が終わらないことを実測して PyTorch に卒業しました。第2章で、コーパスは BPE でトークンになり、長さの揃ったバッチになりました。モデルとデータが揃った今、残るは「学習させる」営みそのもの——訓練ループです。

第3巻4章で線形回帰を勾配降下で訓練したとき、私たちはこう約束しました。**「この4拍子は、第8巻で Transformer を訓練するその日まで、一切変わりません」**。その日が今日です。この章では forward → loss → backward → update の4拍子が、モデルが直線から Transformer に化けても1拍も変わらないことを、自分のコードで確認します。

ただし Transformer の訓練ループには、線形回帰になかった急所が2つあります。1つは **入力と出力を1トークンずらす**こと(3.1)、もう1つは損失に **label smoothing** を入れること(3.2)。論文 Section 5.4 の、第4巻終章で「読める」ようになった2文を、今度は自分の手で**実装**します。

この章のコードは `code/ch03/` にあります。とくに `model.py` の `TinyTransformer` は、第4〜6章がずっと使い続ける共有モデルです。


## 3.1 teacher forcing の確認と、入力・出力を1トークンずらす実装の急所

### 第6巻の予告を回収する

訓練時、decoder には自分の出力ではなく**正解の系列**を入力する——teacher forcing(教師強制)です。第6巻6.4の答えでした。訓練序盤のデタラメな出力を文脈に使うと足場が崩れるので、各ステップを「**正しい**文脈の次の1トークンを当てよ」に固定するのでした。

第6巻6.4にはこんな予告もありました。decoder への入力列は訓練前から全部わかっているので、**各ステップが前のステップを待たないモデルなら、全ステップの損失を一斉に計算できる**——「ここで覚えた teacher forcing を第8巻3章がそのまま使います」。その回収がこの節です。RNN は1ステップずつしか進めませんが、Transformer の decoder は causal mask(第7巻5章)のおかげで全位置の予測を1回の forward で並列に出せます。これが Transformer の訓練が速い根本の理由です。

### 1トークンずらす

「全位置を一斉に」を実装に落とします。対訳ペアの正解側を $y_1, y_2, \ldots, y_n$ とします。decoder の位置 $i$ の仕事は「$y_i$ までの正解文脈を見て次の $y_{i+1}$ を当てる」こと。つまり**入力の列と当てるべき出力の列は、同じ $y$ を1トークンずらしたもの**になります。

$$\texttt{tgt\_in} = [\mathrm{BOS},\ y_1,\ \ldots,\ y_n], \qquad \texttt{tgt\_out} = [y_1,\ \ldots,\ y_n,\ \mathrm{EOS}]$$

どちらも長さ $n+1$ です。位置を揃えて並べると、各位置が独立した「次トークン当てクイズ」になっているのが見えます。

```
図3.1: 1トークンずらし(y = [犬, が, 走る] の場合)

  位置             0      1      2      3
  tgt_in        [BOS]  [犬]   [が]   [走る]      ← decoder への入力
  causal mask    ↓ 位置 i は tgt_in[0..i] だけが見える
  tgt_out        [犬]   [が]   [走る] [EOS]      ← 当てるべき正解
```

- 位置 0 は、まだ何も書かれていない文脈(BOS だけ)から最初の語 $y_1$ を当てます。**BOS は「最初の1語を当てるときの文脈」の座席**です
- 位置 $n$ は、全文を見て EOS を当てます。**EOS を当てる訓練をしないと、モデルは生成をいつ止めればいいか学べません**(生成は第5章)

特殊トークンの規約は第2章のとおり、**PAD = 0, BOS = 1, EOS = 2** です。

### なぜ「急所」なのか

このずらしを「急所」と呼ぶのは、間違えたときの壊れ方が陰険だからです。ずらし忘れて `tgt_in` に `tgt_out` と同じ列を入れたとします。causal mask は位置 $i$ に `tgt_in[i]` **自身**を見せます(隠すのは未来だけ)。すると位置 $i$ の入力に、当てるべき答え `tgt_out[i]` がそのまま印刷されている。モデルは「入力を丸写しする」だけで損失をほぼゼロにでき、**訓練は大成功に見えるのに生成は完全に壊れる**——最悪のかたちで失敗します。第5章5.4の「うまくいかない時の手引き」に「ずらし忘れ」が典型原因として載るのはこのためです(演習2で踏んでみます)。

実装はバッチ単位です。バッチには長さの違うペアが混ざるので(第2章)、短い列の右側を PAD で詰めます。`code/ch03/train.py` の該当部分です。

```python
def collate(pairs):
    """対訳ペアの list → (src, tgt_in, tgt_out) の3つのテンソル(PAD 詰め)。

    急所(3.1): tgt_in = [BOS, y1..yn] / tgt_out = [y1..yn, EOS] の1トークンずらし。
    """
    src_len = max(len(s) for s, _ in pairs)
    tgt_len = max(len(t) for _, t in pairs) + 1          # BOS / EOS の分が1つ増える
    n = len(pairs)
    src = torch.full((n, src_len), PAD, dtype=torch.long)
    tgt_in = torch.full((n, tgt_len), PAD, dtype=torch.long)
    tgt_out = torch.full((n, tgt_len), PAD, dtype=torch.long)
    for i, (s, t) in enumerate(pairs):
        src[i, :len(s)] = s
        tgt_in[i, 0] = BOS
        tgt_in[i, 1:len(t) + 1] = t                      # [BOS, y1, ..., yn]
        tgt_out[i, :len(t)] = t
        tgt_out[i, len(t)] = EOS                         # [y1, ..., yn, EOS]
    return src, tgt_in, tgt_out
```

shape は `src` が `(batch, src_len)`、`tgt_in` と `tgt_out` がともに `(batch, tgt_len)`。mask——src の PAD を見せない pad mask と tgt の未来を見せない causal mask——は、この章の `TinyTransformer` が **forward の内部で `src` と `tgt_in` から自動生成**します。呼ぶ側は `model(src, tgt_in)` と書くだけで logits `(batch, tgt_len, vocab)` が返ります。mask の渡し忘れというバグの入り口を、設計ごと塞ぐ方針です。

## 3.2 損失: cross-entropy + label smoothing の実装

### 各位置は、語彙の上の分類問題

損失の土台は新しくありません。位置ごとの logits に softmax をかければ「次のトークンの確率分布」になり、正解 `tgt_out[i]` との cross-entropy(第4巻5章)が取れます。バッチと系列の全位置で平均すれば損失です。

ただし Transformer 特有の事情が1つ。**PAD の位置は損失から除外する**ことです。PAD はデータではなく詰め物です。除外しないと、モデルの仕事の一部が「PAD の位置で PAD と答える」ことに割かれ、しかもこの問題は簡単すぎて見かけの損失が不当に小さくなります(演習3で実測します)。

### ラスボス: 論文 5.4 を実装する

土台ができたところで、この章のラスボスです。

> *"During training, we employed label smoothing of value ε_ls = 0.1. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score."*
> — Vaswani et al., "Attention Is All You Need", Section 5.4
>
> 訳: 訓練では値 $\epsilon_{ls} = 0.1$ の label smoothing を用いた。モデルがより自信を持たない(more unsure)ように学習するため、これは perplexity を悪化させるが、accuracy と BLEU スコアは改善する。

この2文は、第4巻終章で「すべての単語が読める」ことを確認した箇所です。そこで「直観と実装は第8巻の仕事です」と預けた宿題を、いま回収します。第4巻5.4(KL divergence)の末尾でも「正解の分布をわざと少しなだらかにしてから、モデルの分布とのズレを測る——第8巻で回収します」と予告していました。両方ここで果たします。

### one-hot を smooth な分布にする

普通の cross-entropy の「正解の分布」は one-hot です。正解トークン $t$ に確率1、他の全トークンに確率0——エントロピーがゼロの、極限まで尖った分布でした(第4巻5章)。

この one-hot には教師として意地の悪いところがあります。softmax が確率ちょうど1を出すには、正解の logit を他より**無限に**大きくするしかありません。つまり one-hot を教師にする限り、モデルは「もっと自信を持て、もっと差を広げろ」という圧力を受け続け、訓練データに過剰に確信したモデルになりがちです。第3巻6章の言葉でいえば、過学習に向かう圧力です。

label smoothing は、教師の側の自信を先に削ります。語彙サイズを $K$ として、one-hot の代わりに

$$p'_k = (1 - \epsilon_{ls}) \cdot \mathbb{1}[k = t] + \frac{\epsilon_{ls}}{K}$$

を正解に使います。正解 $t$ から $\epsilon_{ls} = 0.1$ ぶんの確率を削り、**語彙全体に一様に**配り直した分布です(第4巻終章では「他のクラスに薄く配る」と紹介しました。配り方には「他の $K-1$ クラスに配る」流儀と「正解も含めた全 $K$ クラスに配る」流儀があり、本書は式が1行短くなる後者を採ります。$\epsilon_{ls}/K$ は微小なので両者の差は実用上ありません)。正解の確率は $1$ ではなく $(1-\epsilon_{ls}) + \epsilon_{ls}/K \approx 0.9$——「正解はほぼこれ。ただし他の可能性も完全には捨てない」という、少し自信のない教師です。

損失は、この $p'$ とモデル分布 $q$ の cross-entropy にします。第4巻5.4の分解がそのまま効きます。

$$H(p', q) = H(p') + D_{\mathrm{KL}}(p' \,\|\, q)$$

$p'$ は固定なので $H(p')$ は定数です。one-hot のときは $H(p) = 0$ でこの項は見えませんでしたが、smooth にした今は正の定数として居座ります。最小化には影響しないので、**この損失の最小化は「smooth な教師分布 $p'$ に、KL の意味でモデル分布を重ねにいくこと」と同じ問題**です。「one-hot を smooth な分布にして KL を最小化する」——第4巻5.4で予告した言い回しの実体です。

### 実装: 2項に分ける

$H(p', q)$ を定義どおり展開します。$p'$ の中身を代入して和を整理すると、

$$H(p', q) = -\sum_{k=1}^{K} p'_k \log q_k = (1 - \epsilon_{ls}) \cdot \underbrace{(-\log q_t)}_{\text{素の cross-entropy}} + \epsilon_{ls} \cdot \underbrace{\frac{1}{K}\sum_{k=1}^{K} (-\log q_k)}_{\text{全語彙の } -\log q \text{ の平均}}$$

第1項はいつもの「正解の確率の $-\log$」、第2項は「全トークンの $-\log q$ の平均」で、こちらが $\epsilon_{ls}$ の重みで混ざります。`code/ch03/model.py` の該当部分です。

```python
def label_smoothing_loss(logits, targets, eps=0.1, pad_id=0):
    """label smoothing 付き cross-entropy(論文 5.4, ε_ls = 0.1)。

    正解分布を one-hot から p' = (1-ε)·one-hot + ε·一様分布 になだらかにして、
    モデル分布 q との cross-entropy H(p', q) を取る(第4巻5.4: H(p') は定数なので
    KL(p' || q) の最小化と同じ問題)。targets が pad_id の位置は損失から除外する。
    logits (batch, tgt_len, vocab), targets (batch, tgt_len) → スカラー。
    """
    vocab = logits.size(-1)
    log_q = F.log_softmax(logits, dim=-1)                            # (batch, tgt_len, vocab)
    nll = -log_q.gather(-1, targets.unsqueeze(-1)).squeeze(-1)       # 正解項 -log q_t
    uniform = -log_q.mean(dim=-1)                                    # 一様項 (1/K)Σ(-log q_k)
    per_token = (1.0 - eps) * nll + eps * uniform                    # (batch, tgt_len)
    keep = (targets != pad_id).float()
    return (per_token * keep).sum() / keep.sum()
```

最後の2行が PAD の除外です。PAD 位置の損失に 0 を掛けて消し、**残った(本物の)トークン数で割る**——分母を `keep.sum()` にし忘れると、PAD の多いバッチほど損失が薄まります。

`eps=0.0` を渡すと第2項が消え、素の cross-entropy(PyTorch の `F.cross_entropy(..., ignore_index=PAD)`)と一致します。この一致は `test_model.py` で assert しています。

### "This hurts perplexity" の予告

この損失で訓練すると何が起きるか、先に1つだけ計算しておきます。モデルが教師 $p'$ に完璧に重なったとき、正解トークンに置かれる確率は $1$ ではなく約 $0.9$ です。つまり**どれだけ訓練しても素の cross-entropy は $-\log 0.9 \approx 0.105$ より下がらない**。perplexity(第4巻7章)はその指数なので必ず悪化します。論文が "This hurts perplexity" と認めるのはこの構造的な下限のためです。それでも accuracy と BLEU は改善する——本当でしょうか。演習1で確かめます。

## 3.3 [コード] 訓練ループ本体: forward → loss → backward → update の4拍子

### 5巻ぶんの約束を、1つのループで回収する

部品が揃いました。モデル(`TinyTransformer`)、バッチ(`collate`)、損失(`label_smoothing_loss`)。あとはループです。回収する約束を並べておきます。

1. **第3巻4章**: 訓練とは forward → loss → gradient → update の**4拍子**であり、これは第8巻まで一切変わらない
2. **第5巻5章**: 第3拍は手導出から `loss.backward()` に置き換わった。そして `grad` は累積する仕様なので**毎ステップのリセットが必須**——「これは PyTorch でも `optimizer.zero_grad()` として毎ステップ書くことになる、由緒正しい急所です」と予告済み

`code/ch03/train.py` の中心部です。

```python
def train(model, train_batches, val_batches, n_steps, lr=1e-3, eps_ls=0.1,
          eval_every=25, ckpt_path=None, device=None, log=True):
    """訓練ループの最終形。返り値は (loss 履歴, 検証 loss 履歴 [(step, val_loss)])。"""
    device = device or get_device()
    model.to(device)
    model.train()
    # update の道具。Adam の中身は第4章で完全に開ける(ここでは借り物)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history, val_history = [], []
    best_val = float("inf")
    step = 0
    while step < n_steps:
        for src, tgt_in, tgt_out in train_batches:
            if step >= n_steps:
                break
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

            logits = model(src, tgt_in)                              # 1. forward
            loss = label_smoothing_loss(logits, tgt_out,             # 2. loss
                                        eps=eps_ls, pad_id=PAD)
            optimizer.zero_grad()                                    #    grad のリセット(第5巻5章)
            loss.backward()                                          # 3. backward
            optimizer.step()                                         # 4. update

            history.append(loss.item())
            step += 1

            # --- 検証 loss の監視とチェックポイント(3.4) ---
            if step % eval_every == 0 or step == n_steps:
                val_loss = evaluate(model, val_batches, eps_ls, device)
                val_history.append((step, val_loss))
                if log:
                    print(f"step {step:4d}  train {loss.item():.4f}  val {val_loss:.4f}")
                if val_loss < best_val and ckpt_path is not None:
                    best_val = val_loss
                    torch.save({"step": step, "val_loss": val_loss,
                                "model_state": model.state_dict()}, ckpt_path)
    return history, val_history
```

ループの芯は、コメントの番号が付いた4行です。第3巻4章で線形回帰に書いたループと**同じ順番、同じ意味**——1拍目で予測し、2拍目で採点し、3拍目で反省し、4拍目で直す。モデルが `w * x + b` から数十万パラメータの Transformer になり、損失が MSE から label smoothing 付き cross-entropy になり、勾配が手導出から autograd になっても、拍子は1つも増えていません。**訓練ループはモデルから独立している**(第5巻5章)——その最終確認がこの4行です。

細部を3つだけ。

- **`optimizer.zero_grad()`**: 第5巻5章の grad リセットの回収です。`backward()` が勾配を `+=` で累積するのは PyTorch も自作 autograd も同じ理由(同じパラメータへ複数の経路から勾配が届く)で、リセットを忘れたときの壊れ方も第5巻の演習で実体験済みです
- **`torch.optim.Adam`**: 論文 5.3 の指定です。この章では中身を開けず「勾配を受け取ってパラメータを更新する箱」として**借りて**おきます。なぜ第3巻以来の SGD ではいけないのか、$\beta_1, \beta_2, \epsilon$ とは何か——それを完全に読むのが次の第4章です
- **`model.train()` / `model.eval()`**: dropout(第5巻6.4)は訓練中だけ働かせ、評価では切ります。この切り替えを PyTorch ではこの2つのメソッドが担います

### 動かす

データは本来、第2章の `make_corpus` / `make_batches` で作った対訳バッチを流し込みます。ただ、この章の主役はループそのものなので、実行例は**極小のダミー対訳**——「数列を入れると反転した数列が返る」翻訳もどき——で回します。語彙13(特殊トークン3 + 通常トークン10)、`d_model=64`、2層。CPU でも数秒で終わる規模です。`python3 train.py` の出力(抜粋)はこうなります。

```
step   25  train 2.4460  val 2.4197
step  100  train 1.8810  val 1.7599
step  200  train 1.1994  val 0.9143
step  300  train 0.9824  val 0.6418
ok: val loss 2.4197 → 0.6418, accuracy 0.975, best step 300 (val 0.6418) を復元できました
```

検証 loss は 2.42 から 0.64 へ、teacher forcing 下の次トークン正解率は 97.5% に達します。`train.py` の末尾では「検証 loss が初期値の半分未満に下がったか」「正解率が 90% を超えたか」を assert しており、4拍子が実際に坂を下ったことをコードで保証しています。直線のときと同じ4拍子が、Transformer でもそのまま坂を下る——第3巻の約束は果たされました。

## 3.4 検証 loss の監視とチェックポイント

### 訓練 loss を信じない

ループの中に、4拍子以外の塊が1つありました。`eval_every` ステップごとの検証です。これは第3巻6章の実践です。あの章で多項式の次数を上げる実験から学んだ教訓——**訓練 loss は下がり続けるので過学習を検出できない。手放した検証データの loss だけが汎化の物差しになる**——は、Transformer でもそのまま生きています。むしろパラメータが桁違いに多いぶん、丸暗記の余地も桁違いです。だから第2章で分割した検証データの loss を、訓練中ずっと監視します。

検証部分の実装です(`train.py` より)。

```python
@torch.no_grad()
def evaluate(model, batches, eps, device):
    """検証データ全体の平均 loss(訓練と同じ損失で測る)。"""
    model.eval()                                          # dropout を切る(第5巻6.4)
    total, count = 0.0, 0
    for src, tgt_in, tgt_out in batches:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        logits = model(src, tgt_in)
        n_tok = int((tgt_out != PAD).sum())
        total += label_smoothing_loss(logits, tgt_out, eps=eps, pad_id=PAD).item() * n_tok
        count += n_tok
    model.train()
    return total / count
```

`@torch.no_grad()` は「このブロックでは勾配を記録しない」という宣言です。検証は forward だけで backward しないので計算グラフを組む必要がなく、メモリも時間も節約できます。平均の取り方にも1つ細工があります。バッチごとの loss の単純平均ではなく、**トークン数で重み付け**しています。バッチによって PAD でない本物のトークン数が違うからです。

### チェックポイント: 最良の瞬間を保存する

監視するだけでなく保存もします。ループ内のこの部分です。

```python
                if val_loss < best_val and ckpt_path is not None:
                    best_val = val_loss
                    torch.save({"step": step, "val_loss": val_loss,
                                "model_state": model.state_dict()}, ckpt_path)
```

`model.state_dict()` は、モデルの全パラメータを「名前 → テンソル」の辞書として取り出すメソッドです。これを **検証 loss が自己ベストを更新したときだけ** ファイルに保存します。チェックポイント(checkpoint)と呼ばれるこの仕組みの動機は2つです。

1. **過学習への保険**(第3巻6章の実践)。訓練を続けるうちに検証 loss が反転して悪化し始めても、最良だった瞬間のモデルが手元に残ります。「いつ止めるべきだったか」を後から選び直せます
2. **事故への保険**。本物の訓練は数十分〜数時間走ります(第5章)。途中でプロセスが落ちても、チェックポイントがあれば最初からやり直さずに済みます

復元は `torch.load` と `load_state_dict` の2行です。`train.py` の末尾では、保存したチェックポイントを読み戻し、**復元したモデルの検証 loss が保存時の記録と一致する**ことを assert しています。保存できたつもりで復元できない事故を検算で塞いでいます。

```python
    saved = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"])
    restored_val = evaluate(model, val_batches, eps=0.1, device=device)
    assert abs(restored_val - saved["val_loss"]) < 1e-4, "復元したモデルの val loss が一致しない"
```

なお、論文 6.1 には最後の数個のチェックポイントの**平均**を取る技(checkpoint averaging)も出てきます。これは第5章で、生成・評価と一緒に概観します。

これで訓練ループは完成です。`model.py` と `train.py` は、このあと第4章(Adam と warmup)・第5章(本番の訓練と生成)・第6章(評価)がそのまま import して使います。

## まとめ

- 訓練時の decoder には正解を入力する(teacher forcing — 第6巻6.4の回収)。入力 `tgt_in = [BOS, y1..yn]` と出力 `tgt_out = [y1..yn, EOS]` の**1トークンずらし**が実装の急所。ずらし忘れると「訓練は成功に見えて生成が壊れる」
- 損失は cross-entropy + **label smoothing**($\epsilon_{ls} = 0.1$)。one-hot を $p' = (1-\epsilon)\cdot\text{one-hot} + \epsilon\cdot\text{一様}$ になだらかにし、$p'$ との cross-entropy(= 定数 + KL)を最小化する(第4巻5.4・終章の回収)。PAD 位置は損失から除外する
- 訓練ループは **forward → loss → backward → update の4拍子のまま**。第3巻4章の「第8巻まで一切変わりません」という約束は果たされた。`optimizer.zero_grad()` は第5巻5章の grad リセットの回収
- 検証 loss を定期監視し、自己ベスト更新時に `state_dict` をチェックポイントとして保存する(第3巻6章の実践)。復元できることまで assert で確認する
- Adam はこの章では「更新する箱」として借りた。中身の完全読解は第4章

**ラスボスとの距離**: 論文 5.4 の label smoothing の2文は、読めるだけでなく**自分のループの中で動いて**います。Section 5 で残るは 5.3 ——"We used the Adam optimizer with β₁ = 0.9, β₂ = 0.98" と warmup の式だけです。次章で仕留めます。

## 演習

**問1** `python3 ex_label_smoothing.py` を実行してください。同じ初期値・同じデータで $\epsilon_{ls} = 0$ と $\epsilon_{ls} = 0.1$ を訓練し、検証データで比較するスクリプトです。出力の表から、(a) "This hurts perplexity" が実測で成り立っているか、(b) 出力分布の「尖り方」はどちらが強いか、(c) 正解率は犠牲になったか、を読み取ってください。また、$\epsilon_{ls} = 0.1$ 側の「正解確率の平均」の値が、3.2 で計算した理論値とほぼ一致することを確かめてください。

**問2** `train.py` の `collate` を書き換えて、ずらしをやめてみてください(`tgt_in` の行を `tgt_in[i, :len(t)] = t` にして BOS を入れない——つまり `tgt_in` と `tgt_out` の本体を同じにする)。訓練 loss はどうなりますか。その loss を見て「成功した」と判断してよいか、3.1 の議論をもとに説明してください。

**問3** `label_smoothing_loss` から PAD の除外を外してみてください(`keep` を全部1にする)。ダミー対訳は長さ3〜8が混ざるので、バッチにはかなりの PAD が含まれます。訓練後の見かけの loss と、teacher forcing 正解率(PAD 除外で測ったもの)はどう変わりますか。

<details>
<summary>略解</summary>

**問1** 手元の実測(CPU)では次のようになりました。

| (検証データ) | $\epsilon_{ls}=0$ | $\epsilon_{ls}=0.1$ |
|---|---|---|
| 素の cross-entropy | 0.043 | 0.108 |
| perplexity | 1.044 | 1.114 |
| 正解確率の平均 | 0.974 | 0.913 |
| 分布エントロピー | 0.080 | 0.413 |
| 次トークン正解率 | 0.984 | 0.983 |

(a) perplexity は 1.044 → 1.114 と悪化しています(hurts perplexity)。(b) smoothing ありは正解確率が低く、エントロピーが5倍ほど大きい——分布の尖りが弱く、"more unsure" です。(c) 正解率はほぼ同じ。argmax で「どれを選ぶか」は壊れていません。正解確率 0.913 は、理論値 $(1-\epsilon) + \epsilon/K = 0.9 + 0.1/13 \approx 0.908$ とよく一致します——モデルは教師分布 $p'$ にほぼ重なるまで学習した、と読めます。なお、このダミータスクは簡単すぎて「accuracy が改善する」側までは観測できません。そちらは本物の翻訳タスク(第5・6章)の領分です。

**問2** 訓練 loss は数十ステップでほぼ下限(label smoothing の定数項のみ)まで急落します。通常300ステップかけて下がる坂が、一瞬で「終わって」しまう——明らかに速すぎます。これは学習ではありません。causal mask は位置 $i$ に `tgt_in[i]` 自身を見せるので、答えが入力に印刷されており、モデルは丸写しを学んだだけです。BOS から始めて自力で生成させると(第5章)、何も翻訳できません。「loss が不自然に速く下がる」は、ずらし忘れの典型的な症状として覚えておく価値があります。

**問3** 見かけの loss は本文の設定より低く出ます。「PAD の位置で PAD と答える」という、直前のトークンが PAD かどうかを見ればほぼ解ける簡単な問題が平均に混ざるからです。一方、本物のトークンだけで測った正解率は、同じステップ数では本文の設定に届きにくくなります。訓練の容量の一部が PAD 当てに使われるうえ、損失に占める本物のトークンの重みが薄まるためです。「損失の数値は、何で平均したかを言わなければ意味がない」という教訓でもあります。

</details>
</content>
</invoke>
