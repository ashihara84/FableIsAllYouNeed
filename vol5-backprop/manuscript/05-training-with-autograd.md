# 第5章 autograd で学習する — もう手で微分しない

前章で自動微分（`Value` クラス、約130行、全演算が数値微分との照合を通過）を手に入れました。ただし前章で微分したのは $(3x+1)^2$ のような「おもちゃの式」です。この道具の真価は、式が人間の手に負えなくなったときに現れます。第2章2.4で2層MLPの $\partial L / \partial W_1$ を手導出して見た地獄が、この章で消滅します。

この章でやることは3つ。`Value` で MLP を組み、第1章で線形モデルが歯が立たなかったデータを学習させる（5.1）。第3巻・第4巻で手導出した勾配たちを autograd がすべて自動で再現することを確認する（5.2）。スカラー1個ずつの `Value` の致命的な弱点——遅さ——を実測し、行列版へ進化させる（5.3）。この行列版が第6巻から第8巻まで使い続ける道具になります。

コードの完全版は `code/ch05/train_value_mlp.py` にあります（`python3` で全 assert 通過）。本文では中身を流れに沿って分けて掲載します。

## 5.1 [コード] Value ベースの MLP を、4拍子で訓練する

### linear 層と MLP を Value で組む

部品はすべて揃っています。第1巻6章の `linear(X, W, b)` を `Value` 1個ずつで組み直すところから始めます。

```python
class LinearValue:
    """第1巻6章の linear(X, W, b) = X @ W + b を、Value 1個ずつで組んだもの。"""

    def __init__(self, rng, d_in, d_out):
        # 初期値はひとまず小さな乱数(初期化の理論は第6章6.6)
        self.W = [[Value(rng.uniform(-1.0, 1.0)) for _ in range(d_out)]
                  for _ in range(d_in)]
        self.b = [Value(0.0) for _ in range(d_out)]
        self.d_in, self.d_out = d_in, d_out

    def __call__(self, x):
        out = []
        for j in range(self.d_out):
            acc = self.b[j]
            for i in range(self.d_in):
                acc = acc + x[i] * self.W[i][j]
            out.append(acc)
        return out

    def parameters(self):
        return [w for row in self.W for w in row] + self.b
```

行列がないので `X @ W + b` は2重ループに開いて書いてあります。不格好ですが、このループの1周ごとに**計算グラフのノードが生えている**ことに注意してください。forward を書いただけで backward の準備は終わっています。

MLP はこれを重ねて間に ReLU を挟むだけです（第2章）。

```python
class MLPValue:
    """linear → ReLU → linear → …(第2章の MLP)。最後の層は活性化なし。"""

    def __init__(self, rng, sizes):
        self.layers = [LinearValue(rng, d_in, d_out)
                       for d_in, d_out in zip(sizes[:-1], sizes[1:])]

    def __call__(self, x):
        for k, layer in enumerate(self.layers):
            x = layer(x)
            if k < len(self.layers) - 1:
                x = [h.relu() for h in x]
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

データは、第1章1.1で線形モデルを50%台に張り付かせたあの市松模様（データB、200点）を1点も違わず再現して使います。雪辱戦です。

### 訓練の前に: 地獄の手導出と照合する

訓練の前に確かめておきたいことがあります。`backward()` が吐く $\partial L / \partial W_1$ は、第2章2.4で手導出し第3章3.3で行列の形に整理した、あの式と一致するのでしょうか。

完全版では、同じ初期パラメータの2層MLP（2→8→1、sigmoid + log loss）に対して、(1) `loss.backward()` が埋めた `grad` と、(2) 第3章3.3の行列 backprop（$\delta$ を逆向きに流し、$\partial L/\partial W = \text{入力}^T @\, \delta$）を NumPy で手実行した結果を突き合わせています。結論を引用します。

```
検算 OK: 第2章2.4で地獄だった ∂L/∂W_1 が、backward() と一致
```

`np.allclose` の許容誤差は $10^{-12}$。一致は完全です。第2章の地獄は正しい地獄だった——そして、もう二度と通らなくてよい地獄になりました。

### 4拍子は変わらない

訓練ループを書きます。**第3巻4章で命名した4拍子（forward → loss → gradient → update）が1拍も変わらず**そのまま現れることに注目してください。

```python
def train_mlp_value(X, y, hidden=8, lr=0.5, num_steps=120, verbose=True):
    """4拍子(第3巻4章)で Value ベースの MLP を訓練する。所要時間も返す。"""
    rng = np.random.default_rng(0)
    model = MLPValue(rng, [2, hidden, 1])
    n = len(X)
    history = []
    t0 = time.perf_counter()
    for step in range(num_steps):
        # 1. forward: 全サンプルの予測(計算グラフがここで組み上がる)
        probs = [sigmoid_value(model([Value(X[i, 0]), Value(X[i, 1])])[0])
                 for i in range(n)]
        # 2. loss: log loss の平均(第4巻4章)
        loss = sum((log_loss_value(p, y[i]) for i, p in enumerate(probs)),
                   Value(0.0)) * (1.0 / n)
        # 3. gradient: 手導出の代わりに backward() 一発。grad は累積するので先にゼロへ
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()
        # 4. update: 坂を下る(第2巻3章)
        for p in model.parameters():
            p.data -= lr * p.grad
        history.append(loss.data)
        if verbose and step in (0, 10, 40, num_steps - 1):
            print("  step {:>4}: loss = {:.6f}".format(step, loss.data))
    elapsed = time.perf_counter() - t0
    acc = np.mean([(sigmoid_value(model([Value(x1), Value(x2)])[0]).data >= 0.5)
                   == (lab == 1.0) for (x1, x2), lab in zip(X, y)])
    return model, history, acc, elapsed
```

変わったのは第3拍だけです。「4.1で手導出した式」と書かれていた場所に、`loss.backward()` の1行が入りました。手導出した式はモデルを変えるたびに導出し直しでしたが、`backward()` はモデルがどう変わっても同じ1行です。**訓練ループは、モデルから独立した**——これが自動微分がもたらした構造の変化です。

実装の急所が1つ。第3拍の直前で全パラメータの `grad` を 0.0 に戻している2行です。第4章で作ったとおり `backward()` は勾配を `+=` で**累積**します（同じノードへ複数の道から勾配が届くため、そうでなければ正しく動きません）。リセットを忘れると前のステップの勾配が混入したまま坂を下ることになります。PyTorch でも `optimizer.zero_grad()` として毎ステップ書くことになる由緒正しい急所です（演習3で実際に忘れてみます）。

実行結果です。

```
step    0: loss = 1.031911
step   10: loss = 0.095250
step   40: loss = 0.037388
step  119: loss = 0.020870
正解率: 99.0%   所要時間: 5.35 秒
```

第1章で50%台に張り付いた市松模様が、**正解率99.0%**。直線では割れなかったデータが、2→8→1 の小さなMLPで、勾配の式を1行も書かずに割れました。1.4 で予告した「折れ線で何でも近似する」が学習によって現実になった瞬間です。

ただし、所要時間 **5.35秒** という数字を頭の片隅に置いてください。たった200点、120ステップ、パラメータ33個でこの時間です。これが 5.3 の伏線です。

## 5.2 過去2巻の再実装 — 手導出が全部自動で出る

autograd の正しさを、もっと意地悪に確かめます。私たちはこれまでに3つの勾配を手で導出しています。

1. 第3巻4.1: 線形回帰 + MSE の $\partial L/\partial w = \frac{2}{n}\sum r_i x_i$
2. 第4巻4.3: ロジスティック回帰 + log loss の $(p - y)\cdot x$
3. 第4巻6.3: softmax + cross-entropy の $(p - t)$

この3つすべてを、**手導出・数値微分・autograd の三つ巴**で照合します。

### 5.2a 第3巻の線形回帰

データは第3巻2章と同一——あの「勉強時間→点数」の20人（seed 42、真の規則 $y = 7x + 20$）です。MSE を `Value` で組みます。微分の式はどこにも書きません。

```python
def mse_loss_value(w, b, X, y):
    """MSE を Value で組む。微分の式はどこにも書かない。"""
    total = Value(0.0)
    for i in range(len(X)):
        r = w * X[i] + b - y[i]
        total = total + r * r
    return total * (1.0 / len(X))
```

$(w, b) = (0, 0)$ 地点での勾配を、三者で測り比べた結果がこれです。

```
手導出   : grad_w = -670.539664, grad_b = -113.617662
数値微分 : grad_w = -670.539664, grad_b = -113.617662
autograd : grad_w = -670.539664, grad_b = -113.617662
```

小数第6位まで3行が完全に同じです。手導出（理論）と数値微分（実測）と autograd（自動）が同じ坂を指している——第2巻1章から続けてきた「照合の習慣」のひとつの到達点です。

そのまま4拍子で訓練します（lr = 0.01 も第3巻4章と同じ）。

```
学習結果: w = 6.722065, b = 22.092189(第3巻4.4の解析解と一致するか?)
```

第3巻4.4の解析解は $(w^\*, b^\*) = (6.7221, 22.0922)$。一致です。**勾配の式を1行も書かずに、第3巻と同じ答えに着きました。**

### 5.2b 第3巻エピローグの分類 — 凍結地点からの生還、再び

次は因縁の対決です。第3巻エピローグで学習が完全に凍結した初期値 $\mathbf{w} = (-8, -8)$。第4巻4章では「損失を log loss に替えれば生き返る」ことを手導出した勾配で確かめました。今回は同じことを autograd だけでやります。

実装上の工夫を1つ。`Value` で log loss を組むとき、$p = \sigma(z)$ を経由して $\log(1-p)$ と書くと、初期値 $(-8, -8)$ では $p$ が float の精度で 1.0 に張り付き、$\log 0$ の事故になります。そこで $-\log\sigma(z) = \log(1 + e^{-z})$ という恒等式を使い、$z$ から直接損失を組みます。「数式どおりの素直な実装が数値の世界では事故ること」は第4巻6.2（softmax の最大値シフト）で学んだとおりで、これはその log loss 版です。

照合結果です。

```
手導出   : grad_w = [-2.036771, -2.018299], grad_b = -0.004986
数値微分 : grad_w = [-2.036771, -2.018299], grad_b = -0.004986
autograd : grad_w = [-2.036771, -2.018299], grad_b = -0.004986
300ステップ後: loss = 0.013638, 正解率 = 99.5%
```

第3巻エピローグの実測では、この地点の MSE の勾配は $10^{-5}$ のオーダーでした。log loss の勾配は約 2.0——**10万倍**の差です。300ステップ後、正解率 0.5% で凍っていたモデルは 99.5% に到達しました。第4巻4.4と同じ結末ですが、今回は勾配の導出に1秒も使っていません。

### 5.2c 第4巻の softmax 分類

3クラスの小さな問題で、softmax + cross-entropy を `Value` の `exp()` と `log()` から素朴に組み、`backward()` の勾配を第4巻6.3の手導出 $\frac{1}{n} X^T(P - T)$ と照合します。結果は `np.allclose`（許容 $10^{-9}$）で一致。

```
ok: softmax + cross-entropy の勾配 (p - t) も自動で出た
```

これで、このシリーズで行った手導出は**すべて autograd で再現されました**。第3巻4.1も第4巻4.3も6.3も、いまや「backward() が出す答えの検算」という名誉職に退きました。以後の巻で、私たちが勾配を手で導出することは二度とありません。

## 5.3 スカラー Value は遅い — 行列版 tensor_autograd へ

最後に伏線を回収します。5.1 の訓練、たかが200点・120ステップに **5.35秒** かかりました。なぜでしょうか。

理由は第1巻2.5と4.5で実測したとおりです。`Value` は**数1個につきオブジェクト1個**を作ります。`(200, 8)` の行列積1回ぶんの計算が、`Value` の世界では数千個のノード生成と Python のループに化けます。第1巻で測った「forループは数百〜数千倍遅い」差が、計算グラフのオーバーヘッド込みでそのまま襲ってくるのです。

### 設計はそのまま、持ち物を行列に

解決策も第1巻と同じです。**1個ずつやめて、行列で一斉にやる。** `code/ch05/tensor_autograd.py` に、ノードの持ち物を「数1個」から「`np.ndarray` 1枚」に置き換えた `Tensor` クラスを実装しました。設計は第4章の `Value` と寸分違わず——値と勾配と「親と演算」を覚え、`backward()` がトポロジカル順にグラフを逆走します。変わるのは各演算の backward が**行列の式**になることだけ。たとえば行列積はこうなります。

```python
    # --- 行列積(2次元どうしのみ。第3章で手導出した backward の式がそのまま入る) ---
    def __matmul__(self, other):
        assert self.data.ndim == 2 and other.data.ndim == 2  # 最小限: 2次元のみ対応
        out = Tensor(self.data @ other.data, (self, other))

        def _backward():
            self.grad += out.grad @ other.data.T   # ∂L/∂X = δ @ W^T
            other.grad += self.data.T @ out.grad   # ∂L/∂W = X^T @ δ(第3章3.3の式)

        out._backward = _backward
        return out
```

`_backward` の2行を見てください。第3章3.3で導出した $\partial L/\partial W = X^T @\, \delta$ が、**そのまま実装**になっています。あの導出は、行列版 autograd の matmul ノード1個分の設計図だったのです。

ほかに実装したのは、ブロードキャスト対応の加算（`X @ W + b` の `+ b` で複製された勾配を足し戻す `_unbroadcast`。「道が複数なら勾配は足す」——第2巻5章の分配の規則そのもの）、要素ごとの積、`relu` / `log` / `exp` / `sum` / `mean`、そして softmax + cross-entropy を1ノードに融合した数値安定版 `softmax_cross_entropy`（第4巻6.2の最大値シフト入り）。これだけです。全文は `tensor_autograd.py` を読んでください——150行強、もう全行が読めるはずです。照合テストは `test_tensor_autograd.py` にあり、全演算が数値微分と一致します。

### 実測: 671倍

5.1 とまったく同じ課題・同じ4拍子を `Tensor` で書き直して（コードは完全版を参照。forward が `(X_t @ W1 + b1).relu() @ W2 + b2` の1行になります）、時間を測り比べます。

```
スカラー Value: 5.35 秒(5.1 の実測)
行列版 Tensor : 0.0079 秒   正解率: 99.0%
速度比: 約 671 倍
```

正解率は同じ、時間は3桁違い。第1巻のベンチマークの教訓は autograd の世界でもそのまま生きていました。

ここで宣言します。**第6巻からは、この `tensor_autograd.py` の `Tensor` を標準装備として使います。** RNN も（第6巻）、attention の部品も（第7巻）、この150行の上に建ちます。PyTorch という既製品の超高性能版があることは第4章4.5で話したとおりですが、それを解禁する条件と時期は第8巻で——自分の手で建てた装備で行けるところまで行くのが、このシリーズの流儀です。

## まとめ

- 訓練ループの4拍子（第3巻4章）はそのまま、第3拍だけが `loss.backward()` に置き換わった。**訓練ループはモデルから独立した**
- `grad` は累積する仕様なので、毎ステップのリセットが必須（PyTorch の `zero_grad()` に相当する急所）
- 第3巻・第4巻の手導出（MSE、log loss、softmax + cross-entropy）は、すべて autograd が小数第6位まで再現した。手導出の時代は終わり、以後は検算の道具になる
- スカラー `Value` は1個ずつの宿命で遅い（実測 約670倍差）。持ち物を行列に替えた `Tensor`（`tensor_autograd.py`）が第6巻以降の標準装備。matmul の backward には第3章3.3の式がそのまま入っている

**ラスボスとの距離**: 論文 5.3 の "We used the Adam optimizer" の一文のうち、「optimizer が勾配を受け取って W を更新する」という構造が、自分のコードの4拍子として完全に手の中に入りました。残るは Adam という更新規則の中身だけです（第8巻）。

## 演習

**問1** 5.1 の MLP の隠れ層の幅を `hidden=2` と `hidden=32` に変えて訓練し、loss・正解率・所要時間を比べてください。幅2で何が起きますか。

**問2** 3層の MLP（`sizes=[2, 8, 8, 1]`）で 5.1 と同じ訓練を回してください。コードの変更は何文字必要でしたか。同じことを第2章2.4の手導出方式でやるとしたら、何をやり直す必要があったか、1〜2文で述べてください。

**問3** 5.1 のループから「`grad` を 0.0 に戻す2行」を削除して訓練し、何が起きるか観察してください。loss の動きから、混入しているものの正体を推理してください。

<details>
<summary>略解</summary>

**問1** 手元の実測では、`hidden=2` は loss 0.38・正解率 73% で頭打ち、`hidden=32` は正解率 99.0% に達するが所要時間は約23秒（幅8の約4倍）。幅2では折れ線が2本しか使えず、市松模様の4つの塊を囲い切れない（第1章1.4の「折れ線の本数=表現力」の実例）。幅を増やせば表現力は上がるが、スカラー `Value` では計算時間が正直に比例して増える——5.3 の「行列で一斉に」の動機がここでも効きます。

**問2** 変更は `[2, hidden, 1]` を `[2, 8, 8, 1]` にするだけ（10文字前後）。`MLPValue` も訓練ループも `backward()` も一切触りません。手導出方式なら、$\partial L/\partial W_1$ の連鎖律を1層ぶん深く導出し直し（第2章2.4の地獄がもう1段深くなる）、勾配計算のコードも書き直しになります。「モデルの変更が訓練ループに波及しない」ことこそ、自動微分の構造的な恩恵です。

**問3** loss は最初の数ステップこそ下がるものの、やがて振動し、発散します。`grad` に**過去全ステップの勾配の合計**が累積していくため、実質的に学習率が毎ステップ膨らんでいくのと同じことになるからです。「前のステップの勾配が混ざった状態で坂を下る」とどうなるかの実体験として、一度は踏んでおく価値のある地雷です。

</details>
</content>
