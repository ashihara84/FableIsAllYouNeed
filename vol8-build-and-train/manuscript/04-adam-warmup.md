# 第4章 Adam と warmup — 5.3 の完全読解

この章は、シリーズで最も長く待たせた章です。

第3章で訓練ループは完成しました。forward → loss → backward → update の4拍子——第3巻4章から数えて、形は一度も変わっていません。ただし、最後の1拍だけはまだ素朴なままです。私たちの update は、第2巻で覚えた $W \leftarrow W - \eta \cdot \mathrm{grad}_W$、つまり勾配降下法そのものでした。

ところが論文は、その1拍にこう書いています。

> *"We used the Adam optimizer with $\beta_1 = 0.9$, $\beta_2 = 0.98$ and $\epsilon = 10^{-9}$. We varied the learning rate over the course of training, according to the formula:*
> $$lrate = d_{model}^{-0.5} \cdot \min(step\_num^{-0.5},\ step\_num \cdot warmup\_steps^{-1.5})$$
> *This corresponds to increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used warmup_steps = 4000."*
> — Vaswani et al., "Attention Is All You Need", Section 5.3
>
> 訳: 私たちは $\beta_1 = 0.9$、$\beta_2 = 0.98$、$\epsilon = 10^{-9}$ の Adam オプティマイザを使った。learning rate は訓練の経過に応じて、上の式に従って変化させた。これは最初の warmup_steps ステップの間 learning rate を線形に増やし、その後はステップ数の平方根の逆数に比例して減らすことに相当する。warmup_steps = 4000 とした。

後半の lrate の式は、第2巻終章ですでに読みました。山型のスケジュール——最初の 4000 歩は線形に増やす助走、その後は $1/\sqrt{step\_num}$ の減衰。あの読解はいまも有効です。

読めずに残っているのは、最初の一文です。**Adam とは何か。$\beta_1 = 0.9$、$\beta_2 = 0.98$、$\epsilon = 10^{-9}$ という3つのつまみは、何を調整しているのか。** 第2巻終章の「まだ読めないものの棚卸し」で、この一文には「第8巻」という住所を貼りました。シリーズ全体で最も長く放置された宿題が、いまこの章で満期を迎えます。

この章が終わると、セクション 5.3 は一語残らず読めます。そして読めるだけではありません。Adam を自分の手でゼロから書き、PyTorch の `optim.Adam` と1ステップずつ突き合わせて、**完全に同じ軌跡を描くこと**を確認します。さらに warmup を実装し、「もし warmup がなかったら」を実験して、論文がなぜ助走にこだわるのかを自分の損失曲線で目撃します。

## 4.1 SGD の弱点の復習 — 第2巻6.4の伏線回収

第2巻6.4「勾配降下の弱点メモ」で予告した伏線を、ここで回収します。あのとき私たちは2つの弱点を観察し、「処方箋にはモーメンタムと Adam という名前が付いている。中身は第8巻で」と書き残しました。6巻ぶん、待たせました。

弱点を思い出すところから始めます。舞台は細長い谷です。

$$E(w_1, w_2) = w_1^2 + 25\,w_2^2$$

勾配は $(2w_1,\ 50w_2)$ です。$w_2$ 方向(谷を横切る方向)の成分が $w_1$ 方向(谷沿い)の25倍も強いので、勾配はほぼ真横——谷の壁の方——を向きます。第2巻4章で確かめたとおり、勾配は等高線に垂直であり、細長い楕円の等高線に垂直な向きは、谷底へ向かう向きと大きくずれるのです。

ここから2つの弱点が生まれるのでした。

**弱点1: ジグザグ。** $\eta$ を大きめにすると、$w_2$ は毎ステップ谷の壁を跳び越えて反対側へ着地し、左右の往復を繰り返します。進みたいのは $w_1$ 方向なのに、エネルギーの大半が横揺れに費やされる。

**弱点2: 歩幅が全方向共通。** ジグザグの根本治療は「谷を横切る $w_2$ 方向は小股、谷沿いの $w_1$ 方向は大股」と方向ごとに歩幅を変えることです。しかし素の勾配降下法の $\eta$ は1個のスカラーで、全パラメータに同じ歩幅を強制します。第2巻6.3のスケジュールは $\eta$ を時刻によって変えましたが、方向によって変える芸当はできませんでした。

Transformer の訓練では、この「細長い谷」は比喩では済みません。embedding の各行、attention の各重み行列、FFN——数百万個のパラメータは、損失への効き方がてんでばらばらです。頻出単語の embedding には毎バッチ強い勾配が届き、稀な単語にはたまにしか届きません。全員に同じ歩幅を配る optimizer では、誰かに合わせれば誰かが破綻します。

処方箋は2つの部品からできています。弱点1に効くのがモーメンタム(4.2)、弱点2に効くのが勾配の大きさによる自動スケール(4.3)。そして両方を併せ持つのが Adam(4.4)です。1つずつ開けていきます。

## 4.2 モーメンタム — 勾配の移動平均で慣性をつける

ジグザグの正体をもう一度見ます。$w_2$ 方向の勾配成分は、谷の右の壁では正、左の壁では負。ステップごとに**符号が反転**しています。一方 $w_1$ 方向の成分は、ずっと同じ符号で谷の出口を指し続けています。

ここに気づくと、処方箋は半分書けたようなものです。**過去数ステップの勾配を平均してしまえばよい。** 符号が反転し続ける $w_2$ 成分は、平均すればプラスとマイナスが打ち消し合ってほぼゼロになります。符号が揃っている $w_1$ 成分は、平均しても消えずに残ります。横揺れだけが濾し取られ、進むべき向きだけが濾紙に残るのです。

ただし「過去全部の平均」では古すぎる記憶まで均等に効いてしまいます。地形は場所によって変わるので、新しい勾配ほど重く、古い勾配ほど軽く扱いたい。これを1行で実現するのが**指数移動平均(exponential moving average)**です。ステップ $t$ の勾配を $g_t$ として、

$$m_t = \beta_1\, m_{t-1} + (1 - \beta_1)\, g_t$$

$m_t$ が「勾配の移動平均」、$\beta_1$ は 0 と 1 の間の定数です。なぜこれが「新しいほど重い平均」なのかは、漸化式を数ステップ展開すると見えます($m_0 = 0$ から始めて):

$$m_t = (1-\beta_1)\left(g_t + \beta_1\, g_{t-1} + \beta_1^2\, g_{t-2} + \cdots\right)$$

1歩昔にさかのぼるごとに、重みが $\beta_1$ 倍ずつ軽くなっていく加重平均です。$\beta_1 = 0.9$ なら10歩前の勾配の重みは $0.9^{10} \approx 0.35$ 倍、50歩前は $0.005$ 倍。おおまかに「**直近 $1/(1-\beta_1)$ 歩ぶんの記憶**」と読めます。$\beta_1 = 0.9$ で約10歩です。

更新には、生の勾配 $g_t$ の代わりにこの $m_t$ を使います。坂を転がるボールが小さな凹凸で急に向きを変えないように、過去の動きが慣性(モーメンタム)として効く——比喩はこの1つで十分でしょう。実体はあくまで「勾配の指数移動平均」です。

これで弱点1は手当てできました。しかし弱点2——方向ごとの歩幅——にはまだ触れていません。$m_t$ を使っても、それに掛かる $\eta$ は依然として全パラメータ共通です。

## 4.3 RMSProp 的な発想 — 勾配の大きさでスケールを自動調整

弱点2の要求をもう一度言葉にします。「勾配がいつも強い方向($w_2$、勾配 $50w_2$)は歩幅を小さく、いつも弱い方向($w_1$、勾配 $2w_1$)は歩幅を大きくしたい」。

つまり必要なのは、パラメータごとの「**最近の勾配は、だいたいどのくらいの大きさか**」という記録です。大きさを測るのですから、符号を消してから平均します——第4巻2章で分散を作ったときとまったく同じ理屈で、2乗してから移動平均を取ります。

$$v_t = \beta_2\, v_{t-1} + (1 - \beta_2)\, g_t^2$$

$v_t$ は「勾配の2乗の移動平均」、すなわち最近の勾配の大きさの目安です($g_t^2$ は要素ごとの2乗で、$v_t$ はパラメータと同じ shape を持ちます)。そして更新時に、**勾配をこの大きさの平方根で割ります**:

$$W \leftarrow W - \eta \cdot \frac{g_t}{\sqrt{v_t} + \epsilon}$$

何が起きるか、谷の例で確かめます。$w_2$ 方向は勾配が大きいので $v$ も大きく、$\sqrt{v}$ で割られて歩幅が縮みます。$w_1$ 方向は勾配が小さいので $v$ も小さく、割っても大して縮みません。結果として、**どの方向の1歩も、だいたい $\eta$ 程度の大きさに揃います**。勾配のうち「向き」の情報だけを残し、「大きさ」は自動で正規化する——$\eta$ が1個のスカラーのままで、実質的にパラメータごとの歩幅が実現できました。

分母の $\epsilon$ は、ここで初めて顔を出します。勾配がほとんどゼロのパラメータでは $v_t \approx 0$ となり、ゼロ割りが起きます。それを防ぐためだけに足される小さな定数——保険であって、思想ではありません。

このアイデアは RMSProp という名前で知られる手法の核です。最短測地線の原則どおり RMSProp 自体の詳細には立ち入らず、「勾配の2乗の移動平均で歩幅を自動調整する発想」だけを持って先へ進みます。

## 4.4 Adam = 両方 — β1, β2, ε の意味が全部読める

部品は揃いました。4.2の $m_t$(勾配の移動平均=慣性)と、4.3の $v_t$(勾配の2乗の移動平均=スケール)。**Adam はこの2つを単純に併用したものです。** 名前は Adaptive Moment Estimation の略——$m$ と $v$ は統計学の言葉でそれぞれ1次・2次のモーメントと呼ばれ、それを適応的に推定しながら歩くからこの名です。

$$m_t = \beta_1\, m_{t-1} + (1-\beta_1)\, g_t \qquad v_t = \beta_2\, v_{t-1} + (1-\beta_2)\, g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$W \leftarrow W - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

1行目と3行目はもう読めます。読めないのは2行目——$\hat{m}_t$、$\hat{v}_t$ を作っている **バイアス補正(bias correction)** です。なぜ $1 - \beta^t$ などで割るのでしょうか。

### バイアス補正 — 序盤の移動平均は0に偏る

原因は、移動平均の出発点 $m_0 = 0$ にあります。これは便宜上の初期値であって、本当の勾配の平均が0だという意味ではありません。ところが序盤の $m_t$ は、この「根拠のない0」を引きずります。

最初の1歩で確かめましょう。$m_1 = \beta_1 \cdot 0 + (1-\beta_1)\, g_1 = 0.1\, g_1$($\beta_1 = 0.9$ のとき)。直近の勾配の平均を名乗りながら、実際の値は $g_1$ の **1割** しかありません。残りの9割は、初期値0の置き土産です。

これがまぐれでなく系統的な偏りであることは、期待値で言えます。第4巻2章で手に入れた道具——期待値の線形性($\mathbb{E}[aX] = a\,\mathbb{E}[X]$、和の期待値は期待値の和)——を使い、勾配の期待値が当面 $\mathbb{E}[g]$ で一定だとみなして 4.2 の展開式の期待値を取ると:

$$\mathbb{E}[m_t] = (1-\beta_1)\left(1 + \beta_1 + \cdots + \beta_1^{t-1}\right)\mathbb{E}[g] = (1 - \beta_1^t)\,\mathbb{E}[g]$$

(中央の括弧は等比数列の和で $\frac{1-\beta_1^t}{1-\beta_1}$ です。)つまり $m_t$ は平均的に、本来推定したい $\mathbb{E}[g]$ の $(1-\beta_1^t)$ 倍に**必ず**縮んでいます。偏りの倍率が正確にわかっているなら、話は簡単です——**その倍率で割り戻せばよい**。それが $\hat{m}_t = m_t / (1-\beta_1^t)$ の正体です。$t=1$ なら $0.1\,g_1 / 0.1 = g_1$ と全快し、$t$ が大きくなると $\beta_1^t \to 0$ で補正は自然に消えます。$\hat{v}_t$ もまったく同じ理屈です。

補正を怠ると何が起きるかも見ておきます。更新量は比 $\hat{m}/\sqrt{\hat{v}}$ で決まるので、効くのは分子と分母の**縮み方の差**です。$\beta_2$ は $\beta_1$ より1に近いのが通例なので、$v$ の縮み($1-\beta_2^t$)は $m$ の縮み($1-\beta_1^t$)より深刻です。分母の方が余計に縮めば、比は本来より**大きく**なる——つまり補正なしの Adam は、統計がいちばん当てにならない序盤に限って、歩幅を過大にする危険があります(具体的な倍率は演習2で計算します)。

### ラスボスの一文を読む

これで、巻頭の一文の全記号が開きました。読み上げます。

- **Adam optimizer** — 勾配の移動平均 $m$(慣性)と勾配の2乗の移動平均 $v$(自動スケール)を併用し、バイアス補正してから $\hat{m}/(\sqrt{\hat{v}} + \epsilon)$ で歩く optimizer
- **$\beta_1 = 0.9$** — 慣性の記憶の長さ。直近約 $1/(1-0.9) = 10$ 歩の勾配を平均する
- **$\beta_2 = 0.98$** — スケールの記憶の長さ。直近約 $1/(1-0.98) = 50$ 歩で勾配の大きさを見積もる。Adam の原論文が勧める標準値は 0.999(記憶約1000歩)であり、**論文はそれを意図的に短くしています**。なぜか——演習1の題材です
- **$\epsilon = 10^{-9}$** — $\sqrt{\hat{v}}$ がゼロに近いときのゼロ割り防止

最後にもう1つ、この章の後半(4.6)への橋になる観察を。4.3で見たとおり、Adam の1歩の大きさは方向によらずおおむね $\eta$ に揃います。これは便利であると同時に、**$\eta$ の責任が重くなった**ということでもあります。素の勾配降下法では勾配が小さければ歩幅も勝手に小さくなりましたが、Adam は $\eta$ で指定された歩幅を、ほぼ言われたとおりに歩きます。だからこそ論文は、$\eta$ を式で精密に管理するのです——それが lrate の式、warmup の話です。

## 4.5 [コード] Adam をフルスクラッチ実装し、optim.Adam と挙動照合 — 並走はここが唯一の例外

第1章で私たちは自作スタックを卒業し、訓練は PyTorch 一本でやると決めました。並走はしない——維持コストは「論文が読める」に寄与しないからです。**その方針の、シリーズ全体で唯一の例外がこの節です。** optimizer だけはブラックボックスにしません。

理由を述べておきます。optimizer は毎ステップ、**全パラメータを黙って書き換える**部品です。loss や勾配と違って、途中経過が画面に出ることもありません。ここに誤解があると、訓練が壊れたとき(第5章でその診断をやります)に疑える場所がなくなります。逆に、自作 Adam が `optim.Adam` と1ステップも違わず同じ軌跡を描くことを一度確認してしまえば、以後 `optim.Adam` と書くたびに、その中で何が起きているかを正確に言える——第1章1.5の対応表に、自信を持って1行足せます。

4.4の数式を、そのままコードに写します。全文掲載します(`code/ch04/adam_scratch.py`)。

```python
# 第8巻 第4章 4.5: Adam をフルスクラッチ実装し、torch.optim.Adam と軌跡を照合する
# 自作との並走はシリーズ全体でここが唯一の例外 — optimizer だけはブラックボックスにしない
import torch

torch.manual_seed(42)


# ---------------------------------------------------------------
# 自作 Adam(Kingma & Ba 2015, Algorithm 1 をそのまま写す)
# 論文 "Attention Is All You Need" 5.3 の設定: β1=0.9, β2=0.98, ε=1e-9
# ---------------------------------------------------------------
class AdamScratch:
    def __init__(self, params, lr, beta1=0.9, beta2=0.98, eps=1e-9):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.t = 0  # ステップ数(バイアス補正に使う)
        self.m = [torch.zeros_like(p) for p in self.params]  # 1次モーメント(勾配の移動平均)
        self.v = [torch.zeros_like(p) for p in self.params]  # 2次モーメント(勾配の2乗の移動平均)

    def step(self):
        self.t += 1
        with torch.no_grad():
            for p, m, v in zip(self.params, self.m, self.v):
                g = p.grad
                m.mul_(self.beta1).add_((1 - self.beta1) * g)        # m ← β1 m + (1−β1) g
                v.mul_(self.beta2).add_((1 - self.beta2) * g * g)    # v ← β2 v + (1−β2) g²
                m_hat = m / (1 - self.beta1 ** self.t)               # バイアス補正(4.4節)
                v_hat = v / (1 - self.beta2 ** self.t)
                p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = None


# ---------------------------------------------------------------
# 照合実験: 同じ初期値・同じデータ・同じ lr で、
# 自作 Adam と torch.optim.Adam が同じ軌跡を描くか
# モデルは小さな2層 MLP(float64 — 丸め誤差を最小にして厳しく比較する)
# ---------------------------------------------------------------
def make_params(seed):
    g = torch.Generator().manual_seed(seed)
    W1 = (torch.randn(4, 8, generator=g, dtype=torch.float64) * 0.5).requires_grad_()
    b1 = torch.zeros(8, dtype=torch.float64, requires_grad=True)
    W2 = (torch.randn(8, 1, generator=g, dtype=torch.float64) * 0.5).requires_grad_()
    b2 = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    return [W1, b1, W2, b2]


def forward(params, X):
    W1, b1, W2, b2 = params
    return torch.tanh(X @ W1 + b1) @ W2 + b2


# 同一の初期値を持つ2組のパラメータ(seed を揃えてつくる)
params_a = make_params(0)  # 自作 Adam が動かす
params_b = make_params(0)  # torch.optim.Adam が動かす
for pa, pb in zip(params_a, params_b):
    assert torch.equal(pa, pb)  # 出発点は完全に同一

lr = 1e-3
opt_a = AdamScratch(params_a, lr=lr, beta1=0.9, beta2=0.98, eps=1e-9)
opt_b = torch.optim.Adam(params_b, lr=lr, betas=(0.9, 0.98), eps=1e-9)

# 訓練データ(回帰の小問題。中身は何でもよい — 比べたいのは optimizer)
g = torch.Generator().manual_seed(1)
X = torch.randn(64, 4, generator=g, dtype=torch.float64)
y = torch.sin(X.sum(dim=1, keepdim=True))

losses = []
for step in range(50):
    # --- 自作 Adam 側 ---
    opt_a.zero_grad()
    loss_a = ((forward(params_a, X) - y) ** 2).mean()
    loss_a.backward()
    opt_a.step()

    # --- torch.optim.Adam 側 ---
    opt_b.zero_grad()
    loss_b = ((forward(params_b, X) - y) ** 2).mean()
    loss_b.backward()
    opt_b.step()

    # 毎ステップ、全パラメータが一致していることを確認
    for pa, pb in zip(params_a, params_b):
        assert torch.allclose(pa, pb, atol=1e-12), f"step {step+1} で軌跡がずれました"
    losses.append(loss_a.item())

print(f"50ステップ後の loss: 自作 {loss_a.item():.6f} / torch {loss_b.item():.6f}")
print(f"loss の推移(自作): {losses[0]:.4f} -> {losses[9]:.4f} -> {losses[-1]:.4f}")
assert losses[-1] < losses[0]  # ちゃんと学習も進んでいる
print("ok: 自作 Adam と torch.optim.Adam の軌跡が 50 ステップ全てで一致しました(atol=1e-12)")
```

読みどころは2つです。

**`step()` の中身が4.4の数式と1対1対応していること。** $m$ の更新、$v$ の更新、2つのバイアス補正、そして $\hat{m}/(\sqrt{\hat{v}} + \epsilon)$ による更新——5行です。何百万パラメータの Transformer を訓練する optimizer の正体は、この5行の繰り返しに過ぎません。なお `m.mul_(...).add_(...)` の末尾アンダースコアは in-place(その場で書き換える)演算の印で、自作 autograd で言えば「`.data` を直接いじる」操作にあたります(更新は微分の対象ではないので `no_grad` の中で行います)。

**照合の条件を厳密に揃えていること。** 初期値は同じ seed から作って `torch.equal` で確認し、データも loss も共有し、dtype は float64 にして丸め誤差の余地を絞ってあります。比較は最終結果だけでなく**50ステップの毎回**、許容誤差 `atol=1e-12` で行います。実行すると:

```
50ステップ後の loss: 自作 0.616505 / torch 0.616505
ok: 自作 Adam と torch.optim.Adam の軌跡が 50 ステップ全てで一致しました(atol=1e-12)
```

`optim.Adam` の中身は、いま書いたこの5行と同じものです。確認できたので、以後の章では安心して `optim.Adam` を使います——ブラックボックスとしてではなく、中身を一度自分の手で書いた者として。

## 4.6 [コード] warmup スケジュール(第2巻終章の式)の実装 — 切ると訓練が壊れることを実験で確認

残るは lrate の式です。読み方は第2巻終章で完了しています。min の中で「増える候補」と「減る候補」が競い、最初の warmup_steps ステップは線形に増える助走、その後は $1/\sqrt{step\_num}$ の減衰、先頭の $d_{model}^{-0.5}$ は山全体の高さ。今日はそれを**実装し**、そして読解だけでは答えられなかった問いに実験で答えます——**助走は、本当に要るのでしょうか。**

要る理由の見当は、この章ですでについています。4.4の最後に見たとおり、Adam の歩幅はほぼ $\eta$ そのものです。そして訓練の最初の数十歩では、$v$ はまだ数十個の標本から作った頼りない見積もりに過ぎず、パラメータは初期値のまま、勾配のスケールも場所によって荒れています。**いちばん統計が当てにならない時期に、いちばん歩幅を慎重にする**——それが warmup の役割のはずです。本当にそうか、切って確かめます。

実験対象は、論文と同じ構成(post-LN・残差接続・単一ヘッド attention・FFN)の縮小版 Transformer に、ごく簡単なコピータスク(入力列をそのまま出力する)を学ばせるものです。比べたいのはスケジュールの有無だけなので、初期値・データ・Adam の設定は両者で完全に揃えます。warmup_steps は規模に合わせて 4000 から 100 に縮めます。


全文掲載します(`code/ch04/warmup_experiment.py`)。

```python
# 第8巻 第4章 4.6: warmup スケジュール(第2巻終章の式)の実装と、
# 「warmup を切ると訓練が壊れる」ことの実験的確認
#
# 実験には自前の極小 MiniTransformer を使う(第3章 model.py と同じ post-LN 構成。
# 本章単体で再現できるよう独立させてある。本番モデルでの訓練は第5章)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------
# 論文 5.3 の learning rate スケジュール(第2巻終章で読んだ式そのまま)
# lrate = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
# ---------------------------------------------------------------
def lrate(step, d_model, warmup_steps):
    return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)


def lrate_no_warmup(step, d_model, warmup_steps):
    # warmup を切る = min の中の「増える候補」を消し、減衰項だけを残す
    # step=1 から d_model^{-0.5} = 0.125 という大股で歩き出すことになる
    return d_model ** -0.5 * step ** -0.5


# 交差点の確認(第2巻終章の表の再現): step = warmup_steps で2候補が一致する
assert math.isclose(lrate(100, 64, 100), 64 ** -0.5 * 100 ** -0.5)
assert lrate(1, 64, 100) < lrate(100, 64, 100)    # 序盤は小さい(助走)
assert lrate(400, 64, 100) < lrate(100, 64, 100)  # 終盤は減衰


# ---------------------------------------------------------------
# 実験対象の小さな Transformer(post-LN・単一ヘッド・2層)
# 論文と同じ構成の縮小版。完成品モジュール(nn.Transformer 等)は使わない
# ---------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # 式(1)
        x = self.ln1(x + self.wo(F.softmax(scores, dim=-1) @ v))   # post-LN(論文の構成)
        x = self.ln2(x + self.ff2(torch.relu(self.ff1(x))))
        return x


class MiniTransformer(nn.Module):
    def __init__(self, vocab=32, seq_len=16, d_model=64, d_ff=128, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.zeros(seq_len, d_model))  # 学習する位置埋め込み(縮小版の簡略化)
        self.blocks = nn.ModuleList([Block(d_model, d_ff) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab)

    def forward(self, tokens):  # tokens: (batch, seq_len)
        x = self.embed(tokens) + self.pos
        for block in self.blocks:
            x = block(x)
        return self.head(x)  # (batch, seq_len, vocab)


# ---------------------------------------------------------------
# 訓練: コピータスク(入力列をそのまま出力する)。題材は何でもよい —
# 比べたいのはスケジュールの有無だけなので、データと初期値は両者で完全に揃える
# ---------------------------------------------------------------
VOCAB, SEQ_LEN, D_MODEL = 32, 16, 64
STEPS, WARMUP = 400, 100


def train(schedule_fn):
    torch.manual_seed(42)  # 初期値を両者で揃える
    model = MiniTransformer(vocab=VOCAB, seq_len=SEQ_LEN, d_model=D_MODEL)
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    g = torch.Generator().manual_seed(7)  # データも両者で揃える

    losses = []
    for step in range(1, STEPS + 1):
        for group in opt.param_groups:  # 毎ステップ lr を式から計算し直す(5.3 の実装)
            group["lr"] = schedule_fn(step, D_MODEL, WARMUP)
        tokens = torch.randint(0, VOCAB, (64, SEQ_LEN), generator=g)
        logits = model(tokens)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), tokens.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


print("--- warmup あり(論文 5.3 の式そのまま)---")
losses_w = train(lrate)
print(f"loss: step1 {losses_w[0]:.3f} -> step100 {losses_w[99]:.3f} -> step400 {losses_w[-1]:.3f}")

print("--- warmup なし(min の増える候補を消した式)---")
losses_n = train(lrate_no_warmup)
print(f"loss: step1 {losses_n[0]:.3f} -> step100 {losses_n[99]:.3f} -> step400 {losses_n[-1]:.3f}")

# ---------------------------------------------------------------
# 確認: warmup ありは学習が成立し、なしは壊れる
# (この規模では NaN にはならず「序盤に何も学べず、最後まで桁違いに悪い」
#  という壊れ方をする。発散するかどうかは規模と運に依る — 本文 4.6 参照)
# ---------------------------------------------------------------
baseline = math.log(VOCAB)  # 当てずっぽうの loss = ln 32 ≈ 3.47
final_w, final_n = losses_w[-1], losses_n[-1]

# warmup あり: タスクをほぼ完全に解いている
assert final_w < 0.1, f"warmup ありの訓練が想定どおり進んでいません: {final_w}"
# warmup なし: 最初の100ステップは当てずっぽう近傍から動けない
assert math.isnan(losses_n[99]) or losses_n[99] > baseline * 0.9, \
    f"warmup なしが序盤から学習できてしまいました: {losses_n[99]}"
# warmup なし: 400ステップ後も loss が桁違いに高いまま(または NaN)
assert math.isnan(final_n) or final_n > 1.0, \
    f"warmup なしでも学習できてしまいました: {final_n}"

print(f"\n当てずっぽうの loss(ln {VOCAB}): {baseline:.3f}")
print(f"warmup あり: {final_w:.4f}(タスクをほぼ解いた)")
print(f"warmup なし: {final_n if math.isnan(final_n) else round(final_n, 4)}(壊れた — 序盤の損傷を最後まで引きずる)")
print("ok: warmup を切ると訓練が壊れる(桁違いに悪化する)ことを確認しました")
```

実装面の読みどころは2つだけです。スケジュールは関数 `lrate(step, d_model, warmup_steps)` として式を1行で写し、訓練ループの先頭で**毎ステップ lr を計算し直して** optimizer に書き込みます(`param_groups` は `optim.Adam` が設定をまとめて持つ場所です)。そして「warmup を切る」の定義は、min の中の増える候補を消すこと——第2巻終章の言葉で言えば、助走のスイッチを外して、減衰項だけを最初から使うことです。このとき step 1 の lr は $64^{-0.5} \cdot 1^{-0.5} = 0.125$。warmup ありの初手(約 0.000125)の**1000倍**の大股で歩き出すことになります。

実行結果がこちらです。

```
--- warmup あり(論文 5.3 の式そのまま)---
loss: step1 3.510 -> step100 0.001 -> step400 0.000
--- warmup なし(min の増える候補を消した式)---
loss: step1 3.510 -> step100 3.486 -> step400 2.481
```

数字を読みます。当てずっぽう(32種から一様に当てる)の loss は $\ln 32 \approx 3.47$ です。warmup ありは、100ステップでタスクをほぼ完全に解いています(loss 0.001)。warmup なしは、**100ステップたっても当てずっぽうの水準(3.486)から動けていません**。序盤の大股が、初期値の周りの構造——embedding も attention の重みも——を踏み荒らしてしまい、学習が始まらないのです。

注目すべきは、その後です。warmup なしの lr は $1/\sqrt{step}$ で減り続けるので、step 100 では 0.0125 と、warmup ありの頂点と同じ高さまで下がっています。**ここから先の条件はほぼ同じ**なのに、step 400 の loss は 2.48——同じ時点で 0.000 に達した warmup ありとは桁違いのままです。序盤に壊した分を、後から取り返せていません。「最初の数歩の事故が、その後の訓練全体に祟る」——第2巻6.3で4手で発散する実験を見たときの教訓が、Transformer でも(より陰湿な形で)成り立っています。

1つ、正直に書いておきます。この縮小実験での壊れ方は「loss が NaN に飛ぶ派手な発散」ではなく「高止まりして回復しない」型でした。発散するかどうかはモデルの規模・初期値・タスクに依ります。論文規模(数千万パラメータ、長い訓練)では発散や訓練の不安定化として現れることが知られており、だからこそ論文は warmup_steps = 4000 をわざわざ本文に明記しているのです。どちらの型にせよ、結論は同じです——**この式から min を抜いてはいけません。**

これでセクション 5.3 は、読めて、書けて、切ったらどうなるかまで知っている状態になりました。

## まとめ

- 素の勾配降下法の弱点(第2巻6.4の伏線)は2つ: 谷でのジグザグと、全方向共通の歩幅。前者にはモーメンタム $m_t$(勾配の指数移動平均=慣性)、後者には $v_t$(勾配の2乗の移動平均)による自動スケールが効く
- **Adam は両方の併用**。$\beta_1, \beta_2$ はそれぞれ $m, v$ の記憶の長さ(約 $1/(1-\beta)$ 歩)、$\epsilon$ はゼロ割り防止。初期値0に引きずられる序盤の偏りは、期待値の議論(第4巻2章)から倍率 $1-\beta^t$ が正確にわかるので、割り戻して補正する
- 自作 Adam は `torch.optim.Adam` と50ステップ全てで一致した(atol=1e-12)。並走はシリーズでここが唯一の例外——optimizer だけはブラックボックスにしない
- Adam の歩幅はほぼ $\eta$ そのものなので、$\eta$ の管理が重要になる。warmup スケジュールは「統計が当てにならない序盤ほど慎重に歩く」仕掛けで、切ると訓練は序盤に壊れ、後から回復できない

**ラスボスとの距離**: Section 5.3 を一語残らず読了。Section 5 で残るのは 5.2 のハードウェアの数字の読み方だけになり、それは第6章の Table 2(training cost)でまとめて回収します。

## 演習

**問1** 4.6 の実験を `betas=(0.9, 0.999)`(Adam 標準)と `betas=(0.9, 0.98)`(論文)で実行し、loss 曲線を比較してください。そのうえで、論文がなぜ標準の 0.999 ではなく 0.98 を選んだのか、$1/(1-\beta_2)$ の意味から考察してください。

**問2** バイアス補正を**切った** Adam の最初の1歩を手計算してください。$m_1 = (1-\beta_1)g_1$、$v_1 = (1-\beta_2)g_1^2$ から、補正なしの更新量と補正ありの更新量の比($\epsilon$ は無視)を $\beta_1 = 0.9$ と $\beta_2 = 0.98$ の場合、$\beta_2 = 0.999$ の場合について求めてください。

**問3** 4.6 の実験で `WARMUP` を 25 と 400 に変えて実行してください。スケジュールの頂点の高さ $d_{model}^{-0.5} \cdot warmup\_steps^{-0.5}$ を先に計算してから、結果を予想して確かめること。

<details>
<summary>略解</summary>

**問1** この章の縮小実験では、両者の差はほとんど出ません(どちらも loss ≈ 0 までタスクを解きます。手元の実行では最終 loss 0.0001 と 0.0003)。差が問題になるのは論文規模の訓練です。$1/(1-\beta_2)$ は「勾配の大きさを何歩ぶんの記憶で見積もるか」でした——0.999 なら約1000歩、0.98 なら約50歩です。論文の訓練では lrate がスケジュールで動き続け、warmup 中はわずか4000歩で歩幅が1000倍変わり、データも巨大で勾配の景色が変わり続けます。1000歩前の勾配のスケールを引きずる $v$ は「現在の」適正歩幅を見誤りやすく、まれに来る巨大勾配の記憶も長く残りすぎます。記憶を50歩に縮めれば、$v$ は現在の地形に機敏に追従します。つまり 0.98 は「長い訓練を、変化し続ける条件下で安定に走り切る」ための選択です(第6章で読む Table 3 のような感度実験を、著者たちは optimizer に対しても行っていたと推測できます)。

**問2** 更新量の比(補正なし ÷ 補正あり)は $\frac{m_1/\sqrt{v_1}}{\hat{m}_1/\sqrt{\hat{v}_1}} = \frac{1-\beta_1}{\sqrt{1-\beta_2}}$ です。$\beta_2 = 0.98$ なら $0.1/\sqrt{0.02} \approx 0.71$ 倍——やや控えめになる程度。$\beta_2 = 0.999$ なら $0.1/\sqrt{0.001} \approx 3.16$ 倍——いきなり指定の3倍超の大股です。4.4で「分母の方が余計に縮むので歩幅が過大になる」と述べた現象が、標準設定ほど深刻に出ることがわかります。バイアス補正は、$\beta_2$ が1に近いほど命綱になります。

**問3** 頂点の高さは WARMUP = 25 で $0.125 \times 0.2 = 0.025$、100 で 0.0125、400 で 0.00625。短いほど高い山を、少ない助走で駆け上がることになります。手元の実行では、25 はこの規模ではまだ安全圏で(0.025 は 4.6 で事故を起こした 0.125 の5分の1)、高い頂点のぶん収束はむしろ最速でした——ただし頂点直後に loss が小さく跳ね返る兆候が観察できます。400 は序盤が明確に遅く(step 50 時点で loss 約 0.99 — 25 の場合は同時点でほぼ 0)、400 ステップかけてようやく追いつきます。つまり「助走の長さ = 安全と速度のトレードオフ」で、短くするほど速いが頂点が高くなり、どこかで 4.6 の「warmup なし」と同じ事故の領域に入ります。論文の 4000 は、論文の規模でこの綱渡りが成立する長さとして選ばれた値です。

</details>
