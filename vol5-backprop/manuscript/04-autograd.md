# 第4章 ミニ autograd を自作する(クライマックス)

> [目次](../TOC.md) ・ [← 前の章](03-backprop.md) ・ [次の章 →](05-training-with-autograd.md)

ここまでで2つのことがわかりました。勾配の手導出は地獄だということ(第2章)。それでも backprop の**原理**は驚くほど単純で、各ノードは「自分の局所勾配 × 上流から来た勾配」を下流に渡すだけだということ(第3章)。

原理が単純で手作業が地獄。これは自動化の出番です。第2巻5章の最後に「演算ごとに局所微分を1回ずつ実装し、グラフを逆にたどる処理を共通化すれば、backward は自動で書ける。第5巻で回収します」と予告しました。この章がその回収です。これから書く100行強のファイルは練習問題ではありません。**次章のMLPの学習も、第6巻の言語モデルも、第7巻の attention の検証も、この章の `Value` クラスの上で動きます。**

## 4.1 設計方針: 値と勾配と「親と演算」を覚える Value クラス(micrograd 相当、100行強)

自動化の前に、第3章で**手で**やっていたことを観察します。

- forward の途中の値を、あとで使うために覚えておいた(掛け算の局所勾配は「相方の値」だから)
- どの値がどの値から計算されたか——グラフの形——を図に描いて把握していた
- 出力から入力へ、逆順に「局所勾配 × 上流の勾配」を流した

つまり必要な情報は、(1) 各ノードの**値**、(2) 流れてきた**勾配**、(3) その値が**どの親からどの演算で**生まれたか。必要な手続きは、(4) 演算の種類ごとの局所勾配の流し方と、(5) グラフを逆順にたどる仕組みです。

設計の核心は、**forward の計算をしながら、同時にグラフを記録してしまう**ことです。普通の電卓は `2 * 3 + 2` から `8` だけを返して途中経過を捨てますが、私たちが作るのは履歴機能付きの電卓です。`8` に「自分は 6 と 2 の加算で生まれた。6 は 2 と 3 の乗算で生まれた」という出生記録を添えて返す。backward はこの記録を逆にたどるだけです。

これを Python で自然に書ける仕掛けが**演算子オーバーロード(operator overloading)**です。`a + b` は実は `a.__add__(b)` というメソッド呼び出しなので、自作クラスに `__add__` を定義して「足し算しつつ、記録も残す」という意味に上書きすれば、利用者は普通の数式を書くだけで裏でグラフが組み上がります。

このクラスを `Value` と名付けます。設計は、Andrej Karpathy 氏が2020年に公開した約100行の教育用 autograd「**micrograd**」を下敷きにしています。

骨格です。ここから 4.3 節の終わりまで、本文のコードを順につなげると `code/ch04/micrograd.py` が完成します。

```python
# 第5巻 第4章: ミニ autograd — Value クラス(シリーズの基盤モジュール)
# 第5巻5章・第6巻・第7巻・第8巻1章がこのファイルを import する。
# 依存は標準ライブラリのみ。本文(4.1〜4.3節)のコードを順につなげたもの
# + 演習問1の tanh(後の巻で使うため同梱)。
import math


class Value:
    """スカラー1個の値 .data と勾配 .grad を持ち、計算グラフを記録するノード"""

    def __init__(self, data, _parents=(), _op=""):
        self.data = float(data)
        self.grad = 0.0                  # まだ何も流れてきていない
        self._backward = lambda: None    # 葉ノード(入力・パラメータ)は何もしない
        self._parents = set(_parents)    # この値を作るのに使われたノード
        self._op = _op                   # この値を作った演算(デバッグ用の名札)

    def __repr__(self):
        return "Value(data={}, grad={})".format(self.data, self.grad)
```

`data` が値、`grad` が勾配で、先ほどの (1) と (2) です。`grad` は「まだ何の勾配も届いていない」`0.0` で始まります。`_parents` と `_op` が (3) の出生記録で、`Value(2.0)` のように直接作った値は親を持ちません。これは**葉ノード**——入力データやパラメータ $W, b$ に当たります。

見慣れないのは `_backward` です。「上流から届いた勾配を、自分の親たちにどう配るか」を知っている小さな関数で、(4) に当たります。配り方は演算ごとに違うので、初期値は「何もしない」にしておき、**演算が新しい `Value` を作るたびに、その演算専用の配り方を添える**ことにします。

## 4.2 [コード] 演算の実装: 加算・乗算・ReLU・exp・log… 各演算に backward を1個ずつ生やす

最初の演算は加算です。これが全演算の雛形になるので、丁寧に読みます。

```python
    # --- 基本演算: forward で値を計算し、backward(局所勾配の伝え方)を1個ずつ生やす ---

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad        # 加算の局所勾配はどちらの入力にも 1
            other.grad += out.grad       # += なのは「パスが複数あれば足す」ため
        out._backward = _backward
        return out
```

3つの仕事をしています。forward(値を計算)、記録(`out` に親 `(self, other)` と演算名を持たせる)、backward の準備(`out` の勾配が確定したときの配り方を `_backward` として `out` に添える)。加算の局所勾配はどちらの入力にも $1$(第2巻5章)なので、`out.grad` をそのまま両親に渡します。1行目の `isinstance` は、`x + 2` のように普通の数と混ぜて書けるよう相手を `Value` に包む配慮です。

`=` ではなく `+=` なのは、第2巻5章のルール——**パスが複数あれば、足す**——の実装です。`y = x + x` では `x` からのパスが2本あり勾配は両方の合計でなければなりませんが、`=` だと2本目が1本目を上書きしてしまいます。この `+=` 1文字が、枝分かれのあるグラフを正しく扱う仕掛けのすべてです。

乗算も同じ雛形で、局所勾配だけが違います。

```python
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad   # 乗算の局所勾配は「相方の値」
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
```

掛け算ノードの局所勾配は**相方の値**——第2巻5章の結論そのままです。backward で必要になる forward の値はノード自身が `data` として覚えているので、「途中の値を覚えておく」という手作業は設計に自然に溶けました。

2つの演算が入ったところで動かしてみます(この動作確認は `micrograd.py` には含めません)。

```python
from micrograd import Value

a = Value(2.0)
b = Value(3.0)
e = a * b              # この瞬間 e に乗算用の _backward が添えられる
f = e + a              # a は2つの演算に使われている(パスが2本)

f.grad = 1.0           # 出力自身の勾配 df/df = 1 を種として置く
f._backward()          # 加算が配る: e.grad = 1.0, a.grad = 1.0
e._backward()          # 乗算が配る: a.grad += 3.0, b.grad += 2.0
print(a.grad, b.grad)  # 4.0 2.0
```

$f = ab + a$ ですから $\partial f/\partial a = b + 1 = 4$、$\partial f/\partial b = a = 2$。正しい値です。`a.grad` が `1.0 + 3.0` の合計になっているのが `+=` の仕事です。

ただし、いまは `_backward` を**手で、正しい順に**呼びました。`e._backward()` を先に呼ぶと `e.grad` が `0.0` のままなので乗算ノードは親たちに $0$ を配り、答えは狂います。**あるノードの `_backward` は、自分より下流の処理がすべて終わり `grad` が確定してから呼ばなければならない。** この順序問題は 4.3 節で解決することにして、先に演算のレパートリーを揃えます。

べき乗は、第2巻1章のルール $(x^k)' = kx^{k-1}$ をそのまま局所勾配にします。

```python
    def __pow__(self, k):
        # 指数は定数のみ。Value 同士の累乗は exp/log で書ける(演習問3)
        assert isinstance(k, (int, float)), "指数は int か float の定数のみ"
        out = Value(self.data ** k, (self,), "**{}".format(k))

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad   # (x^k)' = k x^(k-1)
        out._backward = _backward
        return out
```

指数 `k` は普通の数に限定しました(`assert` で明示)。指数側にも `Value` が来る $x^y$ は既存の演算の組み合わせで書けるので専用実装は不要です(演習問3)。

次は引き算と割り算ですが、ここで autograd の旨味が最初に現れます。**新しい backward を1つも書かずに**済むのです。

```python
    # --- 派生演算: 上の3つの組み合わせ。backward は書かなくても自動で正しくなる ---

    def __neg__(self):                   # -x
        return self * -1

    def __sub__(self, other):            # x - y
        return self + (-other)

    def __truediv__(self, other):        # x / y
        return self * other ** -1
```

$x - y$ は $x + (-1) \times y$、$x/y$ は $x \times y^{-1}$。既存の演算で式を書けばグラフは加算・乗算・べき乗のノードで自動的に組まれ、backward も正しく流れます。**部品の局所勾配さえ正しければ、組み合わせは何も足さなくても正しい**——深いネットを安心して組める理由の縮図です。

実用の小物をもう1組。`2 + x` では Python はまず `(2).__add__(x)` を試み、`float` は `Value` の足し方を知らないので失敗し、**右側の**オブジェクトの `__radd__` に救済を求めます。そこで右側用も定義しておきます。

```python
    # --- 右側演算子: 2.0 + x のように左辺が普通の数のとき Python が呼ぶ ---

    def __radd__(self, other):           # 定数 + x
        return self + other

    def __rmul__(self, other):           # 定数 * x
        return self * other

    def __rsub__(self, other):           # 定数 - x
        return (-self) + other

    def __rtruediv__(self, other):       # 定数 / x
        return self ** -1 * other
```

最後に非線形関数です。まず、論文の式(2)の主役 $\max(0, x)$——ReLU(第1章)。

```python
    # --- 非線形関数 ---

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out
```

ReLU の局所勾配はゲートです。入力が正なら勾配を素通しし($\times 1$)、負なら遮断する($\times 0$)。折れ目の $x = 0$ では微分が定義されませんが、実装では $0$ と決め打ちするのが慣習です(計算でぴったり $0$ に当たることは実質なく、どちらでも学習は変わりません)。

`exp` と `log` は、第4巻の softmax と cross-entropy を `Value` で書くために必要です。

```python
    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad     # (e^x)' = e^x。forward の結果を再利用
        out._backward = _backward
        return out

    def log(self):
        # 自然対数。定義域は data > 0(損失関数で使うときは中身を正に保つこと)
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad   # (log x)' = 1/x
        out._backward = _backward
        return out
```

`exp` の backward に注目してください。$(e^x)' = e^x$、つまり局所勾配は **forward の出力そのもの**なので、`out.data` を再利用するだけで済みます。

第1章の活性化関数のうち、tanh だけがまだありません。雛形は完全に出揃ったので、演習問1としてあなたに任せます(後の巻で使うため、`micrograd.py` には略解の実装を同梱してあります)。

## 4.3 [コード] トポロジカルソートと backward(): グラフを逆順にたどる

残る仕事は1つ、4.2節で保留にした順序問題です。各ノードの `_backward` を、どの順番で呼ぶか。

条件は言葉にしてあります。「自分より下流の処理がすべて終わってから」。つまりノードたちを**依存関係を壊さない一列**——どのノードも、自分を材料に作られたノードより後ろに来る並び——に整列させ、出力側から逆にたどればよい。この整列操作を**トポロジカルソート(topological sort)**といいます。どの科目も先修科目より後に来るように1本の時間割に並べる、あの操作です。

実装は、グラフを深さ優先でたどりながら「**親を全員リストに積み終えてから、自分を積む**」だけです。リストの先頭側に葉が、末尾に出力が並ぶので、それを逆順に歩きます。

```python
    # --- backward: グラフをトポロジカルソートし、出力から逆順に勾配を流す ---

    def backward(self):
        topo = []                        # 「親が必ず自分より前に来る」順のノード列
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v._parents:
                    build(p)             # 親を先に積む
                topo.append(v)           # 親が全員積まれてから自分を積む
        build(self)

        self.grad = 1.0                  # dL/dL = 1 から伝播を始める
        for v in reversed(topo):         # 出力側から葉に向かって逆順に
            v._backward()
```

`visited` 集合は同じノードを二度積まないための番人です。経路が複数あっても積まれるのは1回だけ——勾配の合算は `_backward` 内の `+=` がすでに引き受けています。`self.grad = 1.0` は伝播の**種**($\partial L/\partial L = 1$)で、4.2節で手で書いた `f.grad = 1.0` をここに引き取りました。

これで `Value` クラスは完成です。仕上げに、第2巻5章で**手で** backward を書いた、あのひし形のグラフ $z = (x + y) \cdot x$ をもう一度流してみます。

```python
from micrograd import Value

x = Value(2.0)
y = Value(3.0)
z = (x + y) * x        # 第2巻5章の「枝分かれのあるグラフ」そのもの
z.backward()
print(z.data)          # 10.0
print(x.grad, y.grad)  # 7.0 2.0
```

`7.0` と `2.0`。第2巻5章の `backward_diamond` が返したのと同じ数です。あのときはパスを目で数え、局所微分を1本ずつ掛けて足しました。いまは `z.backward()` の1行で、しかもこの1行は、ひし形だろうと、次章の数百ノードのMLPだろうと、第7巻の attention だろうと、グラフの形を一切問いません。

注意を1つ。`backward()` をもう一度呼ぶと、`+=` の仕様により勾配が**前回の上に積み増し**されます。学習の反復では毎回 `grad` を $0$ に戻す必要があります——次章の訓練ループで片付けます。

ここまでの本文のコードを順につなげたもの(+演習問1の tanh)が `code/ch04/micrograd.py`、全131行です。依存は標準ライブラリの `math` だけで、NumPy すら使っていません。

## 4.4 [コード] テスト: すべての演算を数値微分と照合する(第2巻からの習慣の集大成)

書き上げた直後の autograd を信用してはいけません。`_backward` のどれか1つに符号ミスがあっても、コードは黙って**もっともらしい嘘の勾配**を返します。学習はなんとなく進み、しかし収束は悪く、原因はどこにも表示されない——勾配のバグは、最も発見しにくい部類のバグです。

幸い、私たちには検算機があります。第2巻1章で作った中心差分です。

$$\frac{df}{dx} \approx \frac{f(x+h) - f(x-h)}{2h}$$

新しい微分が出てくるたび、私たちは必ずこれと突き合わせてきました(第2巻1章で手計算の微分を、第2巻5章で計算グラフの伝播を、第3章の演習で2層MLPの backprop を)。**この習慣の集大成が、いまここです。** autograd は「すべての微分」を生成する機械ですから、演算を1つずつ検算機にかければ機械ごと検証できます。

テストの構造には小さな妙があります。backward は疑わしいが、forward はただの浮動小数演算なので信頼してよい。そこで同じ式 `f` を2役で使い、`Value` を入れて backward で勾配を出し、普通の数を入れて forward だけを中心差分にかける。一致すれば backward は forward と整合しています。骨組みはこうです(これ自体が実行可能な縮約版です)。

```python
# 4.4節: 数値微分との照合の骨組み(完全版は code/ch04/test_micrograd.py)
import math
from micrograd import Value


def numerical_diff(f, x, h=1e-5):
    """中心差分による数値微分(第2巻1章): (f(x+h) - f(x-h)) / 2h"""
    return (f(x + h) - f(x - h)) / (2 * h)


def check_unary(f, xs, name):
    """backward() の勾配と、forward だけを使った数値微分が、全点で一致するか"""
    for x in xs:
        v = Value(x)
        f(v).backward()
        num = numerical_diff(lambda t: f(Value(t)).data, x)
        assert math.isclose(v.grad, num, rel_tol=1e-6, abs_tol=1e-6), (name, x)
    print("ok:", name)


pts = [-2.0, -0.7, 0.5, 1.3, 3.0]    # relu の折れ目 0 は外してある
check_unary(lambda v: v + v, pts, "__add__ (同じ変数を2回 → 勾配 2)")
check_unary(lambda v: v * v * v, pts, "__mul__ (パス3本の合流 → 3x^2)")
check_unary(lambda v: v.relu(), pts, "relu")
check_unary(lambda v: (v ** 2 + 1.0).log().exp() / 2.0, pts, "合成式")
```

完全版の `test_micrograd.py` は、この `check_unary` で**全演算**——`__add__` から `tanh` まで、右側演算子も含めて18本——を1つずつ照合し、さらに全演算入りの深い合成式と、2変数のひし形グラフ(解析解とも照合)を確かめます。評価点にも意味があります。ReLU の折れ目 $x=0$ は外す(微分が定義されず、中心差分もずれた値を返す)。`log` には正の点だけ、除算には $0$ 以外だけ。検算機にも定義域があるのです。

最後のブロックが本当の集大成です。$2 \to 2 \to 1$ の小さなMLPの二乗誤差損失で、`backward()` を**1回**呼ぶだけで9個のパラメータ全部の勾配が出て、その9個すべてが数値微分と一致する。第2章で地獄だった導出が、ここでは1行です。

```
$ python3 test_micrograd.py
ok: __add__ (Value + 定数)
（中略 — 全演算・合成式・2変数の ok が並ぶ）
ok: 2層MLPの全9パラメータの勾配が数値微分と一致

ok: すべての assert を通過しました
```

ならば数値微分だけ使えばいいのでは、と思った人へ。中心差分はパラメータ**1個につき** forward を2回要求します(100万個なら200万回)。backward はグラフ1往復で全部出す。数値微分は本番には遅すぎ、検算には最高の道具です。この使い分けは今後も変わりません。

## 4.5 これが PyTorch の loss.backward() の正体(構造は同じ、規模が違うだけ)

種明かしをします。あなたがいま書き、テストを通したこの131行は玩具ではありません。**PyTorch で `loss.backward()` と書いたとき、その裏で起きていることと構造的に同じもの**です。対応表を見てください。

| この章の micrograd | PyTorch |
|---|---|
| `Value(2.0)` | `torch.tensor(2.0, requires_grad=True)` |
| `.data` / `.grad` | `.data` / `.grad`(名前まで同じ) |
| 演算ごとの `_backward` クロージャ | 演算ごとの `grad_fn`(`AddBackward0` など) |
| `_parents` | `grad_fn.next_functions`(グラフの記録) |
| `backward()` のトポロジカルソートと逆走 | autograd エンジン(C++ 実装) |
| `grad` を 0 に戻してから次の backward | `optimizer.zero_grad()` が必要な理由 |

設計図のレベルでは対応しない部品が1つもありません。PyTorch で `backward()` を呼ぶたび `.grad` が積み増しされ、毎ステップ `zero_grad()` を呼ばされるのも、私たちと同じ「パスは足す」の帰結です。

違いはありますが、規模の違いです。`Value` はスカラー1個を運び、`torch.Tensor` は行列を丸ごと運ぶ(次章5.3節で私たちも同じ拡張をします)。演算は十数種類対数千種類、Python 対 C++ と GPU カーネル。けれど増えているのは品揃えと速度であって、仕組みではありません。**構造は同じ、規模が違うだけ。**

この一文は、シリーズの今後にとって特別な意味を持ちます。第8巻2章で、私たちは自作スタックから PyTorch に乗り換えます。そのとき `loss.backward()` は魔法の呪文ではなく、中身を語れる1行です。**ブラックボックスを使うのではなく、自分が一度作ったものの工業製品版を使う。** だからこのシリーズは第8巻で安心して PyTorch を解禁でき、その根拠は今日のこの章です。

## まとめ

- `Value` は値 `.data`・勾配 `.grad`・出生記録(`_parents`, `_op`)・配り方 `_backward` を持つ計算グラフのノード。演算子オーバーロードにより、**普通の数式を書くだけでグラフが記録される**
- 各演算は「局所勾配 × 上流の勾配」を親に配る `_backward` を1個ずつ持つ。合流は `+=` で**足す**(第2巻5章のルールの実装形)。引き算・割り算は既存演算の組み合わせで済み、backward を書く必要すらない
- `backward()` はトポロジカルソートでノードを依存順に並べ、出力の勾配 $1$ を種に逆順へ流す。グラフの形は問わず、1往復で**全**パラメータの勾配が揃う
- 正しさの保証は数値微分との照合(第2巻1章の中心差分)——シリーズ開始以来の習慣の集大成
- PyTorch の `loss.backward()` は、これと**構造は同じ、規模が違うだけ**。第8巻での PyTorch 解禁はこの章が根拠になる

**ラスボスとの距離**: 論文に載っているのは式(2)をはじめ forward の式だけで、勾配の式は1本もありません——backward は autograd が自動で導くものだからです。今日からあなたの手元にも、その autograd があります。

## 演習

**問1**(tanh を追加する — 本章のメイン演習)`Value` に `tanh()` メソッドを backward 付きで追加してください。forward は `math.tanh`、微分は第1章で出した $\tanh'(x) = 1 - \tanh^2(x)$ です。`exp` と同じ「forward の結果の再利用」が使えます。追加したら、必ず 4.4 節の流儀で数値微分と照合すること。

**問2**(sigmoid を2通りで)$\sigma(x) = \dfrac{1}{1 + e^{-x}}$(第3巻)を、(a) 既存演算の組み合わせ、(b) $\sigma' = \sigma(1 - \sigma)$ を局所勾配に使う専用メソッド、の2通りで実装し、(a)・(b)・数値微分の3つが一致することを確かめてください。

**問3**(`__pow__` が定数指数しか受けない理由)指数も `Value` である $x^y$($x > 0$)は、恒等式 $x^y = e^{y \log x}$ を使えば既存の演算だけで書けます。勾配を `backward()` で求め、解析解 $\partial/\partial x = y x^{y-1}$、$\partial/\partial y = x^y \log x$ と一致することを確かめてください。

<details>
<summary>略解</summary>

**問1** `log` の後ろに次を追加します(`micrograd.py` に同梱済みの実装と同じものです)。

```python
    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad
        out._backward = _backward
        return out
```

局所勾配 $1 - t^2$ は forward の結果 `t` だけで書けます。照合は `check_unary(lambda v: v.tanh(), pts, "tanh")` で、4.4節の完全版テストに含まれています。

**問2** (a) は1行です((b) は問1と同じ雛形で、`_backward` を `self.grad += s * (1.0 - s) * out.grad` とするだけ)。

```python
def sigmoid_combo(v):
    return 1.0 / (1.0 + (-v).exp())

for xv in [-2.0, 0.0, 1.0]:
    v = Value(xv)
    sigmoid_combo(v).backward()
    s = 1 / (1 + math.exp(-xv))
    assert math.isclose(v.grad, s * (1 - s), rel_tol=1e-6)   # 解析解 σ(1-σ)
    assert math.isclose(v.grad, numerical_diff(
        lambda t: sigmoid_combo(Value(t)).data, xv), rel_tol=1e-6)
```

(a) は neg → exp → 加算 → 逆数の4ノードに分解されますが、`_backward` の連鎖がちょうど $\sigma(1-\sigma)$ と同じ値を生みます。部品が正しければ組み合わせも正しい、の再確認です。

**問3**

```python
def powxy(x, y):
    return (y * x.log()).exp()   # x^y = e^{y log x}(x > 0)

x, y = Value(2.0), Value(3.0)
powxy(x, y).backward()
assert math.isclose(x.grad, 3.0 * 2.0 ** 2)             # y x^(y-1) = 12
assert math.isclose(y.grad, 2.0 ** 3 * math.log(2.0))   # x^y log x
```

数値微分との照合も問2と同様にできます。`__pow__` を定数指数に限定したのは、変数指数が必要な場面では常にこの書き換えが効くからです。

</details>

---

本章のコードは `code/ch04/micrograd.py`(本文のコードを順につなげたもの+演習問1の tanh、全131行)と `code/ch04/test_micrograd.py` にまとめてあり、`python3 test_micrograd.py` で全演算の数値微分照合が assert により検算できます。このモジュールは次章以降、第8巻1章まで `from micrograd import Value` の形で使い続けます。

---

> [目次](../TOC.md) ・ [← 前の章](03-backprop.md) ・ [次の章 →](05-training-with-autograd.md)
