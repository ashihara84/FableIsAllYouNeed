# 第3章 埋め込み — トークンをベクトルにする

前の章で、文章をトークンの列に切れるようになりました。BPE が育てた語彙を使えば、どんな文章も「語彙表の何番か」という整数 ID の列になります。第1章の言語モデルの問題設定と合わせると、やりたいことはもう明確です。ID の列を受け取り、次の ID の確率分布を出す——それだけです。

ところが、ここで手が止まります。第1巻から第5巻まで、私たちが組み上げてきた部品——`X @ W + b`、活性化関数、autograd——はすべて、入力が**数値ベクトル** $\mathbf{x}$ であることを前提にしていました。トークン ID は数値ではないか、と思うかもしれませんが、ID の「12」は ID の「3」の4倍大きい単語、という意味ではありません。語彙表の並び順はたまたまであって、ID の大小も差も、意味とは無関係です。この無関係な大小をネットワークに食わせたら、モデルは存在しない構造を学ぼうとしてしまいます。

実は、この問いには第1巻で予告を打ってあります。第1巻1.5——king − man + woman ≈ queen という「意味の算術」を紹介した節で、**どうやってそんなベクトルを手に入れるのか、この伏線は第6巻第3章で回収します**と約束しました。この章が、その第6巻第3章です。約束どおり、5巻ぶんの道具を全部使って回収します。

## 3.1 one-hot ベクトル — すべての単語が等しく無関係

まずは、いちばん素朴な方法から始めましょう。ID の大小が嘘の意味を運んでしまうのが問題なら、大小が発生しない表現にすればよいのです。

語彙サイズを $V$ とします。単語 ID $i$ を、**$V$ 次元のベクトルで、第 $i$ 成分だけが 1、残りすべてが 0** のものに対応させます。これを **one-hot ベクトル(one-hot vector)** と呼びます。「熱い(hot)成分が1つだけ」という意味の名前です。

$$\text{ID } 2 \ \longmapsto\ \mathbf{x} = (0,\ 0,\ 1,\ 0,\ 0) \quad \text{(} V = 5 \text{ の場合。shape は } (V,) \text{)}$$

どの単語も「長さ1で、自分専用の軸を向いたベクトル」になるので、大小関係は消えます。コードで確かめましょう。

```python
import numpy as np

V = 5  # 語彙サイズ(おもちゃの例)
vocab = ["cat", "dog", "king", "queen", "the"]

def one_hot(i, V):
    v = np.zeros(V)
    v[i] = 1.0
    return v

cat, dog, king = one_hot(0, V), one_hot(1, V), one_hot(2, V)
print(cat)   # [1. 0. 0. 0. 0.]

# どの2語を選んでも、内積は 0
assert np.dot(cat, dog) == 0.0
assert np.dot(cat, king) == 0.0
assert np.dot(dog, king) == 0.0
```

最後の3行が、この表現の致命傷です。第1巻第2章で、内積は「似ている度合い」を測る道具でした。one-hot ベクトルどうしの内積は、**異なる単語である限り必ず 0** です。1が立っている位置が違うので、掛けて足すとすべての項が消えます。コサイン類似度で測っても同じく 0 です。

つまり one-hot の世界では、cat と dog の類似度も、cat と the の類似度も、等しく 0。**すべての単語が、他のすべての単語と等しく無関係**なのです。第1巻1.5で見た「意味が成分に分かれて宿る」ベクトルとは正反対の、意味を運ぶ余地が構造的に存在しない表現です。king − man + woman を計算しても、queen には決して着地しません(king の位置が 1、man の位置が −1、woman の位置が 1 という「3語のメモ」ができるだけです)。おまけに次元も巨大で、語彙が3万語なら3万次元、その99.99%が 0 です。

ただし、one-hot を捨てるのは早計です。「意味を運ぶ」道具としては失格でも、「**どの単語かを指し示す**」道具としては完璧だからです。この指し示す能力が、次の節で効いてきます。

## 3.2 埋め込み行列 — lookup はただの行列積

では、内積が意味を語れるベクトルは、どんな形をしているべきでしょうか。中身の数値は後回しにして、まず入れ物の形から決めます。

欲しいのは、単語ごとの、$V$ よりずっと小さい $d$ 次元の(0 だらけではない)ベクトルです。$V$ 個の $d$ 次元ベクトル——と聞いたら、第1巻第3章の見方の出番です。**行列とは行ベクトルの束**でした。$V$ 本のベクトルを行として積み上げれば、1枚の行列になります。

$$E \in \mathbb{R}^{V \times d} \quad \text{shape } (V, d) \text{: 第 } i \text{ 行が、単語 } i \text{ のベクトル}$$

この $E$ を**埋め込み行列(embedding matrix)**、その行を**埋め込み(embedding)**と呼びます。$V$ 次元の one-hot の世界から $d$ 次元の世界へ単語を「埋め込む」、という気持ちの名前です。

単語 $i$ のベクトルが欲しければ $E$ の第 $i$ 行を取り出せばよい——この取り出し(lookup)は、実は前節の one-hot との**行列積そのもの**です。one-hot $\mathbf{x}$ `(V,)` を左から掛けてみます。

$$(\mathbf{x} E)_j = \sum_{i=0}^{V-1} x_i E_{ij}$$

$x_i$ は $i = 2$ のときだけ 1 で、ほかは 0。和の中で生き残るのは $E_{2j}$ の項だけです。つまり $\mathbf{x} E$ は $E$ の第2行、そのものです。

```python
import numpy as np

rng = np.random.default_rng(42)
V, d = 5, 3
E = rng.normal(0.0, 0.1, (V, d))   # 埋め込み行列 E (V, d)。中身はまだ乱数

x = np.zeros(V)
x[2] = 1.0                         # 単語 ID 2 の one-hot (V,)

assert np.allclose(x @ E, E[2])    # one-hot @ E = E の第2行の取り出し
```

「行を取り出すだけなら `E[2]` と書けば済むのに、なぜわざわざ行列積として見るのか」と思うかもしれません。御利益は2つあります。

1つ。**埋め込み層は、バイアスなしの線形層 `X @ W` と同じ形**だとわかります(第1巻第6章)。トークン列 $n$ 個ぶんの one-hot を行に積んだ $X$ `(n, V)` に対して、$XE$ は `(n, V) @ (V, d) = (n, d)`。新しい部品は何ひとつ要りません。

2つ。行列積なら、**第5巻の autograd がそのまま微分を運んでくれます**。第5巻5章の `Tensor` は `@` の backward をすでに知っています。つまり $E$ に勾配を流す仕組みは、もう書き終わっているのです。

なお、実務のライブラリは $V$ が数万になるため、行列積を計算せず添字で直接行を取り出します(結果は同一で、速いだけです)。それでも「数学的な正体は行列積」という見方が、この後ずっと効きます。

## 3.3 E も学習されるパラメータである

入れ物の形は決まりました。残る問いはひとつ——$E$ の中身の数値は、**誰が決めるのか**。

第1巻1.5の採点表を思い出してください。king や queen に「王族度・男性度・女性度・人間度」の4項目で点数を付けた、あの表です。あのとき白状したとおり、あの表は答えから逆算して私たちが手で書いたものでした。同じことを本物の語彙でやるなら、3万語 × 512次元 = 1536万個の数値を手で埋めることになります。不可能ですし、それ以前に「正しい採点項目」が何かを、誰も知りません。

答えはこうです。**決めない。** $E$ を乱数で初期化し、$W$ や $b$ と同じ「学習されるパラメータ」として勾配降下に晒します。埋め込み層は線形層の一種ですから(前節)、損失から $E$ への勾配は autograd が自動で運んでくれます。仕組みとしては何も新しくありません。

新しいのは、問いの立て方です。意味には「正解ラベル」がありません。king の正しいベクトルなど誰も知らないのに、何を目標に学習させればよいのでしょうか。

ここで第1章の作業仮説が効きます。「次(周り)の単語を当てられる = 言語がわかっている」。**周りの単語を当てるタスクなら、正解ラベルはコーパス自身がくれます**。文章中の単語をひとつ選び、その周辺に実際に何が書いてあったかを当てさせればよいのです。ラベル付け作業はゼロです。

このタスクで訓練すると、何が起きるか。似た文脈に現れる単語——たとえば king と queen はどちらも「rules」「palace」「crown」の近くに現れます——は、**同じ文脈語を当てるのが得**になります。同じ出力を出したいなら、入力ベクトルも似ているのが効率的です。勾配降下は、king と queen のベクトルを自然と近くに押し込んでいきます。「似た文脈に現れる単語は似た意味を持つ」という言語学の仮説(分布仮説、distributional hypothesis)が、損失最小化の力学として働くのです。

**意味は、与えるものではなく、タスクから染み込むもの**——これがこの節の結論であり、この後 Transformer まで一貫して変わらない原理です。

## 3.4 [コード] 小さなコーパスで king − man + woman を検算する

道具は揃いました。検算に行きましょう。

タスクは前節の「周りの単語を当てる」を最小構成にしたもの——中心の単語を入力に、窓(前後3語以内)の中の単語を当てる多クラス分類です(word2vec では skip-gram と呼ばれる構成です)。よく見ると、これは**第4巻第6章の softmax 回帰そのもの**です。

$$\text{logits} = \underbrace{X}_{(n,\ V)} \underbrace{E}_{(V,\ d)} \underbrace{W_{out}}_{(d,\ V)} \quad \to \quad \text{softmax cross-entropy}$$

$X$ は中心語 one-hot を $n$ 行積んだ行列、正解は文脈語の ID です。訓練ループは第3巻第4章の4拍子(forward → loss → backward → 更新)そのまま、微分は第5巻の autograd まかせ。本当に、新しい部品はひとつもありません。

ひとつ、先に正直に断っておきます。コーパスは22文・語彙22語の、手で設計したものです。性別の手がかり(he/she、man/woman との共起)と王族の手がかり(palace、crown)がはっきり現れるように文を組んであります。word2vec が数十億語から自然に獲得したものを、数百語の箱庭で意図的に再現するということで、設計なしの生コーパスでこれほど綺麗に出るわけではありません(演習4で実際に壊してみます)。

```python
# 第6巻 第3章 3.4: 小コーパスで埋め込みを学習し、king − man + woman を検算する
# 第1巻1.5で予告した「意味の算術」の伏線回収。
# 第5巻5章の自作 autograd(tensor_autograd.py)を import して使う。
import os
import sys
import warnings

import numpy as np

# 一部の macOS 環境(Accelerate BLAS + NumPy 2.0系)では、正しい行列積でも
# 誤った RuntimeWarning が出ることが知られている(計算結果は正しい)。本筋ではないので非表示にする
warnings.filterwarnings("ignore", message=".*encountered in matmul")

# --- 第5巻の autograd を借りてくる(vol5 のファイルは変更しない) ---
_VOL5 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "..", "vol5-backprop", "code", "ch05")
sys.path.insert(0, _VOL5)
from tensor_autograd import Tensor, softmax_cross_entropy

rng = np.random.default_rng(42)

# --- 小コーパス(訓練データ) ---
# 性別と王族らしさの2軸が文脈に現れるように手で設計してある(本文3.4参照)
corpus = [
    "the king rules the palace",
    "the queen rules the palace",
    "the prince guards the palace",
    "the princess guards the palace",
    "the man works in the village",
    "the woman works in the village",
    "the boy plays in the village",
    "the girl plays in the village",
    "the king is a man",
    "the queen is a woman",
    "the prince is a boy",
    "the princess is a girl",
    "he is the king",
    "she is the queen",
    "he is the prince",
    "she is the princess",
    "he is the man",
    "she is the woman",
    "he is the boy",
    "she is the girl",
    "the king wears the crown",
    "the queen wears the crown",
]

# --- 語彙とトークンID(第2章なら BPE の出番だが、ここでは空白区切りで足りる) ---
sentences = [s.split() for s in corpus]
vocab = sorted(set(w for s in sentences for w in s))
V = len(vocab)                                # 語彙サイズ V = 22
word2id = {w: i for i, w in enumerate(vocab)}

# --- 訓練ペアの作成(skip-gram: 中心語から文脈語を当てる) ---
window = 3
centers, contexts = [], []
for s in sentences:
    for i in range(len(s)):
        for j in range(max(0, i - window), min(len(s), i + window + 1)):
            if j != i:
                centers.append(word2id[s[i]])
                contexts.append(word2id[s[j]])
centers = np.array(centers)
contexts = np.array(contexts)
n = len(centers)                              # n = 372 ペア

# --- one-hot 行列 X (n, V): 第 i 行は centers[i] の位置だけ 1 ---
X_onehot = np.zeros((n, V))
X_onehot[np.arange(n), centers] = 1.0

# 3.1 の確認: 異なる単語の one-hot どうしの内積は必ず 0(すべての単語が等しく無関係)
assert np.dot(X_onehot[0], X_onehot[1]) == 0.0 or centers[0] == centers[1]

# --- パラメータ: 埋め込み行列 E (V, d) と出力層 W_out (d, V) ---
d = 8                                          # 埋め込みの次元(論文なら d_model = 512)
E = Tensor(rng.normal(0.0, 0.1, (V, d)))
W_out = Tensor(rng.normal(0.0, 0.1, (d, V)))
X = Tensor(X_onehot)

# 3.2 の確認: one-hot @ E は「E から行を取り出す」のと同じ(lookup はただの行列積)
assert np.allclose((X @ E).data, E.data[centers])

# --- 訓練ループ(第3巻4章の4拍子: forward → loss → backward → 更新) ---
lr = 5.0
losses = []
for epoch in range(300):
    E.grad = np.zeros_like(E.data)            # 勾配は += で溜まるので毎回ゼロに戻す
    W_out.grad = np.zeros_like(W_out.data)
    logits = (X @ E) @ W_out                  # (n, V): 文脈語の当てっこのスコア
    loss = softmax_cross_entropy(logits, contexts)
    loss.backward()
    E.data -= lr * E.grad
    W_out.data -= lr * W_out.grad
    losses.append(float(loss.data))

# 開始時の損失は「V 択を一様に当てずっぽう」の log V ≈ 3.09 付近。そこから下がったか
assert abs(losses[0] - np.log(V)) < 0.1
assert losses[-1] < losses[0] - 0.5
print("loss: %.3f -> %.3f (log V = %.3f)" % (losses[0], losses[-1], np.log(V)))

# --- 学習した埋め込みで意味の算術を検算する ---
Emb = E.data                                   # (V, d) 学習済み埋め込み


def cosine(a, b):
    """コサイン類似度(第1巻2.3)。正規化してから内積"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def nearest(vec, exclude=()):
    """vec にコサイン類似度が高い順に語彙を並べる(exclude の単語は除く)"""
    scores = [(cosine(vec, Emb[word2id[w]]), w) for w in vocab if w not in exclude]
    return sorted(scores, reverse=True)


def analogy(a, b, c):
    """a − b + c に最も近い単語を返す(word2vec の流儀で a, b, c 自身は候補から除く)"""
    v = Emb[word2id[a]] - Emb[word2id[b]] + Emb[word2id[c]]
    return nearest(v, exclude=(a, b, c))


# 第1巻1.5の伏線回収: king − man + woman ≈ queen
top = analogy("king", "man", "woman")
print("king - man + woman ->", [(w, round(float(s), 3)) for s, w in top[:3]])
assert top[0][1] == "queen"
assert top[0][0] > 0.8

# 同じ算術がもう1組でも成り立つか: prince − boy + girl ≈ princess
top2 = analogy("prince", "boy", "girl")
print("prince - boy + girl ->", [(w, round(float(s), 3)) for s, w in top2[:3]])
assert top2[0][1] == "princess"

# 平行性の確認: 「男 → 女」の差ベクトルが、ペアによらずほぼ同じ向きを向いている
d_kq = Emb[word2id["king"]] - Emb[word2id["queen"]]
d_mw = Emb[word2id["man"]] - Emb[word2id["woman"]]
d_bg = Emb[word2id["boy"]] - Emb[word2id["girl"]]
print("cos(king-queen, man-woman) = %.3f" % cosine(d_kq, d_mw))
print("cos(boy-girl,   man-woman) = %.3f" % cosine(d_bg, d_mw))
assert cosine(d_kq, d_mw) > 0.7
assert cosine(d_bg, d_mw) > 0.7

# --- 演習用: 埋め込み空間の近傍語を観察する ---
for w in ["king", "queen", "man", "village"]:
    print("%-8s の近傍:" % w, [(u, round(float(s), 3)) for s, u in nearest(Emb[word2id[w]], exclude=(w,))[:3]])

print("第3章: すべての assert を通過しました")
```

実行すると(手元では1秒かかりません)、次の出力が得られます。

```
loss: 3.091 -> 2.181 (log V = 3.091)
king - man + woman -> [('queen', 0.927), ('princess', 0.636), ('rules', 0.479)]
prince - boy + girl -> [('princess', 0.924), ('queen', 0.533), ('guards', 0.533)]
cos(king-queen, man-woman) = 0.897
cos(boy-girl,   man-woman) = 0.978
king     の近傍: [('prince', 0.662), ('queen', 0.62), ('crown', 0.497)]
queen    の近傍: [('princess', 0.624), ('king', 0.62), ('crown', 0.393)]
man      の近傍: [('woman', 0.723), ('boy', 0.638), ('crown', 0.485)]
village  の近傍: [('woman', 0.622), ('girl', 0.608), ('in', 0.577)]
第3章: すべての assert を通過しました
```

順に読み解きます。

**損失の出発点が log V なのは偶然ではありません。** 初期の $E$ は乱数なので、モデルは22択を一様に当てずっぽうするしかなく、そのときの cross-entropy はちょうど $\log 22 \approx 3.09$ です(第4巻第5章)。第1章の言葉でいえば perplexity 22——300エポック後には $e^{2.181} \approx 8.9$、「平均9択」まで絞れたことになります。

**そして、本題。** $\mathbf{v}_{king} - \mathbf{v}_{man} + \mathbf{v}_{woman}$ に最も近い単語は **queen**、コサイン類似度 0.927。**第1巻1.5で予告した「意味の算術」が、自分で学習させたベクトルの上で、いま成立しました。** あのときは私たちが答えから逆算して採点表を手で書きましたが、今回は誰も「王族度」や「女性度」という軸を設計していません。乱数から出発した $E$ が、「周りの単語を当てる」というタスクの勾配だけに押されて、この構造を獲得したのです。そして「≈(近い)」の測り方も、第1巻1.5では宿題でした——第1巻第2章の内積とコサイン類似度が、ここでその役を果たしています。2つの伏線、同時回収です。

同じ算術は prince − boy + girl ≈ princess でも成立し、さらに king − queen、man − woman、boy − girl という「男 → 女」の差ベクトルたちが、ほぼ同じ向き(コサイン類似度 0.897〜0.978)を向いています。第1巻1.5の言葉でいえば、「king − man が取り出した差分」が単語ペアによらず共有されている——埋め込み空間の中に「性別の方向」が1本通っている、ということです。意味の算術が成り立つ正体は、この平行性です。

正直な注記も2つ。

1つ。`analogy` は word2vec 以来の流儀どおり、入力した3語自身を候補から除いています。今回のコーパスでは除かなくても queen が1位になりますが、一般には king 自身が1位に来てしまうことが多く、除外は慣例になっています。検算の条件なので、コードに明記しました。

2つ。冒頭で断ったとおり、コーパスは答えが出るように設計したものです。この実験が証明しているのは「word2vec はすごい」ではなく、「**文脈の手がかりさえコーパスにあれば、3.3 の仕組み(タスク + 勾配降下)がそれをベクトルの幾何に変換する**」という原理のほうです。手がかりを消したら何が起きるかは、演習4で確かめます。

## 3.5 論文 3.4 を覗き見 — "learned embeddings of dimension d_model" が読める

巻頭のラスボスのうち、この章が担当する一文を見に行きましょう。論文のセクション 3.4(Embeddings and Softmax)の冒頭です。

> *"Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension $d_{model}$."*
> — Vaswani et al., "Attention Is All You Need", Section 3.4
>
> 訳: 他の系列変換モデルと同様に、私たちは学習された埋め込みを用いて、入力トークンと出力トークンを次元 $d_{model}$ のベクトルに変換する。

一語ずつ、もう読めるはずです。**tokens** は第2章の BPE が切り出したあれ。**embeddings** は本章の $E$ の行。**learned** は 3.3 そのもの——埋め込みは設計するものではなく、学習されるパラメータです。**vectors of dimension $d_{model}$** は、$E$ の shape が $(V, d_{model})$ だということ。本章のおもちゃでは $d = 8$ でしたが、論文では $d_{model} = 512$——第1巻第1章で「1単語 = 512個の数」として眺めた、あの512です。

ひとつだけ、本章との違いがあります。本章の skip-gram は「埋め込みを学習するためだけの独立したタスク」でしたが、Transformer の埋め込みは前処理ではなく、**翻訳というタスク本体と一緒に学習されます**。とはいえ原理は 3.3 と同じです。意味はタスクから染み込む——染み込ませるタスクが「周りの単語当て」から「翻訳」に変わるだけです。

なお、セクション 3.4 には続きの文があります。入出力の埋め込みと softmax 直前の線形層での重み行列の共有、そして $\sqrt{d_{model}}$ 倍。ここはまだ読めなくて構いません。論文読解マップのとおり、第7巻の仕事です。

## まとめ

- one-hot ベクトルは「どの単語か」を指し示す道具としては完璧だが、異なる単語どうしの内積が常に 0 ——**すべての単語が等しく無関係**で、意味を運ぶ余地がない(第1巻第2章の「内積 = 類似度」の言葉で)
- 埋め込み行列 $E$ `(V, d)` は単語ベクトルを行に積んだもの。lookup は **one-hot @ E というただの行列積**であり、埋め込み層はバイアスなしの線形層にすぎない
- $E$ は**学習されるパラメータ**。意味の正解ラベルは存在しないが、「周りの単語を当てる」タスクならラベルはコーパス自身がくれる。意味は与えるものではなく、タスクから染み込む
- 自作 autograd で埋め込みを訓練し、**king − man + woman ≈ queen を検算した(第1巻1.5の伏線回収)**。「≈」はコサイン類似度で測った(第1巻第2章の回収)。算術が成り立つ正体は「男 → 女」方向の平行性
- ただしコーパスは答えが出るよう設計したもの。原理の検証であって、word2vec の再現実験ではない

**ラスボスとの距離**: 論文 3.4 の "learned embeddings ... of dimension $d_{model}$" が読めました。アーキテクチャ図(図1)の最下段、Embedding の箱は攻略済みです。

## 演習

**問1(手計算)** $V = 4$、$d = 2$ とし、$E$ の行を上から $(1, 0)$, $(0, 1)$, $(2, 3)$, $(-1, 1)$ とする。one-hot $\mathbf{x} = (0, 0, 1, 0)$ に対して $\mathbf{x} E$ を成分の式 $\sum_i x_i E_{ij}$ から計算し、$E$ の第2行(0始まり)と一致することを確かめよ。

<details><summary>略解</summary>

$j = 0$ 成分: $0 \cdot 1 + 0 \cdot 0 + 1 \cdot 2 + 0 \cdot (-1) = 2$。$j = 1$ 成分: $0 \cdot 0 + 0 \cdot 1 + 1 \cdot 3 + 0 \cdot 1 = 3$。よって $\mathbf{x} E = (2, 3)$ で、第2行と一致。$x_i$ が 1 の行だけが和に生き残る、というだけの計算です。
</details>

**問2(観察)** 本文のコードの最後の `nearest` ループを、`vocab` の全単語について回せ。king や man のような内容語と、the / is / a のような機能語とで、近傍リストの「読みやすさ」に差はあるか。気づいたことを言葉にせよ。

<details><summary>略解</summary>

内容語は直観に合う近傍を持ちます(king ↔ prince・queen、man ↔ woman・boy など)。一方 the / is / a などの機能語はほぼすべての単語の隣に現れるため文脈で区別がつかず、近傍リストは雑多になりがちです(例: he の近傍に a が来る)。分布仮説の力学は、文脈に偏りのある単語にしか効かないことが観察できます。
</details>

**問3(コード)** `d = 8` を `d = 2` に変えて再訓練し、`analogy("king", "man", "woman")` を実行せよ。1位は変わるか。上位の類似度の「差」はどうなるか。

<details><summary>略解</summary>

seed 42 では 1 位は queen のままですが、類似度が queen 1.000、works 0.9998、plays 0.9998 …と、ほぼ全単語が 1 に張り付きます。2次元では22語ぶんの意味の軸を分担する余地がなく、全員がほぼ同じ向きに押し込められるためです(cos(king, queen) も 1.000 になります)。「勝った」ことよりも「僅差でしか勝てない」ことが本質で、意味を成分に分けて宿らせるには次元の余裕が必要だとわかります。論文の $d_{model} = 512$ は、この余裕を大規模語彙に対して確保した値です。
</details>

**問4(コード)** コーパスから性別の手がかりを消す——「the king is a man」型の4文と「he is the king」型の8文、計12文を削除して再訓練せよ。(a) `analogy("king", "man", "woman")` の1位、(b) `cosine(Emb[king] − Emb[queen], Emb[man] − Emb[woman])`、(c) `cosine(Emb[king], Emb[queen])` をそれぞれ確認し、(a) の結果を信じてよいか論ぜよ。

<details><summary>略解</summary>

seed 42 では (a) は依然 queen(類似度 0.998)ですが、(b) の平行性は 0.897 → 0.245 に崩壊し、(c) は 0.998 まで上がります。性別の手がかりが消えたため king と queen は文脈上ほぼ同一の単語になり(だから (c) がほぼ 1)、「男 → 女」の方向は学習されていません(だから (b) が崩壊)。それでも (a) が queen を返すのは、king − man + woman ≈ king ≈ queen で、king 自身は候補から除外されているから——**算術が成功したのではなく、評価の慣例(入力語の除外)が成功を偽装した**のです。1つの数字だけで結論しないこと、そして意味はコーパスに手がかりがある分しか染み込まないこと(3.3)の、裏側からの確認です。
</details>
