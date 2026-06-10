# 第6巻 第5章 5.2: 文字レベル RNN 言語モデル
# 第5巻5章の tensor_autograd(行列版 autograd)をそのまま import して使う。
# 足りない演算(tanh)だけこのファイルで補い、vol5 のコードには手を入れない。
# 実行すると: n-gram(第4章)と RNN の検証 perplexity を比較し、assert で固定する。
import os
import sys
import time
import warnings

import numpy as np

# macOS の Accelerate + NumPy 2.0 は、正常な行列積でも誤った浮動小数点警告を
# 出すことがある(結果は有限で正しい)。この既知の誤報のみ黙らせる。
warnings.filterwarnings("ignore", message=".*encountered in matmul")

# 第5巻のコードを import パスに足す(巻をまたぐ依存は「後の巻が前の巻を使う」一方向のみ)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, "..", "..", "..", "vol5-backprop", "code", "ch05"))
from tensor_autograd import Tensor, softmax_cross_entropy  # noqa: E402


# --- tanh: 第5巻の Tensor に無い演算は、自分の巻で同じ流儀で足す ---
def tanh(t):
    """RNN の活性化関数。微分は 1 - tanh^2(最大でも 1、両端では 0 に潰れる)。"""
    out = Tensor(np.tanh(t.data), (t,))

    def _backward():
        t.grad += (1.0 - out.data ** 2) * out.grad

    out._backward = _backward
    return out


# --- 小さなコーパス: イソップ寓話(fable)の再話11編。文字レベルで扱う ---
CORPUS = (
    "the north wind and the sun argued about which of them was the stronger. "
    "while they argued, a traveler came along the road, wrapped in a warm cloak. "
    "they agreed that the one who first made the traveler take off his cloak "
    "should be called the stronger of the two. then the north wind blew with "
    "all his might, but the harder he blew, the more closely the traveler "
    "folded the cloak around him. at last the north wind gave up the attempt. "
    "then the sun shone out warmly, and at once the traveler took off his "
    "cloak. and so the north wind had to confess that the sun was the stronger "
    "of the two.\n"
    "one hot day a thirsty fox saw a bunch of grapes hanging from a vine. the "
    "grapes looked ripe and ready to burst with juice. the fox jumped for the "
    "grapes, but he missed them. again and again he jumped, but the grapes "
    "hung too high for him. at last the fox gave up and walked away with his "
    "nose in the air, saying, those grapes are sour, and not ripe as i "
    "thought. it is easy to despise what you cannot get.\n"
    "a hare laughed at a tortoise for being so slow. the tortoise said, slow "
    "as i am, i will beat you in a race. the hare agreed, and the race began. "
    "the hare ran far ahead, and being so sure of winning, he lay down by the "
    "road and fell asleep. the tortoise walked on and on, slow and steady, "
    "and never stopped for a moment. when the hare woke up, he ran as fast as "
    "he could, but the tortoise had already reached the goal. slow and steady "
    "wins the race.\n"
    "a hungry wolf met a house dog and admired how sleek and fat he was. the "
    "dog said, come with me, and you will be fed as well as i am. on the way "
    "the wolf noticed a worn place on the neck of the dog, and asked what had "
    "made it. the dog said, it is only the mark of the collar, for at night "
    "my master chains me up. the wolf stopped and said, then goodbye to you, "
    "my friend. i would rather starve free than be a fat slave.\n"
    "a crow sat on a branch with a piece of cheese in her beak. a fox saw the "
    "cheese and wanted it for himself. he looked up and said, how beautiful "
    "you are, and how bright your eyes. if your voice is as sweet as your "
    "form, you must be the queen of all birds. the crow, pleased by these "
    "words, opened her beak to sing, and the cheese fell to the ground. the "
    "fox snapped it up and said, my dear crow, never trust a flatterer.\n"
    "a lion was asleep when a little mouse ran over his face. the lion woke "
    "up in anger and caught the mouse with his paw. forgive me this time, "
    "said the mouse, and one day i will repay you. the lion laughed at the "
    "idea and let him go. soon after, the lion was caught in a net laid by "
    "hunters. the mouse heard his roar, ran to him, and gnawed the ropes "
    "with his teeth until the lion was free. little friends may prove great "
    "friends.\n"
    "all summer long the grasshopper sang in the sun, while the ant worked "
    "hard, carrying grain to her nest. why do you work so hard, said the "
    "grasshopper, come and sing with me. the ant said nothing and went on "
    "with her work. when winter came, the grasshopper had no food, and came "
    "to the ant to beg for grain. the ant said, you sang all summer, so now "
    "you may dance all winter. there is a time for work and a time for play.\n"
    "a shepherd boy watched his sheep near a village. for a joke, he cried "
    "out, wolf, wolf, and the people of the village ran to help him. when "
    "they came, there was no wolf at all, and the boy laughed at them. he "
    "played the same trick again and again. at last a wolf really came, and "
    "the boy cried, wolf, wolf, as loud as he could. but the people said, he "
    "is lying again, and nobody came. a liar will not be believed, even when "
    "he tells the truth.\n"
    "a town mouse once visited his friend in the country. the country mouse "
    "set out beans and bacon, and the town mouse said, come with me, and i "
    "will show you how to live. in the town they came to a great house and "
    "ate fine food on a fine table. but soon a dog barked at the door, and "
    "the two mice ran away in fear. the country mouse said, better beans in "
    "peace than cakes in fear, and went home to his quiet field.\n"
    "a farmer had a goose that laid a golden egg every morning. the eggs "
    "made him rich, but he wanted more. thinking the goose must be full of "
    "gold inside, he killed her and opened her up. he found no gold at all, "
    "and now he had lost the goose and the eggs together. much wants more "
    "and loses all.\n"
    "a dog with a piece of meat in his mouth crossed a bridge over a "
    "stream. looking down, he saw his own shadow in the water, and took it "
    "for another dog with a bigger piece of meat. he snapped at the shadow "
    "to take the meat, and as he opened his mouth, his own meat fell into "
    "the stream and was lost. grasp at the shadow and lose the substance.\n"
)


def build_vocab(text):
    """文字 ↔ 番号の対応表。第2章のトークン化の最小版(文字 = トークン)。"""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def split_corpus(ids, chunk=100, every=7, offset=3):
    """100文字の塊に切り、7つに1つを検証用に抜く(残りが訓練用)。
    末尾の寓話を丸ごと検証に回すと「訓練で一度も出ない単語だらけのテスト」になって
    しまうため、検証の塊をコーパス全体から満遍なく取る(第3巻6章の分割の規律)。"""
    chunks = [ids[i: i + chunk] for i in range(0, len(ids), chunk)]
    valid_chunks = [c for k, c in enumerate(chunks) if k % every == offset]
    train_ids = np.concatenate([c for k, c in enumerate(chunks) if k % every != offset])
    return train_ids, valid_chunks


def ngram_perplexity(train_ids, valid_chunks, n, V, k=0.1):
    """第4章の n-gram(数えるだけの言語モデル)を検証データで評価する。
    検証側には訓練で一度も見ていない並びが出るため、ゼロ除算(perplexity 無限大)を
    避ける最低限の補修として add-k 平滑化を入れてある(第4章4.3の壁への応急処置)。"""
    counts = {}
    context_counts = {}
    t = list(train_ids)
    for i in range(n - 1, len(t)):
        ctx = tuple(t[i - n + 1: i])
        counts[(ctx, t[i])] = counts.get((ctx, t[i]), 0) + 1
        context_counts[ctx] = context_counts.get(ctx, 0) + 1
    nll = 0.0
    m = 0
    for ch in valid_chunks:
        v = list(ch)
        for i in range(n - 1, len(v)):
            ctx = tuple(v[i - n + 1: i])
            c = counts.get((ctx, v[i]), 0)
            total = context_counts.get(ctx, 0)
            p = (c + k) / (total + k * V)
            nll += -np.log(p)
            m += 1
    return float(np.exp(nll / m))


def init_params(rng, V, d_e, d_h):
    """パラメータ一式。shape を声に出して読むこと(第1巻3章の儀式)。"""
    return {
        "E": Tensor(rng.normal(0, 0.1, size=(V, d_e))),        # 埋め込み行列 (V, d_e)
        "W_x": Tensor(rng.normal(0, 1.0, size=(d_e, d_h)) / np.sqrt(d_e)),   # (d_e, d_h)
        "W_h": Tensor(rng.normal(0, 1.0, size=(d_h, d_h)) / np.sqrt(d_h)),   # (d_h, d_h)
        "b": Tensor(np.zeros(d_h)),                            # (d_h,)
        "W_out": Tensor(rng.normal(0, 0.1, size=(d_h, V))),    # (d_h, V)
        "b_out": Tensor(np.zeros(V)),                          # (V,)
    }


def onehot(ids, V):
    """番号の列 (B,) → one-hot 行列 (B, V)。onehot @ E = E の行の取り出し(第3章3.2)。"""
    out = np.zeros((len(ids), V))
    out[np.arange(len(ids)), ids] = 1.0
    return out


def forward_chunk(params, X_ids, Y_ids):
    """長さ L のひと区切りを 1 トークンずつ読み、各位置の損失の平均を返す。
    X_ids, Y_ids: (B, L) の整数配列。Y は X を 1 文字ずらしたもの(次の文字が正解)。
    この for ループこそが RNN の心臓部であり、同時に痛み1の現場でもある。"""
    B, L = X_ids.shape
    V = params["E"].data.shape[0]
    d_h = params["W_h"].data.shape[0]
    h = Tensor(np.zeros((B, d_h)))                      # h_0: 何も読んでいない状態
    loss = Tensor(0.0)
    for t in range(L):                                  # ← 逐次。t は t-1 を待つ
        x_t = Tensor(onehot(X_ids[:, t], V)) @ params["E"]          # (B, d_e)
        h = tanh(x_t @ params["W_x"] + h @ params["W_h"] + params["b"])  # (B, d_h)
        logits = h @ params["W_out"] + params["b_out"]               # (B, V)
        loss = loss + softmax_cross_entropy(logits, Y_ids[:, t])
    return loss * (1.0 / L)


def make_batch(ids, B, L, rng):
    """訓練テキストからランダムな開始位置で B 本の区切りを取り出す。"""
    starts = rng.integers(0, len(ids) - L - 1, size=B)
    X = np.stack([ids[s: s + L] for s in starts])
    Y = np.stack([ids[s + 1: s + L + 1] for s in starts])
    return X, Y


def train(params, train_ids, steps, B, L, lr, rng, log_every=0):
    """訓練ループの4拍子(第3巻4章)そのまま。モデルが RNN でも拍子は変わらない。"""
    for step in range(steps):
        X, Y = make_batch(train_ids, B, L, rng)
        loss = forward_chunk(params, X, Y)              # 第1拍: forward
        for p in params.values():                       # 第2拍: 勾配リセット
            p.grad = np.zeros_like(p.data)
        loss.backward()                                 # 第3拍: backward
        for p in params.values():                       # 第4拍: 更新
            p.data -= lr * p.grad
        if log_every and (step % log_every == 0 or step == steps - 1):
            print("step %4d  loss %.3f" % (step, loss.data))
    return params


def rnn_perplexity(params, valid_chunks):
    """検証の塊を1本ずつ h=0 から読み、perplexity を返す(forward のみ・NumPy 直書き)。"""
    E = params["E"].data
    W_x, W_h, b = params["W_x"].data, params["W_h"].data, params["b"].data
    W_out, b_out = params["W_out"].data, params["b_out"].data
    nll = 0.0
    m = 0
    for ch in valid_chunks:
        h = np.zeros(W_h.shape[0])
        for t in range(len(ch) - 1):
            x = E[ch[t]]                                # lookup = one-hot @ E の省略形
            h = np.tanh(x @ W_x + h @ W_h + b)
            z = h @ W_out + b_out
            z = z - z.max()
            log_p = z - np.log(np.exp(z).sum())
            nll += -log_p[ch[t + 1]]
            m += 1
    return float(np.exp(nll / m))


if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(42)

    stoi, itos = build_vocab(CORPUS)
    V = len(stoi)
    ids = np.array([stoi[ch] for ch in CORPUS])
    train_ids, valid_chunks = split_corpus(ids)
    n_valid = sum(len(c) for c in valid_chunks)
    print("コーパス %d 文字 / 語彙 %d 文字 / 訓練 %d, 検証 %d(%d 塊)"
          % (len(ids), V, len(train_ids), n_valid, len(valid_chunks)))

    # --- 第4章の n-gram を同じデータで評価(比較台)---
    ppl_bigram = ngram_perplexity(train_ids, valid_chunks, n=2, V=V)
    ppl_trigram = ngram_perplexity(train_ids, valid_chunks, n=3, V=V)
    print("bigram  検証 perplexity: %6.2f" % ppl_bigram)
    print("trigram 検証 perplexity: %6.2f" % ppl_trigram)

    # --- RNN 言語モデルの訓練(終盤は学習率を 1/5 に絞って仕上げる)---
    params = init_params(rng, V, d_e=24, d_h=48)
    train(params, train_ids, steps=400, B=16, L=32, lr=0.7, rng=rng, log_every=100)
    train(params, train_ids, steps=200, B=16, L=32, lr=0.14, rng=rng, log_every=100)
    ppl_rnn = rnn_perplexity(params, valid_chunks)
    print("RNN     検証 perplexity: %6.2f" % ppl_rnn)

    # --- 生成してみる(第4章と同じ「機械が文を書く」体験)---
    gen_rng = np.random.default_rng(42)
    E = params["E"].data
    W_x, W_h, b = params["W_x"].data, params["W_h"].data, params["b"].data
    W_out, b_out = params["W_out"].data, params["b_out"].data
    h = np.zeros(48)
    i = stoi["t"]
    text = "t"
    for _ in range(120):
        h = np.tanh(E[i] @ W_x + h @ W_h + b)
        z = h @ W_out + b_out
        p = np.exp(z - z.max())
        p /= p.sum()
        i = gen_rng.choice(V, p=p)
        text += itos[i]
    print("生成例:", repr(text))

    # --- 比較を assert で固定: RNN は数えるだけの n-gram より一般化する ---
    assert ppl_trigram < ppl_bigram, "文脈を1文字延ばした trigram が bigram に勝つはず"
    assert ppl_rnn < ppl_trigram, "RNN が trigram に負けています"
    assert ppl_rnn < ppl_bigram, "RNN が bigram に負けています"
    print("ok: RNN (%.2f) < trigram (%.2f) < bigram (%.2f) < 語彙サイズ %d(あてずっぽうの上限)"
          % (ppl_rnn, ppl_trigram, ppl_bigram, V))
    print("実行時間: %.1f 秒" % (time.time() - t0))
