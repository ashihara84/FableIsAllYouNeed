"""RNN encoder-decoder(seq2seq)で文字列反転を学習する(第6巻6章)。

- autograd は第5巻の tensor_autograd.Tensor をそのまま import する(vol5 側は変更しない)。
  Tensor に足りない演算(tanh, embedding)は、このファイルで「外付けの演算」として補う。
- タスクはミニチュア翻訳としての文字列反転: "fcahd" → "dhacf"。
  データ生成は make_data() に独立させてある。第7章(attention 付き seq2seq)が
  同じ語彙・同じ make_data を import して性能比較するため、この関数は変更しない契約。
- 6.3 の実測: 訓練に使った長さの範囲内でも、入力が長いほど系列一致率が崩れる
  (固定長ボトルネック)。章末の表の数値を assert で固定してある。
"""

import os
import sys
import time
import warnings

import numpy as np

# 一部の macOS 環境(Accelerate BLAS + NumPy 2.0系)では、正しい行列積でも
# 誤った RuntimeWarning が出ることが知られている(計算結果は正しい)。本筋ではないので非表示にする
warnings.filterwarnings("ignore", message=".*encountered in matmul")

# --- 第5巻の autograd を借りてくる(vol5 のファイルは変更しない) ---
_VOL5 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "..", "vol5-backprop", "code", "ch05")
sys.path.insert(0, os.path.abspath(_VOL5))

from tensor_autograd import Tensor, softmax_cross_entropy  # noqa: E402


# ---------------------------------------------------------------------------
# 語彙とデータ生成(第7章と共有する部分)
# ---------------------------------------------------------------------------

BOS = 0          # 生成開始の合図(decoder の最初の入力)
EOS = 1          # 生成終了の合図(decoder が最後に出すべき記号)
LETTERS = "abcdefgh"
OFFSET = 2       # 文字 ID は 2 始まり(0, 1 は BOS / EOS が使う)
V = OFFSET + len(LETTERS)  # 語彙サイズ 10


def make_data(rng, n, length):
    """文字列反転タスクのペアを n 個生成する。

    戻り値: (src, tgt)
      src: (n, length) — 入力の文字 ID 列(ID は OFFSET..V-1 の一様乱数)
      tgt: (n, length) — src を左右反転した文字 ID 列
    BOS / EOS はここでは付けない(teacher forcing / 生成の側で付ける)。
    第7章の attention 付き seq2seq も同じこの関数でデータを作り、性能を比較する。
    """
    src = rng.integers(OFFSET, V, size=(n, length))
    tgt = src[:, ::-1].copy()
    return src, tgt


def ids_to_str(ids):
    """文字 ID 列を表示用の文字列へ(BOS/EOS は記号で)。"""
    table = {BOS: "<bos>", EOS: "<eos>"}
    return "".join(table.get(int(i), LETTERS[int(i) - OFFSET]) for i in ids)


# ---------------------------------------------------------------------------
# Tensor に足りない演算を外付けで補う(vol5 のファイルは変更しない)
# ---------------------------------------------------------------------------

def tanh(t):
    """tanh ノード。第5巻4章の演習問1(Value 版 tanh)の Tensor 版。
    局所勾配は 1 - tanh^2(第5巻1章)。"""
    out = Tensor(np.tanh(t.data), (t,))

    def _backward():
        t.grad += (1.0 - out.data ** 2) * out.grad

    out._backward = _backward
    return out


def embedding(E, idx):
    """埋め込み行列 E (V, d_emb) から idx (n,) の行を取り出す lookup ノード(第3章)。
    backward は「取り出した行へ勾配を足し戻す」。同じ ID が複数回出ても
    正しく累積するよう np.add.at を使う(ふつうの += は重複 ID を1回しか足さない)。"""
    idx = np.asarray(idx, dtype=int)
    out = Tensor(E.data[idx], (E,))

    def _backward():
        np.add.at(E.grad, idx, out.grad)

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# モデル: RNN encoder-decoder
# ---------------------------------------------------------------------------

D_EMB = 16   # 埋め込み次元
D_H = 32     # 隠れ状態の次元 = 文全体を詰め込む「1本のベクトル」の太さ


def init_params(rng):
    """パラメータ一式。初期化は Xavier(第5巻6.6: 1/sqrt(入力次元))。"""
    def mat(d_in, d_out):
        return Tensor(rng.standard_normal((d_in, d_out)) / np.sqrt(d_in))

    return {
        "E": mat(V, D_EMB),            # 埋め込み(encoder / decoder で共有)
        "Wx_e": mat(D_EMB, D_H),       # encoder: 入力 → 隠れ
        "Wh_e": mat(D_H, D_H),         # encoder: 隠れ → 隠れ
        "b_e": Tensor(np.zeros(D_H)),
        "Wx_d": mat(D_EMB, D_H),       # decoder: 入力 → 隠れ
        "Wh_d": mat(D_H, D_H),         # decoder: 隠れ → 隠れ
        "b_d": Tensor(np.zeros(D_H)),
        "Wo": mat(D_H, V),             # decoder: 隠れ → 語彙スコア(logits)
        "bo": Tensor(np.zeros(V)),
    }


def encode(params, src):
    """encoder: 入力列を1トークンずつ読み、最後の隠れ状態1本 (n, D_H) に要約する。
    第5章の RNN と同じ漸化式 h_t = tanh(x_t @ Wx + h_{t-1} @ Wh + b)。"""
    n, length = src.shape
    h = Tensor(np.zeros((n, D_H)))
    for t in range(length):
        x_t = embedding(params["E"], src[:, t])              # (n, D_EMB)
        h = tanh(x_t @ params["Wx_e"] + h @ params["Wh_e"] + params["b_e"])
    return h  # ボトルネック: 入力が何文字でも、これは (n, D_H) の1本


def decode_step(params, h, idx_in):
    """decoder を1ステップ進める。入力トークン idx_in (n,) と前の隠れ状態 h を受け、
    新しい隠れ状態と語彙スコア logits (n, V) を返す。"""
    x_t = embedding(params["E"], idx_in)
    h = tanh(x_t @ params["Wx_d"] + h @ params["Wh_d"] + params["b_d"])
    logits = h @ params["Wo"] + params["bo"]
    return h, logits


def loss_teacher_forcing(params, src, tgt):
    """teacher forcing の損失(1トークンあたりの平均 cross-entropy)。
    decoder への入力は [BOS, tgt[0], ..., tgt[L-1]](自分の出力ではなく正解)、
    当てるべき出力は [tgt[0], ..., tgt[L-1], EOS]。"""
    n, length = tgt.shape
    h = encode(params, src)
    inputs = np.concatenate([np.full((n, 1), BOS), tgt], axis=1)    # (n, L+1)
    targets = np.concatenate([tgt, np.full((n, 1), EOS)], axis=1)   # (n, L+1)
    total = Tensor(0.0)
    for t in range(length + 1):
        h, logits = decode_step(params, h, inputs[:, t])
        total = total + softmax_cross_entropy(logits, targets[:, t])
    return total * (1.0 / (length + 1))


def generate(params, src, max_steps):
    """自己回帰生成: BOS から始め、自分の出力(argmax)を次の入力に戻す。
    戻り値: (n, max_steps) の出力 ID 列。"""
    h = encode(params, src)
    idx = np.full(src.shape[0], BOS)
    out = []
    for _ in range(max_steps):
        h, logits = decode_step(params, h, idx)
        idx = logits.data.argmax(axis=1)  # 貪欲法: 一番スコアの高い1語
        out.append(idx)
    return np.stack(out, axis=1)


def sequence_accuracy(params, length, n_eval, seed):
    """長さ length の新しいデータ n_eval 個で、系列一致率(全文字一致のみ正解)を測る。
    EOS の位置まで含めて要求する(L 文字 + EOS が全部合って 1 点)。"""
    rng = np.random.default_rng(seed)
    src, tgt = make_data(rng, n_eval, length)
    pred = generate(params, src, length + 1)
    ok = (pred[:, :length] == tgt).all(axis=1) & (pred[:, length] == EOS)
    return ok.mean()


# ---------------------------------------------------------------------------
# 訓練と実測
# ---------------------------------------------------------------------------

MIN_LEN, MAX_LEN = 2, 12   # 訓練に使う入力長の範囲(両端含む一様)。評価もこの内側で行う
BATCH = 32
STEPS = 4000
LR = 0.3


def train(verbose=True):
    rng = np.random.default_rng(42)
    params = init_params(rng)
    for step in range(STEPS):
        length = int(rng.integers(MIN_LEN, MAX_LEN + 1))
        src, tgt = make_data(rng, BATCH, length)
        loss = loss_teacher_forcing(params, src, tgt)   # 1. forward + 2. loss
        for p in params.values():
            p.grad[...] = 0.0
        loss.backward()                                 # 3. gradient
        for p in params.values():
            p.data -= LR * p.grad                       # 4. update
        if verbose and (step % 500 == 0 or step == STEPS - 1):
            print("  step {:>5}: loss = {:.4f} (len {})".format(step, loss.data, length))
    return params


if __name__ == "__main__":
    t0 = time.perf_counter()
    params = train()
    train_time = time.perf_counter() - t0

    print("\n長さ別の系列一致率(訓練に使った範囲 2..12 の内側):")
    accs = {}
    for L in (2, 4, 6, 8, 10, 12):
        accs[L] = sequence_accuracy(params, L, n_eval=200, seed=1000 + L)
        print("  長さ {:>2}: {:.2f}".format(L, accs[L]))
    print("訓練時間: {:.1f} 秒".format(train_time))

    # デモ: 実際の生成例
    rng_demo = np.random.default_rng(7)
    for L in (4, 12):
        src, tgt = make_data(rng_demo, 1, L)
        pred = generate(params, src, L + 1)
        print("入力 {} → 出力 {} (正解 {})".format(
            ids_to_str(src[0]), ids_to_str(pred[0]), ids_to_str(tgt[0]) + "<eos>"))

    # 6.3 の表を固定する assert(seed 42 で再現。手元の実測: 1.00 / 1.00 / 0.81 / 0.26 / 0.04 / 0.00)
    assert accs[2] >= 0.95, "短い入力はほぼ完璧に解けるはず"
    assert accs[8] <= 0.60, "長さ8ですでに大きく崩れるはず"
    assert accs[12] <= 0.10, "長い入力ではほぼ全滅するはず(固定長ボトルネック)"
    assert accs[2] > accs[8] > accs[12], "長いほど悪い、が観察できるはず"
    assert train_time < 60.0, "実行は60秒以内に収まるはず"
    print("ok: 固定長ボトルネック(長いほど崩れる)を実測で確認しました")
