# 第6巻 第5章 5.3 / 5.4: RNN の2つの痛みを実測する
# 痛み1(5.3): 仕事の総量(トークン数)を固定したまま系列長 L を変えると、
#             1ステップの時間が L に比例して伸びる(時間方向は並列化できない)。
# 痛み2(5.4): 最後の位置の損失から、距離 d だけ離れた入力への勾配ノルムが
#             指数的に減衰する(第5巻6.1の勾配消失の、系列方向での再演)。
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul")  # rnn_lm.py と同じ誤報対策

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, "..", "..", "..", "vol5-backprop", "code", "ch05"))
from tensor_autograd import Tensor, softmax_cross_entropy  # noqa: E402
from rnn_lm import CORPUS, build_vocab, init_params, onehot, tanh  # noqa: E402

rng = np.random.default_rng(42)
V = len(build_vocab(CORPUS)[0])     # 5.2 と同じ語彙サイズ(29)
d_e, d_h = 24, 48                   # 5.2 と同じモデルサイズ
params = init_params(rng, V, d_e=d_e, d_h=d_h)


# === 痛み1: 並列化できない —— 総トークン数を固定して L だけ変える ===

def one_step_time(B, L, reps=7):
    """訓練1ステップ(forward + backward)の時間を測る。揺れ対策に reps 回の最小値。"""
    X = rng.integers(0, V, size=(B, L))
    Y = rng.integers(0, V, size=(B, L))
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        h = Tensor(np.zeros((B, d_h)))
        loss = Tensor(0.0)
        for t in range(L):                              # ← この回数だけは絶対に減らせない
            x_t = Tensor(onehot(X[:, t], V)) @ params["E"]
            h = tanh(x_t @ params["W_x"] + h @ params["W_h"] + params["b"])
            logits = h @ params["W_out"] + params["b_out"]
            loss = loss + softmax_cross_entropy(logits, Y[:, t])
        loss = loss * (1.0 / L)
        for p in params.values():
            p.grad = np.zeros_like(p.data)
        loss.backward()
        best = min(best, time.perf_counter() - t0)
    return best


print("=== 痛み1: 総トークン数 B×L = 512 を固定し、系列長 L だけ変える ===")
times = {}
for B, L in [(64, 8), (32, 16), (16, 32), (8, 64), (4, 128)]:
    times[L] = one_step_time(B, L)
    print("B = %3d, L = %3d  ->  1ステップ %6.2f ms(対 L=8 比 %.1f 倍)"
          % (B, L, times[L] * 1000, times[L] / times[8]))

# 比較台: 同じ 512 トークンを「1発の行列積」に流せたなら(時間方向の依存が無い世界)
X_flat = Tensor(onehot(rng.integers(0, V, size=512), V))    # (512, V)
t0 = time.perf_counter()
h_flat = tanh(X_flat @ params["E"] @ params["W_x"])         # (512, d_h) を一斉に
logits = h_flat @ params["W_out"] + params["b_out"]
loss = softmax_cross_entropy(logits, rng.integers(0, V, size=512))
loss.backward()
t_flat = time.perf_counter() - t0
print("(参考)同じ512トークンを1発の行列積で:  %6.2f ms" % (t_flat * 1000))

# 仕事の総量は同じなのに、細切れの回数 L に応じて時間が伸びることを固定する
assert times[16] > times[8] and times[32] > times[16]
assert times[64] > times[32] and times[128] > times[64]
assert times[128] > 4 * times[8], "L を16倍にしたのに時間が4倍も伸びていない?"
assert t_flat < times[8], "一斉の行列積が最短の RNN より遅いのはおかしい"


# === 痛み2: 長距離依存 —— 最後の損失から、遠い入力への勾配ノルム ===

print("\n=== 痛み2: 最後の位置の損失 → 距離 d 離れた入力への勾配ノルム ===")
B, L = 32, 64
X = rng.integers(0, V, size=(B, L))
h = Tensor(np.zeros((B, d_h)))
xs = []                                                 # 各位置の入力ベクトルを覚えておく
for t in range(L):
    x_t = Tensor(onehot(X[:, t], V)) @ params["E"]      # (B, d_e)
    xs.append(x_t)
    h = tanh(x_t @ params["W_x"] + h @ params["W_h"] + params["b"])
logits = h @ params["W_out"] + params["b_out"]          # 損失は最後の1位置だけに置く
loss = softmax_cross_entropy(logits, rng.integers(0, V, size=B))
for p in params.values():
    p.grad = np.zeros_like(p.data)
loss.backward()

# 距離 d = 「損失の位置から何文字前の入力か」。d=1 が直前の入力
norms = {}
for d in [1, 2, 4, 8, 16, 32, 64]:
    g = xs[L - d].grad                                  # (B, d_e)
    norms[d] = float(np.linalg.norm(g, axis=1).mean())  # 1例あたりの勾配ノルムの平均
    print("距離 %2d:  ||∂loss/∂x|| = %.2e(対 d=1 比 %.4f)"
          % (d, norms[d], norms[d] / norms[1]))

# 減衰を assert で固定: 遠いほど小さく、距離64では2桁以上消えている
assert norms[1] > norms[16] > norms[64]
assert norms[64] < 0.05 * norms[1], "距離64の勾配が思ったほど減衰していない?"

print("\nok: 痛み1(時間は L に比例して伸びる)・痛み2(勾配は距離で減衰する)を確認しました")
