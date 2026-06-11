# 第8巻 第1章 1.4: 自作スタックの限界を実測する
# 論文 base model と同じ構成(N=6, d_model=512, d_ff=2048, h=8, 語彙37000)を
# 自作スタックで1台組み、訓練1ステップの所要時間を実測する。
# そこから「論文と同じ訓練(25000トークン/step × 100,000 step)」の総時間を見積もり、
# 物理的に終わらないことを数字で確認する。PyTorch への需要はここで初めて発生する。
# 実行: python3 limit_check.py(1分弱。メモリを 1GB 強使う)
import time

import numpy as np

from transformer import Transformer
from tensor_autograd import softmax_cross_entropy  # 第5巻5章(path は transformer が通す)

rng = np.random.default_rng(42)

# 論文 3.1 Table 3 base / 5.1 の構成そのまま
vocab, d_model, d_ff, h, N = 37000, 512, 2048, 8, 6
src_len = tgt_len = 32                     # 1ステップに流す系列(バッチはこの1対だけ)

print("base model(N=6, d_model=512, d_ff=2048, h=8, 語彙{:,})を組み立て中...".format(vocab))
t0 = time.perf_counter()
model = Transformer(vocab, d_model, d_ff, h, N, max_len=64, rng=rng)
params = model.params()
n_params = model.n_params()
print("  組み立て {:.1f} 秒、パラメータ {:,}(≈ {:.0f}M — 第7巻6章の検算と同じ規模)"
      .format(time.perf_counter() - t0, n_params, n_params / 1e6))
assert n_params > 60e6                     # 確かに「論文サイズ」を持ち上げている

src = rng.integers(1, vocab, size=src_len)
tgt = rng.integers(1, vocab, size=tgt_len)
tgt_in = np.concatenate([[0], tgt[:-1]])


def one_step():
    """4拍子1回 = 訓練1ステップ(forward → loss → backward → update)。"""
    loss = softmax_cross_entropy(model.forward_tensor(src, tgt_in), tgt)
    for p in params:
        p.grad = np.zeros_like(p.data)
    loss.backward()
    for p in params:
        p.data -= 1e-4 * p.grad            # 更新も計測に含める(本番は毎step行うので)
    return loss.data


one_step()                                 # ウォームアップ(初回はキャッシュ等で遅い)
n_trials = 3
t0 = time.perf_counter()
for _ in range(n_trials):
    one_step()
t_step = (time.perf_counter() - t0) / n_trials
print("  訓練1ステップ(32トークンの文対1本): {:.2f} 秒".format(t_step))
assert t_step > 0.0

# --- 見積もり: 論文 5.1・5.3 の訓練量に外挿する ---
# 5.1: 1バッチ ≈ 25000 ターゲットトークン。5.3: base は 100,000 ステップ。
# 自作スタックは1ステップで 32 ターゲットトークンしか処理していないので、
# 同じトークン量を流すには 25000/32 ≈ 781 倍の時間がかかる(線形外挿。
# 実際は系列が長いほど attention が O(n^2) で重くなるから、これでも甘めの見積もり)
tokens_paper = 25000
steps_paper = 100000
scale = tokens_paper / tgt_len
total_sec = t_step * scale * steps_paper
total_days = total_sec / 86400
total_years = total_days / 365.0

print()
print("  論文の訓練量: {:,} トークン/step × {:,} step".format(tokens_paper, steps_paper))
print("  必要時間 = {:.2f} 秒 × {:.0f} × {:,} = {:.2e} 秒".format(
    t_step, scale, steps_paper, total_sec))
print("  = 約 {:,.0f} 日 = 約 {:.1f} 年(このマシン1台、自作スタックで)".format(
    total_days, total_years))
print("  論文の実測: 8 × P100 GPU で 12 時間(5.2)")

assert total_days > 30                     # どんなに速いマシンでも「月」では終わらない
print()
print("limit_check: すべての assert を通過しました — 物理的に終わらない。卒業の時です")
