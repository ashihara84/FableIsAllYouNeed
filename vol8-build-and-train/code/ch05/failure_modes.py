# 第8巻 第5章 5.4: うまくいかない時の手引き — バグを仕込んで症状を観察する
#
# 訓練がうまくいかない時の3大典型(mask 漏れ / learning rate / ずらし忘れ)を、
# 健常な訓練と同一条件(同じデータ・同じ初期値・同じ 1000 ステップ = 5.1 と同じ)で
# 1つずつ実際に仕込み、どんな「症状」が出るかを観察する。
# 各実験の最後の assert が「症状の再現」そのものになっている。
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _ch in ("ch02", "ch03"):
    _p = os.path.normpath(os.path.join(_HERE, "..", _ch))
    if _p not in sys.path:
        sys.path.append(_p)   # 末尾に足す(ch03 にも train.py があり、先頭だと衝突する)

from data import PAD, encode_pair, decode, make_batches, vocab_size   # 第2章
from model import TinyTransformer, label_smoothing_loss, pad_mask     # 第3章
from train import load_data, lrate, token_accuracy, pick_device       # 5.1
from generate import greedy_decode                                    # 5.2


# --- バグ1: causal mask の漏れ -----------------------------------------------
class LeakyTransformer(TinyTransformer):
    """第3章の forward から causal_mask を「足し忘れた」状態を再現する。

    第3章のファイルは変更せず、forward だけ差し替える。tgt_mask が pad_mask
    だけになっている——これが今回仕込むバグのすべて。
    """

    def forward(self, src_ids, tgt_in_ids):
        src_mask = pad_mask(src_ids)
        tgt_mask = pad_mask(tgt_in_ids)          # バグ: + causal_mask(...) を忘れた
        memory = self.encode(src_ids, src_mask)
        y = self.decode(tgt_in_ids, memory, tgt_mask, src_mask)
        return y @ self.embed.weight.T


def run(name, model_cls=TinyTransformer, lr_mode="schedule", shift_bug=False,
        epochs=125, batch_size=32, seed=42, device=None, log=True):
    """健常版と同一条件の短い訓練。lr_mode は "schedule" か定数。

    返り値: (model, [(step, エポック平均 loss), ...])
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or pick_device()
    train_pairs, _, _, _ = load_data()
    model = model_cls(vocab_size, d_model=128, h=4, N=2,
                      d_ff=256, p_drop=0.1, max_len=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

    history, step = [], 0
    for epoch in range(1, epochs + 1):
        batches, _ = make_batches(train_pairs, batch_size, rng=rng)
        model.train()
        epoch_losses = []
        for src, tgt in batches:
            step += 1
            lr = lrate(step) if lr_mode == "schedule" else lr_mode
            for group in opt.param_groups:
                group["lr"] = lr
            src = torch.from_numpy(src).to(device)
            tgt = torch.from_numpy(tgt).to(device)
            tgt_in = tgt[:, :-1]
            if shift_bug:
                tgt_out = tgt[:, :-1]            # バグ: [:, 1:] と書くべき所のずらし忘れ
            else:
                tgt_out = tgt[:, 1:]             # 正しい1トークンずらし(3.1)
            loss = label_smoothing_loss(model(src, tgt_in), tgt_out,
                                        eps=0.1, pad_id=PAD)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        if epoch == 1 or epoch % 25 == 0:
            history.append((step, float(np.mean(epoch_losses))))
    if log:
        print("[%s]" % name)
        print("  loss: " + "  ".join("step%d %.3f" % (s, l) for s, l in history))
    return model, history


def greedy_check(model, raw_pairs, device, n=30):
    """コーパス全体から等間隔に n ペア選び、greedy 完全一致率を測る。"""
    model.eval()
    sample = raw_pairs[::max(1, len(raw_pairs) // n)][:n]
    hit = 0
    for s, t in sample:
        src_ids, _ = encode_pair(s, t)
        if decode(greedy_decode(model, src_ids, device)) == t:
            hit += 1
    return hit / len(sample)


if __name__ == "__main__":
    device = pick_device()
    train_pairs, _, train_raw, _ = load_data()
    demo = ("he eats fish", "かれ は さかな を たべます")

    # --- 基準: 健常な訓練(5.1 と同じ 1000 ステップ) ---------------------------
    model_ok, hist_ok = run("健常(基準)")
    acc_tf = token_accuracy(model_ok, train_pairs, device)
    acc_gen = greedy_check(model_ok, train_raw, device)
    print("  teacher forcing 正解率 %.3f / greedy 完全一致 %.3f" % (acc_tf, acc_gen))
    loss_ok = hist_ok[-1][1]
    assert acc_tf > 0.95 and acc_gen > 0.75

    # --- 症状A: loss は見事に下がるのに、生成すると壊滅 ------------------------
    # 原因: decoder の causal mask 漏れ(訓練中だけ未来=正解が見えている)
    print()
    model_a, hist_a = run("バグ1: causal mask 漏れ", model_cls=LeakyTransformer)
    acc_tf_a = token_accuracy(model_a, train_pairs, device)
    acc_gen_a = greedy_check(model_a, train_raw, device)
    src_ids, _ = encode_pair(*demo)
    print("  teacher forcing 正解率 %.3f / greedy 完全一致 %.3f" % (acc_tf_a, acc_gen_a))
    print("  例: %r -> %r" % (demo[0], decode(greedy_decode(model_a, src_ids, device))))
    assert hist_a[-1][1] < loss_ok + 0.1      # loss は健常版と同等以下(悪くはならない)
    assert acc_tf_a > 0.95                    # teacher forcing の正解率も満点近く「見える」
    assert acc_gen_a < 0.3                    # しかし自力で生成させると壊滅

    # --- 症状B: loss が発散する/高止まりする ---------------------------------
    # 原因: learning rate が大きすぎる(warmup なしの定数 lr = 0.5)
    print()
    model_b, hist_b = run("バグ2a: lr 大きすぎ(定数 0.5)", lr_mode=0.5)
    final_b = hist_b[-1][1]
    assert math.isnan(final_b) or final_b > 3.0   # 当てずっぽう(ln 275 ≈ 5.6)前後で迷走

    # --- 症状B': loss が(ほとんど)下がらない --------------------------------
    # 原因: learning rate が小さすぎる(定数 1e-6)
    model_c, hist_c = run("バグ2b: lr 小さすぎ(定数 1e-6)", lr_mode=1e-6)
    # 1000 ステップ使い切っても、当てずっぽうの loss(ln 275 ≈ 5.6)にすら届かない
    assert hist_c[-1][1] > math.log(vocab_size)

    # --- 症状C: 同じ語を延々と繰り返す -----------------------------------------
    # 原因: 1トークンずらしの忘れ(tgt_out に tgt_in と同じ列を渡している)
    print()
    model_d, hist_d = run("バグ3: ずらし忘れ", shift_bug=True)
    out_ids = greedy_decode(model_d, src_ids, device)
    out_toks = decode(out_ids)
    print("  例: %r -> ids %s..." % (demo[0], out_ids[:8]))
    print("     (decode すると %r — 特殊トークンは読み飛ばされる)" % out_toks)
    assert hist_d[-1][1] < loss_ok                # loss だけ見れば絶好調(コピーは簡単)
    assert len(set(out_ids)) <= 2                 # しかし生成は同じトークンの繰り返し

    print("\nok: 3つのバグの症状(下がりすぎて壊滅 / 発散・停滞 / 繰り返し)を再現")
