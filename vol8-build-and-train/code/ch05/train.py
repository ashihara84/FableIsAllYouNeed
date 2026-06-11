# 第8巻 第5章 5.1: 縮小版 Transformer の訓練実行 — 小さく回して観察する
#
# 第2章の data.py(コーパス・バッチ)と第3章の model.py(TinyTransformer・
# label smoothing)を import し、第4章の learning rate スケジュールで訓練する。
# 規模は「論文の千分の一」: 対訳 250 ペア、語彙 275、パラメータ約 70 万。
# CPU で数分以内に終わる。実行すると loss の数表を表示し、assert で
# 「loss が大きく下がる」「訓練データをほぼ暗記できる」ことを確認する。
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _ch in ("ch02", "ch03"):
    _p = os.path.normpath(os.path.join(_HERE, "..", _ch))
    if _p not in sys.path:
        sys.path.append(_p)   # 末尾に足す(ch03 にも train.py があり、先頭だと衝突する)

from data import PAD, BOS, EOS, make_corpus, encode_pair, decode, \
    make_batches, vocab_size                                       # 第2章
from model import TinyTransformer, label_smoothing_loss, get_device  # 第3章

CKPT_PATH = os.path.join(_HERE, "ch05_checkpoint.pt")
EPS_LS = 0.1      # 論文 5.4 の ε_ls
WARMUP = 400      # 総ステップ 1000 に対する warmup(論文は 4000 / 約10万ステップ。5.1節)
D_MODEL = 128


def pick_device():
    """計算装置の選択。既定は第3章の get_device()(cuda → mps → cpu)。

    環境変数 FABLE_DEVICE で固定できる。本文の実行例は FABLE_DEVICE=cpu で
    採取した——この規模では GPU との速度差がほぼなく、CPU は実行のたびに
    ビットまで同じ結果を返す(GPU は並列和の順序が揺れて結果が微妙に変わる)。
    """
    name = os.environ.get("FABLE_DEVICE")
    return torch.device(name) if name else get_device()


def lrate(step, d_model=D_MODEL, warmup_steps=WARMUP):
    """第4章 4.6(= 論文 5.3 式(3))の learning rate スケジュール。再掲。"""
    return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)


def load_data(n_test=25, seed=42):
    """第2章のコーパスを ID 化し、訓練 225 / 未見(テスト)25 ペアに分ける。"""
    corpus = make_corpus()
    encoded = [encode_pair(s, t) for s, t in corpus]
    rng = np.random.default_rng(seed)
    test_idx = set(rng.permutation(len(corpus))[:n_test].tolist())
    train_pairs = [encoded[i] for i in range(len(corpus)) if i not in test_idx]
    test_pairs = [encoded[i] for i in sorted(test_idx)]
    train_raw = [corpus[i] for i in range(len(corpus)) if i not in test_idx]
    test_raw = [corpus[i] for i in sorted(test_idx)]
    return train_pairs, test_pairs, train_raw, test_raw


def shift(tgt, device):
    """第3章 3.1 の1トークンずらし。tgt (B, L) → (tgt_in, tgt_out)。

    tgt の各行は [BOS, y1..yn, EOS, PAD...]。
    tgt_in = [BOS, y1..yn, ...] / tgt_out = [y1..yn, EOS, ...](PAD は損失で無視)。
    """
    tgt = torch.from_numpy(tgt).to(device)
    return tgt[:, :-1], tgt[:, 1:]


@torch.no_grad()
def mean_loss(model, pairs, device):
    """与えたペア集合の平均 loss(label smoothing 付き。訓練と同じ物差し)。"""
    model.eval()
    rng = np.random.default_rng(0)
    batches, _ = make_batches(pairs, batch_size=len(pairs), rng=rng)
    total, count = 0.0, 0
    for src, tgt in batches:
        src = torch.from_numpy(src).to(device)
        tgt_in, tgt_out = shift(tgt, device)
        n_tok = int((tgt_out != PAD).sum())
        total += label_smoothing_loss(model(src, tgt_in), tgt_out,
                                      eps=EPS_LS, pad_id=PAD).item() * n_tok
        count += n_tok
    model.train()
    return total / count


@torch.no_grad()
def token_accuracy(model, pairs, device):
    """teacher forcing 下の次トークン正解率(PAD 除外)。暗記の進み具合の物差し。"""
    model.eval()
    rng = np.random.default_rng(0)
    batches, _ = make_batches(pairs, batch_size=len(pairs), rng=rng)
    hit, count = 0, 0
    for src, tgt in batches:
        src = torch.from_numpy(src).to(device)
        tgt_in, tgt_out = shift(tgt, device)
        pred = model(src, tgt_in).argmax(dim=-1)
        keep = tgt_out != PAD
        hit += int((pred[keep] == tgt_out[keep]).sum())
        count += int(keep.sum())
    model.train()
    return hit / count


def train_model(epochs=125, batch_size=32, seed=42, device=None, log=True):
    """訓練本体。第3章の4拍子 + 第4章のスケジュール。返り値は (model, 数表の行リスト)。"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or pick_device()

    train_pairs, test_pairs, _, _ = load_data()
    model = TinyTransformer(vocab_size, d_model=D_MODEL, h=4, N=2,
                            d_ff=256, p_drop=0.1, max_len=64).to(device)
    # 論文 5.3 と同じ Adam の設定。lr は毎ステップ式から上書きするので初期値はダミー
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

    if log:
        n_params = sum(p.numel() for p in model.parameters())
        print("device=%s  params=%s  train=%d pairs  test=%d pairs" %
              (device, format(n_params, ","), len(train_pairs), len(test_pairs)))
        print()
        print("%5s  %9s  %9s  %9s  %6s" %
              ("step", "lr", "loss(訓練)", "loss(未見)", "正解率"))

    rows = []
    step = 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        batches, _ = make_batches(train_pairs, batch_size, rng=rng)  # 長さ順(2.3節)
        model.train()
        for src, tgt in batches:
            step += 1
            lr = lrate(step)
            for group in opt.param_groups:
                group["lr"] = lr
            src = torch.from_numpy(src).to(device)
            tgt_in, tgt_out = shift(tgt, device)               # ずらし(3.1)
            logits = model(src, tgt_in)                        # 1. forward
            loss = label_smoothing_loss(logits, tgt_out,       # 2. loss
                                        eps=EPS_LS, pad_id=PAD)
            opt.zero_grad()
            loss.backward()                                    # 3. backward
            opt.step()                                         # 4. update
        if epoch == 1 or epoch % 12 == 0 or epoch == epochs:
            tr = mean_loss(model, train_pairs, device)
            te = mean_loss(model, test_pairs, device)
            acc = token_accuracy(model, train_pairs, device)
            rows.append((step, lr, tr, te, acc))
            if log:
                print("%5d  %9.6f  %9.3f  %9.3f  %6.3f" % (step, lr, tr, te, acc))
    if log:
        print("\n訓練時間: %.1f 秒" % (time.time() - t0))
    return model, rows


def load_or_train(device=None):
    """チェックポイントがあれば復元、なければ訓練して保存(5.2 以降の入口)。"""
    device = device or pick_device()
    model = TinyTransformer(vocab_size, d_model=D_MODEL, h=4, N=2,
                            d_ff=256, p_drop=0.1, max_len=64).to(device)
    if os.path.exists(CKPT_PATH):
        state = torch.load(CKPT_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state)
    else:
        model, _ = train_model(device=device, log=False)
        torch.save(model.state_dict(), CKPT_PATH)
    model.eval()
    return model, device


if __name__ == "__main__":
    device = pick_device()
    model, rows = train_model(device=device)

    # --- assert 1: loss が大きく下がる(最初の記録の 1/5 以下) ---
    first_tr, last_tr = rows[0][2], rows[-1][2]
    assert last_tr < first_tr / 5, "訓練 loss が 1/5 以下まで下がっていない"

    # --- assert 2: 暗記の達成(teacher forcing 下のトークン正解率 95% 以上) ---
    train_pairs, test_pairs, _, _ = load_data()
    acc_train = token_accuracy(model, train_pairs, device)
    assert acc_train > 0.95, "訓練データを暗記できていない: %.3f" % acc_train

    # --- 観察: label smoothing の床(暗記し切っても loss(LS) はゼロにならない) ---
    print("最終 loss(訓練) = %.3f / 素の cross-entropy なら暗記でほぼ 0 になるが、" % last_tr)
    print("label smoothing が ε=0.1 ぶんの床を作る(3.2節・演習で見た現象の再確認)")

    torch.save(model.state_dict(), CKPT_PATH)
    print("checkpoint saved: %s" % os.path.basename(CKPT_PATH))
    print("ok: loss %.3f → %.3f(1/%d)、訓練データの次トークン正解率 %.3f" %
          (first_tr, last_tr, round(first_tr / last_tr), acc_train))
