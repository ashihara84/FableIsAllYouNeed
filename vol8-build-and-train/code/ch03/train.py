# 第8巻 第3章 3.3〜3.4: 訓練ループ本体 — 4拍子の最終形
# forward → loss → backward → update(第3巻4章の4拍子そのまま)+ 検証 loss の監視と
# チェックポイント(第3巻6章の実践)。CPU の極小規模で2分以内に全 assert が通る。
import os
import sys

import torch

from model import (PAD, BOS, EOS, TinyTransformer, label_smoothing_loss,
                   get_device)

# --- データ: 第2章の data.py があればそれを使う -------------------------------
# 第2章(並列執筆中)は make_corpus / make_batches / PAD=0 / BOS=1 / EOS=2 を提供
# する契約。まだ無い場合は、同じ規約の極小ダミー対訳(数列の反転タスク)で代用する。
# 本章はあえて自前のダミー対訳で訓練する(ループの仕組みが主役。実コーパスでの本訓練は第5章)
_CH02 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ch02")
sys.path.insert(0, _CH02)
try:
    from data import make_corpus, make_batches  # noqa: F401
    HAVE_CH02 = True
except ImportError:
    HAVE_CH02 = False


def make_toy_pairs(n_pairs, vocab_size, min_len=3, max_len=8, seed=42):
    """ダミー対訳: src は通常トークン(id 3 以上)のランダム列、tgt はその反転。"""
    g = torch.Generator().manual_seed(seed)
    pairs = []
    for _ in range(n_pairs):
        length = int(torch.randint(min_len, max_len + 1, (1,), generator=g))
        src = torch.randint(3, vocab_size, (length,), generator=g)
        tgt = torch.flip(src, dims=[0])
        pairs.append((src, tgt))
    return pairs


def collate(pairs):
    """対訳ペアの list → (src, tgt_in, tgt_out) の3つのテンソル(PAD 詰め)。

    急所(3.1): tgt_in = [BOS, y1..yn] / tgt_out = [y1..yn, EOS] の1トークンずらし。
    """
    src_len = max(len(s) for s, _ in pairs)
    tgt_len = max(len(t) for _, t in pairs) + 1          # BOS / EOS の分が1つ増える
    n = len(pairs)
    src = torch.full((n, src_len), PAD, dtype=torch.long)
    tgt_in = torch.full((n, tgt_len), PAD, dtype=torch.long)
    tgt_out = torch.full((n, tgt_len), PAD, dtype=torch.long)
    for i, (s, t) in enumerate(pairs):
        src[i, :len(s)] = s
        tgt_in[i, 0] = BOS
        tgt_in[i, 1:len(t) + 1] = t                      # [BOS, y1, ..., yn]
        tgt_out[i, :len(t)] = t
        tgt_out[i, len(t)] = EOS                         # [y1, ..., yn, EOS]
    return src, tgt_in, tgt_out


@torch.no_grad()
def evaluate(model, batches, eps, device):
    """検証データ全体の平均 loss(訓練と同じ損失で測る)。"""
    model.eval()                                          # dropout を切る(第5巻6.4)
    total, count = 0.0, 0
    for src, tgt_in, tgt_out in batches:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        logits = model(src, tgt_in)
        n_tok = int((tgt_out != PAD).sum())
        total += label_smoothing_loss(logits, tgt_out, eps=eps, pad_id=PAD).item() * n_tok
        count += n_tok
    model.train()
    return total / count


@torch.no_grad()
def token_accuracy(model, batches, device):
    """teacher forcing 下の次トークン正解率(PAD 除外)。"""
    model.eval()
    hit, count = 0, 0
    for src, tgt_in, tgt_out in batches:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        pred = model(src, tgt_in).argmax(dim=-1)
        keep = tgt_out != PAD
        hit += int((pred[keep] == tgt_out[keep]).sum())
        count += int(keep.sum())
    model.train()
    return hit / count


def train(model, train_batches, val_batches, n_steps, lr=1e-3, eps_ls=0.1,
          eval_every=25, ckpt_path=None, device=None, log=True):
    """訓練ループの最終形。返り値は (loss 履歴, 検証 loss 履歴 [(step, val_loss)])。"""
    device = device or get_device()
    model.to(device)
    model.train()
    # update の道具。Adam の中身は第4章で完全に開ける(ここでは借り物)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history, val_history = [], []
    best_val = float("inf")
    step = 0
    while step < n_steps:
        for src, tgt_in, tgt_out in train_batches:
            if step >= n_steps:
                break
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

            logits = model(src, tgt_in)                              # 1. forward
            loss = label_smoothing_loss(logits, tgt_out,             # 2. loss
                                        eps=eps_ls, pad_id=PAD)
            optimizer.zero_grad()                                    #    grad のリセット(第5巻5章)
            loss.backward()                                          # 3. backward
            optimizer.step()                                         # 4. update

            history.append(loss.item())
            step += 1

            # --- 検証 loss の監視とチェックポイント(3.4) ---
            if step % eval_every == 0 or step == n_steps:
                val_loss = evaluate(model, val_batches, eps_ls, device)
                val_history.append((step, val_loss))
                if log:
                    print(f"step {step:4d}  train {loss.item():.4f}  val {val_loss:.4f}")
                if val_loss < best_val and ckpt_path is not None:
                    best_val = val_loss
                    torch.save({"step": step, "val_loss": val_loss,
                                "model_state": model.state_dict()}, ckpt_path)
    return history, val_history


if __name__ == "__main__":
    torch.manual_seed(42)
    device = get_device()
    VOCAB = 13          # PAD, BOS, EOS + 通常トークン 10 種
    BATCH = 64

    if HAVE_CH02:
        print("note: ch02 data.py を検出(実コーパスでの本訓練は第5章 train.py で行う)")
    pairs = make_toy_pairs(n_pairs=1280, vocab_size=VOCAB, seed=42)
    train_pairs, val_pairs = pairs[:1024], pairs[1024:]   # 訓練/検証分割(第3巻6章)
    train_batches = [collate(train_pairs[i:i + BATCH])
                     for i in range(0, len(train_pairs), BATCH)]
    val_batches = [collate(val_pairs[i:i + BATCH])
                   for i in range(0, len(val_pairs), BATCH)]

    model = TinyTransformer(vocab_size=VOCAB, d_model=64, h=4, N=2, d_ff=128,
                            p_drop=0.1, max_len=16)
    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint_best.pt")
    history, val_history = train(model, train_batches, val_batches,
                                 n_steps=300, lr=1e-3, eps_ls=0.1,
                                 eval_every=25, ckpt_path=ckpt, device=device)

    # --- 検証: 4拍子が実際に坂を下ったか ---
    first_val, last_val = val_history[0][1], val_history[-1][1]
    best_step, best_val = min(val_history, key=lambda t: t[1])
    assert last_val < 0.5 * first_val, "検証 loss が下がっていない"

    acc = token_accuracy(model, val_batches, device)
    assert acc > 0.90, f"teacher forcing 正解率が低すぎる: {acc:.3f}"

    # --- 検証: チェックポイントから best モデルを復元できるか ---
    assert os.path.exists(ckpt)
    saved = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"])
    restored_val = evaluate(model, val_batches, eps=0.1, device=device)
    assert abs(restored_val - saved["val_loss"]) < 1e-4, "復元したモデルの val loss が一致しない"

    print(f"ok: val loss {first_val:.4f} → {last_val:.4f}, "
          f"accuracy {acc:.3f}, best step {best_step} (val {best_val:.4f}) を復元できました")
