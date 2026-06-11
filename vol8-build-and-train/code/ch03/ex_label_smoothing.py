# 第8巻 第3章 演習1: label smoothing の有無で loss と出力分布の尖り方を比較する
# 論文 5.4 の "This hurts perplexity, as the model learns to be more unsure" を
# 自分の実測で確認する。同条件(同じ初期値・同じデータ)で ε=0 と ε=0.1 を訓練し、
#   - 検証データの素の cross-entropy(ε=0 で測る)と perplexity = exp(CE)
#   - 正解位置に置いた確率の平均と、出力分布のエントロピー(尖り方の物差し)
#   - 次トークン正解率
# を比べる。CPU で2分以内。
import math

import torch
import torch.nn.functional as F

from model import PAD, TinyTransformer, label_smoothing_loss, get_device
from train import make_toy_pairs, collate, train, token_accuracy


@torch.no_grad()
def measure(model, batches, device):
    """素の CE(= ε=0 の損失)・perplexity・正解確率の平均・分布エントロピーの平均。"""
    model.eval()
    ce_sum, p_sum, ent_sum, count = 0.0, 0.0, 0.0, 0
    for src, tgt_in, tgt_out in batches:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        logits = model(src, tgt_in)
        keep = tgt_out != PAD
        ce_sum += label_smoothing_loss(logits, tgt_out, eps=0.0,
                                       pad_id=PAD).item() * int(keep.sum())
        q = F.softmax(logits, dim=-1)                              # (batch, len, vocab)
        p_correct = q.gather(-1, tgt_out.unsqueeze(-1)).squeeze(-1)
        entropy = -(q * torch.log(q + 1e-12)).sum(dim=-1)          # 各位置の H(q)
        p_sum += float(p_correct[keep].sum())
        ent_sum += float(entropy[keep].sum())
        count += int(keep.sum())
    ce = ce_sum / count
    return {"ce": ce, "ppl": math.exp(ce),
            "p_correct": p_sum / count, "entropy": ent_sum / count}


def run(eps_ls, train_batches, val_batches, device, n_steps=600):
    torch.manual_seed(42)   # 両者を同じ初期値から出発させる
    model = TinyTransformer(vocab_size=13, d_model=64, h=4, N=2, d_ff=128,
                            p_drop=0.1, max_len=16)
    train(model, train_batches, val_batches, n_steps=n_steps, lr=1e-3,
          eps_ls=eps_ls, eval_every=100, ckpt_path=None, device=device, log=False)
    stats = measure(model, val_batches, device)
    stats["acc"] = token_accuracy(model, val_batches, device)
    return stats


if __name__ == "__main__":
    device = get_device()
    pairs = make_toy_pairs(n_pairs=1280, vocab_size=13, seed=42)
    train_pairs, val_pairs = pairs[:1024], pairs[1024:]
    train_batches = [collate(train_pairs[i:i + 64]) for i in range(0, 1024, 64)]
    val_batches = [collate(val_pairs[i:i + 64]) for i in range(0, 256, 64)]

    plain = run(eps_ls=0.0, train_batches=train_batches,
                val_batches=val_batches, device=device)
    smooth = run(eps_ls=0.1, train_batches=train_batches,
                 val_batches=val_batches, device=device)

    print(f"{'(検証データ)':<22}{'ε=0':>10}{'ε=0.1':>10}")
    print(f"{'素の cross-entropy':<22}{plain['ce']:>10.4f}{smooth['ce']:>10.4f}")
    print(f"{'perplexity':<22}{plain['ppl']:>10.4f}{smooth['ppl']:>10.4f}")
    print(f"{'正解確率の平均':<22}{plain['p_correct']:>10.4f}{smooth['p_correct']:>10.4f}")
    print(f"{'分布エントロピー':<22}{plain['entropy']:>10.4f}{smooth['entropy']:>10.4f}")
    print(f"{'次トークン正解率':<22}{plain['acc']:>10.4f}{smooth['acc']:>10.4f}")

    # --- 論文 5.4 の主張の確認 ---
    # (1) "This hurts perplexity": smoothing ありの方が perplexity は悪い
    assert smooth["ppl"] > plain["ppl"], "perplexity が悪化していない"
    # (2) "the model learns to be more unsure": 分布の尖りが弱い
    #     (正解確率は低く、エントロピーは高い)
    assert smooth["p_correct"] < plain["p_correct"]
    assert smooth["entropy"] > plain["entropy"]
    # (3) それでも「どれを選ぶか」はほぼ壊れていない(accuracy は同水準)
    assert smooth["acc"] > 0.90 and plain["acc"] > 0.90
    print("ok: smoothing は perplexity を悪化させ、分布をなだらかにし、"
          "正解率はほぼ保たれました")
