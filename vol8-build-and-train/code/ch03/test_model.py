# 第8巻 第3章: model.py の契約テスト
# python3 test_model.py で全 assert が通ること。CPU の極小規模で数秒。
import torch
import torch.nn.functional as F

from model import (PAD, BOS, EOS, TinyTransformer, label_smoothing_loss,
                   get_device)

torch.manual_seed(42)
VOCAB = 20


def make_model():
    torch.manual_seed(42)
    m = TinyTransformer(vocab_size=VOCAB, d_model=32, h=4, N=2, d_ff=64,
                        p_drop=0.1, max_len=16)
    m.eval()  # dropout を切って決定的に
    return m


# --- (1) shape の契約: forward(src, tgt_in) → (batch, tgt_len, vocab) ---
model = make_model()
src = torch.tensor([[5, 6, 7, 8, PAD], [9, 10, PAD, PAD, PAD]])
tgt_in = torch.tensor([[BOS, 5, 6, 7], [BOS, 9, 10, PAD]])
logits = model(src, tgt_in)
assert logits.shape == (2, 4, VOCAB)

# --- (2) causal mask: 位置 j の入力を変えても、位置 < j の logits は変わらない ---
tgt_in2 = tgt_in.clone()
tgt_in2[0, 3] = 13                     # 末尾(位置3)だけ変える
logits2 = model(src, tgt_in2)
assert torch.allclose(logits[0, :3], logits2[0, :3], atol=1e-5), "未来が過去に漏れている"
assert not torch.allclose(logits[0, 3], logits2[0, 3], atol=1e-5), "自分の位置には効くはず"

# --- (3) pad mask: src の末尾に PAD を足しても出力は変わらない ---
src_more_pad = torch.cat([src, torch.full((2, 3), PAD)], dim=1)   # (2, 8)
logits3 = model(src_more_pad, tgt_in)
assert torch.allclose(logits, logits3, atol=1e-5), "PAD が encoder 経由で漏れている"

# --- (4) weight sharing: 出力射影は埋め込み E と同一物(専用パラメータを持たない) ---
n_emb = model.embed.weight.numel()
n_total = sum(p.numel() for p in model.parameters())
with torch.no_grad():
    model.embed.weight.zero_()
zeroed = model(src, tgt_in)
assert zeroed.abs().max().item() == 0.0, "E を 0 にしたら logits も 0 のはず(共有の証拠)"
assert n_emb == VOCAB * 32 and n_total > n_emb

# --- (5) label_smoothing_loss: eps=0 は素の cross-entropy(PAD 除外)と一致 ---
model = make_model()
logits = model(src, tgt_in)
tgt_out = torch.tensor([[5, 6, 7, EOS], [9, 10, EOS, PAD]])
ls0 = label_smoothing_loss(logits, tgt_out, eps=0.0, pad_id=PAD)
ce = F.cross_entropy(logits.reshape(-1, VOCAB), tgt_out.reshape(-1),
                     ignore_index=PAD)
assert torch.allclose(ls0, ce, atol=1e-6), "eps=0 で素の cross-entropy に一致しない"

# --- (6) PAD 位置の除外: targets が PAD の位置の logits を変えても損失は不変 ---
logits_perturbed = logits.clone()
logits_perturbed[1, 3] += 100.0        # tgt_out[1,3] == PAD の位置
l_a = label_smoothing_loss(logits, tgt_out, eps=0.1, pad_id=PAD)
l_b = label_smoothing_loss(logits_perturbed, tgt_out, eps=0.1, pad_id=PAD)
assert torch.allclose(l_a, l_b), "PAD 位置が損失に混入している"

# --- (7) smoothing の向き: 正解に確信を持ちすぎた分布では eps>0 の方が損失が大きい ---
sharp = torch.full((1, 1, VOCAB), -20.0)
sharp[0, 0, 5] = 20.0                  # ほぼ one-hot な自信過剰の出力
t = torch.tensor([[5]])
assert label_smoothing_loss(sharp, t, eps=0.1) > label_smoothing_loss(sharp, t, eps=0.0)

# --- (8) デバイスのフォールバック関数が動く ---
dev = get_device()
assert dev.type in ("cuda", "mps", "cpu")

print("ok: test_model.py の assert がすべて通りました(device:", str(dev) + ")")
