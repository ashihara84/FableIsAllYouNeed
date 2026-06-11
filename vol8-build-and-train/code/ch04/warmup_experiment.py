# 第8巻 第4章 4.6: warmup スケジュール(第2巻終章の式)の実装と、
# 「warmup を切ると訓練が壊れる」ことの実験的確認
#
# 実験には自前の極小 MiniTransformer を使う(第3章 model.py と同じ post-LN 構成。
# 本章単体で再現できるよう独立させてある。本番モデルでの訓練は第5章)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------
# 論文 5.3 の learning rate スケジュール(第2巻終章で読んだ式そのまま)
# lrate = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
# ---------------------------------------------------------------
def lrate(step, d_model, warmup_steps):
    return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)


def lrate_no_warmup(step, d_model, warmup_steps):
    # warmup を切る = min の中の「増える候補」を消し、減衰項だけを残す
    # step=1 から d_model^{-0.5} = 0.125 という大股で歩き出すことになる
    return d_model ** -0.5 * step ** -0.5


# 交差点の確認(第2巻終章の表の再現): step = warmup_steps で2候補が一致する
assert math.isclose(lrate(100, 64, 100), 64 ** -0.5 * 100 ** -0.5)
assert lrate(1, 64, 100) < lrate(100, 64, 100)    # 序盤は小さい(助走)
assert lrate(400, 64, 100) < lrate(100, 64, 100)  # 終盤は減衰


# ---------------------------------------------------------------
# 実験対象の小さな Transformer(post-LN・単一ヘッド・2層)
# 論文と同じ構成の縮小版。完成品モジュール(nn.Transformer 等)は使わない
# ---------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # 式(1)
        x = self.ln1(x + self.wo(F.softmax(scores, dim=-1) @ v))   # post-LN(論文の構成)
        x = self.ln2(x + self.ff2(torch.relu(self.ff1(x))))
        return x


class MiniTransformer(nn.Module):
    def __init__(self, vocab=32, seq_len=16, d_model=64, d_ff=128, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.zeros(seq_len, d_model))  # 学習する位置埋め込み(縮小版の簡略化)
        self.blocks = nn.ModuleList([Block(d_model, d_ff) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab)

    def forward(self, tokens):  # tokens: (batch, seq_len)
        x = self.embed(tokens) + self.pos
        for block in self.blocks:
            x = block(x)
        return self.head(x)  # (batch, seq_len, vocab)


# ---------------------------------------------------------------
# 訓練: コピータスク(入力列をそのまま出力する)。題材は何でもよい —
# 比べたいのはスケジュールの有無だけなので、データと初期値は両者で完全に揃える
# ---------------------------------------------------------------
VOCAB, SEQ_LEN, D_MODEL = 32, 16, 64
STEPS, WARMUP = 400, 100


def train(schedule_fn):
    torch.manual_seed(42)  # 初期値を両者で揃える
    model = MiniTransformer(vocab=VOCAB, seq_len=SEQ_LEN, d_model=D_MODEL)
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    g = torch.Generator().manual_seed(7)  # データも両者で揃える

    losses = []
    for step in range(1, STEPS + 1):
        for group in opt.param_groups:  # 毎ステップ lr を式から計算し直す(5.3 の実装)
            group["lr"] = schedule_fn(step, D_MODEL, WARMUP)
        tokens = torch.randint(0, VOCAB, (64, SEQ_LEN), generator=g)
        logits = model(tokens)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), tokens.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


print("--- warmup あり(論文 5.3 の式そのまま)---")
losses_w = train(lrate)
print(f"loss: step1 {losses_w[0]:.3f} -> step100 {losses_w[99]:.3f} -> step400 {losses_w[-1]:.3f}")

print("--- warmup なし(min の増える候補を消した式)---")
losses_n = train(lrate_no_warmup)
print(f"loss: step1 {losses_n[0]:.3f} -> step100 {losses_n[99]:.3f} -> step400 {losses_n[-1]:.3f}")

# ---------------------------------------------------------------
# 確認: warmup ありは学習が成立し、なしは壊れる
# (この規模では NaN にはならず「序盤に何も学べず、最後まで桁違いに悪い」
#  という壊れ方をする。発散するかどうかは規模と運に依る — 本文 4.6 参照)
# ---------------------------------------------------------------
baseline = math.log(VOCAB)  # 当てずっぽうの loss = ln 32 ≈ 3.47
final_w, final_n = losses_w[-1], losses_n[-1]

# warmup あり: タスクをほぼ完全に解いている
assert final_w < 0.1, f"warmup ありの訓練が想定どおり進んでいません: {final_w}"
# warmup なし: 最初の100ステップは当てずっぽう近傍から動けない
assert math.isnan(losses_n[99]) or losses_n[99] > baseline * 0.9, \
    f"warmup なしが序盤から学習できてしまいました: {losses_n[99]}"
# warmup なし: 400ステップ後も loss が桁違いに高いまま(または NaN)
assert math.isnan(final_n) or final_n > 1.0, \
    f"warmup なしでも学習できてしまいました: {final_n}"

print(f"\n当てずっぽうの loss(ln {VOCAB}): {baseline:.3f}")
print(f"warmup あり: {final_w:.4f}(タスクをほぼ解いた)")
print(f"warmup なし: {final_n if math.isnan(final_n) else round(final_n, 4)}(壊れた — 序盤の損傷を最後まで引きずる)")
print("ok: warmup を切ると訓練が壊れる(桁違いに悪化する)ことを確認しました")
