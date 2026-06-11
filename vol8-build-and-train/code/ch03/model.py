# 第8巻 第3章: TinyTransformer — 第4〜6章が import する共有モデル
#
# 第7巻で部品単位に自作・テスト済みの Transformer を、PyTorch の素の部品
# (nn.Linear / nn.Embedding / nn.LayerNorm / nn.Dropout / F.softmax)だけで
# 再実装したもの。nn.Transformer / nn.MultiheadAttention 等の完成品は使わない
# (使ったら第7巻が無意味になる)。各部品の「自作版のどれにあたるか」は
# 第1章1.5の API 対応表を参照。
#
# 特殊トークン規約(第2章): PAD=0, BOS=1, EOS=2
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 0
BOS = 1
EOS = 2

NEG = -1e9  # mask の「見せない」印。softmax 後にほぼ 0 になる(-inf だと全滅行で NaN の危険)


def get_device():
    """cuda → mps → cpu の順で、使える計算装置を返す。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def positional_encoding(max_len, d_model):
    """論文 3.5 の sin/cos 固定テーブル (max_len, d_model)。第7巻7章の PyTorch 翻訳。"""
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)   # (max_len, 1)
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)        # (d_model/2,)
    angle = pos / torch.pow(10000.0, two_i / d_model)               # (max_len, d_model/2)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe


def pad_mask(ids):
    """PAD 位置を「key として見せない」加算 mask。ids (batch, len) → (batch, 1, 1, len)。"""
    return (ids == PAD).float()[:, None, None, :] * NEG


def causal_mask(length, device):
    """未来を見せない上三角 mask (1, 1, len, len)。第7巻5章の causal mask と同じもの。"""
    upper = torch.triu(torch.ones(length, length, device=device), diagonal=1)
    return upper[None, None, :, :] * NEG


class MultiHeadAttention(nn.Module):
    """論文 3.2.1〜3.2.2(第7巻3〜4章)。mask は scores への加算で効かせる。"""

    def __init__(self, d_model, h, p_drop):
        super().__init__()
        assert d_model % h == 0, "d_model は h で割り切れること(d_k = d_model / h)"
        self.h = h
        self.d_k = d_model // h
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(p_drop)

    def forward(self, q_in, k_in, v_in, mask=None):
        # q_in (batch, q_len, d_model), k_in/v_in (batch, k_len, d_model)
        batch, q_len, _ = q_in.shape
        k_len = k_in.size(1)

        def split_heads(x, length):
            # (batch, len, d_model) → (batch, h, len, d_k)
            return x.view(batch, length, self.h, self.d_k).transpose(1, 2)

        Q = split_heads(self.W_q(q_in), q_len)
        K = split_heads(self.W_k(k_in), k_len)
        V = split_heads(self.W_v(v_in), k_len)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (batch, h, q_len, k_len)
        if mask is not None:
            scores = scores + mask
        attn = self.drop(F.softmax(scores, dim=-1))
        out = attn @ V                                           # (batch, h, q_len, d_k)
        out = out.transpose(1, 2).contiguous().view(batch, q_len, self.h * self.d_k)
        return self.W_o(out)


class FeedForward(nn.Module):
    """論文 3.3 式(2): FFN(x) = max(0, x W1 + b1) W2 + b2(第7巻6.1)。"""

    def __init__(self, d_model, d_ff, p_drop):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        return self.lin2(self.drop(F.relu(self.lin1(x))))


class EncoderLayer(nn.Module):
    """self-attention + FFN。residual と layer norm は論文 5.4 どおり post-LN。"""

    def __init__(self, d_model, h, d_ff, p_drop):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, h, p_drop)
        self.ffn = FeedForward(d_model, d_ff, p_drop)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.drop(self.attn(x, x, x, src_mask)))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    """masked self-attention + cross-attention + FFN(第7巻5章の3種の attention)。"""

    def __init__(self, d_model, h, d_ff, p_drop):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, p_drop)
        self.cross_attn = MultiHeadAttention(d_model, h, p_drop)
        self.ffn = FeedForward(d_model, d_ff, p_drop)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p_drop)

    def forward(self, y, memory, tgt_mask, src_mask):
        y = self.norm1(y + self.drop(self.self_attn(y, y, y, tgt_mask)))
        y = self.norm2(y + self.drop(self.cross_attn(y, memory, memory, src_mask)))
        y = self.norm3(y + self.drop(self.ffn(y)))
        return y


class TinyTransformer(nn.Module):
    """縮小版 Transformer(encoder-decoder)。

    - 語彙は src/tgt 共有。埋め込み E 1枚を入力埋め込み(両側)と出力射影の
      3役で共有する(weight sharing — 第7巻6.3)。埋め込みは √d_model 倍(論文 3.4)
    - mask(src の pad mask + tgt の pad mask & causal mask)は forward 内部で生成する
    """

    def __init__(self, vocab_size, d_model=128, h=4, N=2, d_ff=256,
                 p_drop=0.1, max_len=64):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, d_model)
        # √d_model 倍の補正(論文 3.4)の前提に合わせ、E は小さめに初期化する
        nn.init.normal_(self.embed.weight, mean=0.0, std=d_model ** -0.5)
        self.register_buffer("pe", positional_encoding(max_len, d_model))
        self.drop = nn.Dropout(p_drop)
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, h, d_ff, p_drop) for _ in range(N)])
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, h, d_ff, p_drop) for _ in range(N)])
        # 出力射影の重みは self.embed.weight をそのまま使う(専用パラメータを持たない)

    def encode(self, src_ids, src_mask):
        seq = src_ids.size(1)
        x = self.drop(self.embed(src_ids) * math.sqrt(self.d_model) + self.pe[:seq])
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt_in_ids, memory, tgt_mask, src_mask):
        seq = tgt_in_ids.size(1)
        y = self.drop(self.embed(tgt_in_ids) * math.sqrt(self.d_model) + self.pe[:seq])
        for layer in self.dec_layers:
            y = layer(y, memory, tgt_mask, src_mask)
        return y

    def forward(self, src_ids, tgt_in_ids):
        # src_ids (batch, src_len), tgt_in_ids (batch, tgt_len) → logits (batch, tgt_len, vocab)
        assert src_ids.size(1) <= self.max_len and tgt_in_ids.size(1) <= self.max_len
        src_mask = pad_mask(src_ids)                                  # (batch, 1, 1, src_len)
        tgt_mask = pad_mask(tgt_in_ids) + causal_mask(tgt_in_ids.size(1),
                                                      tgt_in_ids.device)
        memory = self.encode(src_ids, src_mask)
        y = self.decode(tgt_in_ids, memory, tgt_mask, src_mask)
        logits = y @ self.embed.weight.T   # weight sharing: 出口も E(第7巻6.3)
        return logits


def label_smoothing_loss(logits, targets, eps=0.1, pad_id=0):
    """label smoothing 付き cross-entropy(論文 5.4, ε_ls = 0.1)。

    正解分布を one-hot から p' = (1-ε)·one-hot + ε·一様分布 になだらかにして、
    モデル分布 q との cross-entropy H(p', q) を取る(第4巻5.4: H(p') は定数なので
    KL(p' || q) の最小化と同じ問題)。targets が pad_id の位置は損失から除外する。
    logits (batch, tgt_len, vocab), targets (batch, tgt_len) → スカラー。
    """
    vocab = logits.size(-1)
    log_q = F.log_softmax(logits, dim=-1)                            # (batch, tgt_len, vocab)
    nll = -log_q.gather(-1, targets.unsqueeze(-1)).squeeze(-1)       # 正解項 -log q_t
    uniform = -log_q.mean(dim=-1)                                    # 一様項 (1/K)Σ(-log q_k)
    per_token = (1.0 - eps) * nll + eps * uniform                    # (batch, tgt_len)
    keep = (targets != pad_id).float()
    return (per_token * keep).sum() / keep.sum()


if __name__ == "__main__":
    # スモークテスト(詳細は test_model.py)
    torch.manual_seed(42)
    model = TinyTransformer(vocab_size=20, d_model=32, h=4, N=2, d_ff=64, max_len=16)
    src = torch.tensor([[5, 6, 7, PAD], [8, 9, PAD, PAD]])
    tgt_in = torch.tensor([[BOS, 5, 6], [BOS, 8, PAD]])
    logits = model(src, tgt_in)
    assert logits.shape == (2, 3, 20)
    tgt_out = torch.tensor([[5, 6, EOS], [8, EOS, PAD]])
    loss = label_smoothing_loss(logits, tgt_out, eps=0.1, pad_id=PAD)
    assert loss.item() > 0
    print(f"ok: logits {tuple(logits.shape)}, loss {loss.item():.4f}, device {get_device()}")
