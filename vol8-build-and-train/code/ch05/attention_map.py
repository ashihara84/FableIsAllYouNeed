# 第8巻 第5章 5.3: attention マップの観察 — 訓練済みモデルは「どこを見ているか」
#
# 第6巻7.3で attention 付き seq2seq の重み行列に「反転の手順」が浮かび上がるのを
# 見た。同じ図を、今度は自作の Transformer(の cross-attention)で描く。
# 第3章の model.py は attention 重みを保存しない設計なので、第3章のファイルには
# 手を入れず、PyTorch の forward hook で「覗き穴」を開けて重みを取り出す。
import math
import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
for _ch in ("ch02", "ch03"):
    _p = os.path.normpath(os.path.join(_HERE, "..", _ch))
    if _p not in sys.path:
        sys.path.append(_p)   # 末尾に足す(ch03 にも train.py があり、先頭だと衝突する)

from data import BOS, encode_pair, decode, itos                       # 第2章
from train import load_or_train                                       # 5.1
from generate import greedy_decode                                    # 5.2


def probe_attention(mha):
    """MultiHeadAttention に覗き穴を開ける。

    forward hook で入力 (q_in, k_in, v_in, mask) を横取りし、モジュール自身の
    W_q, W_k を使って attention 重みを計算し直す(第7巻3章の3拍子の前半2拍)。
    eval モードでは dropout が恒等写像なので、再計算した重みは forward 内で
    実際に使われた重みと一致する。返り値: (結果の入れ物, hook のハンドル)。
    """
    store = {}

    def hook(module, inputs, output):
        q_in, k_in = inputs[0], inputs[1]
        mask = inputs[3] if len(inputs) > 3 else None
        B, q_len, _ = q_in.shape
        k_len = k_in.shape[1]

        def split_heads(x, length):
            return x.view(B, length, module.h, module.d_k).transpose(1, 2)

        Q = split_heads(module.W_q(q_in), q_len)
        K = split_heads(module.W_k(k_in), k_len)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(module.d_k)   # 式(1)の中身
        if mask is not None:
            scores = scores + mask
        store["A"] = F.softmax(scores, dim=-1).detach().cpu()      # (B, h, q_len, k_len)

    return store, mha.register_forward_hook(hook)


@torch.no_grad()
def cross_attention_map(model, src_ids, device):
    """greedy で翻訳し、最終 decoder 層の cross-attention(ヘッド平均)を返す。

    返り値: (出力トークン列, 重み行列 A (出力長, 入力長))。A の各行は合計1。
    """
    out_ids = greedy_decode(model, src_ids, device)
    store, handle = probe_attention(model.dec_layers[-1].cross_attn)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    tgt_in = torch.tensor([[BOS] + out_ids[:-1]], dtype=torch.long, device=device)
    model(src, tgt_in)                       # 翻訳をなぞる1回の forward で重みを採取
    handle.remove()
    return out_ids, store["A"][0].mean(dim=0)  # 4ヘッドの平均 (q_len, k_len)


def print_map(src_ids, out_ids, A):
    """重み行列を数値表で表示する(行=出力トークン, 列=入力トークン)。"""
    src_toks = [itos[i] for i in src_ids]
    out_toks = [itos[i] for i in out_ids]
    print(" " * 12 + "".join("%10s" % w for w in src_toks))
    for r, tok in enumerate(out_toks):
        print("%12s" % tok + "".join("%10.2f" % v for v in A[r]))


# 図5.1 の描画コード(掲載のみ。第6巻7.3の図7.1と同じ形式): A は上の重み行列
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.imshow(A, cmap="Greys", vmin=0.0, vmax=1.0)
# ax.set_xticks(range(len(src_toks)), src_toks)
# ax.set_yticks(range(len(out_toks)), out_toks)
# ax.set_xlabel("input position (key)")
# ax.set_ylabel("output step (query)")
# plt.show()


if __name__ == "__main__":
    model, device = load_or_train()

    # --- 観察1: 語順の入れ替え(英 S-V-O → 日 S-O-V)が模様に出る -------------
    s, t = "he eats fish", "かれ は さかな を たべます"
    src_ids, _ = encode_pair(s, t)
    out_ids, A = cross_attention_map(model, src_ids, device)
    print("src: %r -> 出力: %r" % (s, decode(out_ids)))
    print("最終 decoder 層の cross-attention(4ヘッドの平均):")
    print_map(src_ids, out_ids, A)

    src_toks = [itos[i] for i in src_ids]
    out_toks = [itos[i] for i in out_ids]
    col = {w: i for i, w in enumerate(src_toks)}
    row = {w: i for i, w in enumerate(out_toks)}

    assert torch.allclose(A.sum(dim=1), torch.ones(len(out_ids)), atol=1e-4)
    # 内容語の対応: かれ→he, さかな→fish, たべます→eats(行の argmax で確認)
    assert int(A[row["かれ</w>"]].argmax()) == col["he</w>"]
    assert int(A[row["さかな</w>"]].argmax()) == col["fish</w>"]
    assert int(A[row["たべます</w>"]].argmax()) == col["eats</w>"]

    # --- 観察2: 助詞「を」「が」は動詞を見て書かれる ---------------------------
    print("\n助詞の行だけ取り出す(を/が を書く瞬間、モデルはどこを見るか):")
    for s, t in [("he eats fish", "かれ は さかな を たべます"),
                 ("he wants fish", "かれ は さかな が ほしい です")]:
        src_ids, _ = encode_pair(s, t)
        out_ids, A = cross_attention_map(model, src_ids, device)
        src_toks = [itos[i] for i in src_ids]
        out_toks = [itos[i] for i in out_ids]
        particle = "を</w>" if "を</w>" in out_toks else "が</w>"
        r = out_toks.index(particle)
        peak = src_toks[int(A[r].argmax())]
        print("  %r: %s の行の最大重み %.2f は %s の列" %
              (s, particle, float(A[r].max()), peak))

    print("\nok: cross-attention に翻訳の対応関係が浮かび上がった")
