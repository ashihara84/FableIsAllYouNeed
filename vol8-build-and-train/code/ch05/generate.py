# 第8巻 第5章 5.2: 生成 — greedy / 温度サンプリング / beam search
#
# 訓練済みの縮小版 Transformer(train.py のチェックポイント)で翻訳を生成する。
# - greedy: 各ステップで argmax を取る
# - 温度サンプリング: softmax(z/τ) からサンプルする(第4巻6.5の温度の回収)
# - beam search: 論文 6.1 "beam search with a beam size of 4 and length penalty
#   α = 0.6" の実装
# 実行すると3方式の出力と assert による動作確認が走る。
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
for _ch in ("ch02", "ch03"):
    _p = os.path.normpath(os.path.join(_HERE, "..", _ch))
    if _p not in sys.path:
        sys.path.append(_p)   # 末尾に足す(ch03 にも train.py があり、先頭だと衝突する)

from data import PAD, BOS, EOS, encode_pair, decode                  # 第2章
from train import load_data, load_or_train, train_model              # 5.1

MAX_LEN = 20      # 生成の打ち切り長(コーパス最長の倍以上あれば十分)


@torch.no_grad()
def greedy_decode(model, src_ids, device, max_len=MAX_LEN):
    """各ステップで最大確率のトークンを選ぶ。返り値は BOS を除いた ID 列。"""
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    ys = [BOS]
    for _ in range(max_len):
        tgt_in = torch.tensor([ys], dtype=torch.long, device=device)
        logits = model(src, tgt_in)                  # (1, len(ys), vocab)
        next_id = int(logits[0, -1].argmax())        # 最後の位置の予測だけ使う
        ys.append(next_id)
        if next_id == EOS:
            break
    return ys[1:]


@torch.no_grad()
def sample_decode(model, src_ids, tau, device, g, max_len=MAX_LEN):
    """温度 τ 付きサンプリング。softmax(z/τ) から1トークンずつ引く(第4巻6.5)。"""
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    ys = [BOS]
    for _ in range(max_len):
        tgt_in = torch.tensor([ys], dtype=torch.long, device=device)
        z = model(src, tgt_in)[0, -1]                # 最後の位置のスコア (vocab,)
        probs = F.softmax(z / tau, dim=-1).cpu()     # τ で割ってから softmax
        next_id = int(torch.multinomial(probs, 1, generator=g))
        ys.append(next_id)
        if next_id == EOS:
            break
    return ys[1:]


def length_penalty(n_tokens, alpha=0.6):
    """GNMT 流の長さ補正 lp(Y) = ((5 + |Y|) / 6)^α。論文 6.1 が参照する形。"""
    return ((5 + n_tokens) / 6.0) ** alpha


@torch.no_grad()
def beam_search(model, src_ids, device, beam_size=4, alpha=0.6, max_len=MAX_LEN):
    """論文 6.1: beam size 4・length penalty α=0.6 の beam search。

    各仮説は (トークン列, 累積 log 確率)。毎ステップ、生きている仮説を
    beam_size 通りずつ延長し、全候補から上位 beam_size 本だけ残す。
    EOS に到達した仮説は完了プールへ移し、最後に長さ補正付きスコア
    logP / lp(Y) で勝者を決める。返り値は (BOS を除いた ID 列, スコア)。
    """
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    alive = [([BOS], 0.0)]
    finished = []
    for _ in range(max_len):
        candidates = []
        for ys, logp in alive:
            tgt_in = torch.tensor([ys], dtype=torch.long, device=device)
            log_probs = F.log_softmax(model(src, tgt_in)[0, -1], dim=-1).cpu()
            top = torch.topk(log_probs, beam_size)
            for lp_tok, tok in zip(top.values.tolist(), top.indices.tolist()):
                candidates.append((ys + [tok], logp + lp_tok))
        candidates.sort(key=lambda c: c[1], reverse=True)
        alive = []
        for ys, logp in candidates:
            if ys[-1] == EOS:
                finished.append((ys, logp))
            else:
                alive.append((ys, logp))
            if len(alive) == beam_size:
                break
        if not alive:                                 # 全仮説が EOS に到達
            break
    finished.extend(alive)                            # 打ち切られた仮説も候補に残す
    best, best_logp = max(
        finished, key=lambda c: c[1] / length_penalty(len(c[0]) - 1, alpha))
    return best[1:], best_logp / length_penalty(len(best) - 1, alpha)


def seq_accuracy(model, raw_pairs, device, decode_fn):
    """生成文と正解文の完全一致率(文字列で比較)。"""
    hit = 0
    for s, t in raw_pairs:
        src_ids, _ = encode_pair(s, t)
        if decode(decode_fn(model, src_ids, device)) == t:
            hit += 1
    return hit / len(raw_pairs)


if __name__ == "__main__":
    model, device = load_or_train()
    _, _, train_raw, test_raw = load_data()

    # --- (1) greedy: 暗記の確認と、未見ペアの観察 ----------------------------
    acc_train = seq_accuracy(model, train_raw, device, greedy_decode)
    acc_test = seq_accuracy(model, test_raw, device, greedy_decode)
    print("greedy 完全一致率: 訓練 %.3f / 未見 %.3f" % (acc_train, acc_test))
    assert acc_train > 0.85, "訓練ペアの大半を翻訳できていない: %.3f" % acc_train

    print("\n未見ペアの greedy 翻訳(左: モデル出力, 右: 正解):")
    for s, t in test_raw[:6]:
        src_ids, _ = encode_pair(s, t)
        out = decode(greedy_decode(model, src_ids, device))
        mark = "o" if out == t else "x"
        print("  %s %-28s -> %s | %s" % (mark, s, out, t))

    print("\ngreedy が外した訓練ペア(暗記し損ねた場所):")
    for s, t in train_raw:
        src_ids, _ = encode_pair(s, t)
        out = decode(greedy_decode(model, src_ids, device))
        if out != t:
            print("  %r -> %r(正解: %r)" % (s, out, t))

    # --- (2) 温度サンプリング(第4巻6.5の回収) ------------------------------
    demo = ("i like apples", "わたし は りんご が すき です")
    s, t = demo if demo in train_raw else train_raw[0]
    src_ids, _ = encode_pair(s, t)
    print("\n温度サンプリング: src = %r(正解: %r)、各温度で30回" % (s, t))
    g = torch.Generator().manual_seed(42)
    results = {}
    for tau in [0.5, 1.0, 2.0]:
        outs = [decode(sample_decode(model, src_ids, tau, device, g))
                for _ in range(30)]
        n_ok = sum(o == t for o in outs)
        n_distinct = len(set(outs))
        results[tau] = (n_ok, n_distinct)
        print("  tau=%.1f: 正解 %2d/30, 異なる出力 %2d 種  外した例: %r" %
              (tau, n_ok, n_distinct,
               next((o for o in outs if o != t), "(なし)")))
    assert results[0.5][0] >= results[2.0][0], "低温の方が堅実なはず"
    assert results[2.0][1] >= results[0.5][1], "高温の方が多彩なはず"

    # --- (3) beam search(論文 6.1: beam size 4, α=0.6) ----------------------
    # 暗記が完了したモデルでは greedy はほとんど詰まらない。greedy の構造的な
    # 弱点は、確信が固まる前のモデルでよく見える——25エポックで打ち切った
    # 「訓練途中」のモデルを作り、greedy が外したペアを beam search と比べる
    print("\n訓練途中(25エポック)のモデルで、greedy が外した訓練ペア(最大3件):")
    half, _ = train_model(epochs=25, device=device, log=False)
    half.eval()
    shown = 0
    for s, t in train_raw:
        src_ids, _ = encode_pair(s, t)
        out_g = decode(greedy_decode(half, src_ids, device))
        if out_g == t:
            continue
        out_b, score = beam_search(half, src_ids, device)
        print("  src: %r(正解: %r)" % (s, t))
        print("    greedy: %r" % out_g)
        print("    beam  : %r(score %.3f)" % (decode(out_b), score))
        shown += 1
        if shown == 3:
            break
    if shown == 0:
        print("  (この環境では25エポックで全ペア正解 — epochs を減らして再観察を)")

    # 仕上げ: 暗記完了モデルでも beam は greedy を下回らない(この規模では同点が多い)
    acc_beam = seq_accuracy(
        model, test_raw, device,
        lambda m, ids, dev: beam_search(m, ids, dev)[0])
    print("\nbeam(未見)完全一致率: %.3f(greedy は %.3f)" % (acc_beam, acc_test))
    assert acc_beam >= acc_test - 1e-9, "この規模では beam が greedy を下回らないはず"

    print("\nok: greedy / 温度サンプリング / beam search の3方式が動作")
