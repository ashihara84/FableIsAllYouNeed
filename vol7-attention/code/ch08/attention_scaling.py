# 第7巻 第8章: self-attention の実行時間スケーリング実測
#   Table 1 の self-attention 行「層あたり計算量 O(n²·d)」を、自分の第3章の
#   実装で確かめる。n を倍々に増やし、実行時間が n に対して「線形より明確に
#   速く」伸びること(n² 傾向)を観測する。
#
#   assert は意図的に緩い(単調増加 + 超線形)。実行時間は BLAS・CPU・他の
#   プロセスの影響でぶれるため、「4.00倍ぴったり」を固定すると環境差で落ちる。
#   検証したいのは係数ではなく「自乗で効く」という傾向そのものである。
import sys
import time
from pathlib import Path

import numpy as np

# 第3章で実装した自分の attention をそのまま使う(精読の検算は自分のコードで)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch03"))
from attention import attention

rng = np.random.default_rng(42)

D_K = 64                            # 論文 base model の1ヘッドあたりの次元
NS = [256, 512, 1024, 2048, 4096]   # 系列長 n を倍々に振る
REPEATS = 5                         # 各 n で5回測り最小値を採る(ぶれ対策)


def measure(n, d_k, repeats=REPEATS):
    """系列長 n の self-attention 1回の実行時間(秒)。repeats 回の最小値。

    self-attention なので Q, K, V はすべて同じ系列由来 → shape は全部 (n, d_k)。
    配列の生成は計測の外に置く(測りたいのは attention 本体だけ)。
    """
    Q = rng.standard_normal((n, d_k))
    K = rng.standard_normal((n, d_k))
    V = rng.standard_normal((n, d_k))
    # 一部の BLAS(macOS の Accelerate 等)は、大きな行列積で浮動小数点の
    # 状態フラグを誤って立て、結果は正常なのに警告だけを出すことがある。
    # 結果の有限性は下の assert で直接確かめた上で、偽の警告は黙らせる
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        output, _ = attention(Q, K, V)
        assert np.isfinite(output).all(), "出力に inf / nan — これは本物の異常"
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            attention(Q, K, V)
            best = min(best, time.perf_counter() - t0)
    return best


if __name__ == "__main__":
    measure(NS[0], D_K, repeats=1)  # ウォームアップ(初回の準備コストを捨てる)

    print(f"d_k = {D_K} 固定、系列長 n を倍々に(各 {REPEATS} 回計測の最小値)")
    print()
    print("     n | 時間 (ms) | 直前との比 (n²なら≈4, 線形なら≈2)")
    print("-" * 55)
    times = []
    for n in NS:
        t = measure(n, D_K)
        ratio = f"{t / times[-1]:.2f}" if times else "   -"
        times.append(t)
        print(f"{n:6d} | {t * 1000:9.2f} | {ratio}")

    # log-log の傾き: 時間 ∝ n^slope の slope を最小二乗で推定(n² なら ≈ 2)
    slope = np.polyfit(np.log(NS), np.log(times), 1)[0]
    print("-" * 55)
    print(f"log-log の傾き: {slope:.2f}(n² 傾向なら 2 前後)")

    # === assert(緩め): 係数ではなく「傾向」だけを固定する ===
    # 1. n を増やすと時間は単調に増える
    for t_prev, t_next in zip(times, times[1:]):
        assert t_next > t_prev, "n を倍にしたのに時間が増えていない"

    # 2. 線形より明確に速く伸びる: n が16倍(256→4096)のとき、線形なら時間も
    #    16倍止まり。n² なら理論上256倍。中間の16倍超なら「超線形」と言える
    assert times[-1] / times[0] > (NS[-1] / NS[0]), (
        "時間の伸びが線形以下 — n² 傾向が観測できていない"
    )

    print()
    print("ok: n を増やすと実行時間は単調に増え、線形を明確に超えて伸びる")
    print("    (Table 1 の self-attention 行 O(n²·d) と整合)")
