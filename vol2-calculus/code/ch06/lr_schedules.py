# 第2巻 第6章 6.3: learning rate を途中で変える — 最初は大股、あとは小股
# 実験1: 減衰(decay) -- 同じ初期歩幅でも、途中で縮めれば発散せず速い
# 実験2: ウォームアップ(warmup) -- 出発点が急斜面なら、最初は小股で様子を見る
import numpy as np

# --- 6.2 と同じデータ・同じ部品 ---
X = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
y = np.array([-1.0, 1.0, 3.0, 5.0, 7.0])


def error_total(X, y, w, b):
    return np.sum((w * X + b - y) ** 2)


def gradients(X, y, w, b):
    diff = w * X + b - y
    return np.sum(2 * diff * X), np.sum(2 * diff)


def run(lr_of_step, T):
    """スケジュール lr_of_step(t) に従って T ステップ降下し、E の推移を返す"""
    w, b = 0.0, 0.0
    history = [error_total(X, y, w, b)]
    for t in range(T):
        grad_w, grad_b = gradients(X, y, w, b)
        lr = lr_of_step(t)
        w -= lr * grad_w
        b -= lr * grad_b
        E = error_total(X, y, w, b)
        history.append(E)
        if E > 1e9:                      # 発散したら打ち切り(オーバーフロー回避)
            break
    return w, b, history


T = 200

# --- 実験1: 定数の小股・定数の大股・「最初は大股、あとは小股」 ---
_, _, hist_small = run(lambda t: 0.002, T)               # 小股のまま
_, _, hist_large = run(lambda t: 0.04, T)                # 大股のまま
_, _, hist_decay = run(lambda t: 0.04 / np.sqrt(t + 1), T)   # 大股から 1/sqrt(step) で縮める

assert hist_large[-1] > 1e9          # 大股のまま: 発散
assert hist_small[-1] > 0.1          # 小股のまま: 200ステップではまだ遠い
assert hist_decay[-1] < 1e-2         # 同じ 0.04 から出発しても、縮めれば収束する
assert hist_decay[-1] < hist_small[-1]

# --- 実験2: 急斜面スタートでは warmup が効く(地形だけ取り出した実験: E(w) = w^4) ---
def run_quartic(lr_of_step, T, w0=2.0):
    w = w0
    for t in range(T):
        w -= lr_of_step(t) * (4 * w ** 3)    # E'(w) = 4 w^3
        if abs(w) > 1e6:                     # 発散したら打ち切り
            break
    return w

T2 = 300
w_large = run_quartic(lambda t: 0.2, T2)     # 最初から大股
w_small = run_quartic(lambda t: 0.01, T2)    # ずっと小股
w_warmup = run_quartic(lambda t: 0.2 * min(1.0, (t + 1) / 30), T2)  # 30ステップかけて大股へ

assert abs(w_large) > 1e6                    # いきなり大股: 初手の急斜面で発散
assert abs(w_small) > 0.15                   # ずっと小股: 底の平らな区間で足踏み
assert abs(w_warmup) < 0.05                  # 小股で降りてから大股: 生き残り、かつ速い
assert abs(w_warmup) < abs(w_small)

print("ok: すべての assert を通過しました")
print(f"  実験1 (200ステップ後の E): 小股 {hist_small[-1]:.3f} / "
      f"大股 発散 / 減衰 {hist_decay[-1]:.2e}")
print(f"  実験2 (300ステップ後の |w|): 大股 発散 / 小股 {abs(w_small):.3f} / "
      f"warmup {abs(w_warmup):.4f}")
