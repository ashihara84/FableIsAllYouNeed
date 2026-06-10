# 第5巻 第1章 1.1: XOR的なデータは線形モデルでは原理的に解けない — その観測
# このファイルがやるのは「観測」だけ。救済策(線形と線形の間に非線形を挟む)は第2章で実装する。
# モデル・損失・勾配は第4巻第4章のロジスティック回帰を、訓練ループは第3巻第4章の4拍子を、そのまま流用している。
import numpy as np

rng = np.random.default_rng(42)
n = 50  # 1つの塊あたりの点数


def blob(cx, cy):
    """中心 (cx, cy) のまわりに点を n 個ばらまく。(n, 2) を返す"""
    return rng.normal(loc=[cx, cy], scale=0.7, size=(n, 2))


# === データA(対照用): 2つの塊。第3巻エピローグと同じ、直線で割れる配置 ===
X_A = np.vstack([blob(-2.0, -2.0), blob(2.0, 2.0)])      # (100, 2)
y_A = np.concatenate([np.zeros(n), np.ones(n)])          # (100,)

# === データB(本題): 4つの塊を市松模様に置く(XOR的な配置) ===
# 対角どうしが同じラベル。座標の符号が「同じ」なら0、「異なる」なら1 — 排他的論理和(XOR)の配置。
X_B = np.vstack([blob(-2.0, -2.0), blob(2.0, 2.0),       # ラベル0(左下・右上)
                 blob(-2.0, 2.0), blob(2.0, -2.0)])      # ラベル1(左上・右下)
y_B = np.concatenate([np.zeros(2 * n), np.ones(2 * n)])  # (200,)


# --- モデルと損失: 第4巻第4章のロジスティック回帰そのまま ---
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def predict(X, w, b):
    """p = sigmoid(Xw + b)。クラス1である「自信」を 0〜1 で返す。(n,)"""
    return sigmoid(X @ w + b)


def log_loss(X, y, w, b):
    """第4巻4.2で導出した負の対数尤度(log loss)"""
    eps = 1e-12
    p = np.clip(predict(X, w, b), eps, 1.0 - eps)
    return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def accuracy(X, y, w, b):
    """0.5 以上をクラス1と判定したときの正解率"""
    return np.mean((predict(X, w, b) >= 0.5) == (y == 1))


def train(X, y, w0, b0, lr=0.5, steps=5000):
    """第3巻第4章の4拍子。勾配は第4巻4.3で手導出した (p − y) の式(導出済みなので数値微分は不要)"""
    w = np.array(w0, dtype=float)
    b = float(b0)
    history = [log_loss(X, y, w, b)]
    for _ in range(steps):
        p = predict(X, w, b)                 # forward
        grad_w = X.T @ (p - y) / len(y)      # gradient(loss は式の中で済んでいる)
        grad_b = np.mean(p - y)
        w -= lr * grad_w                     # update
        b -= lr * grad_b
        history.append(log_loss(X, y, w, b))
    return w, b, history


# === 実験1(対照): 直線で割れるデータAなら、ロジスティック回帰は何の問題もなく解ける ===
w_A, b_A, hist_A = train(X_A, y_A, w0=[0.0, 0.0], b0=0.0)
print("実験1(対照: 2つの塊):")
print(f"  loss: {hist_A[0]:.4f} -> {hist_A[-1]:.4f}   正解率: {accuracy(X_A, y_A, w_A, b_A):.1%}")

assert hist_A[-1] < 0.05                       # 損失はほぼゼロまで下がり、
assert accuracy(X_A, y_A, w_A, b_A) >= 0.99    # ほぼ全問正解。道具は壊れていない

# === 実験2(本題): XOR的なデータBでは、同じ道具・同じ手順が50%台で止まる ===
w_B, b_B, hist_B = train(X_B, y_B, w0=[0.0, 0.0], b0=0.0, steps=20000)
acc_B = accuracy(X_B, y_B, w_B, b_B)
print("実験2(本題: 4つの塊の市松模様):")
print(f"  loss: {hist_B[0]:.4f} -> {hist_B[-1]:.4f}   正解率: {acc_B:.1%}")
print(f"  (参考)log 2 = {np.log(2):.4f} = 全員に「自信0.5」と答えるときの損失")

assert hist_B[0] - hist_B[-1] < 0.05    # 2万ステップ回しても、損失はほぼ動かず
assert hist_B[-1] > 0.66                # log 2 ≈ 0.693(完全な無知)の近くに張りついたまま
assert 0.40 <= acc_B <= 0.62            # 正解率はコイン投げなみ

# === 実験3: 初期値を10通り変えて再訓練 — どこから出発しても同じ場所に落ちる ===
# (第2巻2章の「初期値が悪くて局所的な谷にはまった」疑惑をつぶすための実験)
accs = []
for _ in range(10):
    w0 = rng.normal(0.0, 2.0, size=2)
    b0 = float(rng.normal(0.0, 2.0))
    w, b, _ = train(X_B, y_B, w0=w0, b0=b0, steps=20000)
    accs.append(accuracy(X_B, y_B, w, b))
print("実験3(初期値を10通り):")
print(f"  正解率: 最小 {min(accs):.1%} / 最大 {max(accs):.1%}")

assert max(accs) <= 0.62   # どの初期値から始めても、50%台を抜け出せない

# === 実験4: そもそも4つの中心を分ける直線は存在しない — 10万本の直線で総当たり ===
centers = np.array([[-2.0, -2.0], [2.0, 2.0],
                    [-2.0, 2.0], [2.0, -2.0]])        # (4, 2)
labels = np.array([False, False, True, True])         # (4,) 上と同じラベル
W_try = rng.normal(size=(100000, 2))                  # 直線の向き (w1, w2) を10万通り
b_try = rng.normal(scale=4.0, size=100000)            # 切片 b も散らす
# 各直線について、4つの中心のスコア w1*x1 + w2*x2 + b の符号を一斉に調べる
score = (W_try[:, 0:1] * centers[:, 0]               # (100000, 1) * (4,) → (100000, 4)
         + W_try[:, 1:2] * centers[:, 1]
         + b_try[:, None])
side = score > 0                                      # (100000, 4) 各中心がどちら側か
ok_fwd = (side == labels).all(axis=1)                 # 「正の側 = ラベル1」で4点とも正解
ok_rev = (side == ~labels).all(axis=1)                # ラベルの割り当てを逆にしても試す
n_separating = int((ok_fwd | ok_rev).sum())
print("実験4(総当たり):")
print(f"  4つの中心を完全に分けられた直線: {n_separating} / 100000 本")

assert n_separating == 0   # 1本もない(本文の2行の証明のとおり、原理的に存在しない)

print()
print("ok: すべての assert を通過しました")
print("  - 同じモデル・同じ訓練で、直線で割れるデータは解け、XOR的なデータは50%台で止まった")
print("  - 初期値のせいではない。直線が存在しないのだから、探し方の問題ではない")
print("  - 直線しか引けないことが限界 → 第2章で「曲がる」モデルを作る")
