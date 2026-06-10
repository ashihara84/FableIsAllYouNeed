# 第4巻 第7章: 内積の分散と √d_k — 伏線回収の実験
# d_k を変えて (1) 内積のばらつき (2) softmax 後の分布の尖り (3) 勾配の大きさ を観測し、
# √d_k で割るとすべて治ることを assert で固定する(論文 脚注4 の数値的な再現)。
import numpy as np

rng = np.random.default_rng(42)


# --- 道具の再掲(第5・6章で作ったもの) ---
def softmax(z):
    """スコアの列を確率分布に変える(第6章)。最大値シフトで数値安定化"""
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def entropy(p):
    """分布の不確かさ(第5章)。単位はナット"""
    return -np.sum(p * np.log(p + 1e-12))


def softmax_jacobian(p):
    """softmax の勾配(ヤコビ行列): J[i, j] = p_i (δ_ij − p_j)。(n, n)"""
    return np.diag(p) - np.outer(p, p)


# --- ヤコビ行列の式の検算: 数値微分(第2巻第1章の中心差分)と一致するか ---
z_check = rng.normal(size=5)
J_analytic = softmax_jacobian(softmax(z_check))
h = 1e-6
J_numeric = np.zeros((5, 5))
for j in range(5):
    e_j = np.zeros(5)
    e_j[j] = h
    J_numeric[:, j] = (softmax(z_check + e_j) - softmax(z_check - e_j)) / (2 * h)
assert np.allclose(J_analytic, J_numeric, atol=1e-8)  # 式と実測が一致

# --- 本体: d_k を変えて、内積 → softmax → 勾配 を観測する ---
n_keys = 64      # クエリ1本あたりのキーの本数。softmax は 64 択の分布になる
n_trials = 1000  # 乱数実験の反復回数

print(f"設定: キー {n_keys} 本, 試行 {n_trials} 回, 成分は平均0・分散1の正規乱数")
print(f"エントロピーと勾配ノルムは試行の中央値(典型的な1回の試行の姿)で報告する")
print()
print("d_k   | 内積のstd   √d_k  | エントロピー 生/÷√d_k | 勾配ノルム 生/÷√d_k")
print("-" * 74)

stats = {}
for d_k in [4, 64, 512]:
    all_scores = []
    ent_raw, ent_scaled = [], []
    grad_raw, grad_scaled = [], []
    for _ in range(n_trials):
        q = rng.normal(size=d_k)             # クエリ   (d_k,)   成分は平均0・分散1
        K = rng.normal(size=(n_keys, d_k))   # キーの束 (n_keys, d_k)
        scores = (K * q).sum(axis=1)         # 内積64本 (n_keys,)。K @ q と同じ(第1巻第4章)
        all_scores.append(scores)

        p_raw = softmax(scores)                       # そのまま softmax(割らない)
        p_scaled = softmax(scores / np.sqrt(d_k))     # √d_k で割ってから softmax
        ent_raw.append(entropy(p_raw))
        ent_scaled.append(entropy(p_scaled))
        # 勾配の大きさ: softmax のヤコビ行列のフロベニウスノルム(全成分の二乗和の平方根)
        # この行列がゼロ行列に近い = スコアを動かしても出力が動かない = 勾配が死んでいる
        grad_raw.append(np.linalg.norm(softmax_jacobian(p_raw)))
        grad_scaled.append(np.linalg.norm(softmax_jacobian(p_scaled)))

    s = {
        "mean": np.mean(all_scores),
        "std": np.std(all_scores),
        "ent_raw": np.median(ent_raw),
        "ent_scaled": np.median(ent_scaled),
        "grad_raw": np.median(grad_raw),
        "grad_scaled": np.median(grad_scaled),
    }
    stats[d_k] = s
    print(f"{d_k:5d} | {s['std']:8.2f} {np.sqrt(d_k):6.2f} | "
          f"{s['ent_raw']:6.3f} / {s['ent_scaled']:6.3f}   | "
          f"{s['grad_raw']:.4f} / {s['grad_scaled']:.4f}")

# === 観測1: 内積の平均は0、標準偏差は √d_k(分散は d_k)— 脚注4の再現 ===
for d_k in [4, 64, 512]:
    assert abs(stats[d_k]["mean"]) < 0.1 * np.sqrt(d_k)          # 平均はほぼ 0
    assert abs(stats[d_k]["std"] / np.sqrt(d_k) - 1.0) < 0.05    # 標準偏差は √d_k に比例(比 ≈ 1)
    # ↑ 比 ≈ 1 は「√d_k で割れば標準偏差1に戻る」ことの確認でもある

# === 観測2: 割らないと、d_k が大きいほど softmax が尖る(エントロピー低下) ===
assert stats[4]["ent_raw"] > stats[64]["ent_raw"] > stats[512]["ent_raw"]
assert stats[512]["ent_raw"] < 0.05      # 64択なのに不確かさはほぼゼロ(尖りきっている)

# === 観測3: 割らないと、d_k が大きいほど勾配が死ぬ(第3巻エピローグと同じ病気) ===
assert stats[4]["grad_raw"] > stats[64]["grad_raw"] > stats[512]["grad_raw"]
assert stats[512]["grad_raw"] < 0.05 * stats[512]["grad_scaled"]  # 勾配の通り道が痩せ細る

# === 治療: √d_k で割れば、尖りも勾配も d_k によらずほぼ一定 ===
ents = [stats[d]["ent_scaled"] for d in [4, 64, 512]]
grads = [stats[d]["grad_scaled"] for d in [4, 64, 512]]
assert max(ents) - min(ents) < 0.15      # エントロピーは d_k に依存しない
assert max(grads) - min(grads) < 0.05    # 勾配の大きさも

print()
print("ok: すべての assert を通過しました")
print("  - 内積の標準偏差は √d_k に比例する(分散は d_k。脚注4の再現)")
print("  - 割らずに softmax へ入れると d_k とともに分布が尖り、勾配が死ぬ")
print("  - √d_k で割ると、d_k によらず分布の尖りも勾配も一定に保たれる(scaled の意味)")
