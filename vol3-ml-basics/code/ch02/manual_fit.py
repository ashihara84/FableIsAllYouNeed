# 第3巻 第2章 2.2: 直線を「手で」当てはめる
# make_data() が作る20人のデータは、第3章(損失地形)・第4章(勾配降下)・
# 第5章(ミニバッチ)でも同じものを使い続ける。seed=42 を変えないこと。
import numpy as np


def linear(X, W, b):
    """全結合層: (バッチ, d_in) @ (d_in, d_out) + (d_out,) -> (バッチ, d_out)"""
    return X @ W + b


def make_data():
    """勉強時間 -> 点数 の合成データ(20人)。

    真の規則: 点数 = 7.0 * 勉強時間 + 20.0 + ばらつき(標準偏差 6.0)。
    「真の規則」は出題者だけが知っている。学習する側は X と y しか見ない。
    """
    rng = np.random.default_rng(42)
    n = 20
    X = rng.uniform(0, 9, size=(n, 1))   # 勉強時間(0〜9時間) (20, 1)
    noise = rng.normal(0, 6.0, size=(n, 1))
    y = 7.0 * X + 20.0 + noise           # 点数 (20, 1)
    return X, y


def try_line(w_val, b_val, X, y):
    """候補 (w, b) を1つ試し、予測とズレ(y - y_hat)を返す。"""
    w = np.array([[float(w_val)]])       # (1, 1)
    b = np.array([float(b_val)])         # (1,)
    y_hat = linear(X, w, b)              # (20, 1) @ (1, 1) + (1,) -> (20, 1)
    return y_hat, y - y_hat              # ズレ: 正なら点が直線の上、負なら下


X, y = make_data()

# --- データの確認: shape と再現性(後続章はこの3人の値で同一データと照合する) ---
assert X.shape == (20, 1) and y.shape == (20, 1)
assert np.allclose(X[:3].ravel(), [6.96560444, 3.94990596, 7.72738128])
assert np.allclose(y[:3].ravel(), [67.65005688, 43.56376444, 81.42691699])

# --- 2.1 の確認: 線形回帰のモデルは第1巻6章の linear そのもの ---
w = np.array([[7.0]])                    # (1, 1)
b = np.array([20.0])                     # (1,)
y_hat = linear(X, w, b)
assert y_hat.shape == (20, 1)
for i in range(20):                      # 1人ずつの y = w*x + b と一致する
    assert np.allclose(y_hat[i, 0], 7.0 * X[i, 0] + 20.0)

# --- 2.2: 候補 (w, b) を試してズレを目視する ---
for w_val, b_val in [(10.0, 0.0), (4.0, 40.0), (7.0, 20.0), (6.5, 25.0)]:
    _, diff = try_line(w_val, b_val, X, y)
    print("w={:4.1f}, b={:4.1f}  ズレ: {}".format(
        w_val, b_val, np.round(diff.ravel(), 1)))

# --- 本文で引用した数値の検証 ---
_, diff_A = try_line(10.0, 0.0, X, y)    # 候補A: 直線が全体に低い
assert (diff_A > 0).sum() == 14          # 20人中14人がプラスのズレ
assert np.allclose(diff_A.max(), 22.8, atol=0.05)

_, diff_B = try_line(4.0, 40.0, X, y)    # 候補B: 今度はマイナスが多い
assert (diff_B < 0).sum() == 13

_, diff_C = try_line(7.0, 20.0, X, y)    # 候補C: ほとんどの人が ±5 前後
_, diff_D = try_line(6.5, 25.0, X, y)    # 候補D: こちらも悪くない
assert (np.abs(diff_C) < 13).all()
assert (np.abs(diff_D) < 10).all()       # CとDの優劣は目視では決められない

# --- 演習・問4の数値の検証: 「ズレの合計」では悪さを測れない ---
_, diff_E = try_line(10.0, 5.0, X, y)    # 明らかに急すぎる直線なのに
assert abs(diff_E.sum()) < 4.0           # 合計はほぼ 0(プラスとマイナスが打ち消す)
assert (np.abs(diff_E) > 12).sum() >= 5  # それでも個々のズレは大きい

print("ok: すべての assert を通過しました")
