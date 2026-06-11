# 第8巻 第4章 4.5: Adam をフルスクラッチ実装し、torch.optim.Adam と軌跡を照合する
# 自作との並走はシリーズ全体でここが唯一の例外 — optimizer だけはブラックボックスにしない
import torch

torch.manual_seed(42)


# ---------------------------------------------------------------
# 自作 Adam(Kingma & Ba 2015, Algorithm 1 をそのまま写す)
# 論文 "Attention Is All You Need" 5.3 の設定: β1=0.9, β2=0.98, ε=1e-9
# ---------------------------------------------------------------
class AdamScratch:
    def __init__(self, params, lr, beta1=0.9, beta2=0.98, eps=1e-9):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.t = 0  # ステップ数(バイアス補正に使う)
        self.m = [torch.zeros_like(p) for p in self.params]  # 1次モーメント(勾配の移動平均)
        self.v = [torch.zeros_like(p) for p in self.params]  # 2次モーメント(勾配の2乗の移動平均)

    def step(self):
        self.t += 1
        with torch.no_grad():
            for p, m, v in zip(self.params, self.m, self.v):
                g = p.grad
                m.mul_(self.beta1).add_((1 - self.beta1) * g)        # m ← β1 m + (1−β1) g
                v.mul_(self.beta2).add_((1 - self.beta2) * g * g)    # v ← β2 v + (1−β2) g²
                m_hat = m / (1 - self.beta1 ** self.t)               # バイアス補正(4.4節)
                v_hat = v / (1 - self.beta2 ** self.t)
                p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = None


# ---------------------------------------------------------------
# 照合実験: 同じ初期値・同じデータ・同じ lr で、
# 自作 Adam と torch.optim.Adam が同じ軌跡を描くか
# モデルは小さな2層 MLP(float64 — 丸め誤差を最小にして厳しく比較する)
# ---------------------------------------------------------------
def make_params(seed):
    g = torch.Generator().manual_seed(seed)
    W1 = (torch.randn(4, 8, generator=g, dtype=torch.float64) * 0.5).requires_grad_()
    b1 = torch.zeros(8, dtype=torch.float64, requires_grad=True)
    W2 = (torch.randn(8, 1, generator=g, dtype=torch.float64) * 0.5).requires_grad_()
    b2 = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    return [W1, b1, W2, b2]


def forward(params, X):
    W1, b1, W2, b2 = params
    return torch.tanh(X @ W1 + b1) @ W2 + b2


# 同一の初期値を持つ2組のパラメータ(seed を揃えてつくる)
params_a = make_params(0)  # 自作 Adam が動かす
params_b = make_params(0)  # torch.optim.Adam が動かす
for pa, pb in zip(params_a, params_b):
    assert torch.equal(pa, pb)  # 出発点は完全に同一

lr = 1e-3
opt_a = AdamScratch(params_a, lr=lr, beta1=0.9, beta2=0.98, eps=1e-9)
opt_b = torch.optim.Adam(params_b, lr=lr, betas=(0.9, 0.98), eps=1e-9)

# 訓練データ(回帰の小問題。中身は何でもよい — 比べたいのは optimizer)
g = torch.Generator().manual_seed(1)
X = torch.randn(64, 4, generator=g, dtype=torch.float64)
y = torch.sin(X.sum(dim=1, keepdim=True))

losses = []
for step in range(50):
    # --- 自作 Adam 側 ---
    opt_a.zero_grad()
    loss_a = ((forward(params_a, X) - y) ** 2).mean()
    loss_a.backward()
    opt_a.step()

    # --- torch.optim.Adam 側 ---
    opt_b.zero_grad()
    loss_b = ((forward(params_b, X) - y) ** 2).mean()
    loss_b.backward()
    opt_b.step()

    # 毎ステップ、全パラメータが一致していることを確認
    for pa, pb in zip(params_a, params_b):
        assert torch.allclose(pa, pb, atol=1e-12), f"step {step+1} で軌跡がずれました"
    losses.append(loss_a.item())

print(f"50ステップ後の loss: 自作 {loss_a.item():.6f} / torch {loss_b.item():.6f}")
print(f"loss の推移(自作): {losses[0]:.4f} -> {losses[9]:.4f} -> {losses[-1]:.4f}")
assert losses[-1] < losses[0]  # ちゃんと学習も進んでいる
print("ok: 自作 Adam と torch.optim.Adam の軌跡が 50 ステップ全てで一致しました(atol=1e-12)")
