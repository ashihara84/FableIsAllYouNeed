# 第6巻 第6章 演習問1: 入力長と系列一致率の関係をプロットする
# 注意: matplotlib は掲載のみの扱い(STYLE.md)。実行には matplotlib の導入が必要。
import matplotlib.pyplot as plt

from seq2seq_rnn import MAX_LEN, MIN_LEN, sequence_accuracy, train

params = train(verbose=False)
lengths = list(range(MIN_LEN, MAX_LEN + 1))
accs = [sequence_accuracy(params, L, n_eval=200, seed=1000 + L) for L in lengths]

plt.figure(figsize=(6, 4))
plt.plot(lengths, accs, marker="o")
plt.xlabel("input length")
plt.ylabel("sequence accuracy")
plt.title("fixed-length bottleneck: longer input, lower accuracy")
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.savefig("length_vs_accuracy.png", dpi=150)
print("saved: length_vs_accuracy.png")
