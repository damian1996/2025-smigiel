import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Global font configuration (BIGGER TEXT)
# ----------------------------
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
})

exp_name = 'logs/bielik2base' # logs/bielik3
path = f'logs/{exp_name}'
N = 1000

def load_data():
    with open(path, 'r') as f:
        ppls, xppls = [], []
        for idx, line in enumerate(f):
            if idx > N:
                continue
            values = line.strip().split()
            ppl, xppl = values[0], values[1]
            ppls.append(ppl)
            xppls.append(xppl)
        return ppls, xppls


perplexity, cross_perplexity = load_data()
perplexity = np.array(perplexity, dtype=float)
cross_perplexity = np.array(cross_perplexity, dtype=float)

np.random.seed(42)

perplexity = np.clip(perplexity, 0, 7)
cross_perplexity = np.clip(cross_perplexity, 0, 7)

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(9, 7))

plt.scatter(
    cross_perplexity,
    perplexity,
    color='#4678A7',
    s=70,        # slightly larger markers
    alpha=1.0
)

plt.title('Cross-Perplexity vs Perplexity', pad=12)
plt.xlabel('Cross-Perplexity', labelpad=10)
plt.ylabel('Perplexity', labelpad=10)

plt.xlim(0, 5)
plt.ylim(0, 5)

plt.xticks(np.arange(0, 6, 1))
plt.yticks(np.arange(0, 6, 1))

plt.tight_layout()
plt.savefig(f'{exp_name}_v2.pdf', bbox_inches='tight')
# plt.show()
