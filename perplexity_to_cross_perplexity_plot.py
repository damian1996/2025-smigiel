import matplotlib.pyplot as plt
import numpy as np

path = 'logs/bielik3' # 'logs/bielik2base'
N = 1000

def load_data():
    with open(path, 'r') as f:
        ppls, xppls = [], []
        for idx, line in enumerate(f):
            if idx > N: continue

            values = line.strip().split()
            ppl, xppl = values[0], values[1]
            ppls.append(ppl)
            xppls.append(xppl)
        
        return ppls, xppls


perplexity, cross_perplexity = load_data()
perplexity = np.array(perplexity).astype(float)
cross_perplexity = np.array(cross_perplexity).astype(float)

np.random.seed(42)

cross_perplexity = np.clip(cross_perplexity, 0, 7)
perplexity = np.clip(perplexity, 0, 7)


plt.figure(figsize=(8, 6)) # Adjust figure size for better visualization

plt.scatter(
    cross_perplexity,  # x-axis data
    perplexity,        # y-axis data
    color='#4678A7',   # A blue color similar to the one in the image
    s=50,              # Marker size (adjust as needed)
    alpha=1.0          # Opacity
)

# Set labels and title
plt.title('Cross-Perplexity vs Perplexity', fontsize=14)
plt.xlabel('Cross-Perplexity', fontsize=12)
plt.ylabel('Perplexity', fontsize=12)

# Set axis limits to match the original image (0 to 12)
plt.xlim(0, 7)
plt.ylim(0, 7)

# Set ticks for the axes
plt.xticks(np.arange(0, 7, 1))
plt.yticks(np.arange(0, 7, 1))

# plt.show()
plt.savefig('bielik3.pdf')
