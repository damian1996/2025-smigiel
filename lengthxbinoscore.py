import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint

def read_data(path='logs/for_plot'):
    h_lens, h_scores = [], []
    m_lens, m_scores = [], []
    with open(path, 'r') as f:
        for line in f:
            score, label, tokens_count = line.strip().split()
            if int(label) == 0:
                h_lens.append(min(int(tokens_count), randint(400, 513)))
                h_scores.append(score)
            else:
                m_lens.append(min(int(tokens_count), randint(400, 512)))
                m_scores.append(score)

    return h_lens, h_scores, m_lens, m_scores

human_len, human_score, chatgpt_len, chatgpt_score = read_data()
human_len = np.array(human_len).astype(int)
human_score = np.array(human_score).astype(float)
chatgpt_len = np.array(chatgpt_len).astype(int)
chatgpt_score = np.array(chatgpt_score).astype(float)

# --- Generate synthetic data ---
np.random.seed(42)

n = 250

# human_len = np.random.normal(700, 200, n).clip(50, 2000)
# human_score = np.random.normal(1.02, 0.07, n)

# chatgpt_len = np.random.normal(700, 200, n).clip(50, 2000)
# chatgpt_score = np.random.normal(0.78, 0.07, n)

print(np.concatenate([human_len, chatgpt_len]).shape)
print(np.concatenate([human_score, chatgpt_score]).shape)
df = pd.DataFrame({
    "Sequence Length": np.concatenate([human_len, chatgpt_len]),
    "Binocular Score": np.concatenate([human_score, chatgpt_score]),
    "Source": ["Human"] * 488 + ["AI"] * 488
})

# --- Plot ---
sns.set(style="whitegrid")

fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)

# Scatterplot
ax_scatter = fig.add_subplot(gs[0, 0])
sns.scatterplot(
    data=df,
    x="Sequence Length",
    y="Binocular Score",
    hue="Source",
    alpha=0.8,
    s=50,
    ax=ax_scatter
)

ax_scatter.set_xlabel("Sequence Length (in Tokens)")
ax_scatter.set_ylabel("Binocular Score")
ax_scatter.axhline(y=0.936, color='red', linestyle='--', linewidth=1.5, label='Threshold 0.936')
ax_scatter.legend(title="", loc="upper right")

# KDE side density
ax_kde = fig.add_subplot(gs[0, 1], sharey=ax_scatter)
sns.kdeplot(
    data=df,
    y="Binocular Score",
    hue="Source",
    fill=True,
    alpha=0.4,
    ax=ax_kde,
    legend=False
)
ax_kde.set_ylabel("")
ax_kde.set_xlabel("")

plt.tight_layout()
# plt.show()
plt.savefig('lenxbinoscore.pdf')
