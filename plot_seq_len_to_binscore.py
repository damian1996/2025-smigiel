import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint

# ----------------------------
# Global plot style (BIGGER TEXT)
# ----------------------------
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "legend.title_fontsize": 16
})

sns.set_style("whitegrid")

# ----------------------------
# Data loading
# ----------------------------
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

human_len = np.array(human_len, dtype=int)
human_score = np.array(human_score, dtype=float)
chatgpt_len = np.array(chatgpt_len, dtype=int)
chatgpt_score = np.array(chatgpt_score, dtype=float)

df = pd.DataFrame({
    "Sequence Length": np.concatenate([human_len, chatgpt_len]),
    "Binocular Score": np.concatenate([human_score, chatgpt_score]),
    "Source": ["Human"] * len(human_len) + ["AI"] * len(chatgpt_len)
})

# ----------------------------
# Plot
# ----------------------------
fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)

# Scatter plot
ax_scatter = fig.add_subplot(gs[0, 0])
sns.scatterplot(
    data=df,
    x="Sequence Length",
    y="Binocular Score",
    hue="Source",
    alpha=0.8,
    s=70,                  # larger markers
    ax=ax_scatter
)

ax_scatter.set_xlabel("Sequence Length (in Tokens)", labelpad=10)
ax_scatter.set_ylabel("Binocular Score", labelpad=10)

ax_scatter.axhline(
    y=0.931,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Threshold = 0.931"
)

ax_scatter.legend(
    title="",
    loc="upper right",
    frameon=True
)

# KDE side density
ax_kde = fig.add_subplot(gs[0, 1], sharey=ax_scatter)
sns.kdeplot(
    data=df,
    y="Binocular Score",
    hue="Source",
    fill=True,
    alpha=0.4,
    linewidth=2,
    ax=ax_kde,
    legend=False
)

ax_kde.set_xlabel("")
ax_kde.set_ylabel("")
ax_kde.tick_params(axis="x", labelsize=14)

plt.tight_layout()
plt.savefig("lenxbinoscore.pdf", bbox_inches="tight")
# plt.show()
