import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_DIR = os.path.join(BASE_DIR, "outputs", "predictions")
JSON_PATH = os.path.join(PREDICTIONS_DIR, "checkpoint_comparison.json")
OUTPUT_LEARNING_CURVE = os.path.join(PREDICTIONS_DIR, "learning_curve.png")
OUTPUT_BAR_CHART = os.path.join(PREDICTIONS_DIR, "performance_comparison.png")

# Load data
with open(JSON_PATH, "r") as f:
    data = json.load(f)

metrics = data["metrics"]
df = pd.DataFrame(metrics).T
df = df.reset_index().rename(columns={"index": "Model"})

# Set aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

# 1. Performance Comparison (Bar Chart)
plt.figure(figsize=(10, 6))
models_to_compare = ["Zero-shot", "Prompt-engineered", "ckpt-200"]
compare_df = df[df["Model"].isin(models_to_compare)].copy()
compare_df["Model"] = pd.Categorical(compare_df["Model"], categories=models_to_compare, ordered=True)
compare_df = compare_df.sort_values("Model")
compare_df_melted = compare_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

ax = sns.barplot(x="Model", y="Score", hue="Metric", data=compare_df_melted, palette="viridis")
plt.title("Model Performance Comparison: Baseline vs. Best Checkpoint (ROUGE)", fontsize=16, fontweight='bold', pad=15)
plt.ylabel("ROUGE Score", fontsize=14)
plt.xlabel("")
plt.ylim(0, 30)

# Add values on top of bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3)

plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(OUTPUT_BAR_CHART, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved {OUTPUT_BAR_CHART}")

# 2. Learning Curve (Line Graph)
plt.figure(figsize=(10, 6))
# Filter only checkpoints and sort them by steps
checkpoints = [m for m in df["Model"] if m.startswith("ckpt-")]
ckpt_df = df[df["Model"].isin(checkpoints)].copy()
ckpt_df["Steps"] = ckpt_df["Model"].str.extract(r'(\d+)').astype(int)
ckpt_df = ckpt_df.sort_values("Steps")

# Plot each metric
plt.plot(ckpt_df["Steps"], ckpt_df["ROUGE-1"], marker='o', linewidth=2.5, label='ROUGE-1', color='#440154')
plt.plot(ckpt_df["Steps"], ckpt_df["ROUGE-L"], marker='s', linewidth=2.5, label='ROUGE-L', color='#21918c')
plt.plot(ckpt_df["Steps"], ckpt_df["ROUGE-2"], marker='^', linewidth=2.5, label='ROUGE-2', color='#fde725')

# Highlight best checkpoint
best_step = 200
best_r1 = float(ckpt_df[ckpt_df["Steps"] == best_step]["ROUGE-1"].iloc[0])
plt.axvline(x=best_step, color='red', linestyle='--', alpha=0.7, label='Best Checkpoint (200)')
plt.scatter([best_step], [best_r1], color='red', s=100, zorder=5)
plt.annotate('Peak Performance', xy=(best_step, best_r1), xytext=(best_step-20, best_r1+2),
             arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8),
             fontsize=12, fontweight='bold', color='red')

plt.title("Learning Curve: ROUGE Metrics across Training Steps", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Training Steps", fontsize=14)
plt.ylabel("ROUGE Score", fontsize=14)
plt.xticks(ckpt_df["Steps"])
plt.ylim(0, 30)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right')

plt.savefig(OUTPUT_LEARNING_CURVE, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved {OUTPUT_LEARNING_CURVE}")
