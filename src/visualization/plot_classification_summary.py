import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

root = Path(__file__).resolve().parents[2]
summary_csv = root / "results/classification/classification_test_summary.csv"
save_dir = root / "results/classification/summary_plots"
save_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(summary_csv)

# 把 Horizon 转成字符串标签，方便显示
h_label_map = {1: "t+1", 6: "t+6", 12: "t+12", 24: "t+24"}
df["HorizonLabel"] = df["Horizon"].map(h_label_map)

# 1. 所有模型在各个 horizon 的准确率对比
plt.figure(figsize=(10, 5))
sns.barplot(
    data=df,
    x="HorizonLabel",
    y="Accuracy",
    hue="Model"
)
plt.ylabel("Accuracy")
plt.xlabel("Horizon")
plt.title("Accuracy Comparison Across Models & Horizons")
plt.tight_layout()
plt.savefig(save_dir / "all_models_accuracy.png", dpi=200)
plt.close()

# 2. Baseline vs 最佳模型（每个 horizon 只保留 Accuracy 最高的一个模型）
baseline_df = df[df["Model"] == "Baseline"]
non_baseline = df[df["Model"] != "Baseline"]
best_per_h = non_baseline.loc[non_baseline.groupby("Horizon")["Accuracy"].idxmax()]

plt.figure(figsize=(10, 5))
plt.plot(baseline_df["Horizon"], baseline_df["Accuracy"], marker="o", label="Baseline", linewidth=2)
plt.plot(best_per_h["Horizon"], best_per_h["Accuracy"], marker="o", label="Best Model", linewidth=2)
plt.xticks([1, 6, 12, 24], ["t+1", "t+6", "t+12", "t+24"])
plt.ylabel("Accuracy")
plt.xlabel("Horizon")
plt.title("Baseline vs Best Model Accuracy")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(save_dir / "baseline_vs_best.png", dpi=200)
plt.close()
