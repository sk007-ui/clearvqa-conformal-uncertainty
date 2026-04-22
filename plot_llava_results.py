import json
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    try:
        with open('cp_llava_clearvqa_results.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: cp_llava_clearvqa_results.json not found.")
        return

    methods  = ['LAC', 'APS']
    plot_data = {"Method": [], "Question Type": [], "Average Set Size": []}

    for method in methods:
        plot_data["Method"].append(method)
        plot_data["Question Type"].append("Clear")
        plot_data["Average Set Size"].append(data["clear"][method]["avg_set_size"])

        plot_data["Method"].append(method)
        plot_data["Question Type"].append("Ambiguous")
        plot_data["Average Set Size"].append(data["ambig"][method]["avg_set_size"])

    # ── Identical styling to all previous plots ──────────
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.barplot(
        data=plot_data,
        x="Method",
        y="Average Set Size",
        hue="Question Type",
        palette=["#3498db", "#e74c3c"],
        ax=ax,
        edgecolor=".1",
        linewidth=1.2,
        capsize=0.05
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=5, fontsize=12, fontweight='bold', color='#333333')

    # Specific to ClearVQA dataset (rephrased questions vs blurred text)
    ax.set_title("Conformal Prediction Average Set Sizes\nClear vs. Ambiguous Questions (LLaVA-1.5-7B)", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Conformal Prediction Method", fontsize=14, fontweight='bold')
    ax.set_ylabel("Average Set Size", fontsize=14, fontweight='bold')

    max_val = max(plot_data["Average Set Size"])
    ax.set_ylim(0, max_val * 1.3)

    plt.legend(title="Question Type", title_fontsize='13', fontsize='12', loc='upper left')
    sns.despine(left=True, bottom=False)
    plt.tight_layout()

    output_filename = "llava_clearvqa_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Plot successfully saved to {output_filename}")

if __name__ == "__main__":
    main()
