import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    # Load the JSON data
    try:
        with open('cp_vcr_results.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: cp_vcr_results.json not found.")
        return

    # Extract data for plotting
    methods = ['LAC', 'APS']

    # Format data to work well with seaborn
    # Layout: Method | Image Type | Set Size
    plot_data = {"Method": [], "Image Type": [], "Average Set Size": []}

    for method in methods:
        # Clear data
        plot_data["Method"].append(method)
        plot_data["Image Type"].append("Clear")
        plot_data["Average Set Size"].append(data["clear"][method]["avg_set_size"])

        # Ambiguous (blurred) data
        plot_data["Method"].append(method)
        plot_data["Image Type"].append("Ambiguous")
        plot_data["Average Set Size"].append(data["ambig"][method]["avg_set_size"])

    # Configure publication-quality seaborn aesthetics — identical to cp_comparison_plot
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create the grouped bar chart
    sns.barplot(
        data=plot_data,
        x="Method",
        y="Average Set Size",
        hue="Image Type",
        palette=["#3498db", "#e74c3c"],  # Identical palette to ClearVQA plot
        ax=ax,
        edgecolor=".1",
        linewidth=1.2,
        capsize=0.05
    )

    # Add numeric labels on top of the bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=5, fontsize=12, fontweight='bold', color='#333333')

    # Customize axes and labels
    ax.set_title("Conformal Prediction Average Set Sizes\nClear vs. Ambiguous Images (VCR)", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Conformal Prediction Method", fontsize=14, fontweight='bold')
    ax.set_ylabel("Average Set Size", fontsize=14, fontweight='bold')

    # Dynamic Y limit with headroom for bar labels
    max_val = max(plot_data["Average Set Size"])
    ax.set_ylim(0, max_val * 1.3)

    # Legend — identical position and styling
    plt.legend(title="Image Type", title_fontsize='13', fontsize='12', loc='upper left')

    # Despine for a cleaner look
    sns.despine(left=True, bottom=False)

    # Tight layout ensures no clipping
    plt.tight_layout()

    # Save as high-resolution PNG
    output_filename = "vcr_comparison_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Plot successfully saved to {output_filename}")

if __name__ == "__main__":
    main()
