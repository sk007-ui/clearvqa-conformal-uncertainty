import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    # Load the JSON data
    try:
        with open('cp_results.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: cp_results.json not found.")
        return

    # Extract data for plotting
    methods = ['LAC', 'APS']
    q_types = ['Clear', 'Ambiguous']
    
    # We will format data to work well with seaborn
    # We want a layout like: Method | Question Type | Set Size
    plot_data = {"Method": [], "Question Type": [], "Average Set Size": []}

    for method in methods:
        # Clear data
        plot_data["Method"].append(method)
        plot_data["Question Type"].append("Clear")
        plot_data["Average Set Size"].append(data["clear"][method]["avg_set_size"])
        
        # Ambiguous data
        plot_data["Method"].append(method)
        plot_data["Question Type"].append("Ambiguous")
        plot_data["Average Set Size"].append(data["ambig"][method]["avg_set_size"])

    # Configure publication-quality seaborn aesthetics
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create the grouped bar chart
    g = sns.barplot(
        data=plot_data,
        x="Method",
        y="Average Set Size",
        hue="Question Type",
        palette=["#3498db", "#e74c3c"], # Nice professional colors
        ax=ax,
        edgecolor=".1",
        linewidth=1.2,
        capsize=0.05
    )

    # Add numeric labels on top of the bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=5, fontsize=12, fontweight='bold', color='#333333')

    # Customize axes and labels
    ax.set_title("Conformal Prediction Average Set Sizes\nClear vs. Ambiguous Questions", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Conformal Prediction Method", fontsize=14, fontweight='bold')
    ax.set_ylabel("Average Set Size", fontsize=14, fontweight='bold')
    
    # Setting an appropriate Y limit dynamically
    max_val = max(plot_data["Average Set Size"])
    ax.set_ylim(0, max_val * 1.3)  # Leave room for labels

    # Ensure legend is clear and styled well
    plt.legend(title="Question Type", title_fontsize='13', fontsize='12', loc='upper left')

    # Despine for a cleaner look
    sns.despine(left=True, bottom=False)

    # Tight layout ensures no clipping
    plt.tight_layout()

    # Save as high-resolution PNG
    output_filename = "cp_comparison_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Plot successfully saved to {output_filename}")

if __name__ == "__main__":
    main()
