import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
I did not do anything in the chart scripts, I just asked AI to do everything here,

Ofcourse I still decided how they should look and implemented feedback from my supervisors 
"""


# ------------------------------------------------------------
# Input / output settings
# ------------------------------------------------------------

SUMMARY_CSV = "evaluation_metrics_runtime_mean_std_summary.csv"
OUTPUT_DIR = "summarycharts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

topics = [10, 20, 30, 40, 50, 60]


# ------------------------------------------------------------
# Global plot styling
# ------------------------------------------------------------

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
})


# ------------------------------------------------------------
# Load aggregated mean/std results
# ------------------------------------------------------------

def require_columns(df, required_columns, csv_path):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {missing}\n"
            f"Available columns are: {list(df.columns)}"
        )


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_summary(csv_path=SUMMARY_CSV):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find {csv_path}. Run the evaluation summary script first."
        )

    df = pd.read_csv(csv_path)

    required = [
        "Model", "Variant", "Embedding", "Vocab", "K",
        "NPMI_mean", "Diversity_mean", "Runtime_Seconds_mean",
    ]
    require_columns(df, required, csv_path)

    for col in ["Model", "Variant", "Embedding", "Vocab"]:
        df[col] = df[col].apply(normalize_text)

    df["K"] = pd.to_numeric(df["K"], errors="coerce").astype("Int64")
    df["NPMI_mean"] = pd.to_numeric(df["NPMI_mean"], errors="coerce")
    df["Diversity_mean"] = pd.to_numeric(df["Diversity_mean"], errors="coerce")
    df["Runtime_Seconds_mean"] = pd.to_numeric(df["Runtime_Seconds_mean"], errors="coerce")

    return df


def subset_by_keys(df, model, variant=None, embedding=None, vocab=None):
    result = df[df["Model"] == model].copy()

    if variant is not None:
        result = result[result["Variant"] == variant]

    if embedding is not None:
        result = result[result["Embedding"] == embedding]

    if vocab is not None:
        result = result[result["Vocab"] == vocab]

    return result.sort_values("K")


def values_for(df, model, variant=None, embedding=None, vocab=None, value_col="NPMI_mean"):
    sub = subset_by_keys(df, model=model, variant=variant, embedding=embedding, vocab=vocab)

    values = []
    missing_k = []

    for k in topics:
        row = sub[sub["K"] == k]
        if row.empty:
            values.append(np.nan)
            missing_k.append(k)
        else:
            values.append(float(row.iloc[0][value_col]))

    if missing_k:
        label_parts = [model]
        if variant:
            label_parts.append(variant)
        if embedding:
            label_parts.append(embedding)
        if vocab:
            label_parts.append(vocab)
        print(f"Warning: missing K values {missing_k} for {' / '.join(label_parts)}")

    return values


def build_data_from_summary(summary_df):
    """
    Reconstructs the original chart data dictionary from the new
    mean/std summary CSV.

    The charts use the mean values:
      - NPMI_mean as TC
      - Diversity_mean as TD
      - Runtime_Seconds_mean as time

    The generated charts keep the same structure and filenames as the old
    hard-coded chart script.
    """

    data = {
        "BERTopic Base": pd.DataFrame({
            "Topics": topics,
            "Light_TC": values_for(summary_df, "BERTopic", "Base fullK", "Light", value_col="NPMI_mean"),
            "Light_TD": values_for(summary_df, "BERTopic", "Base fullK", "Light", value_col="Diversity_mean"),
            "Light_time": values_for(summary_df, "BERTopic", "Base fullK", "Light", value_col="Runtime_Seconds_mean"),
            "BoW_TC": values_for(summary_df, "BERTopic", "Base fullK", "Heavy", value_col="NPMI_mean"),
            "BoW_TD": values_for(summary_df, "BERTopic", "Base fullK", "Heavy", value_col="Diversity_mean"),
            "BoW_time": values_for(summary_df, "BERTopic", "Base fullK", "Heavy", value_col="Runtime_Seconds_mean"),
        }),

        "BERTopic-HDBSCAN": pd.DataFrame({
            "Topics": topics,
            "Light_TC": values_for(summary_df, "BERTopic", "NoiseParams fullK", "Light", value_col="NPMI_mean"),
            "Light_TD": values_for(summary_df, "BERTopic", "NoiseParams fullK", "Light", value_col="Diversity_mean"),
            "Light_time": values_for(summary_df, "BERTopic", "NoiseParams fullK", "Light", value_col="Runtime_Seconds_mean"),
            "BoW_TC": values_for(summary_df, "BERTopic", "NoiseParams fullK", "Heavy", value_col="NPMI_mean"),
            "BoW_TD": values_for(summary_df, "BERTopic", "NoiseParams fullK", "Heavy", value_col="Diversity_mean"),
            "BoW_time": values_for(summary_df, "BERTopic", "NoiseParams fullK", "Heavy", value_col="Runtime_Seconds_mean"),
        }),

        "BERTopic-Reassign": pd.DataFrame({
            "Topics": topics,
            "Light_TC": values_for(summary_df, "BERTopic", "Reassignment fullK", "Light", value_col="NPMI_mean"),
            "Light_TD": values_for(summary_df, "BERTopic", "Reassignment fullK", "Light", value_col="Diversity_mean"),
            "Light_time": values_for(summary_df, "BERTopic", "Reassignment fullK", "Light", value_col="Runtime_Seconds_mean"),
            "BoW_TC": values_for(summary_df, "BERTopic", "Reassignment fullK", "Heavy", value_col="NPMI_mean"),
            "BoW_TD": values_for(summary_df, "BERTopic", "Reassignment fullK", "Heavy", value_col="Diversity_mean"),
            "BoW_time": values_for(summary_df, "BERTopic", "Reassignment fullK", "Heavy", value_col="Runtime_Seconds_mean"),
        }),

        "CTM": pd.DataFrame({
            "Topics": topics,
            "Light_TC": values_for(summary_df, "CTM", "Base", "Light", value_col="NPMI_mean"),
            "Light_TD": values_for(summary_df, "CTM", "Base", "Light", value_col="Diversity_mean"),
            "Light_time": values_for(summary_df, "CTM", "Base", "Light", value_col="Runtime_Seconds_mean"),
            "BoW_TC": values_for(summary_df, "CTM", "Base", "Heavy", value_col="NPMI_mean"),
            "BoW_TD": values_for(summary_df, "CTM", "Base", "Heavy", value_col="Diversity_mean"),
            "BoW_time": values_for(summary_df, "CTM", "Base", "Heavy", value_col="Runtime_Seconds_mean"),
        }),

        "ETM": pd.DataFrame({
            "Topics": topics,
            "V2000_TC": values_for(summary_df, "ETM", "Base", "Heavy", "Vocab2000", value_col="NPMI_mean"),
            "V2000_TD": values_for(summary_df, "ETM", "Base", "Heavy", "Vocab2000", value_col="Diversity_mean"),
            "V2000_time": values_for(summary_df, "ETM", "Base", "Heavy", "Vocab2000", value_col="Runtime_Seconds_mean"),
            "V6000_TC": values_for(summary_df, "ETM", "Base", "Heavy", "Vocab6000", value_col="NPMI_mean"),
            "V6000_TD": values_for(summary_df, "ETM", "Base", "Heavy", "Vocab6000", value_col="Diversity_mean"),
            "V6000_time": values_for(summary_df, "ETM", "Base", "Heavy", "Vocab6000", value_col="Runtime_Seconds_mean"),
        }),

        "LDA": pd.DataFrame({
            "Topics": topics,
            "BoW_TC": values_for(summary_df, "LDA", "Base", "Heavy", "Vocab2000", value_col="NPMI_mean"),
            "BoW_TD": values_for(summary_df, "LDA", "Base", "Heavy", "Vocab2000", value_col="Diversity_mean"),
            "BoW_time": values_for(summary_df, "LDA", "Base", "Heavy", "Vocab2000", value_col="Runtime_Seconds_mean"),
        }),
    }

    return data


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def safe_name(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def display_version(model_name, version):
    """
    Controls how versions are displayed in legends and labels.

    Light -> D1
    BoW   -> D2
    Heavy -> D2
    """

    if version == "Light":
        return "D1"

    if version == "BoW":
        return "D2"

    if version == "Heavy":
        return "D2"

    return version


def get_bar_gradient_colors(n):
    """
    Creates a left-to-right color gradient for charts with a single bar series.
    Used for LDA charts and summary bar charts.
    """

    return plt.cm.viridis(np.linspace(0.25, 0.85, n))


def save_figure(filename):
    """
    Saves PNG version with tight bounding box.
    """

    png_path = os.path.join(OUTPUT_DIR, f"{filename}.png")

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.15
    )

    print(f"Saved: {png_path}")


# ------------------------------------------------------------
# Grouped bar chart function
# ------------------------------------------------------------

def plot_grouped_bars(
    df,
    model_name,
    metric_suffix,
    y_label,
    title,
    y_min=None,
    y_max=None
):
    """
    Creates a grouped bar chart.

    X-axis: number of topics
    Bars: model versions
    Y-axis: selected metric

    If there is only one model version, the bars use a left-to-right gradient.
    """

    x = np.arange(len(df["Topics"]))

    metric_columns = [
        col for col in df.columns
        if col.endswith(f"_{metric_suffix}")
    ]

    labels = [
        display_version(model_name, col.replace(f"_{metric_suffix}", ""))
        for col in metric_columns
    ]

    bar_width = 0.8 / len(metric_columns)

    plt.figure(figsize=(10.5, 6))

    if len(metric_columns) == 1:
        col = metric_columns[0]

        plt.bar(
            x,
            df[col],
            width=0.65,
            label=labels[0],
            color=get_bar_gradient_colors(len(x))
        )

    else:
        for i, col in enumerate(metric_columns):
            offset = (i - (len(metric_columns) - 1) / 2) * bar_width

            plt.bar(
                x + offset,
                df[col],
                width=bar_width,
                label=labels[i]
            )

    plt.xlabel("Number of topics")
    plt.ylabel(y_label)
    plt.title(title, pad=12)
    plt.xticks(x, df["Topics"])

    plt.legend(title="Model version", frameon=True)

    plt.grid(axis="y", linestyle="--", alpha=0.4)

    if y_min is not None or y_max is not None:
        plt.ylim(y_min, y_max)

    plt.tight_layout()

    filename = f"{safe_name(model_name)}_{metric_suffix.lower()}"
    save_figure(filename)

    plt.close()


# ------------------------------------------------------------
# Convert all model data to long format
# ------------------------------------------------------------

def make_long_results(data):
    """
    Converts the dictionary of model DataFrames into one long DataFrame.
    """

    rows = []

    for model_name, df in data.items():
        coherence_columns = [
            col for col in df.columns
            if col.endswith("_TC")
        ]

        for coherence_col in coherence_columns:
            version = coherence_col.replace("_TC", "")

            diversity_col = f"{version}_TD"
            runtime_col = f"{version}_time"

            if diversity_col not in df.columns or runtime_col not in df.columns:
                continue

            shown_version = display_version(model_name, version)

            for _, row in df.iterrows():
                rows.append({
                    "Model": model_name,
                    "Version": shown_version,
                    "Original_Version": version,
                    "Topics": int(row["Topics"]),
                    "Coherence": row[coherence_col],
                    "Diversity": row[diversity_col],
                    "Runtime": row[runtime_col],
                    "Label": f"{model_name}\n{shown_version}, K={int(row['Topics'])}",
                    "Summary_Label": model_name
                })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Best version comparison
# ------------------------------------------------------------

def get_best_runs_by_coherence(long_df):
    """
    Selects the best run for each model based on highest coherence.
    """

    best_rows = []

    for model_name, group in long_df.groupby("Model"):
        best_row = group.loc[group["Coherence"].idxmax()]
        best_rows.append(best_row)

    best_df = pd.DataFrame(best_rows)

    best_df = best_df.sort_values(
        by="Coherence",
        ascending=False
    ).reset_index(drop=True)

    return best_df


def plot_best_model_comparison(
    best_df,
    metric,
    y_label,
    title,
    y_min=None,
    y_max=None
):
    """
    Bar chart comparing the best run from each model.

    The summary chart only uses the model name as the x-axis label.
    The configuration details are handled separately in the Typst table.

    Summary bars use a left-to-right gradient.
    """

    x = np.arange(len(best_df))

    plt.figure(figsize=(10.5, 6))

    plt.bar(
        x,
        best_df[metric],
        color=get_bar_gradient_colors(len(best_df))
    )

    plt.xticks(
        x,
        best_df["Summary_Label"],
        rotation=25,
        ha="right"
    )

    plt.ylabel(y_label)
    plt.title(title, pad=12)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    if y_min is not None or y_max is not None:
        plt.ylim(y_min, y_max)

    for i, value in enumerate(best_df[metric]):
        if metric == "Runtime":
            text = f"{value:.1f}s"
        else:
            text = f"{value:.3f}"

        plt.text(
            i,
            value,
            text,
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()

    filename = f"summary_best_models_{safe_name(metric)}"
    save_figure(filename)

    plt.close()


def plot_best_models_scatter(best_df):
    """
    Scatter plot comparing the best model runs.

    X-axis: coherence
    Y-axis: diversity

    The legend only uses model names.
    The configuration details are handled separately in the Typst table.
    """

    markers = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">"]

    plt.figure(figsize=(9.5, 6.8))

    for i, (_, row) in enumerate(best_df.iterrows()):
        label = row["Summary_Label"]

        plt.scatter(
            row["Coherence"],
            row["Diversity"],
            s=130,
            marker=markers[i % len(markers)],
            label=label,
            edgecolors="black",
            linewidths=0.7
        )

    plt.xlabel("Topic coherence")
    plt.ylabel("Topic diversity")
    plt.title("Best runs by coherence: coherence vs diversity", pad=12)
    plt.xlim(0, 0.22)
    plt.ylim(0, 1.02)
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.legend(
        title="Model",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=2,
        frameon=True,
        framealpha=0.9
    )

    plt.tight_layout()

    filename = "summary_best_models_coherence_vs_diversity_scatter"
    save_figure(filename)

    plt.close()


# ------------------------------------------------------------
# Typst summary table
# ------------------------------------------------------------

def print_typst_best_runs_table(best_df):
    """
    Prints a Typst table containing the configuration details for the
    best run from each model.

    This table is meant to accompany the summary plots, where only the
    model names are shown.
    """

    table_df = best_df.copy()

    models = table_df["Model"].tolist()
    versions = table_df["Version"].tolist()
    topics = table_df["Topics"].tolist()
    coherence = table_df["Coherence"].tolist()
    diversity = table_df["Diversity"].tolist()
    runtime = table_df["Runtime"].tolist()

    num_columns = len(models) + 1
    columns = ", ".join(["auto"] * num_columns)

    print("\nTypst table for best-run configurations:\n")

    print("#figure(")
    print("  table(")
    print(f"    columns: ({columns}),")
    print("    align: center + horizon,")
    print("    inset: 5pt,")
    print("    stroke: 1pt + luma(10),")
    print("")
    print("    fill: (x, y) => {")
    print("      if y == 0 or x == 0 and y > 0 {")
    print("        header-fill1")
    print("      } else if calc.even(x) and y > 0 {")
    print("        row-fill1")
    print("      } else {")
    print("        white")
    print("      }")
    print("    },")
    print("")
    print("    table.header(")
    print("      [*Metric*],")
    for model in models:
        comma = "," if model != models[-1] else ""
        print(f"      [*{model}*]{comma}")
    print("    ),")
    print("")

    def print_row(row_name, values):
        print(f"    [*{row_name}*],", end=" ")
        formatted_values = []
        for value in values:
            formatted_values.append(f"[{value}]")
        print(", ".join(formatted_values) + ",")

    print_row("Configuration", versions)
    print_row("$K$", topics)
    print_row("@tc", [f"{v:.3f}" for v in coherence])
    print_row("@td", [f"{v:.3f}" for v in diversity])
    print_row("t (s)", [f"{v:.1f}" for v in runtime])

    print("  ),")
    print("  caption: [Best run selected for each model according to highest Topic Coherence (@tc). The table shows the selected configuration, number of topics ($K$), Topic Coherence (@tc), Topic Diversity (@td), and runtime (t) in seconds.]")
    print(")<table:best-model-configurations>")


# ------------------------------------------------------------
# Main script
# ------------------------------------------------------------

summary_df = load_summary(SUMMARY_CSV)
data = build_data_from_summary(summary_df)

print(f"\nLoaded summary input from: {SUMMARY_CSV}")
print("Generating charts from mean values in the summary file.")


# ------------------------------------------------------------
# Generate all per-model bar charts
# ------------------------------------------------------------

print("\nGenerating per-model charts...\n")

for model_name, df in data.items():

    print(f"Generating charts for: {model_name}")

    # 1. Topic coherence bar chart
    plot_grouped_bars(
        df=df,
        model_name=model_name,
        metric_suffix="TC",
        y_label="Topic coherence",
        title=f"{model_name}: Topic coherence by number of topics",
        y_min=0,
        y_max=0.22
    )

    # 2. Topic diversity bar chart
    plot_grouped_bars(
        df=df,
        model_name=model_name,
        metric_suffix="TD",
        y_label="Topic diversity",
        title=f"{model_name}: Topic diversity by number of topics",
        y_min=0,
        y_max=1
    )

    # 3. Runtime bar chart
    plot_grouped_bars(
        df=df,
        model_name=model_name,
        metric_suffix="time",
        y_label="Runtime, t (s)",
        title=f"{model_name}: Runtime by number of topics"
    )


# ------------------------------------------------------------
# Generate final summary charts
# ------------------------------------------------------------

print("\nGenerating summary charts...\n")

long_results = make_long_results(data)
best_runs = get_best_runs_by_coherence(long_results)

# Save the full long results and the best runs as CSV files
long_results.to_csv(
    os.path.join(OUTPUT_DIR, "all_results_long_format.csv"),
    index=False
)

best_runs.to_csv(
    os.path.join(OUTPUT_DIR, "best_runs_by_coherence.csv"),
    index=False
)

print("\nBest runs by coherence:")
print(best_runs[[
    "Model",
    "Version",
    "Topics",
    "Coherence",
    "Diversity",
    "Runtime"
]])

# Summary chart 1: coherence
plot_best_model_comparison(
    best_df=best_runs,
    metric="Coherence",
    y_label="Topic coherence",
    title="Best run from each model by coherence",
    y_min=0,
    y_max=0.22
)

# Summary chart 2: diversity
plot_best_model_comparison(
    best_df=best_runs,
    metric="Diversity",
    y_label="Topic diversity",
    title="Diversity of each model's best coherence run",
    y_min=0,
    y_max=1
)

# Summary chart 3: runtime
plot_best_model_comparison(
    best_df=best_runs,
    metric="Runtime",
    y_label="Runtime, t (s)",
    title="Runtime of each model's best coherence run"
)

# Summary chart 4: coherence-diversity scatter
plot_best_models_scatter(best_runs)

# Print Typst table for selected model configurations
print_typst_best_runs_table(best_runs)


# ------------------------------------------------------------
# Final confirmation
# ------------------------------------------------------------

expected_files = [
    "bertopic_base_tc.png",
    "bertopic_base_td.png",
    "bertopic_base_time.png",

    "bertopic_hdbscan_tc.png",
    "bertopic_hdbscan_td.png",
    "bertopic_hdbscan_time.png",

    "bertopic_reassign_tc.png",
    "bertopic_reassign_td.png",
    "bertopic_reassign_time.png",

    "ctm_tc.png",
    "ctm_td.png",
    "ctm_time.png",

    "etm_tc.png",
    "etm_td.png",
    "etm_time.png",

    "lda_tc.png",
    "lda_td.png",
    "lda_time.png",

    "summary_best_models_coherence.png",
    "summary_best_models_diversity.png",
    "summary_best_models_runtime.png",
    "summary_best_models_coherence_vs_diversity_scatter.png",

    "all_results_long_format.csv",
    "best_runs_by_coherence.csv",
]

print("\nDone. The following files should now be in the charts folder:\n")
for filename in expected_files:
    path = os.path.join(OUTPUT_DIR, filename)
    status = "OK" if os.path.exists(path) else "MISSING"
    print(f"{status}: {path}")
