import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



"""
I did not do anything in the chart scripts, I just asked AI to do everything here,

Ofcourse I still decided how they should look and implemented feedback from my supervisors 
"""

# ------------------------------------------------------------
# Output folder: new folder, does NOT overwrite old charts
# ------------------------------------------------------------

OUTPUT_DIR = "charts_2M_best_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

EVAL_CSV = "evaluation_metrics_2M_runs.csv"
TIMING_CSV = "2Mrun_model_timing_log.csv"


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
})


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def save_figure(filename):
    path = os.path.join(OUTPUT_DIR, f"{filename}.png")

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.15
    )

    print(f"Saved: {path}")


def get_bar_gradient_colors(n):
    return plt.cm.viridis(np.linspace(0.25, 0.85, n))


def find_eval_row(eval_df, model, k):
    """
    Finds the evaluation row for one model and K.
    """

    subset = eval_df[
        (eval_df["Model"].astype(str).str.lower() == model.lower()) &
        (eval_df["K"].astype(int) == int(k)) &
        (eval_df["Status"].astype(str) == "ok")
    ]

    if subset.empty:
        raise ValueError(f"Could not find evaluation row for {model}, K={k}")

    # If several rows exist, use the one with highest NPMI.
    subset = subset.sort_values("NPMI", ascending=False)

    return subset.iloc[0]


def find_runtime_seconds(timing_df, possible_run_names, fallback_seconds=None):
    """
    Finds runtime in seconds from timing log.
    If no match is found, uses fallback_seconds.
    """

    if timing_df is None or timing_df.empty:
        if fallback_seconds is not None:
            return fallback_seconds
        raise ValueError("Timing CSV missing and no fallback runtime was provided.")

    for run_name in possible_run_names:
        subset = timing_df[timing_df["run_name"].astype(str) == run_name]

        if not subset.empty:
            return float(subset.iloc[-1]["time_seconds"])

    if fallback_seconds is not None:
        return fallback_seconds

    raise ValueError(f"Could not find runtime for any of: {possible_run_names}")


def plot_metric(df, metric, y_label, title, filename, y_min=None, y_max=None):
    x = np.arange(len(df))

    plt.figure(figsize=(8.5, 5.6))

    plt.bar(
        x,
        df[metric],
        color=get_bar_gradient_colors(len(df))
    )

    plt.xticks(
        x,
        df["Model_Label"],
        rotation=20,
        ha="right"
    )

    plt.ylabel(y_label)
    plt.title(title, pad=12)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    if y_min is not None or y_max is not None:
        plt.ylim(y_min, y_max)

    for i, value in enumerate(df[metric]):
        if metric == "Runtime_seconds":
            text = f"{value / 60:.1f} min"
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
    save_figure(filename)
    plt.close()


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

eval_df = pd.read_csv(EVAL_CSV)

if os.path.exists(TIMING_CSV):
    timing_df = pd.read_csv(TIMING_CSV)
else:
    print(f"Warning: {TIMING_CSV} not found. Using fallback runtimes.")
    timing_df = None


# ------------------------------------------------------------
# Select only the best 2M models
# ------------------------------------------------------------
# These are the models you want in the presentation:
#
# BERTopic: best BERTopic full-corpus assignment run
# CTM:      best CTM full-corpus assignment run
# LDA:      K=30, because it was the best LDA setting in your main results
#
# Fallback runtimes are included from your logs:
# BERTopic: 18.54 min
# CTM:      40.46 min
# LDA K30:  43.57 min

selected_models = [
    {
        "model": "BERTopic",
        "label": "BERTopic\nD2, K=30",
        "k": 30,
        "runtime_names": [
            "Best2_2M_K30_fit150k_transform_all",
            "BERTopic Base D2 K30 fit150k transform all",
        ],
        "fallback_seconds": 18.54 * 60,
    },
    {
        "model": "CTM",
        "label": "CTM\nD1, K=20",
        "k": 20,
        "runtime_names": [
            "CTM_LIGHT_2M_K20_fit150k_transform_all",
            "CTM Light K20 fit150k transform all",
        ],
        "fallback_seconds": 40.46 * 60,
    },
    {
        "model": "LDA",
        "label": "LDA\nV2000, K=30",
        "k": 30,
        "runtime_names": [
            "LDA_2M_VOCAB2000_K30_fullcorpus",
            "LDA Vocab2000 K30 full corpus",
        ],
        "fallback_seconds": 43.57 * 60,
    },
]


rows = []

for item in selected_models:
    eval_row = find_eval_row(
        eval_df=eval_df,
        model=item["model"],
        k=item["k"],
    )

    runtime_seconds = find_runtime_seconds(
        timing_df=timing_df,
        possible_run_names=item["runtime_names"],
        fallback_seconds=item["fallback_seconds"],
    )

    rows.append({
        "Model": item["model"],
        "Model_Label": item["label"],
        "K": item["k"],
        "Coherence": float(eval_row["NPMI"]),
        "Diversity": float(eval_row["Diversity"]),
        "Runtime_seconds": runtime_seconds,
        "Valid_Topics": int(eval_row["Valid_Topics"]),
        "Topic_File": eval_row["Topic_File"],
    })


best_df = pd.DataFrame(rows)

# Sort by coherence for coherence/diversity charts
best_df = best_df.sort_values("Coherence", ascending=False).reset_index(drop=True)

summary_path = os.path.join(OUTPUT_DIR, "best_2M_models_summary.csv")
best_df.to_csv(summary_path, index=False)

print("\nSelected 2M models:")
print(best_df[[
    "Model",
    "K",
    "Coherence",
    "Diversity",
    "Runtime_seconds",
    "Valid_Topics",
]])

print(f"\nSaved summary CSV: {summary_path}")


# ------------------------------------------------------------
# Generate 3 charts
# ------------------------------------------------------------

plot_metric(
    df=best_df,
    metric="Coherence",
    y_label="Topic coherence",
    title="2M runs: Topic coherence",
    filename="2M_best_models_coherence",
    y_min=0,
    y_max=0.22,
)

plot_metric(
    df=best_df,
    metric="Diversity",
    y_label="Topic diversity",
    title="2M runs: Topic diversity",
    filename="2M_best_models_diversity",
    y_min=0,
    y_max=1,
)

# For runtime, sort fastest to slowest
runtime_df = best_df.sort_values("Runtime_seconds", ascending=True).reset_index(drop=True)

plot_metric(
    df=runtime_df,
    metric="Runtime_seconds",
    y_label="Runtime, t (s)",
    title="2M runs: Runtime",
    filename="2M_best_models_runtime",
)

print("\nDone. Created:")
print(os.path.join(OUTPUT_DIR, "2M_best_models_coherence.png"))
print(os.path.join(OUTPUT_DIR, "2M_best_models_diversity.png"))
print(os.path.join(OUTPUT_DIR, "2M_best_models_runtime.png"))
print(os.path.join(OUTPUT_DIR, "best_2M_models_summary.csv"))