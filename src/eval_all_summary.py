import os
import re
import pandas as pd

"""
I used AI to generate this script based on my previous evaluations scripts,
it now takes the evaluated metrics and the timing log and merges them into a single summary with mean/std values for each metric and runtime.
"""



# ── Settings ────────────────────────────────────────────────────────────────
EVALUATION_CSV = "evaluation_metrics_all_runs.csv"
TIMING_CSV = "final_model_timing_log.csv"

OUTPUT_SUMMARY_CSV = "evaluation_metrics_runtime_mean_std_summary.csv"
OUTPUT_DETAIL_CSV = "evaluation_metrics_runtime_with_group_keys.csv"
OUTPUT_TIMING_PARSED_CSV = "model_timing_log_parsed.csv"

METRIC_COLUMNS = ["NPMI", "Diversity", "Valid_Topics", "Dropped_Topics"]
GROUP_COLUMNS = ["Model", "Variant", "Embedding", "Vocab", "K"]
GROUP_REPEAT_COLUMNS = GROUP_COLUMNS + ["Repeat"]


def normalize_empty(value):
    if pd.isna(value):
        return ""
    return str(value)


def parse_repeat(run_name):
    run_name = str(run_name)
    if "final3" in run_name:
        return "Run3"
    if "final2" in run_name:
        return "Run2"
    if "final" in run_name:
        return "Run1"
    return "Unknown"


def parse_timing_row(row):
    """Parse model_timing_log.csv run names into the same keys as the eval CSV.

    Handles names such as:
      LIGHT_EMBfinal_K10
      LIGHT_EMBfinal2_fullK10
      LIGHT_EMBfinal_noiseparams_K10
      LIGHT_EMBfinal2_noiseparams_fullK10
      LIGHT_EMBfinal_reassignment_K10
      ctm style: LIGHT_EMBfinal_K10
      ETM/LDA style: HEAVY_final2_VOCAB2000_K10
    """
    model = str(row.get("model", "")).strip()
    run_name = str(row.get("run_name", ""))

    # K should already exist in the timing log
    k_value = row.get("Timing_K", None)
    if pd.isna(k_value):
        match = re.search(r"K(\d+)", run_name)
        k_value = int(match.group(1)) if match else None

    # Embedding / preprocessing label
    if "LIGHT" in run_name.upper():
        embedding = "Light"
    elif "HEAVY" in run_name.upper():
        embedding = "Heavy"
    else:
        embedding = ""

    # Repeat
    repeat = parse_repeat(run_name)

    # Vocabulary
    vocab_match = re.search(r"VOCAB(\d+)", run_name, flags=re.IGNORECASE)
    vocab = f"Vocab{vocab_match.group(1)}" if vocab_match else ""

    # Variant
    if model == "BERTopic":
        if "noiseparams" in run_name.lower():
            variant = "NoiseParams fullK"
        elif "reassignment" in run_name.lower():
            variant = "Reassignment fullK"
        else:
            variant = "Base fullK"
    else:
        variant = "Base"

    return pd.Series({
        "Model": model,
        "Variant": variant,
        "Embedding": embedding,
        "Vocab": vocab,
        "K": int(k_value) if k_value is not None and not pd.isna(k_value) else None,
        "Repeat": repeat,
    })


def load_evaluation_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Evaluation CSV not found: {path}")

    print(f"Loading evaluation metrics from {path}")
    df_eval = pd.read_csv(path)

    required = set(GROUP_REPEAT_COLUMNS + METRIC_COLUMNS + ["Source_File"])
    missing = sorted(required - set(df_eval.columns))
    if missing:
        raise ValueError(f"Evaluation CSV is missing required columns: {missing}")

    for col in GROUP_COLUMNS + ["Repeat"]:
        df_eval[col] = df_eval[col].apply(normalize_empty)

    df_eval["K"] = pd.to_numeric(df_eval["K"], errors="coerce").astype("Int64")

    for col in METRIC_COLUMNS:
        df_eval[col] = pd.to_numeric(df_eval[col], errors="coerce")

    return df_eval

def load_timing_csv(path):
    if not os.path.exists(path):
        print(f"Warning: timing file not found: {path}")
        print("Runtime columns will be empty in the final summary.")
        return None

    print(f"Loading runtime data from {path}")
    df_time = pd.read_csv(path)

    required = {"model", "run_name", "K", "time_seconds", "time_minutes"}
    missing = sorted(required - set(df_time.columns))
    if missing:
        raise ValueError(f"Timing CSV is missing required columns: {missing}")

    # Keep the original timing K under a different name to avoid duplicate K columns.
    df_time = df_time.rename(columns={"K": "Timing_K"})

    parsed = df_time.apply(parse_timing_row, axis=1)
    df_time = pd.concat([df_time, parsed], axis=1)

    for col in GROUP_COLUMNS + ["Repeat"]:
        df_time[col] = df_time[col].apply(normalize_empty)

    df_time["K"] = pd.to_numeric(df_time["K"], errors="coerce").astype("Int64")
    df_time["time_seconds"] = pd.to_numeric(df_time["time_seconds"], errors="coerce")
    df_time["time_minutes"] = pd.to_numeric(df_time["time_minutes"], errors="coerce")

    df_time.to_csv(OUTPUT_TIMING_PARSED_CSV, index=False)
    print(f"Saved parsed timing log to {OUTPUT_TIMING_PARSED_CSV}")

    return df_time

    
def summarize_timing_by_repeat(df_time):
    """Return one timing row per Model/Variant/Embedding/Vocab/K/Repeat.

    If the timing log contains duplicate entries for the same run, this averages
    them and reports how many were found. This is safer than crashing if the log
    contains old runs mixed with the new repeated runs.
    """
    if df_time is None:
        return None

    timing_by_repeat = (
        df_time
        .groupby(GROUP_REPEAT_COLUMNS, dropna=False)
        .agg(
            Runtime_Seconds=("time_seconds", "mean"),
            Runtime_Minutes=("time_minutes", "mean"),
            Runtime_Log_Entries=("time_seconds", "count"),
            Timing_Run_Names=("run_name", lambda s: "; ".join(map(str, s))),
        )
        .reset_index()
    )

    duplicates = timing_by_repeat[timing_by_repeat["Runtime_Log_Entries"] > 1]
    if len(duplicates) > 0:
        print("\nWarning: multiple timing log entries matched some evaluation runs.")
        print("Those runtime values were averaged. Check model_timing_log_parsed.csv if this is unexpected.")
        print(duplicates[GROUP_REPEAT_COLUMNS + ["Runtime_Log_Entries", "Timing_Run_Names"]].to_string(index=False))

    return timing_by_repeat


def main():
    df_eval = load_evaluation_csv(EVALUATION_CSV)
    df_time = load_timing_csv(TIMING_CSV)
    df_time_by_repeat = summarize_timing_by_repeat(df_time)

    if df_time_by_repeat is not None:
        df_detail = df_eval.merge(
            df_time_by_repeat,
            on=GROUP_REPEAT_COLUMNS,
            how="left",
        )

        missing_runtime = df_detail["Runtime_Seconds"].isna().sum()
        if missing_runtime > 0:
            print(f"\nWarning: {missing_runtime} evaluated rows did not find a matching runtime entry.")
            print("Rows without runtime are still included in the metric summary.")
            missing_rows = df_detail[df_detail["Runtime_Seconds"].isna()]
            print(missing_rows[GROUP_REPEAT_COLUMNS + ["Source_File"]].to_string(index=False))
    else:
        df_detail = df_eval.copy()
        df_detail["Runtime_Seconds"] = pd.NA
        df_detail["Runtime_Minutes"] = pd.NA
        df_detail["Runtime_Log_Entries"] = pd.NA
        df_detail["Timing_Run_Names"] = ""

    df_detail.to_csv(OUTPUT_DETAIL_CSV, index=False)

    # Convert runtime columns to numeric after merge.
    df_detail["Runtime_Seconds"] = pd.to_numeric(df_detail["Runtime_Seconds"], errors="coerce")
    df_detail["Runtime_Minutes"] = pd.to_numeric(df_detail["Runtime_Minutes"], errors="coerce")

    summary = (
        df_detail
        .groupby(GROUP_COLUMNS, dropna=False)
        .agg(
            Runs=("Repeat", "count"),
            Metric_Runs=("NPMI", "count"),
            Runtime_Runs=("Runtime_Seconds", "count"),
            NPMI_mean=("NPMI", "mean"),
            NPMI_std=("NPMI", "std"),
            Diversity_mean=("Diversity", "mean"),
            Diversity_std=("Diversity", "std"),
            Valid_Topics_mean=("Valid_Topics", "mean"),
            Valid_Topics_std=("Valid_Topics", "std"),
            Dropped_Topics_mean=("Dropped_Topics", "mean"),
            Dropped_Topics_std=("Dropped_Topics", "std"),
            Runtime_Seconds_mean=("Runtime_Seconds", "mean"),
            Runtime_Seconds_std=("Runtime_Seconds", "std"),
            Runtime_Minutes_mean=("Runtime_Minutes", "mean"),
            Runtime_Minutes_std=("Runtime_Minutes", "std"),
        )
        .reset_index()
        .sort_values(["Model", "Variant", "Embedding", "Vocab", "K"])
    )

    rounded = summary.copy()
    for col in rounded.columns:
        if col.endswith("_mean") or col.endswith("_std"):
            rounded[col] = rounded[col].round(4)

    rounded.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print("\n=== Aggregated mean/std summary with runtime ===")
    print(rounded.to_string(index=False))
    print(f"\nSaved summary to {OUTPUT_SUMMARY_CSV}")
    print(f"Saved detail file to {OUTPUT_DETAIL_CSV}")


if __name__ == "__main__":
    main()
