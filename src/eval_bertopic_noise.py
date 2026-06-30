import pandas as pd
import glob
import os
import re


def parse_bertopic_assignment_filename(filename):
    """
    Parses filenames like:
      bertopic_assignments_LIGHT_EMBfinal_fullK10.csv
      bertopic_assignments_LIGHT_EMBfinal2_fullK10.csv
      bertopic_assignments_LIGHT_EMBfinal3_fullK10.csv

      bertopic_assignments_LIGHT_EMBfinal_noiseparams_fullK10.csv
      bertopic_assignments_LIGHT_EMBfinal2_noiseparams_fullK10.csv
      bertopic_assignments_LIGHT_EMBfinal3_noiseparams_fullK10.csv

      bertopic_assignments_LIGHT_EMBfinal_reassignment_fullK10.csv
      bertopic_assignments_LIGHT_EMBfinal2_reassignment_fullK10.csv
      bertopic_assignments_LIGHT_EMBfinal3_reassignment_fullK10.csv
    """
    base = os.path.basename(filename)
    name = base.replace("bertopic_assignments_", "").replace(".csv", "")

    embedding = "Light" if "LIGHT" in name.upper() else "Heavy" if "HEAVY" in name.upper() else ""

    if "noiseparams" in name.lower():
        variant = "NoiseParams"
    elif "reassignment" in name.lower():
        variant = "Reassignment"
    else:
        variant = "Base"

    if "final3" in name:
        repeat = "Run3"
    elif "final2" in name:
        repeat = "Run2"
    elif "final" in name:
        repeat = "Run1"
    else:
        repeat = "Unknown"

    k_match = re.search(r"K(\d+)", name)
    k_value = int(k_match.group(1)) if k_match else None

    return {
        "Experiment": name,
        "Model": "BERTopic",
        "Variant": variant,
        "Embedding": embedding,
        "K": k_value,
        "Repeat": repeat,
    }


def calculate_noise_metrics():
    results = []
    current_dir = os.getcwd()

    variations = ["LIGHT", "HEAVY"]

    csv_files = []

    for var in variations:
        patterns = [
            # Base
            os.path.join(current_dir, f"bertopic_assignments_{var}_EMBfinal_fullK*.csv"),
            os.path.join(current_dir, f"bertopic_assignments_{var}_EMBfinal2_fullK*.csv"),
            os.path.join(current_dir, f"bertopic_assignments_{var}_EMBfinal3_fullK*.csv"),

            # NoiseParams
            os.path.join(current_dir, f"bertopic_assignments_{var}_EMBfinal_noiseparams_fullK*.csv"),
            os.path.join(current_dir, f"bertopic_assignments_{var}_EMBfinal2_noiseparams_fullK*.csv"),
            os.path.join(current_dir, f"bertopic_assignments_{var}_EMBfinal3_noiseparams_fullK*.csv"),

            # Reassignment
            os.path.join(current_dir, f"bertopic_assignments_{var}_EMBfinal_reassignment_fullK*.csv"),
            os.path.join(current_dir, f"bertopic_assignments_{var}_EMBfinal2_reassignment_fullK*.csv"),
            os.path.join(current_dir, f"bertopic_assignments_{var}_EMBfinal3_reassignment_fullK*.csv"),
        ]

        for pattern in patterns:
            csv_files.extend(glob.glob(pattern))

    csv_files = sorted(set(csv_files))

    if not csv_files:
        print("Error: No files found matching BERTopic assignment CSV patterns.")
        return None, None

    print(f"Found {len(csv_files)} BERTopic assignment files.")

    for file in csv_files:
        df = pd.read_csv(file)
        filename = os.path.basename(file)

        target_col = "topic_id"

        if target_col not in df.columns:
            print(f"Skipping {filename}: '{target_col}' column not found.")
            continue

        total_docs = len(df)
        noise_docs = len(df[df[target_col] == -1])
        noise_percentage = (noise_docs / total_docs) * 100 if total_docs > 0 else float("nan")

        unique_topics = df[target_col].unique()
        num_topics = len([t for t in unique_topics if t != -1])

        parsed = parse_bertopic_assignment_filename(filename)

        results.append({
            **parsed,
            "Source_File": filename,
            "Total_Docs": total_docs,
            "Noise_Docs": noise_docs,
            "Noise_Percent": noise_percentage,
            "Final_Topics_Found": num_topics,
        })

    detail_df = pd.DataFrame(results)

    if detail_df.empty:
        return None, None

    detail_df = detail_df.sort_values(["Variant", "Embedding", "K", "Repeat"])

    summary_df = (
        detail_df
        .groupby(["Model", "Variant", "Embedding", "K"], dropna=False)
        .agg(
            Runs=("Repeat", "count"),
            Total_Docs_mean=("Total_Docs", "mean"),
            Noise_Docs_mean=("Noise_Docs", "mean"),
            Noise_Docs_std=("Noise_Docs", "std"),
            Noise_Percent_mean=("Noise_Percent", "mean"),
            Noise_Percent_std=("Noise_Percent", "std"),
            Final_Topics_Found_mean=("Final_Topics_Found", "mean"),
            Final_Topics_Found_std=("Final_Topics_Found", "std"),
        )
        .reset_index()
        .sort_values(["Variant", "Embedding", "K"])
    )

    return detail_df, summary_df


# Execute
detail, summary = calculate_noise_metrics()

if detail is not None and summary is not None:
    print("\n--- NOISE ANALYSIS DETAIL ---")
    print(detail.to_string(index=False))

    print("\n--- NOISE ANALYSIS MEAN / STD SUMMARY ---")
    rounded_summary = summary.copy()
    for col in rounded_summary.columns:
        if col.endswith("_mean") or col.endswith("_std"):
            rounded_summary[col] = rounded_summary[col].round(3)

    print(rounded_summary.to_string(index=False))

    detail.to_csv("bertopic_noise_detail_all_runs.csv", index=False)
    rounded_summary.to_csv("bertopic_noise_mean_std_summary.csv", index=False)

    print("\nSaved:")
    print("  bertopic_noise_detail_all_runs.csv")
    print("  bertopic_noise_mean_std_summary.csv")