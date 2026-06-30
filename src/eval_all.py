import json
import os
import sys
import unicodedata
import string
import pandas as pd
from gensim.corpora import Dictionary

from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity


"""
I messed up the naming convention for the topic files
they are called final, final2, final3, instead of havng the first one called final1.
I used AI for the legic to acount for this, instead of renaming the files.
"""


# ── Settings ────────────────────────────────────────────────────────────────
OUTPUT_CSV = "evaluation_metrics_all_runs.csv"
OUTPUT_LOG = "evaluation_results_log_all_runs.txt"
STRICT_FILE_CHECK = True  # if True, stop before evaluation if any topic JSON is missing


# ── Logging Setup ────────────────────────────────────────────────────────────
class Logger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger(OUTPUT_LOG)
results_data = []
experiments = []


def first_existing_file(candidates):
    """Return the first existing file from a list of possible names.

    This lets the script handle the naming differences between files that use
    _fullK and files that only use _K, without changing your experiment layout.
    """
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def add_experiment(model, variant, embedding, vocab, K, repeat, json_file_path, timing_run_name):
    """Stores one experiment specification for later file checking and evaluation."""
    run_name_parts = [model]
    if variant:
        run_name_parts.append(variant)
    if embedding:
        run_name_parts.append(embedding)
    if vocab:
        run_name_parts.append(str(vocab))
    run_name_parts.append(f"K{K}")
    run_name_parts.append(f"{repeat}")
    run_name = " | ".join(run_name_parts)

    experiments.append({
        "Model": model,
        "Variant": variant,
        "Embedding": embedding,
        "Vocab": vocab,
        "K": K,
        "Repeat": repeat,
        "Experiment": run_name,
        "Source_File": json_file_path,
        "Timing_Run_Name": timing_run_name,
    })


# === 0. Define all topic files to evaluate ===
K_VALUES = [10, 20, 30, 40, 50, 60]

# Repetition names follow the naming you used in the model scripts:
#   final  = first run
#   final2 = second run
#   final3 = third run
REPEATS = [
    ("final", "Run1"),
    ("final2", "Run2"),
    ("final3", "Run3"),
]


# ── BERTopic Base, fullK ────────────────────────────────────────────────────
# This accepts either of these topic filename styles:
#   bertopic_topics_LIGHT_EMBfinal_fullK10.json
#   bertopic_topics_LIGHT_EMBfinal_K10.json
for final_tag, repeat_label in REPEATS:
    for K in K_VALUES:
        bertopic_base_fulllight = first_existing_file([
            f"bertopic_topics_LIGHT_EMB{final_tag}_fullK{K}.json",
            f"bertopic_topics_LIGHT_EMB{final_tag}_K{K}.json",
        ])
        bertopic_base_fullheavy = first_existing_file([
            f"bertopic_topics_HEAVY_EMB{final_tag}_fullK{K}.json",
            f"bertopic_topics_HEAVY_EMB{final_tag}_K{K}.json",
        ])

        add_experiment(
            "BERTopic", "Base fullK", "Light", "", K, repeat_label,
            bertopic_base_fulllight,
            f"LIGHT_EMB{final_tag}_K{K}",
        )
        add_experiment(
            "BERTopic", "Base fullK", "Heavy", "", K, repeat_label,
            bertopic_base_fullheavy,
            f"HEAVY_EMB{final_tag}_K{K}",
        )


# ── BERTopic Noise Parameters, fullK ────────────────────────────────────────
# This accepts either of these topic filename styles:
#   bertopic_topics_LIGHT_EMBfinal_noiseparams_fullK10.json
#   bertopic_topics_LIGHT_EMBfinal_noiseparams_K10.json
for final_tag, repeat_label in REPEATS:
    for K in K_VALUES:
        bertopic_noise_fulllight = first_existing_file([
            f"bertopic_topics_LIGHT_EMB{final_tag}_noiseparams_fullK{K}.json",
            f"bertopic_topics_LIGHT_EMB{final_tag}_noiseparams_K{K}.json",
        ])
        bertopic_noise_fullheavy = first_existing_file([
            f"bertopic_topics_HEAVY_EMB{final_tag}_noiseparams_fullK{K}.json",
            f"bertopic_topics_HEAVY_EMB{final_tag}_noiseparams_K{K}.json",
        ])

        add_experiment(
            "BERTopic", "NoiseParams fullK", "Light", "", K, repeat_label,
            bertopic_noise_fulllight,
            f"LIGHT_EMB{final_tag}_noiseparams_K{K}",
        )
        add_experiment(
            "BERTopic", "NoiseParams fullK", "Heavy", "", K, repeat_label,
            bertopic_noise_fullheavy,
            f"HEAVY_EMB{final_tag}_noiseparams_K{K}",
        )


# ── BERTopic Reassignment, fullK ────────────────────────────────────────────
# This accepts either of these topic filename styles:
#   bertopic_topics_LIGHT_EMBfinal_reassignment_fullK10.json
#   bertopic_topics_LIGHT_EMBfinal_reassignment_K10.json
for final_tag, repeat_label in REPEATS:
    for K in K_VALUES:
        bertopic_reassign_fulllight = first_existing_file([
            f"bertopic_topics_LIGHT_EMB{final_tag}_reassignment_fullK{K}.json",
            f"bertopic_topics_LIGHT_EMB{final_tag}_reassignment_K{K}.json",
        ])
        bertopic_reassign_fullheavy = first_existing_file([
            f"bertopic_topics_HEAVY_EMB{final_tag}_reassignment_fullK{K}.json",
            f"bertopic_topics_HEAVY_EMB{final_tag}_reassignment_K{K}.json",
        ])

        add_experiment(
            "BERTopic", "Reassignment fullK", "Light", "", K, repeat_label,
            bertopic_reassign_fulllight,
            f"LIGHT_EMB{final_tag}_reassignment_K{K}",
        )
        add_experiment(
            "BERTopic", "Reassignment fullK", "Heavy", "", K, repeat_label,
            bertopic_reassign_fullheavy,
            f"HEAVY_EMB{final_tag}_reassignment_K{K}",
        )


# ── CTM ─────────────────────────────────────────────────────────────────────
# Expected examples:
#   ctm_topics_LIGHT_EMBfinal_K10.json
#   ctm_topics_LIGHT_EMBfinal2_K10.json
#   ctm_topics_LIGHT_EMBfinal3_K10.json
for final_tag, repeat_label in REPEATS:
    for K in K_VALUES:
        ctm_light = f"ctm_topics_LIGHT_EMB{final_tag}_K{K}.json"
        ctm_heavy = f"ctm_topics_HEAVY_EMB{final_tag}_K{K}.json"

        add_experiment("CTM", "Base", "Light", "", K, repeat_label, ctm_light, f"LIGHT_EMB{final_tag}_K{K}")
        add_experiment("CTM", "Base", "Heavy", "", K, repeat_label, ctm_heavy, f"HEAVY_EMB{final_tag}_K{K}")


# ── ETM, vocabulary sensitivity ─────────────────────────────────────────────
# Expected examples:
#   etm_topics_HEAVY_final_VOCAB2000_K10.json
#   etm_topics_HEAVY_final2_VOCAB2000_K10.json
#   etm_topics_HEAVY_final3_VOCAB2000_K10.json
for final_tag, repeat_label in REPEATS:
    for K in K_VALUES:
        etm_2000 = f"etm_topics_HEAVY_{final_tag}_VOCAB2000_K{K}.json"
        etm_6000 = f"etm_topics_HEAVY_{final_tag}_VOCAB6000_K{K}.json"

        add_experiment("ETM", "Base", "Heavy", "Vocab2000", K, repeat_label, etm_2000, f"HEAVY_{final_tag}_VOCAB2000_K{K}")
        add_experiment("ETM", "Base", "Heavy", "Vocab6000", K, repeat_label, etm_6000, f"HEAVY_{final_tag}_VOCAB6000_K{K}")


# ── LDA baseline ────────────────────────────────────────────────────────────
# Expected examples:
#   lda_topics_HEAVY_final_VOCAB2000_K10.json
#   lda_topics_HEAVY_final2_VOCAB2000_K10.json
#   lda_topics_HEAVY_final3_VOCAB2000_K10.json
for final_tag, repeat_label in REPEATS:
    for K in K_VALUES:
        lda_2000 = f"lda_topics_HEAVY_{final_tag}_VOCAB2000_K{K}.json"
        add_experiment("LDA", "Base", "Heavy", "Vocab2000", K, repeat_label, lda_2000, f"HEAVY_{final_tag}_VOCAB2000_K{K}")


# === 1. Check that all expected files exist before evaluation ===
print("=== 1. Checking topic files ===")
missing_files = [exp["Source_File"] for exp in experiments if not os.path.exists(exp["Source_File"])]

print(f"Expected topic files: {len(experiments)}")
print(f"Found topic files:    {len(experiments) - len(missing_files)}")
print(f"Missing topic files:  {len(missing_files)}")

if missing_files:
    print("\nMissing files:")
    for path in missing_files:
        print(f"  - {path}")

    if STRICT_FILE_CHECK:
        raise FileNotFoundError(
            f"Found {len(missing_files)} missing topic JSON files. "
            "Evaluation stopped before starting so it will not crash in the middle. "
            "Either generate the missing files or update the filename list in this script."
        )


# === 2. Reconstruct the Reference Corpus ===
print("\n=== 2. Reconstructing the Reference Corpus ===")
# df_golden = pd.read_parquet("riksarkivet_GOLDEN_neural_150k.parquet")
# dictionary = Dictionary.load("riksarkivet_GOLDEN_bow_dictionary_0stopwords2_20.dict")
df_golden = pd.read_parquet("riksarkivet_final_neural_sample_150k.parquet")
dictionary = Dictionary.load("riksarkivet_final_bow_sample_dictionary_20.dict")  # Match your saved filename!
valid_words = set(dictionary.token2id.keys())
remove_punct = str.maketrans('', '', string.punctuation)

light_texts = df_golden['clean_text'].fillna("").tolist()

tokenized_reference_corpus = []
for text in light_texts:
    clean_str = unicodedata.normalize('NFC', str(text).lower())
    clean_str = clean_str.translate(remove_punct)
    words = [w for w in clean_str.split() if w in valid_words]
    tokenized_reference_corpus.append(words)

print(f"Constructed reference corpus with {len(tokenized_reference_corpus)} documents.")


# === 3. Initialize OCTIS Metrics ===
print("\n=== 3. Initializing OCTIS Metrics ===")
npmi_metric = Coherence(texts=tokenized_reference_corpus, topk=10, measure='c_npmi')
diversity_metric = TopicDiversity(topk=10)


def evaluate_topic_file(exp):
    json_file_path = exp["Source_File"]
    run_name = exp["Experiment"]

    print(f"\n--- Evaluating {run_name} ---")
    print(f"  -> File: {json_file_path}")

    with open(json_file_path, "r", encoding="utf-8") as f:
        raw_topics = json.load(f)

    clean_topics = []
    dropped_topics = 0

    for topic in raw_topics:
        # Only filter genuinely empty or whitespace-only strings.
        # Do NOT filter by dictionary membership — that inflates scores artificially.
        words = [w for w in topic if w and str(w).strip()]
        if len(words) >= 10:
            clean_topics.append(words[:10])
        else:
            dropped_topics += 1

    if dropped_topics > 0:
        print(f"  -> Dropped {dropped_topics} degenerate topics (fewer than 10 words).")

    if len(clean_topics) == 0:
        print("  -> ERROR: No valid topics. Score is NaN.")
        npmi_score = float("nan")
        diversity_score = float("nan")
    else:
        model_output = {"topics": clean_topics}
        npmi_score = npmi_metric.score(model_output)
        diversity_score = diversity_metric.score(model_output)

        print(f"  -> Topic Coherence (NPMI): {npmi_score:.4f}")
        print(f"  -> Topic Diversity:        {diversity_score:.4f}")
        print(f"  -> Successfully evaluated {len(clean_topics)} valid topics")

    results_data.append({
        "Model": exp["Model"],
        "Variant": exp["Variant"],
        "Embedding": exp["Embedding"],
        "Vocab": exp["Vocab"],
        "K": exp["K"],
        "Repeat": exp["Repeat"],
        "Experiment": exp["Experiment"],
        "Source_File": exp["Source_File"],
        "Timing_Run_Name": exp["Timing_Run_Name"],
        "NPMI": npmi_score,
        "Diversity": diversity_score,
        "Valid_Topics": len(clean_topics),
        "Dropped_Topics": dropped_topics,
    })


# === 4. Run the Evaluations ===
print("\n=== 4. Running Evaluations ===")

current_section = None
for exp in experiments:
    section = (exp["Model"], exp["Variant"])
    if section != current_section:
        print(f"\nEvaluating {exp['Model']} {exp['Variant']} =========")
        current_section = section

    evaluate_topic_file(exp)


# === 5. Save results ===
print("\n=== Final Summary Table ===")
df_final = pd.DataFrame(results_data)
print(df_final.to_string(index=False))

df_final.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved structured evaluation data to {OUTPUT_CSV}")
print(f"Saved log to {OUTPUT_LOG}")
print("\nEvaluation Complete!")
