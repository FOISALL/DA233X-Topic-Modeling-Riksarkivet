# Workflow and Script Order

This file records the overall order in which the thesis scripts were used.

## 1. Raw Parsing

Start with:

- `src/dataparsing.py`

This parses the raw archival/transcription files and produces the raw parsed data used by later preprocessing scripts.

## 2. Neural Dataset Preprocessing

Run:

- `src/preprocess_neural.py`

This creates the main cleaned neural dataset:

- `riksarkivet_neural_cleaned.parquet`

This file is an important shared input. It is used for model fine-tuning, sampling, full-corpus bag-of-words preprocessing, and full-corpus embedding generation.

## 3. Sentence-Transformer Fine-Tuning

Run:

- `src/finetune.py`

Input:

- `riksarkivet_neural_cleaned.parquet`

Output:

- `riksarkivet-sbert-historical-swedish-200k`

This fine-tuned model is later used when generating the embedding files.

## 4. Sample Dataset

Run:

- `src/neural_sample.py`

Input:

- `riksarkivet_neural_cleaned.parquet`

Output:

- `riksarkivet_final_neural_sample_150k.parquet`

This 150k sample was used to make smaller model experiments feasible.

## 5. Bag-of-Words Preprocessing

The files with `BoW` in their names create the preprocessed data used to build bag-of-words representations for the models.

For sample-based runs:

- `src/bow_sample.py`

Input:

- `riksarkivet_final_neural_sample_150k.parquet`

Typical outputs:

- `riksarkivet_final_bow_sample_dictionary_20.dict`
- `riksarkivet_final_bow_sample_corpus_20.mm`
- `riksarkivet_final_bow_sample_corpus_20.mm.index`

For full-corpus/2M runs:

- `src/preprocess_BoW_2M.py`

Input:

- `riksarkivet_neural_cleaned.parquet`

Typical outputs:

- `riksarkivet_2M_bow_dictionary_20.dict`
- `riksarkivet_2M_bow_corpus_20.mm`
- `riksarkivet_2M_bow_corpus_20.mm.index`

## 6. Embedding Generation

For final sample-based runs:

- `src/gen_embeddings.py`

Input:

- `riksarkivet_final_neural_sample_150k.parquet`

Outputs:

- `riksarkivet_final_embeddings_LIGHT.npy`
- `riksarkivet_final_embeddings_HEAVY.npy`

For full-corpus/2M runs:

- `src/gen_embeddings_full_light.py`
- `src/gen_embeddings_full.py`

Input:

- `riksarkivet_neural_cleaned.parquet`

Outputs:

- `riksarkivet_2M_embeddings_LIGHT.npy`
- `riksarkivet_2M_embeddings_HEAVY.npy`

## 7. Sample-Based Model Runs

The wrapper script:

- `src/runall.py`

launches several sample-based model experiments in sequence, including:

- `src/runbertopicbase.py`
- `src/runbertopicnoiseparams.py`
- `src/runbertopicreassignment.py`
- `src/runetm.py`
- `src/runlda.py`

The main topic models were:

| Model | Scripts |
| --- | --- |
| BERTopic | `src/runbertopicbase.py`, `src/runbertopicnoiseparams.py`, `src/runbertopicreassignment.py` |
| CTM | `src/runctm.py` |
| ETM | `src/runetm.py` |
| LDA | `src/runlda.py` |

## 8. Full-Corpus / 2M Model Runs

Separate scripts were used for full-corpus experiments:

| Model | Script |
| --- | --- |
| BERTopic | `src/runbertopicbase2M.py` |
| CTM | `src/runctm2M.py` |
| ETM | `src/runETM2M.py` |
| LDA | `src/runlda2M.py` |

## 9. Evaluation

Main evaluation scripts:

- `src/eval_all.py`
- `src/eval_all_summary.py`
- `src/eval2M.py`
- `src/eval2Mbertopic.py`
- `src/eval_bertopic_noise.py`

`src/eval_all.py` computes NPMI coherence and topic diversity for the sample-based topic files and saves:

- `evaluation_metrics_all_runs.csv`

`src/eval_all_summary.py` merges evaluation output with runtime information and creates summary tables such as:

- `evaluation_metrics_runtime_mean_std_summary.csv`

`src/eval2M.py` and `src/eval2Mbertopic.py` evaluate the full-corpus/2M experiments.

`src/eval_bertopic_noise.py` analyzes BERTopic assignment outputs and saves:

- `bertopic_noise_detail_all_runs.csv`
- `bertopic_noise_mean_std_summary.csv`

## 10. Chart Generation

Main chart scripts:

- `src/genchartsum.py`
- `src/genchartsfullruns.py`

`src/genchartsum.py` rebuilds the main summary charts from the aggregated evaluation summary CSV.

`src/genchartsfullruns.py` produces charts for the best full-corpus/2M model results.

These chart scripts were largely composed with AI assistance.

## Notes

- Some original script/file names were inconsistent during the project. For example, earlier notes used `datapasing.py`, but the script in this repository is `src/dataparsing.py`.
- Some files named `BoW` are preprocessing scripts for data that was later used to create bag-of-words representations.
- Several large intermediate files are not included in this cleaned repository. See `README.md` for the missing/large-file list.
