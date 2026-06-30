# DA233X Topic Modeling Riksarkivet

This repository contains the code and selected experiment artifacts for a master's thesis on topic modeling historical Swedish archival text from Riksarkivet.

The repository is organized as a compact handoff version: it keeps the scripts, evaluated model outputs, evaluation tables, final charts, and report material. Very large raw/intermediate files are documented but not included.

## Repository Structure

| Path | Contents | Purpose |
| --- | --- | --- |
| `src/` | Python scripts for preprocessing, model runs, evaluation, and chart generation | Main code used in the experimental pipeline |
| `data/` | Stopword files | Small input artifacts kept in the repository |
| `topics/` | Evaluated topic JSON files, topic-word matrices, and small assignment samples | Model outputs and examples used to inspect the evaluation workflow |
| `evaluation/` | Evaluation metrics, runtime summaries, noise summaries, and logs | Quantitative results used for analysis |
| `charts_used/` | Final summary charts and full-corpus/2M charts | Figures used in the report |
| `report/` | Thesis report source and report assets | Written thesis material |
| `filerunorder.md` | Detailed workflow notes | Step-by-step record of how the scripts relate to each other |

Large assignment and corpus artifacts were moved out of the Git repository to `../tobig/`, and `topics/samples/` contains small uncompressed examples for quick inspection.

See `filerunorder.md` for the detailed script-by-script workflow.

## Key Scripts

| Script | Role |
| --- | --- |
| `src/dataparsing.py` | Parses raw archival files |
| `src/preprocess_neural.py` | Creates the cleaned neural dataset |
| `src/neural_sample.py` | Creates the 150k sample |
| `src/bow_sample.py` | Creates sample-based bag-of-words artifacts |
| `src/preprocess_BoW_2M.py` | Creates full-corpus bag-of-words artifacts |
| `src/finetune.py` | Fine-tunes the sentence-transformer model |
| `src/gen_embeddings.py` | Generates sample-based embeddings |
| `src/gen_embeddings_full_light.py` | Generates full-corpus light embeddings |
| `src/gen_embeddings_full.py` | Generates full-corpus heavy embeddings |
| `src/runall.py` | Wrapper for main sample-based model runs |
| `src/runbertopicbase.py`, `src/runbertopicnoiseparams.py`, `src/runbertopicreassignment.py` | BERTopic sample-based runs |
| `src/runctm.py`, `src/runetm.py`, `src/runlda.py` | CTM, ETM, and LDA sample-based runs |
| `src/runbertopicbase2M.py`, `src/runctm2M.py`, `src/runETM2M.py`, `src/runlda2M.py` | Full-corpus/2M model runs |
| `src/eval_all.py` | Evaluates sample-based topic files |
| `src/eval2M.py`, `src/eval2Mbertopic.py` | Evaluates full-corpus/2M topic files |
| `src/eval_bertopic_noise.py` | Computes BERTopic noise summaries from assignment files |
| `src/eval_all_summary.py` | Aggregates evaluation and runtime results |
| `src/genchartsum.py` | Generates summary charts |
| `src/genchartsfullruns.py` | Generates full-corpus/2M charts |

## Charts

The final charts are in `charts_used/`:

- `charts_used/summarycharts/` contains the main summary charts.
- `charts_used/2M_best_models/` contains the full-corpus/2M best-model charts.


## Reproducibility Note

This repository is intended to document the thesis workflow and preserve the evaluated outputs. Full reruns may require access to large local data files that are not included in this Git repository.
