# Hidden Nations, Hidden Bias
### Benchmarking Biases of Large Language Models Toward Stateless Nations

This repository contains the code and data supplementing a paper under peer review.

Large Language Models (LLMs) increasingly shape public perception of social groups, yet their treatment of *stateless nations*—groups lacking full state recognition—remains largely unexplored. Existing bias benchmarks rarely include these communities, and no multi‑LLM, multilingual analysis has been available to date.

This project provides the first **prompt‑based, cross‑model bias assessment** targeting stateless nations. It includes both **explicit** and **implicit** bias measurements across multiple LLMs, languages, and evaluation conditions.

---

## Overview

Our study focuses on six stateless nations:

- **Catalans**
- **Kurds**
- **Palestinians**
- **Rohingya**
- **Tibetans**
- **Uyghurs**

We evaluate four widely used LLMs across two complementary methods:

### Method A — Explicit Consent Analysis
Models respond to structured statements about stateless nations using controlled multiple‑choice formats. This measures *explicit value judgments* (e.g. empathy, cultural appreciation, political recognition).

### Method B — Implicit Bias Analysis
Models generate free‑form descriptions of stateless nations. These descriptions are then *evaluated by other LLMs* for distortions, stereotypes, or biased framing.

Running both methods in parallel reveals the gap between **explicit self‑reported attitudes** and **implicit patterns in generated text**.

---

## Repository structure

```
modells.py                        LLM API wrapper (ModelAPI) used by both methods
explicit_analysis/                Method A
  data/                           statement catalogue, raw and processed scores
    run_fragenkatalog.py          queries the models
    processed/                    score post-processing and run merging
  results/                        per-run statistics and heatmaps
  factors_analysis/               score contributions per factor
  radar_charts/                   radar chart generation
  missing_combinations/           audit of missing model responses
implicit_analysis/                Method B
  data/
    create_descriptions.py        generates the descriptions
    evaluate_descriptions.py      scores the descriptions with each model
    descriptions_1/, descriptions_2/   the two generation runs
    scoring_raw/, scoring_processed/   raw and processed scores
  results/                        per-run aggregates (run_1_1 … run_2_3)
  heatmaps/, factors_analysis/    visualisations
Data/                             combined datasets used by the notebook
Evaluation_Validation_V2.ipynb    human validation, Krippendorff's alpha, MBS
Human_validation_dataset.xlsx     human annotator ratings
figures/                          figures used in the paper
```

---

## Setup

Requires Python 3.10 or newer:

```bash
pip install -r requirements.txt
```

The analysis packages are pinned to the versions the results were produced with. The two API clients (`openai`, `google-generativeai`) are intentionally left unpinned — see the comments in `requirements.txt`.

All API keys are read from environment variables; none are stored in this repository:

```bash
export OPENAI_API_KEY=...       # ChatGPT-4o
export GEMINI_API_KEY=...       # Gemini-1.5-flash
export DASHSCOPE_API_KEY=...    # Qwen-Plus
export DEEPSEEK_API_KEY=...     # DeepSeek-V3
```

**All scripts use paths relative to the repository root, so run them from there:**

```bash
python explicit_analysis/data/run_fragenkatalog.py
python implicit_analysis/data/evaluate_descriptions.py
```

---

## Models and API configuration

| Paper | API identifier | Provider |
|---|---|---|
| ChatGPT-4o | `gpt-4o` | OpenAI |
| Gemini-1.5-flash | `gemini-1.5-flash` | Google |
| Qwen-Plus | `qwen-plus` | Alibaba DashScope |
| DeepSeek-V3 | `deepseek-chat` | DeepSeek |

No sampling parameters are set in `modells.py`, so the respective provider defaults apply. All models were queried between **1 and 18 March 2025**.

---

## Execution order

All commands are run from the repository root. Steps 1 of each method issue paid API calls; every later step works purely on the CSV files already contained in this repository, so the analysis can be reproduced without any API access.

> **Do not run step 1 unless you intend to collect new data.** `run_fragenkatalog.py`, `create_descriptions.py` and `evaluate_descriptions.py` write to fixed filenames and overwrite — `evaluate_descriptions.py` even deletes its target file before starting. Running them replaces the published raw data with a fresh sample from the models, and the numbers in all downstream results and figures will change. To reproduce the published results, start at step 2.

**Note:** most scripts carry the run identifier as a module-level constant near the top of the file (for example `run = 'run_2_2'`, or the output filename in `run_fragenkatalog.py`). Reproducing all runs means editing that constant and re-running the script once per run.

### Method A — explicit consent analysis

```bash
python explicit_analysis/data/run_fragenkatalog.py                    # 1. query the models -> data/raw/scoring_run_N.csv
python explicit_analysis/data/processed/update_scoring.py             # 2. post-process    -> data/processed/scoring_processed_run_N.csv
python explicit_analysis/data/processed/merge_runs.py                 # 3. merge runs 1-3  -> data/processed/scoring_processed_combined.csv
python explicit_analysis/results/compute_explicit_statistics_overall.py   # 4. per-run statistics -> results/results_run_N/
python explicit_analysis/results/generate_axis_stats.py               # 5. per-axis heatmaps
python explicit_analysis/results/results_combined/combine_runs.py     # 6. combine runs    -> results/results_combined/
python explicit_analysis/results/results_combined/heatmaps_combined/overall_heatmap.py   # 7. overall heatmap
python explicit_analysis/radar_charts/create_radar_chart_overall.py   # 8. radar charts
python explicit_analysis/factors_analysis/score_contribution.py       # 9. factor contributions
python explicit_analysis/factors_analysis/formulation/score_contribution_per_formulation.py
```

Optional audit of missing model responses:

```bash
python explicit_analysis/missing_combinations/find_missing_combinations.py
python explicit_analysis/missing_combinations/create_charts_missing_combination.py
```

### Method B — implicit bias analysis

```bash
python implicit_analysis/data/create_descriptions.py                  # 1. generate descriptions -> data/descriptions_2/
python implicit_analysis/data/evaluate_descriptions.py                # 2. score them            -> data/scoring_raw/scoring_raw_run_X_Y.csv
python implicit_analysis/data/scoring_processed/remove_dots.py        # 3. post-process          -> data/scoring_processed/
python implicit_analysis/data/scoring_processed/merge_scores_processed.py   # 4. merge the six runs
python implicit_analysis/results/compute_model_group.py               # 5. aggregate per group  -> results/combined/
python implicit_analysis/results/compute_model_model.py               #    aggregate per judge
python implicit_analysis/heatmaps/generate_heatmap_model_group.py     # 6. heatmaps
python implicit_analysis/heatmaps/generate_heatmap_model_model.py
python implicit_analysis/factors_analysis/score_contributions.py      # 7. factor contributions
python implicit_analysis/results/combined/combine_model_model.py      # 8. combine per-run results
```

### Human validation and paper figures

`Evaluation_Validation_V2.ipynb` computes the inter-rater agreement (Krippendorff's alpha), the Mean Bias Score and the figures of the paper. It reads three files from `Data/` — the human ratings, `all_descriptions.csv` and `scoring_processed_combined.csv` — and writes `bias_consensus_table.csv` together with the `figure4_*` files into the working directory.

---

## Reproducibility

The full experimental pipeline is designed for repeatability:

- **Method A:** the complete set of prompts was run three times with an identical setup (`data/raw/scoring_run_1..3.csv`).
- **Method B:** descriptions were generated in two independent runs (`descriptions_1/`, `descriptions_2/`), each scored three times independently by every judge model — six evaluations per description and choice set (`scoring_raw/scoring_raw_run_1_1..2_3.csv`).
- Multilingual prompts (German/English), multiple paraphrases, and three answer-set variants per method.
- Evaluation by all four participating LLMs.

This design enables **longitudinal monitoring** of bias across future LLM versions.

### Known limitations

- `implicit_analysis/results/combined/combine_model_group.py` and `implicit_analysis/results/generate_visualisations.py` still expect the run directories `run_1`, `run_2`, `run_3`; the implicit results use the naming `run_1_1` … `run_2_3`.
- Some scripts write into output directories that must already exist.
- `Evaluation_Validation_V2.ipynb` resolves its input files relative to the working directory (`base_path = Path('.')`), so it has to be started from the repository root.

---

## Citation

Will be added once the paper is published.

---

## Contributing

Contributions are welcome, especially:

- new groups or minorities to monitor
- additional languages
- alternative bias dimensions
- improved evaluation schemes

Please open an issue or pull request.

---

## License

This repository is released under the MIT License unless otherwise stated.

---

## Contact

For questions or collaborations:
**Beat Tödtli**, OST – Eastern Switzerland University of Applied Sciences
📧 beat.toedtli@ost.ch
