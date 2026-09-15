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

Requires Python 3.10+ with `openai`, `google-generativeai`, `pandas`, `numpy`, `matplotlib`, `seaborn` and `requests`. The validation notebook additionally needs `simpledorff`, `scipy` and `statsmodels`.

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
