# Hidden Nations, Hidden Bias
### Benchmarking Biases of Large Language Models Toward Stateless Nations

This repository contains the code and data supplementing a paper under peer review


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

### **Method A — Explicit Consent Analysis**
Models respond to structured statements about stateless nations using controlled multiple‑choice formats.  
This measures *explicit value judgments* (e.g., empathy, cultural appreciation, political recognition).

### **Method B — Implicit Bias Analysis**
Models generate free‑form descriptions of stateless nations.  
These descriptions are then *evaluated by other LLMs* for distortions, stereotypes, or biased framing.

Running both methods in parallel reveals the gap between **explicit self‑reported attitudes** and **implicit patterns in generated text**.

---

## What’s in this repository?

- **/explizite_Analyse/** — code and data for Method A  
- **/factors_analysis/** — code and heatmap generation for Method B  
- **/data/** — prompts, answer sets, and evaluation templates  
- **Notebooks and scripts** to:
  - generate prompts  
  - submit queries to LLM APIs  
  - evaluate and aggregate results  
  - compute statistical summaries and produce visualizations  

All models were queried via official APIs between **February–March 2025** using default parameters.

---

## Reproducibility

The full experimental pipeline is designed for repeatability:
- Multi‑run sampling to reduce stochastic variance  
- Multilingual prompts (English/German)  
- Multiple paraphrases and Likert-scale variants  
- Evaluation by all participating LLMs  

This design enables **longitudinal monitoring** of bias across future LLM versions.

---

## Citation

will be added later.

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
