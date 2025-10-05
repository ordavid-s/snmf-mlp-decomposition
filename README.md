# Decomposing MLP Activations into Interpretable Features via Semi-Nonnegative Matrix Factorization

This repository contains the official code for the paper: **“Decomposing MLP Activations into Interpretable Features via Semi-Nonnegative Matrix Factorization”** (2025).

---

## Quick Links

1. [Setup](#setup)
2. [Data](#data)
3. [Tutorials](#tutorials)
4. [Experiments](#experiments)

---

## Setup

### Install

```bash
git clone https://github.com/ordavid-s/snmf-mlp-decomposition.git
cd snmf-mlp-decomposition
pip install -r requirements.txt
```

### Environment

To run experiments, configure the following in a `.env` file:

* `OPENAI_API_KEY`: a functioning API key for OpenAI (used for evaluation)
* `GOOGLE_API_KEY`: a functioning API key for Gemini (used for DiffMean sentence generation)

**Example:**

```bash
OPENAI_API_KEY=sk-....
GOOGLE_API_KEY=ABC....
```

---

## Data

This project’s data directory contains datasets compatible with the repo’s dataset abstraction:

1. **final_dataset_20_concepts.json** – Dataset of randomly sampled concepts used to train SNMF in the paper
2. **hier_concepts.json** – Dataset constructed with ConceptNet containing hierarchical concepts
3. **gpt2_mlp_features.json** – Randomly sampled features from GPT-2 SAE used in the research
4. **gemma_mlp_features.json** – Randomly sampled features from Gemmascope SAE used in the research
5. **llama_mlp_features.json** – Randomly sampled features from Llamascope SAE used in the research
6. **languages.json** – Small dataset helpful for experimenting on a very small scale

---

## Tutorials

1. **snmf_tutorial.ipynb** – Train SNMF end-to-end: process data, factorize, and analyze discovered factors.
2. **hierarchial_nmf_tutorial.ipynb** – Train recursive SNMF and visualize concept trees identified in the MLP.

---

## Experiments

Change into the `experiments` directory and run the desired experiment:

1. **Concept Detection**

   * SNMF: `./run_snmf_concept_detection.sh`
   * SAE: `./run_sae_concept_detection.sh`

2. **Concept Steering**

   * SNMF: `./run_snmf_steering.sh`
   * SAE: `./run_sae_steering.sh`
   * DiffMeans: `./run_diffmean_steering.sh`

     * Must run SNMF and SAE first
     * Must update paths in `run_diffmean_steering.sh` to utilize generated SAE or SNMF data

3. **Qualitative Analysis**
   Use the provided hierarchical features tutorial.

---

**Questions or issues?** Please open an issue and we will do our best to respond in a timely manner.
