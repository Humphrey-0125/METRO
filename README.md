# METRO: Strategy Induction for Non-collaborative Dialogues

**ACL 2026**

Official implementation of:
> "METRO: Towards Strategy Induction from Expert Dialogue Transcripts for Non-collaborative Dialogues"

📄 Paper: [arxiv](https://arxiv.org/abs/2604.11427) &nbsp;|&nbsp; 🧪 Baselines: [non-cooperative-dialogue-baseline](https://github.com/Humphrey-0125/non-cooperative-dialogue-baseline)

---

## Overview

METRO learns structured dialogue strategies directly from expert transcripts and applies them online via retrieval-augmented inference.

<p align="center">
  <img src="assets/metro_architecture.png" alt="METRO Architecture" width="800"/>
  <br>
  <i>Figure 1: Offline Strategy Forest induction + online retrieval-augmented inference.</i>
</p>

**Supported tasks:**

| Dataset | Task | Agent role |
|---------|------|-----------|
| P4G | Persuasion for Good | Persuader |
| CB | CraigslistBargain | Buyer |
| ESC | Emotional Support Conversation | Supporter |

---

## Installation

```bash
git clone https://github.com/Humphrey-0125/METRO.git
cd METRO
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**API keys** — copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
# then edit .env:
# OPENAI_API_KEY=...
# SILICONFLOW_API_KEY=...   (or any OpenAI-compatible endpoint)
```

---

## Quick Start: run METRO in one command

Once the [offline Strategy Forest](#offline-build-the-strategy-forest) has been built, running the full METRO evaluation is a single command:

```bash
# Edit DATASET at the top of run.sh (p4g | cb | esc), then:
bash run.sh
```

`run.sh` automatically:
- splits the evaluation set into batches and launches them **in parallel**;
- writes all results to a single timestamped JSON file with **incremental saving** (crash-safe);
- skips already-completed dialogues on **resume**.

Default parameters per dataset:

| Dataset | Cluster config | Samples | Batch size |
|---------|---------------|---------|-----------|
| P4G | `kmeans_k150` | 200 | 20 |
| CB  | `kmeans_k80`  | 200 | 15 |
| ESC | `kmeans_k150` | 120 | 40 |

Results are saved to `outputs/{P4G,CB,ESC}/evaluate/history/<cluster_config>/results_<timestamp>.json`.

---

## Pipeline

METRO has two phases: **offline** (build the Strategy Forest once) and **online** (retrieval-augmented inference, via `run.sh`).

### Offline: build the Strategy Forest

```
dialogue transcripts
      │
      ▼
  1. State     — embed dialogue history into vectors
      │
      ▼
  2. Cluster   — group similar dialogue states (KMeans / OPTICS / HDBSCAN)
      │
      ▼
  3. MCT       — induce strategy trees per cluster (Monte Carlo Tree)
      │
      ▼
  4. Split     — assign micro-principles to each cluster
```

Run all steps with one command:

```bash
python src/data_process/pipeline_dataset.py \
    --dataset p4g \
    --steps all \
    --k 150 \
    --normalize
```

Or run individual steps:

```bash
# Step 1: embed dialogue history
python src/data_process/1state.py --task p4g

# Step 2: cluster embeddings
python src/data_process/2cluster.py \
    --task P4G --state_type history \
    --cluster-method kmeans --k 150 --normalize

# Step 3: induce strategy trees
python src/data_process/3MCT_topk.py \
    --dataset p4g \
    --states  outputs/P4G/embedding/history/history_embeddings_train_test.json \
    --labels  outputs/P4G/cluster/history/kmeans_k150/labels.jsonl \
    --dialogs data/P4G/result.json \
    --outdir  outputs/P4G/cluster/history/kmeans_k150

# Step 4: assign micro-principles to clusters
python src/data_process/4split_principles_by_cluster.py \
    --principles outputs/P4G/principles/micro_principles.jsonl \
    --labels     outputs/P4G/cluster/history/kmeans_k150/labels.jsonl
```

**Key pipeline options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | — | `p4g` / `cb` / `esc` |
| `--k` | 100 | KMeans cluster count |
| `--normalize` | off | L2-normalize embeddings before clustering |
| `--cluster-method` | `kmeans` | `kmeans` / `optics` / `hdbscan` |
| `--topk` | 5 | Strategy chains kept per cluster |
| `--beam` | 20 | MCT beam width |
| `--max-depth` | 9 | Maximum strategy tree depth |

---

### Online: run METRO (via `run.sh`)

`run.sh` is the recommended way to run METRO. It handles parallelism, output paths, and resume automatically. Just set `DATASET=` at the top and run:

```bash
bash run.sh
```

To invoke the runtime scripts directly (e.g., for debugging):

```bash
# P4G — single batch
python src/runtime/metro_p4g.py \
    --cluster_config kmeans_k150 \
    --max_samples 200 \
    --start_index 0 --end_index 49 \
    --output_file outputs/P4G/evaluate/history/kmeans_k150/results.json \
    --verbose

# CB
python src/runtime/metro_cb.py \
    --cluster_config kmeans_k80 \
    --dev_path data/CB/dev.json \
    --persona_path outputs/P4G/personas/personas_eval.jsonl \
    --max_samples 200

# ESC
python src/runtime/metro_esc.py \
    --cluster_config kmeans_k150 \
    --dev_path data/ESC/test.json \
    --max_samples 120
```

---

## Evaluation

```bash
python src/evaluate/metrics.py <results_file> --dataset p4g
python src/evaluate/metrics.py <results_file> --dataset cb
python src/evaluate/metrics.py <results_file> --dataset esc
```


