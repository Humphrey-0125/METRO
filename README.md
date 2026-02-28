# Persuasion Agent

Language / 语言: [中文](#中文) | [English](#english)

---

## 中文

### 1. 项目简介

本仓库用于三类说服对话任务（`P4G` / `CB` / `ESC`）的端到端实验，核心流程包括：

1. 对话状态表征（history embedding）
2. 状态聚类（KMeans / OPTICS / HDBSCAN）
3. 每个簇构建 MCT 策略树并导出 Top-K 策略链
4. 按簇切分原则（principles）
5. 运行对话仿真（history principle runtime）
6. 评测（CB 或 P4G/ESC 指标）

---

### 2. 目录结构

```text
persuasion-agent/
├─ data/                       # 原始数据（CB/ESC/P4G）
├─ src/
│  ├─ data_process/            # 预处理与聚类流水线
│  │  ├─ 1state.py
│  │  ├─ 2cluster.py
│  │  ├─ 3MCT_topk.py
│  │  ├─ 4split_principles_by_cluster.py
│  │  └─ pipeline_dataset.py   # 一键串联 1~4
│  ├─ runtime/                 # 对话仿真
│  │  ├─ history_principle.py      # P4G
│  │  ├─ history_principle_cb.py   # CB
│  │  └─ history_principle_esc.py  # ESC
│  ├─ evaluate/                # 评测
│  │  ├─ metrics.py            # 统一评测入口（cb/p4g/esc）
│  │  ├─ metrics_cb.py         # 兼容入口
│  │  ├─ metrics_p4g.py        # 兼容入口
│  │  └─ metrics_esc.py        # 兼容入口
│  └─ utils/                   # API / embedding / 检索等工具
├─ run_all.sh                  # 多进程批量运行入口
└─ requirements.txt
```

---

### 3. 环境安装

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
# source .venv/bin/activate

pip install -r requirements.txt
```

> 说明：`requirements.txt` 包含 `cupy`，如果本机无 CUDA 环境，安装可能失败。可按需改为 CPU 版本依赖。

---

### 4. 配置 API Key

建议使用环境变量，不要在代码中写死 key。

```bash
# Embedding
set SILICONFLOW_API_KEY=your_key
set SILICONFLOW_EMBED_URL=https://api.siliconflow.cn/v1/embeddings
set SILICONFLOW_EMBED_MODEL=BAAI/bge-large-en-v1.5

# LLM（按你实际使用的后端设置）
set OPENAI_API_KEY=your_key
set PLATO_API_KEY=your_key
```

Linux/Mac 把 `set` 改成 `export`。

---

### 5. 数据准备

默认读取以下路径（可在命令行覆盖）：

- `data/P4G/*.json`
- `data/CB/*.json`
- `data/ESC/*.json`

此外，runtime 默认需要 persona 文件：

- `outputs/P4G/personas/personas_eval.jsonl`

---

### 6. 一键跑预处理流水线（推荐）

使用统一脚本 `src/data_process/pipeline_dataset.py`：

```bash
# P4G：执行 state -> cluster -> mct -> split
python src/data_process/pipeline_dataset.py --dataset p4g --steps all --k 100 --normalize

# CB：只跑前三步
python src/data_process/pipeline_dataset.py --dataset cb --steps state,cluster,mct --k 80 --normalize

# ESC：只跑 mct + split（使用已有聚类结果）
python src/data_process/pipeline_dataset.py --dataset esc --steps mct,split \
  --labels outputs/ESC/cluster/history/kmeans_k150_llm/labels.jsonl
```

`--steps` 可选：`state,cluster,mct,split`。

---

### 7. 分步运行（手动）

```bash
# 1) state
python src/data_process/1state.py --task p4g

# 2) cluster
python src/data_process/2cluster.py --task P4G --state_type history --normalize --cluster-method kmeans --k 100

# 3) mct topk
python src/data_process/3MCT_topk.py \
  --dataset p4g \
  --states outputs/P4G/embedding/history/history_embeddings_train_test.json \
  --labels outputs/P4G/cluster/history/kmeans_k100_llm/labels.jsonl \
  --dialogs data/P4G/result.json \
  --outdir outputs/P4G/cluster/history/kmeans_k100_llm

# 4) split principles
python src/data_process/4split_principles_by_cluster.py \
  --principles outputs/P4G/principles/micro_principles.jsonl \
  --labels outputs/P4G/cluster/history/kmeans_k100_llm/labels.jsonl
```

---

### 8. 运行仿真

#### 8.1 使用统一批处理脚本（推荐）

```bash
# P4G
bash run_all.sh --dataset p4g --cluster-config kmeans_k150

# CB
bash run_all.sh --dataset cb --cluster-config kmeans_k80

# ESC
bash run_all.sh --dataset esc --cluster-config kmeans_k150
```

常用参数：

- `--start-index` / `--end-index`：分片跑
- `--batch-size`：单批规模
- `--max-parallel`：并发 Python 进程数
- `--ablation-mode`：`none | w/o_depth | w/o_breadth | w/o_both | w/o_expend`
- `--output-file`：结果输出路径

#### 8.2 直接运行 runtime 脚本

```bash
python src/runtime/history_principle.py --max_samples 200 --cluster_config kmeans_k150
python src/runtime/history_principle_cb.py --max_samples 200 --cluster_config kmeans_k80
python src/runtime/history_principle_esc.py --max_samples 200 --cluster_config kmeans_k150
```

---

### 9. 评测

统一评测入口：`src/evaluate/metrics.py`

```bash
# CB
python src/evaluate/metrics.py data/CB/result.json --dataset cb --max_turns 10

# P4G
python src/evaluate/metrics.py outputs/P4G/evaluate/history/kmeans_k150/none/test.json \
  --dataset p4g --threshold 0.6 --mapping_json "{\"A\":-1,\"B\":-0.5,\"C\":0.1,\"D\":1}"

# ESC
python src/evaluate/metrics.py outputs/ESC/evaluate/kmeans_k150/history/none/standard.json \
  --dataset esc --threshold 0.6 --strict_valid
```



---

## English

### 1. Overview

This repository provides an end-to-end pipeline for persuasion tasks on `P4G`, `CB`, and `ESC`, including:

1. State representation (history embedding)
2. State clustering (KMeans / OPTICS / HDBSCAN)
3. MCT strategy-tree building and Top-K chain export per cluster
4. Principle splitting by cluster
5. Runtime dialogue simulation
6. Evaluation metrics

---

### 2. Project Layout

```text
persuasion-agent/
├─ data/
├─ src/
│  ├─ data_process/
│  │  ├─ 1state.py
│  │  ├─ 2cluster.py
│  │  ├─ 3MCT_topk.py
│  │  ├─ 4split_principles_by_cluster.py
│  │  └─ pipeline_dataset.py
│  ├─ runtime/
│  │  ├─ history_principle.py
│  │  ├─ history_principle_cb.py
│  │  └─ history_principle_esc.py
│  ├─ evaluate/
│  │  ├─ metrics.py
│  │  ├─ metrics_cb.py
│  │  ├─ metrics_p4g.py
│  │  └─ metrics_esc.py
│  └─ utils/
├─ run_all.sh
└─ requirements.txt
```

---

### 3. Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
# source .venv/bin/activate

pip install -r requirements.txt
```

> Note: `requirements.txt` includes `cupy`. If CUDA is unavailable, installation may fail; use a CPU-compatible dependency set.

---

### 4. Environment Variables

Use environment variables instead of hardcoded keys.

```bash
# Embedding
export SILICONFLOW_API_KEY=your_key
export SILICONFLOW_EMBED_URL=https://api.siliconflow.cn/v1/embeddings
export SILICONFLOW_EMBED_MODEL=BAAI/bge-large-en-v1.5

# LLM backend (set what you actually use)
export OPENAI_API_KEY=your_key
export PLATO_API_KEY=your_key
```

For Windows CMD, replace `export` with `set`.

---

### 5. Data

Default inputs are under:

- `data/P4G/*.json`
- `data/CB/*.json`
- `data/ESC/*.json`

Runtime also expects persona file by default:

- `outputs/P4G/personas/personas_eval.jsonl`

---

### 6. One-command Data Pipeline (Recommended)

Use `src/data_process/pipeline_dataset.py`:

```bash
python src/data_process/pipeline_dataset.py --dataset p4g --steps all --k 100 --normalize
python src/data_process/pipeline_dataset.py --dataset cb --steps state,cluster,mct --k 80 --normalize
python src/data_process/pipeline_dataset.py --dataset esc --steps mct,split \
  --labels outputs/ESC/cluster/history/kmeans_k150_llm/labels.jsonl
```

`--steps` can be any combination of: `state,cluster,mct,split`.

---

### 7. Run Each Step Manually

```bash
python src/data_process/1state.py --task p4g

python src/data_process/2cluster.py --task P4G --state_type history --normalize --cluster-method kmeans --k 100

python src/data_process/3MCT_topk.py \
  --dataset p4g \
  --states outputs/P4G/embedding/history/history_embeddings_train_test.json \
  --labels outputs/P4G/cluster/history/kmeans_k100_llm/labels.jsonl \
  --dialogs data/P4G/result.json \
  --outdir outputs/P4G/cluster/history/kmeans_k100_llm

python src/data_process/4split_principles_by_cluster.py \
  --principles outputs/P4G/principles/micro_principles.jsonl \
  --labels outputs/P4G/cluster/history/kmeans_k100_llm/labels.jsonl
```

---

### 8. Runtime Simulation

#### 8.1 Using `run_all.sh` (recommended)

```bash
bash run_all.sh --dataset p4g --cluster-config kmeans_k150
bash run_all.sh --dataset cb --cluster-config kmeans_k80
bash run_all.sh --dataset esc --cluster-config kmeans_k150
```

Useful options:

- `--start-index`, `--end-index`
- `--batch-size`
- `--max-parallel`
- `--ablation-mode` (`none | w/o_depth | w/o_breadth | w/o_both | w/o_expend`)
- `--output-file`

#### 8.2 Run runtime scripts directly

```bash
python src/runtime/history_principle.py --max_samples 200 --cluster_config kmeans_k150
python src/runtime/history_principle_cb.py --max_samples 200 --cluster_config kmeans_k80
python src/runtime/history_principle_esc.py --max_samples 200 --cluster_config kmeans_k150
```

---

### 9. Evaluation

Unified entry: `src/evaluate/metrics.py`

```bash
python src/evaluate/metrics.py data/CB/result.json --dataset cb --max_turns 10

python src/evaluate/metrics.py outputs/P4G/evaluate/history/kmeans_k150/none/test.json \
  --dataset p4g --threshold 0.6 --mapping_json "{\"A\":-1,\"B\":-0.5,\"C\":0.1,\"D\":1}"

python src/evaluate/metrics.py outputs/ESC/evaluate/kmeans_k150/history/none/standard.json \
  --dataset esc --threshold 0.6 --strict_valid
```

Compatibility wrappers are still available:

- `python src/evaluate/metrics_cb.py --input ...`
- `python src/evaluate/metrics_p4g.py ...`
- `python src/evaluate/metrics_esc.py ...`

---