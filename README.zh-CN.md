# 假新闻检测 — ISOT（二分类文本分类）

[English](README.md) | **简体中文**

根据新闻的 **标题 + 正文**，判断它是 **虚假新闻**（`label=1`）还是 **真实新闻**（`label=0`）。
数据集为 ISOT Fake and Real News
（[Kaggle: clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)）。

本仓库按照 [`docs/coding_plan.md`](docs/coding_plan.md) 分阶段实现。当前进度：

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 仓库骨架、配置系统、测试、CI | ✅ |
| 1 | 数据下载 + EDA 报告 | ✅ |
| 2 | 清洗、去重/冲突审计、70/10/20 划分 | ✅ |
| 3 | TF-IDF + 逻辑回归 / 线性 SVM | ✅ |
| 4 | shortcut 格式特征 baseline | ⬜ |
| 5 | Bi-LSTM | ⬜ |
| 6 | DistilBERT | ⬜ |
| 7 | 评估、bootstrap 置信区间、错误分析 | ⬜ |
| 8 | 来源线索遮蔽实验 | ⬜ |
| 9 | 图表 + 文档 | ⬜ |

## 数据

- `True.csv` — 路透社（Reuters）文章，`label = 0`（真实）
- `Fake.csv` — 来自被标记网站的文章，`label = 1`（虚假）
- 许可：**数据 CC BY-NC-SA 4.0**；代码 MIT。CSV 文件**不**包含在本仓库中。

## 快速开始

需要 Python **3.10+**（在 Windows 上使用 3.11 测试）。在仓库根目录执行：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

神经模型阶段（Bi-LSTM / DistilBERT）还需要 torch + transformers：

```bash
pip install -e ".[dev,neural]"
```

### 1. 获取数据

在 Kaggle 页面（*Settings* → *API* → *Create New Token*）创建 API Token，并导出到环境变量（**不要提交到 git**）：

```powershell
$env:KAGGLE_API_TOKEN = "KGAT_xxx"        # PowerShell
```

```bash
export KAGGLE_API_TOKEN="KGAT_xxx"        # bash
```

```bash
python -m fakenews.data.download --config configs/base.yaml
```

`download.py` 会校验两个 CSV、写入 `data/raw/CHECKSUMS.txt`，并优先使用 Kaggle API。
如果所在网络阻断了到 Google Storage 的 TLS 连接（某些网络下常见），脚本会用 HTTP
重试同一个签名 URL 并打印警告。手动下载的回退方式：

```bash
python -m fakenews.data.download --archive path/to/archive.zip
```

### 2. EDA → 清洗 → 划分

```bash
python -m fakenews.data.eda    --config configs/base.yaml
python -m fakenews.data.clean  --config configs/base.yaml --csv
python -m fakenews.data.split  --config configs/base.yaml
```

输出：

```text
results/eda_report.md
results/figures/fig_eda_overview.{png,pdf}
data/processed/articles_clean.parquet   # 训练用主格式
data/processed/articles_clean.csv       # 便于人工查看的副本
data/processed/clean_log.json           # 清洗计数
data/processed/{train,val,test}.parquet # 70 / 10 / 20，分层，seed 42
data/processed/split_manifest.json      # 行数、类别计数、sha256
```

清洗顺序：去 HTML/控制字符 → 压缩空白 → 删除正文 `<= 50` 字符的样本 →
删除在真假两类中都出现过的 `dup_key` 整组 → 删除精确重复（`keep="first"`）。
URL 和 `(Reuters)` dateline 在这一步**保留**；遮蔽它们是单独的 Stage 8 实验。

## 当前数据快照

2026-10-23 运行，`min_text_len=50`、`seed=42`：

| 项目 | 数值 |
|---|---|
| 原始行数 | 44,898（`True.csv` 21,417 + `Fake.csv` 23,481） |
| 空正文 | 631 |
| 删除：正文 ≤ 50 字符 | 844 |
| 删除：跨标签冲突 | 0 |
| 删除：精确重复 | 5,551 |
| **清洗后行数** | **38,503**（21,194 real / 17,309 fake） |
| train / val / test | 26,951 / 3,851 / 7,701 |

`subject` 在两类之间完全不重叠（real：`politicsNews`、`worldnews`；
fake：另外六个取值），日期分布也不同，因此两者都不进入任何特征矩阵。
清洗后的 CSV 约 293 MB，因为它同时保留了 `text_raw`、`text_clean` 和 `dup_key`；
训练请使用 Parquet 文件。

## 经典 baseline（阶段 3）

`python -m fakenews.train.classical --config configs/base.yaml --model {lr,svm} --final`

单 seed（42），TF-IDF 只在 train 上 fit，`alpha` 和阈值只在 val 上选，
在 train+val 上 refit 后只读取一次 test。F1（正类 = fake）：

| 模型 | title_only | body_only | concat | fusion | alpha |
|---|---|---|---|---|---|
| LR val | 0.9387 | 0.9916 | 0.9939 | **0.9951** | 0.40 |
| LR test | 0.9433 | 0.9940 | 0.9960 | **0.9962** | 0.40 |
| SVM val | 0.9396 | 0.9939 | 0.9959 | **0.9960** | 0.50 |
| SVM test | 0.9422 | 0.9954 | 0.9963 | **0.9970** | 0.50 |

test F1 使用 val 上为同一条件选出的阈值。逐样本概率在 `results/predictions/`
（已被 git 忽略）；汇总指标在 `results/metrics.csv`，alpha 曲线在 `results/fusion.json`。
title-only 明显弱于 body-only，但二者融合仍能带来小幅提升。

完整结果、统计检验和解释性警告见
[`results/RESULTS.md`](results/RESULTS.md)。简述：body 占主导，fusion 与 early concat
在统计上无法区分；而且一条 `(Reuters)` 规则就已经达到 F1 0.994 —— 这个 benchmark
已经被来源线索饱和，因此这些数字不能作为真实场景事实核查能力的证据。

## 配置系统

`configs/base.yaml` 是唯一配置源；各模型配置只写覆盖项，加载时递归合并。
所有 CLI 都支持：

```bash
--config configs/base.yaml --override data.min_text_len=80
```

## 测试与 lint

```bash
pytest -q
ruff check .
```

CI（`.github/workflows/ci.yml`）在 Ubuntu 上运行 `ruff` 和 `pytest -m "not slow"`。

## 硬件要求

- TF-IDF + LR/SVM：笔记本 CPU，分钟级。
- Bi-LSTM：CPU/GPU，每个 epoch 约 20–60 分钟。
- DistilBERT：Colab T4，每个 seed 约 20–40 分钟。

## 已知限制

- 标签来自**来源**，不是逐篇事实核查；模型可能学到来源风格而非真实性。
- `subject` 和 `date` 只用于审计，绝不能作为特征。
- 单一数据集、仅英文；神经模型会对正文做截断。

## 许可证

代码：MIT（见 `LICENSE`）。数据：CC BY-NC-SA 4.0 —— 见 `data/README.md` 和
`docs/data_card.md`。
