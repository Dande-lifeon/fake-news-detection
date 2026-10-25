# ISOT 假新闻检测

[English](README.md)

这个仓库在 ISOT Fake and Real News 数据集上训练并比较四个模型：TF-IDF 配逻辑回归、TF-IDF 配线性 SVM、Bi-LSTM，以及微调后的 DistilBERT。标题和正文分开编码后再融合，同时保留直接拼接的基线。

真正值得说的不是最高 F1。只用一条 Reuters dateline 规则就能到 0.994，18 个格式特征能到 0.995，完整 TF-IDF 模型是 0.996。ISOT 的标签几乎可以由来源和排版线索直接解出，所以这里的高分不能当作事实核查能力的证据。

数据集：<https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset>

Milestone 2 交付物：[report PDF](reports/Milestone_Report.pdf) ·
[slides PDF](reports/Milestone_Slides.pdf)。

## 状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 包结构、配置、测试、CI | ✅ 完成 |
| 1 | 数据下载与 EDA | ✅ 完成 |
| 2 | 清洗、去重、70/10/20 划分 | ✅ 完成 |
| 3 | TF-IDF 逻辑回归与线性 SVM | ✅ 完成 |
| 4 | 格式特征 shortcut 基线 | ✅ 完成 |
| 5 | Bi-LSTM | 🟡 代码就绪，未训练 |
| 6 | DistilBERT | 🟡 代码就绪，未训练 |
| 7 | bootstrap、McNemar、校准、错误分析 | 🟡 部分完成 |
| 8 | 来源线索遮蔽 | ✅ 完成 |
| 9 | 结果图与文档 | 🟡 进行中 |
| 10 | 时间划分、LIAR、近重复划分 | ⏸ 暂缓 |

目前可以直接跑的是阶段 0-4、阶段 8、指标汇总和结果图。神经模型代码已经写好，但还没有训练结果：本机没有安装 torch/transformers，且 PyPI 不可达。

## 目录结构

```text
configs/            默认配置和模型覆盖配置
data/               原始与处理后数据（git 忽略）
docs/               数据卡、实验协议、开发计划、环境记录
environment.yml     conda 基础环境
.python-version     参考 Python 版本
fakenews/
  data/             下载、EDA、清洗、划分
  features/         TF-IDF、shortcut 特征、遮蔽
  models/           经典模型、融合层
  train/            经典模型与 shortcut 训练入口
  eval/             指标、bootstrap、指标汇总
  experiments/      来源线索遮蔽
scripts/            run_all.sh、smoke.sh、env_report.py
tests/              单元与集成测试
results/            纳入版本库的报告与指标；逐样本预测被忽略
```

## 环境

参考版本是 Python 3.11。`requirements.txt` 是在 3.11 上生成的锁定版本，项目要求 3.11 或更高。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

如果要用锁定版本：

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

Conda 用户可以从基础环境开始：

```bash
conda env create -f environment.yml
conda activate fakenews
pip install -e ".[dev]"
```

报告所用环境的具体版本在 [`docs/environment.md`](docs/environment.md)。重新生成：

```bash
python scripts/env_report.py --output docs/environment.md
```

`Makefile` 是可选的。Windows 上没有 `make` 时，直接运行下面的 `python -m ...` 命令。

神经模型需要额外的包：

```bash
pip install -e ".[dev,neural]"
```

## 数据

两个 CSV 不放进仓库。

- `True.csv`：Reuters 文章，`label=0`
- `Fake.csv`：来自被标记站点的文章，`label=1`

用 Kaggle API token 下载：

```powershell
$env:KAGGLE_API_TOKEN = "KGAT_xxx"
```

```bash
export KAGGLE_API_TOKEN="KGAT_xxx"
python -m fakenews.data.download --config configs/base.yaml
```

如果下载被网络挡住，从 Kaggle 下载 `archive.zip` 后解压：

```bash
python -m fakenews.data.download --archive path/to/archive.zip
```

下载器会检查两个文件，把 SHA-256 写进 `data/raw/CHECKSUMS.txt`，并生成 `data/raw/raw_summary.json`。

## 运行流程

### 1. EDA、清洗、划分

```bash
python -m fakenews.data.eda   --config configs/base.yaml
python -m fakenews.data.clean --config configs/base.yaml --csv
python -m fakenews.data.split --config configs/base.yaml
```

输出：

```text
results/eda_report.md
results/figures/fig_eda_overview.{png,pdf}
data/processed/articles_clean.parquet
data/processed/articles_clean.csv
data/processed/clean_log.json
data/processed/{train,val,test}.parquet
data/processed/split_manifest.json
```

清洗顺序是：去 HTML 和控制字符、压缩空白、丢掉 50 字符及以下的正文、删掉同一个 `dup_key` 同时出现两种标签的所有行、精确重复保留第一条。URL 和 Reuters dateline 在这一步保留，删除它们是单独的遮蔽实验。

当前快照使用 `min_text_len=50`、`seed=42`：

| 项目 | 数值 |
|---|---:|
| 原始行数 | 44,898（`True.csv` 21,417，`Fake.csv` 23,481） |
| 空正文 | 631 |
| 删除：正文 <= 50 字符 | 844 |
| 删除：跨标签冲突 | 0 |
| 删除：精确重复 | 5,551 |
| 清洗后 | 38,503（21,194 真，17,309 假） |
| train / val / test | 26,951 / 3,851 / 7,701 |

`subject` 在两类之间完全不重叠，`date` 分布也不同，所以两者都不进入特征。CSV 副本约 293 MB，因为保留了 `text_raw`、`text_clean` 和 `dup_key`；训练读 Parquet。

631 条空正文是 844 条“正文 <= 50 字符”的子集，不重复相减：
44,898 - 844 - 5,551 = 38,503。

### 2. 经典基线

```bash
python -m fakenews.train.classical --config configs/base.yaml --model lr  --final
python -m fakenews.train.classical --config configs/base.yaml --model svm --final
```

两个模型用同一份划分。TF-IDF 只在 train 上 fit；`C` 在 train 上做 5 折 CV；`alpha` 和阈值在 val 上选；最后在 train+val 上重训，再对 test 评估一次。

fake 类的 test F1：

| 模型 | title_only | body_only | concat | fusion | alpha |
|---|---:|---:|---:|---:|---:|
| LR | 0.9429 | 0.9929 | 0.9962 | 0.9962 | 0.40 |
| SVM | 0.9425 | 0.9957 | 0.9975 | 0.9970 | 0.50 |

主表用 0.5 阈值。按 val 调阈值最多让 test F1 变化 0.0012，所以 0.5 更稳。`alpha` 是概率混合权重，不代表两个分支的重要性。

逐样本概率写在 `results/predictions/`（git 忽略）。汇总表是 `results/all_metrics.csv`，逐条件 95% CI 在 `results/cis.csv`，四张结果图在 `results/figures/`，FP/FN 表在 `results/error_examples.md`。

### 3. shortcut 与遮蔽实验

```bash
python -m fakenews.train.shortcut --config configs/shortcut.yaml --final
python -m fakenews.experiments.source_cue --config configs/base.yaml --model lr
python -m fakenews.experiments.source_cue --config configs/base.yaml --model svm
python -m fakenews.eval.evaluate --config configs/base.yaml
```

`shortcut` 只用 18 个表面特征训练逻辑回归：dateline、URL、Getty 与 `featured image` 署名、大小写、标点和长度。它不使用正文词义，正文只贡献表层统计。

`source_cue` 对清洗后的正文做遮蔽，用同一套 LR/SVM 配置在遮蔽后的 train 上重训，再把遮蔽后的 test 预测和原始预测按同一条样本对齐。`C`、`alpha`、阈值都固定，所以唯一变化就是被删掉的线索。

## 结果

完整台账、置信区间、McNemar 检验、校准和错误分析在 [`results/RESULTS.md`](results/RESULTS.md)。
第二次 milestone 的 1 页 report 和 slides 素材在 [`results/MILESTONE.md`](results/MILESTONE.md)。

简要结论：

- 正文远强于标题。title-only 的 F1 约 0.94，body-only 约 0.99。
- fusion 和 early concat 在统计上没有区别。
- 单条 `(Reuters)` 规则到 test F1 0.9943。
- 18 个格式特征到 0.9953。
- 全文 LR 和 SVM 到 0.9962 和 0.9970。
- 遮掉 dateline、URL 和媒体署名后，fusion 仍有 0.9903-0.9944。

最后一条是重点。删掉最明显的线索并不能删掉捷径，因为语料里还有多种互相冗余的来源和话题信号。

### 原始样本

前 100 行就能看出模式。`True.csv` 里 91/100 的正文含 `(Reuters)`，0 条含 URL，0 条含 `getty`。`Fake.csv` 里 0 条含 `(Reuters)`，23 条含 URL，79 条含 `getty`，58 条含 `featured image`。

| 标签 | 来源 | 片段 | 线索 |
|---|---|---|---|
| 真 | `True.csv` #0 | `WASHINGTON (Reuters) - The head of a conservative Republican faction...` | 城市 + Reuters dateline |
| 真 | `True.csv` #4 | `SEATTLE/WASHINGTON (Reuters) - President Donald Trump called on...` | 多城市 dateline |
| 假 | `Fake.csv` #0 | 标题以 `This is Disturbing` 结尾；正文以 `Photo by Andrew Burton/Getty Images.` 结尾 | 标题党 + Getty 署名 |
| 假 | `Fake.csv` #5 | 正文以 `Featured image via David McNew/Stringer/Getty Images` 结尾 | featured image + Getty 署名 |

## 复现

- 每个入口都调用 `seed_everything(42)`。
- 划分只生成一次，test 文件设为只读。
- `split_manifest.json` 记录大小、类别计数、SHA-256 和 pandas 版本。
- 每个 run 在 `results/logs/{run_id}/` 下保存 `config.yaml`、`environment.txt`、`git_commit.txt` 和 `train.log`。
- `eval.collect_metrics` 会汇总它找到的每一个预测文件。同一个 `(model, condition, seed)` 只保留一次 test 运行，否则台账会出现重复行。

实验协议写在 [`docs/protocol.md`](docs/protocol.md)。

## 配置

`configs/base.yaml` 是默认值。模型配置只写覆盖项，加载时递归合并。所有 CLI 都支持：

```bash
--config configs/base.yaml --override data.min_text_len=80
```

## 测试与 CI

```bash
pytest -q
ruff check .
```

CI 在 Ubuntu、Python 3.11 上跑 `ruff` 和 `pytest -m "not slow"`。单元测试覆盖清洗、划分、TF-IDF、融合、指标、遮蔽和 Kaggle CLI 回退，不需要 ISOT 原始 CSV。

## 硬件

经典流程在笔记本 CPU 上几分钟跑完。shortcut 和遮蔽实验是秒到几分钟。Bi-LSTM 和 DistilBERT 代码已经写好但还没有结果；训练需要 torch/transformers，本机未安装。计划是 Bi-LSTM 用笔记本 CPU，DistilBERT 用免费 Colab T4。

## 限制

- 标签来自发布来源，不是逐篇事实核查。模型可能学到来源风格，而不是真实性。
- `subject` 和 `date` 只用于审计，不能作为特征。
- 单一数据集，只有英文；神经模型会截断长正文。
- 数据集接近饱和，模型之间很小的 F1 差异，没有配对检验就不要当成真实差异。

## 许可

代码用 MIT（见 `LICENSE`）。数据用 CC BY-NC-SA 4.0，见 `data/README.md` 和 [`docs/data_card.md`](docs/data_card.md)。

## 常见问题

- `download` 失败：检查 `KAGGLE_API_TOKEN`，或者用 `--archive` / 手动把 CSV 放进 `data/raw/`。
- `split` 拒绝运行：它在保护已有的划分。只有确实想重新生成 train/val/test 时才加 `--force`。
- Windows 上没有 `make`：直接跑上面的 `python -m ...` 命令。
- `source_cue` 找不到预测：先跑经典基线。
