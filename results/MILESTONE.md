# Week 9 milestone summary

> 用途：作为第二次 milestone 的 1 页 report 和 5 分钟 slides 的主要素材。
> 数据来源：`results/all_metrics.csv`、`results/fusion.json`、`results/shortcut.json`、
> `results/source_cue.json`、`results/RESULTS.md`、`results/eda_report.md`。
> 核对日期：2026-10-25。文中数字均可在上述文件中复查。

---

## 0. 一句话结论

项目已经完成数据管线、两个经典基线、18 特征 shortcut 基线和 LR/SVM 的来源线索遮蔽实验。所有模型在 ISOT 上都接近 0.99，遮掉 Reuters dateline、URL 和媒体署名后仍有 0.99。当前阶段的主要产出不是“更高的 F1”，而是一个可复现的 benchmark audit：ISOT 的高分主要来自来源和排版线索，单点遮蔽不足以消除这些线索。

---

## 1. 与 milestone 要求的对应

| 要求 | 当前状态 | 证据 |
|---|---|---|
| Describe the data in detail | 已有完整数字、泄漏审计和划分 | `results/eda_report.md`、`clean_log.json`、`split_manifest.json` |
| Explain which models you may use | LR/SVM 已完成；Bi-LSTM/DistilBERT 已设计但暂缓 | `README.md` 状态表、`docs/coding_plan.md` |
| Report current progress | 两个基线 + shortcut + 遮蔽实验 + 统计检查 | `results/RESULTS.md`、`all_metrics.csv` |
| Discuss problems | 已有 3 个明确瓶颈 | 本文第 7 节 |
| Lay out next plan | 见第 6 节 | 与当前结果直接对应 |
| 1 页 report，未迟交 | 尚未撰写 | 本文第 8 节给出提纲 |
| 代码仓库 | 已建并推送 | GitHub `Dande-lifeon/fake-news-detection` |

---

## 2. 当前阶段

### 2.1 已完成

- 数据下载、校验、EDA、清洗、去重、70/10/20 分层划分和 manifest。
- Stage 3：TF-IDF + LR、TF-IDF + Linear SVM，四个条件 `title_only / body_only / concat / fusion`。
- Stage 4：18 个格式特征 + LR 的 shortcut 基线；单条 `(Reuters)` 规则。
- Stage 8：LR/SVM 的两种遮蔽强度 `dateline_url` 和 `format`，含逐样本预测和 top 特征审计。
- 统计检查：bootstrap CI、McNemar、校准（Brier/ECE）、错误计数。
- 工程：50 个测试通过，ruff 通过，CI、环境记录、README 和运行日志齐全。

### 2.2 部分完成

- Stage 7 的 bootstrap、McNemar、校准和错误分析已经做在 `RESULTS.md` 与分析脚本里，但还没有统一的 `evaluate.py` CLI。
- 结果图只有 EDA 总览，还没有模型对比图、混淆矩阵、ΔF1 图和 alpha 曲线。
- 错误分析只有类别汇总，还没有可用于 slides 的具体 FP/FN 样本表。

### 2.3 未完成

- Bi-LSTM、DistilBERT（Stage 5/6）。
- 时间划分、LIAR 跨数据集、近重复分组划分（Stage 10）。
- 最终 report、slides、结果图。

---

## 3. 数据

### 3.1 口径校正

此前材料里出现过 `12,600`（旧 proposal）和 `23,502`（课件/部分文档）两个数字。当前统一以 `pandas.read_csv` 实际加载为准：

| 项目 | 数值 |
|---|---:|
| 原始行数 | 44,898 |
| `True.csv` | 21,417 |
| `Fake.csv` | 23,481 |
| 空正文 | 631 |
| 删除：正文 <= 50 字符 | 844 |
| 删除：跨标签冲突 | 0 |
| 删除：精确重复 | 5,551 |
| 清洗后 | 38,503 |
| real / fake | 21,194 / 17,309 |
| 类别比例 | 55.0 / 45.0 |
| train / val / test | 26,951 / 3,851 / 7,701 |

### 3.2 数据层面的主要发现

- 去重后 fake 占比从 52.3% 降到 45.0%。fake 侧的重复和空正文远多于 real，说明这个数据集的“假新闻”里有大量转载和模板内容。
- `subject` 在两类之间完全不重叠（real 只有 `politicsNews` / `worldnews`，fake 是另外六个值），是直接泄漏，已排除。
- `date` 分布不同，只用于审计，不进入特征。
- 1.3% 的 test 行与前 200 字符存在近重复；这些行 F1=1.0，其余 0.9962。影响不大，但 Stage 10 需要按近重复簇重划分确认。
- 测试集在每个 `(model, condition, seed)` 上只评估一次，写入 manifest 并设为只读。

---

## 4. 实验结果

### 4.1 主基线（test F1，阈值 0.5）

| 模型 | title_only | body_only | concat | fusion | alpha |
|---|---:|---:|---:|---:|---:|
| LR | 0.9429 | 0.9929 | 0.9962 | 0.9962 | 0.40 |
| SVM | 0.9425 | 0.9957 | 0.9975 | 0.9970 | 0.50 |

val 上的 fusion F1 为 LR 0.9951、SVM 0.9960，和 test 很接近，没有明显过拟合。

fusion 的完整 test 指标：

| 模型 | Acc | Precision | Recall | F1 | ROC-AUC | PR-AUC | TP/FP/FN/TN |
|---|---:|---:|---:|---:|---:|---:|---|
| LR | 0.9966 | 0.9968 | 0.9957 | 0.9962 | 0.9998 | 0.9997 | 3447/11/15/4228 |
| SVM | 0.9973 | 0.9963 | 0.9977 | 0.9970 | 0.9998 | 0.9997 | 3454/13/8/4226 |

### 4.2 shortcut 与单规则

| 探针 | test F1 | test Acc |
|---|---:|---:|
| 单条 `(Reuters)` 规则 | 0.9943 | 0.9948 |
| 18 个格式特征 + LR | 0.9953 | 0.9957 |
| LR fusion（全文） | 0.9962 | 0.9966 |
| SVM fusion（全文） | 0.9970 | 0.9973 |

18 个格式特征只用 dateline、URL、Getty/featured image、大小写、标点和长度，不读正文。全文模型只比它高 0.0009–0.0017。

### 4.3 来源线索遮蔽

| 模型 | 遮蔽 | fusion 原 F1 | 遮蔽后 F1 | ΔF1 | 95% CI |
|---|---|---:|---:|---:|---|
| LR | dateline_url | 0.9962 | 0.9928 | -0.0035 | [-0.0051, -0.0022] |
| LR | format | 0.9962 | 0.9903 | -0.0059 | [-0.0080, -0.0041] |
| SVM | dateline_url | 0.9970 | 0.9944 | -0.0026 | [-0.0038, -0.0014] |
| SVM | format | 0.9970 | 0.9913 | -0.0056 | [-0.0075, -0.0039] |

- title-only 在遮蔽前后完全不变（Δ=0，p=1），说明遮蔽只动了正文。
- body-only 掉得比 fusion 多，因为 fusion 还有标题。
- 即使使用 `format` 遮蔽，fusion 仍有 0.9903–0.9913。
- 遮蔽后 LR body 的 top 特征仍有 `said`、`president donald`、`reuters`、`just`、`gop`，说明来源/话题/风格信号不止 dateline 和 URL 这一层。

### 4.4 统计检查

- fusion vs body：LR +0.0033（95% CI [0.0015, 0.0052]，p=0.001）；SVM +0.0013（[-0.0004, 0.0030]，p=0.176）。
- fusion vs concat：LR 0.0000（[-0.0012, 0.0012]，p=0.80）；SVM -0.0006（[-0.0016, 0.0003]，p=0.39）。两者可以视为同等强基线。
- LR vs SVM fusion：-0.0007（[-0.0018, 0.0003]），没有显著差异。
- 校准：fusion test Brier 0.0103（LR）/ 0.0112（SVM）；ECE 0.046 / 0.042。中间概率分箱样本少，极端分箱很准。
- test 错误数：LR fusion 26（FP 11，FN 15）；SVM fusion 21（FP 13，FN 8）。遮蔽后错误数分别升到 50–67 和 39–60，但 F1 只动 0.003–0.006。

---

## 5. 这些结果说明什么

### 5.1 创新性

不是“又训练了几个模型”，而是把 ISOT 当成一个待审计的 benchmark：

- 用三个独立探针测同一个问题：单条规则、18 特征格式模型、全文模型。
- 再做单点遮蔽，观察分数是否下降。
- 结果是三个探针都落在 0.99–0.997，遮蔽只带来 0.0026–0.0059 的下降。

这说明 ISOT 的高分不是“内容理解”的结果，而是来源、排版和话题线索共同作用的结果。单点遮蔽不够，因为线索是冗余的。

### 5.2 工程性

- 数据、划分、模型、预测、指标都可追溯；split manifest 有 SHA-256，test 只读。
- 所有实验由 `configs/*.yaml` 驱动，run 目录保存 config、环境、git commit 和日志。
- 50 个测试覆盖清洗、划分、TF-IDF、融合、指标、遮蔽和 Kaggle CLI 回退；CI 和 ruff 通过。
- 环境记录在 `docs/environment.md`，可用 `scripts/env_report.py` 重新生成。

### 5.3 思维性

- 同一批 test 样本、同一阈值、同一 C/alpha 下比较，避免“调参调出来的差异”。
- 对“模型打平”的结论使用 CI 和 McNemar，而不是只看点估计。
- 先承认 benchmark 饱和，再设计能暴露问题的实验，而不是继续追 0.997。

---

## 6. 接下来做什么

### 6.1 Week 9 之前

不再开新实验。把现有结果整理成：

- 1 页 report：数据口径、协议、主表、shortcut/遮蔽结论、下一步。
- 8 页 slides：数据、模型、结果、问题、下一步。
- 2 张图：四路 F1 柱状图；单规则/18 特征/全文/遮蔽后的 F1 对比。
- 2 个 FP/FN 例子：从 `results/predictions` 取原文片段。

### 6.2 Week 9 之后

1. **加强遮蔽梯度**：不只遮 dateline/URL，再遮 `getty`、`featured image`、标题风格和话题词，做 dose-response 曲线。原因是单点遮蔽后仍有 0.99，需要找到残余线索。
2. **制造有区分度的测试集**：cue-balanced test、时间划分、近重复分组划分。原因是当前 test 只有 21–67 个错误，模型差异被天花板压平。
3. **神经模型放进同一协议**：Bi-LSTM 和 DistilBERT 分别跑 original 和 masked，重点比较 ΔF1，而不是原始 F1。原因是如果神经模型也依赖来源线索，那么“更强模型”并不能解决问题。
4. **标签审计**：抽 100–200 条人工核对，估计来源标签的噪声。原因是所有结论都受标签质量限制。

---

## 7. 三个实际问题

### 问题 1：遮蔽后仍然 0.99，单点消融无法定位“真正的线索”

dateline/URL/媒体署名都去掉后，fusion 仍有 0.9903–0.9913。说明线索分布在标题、话题词、句式、长度和残余来源词里。继续单点遮蔽会陷入“遮一个、模型换一个”的循环。

痛点：很难给出一个干净的“content-only”模型，也很难把 ΔF1 解释成单一因果效应。

### 问题 2：天花板效应让模型比较失去统计空间

test 只有 7,701 条，fusion 的错误数是 21–26；模型间 F1 差异只有 0.001–0.006，CI 经常跨 0。继续加模型、调参，得到的差异很可能只是噪声。

痛点：在当前 test 分布上，“哪个模型更好”不是一个有意义的问题。需要更难的测试集，而不是更多模型。

### 问题 3：标签构造和测试复用限制了结论强度

标签是来源标签，不是逐篇事实核查；去重后类别比例变成 55:45；test 已经用于 8 个预注册条件；1.3% 的近重复仍未按簇划分。

痛点：任何“模型能识别假新闻”的说法都站不住；我们只能声称“模型能识别来源/排版”。要做更强的因果或泛化结论，需要 label audit、group split 和新的外部数据。

---

## 8. 1 页 report 与 slides 提纲

### 1 页 report

1. 标题 + repo 链接。
2. 数据：44,898 → 38,503；55:45；去重/空正文；subject 泄漏。
3. 协议：70/10/20；TF-IDF 只在 train 上 fit；alpha/阈值只在 val 上选；test 只读。
4. 结果：LR/SVM 四路 F1 表；shortcut 和遮蔽各一行。
5. 结论：单规则 0.9943，18 特征 0.9953，全文 0.9962；遮蔽后仍 0.99。
6. 下一步：更强遮蔽、难测试集、神经模型同协议。

### 8 页 slides

1. 标题 + 一句话结论。
2. 数据：44,898 → 38,503；55:45；subject 泄漏。
3. 协议：split、只读 test、只在 val 上调参。
4. 模型：LR/SVM 双编码 + 融合；四路条件。
5. 结果：四路 F1 表。
6. 转折：单规则/18 特征/全文/遮蔽后的对比。
7. 问题：遮蔽后仍 0.99；1 个 FP/FN 例子。
8. 下一步 + repo。

---

## 9. 还缺的实验数据（提醒）

- **accuracy 不缺**：`results/all_metrics.csv` 已包含所有 50 个条件的 accuracy、precision、recall、F1、ROC-AUC、PR-AUC 和混淆矩阵。
- **缺逐条件的 bootstrap CI**：`RESULTS.md` 只有关键比较的 CI；如果 slides 要给每个模型画 error bar，需要从 predictions 重新算。
- **缺结果图**：目前只有 EDA 图；四路 F1、ΔF1、混淆矩阵、alpha 曲线都还没画。
- **缺具体错误样本表**：需要从 predictions 取 FP/FN 各 2 条，附标题/正文片段。
- **缺神经模型结果**：Bi-LSTM、DistilBERT 尚未实现，因此没有 original/masked 的 2×2 对比。
- **缺时间划分、LIAR、近重复分组结果**：Stage 10 未做。
- **`feature_audit.csv` 的 category 列还是空的**：top 特征尚未人工标注为 source/format/content。
