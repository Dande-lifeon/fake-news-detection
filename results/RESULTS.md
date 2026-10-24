# 模型结果台账（Model Results）

> 本文件是项目唯一的模型结果汇总，后续每个阶段（shortcut / Bi-LSTM / DistilBERT / 来源线索 / 时间划分等）的结果都追加到本文件。
> 机器可读产物：`results/metrics.csv`、`results/fusion.json`、`results/predictions/*.parquet`（逐样本预测，git-ignored）。
> EDA 报告单独放在 `results/eda_report.md`。

最后更新：2026-10-24

---

## 1. 实验设置

| 项目 | 值 |
|---|---|
| 数据集 | ISOT Fake and Real News（Kaggle `clmentbisaillon/fake-and-real-news-dataset`） |
| 清洗后样本 | 38,503（real 21,194 / fake 17,309） |
| 划分 | train 26,951 / val 3,851 / test 7,701，分层，seed=42 |
| 正类 | `label=1`（fake） |
| 文本分支 | title TF-IDF（≤20k）、body TF-IDF（≤50k），只在 train 上 fit |
| 融合 | `alpha * p_title + (1-alpha) * p_body`，alpha 只在 val 上选 |
| 测试纪律 | test 只在 `--final` 读取一次；模型在 train+val 上 refit 后评估 |
| 运行时间 | LR 约 45s；SVM 约 95s（CPU） |
| run_id | `lr_seed42_20261024_123834`、`svm_seed42_20261024_123921` |
| 代码版本 | `91b215a`（结果提交 `a203df6`） |

---

## 2. Stage 3 — TF-IDF + Logistic Regression / Linear SVM

### 2.1 指标（seed 42）

| 模型 | 条件 | val F1 | test F1 | test Precision | test Recall | test ROC-AUC | alpha |
|---|---|---:|---:|---:|---:|---:|---:|
| LR | title_only | 0.9387 | 0.9433 | 0.9393 | 0.9474 | 0.9885 | — |
| LR | body_only | 0.9916 | 0.9940 | 0.9905 | 0.9974 | 0.9997 | — |
| LR | concat | 0.9939 | 0.9960 | 0.9948 | 0.9971 | 0.9998 | — |
| LR | **fusion** | **0.9951** | **0.9962** | **0.9968** | **0.9957** | **0.9998** | **0.40** |
| SVM | title_only | 0.9396 | 0.9422 | 0.9407 | 0.9437 | 0.9885 | — |
| SVM | body_only | 0.9939 | 0.9954 | 0.9925 | 0.9983 | 0.9999 | — |
| SVM | concat | 0.9959 | 0.9963 | 0.9937 | 0.9988 | 0.9999 | — |
| SVM | **fusion** | **0.9960** | **0.9970** | **0.9963** | **0.9977** | **0.9998** | **0.50** |

test 指标使用 val 上按条件选出的阈值；`metrics.csv` 同时保留 val@0.5 与 val@tuned。
fusion 的 test 混淆矩阵：LR TP=3447 / FP=11 / FN=15 / TN=4228；SVM TP=3454 / FP=13 / FN=8。

### 2.2 alpha 与阈值

- LR：alpha = **0.40**，val F1 从 alpha=0 的 0.9916 升到 0.9951，alpha>0.5 后快速下降（alpha=1 为 0.9387）。
- SVM：alpha = **0.50**，val F1 在 0.35–0.50 几乎持平（0.9959–0.9960），alpha>0.5 后下降。
- val 选出的阈值普遍低于 0.5：LR 为 0.32–0.50，SVM 为 0.23–0.50；但在 test 上换成 tuned 阈值几乎没有收益，SVM concat 反而下降 0.0013。
- test@0.5 与 test@tuned 对比：LR concat 0.9962 vs 0.9960；SVM concat 0.9975 vs 0.9963；fusion 两者相同（LR 0.9962，SVM 0.9970）。
- **结论：0.5 是更稳健的默认阈值**；tuned 阈值只作为敏感性分析，不建议作为主结果。

### 2.3 关键发现

1. **body 远强于 title**：body-only 的 test F1 为 0.9940/0.9954，title-only 只有 0.9433/0.9422；标题携带的信号有限。
2. **融合略好于 body，但提升很小**：LR fusion−body = +0.0023（95% CI [+0.0004, +0.0042]，McNemar p=0.021）；SVM = +0.0016（95% CI [−0.0000, +0.0033]，p=0.082），属于边缘显著。
3. **late fusion 没有显著优于 early concat**：LR fusion−concat = +0.0003（95% CI [−0.0007, +0.0014]，p=0.80）；SVM = +0.0007（95% CI [−0.0004, +0.0019]，p=0.33）。两者可以视为同等强 baseline。
4. **LR 与 SVM 没有显著差异**：fusion test F1 差 −0.0007（95% CI [−0.0018, +0.0003]）。SVM 只在 body/concat 上略好。
5. **阈值不敏感区间宽**：fusion test F1 在阈值 0.4–0.6 为 0.9948–0.9962（LR）/ 0.9904–0.9970（SVM）；阈值 <0.3 或 >0.7 才明显下降。
6. **校准中等**：fusion test Brier 0.0103（LR）/ 0.0112（SVM），ECE 0.046 / 0.042；0.4–0.6 中间概率分箱样本少且不可靠，极端分箱（<0.1、>0.9）非常准。
7. **错误集中且可解释**：LR fusion FP=11、FN=15；SVM FP=13、FN=8。FP 多为风格特殊的 Reuters 长文（如 "Mitt Romney speech…"、"Pictures of the Year"）；FN 多为 2015 年 fake 的全大写/正式风格旧文。按年份看，2015 年切片 F1 约 0.992，2016–2017 约 0.997。
8. **title/body 互补性有限**：test 中 "title 错但 body 对" 约 384–393 条，"body 错但 title 对" 仅 24–32 条；fusion 主要修正少量 body 错误。
9. **近重复泄漏很小**：train∩test 的 `text_clean` 完全相同 4 条（同标签，标题略有差异）；前 200 字符重合 102 条 test 行（1.3%），这些行 F1=1.0，其余 F1=0.9962；没有跨标签冲突。对结论影响可忽略，但说明去重只按 `title+text`，不是按 text 单独去重。

### 2.4 来源线索捷径（最重要 caveat）

| 线索 | real | fake |
|---|---|---|
| 正文含 `(Reuters)` | 99.08% | 0.03% |
| 正文含 URL | 0.14% | 14.73% |
| 含 `getty` / `featured image` | 极少 | 大量（fake 专属） |

- 单条规则 **"没有 (Reuters) 就判 fake"** 在 test 上已经达到 **F1=0.9943、Accuracy=0.9948**；完整 LR fusion 为 0.9962，只高 0.0019。
- LR body 分支权重最大的 real 特征：`reuters`(−47.96)、`said`(−21.02)、`washington reuters`(−18.10)；最大的 fake 特征：`featured image`(+10.81)、`image`(+10.60)、`getty`(+7.08)。
- 因此 **99.6%+ 的 F1 主要反映"来源/排版风格"，而不是"新闻真实性"**。这是 ISOT 数据集的已知缺陷，不是训练代码 bug；也是 Stage 8 必须做来源线索遮蔽实验的原因。

### 2.5 是否符合预期

**符合。** ISOT 是一个接近饱和的 benchmark，TF-IDF + LR/SVM 的公开结果通常就在 99% 以上；本项目的 body 0.994、fusion 0.9962–0.9970、title 0.942–0.943，与文献和常识一致：

- 模型没有出现异常高或异常低的分数；
- val/test 差距很小（test 甚至略高），没有明显过拟合；
- α 在 val 上选择合理（LR 0.40，SVM 0.50）；
- 错误样本可解释，不是数据/代码错位；
- 四路消融方向正确：body > title，concat ≈ fusion。

**但不能据此认为模型具备真实假新闻识别能力。** 单条 Reuters 正则就有 0.994 的 F1，说明数据集把"来源"泄漏成了"标签"。下一步应优先做：

1. Stage 8 来源线索遮蔽（去 `(Reuters)` dateline、URL、`getty`/`featured image` 后重训），看 ΔF1；
2. Stage 10 近重复分组划分，排除 1.3% 近重复的残余影响；
3. Stage 4 shortcut baseline（18 个格式特征 + LR），量化"只靠格式能走多远"；
4. 在报告中明确写"benchmark 饱和 + 来源捷径"，不要把高 F1 当作真实场景结论。

---

## 3. 后续结果（占位，完成后更新）

| 阶段 | 内容 | 状态 |
|---|---|---|
| 4 | shortcut / 格式特征 baseline | 待做 |
| 5 | Bi-LSTM（title/body 双分支 + 门控融合） | 待做 |
| 6 | DistilBERT（共享编码器 + 门控融合） | 待做 |
| 7 | bootstrap CI、McNemar、错误分析、校准 | 部分完成（本页已含 Stage 3 的 CI/McNemar/校准） |
| 8 | 来源线索遮蔽 ΔF1 | 待做 |
| 10 | 时间划分 / LIAR / 近重复分组 | 待做 |

> 更新约定：每完成一个阶段，在本文件追加一节（指标表 + 关键发现 + 是否符合预期），并在 `results/metrics.csv` 追加对应行；不要覆盖旧结果。
