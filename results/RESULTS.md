# 模型结果台账（Model Results）

> 本文件是项目唯一的模型结果汇总，后续每个阶段（shortcut / Bi-LSTM / DistilBERT / 来源线索 / 时间划分等）的结果都追加到本文件。
> 机器可读产物：`results/all_metrics.csv`、`results/fusion.json`、`results/shortcut.json`、
> `results/source_cue.json`、`results/predictions/*.parquet`（逐样本预测，git-ignored）。
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
| 代码版本 | Stage 3 `91b215a`；Stage 4/8 `01cc07a` |

---

## 2. Stage 3: TF-IDF + logistic regression / linear SVM

### 2.1 test 指标（@0.5）

| model | condition | Acc | P | R | F1 | ROC-AUC | PR-AUC | TP | FP | FN | TN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LR | title_only | 0.9494 | 0.9552 | 0.9310 | 0.9429 | 0.9885 | 0.9862 | 3223 | 151 | 239 | 4088 |
| LR | body_only | 0.9936 | 0.9948 | 0.9910 | 0.9929 | 0.9997 | 0.9996 | 3431 | 18 | 31 | 4221 |
| LR | concat | 0.9966 | 0.9974 | 0.9951 | 0.9962 | 0.9998 | 0.9997 | 3445 | 9 | 17 | 4230 |
| LR | **fusion** | **0.9966** | **0.9968** | **0.9957** | **0.9962** | **0.9998** | **0.9997** | **3447** | **11** | **15** | **4228** |
| SVM | title_only | 0.9486 | 0.9480 | 0.9370 | 0.9425 | 0.9885 | 0.9864 | 3244 | 178 | 218 | 4061 |
| SVM | body_only | 0.9961 | 0.9945 | 0.9968 | 0.9957 | 0.9999 | 0.9998 | 3451 | 19 | 11 | 4220 |
| SVM | concat | 0.9978 | 0.9977 | 0.9974 | 0.9975 | 0.9999 | 0.9999 | 3453 | 8 | 9 | 4231 |
| SVM | **fusion** | **0.9973** | **0.9963** | **0.9977** | **0.9970** | **0.9998** | **0.9997** | **3454** | **13** | **8** | **4226** |

### 2.1b validation 指标（@0.5）

| model | condition | Acc | P | R | F1 | ROC-AUC | PR-AUC | TP | FP | FN | TN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LR | title_only | 0.9457 | 0.9530 | 0.9249 | 0.9387 | 0.9887 | 0.9877 | 1601 | 79 | 130 | 2041 |
| LR | body_only | 0.9925 | 0.9988 | 0.9844 | 0.9916 | 0.9993 | 0.9993 | 1704 | 2 | 27 | 2118 |
| LR | concat | 0.9945 | 0.9983 | 0.9896 | 0.9939 | 0.9999 | 0.9999 | 1713 | 3 | 18 | 2117 |
| LR | **fusion** | **0.9956** | **0.9994** | **0.9908** | **0.9951** | **0.9998** | **0.9998** | **1715** | **1** | **16** | **2119** |
| SVM | title_only | 0.9460 | 0.9451 | 0.9341 | 0.9396 | 0.9884 | 0.9874 | 1617 | 94 | 114 | 2026 |
| SVM | body_only | 0.9945 | 0.9971 | 0.9908 | 0.9939 | 0.9995 | 0.9995 | 1715 | 5 | 16 | 2115 |
| SVM | concat | 0.9964 | 0.9983 | 0.9936 | 0.9959 | 0.9999 | 0.9999 | 1720 | 3 | 11 | 2117 |
| SVM | **fusion** | **0.9964** | **0.9971** | **0.9948** | **0.9960** | **0.9998** | **0.9998** | **1722** | **5** | **9** | **2115** |

主表统一用阈值 0.5。val 上调出的 per-condition 阈值只作敏感性分析：
LR test（title/body/concat/fusion）= 0.9433 / 0.9940 / 0.9960 / 0.9962；
SVM = 0.9422 / 0.9954 / 0.9963 / 0.9970；相对 @0.5 的最大变化为 0.0012。
所有模型/条件/划分的同一套指标（50 行）也保存在 `results/all_metrics.csv`。

### 2.2 alpha 与阈值

- LR：alpha = **0.40**，val F1 从 alpha=0 的 0.9916 升到 0.9951，alpha>0.5 后快速下降（alpha=1 为 0.9387）。
- SVM：alpha = **0.50**，val F1 在 0.35–0.50 几乎持平（0.9959–0.9960），alpha>0.5 后下降。
- **alpha 是概率混合权重，不是分支重要性。** title 的 Brier 明显更差（LR 0.0390 vs body 0.0055；SVM 0.0387 vs body 0.0030），概率平均需要给 title 更大权重来补偿校准差异。
- val 选出的阈值普遍低于 0.5：LR 为 0.32–0.50，SVM 为 0.23–0.50；但在 test 上换成 tuned 阈值几乎没有收益，SVM concat 反而下降 0.0013。
- test@0.5 与 test@tuned 对比：LR concat 0.9962 vs 0.9960；SVM concat 0.9975 vs 0.9963；fusion 两者相同（LR 0.9962，SVM 0.9970）。
- **结论：0.5 是更稳健的默认阈值**；tuned 阈值只作为敏感性分析，主表已统一为 0.5。

#### alpha 的稳健性：概率平均 vs logit 平均

| 模型 | 融合方式 | 最优 alpha | val F1 | test F1 |
|---|---|---:|---:|---:|
| LR | 概率平均 | 0.40 | 0.9951 | 0.9962 |
| LR | logit 平均 | 0.30 | 0.9948 | 0.9961 |
| SVM | 概率平均 | 0.50 | 0.9960 | 0.9970 |
| SVM | logit 平均 | 0.45 | 0.9962 | 0.9975 |

换成 logit 平均后最优 alpha 降到 0.30/0.45，但 test F1 基本不变（−0.0001 / +0.0005），
0.40 是概率尺度下的选择，不是 bug。rank 平均在固定 0.5 阈值下明显更差
（LR 0.9701，SVM 0.9733），因为丢掉了置信度信息，不作为替代方案。

### 2.3 主要发现

1. **body 远强于 title**：@0.5 的 body-only test F1 为 0.9929/0.9957，title-only 只有 0.9429/0.9425；标题携带的信号有限。
2. **融合略好于 body，但提升很小**：@0.5 的 LR fusion−body = +0.0033（95% CI [+0.0015, +0.0052]，McNemar p=0.001）；SVM = +0.0013（95% CI [−0.0004, +0.0030]，p=0.176），SVM 不显著。
3. **late fusion 没有显著优于 early concat**：@0.5 的 LR fusion−concat = 0.0000（95% CI [−0.0012, +0.0012]，p=0.80）；SVM = −0.0006（95% CI [−0.0016, +0.0003]，p=0.39）。两者可以视为同等强 baseline，SVM concat 甚至略高。
4. **LR 与 SVM 没有显著差异**：fusion test F1 差 −0.0007（95% CI [−0.0018, +0.0003]）。SVM 只在 body/concat 上略好。
5. **阈值不敏感区间宽**：fusion test F1 在阈值 0.4–0.6 为 0.9930–0.9962（LR）/ 0.9863–0.9970（SVM）；阈值 <0.3 或 >0.7 才明显下降。
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

**但不能据此认为模型具备真实假新闻识别能力。** 单条 Reuters 正则就有 0.994 的 F1，说明数据集把"来源"泄漏成了"标签"。Stage 4 和 Stage 8 就是对这一点的直接检验。

---

## 3. Stage 4: shortcut / 格式特征 baseline

只用 18 个格式特征（`fakenews/features/shortcut.py`），不读正文内容：
Reuters dateline、URL、`via`/`getty`/`featured image`、关注/订阅、标题问号/感叹号、
大写比例、长度、数字比例、引号数等。LR + 5 折 CV 选 C，train+val refit 后 test 一次。

| 项目 | 值 |
|---|---|
| 特征数 | 18 |
| 最优 C | 1.0 |
| val F1 @0.5 / @0.55 | 0.9957 / 0.9968 |
| **test F1 @0.5** | **0.9953**（@0.55 为 0.9954） |
| test Precision / Recall | 0.9908 / 0.9997 |
| test ROC-AUC / PR-AUC | 0.9999 / 0.9998 |

完整指标：

| split | Acc | P | R | F1 | ROC-AUC | PR-AUC | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0.9961 | 0.9920 | 0.9994 | 0.9957 | 0.9993 | 0.9977 | 1730 | 14 | 1 | 2106 |
| test | 0.9957 | 0.9908 | 0.9997 | 0.9953 | 0.9999 | 0.9998 | 3461 | 32 | 1 | 4207 |

Top 系数（标准化后）：`has_reuters_dateline` −4.65、`starts_with_city_dateline` −1.63、
`quote_count` −1.44、`title_len` +1.37、`has_getty` +1.36、`has_via` +1.34、
`title_has_exclamation` +1.21。方向完全符合"来源/排版"解释。

单特征规则（test）：

| 规则 | F1 | Accuracy |
|---|---:|---:|
| 无 `(Reuters)` → fake | 0.9943 | 0.9948 |
| 无 Reuters / 有 URL / 有 getty → fake | 0.9934 | 0.9940 |
| 有 URL → fake | 0.2564 | 0.6159 |
| 有 getty → fake | 0.3723 | 0.6532 |

**对比 Stage 3：**18 个格式特征 test F1@0.5=0.9953，LR fusion=0.9962（仅高 0.0009），
SVM fusion=0.9970（高 0.0017）。配对 bootstrap：shortcut−LR fusion = −0.0010
（95% CI [−0.0031, +0.0010]，McNemar p=0.43）；shortcut−SVM fusion = −0.0017
（95% CI [−0.0039, +0.0003]，p=0.13）。**18 个格式特征与完整正文模型在统计上
无法区分。** 它甚至略高于 LR body-only @0.5（0.9929），与 SVM body-only（0.9957）
基本持平。**不读正文语义，只用 18 个表层格式特征，就已经达到完整模型的 99.9% 水平。**

---

## 4. Stage 8: 来源线索遮蔽实验

在 train/val/test 上遮蔽正文后重训 LR/SVM，C 和 alpha 沿用 Stage 3，不再调参；
test 只评估一次。两种遮蔽强度：

- `dateline_url`：去 `CITY (Reuters) -` 与 URL（计划中的主实验）；
- `format`：再额外去 `featured image`、`Photo by ... / Getty Images`、`Image via ...`、
  `via`、`follow us`/`subscribe`、`read more`。

### 4.1 命中数（清洗后全量）

| 线索 | real（21,194） | fake（17,309） |
|---|---:|---:|
| `(Reuters)` | 21,027 | 7 |
| URL | 41 | 2,504 |
| `getty` | 4 | 4,060 |
| `featured image` | 0 | 8,160 |
| `via` | 465 | 9,620 |

### 4.2 test F1（同一批样本、阈值 0.5）

| 模型 | 遮蔽 | fusion 原 | fusion 遮蔽后 | Δfus | 95% CI | body Δ | title Δ |
|---|---|---|---:|---:|---:|---|---:|---:|
| LR | dateline_url | 0.9962 | 0.9928 | −0.0035 | [−0.0051, −0.0022] | −0.0070 | 0.0000 |
| LR | format | 0.9962 | 0.9903 | −0.0059 | [−0.0080, −0.0041] | −0.0087 | 0.0000 |
| SVM | dateline_url | 0.9970 | 0.9944 | −0.0026 | [−0.0038, −0.0014] | −0.0069 | 0.0000 |
| SVM | format | 0.9970 | 0.9913 | −0.0056 | [−0.0075, −0.0039] | −0.0098 | 0.0000 |

- 所有非 title 条件的 McNemar p ≤ 0.0003（format 全部 p<0.0001）；title-only 完全
  不变（p=1），证明遮蔽只动了正文，没有误伤标题。
- body-only 掉得比 fusion 多（−0.007～−0.010），因为 fusion 还有未遮蔽的标题。
- **即使去掉 dateline、URL 和媒体署名，fusion 仍有 0.9903–0.9944。**
- 遮蔽后 LR body 的 top 特征变成 `said`(−27.99)、`president donald`(−15.56)、
  `just`(+12.57)、`gop`(+12.25)、`reuters`(−11.67)：仍有残余来源词、话题词和风格词，
  说明"来源/话题"信号不止 dateline/URL 这一层。

### 4.2b 完整指标（test，fusion）

| model | mask | Acc | P | R | F1 | TP | FP | FN | TN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LR | original | 0.9966 | 0.9968 | 0.9957 | 0.9962 | 3447 | 11 | 15 | 4228 |
| LR | dateline_url | 0.9935 | 0.9948 | 0.9908 | 0.9928 | 3430 | 18 | 32 | 4221 |
| LR | format | 0.9913 | 0.9930 | 0.9876 | 0.9903 | 3419 | 24 | 43 | 4215 |
| SVM | original | 0.9973 | 0.9963 | 0.9977 | 0.9970 | 3454 | 13 | 8 | 4226 |
| SVM | dateline_url | 0.9949 | 0.9937 | 0.9951 | 0.9944 | 3445 | 22 | 17 | 4217 |
| SVM | format | 0.9923 | 0.9925 | 0.9905 | 0.9913 | 3429 | 27 | 33 | 4212 |

### 4.3 预期 vs 实测：为什么 ΔF1 只有 0.003–0.006

执行方案里给的验收示例是 `Δ=0.048`（0.991→0.943），但原文写的是"**例如**"，不是硬预测。
实测只有 −0.0035～−0.0059，偏差来自三个原因：

1. **错误数其实翻倍了**：LR fusion 的 test 错误 26 → 50（dateline/URL）→ 67（format）；
   SVM 21 → 39 → 60。F1 在 99% 以上被 TP 主导，错误数翻倍只让 F1 动 0.003–0.006。
2. **遮蔽只删了 token，没删完来源/话题**：标题未动；`said`、`washington`、`president donald`
   等 Reuters 风格词，以及 `just`、`gop`、`obama`、`watch`、全大写等 fake 风格词仍在。
3. **重训会适应**：模型在遮蔽文本上重新分配权重，残余线索被重新利用；这不是"遮蔽没生效"。

更准确的说法是：显式来源线索可测且显著，但捷径是分布式的，单点遮蔽无法消除。若一定要看到
0.048 量级的下降，需要同时遮蔽标题、做风格/大小写归一化，或按话题/来源分组评估；这已经
超出 dateline/URL 遮蔽的范围。

### 4.4 产物

`results/source_cue.json`（两种模型 × 两种遮蔽 × 四个条件，含 CI/McNemar/命中数）、
`results/feature_audit.csv`（original/dateline_url/format 的 top 特征）、
`results/all_metrics.csv`（全部模型/条件/划分的统一 Acc/P/R/F1/AUC/CM，50 行）、
`results/cis.csv`（逐条件 bootstrap 95% CI，200 行）、
`results/error_examples.md`（LR fusion 的 2 个 FP、2 个 FN）、
`results/figures/fig_*.{pdf,png}`（四路 F1、探针对比、混淆矩阵、alpha 曲线）、
`results/predictions/*_source_cue_*.parquet`（逐样本预测，git-ignored）。

---

## 5. 跨阶段诊断：benchmark 确实饱和

| 探针 | test F1 |
|---|---:|
| 单条 `(Reuters)` 规则 | 0.9943 |
| 18 个格式特征 + LR | 0.9953 |
| LR body-only @0.5 | 0.9929 |
| LR fusion（全文） | 0.9962 |
| SVM fusion（全文） | 0.9970 |
| LR fusion + dateline/URL 遮蔽 | 0.9928 |
| LR fusion + format 遮蔽 | 0.9903 |
| SVM fusion + dateline/URL 遮蔽 | 0.9944 |
| SVM fusion + format 遮蔽 | 0.9913 |

三个独立探针（正则、18 特征、全文模型）都落在 0.99–0.997；全文模型只比 18 个
格式特征高 0.0009–0.0017；遮蔽来源线索只损失 0.0026–0.0059，仍然 99% 以上。
**结论：ISOT 的标签几乎可以被"来源/排版/话题风格"直接解出，高 F1 不是内容理解
或真实性推理的证据。** 这与公开文献中 ISOT 接近饱和的结论一致，属于数据集性质，
不是实现错误。

---

## 6. 后续结果（状态）

| 阶段 | 内容 | 状态 |
|---|---|---|
| 4 | shortcut / 格式特征 baseline | ✅ 完成（test F1 0.9953） |
| 5 | Bi-LSTM（title/body 双分支 + 门控融合） | ⏸ 按当前范围暂缓 |
| 6 | DistilBERT（共享编码器 + 门控融合） | ⏸ 按当前范围暂缓 |
| 7 | bootstrap CI、McNemar、错误分析、校准 | 🟡 已实现 bootstrap/McNemar/校准；`evaluate.py` 汇总器待做 |
| 8 | 来源线索遮蔽 ΔF1 | ✅ 完成（两种遮蔽 × LR/SVM） |
| 9 | 图表 + 文档 | ⬜ 部分文档已更新 |
| 10 | 时间划分 / LIAR / 近重复分组 | ⏸ 按要求暂缓 |

> 更新约定：每完成一个阶段，在本文件追加一节（指标表、主要发现、是否符合预期），并在 `results/all_metrics.csv` 追加对应行；不要覆盖旧结果。Stage 4/8 的逐样本预测和 JSON 已保存在 `results/` 下。
