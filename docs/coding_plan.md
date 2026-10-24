# 假新闻检测项目 · 代码开发计划（Coding Plan）

> 这份文档是给**开发会话**用的：读完第 1–4 节就能建仓库、装环境、按第 7 节顺序开始写代码。
> 只保留工程与实现相关内容；
> 配套文件：`Fake_News_Project_满分执行方案.md`（项目背景与实验设计），本文是它的“可执行代码版”。
> 建议把本文件放进仓库的 `docs/coding_plan.md`，作为唯一的开发入口文档。
>
> **Implementation status (2026-10-24):** Stages 0–4 and 8 are implemented; Stage 7 is partial. Commands for `fakenews.train.bilstm`, `fakenews.train.distilbert`, `fakenews.eval.evaluate`, and `fakenews.viz.figures` are future work and will fail until those modules exist. See `README.md` for the current runnable path.

---

## 0. 使用说明

1. 先读第 1 节（做什么）、第 2 节（数据长什么样）、第 3 节（仓库结构）。
2. 按第 7 节的顺序实现；每一阶段都有“完成标准”，没达到不要进入下一阶段。
3. 所有路径相对仓库根目录；不要写绝对路径。
4. 所有实验必须能由 `configs/*.yaml` + 命令行复现；禁止在 notebook 里手工改数据。
5. 代码风格：Python 3.11+，类型标注，`ruff` 通过，`pytest` 通过。

---

## 1. 项目技术摘要

**任务**：二分类。输入一篇新闻的 `title` + `body`，预测 `label=1`（fake）或 `label=0`（real）。

**数据**：ISOT Fake and Real News（Kaggle `clmentbisaillon/fake-and-real-news-dataset`），两个 CSV：`True.csv`（Reuters，label=0）、`Fake.csv`（被标记站点，label=1）。

**四个模型**：

| 模型 | 标题编码 | 正文编码 | 融合 |
|---|---|---|---|
| Logistic Regression | TF-IDF（≤20k） | TF-IDF（≤50k） | 晚融合 `α·p_title + (1−α)·p_body` |
| Linear SVM | 同上 | 同上 | 两分支校准后晚融合 |
| Bi-LSTM | 独立 Bi-LSTM（≤32 token） | 独立 Bi-LSTM（≤300 token） | 门控融合 |
| DistilBERT | 共享权重编码器（≤32 token） | 共享权重编码器（≤256 token） | 门控融合 |

**必须保留的对照**：

- `title-only`、`body-only`、`early concat`、`late fusion` 四路消融（四个模型至少经典模型要全做）。
- `shortcut-only` 基线：只用 15–20 个手工格式特征 + LR。
- 来源线索遮蔽：去掉正文里的 `CITY (Reuters) - ` 与 URL 后重训，输出 ΔF1。

**硬约束**：

- 主模型的 test 只在最终评估时使用一次；所有超参、α、阈值只在 train/val 上选（预注册分析例外见 9.5）。
- 全流程能在笔记本 CPU 或免费 Colab T4 上跑通。
- 固定随机种子；保存逐样本预测，而不只是指标。

**非目标**：Web 服务、前端、在线 API、移动端。最多做一个离线 CLI demo。

---

## 2. 数据契约

### 2.1 原始数据

```text
data/raw/True.csv    # 列：title, text, subject, date；label=0
data/raw/Fake.csv    # 列：title, text, subject, date；label=1
```

- 原始文件**只读**，不覆盖、不改名；下载后记录 `sha256` 到 `data/raw/CHECKSUMS.txt`。
- 行数以 `pd.read_csv` 实际加载为准；不要硬编码成 21,417 / 23,502。
- 已知问题：正文重复、`subject` 泄露标签、`date` 分布不同、标签按来源而非逐篇核查。

### 2.2 清洗后数据 `data/processed/articles_clean.parquet`

| 列 | 类型 | 说明 |
|---|---|---|
| `id` | int64 | 合并后、清洗前的稳定行号；所有实验按它对齐 |
| `title` | string | 原始标题，可空 |
| `text_raw` | string | 原始正文，**不做任何删改**；来源线索特征从这列提 |
| `text_clean` | string | 去 HTML、压缩空白、strip 后的正文；建模用 |
| `label` | int8 | 1=fake, 0=real |
| `subject` | string | 仅用于审计，**不得进入任何特征** |
| `date` | string | 原始日期字符串；按时间划分时再解析 |
| `dup_key` | string | `normalized(title + "\n" + text_clean)`，仅审计用 |

### 2.3 划分文件 `data/processed/{train,val,test}.parquet`

- 同一 schema；`train:val:test = 70:10:20`，分层，`seed=42`。
- 三份文件必须无 `id` 交集；类别比例与全集一致（±0.5%）。
- `split_manifest.json` 记录：seed、各集合行数、类别计数、每个 parquet 的 `sha256`、pandas 版本、生成时间。
- `test.parquet` 生成后设为只读；训练脚本不得读取它。

### 2.4 逐样本预测 `results/predictions/{run_id}.parquet`

| 列 | 类型 | 说明 |
|---|---|---|
| `id` | int64 | 对应 `articles_clean.id` |
| `y_true` | int8 | 真实标签 |
| `y_prob` | float32 | 预测为 fake 的概率 |
| `split` | string | train/val/test |
| `model` | string | `lr` / `svm` / `bilstm` / `distilbert` / `shortcut` |
| `condition` | string | `title_only` / `body_only` / `concat` / `fusion` |
| `seed` | int | 随机种子 |

### 2.5 指标表 `results/metrics.csv`

列：

```text
run_id, model, condition, seed, split, threshold,
precision, recall, f1, accuracy, roc_auc, pr_auc,
tp, fp, fn, tn, n, alpha, gate_mean,
train_time_sec, params, git_commit, config_hash, created_at
```

- 正类固定为 `label=1`（fake）。
- 每次评估一行；不要只保留最佳结果。

---

## 3. 仓库结构

```text
fake-news-detection/
├── README.md
├── LICENSE                     # 代码 MIT；数据 CC BY-NC-SA 4.0
├── pyproject.toml
├── requirements.txt
├── Makefile
├── .gitignore
├── .github/
│   └── workflows/ci.yml
├── configs/
│   ├── base.yaml
│   ├── classical_lr.yaml
│   ├── classical_svm.yaml
│   ├── bilstm.yaml
│   ├── distilbert.yaml
│   └── shortcut.yaml
├── data/
│   ├── README.md               # 数据来源、许可、下载方式
│   ├── raw/                    # gitignored
│   └── processed/              # gitignored
├── fakenews/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py
│   │   ├── eda.py
│   │   ├── clean.py
│   │   ├── split.py
│   │   └── dataset.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── tfidf.py
│   │   ├── shortcut.py
│   │   └── masking.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── fusion.py
│   │   ├── classical.py
│   │   ├── bilstm.py
│   │   └── distilbert.py
│   ├── train/
│   │   ├── __init__.py
│   │   ├── common.py
│   │   ├── classical.py
│   │   ├── bilstm.py
│   │   ├── distilbert.py
│   │   └── shortcut.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── bootstrap.py
│   │   ├── error_analysis.py
│   │   └── evaluate.py
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── source_cue.py
│   │   ├── temporal.py
│   │   ├── cross_dataset.py
│   │   ├── chunking.py
│   │   └── near_dup.py
│   └── viz/
│       ├── __init__.py
│       └── figures.py
├── scripts/
│   ├── run_all.sh
│   └── smoke.sh
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   └── tiny.csv
│   ├── test_clean.py
│   ├── test_split.py
│   ├── test_tfidf.py
│   ├── test_fusion.py
│   ├── test_metrics.py
│   ├── test_bootstrap.py
│   └── test_masking.py
├── results/
│   ├── .gitkeep
│   ├── metrics.csv
│   ├── predictions/
│   ├── figures/
│   ├── logs/
│   └── source_cue.json
└── docs/
    ├── data_card.md
    ├── protocol.md
    └── coding_plan.md          # 本文件
```

`.gitignore` 至少包含：

```gitignore
data/raw/
data/processed/
results/logs/
results/predictions/
results/figures/
models/
__pycache__/
*.pyc
.venv/
.ipynb_checkpoints/
.pytest_cache/
.ruff_cache/
```

---

## 4. 配置系统

### 4.1 `configs/base.yaml`

```yaml
seed: 42

paths:
  raw_dir: data/raw
  processed_dir: data/processed
  results_dir: results
  models_dir: models

data:
  true_file: True.csv
  fake_file: Fake.csv
  min_text_len: 50
  val_size: 0.10
  test_size: 0.20
  drop_cross_label_conflicts: true

features:
  title:
    ngram_max: 2
    min_df: 2
    max_features: 20000
    stop_words: english
    sublinear_tf: true
  body:
    ngram_max: 2
    min_df: 3
    max_features: 50000
    stop_words: english
    sublinear_tf: true

classical:
  C_grid: [0.1, 1.0, 10.0]
  cv_folds: 5
  alpha_grid: [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
               0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
  threshold: 0.5

bilstm:
  max_title_len: 32
  max_body_len: 300
  vocab_size: 30000
  min_freq: 2
  embed_dim: 128
  hidden_dim: 128
  dropout: 0.2
  batch_size: 64
  lr: 0.001
  epochs: 10
  patience: 2

distilbert:
  model_name: distilbert-base-uncased
  max_title_len: 32
  max_body_len: 256
  head_tokens: 198
  tail_tokens: 56
  batch_size: 16
  lr: 0.00002
  epochs: 3
  warmup_ratio: 0.1
  weight_decay: 0.01
  fp16: true

eval:
  bootstrap_n: 1000
  ci: 0.95
  positive_label: 1
```

### 4.2 配置加载

`classical_lr.yaml`、`classical_svm.yaml`、`bilstm.yaml`、`distilbert.yaml`、`shortcut.yaml` 只需要写**覆盖项**（例如 `bilstm.epochs: 5`），加载时与 `base.yaml` 递归合并；不要复制整份 base。

`fakenews/config.py`：

```python
from pathlib import Path
import yaml

class Config(dict):
    """dict + 属性访问：cfg.data.min_text_len"""
    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))

def merge(base: dict, override: dict) -> dict:
    """递归合并，override 优先。"""
    ...
```

- 所有脚本接受 `--config configs/base.yaml`，可选 `--override key=value`（用 `merge` 覆盖）。
- 每个 run 保存 `results/logs/{run_id}/config.yaml` 和 `git rev-parse HEAD`。

---

## 5. 模块实现规格

### 5.1 `fakenews/utils.py`

```python
def seed_everything(seed: int = 42) -> None:
    """random / numpy / torch / CUDA 全部固定；torch 开启 deterministic（可选）。"""

def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """同时输出到 stdout 和文件。"""

def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str: ...

def ensure_dir(path: Path) -> Path: ...

def save_json(obj, path: Path) -> None: ...

def load_json(path: Path) -> dict: ...

@contextmanager
def timer() -> Iterator[float]:
    """yield 一个 list，退出时写入耗时秒数。"""
```

### 5.2 `fakenews/data/download.py`

```python
DATASET_SLUG = "clmentbisaillon/fake-and-real-news-dataset"

def download_dataset(raw_dir: Path, force: bool = False) -> tuple[Path, Path]:
    """
    1) 若 True.csv / Fake.csv 已存在且非空，直接返回；
    2) 尝试 `kaggle datasets download -d ... -p raw_dir --unzip`；
    3) 失败则抛出 RuntimeError，提示手动下载 archive.zip 并解压；
    4) 校验列名、行数 > 0，写 CHECKSUMS.txt。
    """

def validate_raw(true_path: Path, fake_path: Path) -> dict:
    """返回 rows、columns、dtypes、missing、sha256。"""
```

### 5.3 `fakenews/data/eda.py`

```python
def run_eda(df: pd.DataFrame, out_dir: Path) -> dict:
    """
    输出 results/eda_report.md + 图：
    - 行数、缺失、空文本、短文本
    - subject × label 交叉表
    - date 范围与按月分布
    - 文本长度分布（title/body，按 label）
    - 重复 key 数、跨标签冲突 key 数
    - 类别比例
    """
```

要求：`subject` 交叉表要能直接看出真假取值不重叠。

### 5.4 `fakenews/data/clean.py`

```python
HTML_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    """去 HTML、去控制字符、压缩空白、strip；不删 URL、不删 (Reuters)。
    非字符串（NaN/None）返回空串。"""

def make_dup_key(title: str, text_clean: str) -> str:
    """lower + 空白归一化 + title + '\\n' + text_clean。"""

def load_raw(raw_dir: Path) -> pd.DataFrame:
    """读两个 CSV，加 label（True=0, Fake=1）、source、id；concat 后 reset_index。
    title/text 先 fillna('')，避免后续 str 操作报错。"""

def clean_dataset(df: pd.DataFrame, min_len: int = 50,
                  drop_conflicts: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    顺序很重要：
    1) text_clean = clean_text(text)
    2) 丢掉 text_clean 长度 <= min_len
    3) dup_key = make_dup_key(title, text_clean)
    4) 统计跨标签冲突：groupby(dup_key).label.nunique() > 1
       - drop_conflicts=True 时整组删除
    5) 精确去重：duplicated(keep='first')
    6) 返回 df + stats
    """

def save_clean(df: pd.DataFrame, path: Path) -> None:
    """写 parquet；列顺序固定；id 唯一。"""
```

`stats` 必须包含：

```json
{
  "loaded_total": 0,
  "loaded_true": 0,
  "loaded_fake": 0,
  "missing_title": 0,
  "missing_text": 0,
  "short_text_dropped": 0,
  "conflict_keys": 0,
  "conflict_rows_dropped": 0,
  "duplicate_rows_dropped": 0,
  "remaining_total": 0,
  "remaining_true": 0,
  "remaining_fake": 0
}
```

验收：`loaded_total - short_text_dropped - conflict_rows_dropped - duplicate_rows_dropped == remaining_total`。

### 5.5 `fakenews/data/split.py`

```python
def split_dataset(df: pd.DataFrame, val_size: float, test_size: float,
                  seed: int) -> dict[str, pd.DataFrame]:
    """
    先分 test，再从剩余分 val：
    train_tmp, test = train_test_split(df, test_size=test_size, stratify=df.label, random_state=seed)
    train, val = train_test_split(train_tmp, test_size=val_size/(1-test_size),
                                  stratify=train_tmp.label, random_state=seed)
    """

def assert_split_invariants(train, val, test) -> None:
    """id 无交集；并集大小 == 总数；每份类别比例与全集差 <= 0.005；无空文本。"""

def save_splits(splits: dict[str, pd.DataFrame], out_dir: Path,
                manifest_path: Path) -> dict:
    """写 parquet + manifest（seed、sizes、label_counts、sha256、pandas_version、created_at）。"""
```

### 5.6 `fakenews/features/tfidf.py`

```python
@dataclass
class TfidfMatrices:
    title_train: csr_matrix
    title_val: csr_matrix
    title_test: csr_matrix
    body_train: csr_matrix
    body_val: csr_matrix
    body_test: csr_matrix
    title_vec: TfidfVectorizer
    body_vec: TfidfVectorizer

def build_tfidf(train, val, test, cfg, text_col: str = "text_clean") -> TfidfMatrices:
    """
    两个 vectorizer 都只在 train 上 fit_transform；val/test 只 transform。
    text_col 可传 "text_masked" 给来源线索实验。
    dtype 转 float32，节省内存。
    """
```

- title：`ngram_range=(1, cfg.ngram_max)`, `min_df`, `max_features`, `sublinear_tf`, `stop_words`。
- body：同上，但 `min_df=3`、`max_features=50000`。
- 保存 vectorizer：`joblib.dump` 到 `models/tfidf_{title,body}.joblib`。

### 5.7 `fakenews/features/shortcut.py`

```python
def extract_shortcut_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    只用 text_raw / title，不用 text_clean，不用 subject/date。
    返回与 df 等长的数值 DataFrame，列名固定。
    """

FEATURES = [
    "has_reuters_dateline", "has_url", "has_via", "has_getty",
    "has_follow_us", "has_our_standards", "starts_with_city_dateline",
    "title_has_question", "title_has_exclamation",
    "exclamation_count", "question_count",
    "uppercase_ratio", "all_caps_word_count",
    "title_len", "text_len", "digit_ratio", "quote_count",
    "body_has_url",
]
```

规则示例（`text_raw`/`title` 先 `fillna("")`）：

```python
has_reuters_dateline = text_raw.str.contains(r"\(Reuters\)", case=False, regex=True)
has_url = text_raw.str.contains(r"https?://|www\.", case=False, regex=True)
has_via = text_raw.str.contains(r"\bvia\b", case=False, regex=True)
has_getty = text_raw.str.contains(r"getty", case=False, regex=True)
has_follow_us = text_raw.str.contains(r"follow us|subscribe", case=False, regex=True)
has_our_standards = text_raw.str.contains(r"our standards", case=False, regex=True)
starts_with_city_dateline = text_raw.str.contains(r"^[A-Z][A-Za-z .'\-]{1,40}\(Reuters\)", regex=True)
uppercase_ratio = text_raw.map(lambda s: sum(c.isupper() for c in s) / max(sum(c.isalpha() for c in s), 1))
all_caps_word_count = text_raw.map(lambda s: sum(w.isupper() and len(w) > 1 for w in s.split()))
```

### 5.8 `fakenews/features/masking.py`

```python
DATELINE_RE = re.compile(r"[A-Za-z .'\-/]{0,60}\(Reuters\)\s*[-–—]\s*")
URL_RE = re.compile(r"https?://\S+|www\.\S+")

def mask_body(text: str) -> str:
    """只用于正文；先去 dateline，再去 URL；不碰标题。"""

def mask_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """复制 df，新增 text_masked 列；text_clean 不变。"""

def count_hits(df: pd.DataFrame) -> dict:
    """按 label 统计 dateline/url 命中数，用于 sanity check。"""
```

### 5.9 `fakenews/models/fusion.py`

```python
class GatedFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(2 * dim, dim)

    def forward(self, h_t: Tensor, h_b: Tensor) -> tuple[Tensor, Tensor]:
        g = torch.sigmoid(self.gate(torch.cat([h_t, h_b], dim=-1)))
        return g * h_t + (1 - g) * h_b, g.mean(dim=-1)
```

```python
class ConcatFusion(nn.Module):
    def forward(self, h_t, h_b):
        return torch.cat([h_t, h_b], dim=-1), None
```

### 5.10 `fakenews/models/classical.py`

```python
@dataclass
class BranchModel:
    vectorizer: TfidfVectorizer
    clf: BaseEstimator
    name: str

    def predict_proba(self, texts: pd.Series) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.clf.predict_proba(X)[:, 1]

class DualTextModel:
    def __init__(self, model_type: Literal["lr", "svm"], cfg: Config,
                 text_col: str = "text_clean"):
        ...

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "DualTextModel":
        """
        1) 在 train 上 fit 两个 vectorizer；
        2) 对每个分支用 GridSearchCV(C) 在 train 上做 5 折 CV；
        3) 用最优 C 在 train 上 fit 分支；
        4) 在 val 上算 p_title / p_body，网格搜 α；
        5) 保存 α、C、阈值。
        """

    def predict_proba(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """返回 {'title': ..., 'body': ..., 'fusion': ...}。"""

    def refit_full(self, train_val: pd.DataFrame) -> "DualTextModel":
        """用 train+val 重新 fit 两个分支；α 和 C 不变；用于最终 test。"""

    def save(self, out_dir: Path) -> None: ...
    @classmethod
    def load(cls, out_dir: Path) -> "DualTextModel": ...
```

细节：

- LR：`LogisticRegression(C=best_C, max_iter=1000, solver="liblinear")`。
- SVM：`CalibratedClassifierCV(LinearSVC(C=best_C), cv=5)`，这样有 `predict_proba`；`cv="prefit"` 在新版 sklearn 已弃用，不要用。
- `text_col` 默认 `text_clean`；来源线索实验传 `text_masked`，title 列保持不变。
- early concat 条件：`hstack([X_title, X_body])` + 单分支 LR；单独一个方法 `fit_concat` / `predict_concat`。
- `refit_full` **只重训分类器**，vectorizer 保持 train 上 fit 的版本；不要在 train+val 上重新 fit vectorizer，否则 α/C 与最终模型失配。
- α 选择：`argmax F1`；并列时取更接近 0.5 的 α（写死规则）。
- 阈值：默认 0.5；另在 val 上选 F1 最大阈值，存 `threshold_tuned`。

### 5.11 `fakenews/models/bilstm.py`

```python
class Vocab:
    """特殊 token：<pad>=0、<unk>=1；只用 train 的 title+body 构建，按频次截断到 max_size。"""
    def __init__(self, stoi: dict[str, int]): ...
    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 2,
              max_size: int = 30000) -> "Vocab": ...
    def encode(self, text: str, max_len: int) -> list[int]: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "Vocab": ...

class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(4 * hidden_dim, hidden_dim)  # mean+max，各 2*hidden

    def forward(self, ids: Tensor, lengths: Tensor) -> Tensor:
        emb = self.dropout(self.embed(ids))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        mask = (ids != 0).unsqueeze(-1)
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        max_ = out.masked_fill(~mask, -1e4).max(1).values
        return self.proj(torch.cat([mean, max_], dim=-1))

class DualBiLSTM(nn.Module):
    def __init__(self, vocab_size, cfg):
        ...
        self.title_enc = BiLSTMEncoder(...)
        self.body_enc = BiLSTMEncoder(...)
        self.fusion = GatedFusion(cfg.hidden_dim)
        self.classifier = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, title_ids, title_lens, body_ids, body_lens):
        h_t = self.title_enc(title_ids, title_lens)
        h_b = self.body_enc(body_ids, body_lens)
        h, gate = self.fusion(h_t, h_b)
        return {"logit": self.classifier(h).squeeze(-1), "gate": gate}
```

`concat` 条件用单个 `BiLSTMEncoder` 处理 `[title; body]` 拼接后的 ≤300 序列，单独写 `ConcatBiLSTM`；`fusion` 才用 `DualBiLSTM`。两者必须用同一词表和同一训练协议。

### 5.12 `fakenews/models/distilbert.py`

```python
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput

@dataclass
class DualOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    gate: torch.Tensor | None = None

class DualDistilBert(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.fusion = GatedFusion(hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, title_ids, title_mask, body_ids, body_mask):
        # 两次 forward：title 32 token、body 256 token；title 开销很小
        h_t = self.encoder(input_ids=title_ids, attention_mask=title_mask).last_hidden_state[:, 0]
        h_b = self.encoder(input_ids=body_ids, attention_mask=body_mask).last_hidden_state[:, 0]
        h, gate = self.fusion(h_t, h_b)
        logits = self.classifier(self.dropout(h)).squeeze(-1)
        return DualOutput(logits=logits, gate=gate)
```

> 不要试图把 title/body 沿 batch 维拼成一次 forward：两者序列长度不同（32 vs 256），必须先 padding 到同长才能拼，反而更慢。两次 forward 的总开销约为单编码的 1.1–1.2 倍。

`concat` 条件不要用 `DualDistilBert` 硬拼：单独实现 `ConcatDistilBert`，输入 `[CLS] title [SEP] body [SEP]`，总长 288（title 32 + body 253 + 3 个特殊 token）；`fusion` 条件才用上面的双编码。

用 `Trainer` 时，自定义 `compute_loss`，避免依赖模型返回 `loss`：

```python
class DualTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.binary_cross_entropy_with_logits(outputs.logits, labels.float())
        return (loss, outputs) if return_outputs else loss
```

- `TrainingArguments(..., remove_unused_columns=False)`，否则 Trainer 可能丢掉 `title_ids` 等自定义列。
- `compute_metrics` 里对 `predictions` 做 `sigmoid`，再调用 `compute_metrics`。

### 5.13 `fakenews/data/dataset.py`

```python
class PairDataset(Dataset):
    """返回 dict：title_ids/title_mask/body_ids/body_mask（BERT）
    或 title_ids/title_lens/body_ids/body_lens（Bi-LSTM）。"""

def make_distilbert_dataset(df, tokenizer, cfg) -> Dataset:
    """
    title：截断到 32，正常加 [CLS]/[SEP]。
    body：head 198 + tail 56 = 254 个内容 token，加 [CLS]/[SEP] 正好 256。
    """

def encode_body(tokenizer, text: str, head: int = 198, tail: int = 56,
                max_len: int = 256) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) > head + tail:
        ids = ids[:head] + ids[-tail:]
    ids = [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]
    ids = ids[:max_len]
    return ids + [tokenizer.pad_token_id] * (max_len - len(ids))

def make_bilstm_dataset(df, vocab, cfg) -> Dataset:
    """title/body 分别 encode 到固定长度；返回 lengths。"""

def make_loader(dataset, batch_size, shuffle, num_workers=0) -> DataLoader:
    ...
```

### 5.14 `fakenews/train/common.py`

```python
def make_run_id(model: str, condition: str, seed: int) -> str:
    return f"{model}_{condition}_seed{seed}_{datetime.now():%Y%m%d_%H%M%S}"

def save_predictions(df_ids, y_true, y_prob, meta: dict, path: Path) -> None: ...

def append_metrics(row: dict, path: Path) -> None:
    """不存在则写表头；存在则 append。"""

def compute_git_commit() -> str: ...

def snapshot_config(cfg, out_dir: Path) -> str:
    """写 config.yaml，返回 sha256。"""
```

### 5.15 `fakenews/train/classical.py`

CLI：

```bash
python -m fakenews.train.classical --config configs/base.yaml --model lr
python -m fakenews.train.classical --config configs/base.yaml --model svm
```

流程：

1. 读取 `articles_clean.parquet` 和 `train/val/test`。
2. `DualTextModel.fit(train, val)`。
3. 计算 val 的 title/body/fusion 指标；写 predictions。
4. `refit_full(train+val)`；只在这里读取 test，计算 test 指标；写 predictions。
5. 保存模型、α、阈值、metrics 行。

条件：`title_only`、`body_only`、`concat`、`fusion` 全部输出。

### 5.16 `fakenews/train/bilstm.py`

CLI：

```bash
python -m fakenews.train.bilstm --config configs/bilstm.yaml --seed 42 --condition fusion
```

流程：

1. 建词表（只用 train）。
2. 建 Dataset/DataLoader。
3. 训练：BCEWithLogits、Adam、grad clip 1.0、early stop on val F1。
4. 保存 best state dict、vocab、config、metrics。
5. 在 test 上评估一次（只有 `--final` 时）。
6. `condition` 支持 `title_only` / `body_only` / `concat` / `fusion`：
   - `title_only`：`h = h_t`，gate 固定 1；
   - `body_only`：`h = h_b`，gate 固定 0；
   - `concat`：单个 `BiLSTMEncoder` 处理 `[title; body]`；
   - `fusion`：`DualBiLSTM` + 门控。

### 5.17 `fakenews/train/distilbert.py`

CLI：

```bash
python -m fakenews.train.distilbert --config configs/distilbert.yaml --seed 42 --condition fusion
```

- 用 `transformers.Trainer` + 自定义 `DataCollator`。
- `TrainingArguments`：

```python
TrainingArguments(
    output_dir=str(out_dir),
    learning_rate=cfg.lr,
    num_train_epochs=cfg.epochs,
    per_device_train_batch_size=cfg.batch_size,
    per_device_eval_batch_size=cfg.batch_size,
    warmup_ratio=cfg.warmup_ratio,
    weight_decay=cfg.weight_decay,
    fp16=cfg.fp16 and torch.cuda.is_available(),
    eval_strategy="epoch",          # 旧版写 evaluation_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    remove_unused_columns=False,
    save_total_limit=1,
    seed=seed,
    report_to="none",
)
```

> `eval_strategy` 需要 `transformers>=4.41`；如果锁定旧版，改回 `evaluation_strategy`。

- `compute_metrics` 用 `fakenews.eval.metrics.compute_metrics`。Trainer 传入的是 logits，记得先 sigmoid：

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-logits.squeeze()))
    return compute_metrics(labels, probs)
```

- 先跑 `concat` 1 个 seed，再跑 `fusion` 1 个 seed；确认 pipeline 后再 3 seeds。

### 5.18 `fakenews/train/shortcut.py`

CLI：

```bash
python -m fakenews.train.shortcut --config configs/shortcut.yaml
```

- 提特征 → 5 折 CV 选 C → train/val/test 评估。
- 输出 `results/shortcut.json` + predictions。

### 5.19 `fakenews/eval/metrics.py`

```python
def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """
    返回 precision/recall/f1/accuracy/roc_auc/pr_auc/tp/fp/fn/tn/n。
    正类 = 1（fake）。单类时 roc_auc/pr_auc 返回 nan。
    """

def tune_threshold(y_true, y_prob, grid=np.arange(0.05, 0.96, 0.01)) -> tuple[float, float]:
    """返回 (best_threshold, best_f1)。"""

def mcnemar(y_true, pred_a, pred_b) -> dict:
    """返回 b, c, chi2, p_value；连续性校正。"""

def brier_score(y_true, y_prob) -> float:
    """mean((y_prob - y_true) ** 2)。"""

def reliability_curve(y_true, y_prob, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """返回每个 bin 的平均预测概率与经验正类率。"""
```

### 5.20 `fakenews/eval/bootstrap.py`

```python
def bootstrap_f1(y_true, y_prob, threshold=0.5, n=1000, alpha=0.05,
                 seed=42, stratified=True) -> tuple[float, float]:
    """按类别分层重采样；跳过只有单类的 replicate；返回 percentile CI。"""

def paired_bootstrap_delta(y_true, prob_a, prob_b, threshold=0.5,
                           n=1000, seed=42) -> dict:
    """同一批样本上 F1(a)-F1(b) 的 CI；用于 fusion vs concat、masked vs original。"""
```

### 5.21 `fakenews/eval/evaluate.py`

- 扫描 `results/predictions/*.parquet`，按 `run_id` 分组。
- 对 val：算 0.5 与 tuned threshold；对 test：用 val 的 threshold。
- 写 `results/metrics.csv`（追加/去重）。
- 生成 `results/summary.md`（按 model/condition 的 mean±std）。
- 断言：同一 `(model, condition, seed)` 的 test 只出现一次。

### 5.21b `fakenews/eval/error_analysis.py`

```python
def run_error_analysis(preds: pd.DataFrame, df: pd.DataFrame,
                       out_dir: Path, k: int = 20) -> None:
    """
    1) 按 fusion 概率取 FP / FN 各 k 条；
    2) 对每条附 title、text 前 200 字、p_title、p_body、p_fusion；
    3) 统计 title-only 与 body-only 预测不一致的样本数；
    4) 写 results/error_analysis.md。
    """
```

- `preds` 至少包含 `id, y_true, y_prob, condition`；如果有多条件，先 pivot 成宽表。
- 输出是给人看的 Markdown 表格，不要只打印到 stdout。

### 5.22 `fakenews/experiments/source_cue.py`

```python
def run_source_cue(cfg, model: str = "lr") -> dict:
    """
    1) 读 train/val/test；
    2) 每份加 text_masked；
    3) 用 masked 文本重训 LR fusion（或最佳神经）；
    4) 在 test 上比较 original vs masked 的 F1/PR-AUC；
    5) 分别输出 title_only/body_only/fusion 的 ΔF1；
    6) 保存 results/source_cue.json + 逐样本预测。
    """
```

- 超参、α 沿用原配置；不要在 masked 上重新调参。
- 统计命中数：dateline/url 各多少条，按 label 分。
- 特征审计：`export_top_features(model, k=50)` → `results/feature_audit.csv`，列 `branch, feature, coef, category`，`category` 留空，人工填 `source/format/content`。

### 5.23 `fakenews/experiments/temporal.py`

- 解析 `date`（`errors="coerce"`），选 cutoff（默认 `2017-06-01`）。
- train = date < cutoff；test = date >= cutoff；val 从 train 里再切 10%。
- LR fusion；输出类别比例与 F1。
- 输出 `results/temporal.json`。

### 5.24 `fakenews/experiments/cross_dataset.py`

- 读 LIAR TSV（`liar_dataset/train.tsv`，列：id, label, statement, ...）。
- 映射：`true/mostly-true/half-true` → 0；`false/pants-fire` → 1；`barely-true` 丢弃。
- 用 ISOT 训好的 LR fusion 直接预测；输出 P/R/F1 与样本数。
- 输出 `results/cross_dataset.json`。
- 说明：这是 domain shift 压力测试，不是主结果。

### 5.25 `fakenews/experiments/near_dup.py`

- 5-gram shingles + `datasketch.MinHashLSH`（threshold=0.8）。
- 并查集聚类；按簇做 `GroupShuffleSplit`（70/10/20）。
- 用 LR fusion 对比 random split vs group split。
- 输出 `results/near_dup.json`。
- 可选依赖：`datasketch`；没有就跳过并打印提示。

### 5.25b `fakenews/experiments/chunking.py`（可选）

- 只对 DistilBERT 的 body 做：256 token、stride 128 切窗；每个窗编码后 mean/max pool。
- 取 test 的 5k 子集、1 个 seed；和 head+tail 版本比 F1/PR-AUC。
- 输出 `results/chunking.json`；涨幅 <0.2 F1 就保留 head+tail。
- 这是可选实验，不要阻塞主线。

### 5.26 `fakenews/viz/figures.py`

函数与输出：

| 函数 | 输出 | 内容 |
|---|---|---|
| `plot_class_distribution(df, out)` | `fig_class_distribution.pdf` | 类别 + subject 分布 |
| `plot_length_distribution(df, out)` | `fig_length.pdf` | title/body 长度 |
| `plot_model_bars(metrics, out)` | `fig_model_bars.pdf` | 四模型 F1 + CI |
| `plot_fusion_ablation(metrics, out)` | `fig_fusion_ablation.pdf` | title/body/concat/fusion |
| `plot_confusion(cm, out)` | `fig_confusion.pdf` | 最佳模型 |
| `plot_delta_f1(source_cue, out)` | `fig_delta_f1.pdf` | masked vs original |
| `plot_top_features(audit, out)` | `fig_top_features.pdf` | top-20 权重 |
| `plot_alpha_curve(alphas, out)` | `fig_alpha.pdf` | α vs val F1 |

风格：`matplotlib`，`figsize=(6,4)`，`dpi=200`，同时存 PDF 和 PNG；不要用截图。

---

## 6. 实验矩阵

### 6.1 必做

| 实验 | 模型 | 条件 | seeds | 输出 |
|---|---|---|---|---|
| 主对比 | LR / SVM | title_only, body_only, concat, fusion | 42–44 | metrics + preds |
| 主对比 | Bi-LSTM / DistilBERT | concat, fusion | 42–44 | metrics + preds |
| 四路消融 | LR | title/body/concat/fusion | 42 | `fusion.json` |
| shortcut | LR | 18 个手工特征 | 42 | `shortcut.json` |
| 来源线索 | LR fusion | original vs masked | 42 | `source_cue.json` |
| 来源线索 | best neural | original vs masked | 42 | `source_cue.json` |
| 错误分析 | best model | val | 42 | `error_analysis.md` |

### 6.2 可做（看余力）

| 实验 | 说明 | 输出 |
|---|---|---|
| 按时间划分 | LR fusion，cutoff 2017-06 | `temporal.json` |
| 跨数据集 | LIAR 零样本 | `cross_dataset.json` |
| 近重复分组 | MinHash + group split | `near_dup.json` |
| chunking | DistilBERT，5k 子集，1 seed | `chunking.json` |
| 相似度特征 | `cos(h_t,h_b)` 拼进融合 | 并入 metrics |
| 校准 | reliability + Brier | `calibration.json` |

---

## 7. 实施顺序与完成标准

> 严格按顺序；每步的“完成标准”通过后再进入下一步。

### 阶段 0：仓库骨架

- 建目录、`pyproject.toml`、`requirements.txt`、`.gitignore`、`Makefile`、CI。
- `pip install -e ".[dev]"` 成功；`ruff check .` 通过；`pytest` 通过（空测试也算）。

**完成标准**：GitHub 上能 clone，README 能按步骤装好环境。

### 阶段 1：数据下载与 EDA

- 实现 `download.py`、`eda.py`。
- 跑 `python -m fakenews.data.download --config configs/base.yaml`，再 `python -m fakenews.data.eda ...`。
- 检查 `results/eda_report.md`：行数、subject 交叉表、重复数、冲突数。

**完成标准**：原始文件只读；EDA 能回答“多少行、多少重复、subject 是否泄露、真假时间范围”。

### 阶段 2：清洗与划分

- 实现 `clean.py`、`split.py`、`tests/test_clean.py`、`tests/test_split.py`。
- 产出 `articles_clean.parquet`、`train/val/test.parquet`、`split_manifest.json`。
- 跑测试：无重复 key、无跨标签冲突、三份无交集、比例正确。

**完成标准**：`clean_log.json` 的数字能加总；`assert_split_invariants` 通过。

### 阶段 3：TF-IDF + 经典模型

- 实现 `tfidf.py`、`classical.py`、`train/classical.py`。
- 先跑 LR 的 title/body/fusion；再跑 concat；再跑 SVM。
- 产出 `results/metrics.csv`、`results/predictions/`、`results/fusion.json`。

**完成标准**：四路指标都有；α 在 val 上选出；vectorizer 只在 train 上 fit；test 只读一次。

### 阶段 4：shortcut baseline

- 实现 `shortcut.py`、`train/shortcut.py`。
- 18 个特征；5 折 CV；test 一次。

**完成标准**：`results/shortcut.json` 有 F1/PR-AUC；和完整模型的对比有 JSON 输出。

### 阶段 5：Bi-LSTM

- 实现 `dataset.py`（Vocab + Dataset）、`bilstm.py`、`train/bilstm.py`。
- 先跑 fusion 1 个 seed；再跑 concat；再补 3 seeds。
- 保存 vocab、state_dict、metrics、preds。

**完成标准**：val F1 ≥ 0.97；gate 有输出；title/body 分支可单独跑。

### 阶段 6：DistilBERT

- 实现 `distilbert.py`、`train/distilbert.py`。
- 先跑 concat 1 seed，再 fusion 1 seed；确认显存/时间后补 3 seeds。
- 保存 checkpoint、tokenizer、metrics、preds。

**完成标准**：T4 上 1 seed ≤ 40 分钟；test 只用 val 选出的 checkpoint。

### 阶段 7：评估与统计

- 实现 `metrics.py`、`bootstrap.py`、`evaluate.py`、`error_analysis.py`。
- 汇总 `metrics.csv`；算 95% CI；McNemar；错误分析。

**完成标准**：每个结论都有 CI 或 p 值；同一配置的 test 只出现一次。

### 阶段 8：来源线索

- 实现 `masking.py`、`source_cue.py`。
- 在 train/val/test 上分别 mask；重训 LR fusion；评估；特征审计。
- 产出 `source_cue.json`、`feature_audit.csv`。

**完成标准**：命中数 sanity check 通过；ΔF1 + CI 有数字；title-only 几乎不变。

### 阶段 9：图表与文档

- 实现 `figures.py`；生成 6 张图。
- 更新 `README.md`、`docs/data_card.md`、`docs/protocol.md`。

**完成标准**：新机器按 README 能复现基线；图都是矢量图。

### 阶段 10：可选项

- 时间划分、LIAR、近重复、chunking、校准。
- 每个都是独立脚本，不阻塞主线。

---

## 8. 测试计划

### 8.1 单元测试

| 文件 | 测什么 |
|---|---|
| `test_clean.py` | HTML 去除、空白压缩、短文本删除、dup_key 归一化 |
| `test_clean.py` | 冲突检测在 keep-first 之前；冲突行全部删除 |
| `test_split.py` | 无交集、并集完整、比例误差 <0.5% |
| `test_tfidf.py` | 只在 train 上 fit；val/test transform 后维度一致 |
| `test_fusion.py` | α=0/1 退化为单分支；gate 输出在 [0,1] |
| `test_metrics.py` | 已知混淆矩阵的 P/R/F1；单类 AUC 返回 nan |
| `test_bootstrap.py` | CI 包含点估计；固定 seed 可复现 |
| `test_masking.py` | dateline/URL 被删；title 不变；命中数按 label 统计 |

### 8.2 集成冒烟

`tests/fixtures/tiny.csv`：40 行（20 真/20 假），字段齐全，含 2 条重复、1 条冲突、1 条超短。

```bash
python -m fakenews.data.clean --input tests/fixtures/tiny.csv --output /tmp/tiny_clean.parquet
python -m fakenews.train.classical --config configs/base.yaml --smoke
pytest -q
```

- `--smoke` 只跑 1 个 seed、1 个 C、1 个 α，2 分钟内完成。
- CI 只跑 `-m "not slow"`；神经模型测试标 `@pytest.mark.slow`，本地跑。

### 8.3 CI

`.github/workflows/ci.yml`：

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: pytest -q -m "not slow"
```

---

## 9. 复现与工程规范

### 9.1 随机性

- 每个入口先 `seed_everything(seed)`。
- DataLoader：`worker_init_fn` 固定；`generator` 固定。
- `torch.backends.cudnn.deterministic = True`；`benchmark = False`。
- 同一 run 保存 seed 到 metrics。

### 9.2 运行记录

每个 run 目录：

```text
results/logs/{run_id}/
├── config.yaml
├── environment.txt        # pip freeze + python + CUDA
├── git_commit.txt
├── train.log
└── metrics.json
```

### 9.3 数据与模型版本

- `data/raw/CHECKSUMS.txt`：两个 CSV 的 sha256。
- `data/processed/split_manifest.json`：三份 parquet 的 sha256。
- 模型目录：`models/{model}/{run_id}/`，含 `state_dict.pt`、`vocab.json`、`tokenizer/`、`config.yaml`。

### 9.4 代码规范

- `ruff`：line-length 100，import 排序。
- 类型标注；公共函数写 docstring。
- 不提交 `data/`、`models/`、`results/predictions/`、`results/figures/`。
- 不在 notebook 里做正式实验；notebook 只做探索。

### 9.5 测试集纪律

- `test.parquet` 只读。
- 训练脚本默认不读 test；只有 `--final` 才读。
- `evaluate.py` 检查同一 `(model, condition, seed)` 的 test 记录是否已存在；已存在则拒绝重复评估。
- **例外**：`source_cue`、`temporal`、`cross_dataset` 是预注册分析，允许在 test 上评估一次；但它们不能反过来用于调参、选模型或改 α。

### 9.6 Git 工作流

- `main` 保持可运行；每个功能开 `feat/...` 分支，PR 合并。
- 建议的 tag：`v0.1-data`（清洗+划分）、`v0.2-baselines`（LR/SVM）、`v1.0-final`。
- 不要在 PR 里提交数据、模型权重、预测文件；只提交代码、配置、测试和文档。
- 每个 PR 至少包含：改了什么、怎么跑、预期输出、是否影响 test 纪律。
- commit message 用 `feat: ...` / `fix: ...` / `exp: ...`，方便回溯实验。

---

## 10. 算力与降级

| 阶段 | 硬件 | 预计 | 降级 |
|---|---|---|---|
| TF-IDF + LR/SVM | 笔记本 CPU | 分钟级 | 无 |
| Bi-LSTM | CPU/GPU | 20–60 min/epoch | batch 64、max_body 200、3 epochs |
| DistilBERT | Colab T4 | 20–40 min/seed | max_body 192、1 seed、grad accumulation |
| 3 seeds 神经 | T4 | 2–3 h | 只对 fusion 跑 3 seeds，concat 1 seed |

- Colab 断开：checkpoint 存 Google Drive；`resume_from_checkpoint`。
- 显存不足：`batch_size=8` + `gradient_accumulation_steps=2`；`fp16=True`。

---

## 11. 常见坑

1. **行数对不上**：CSV 解析差异；记录 pandas 版本，以实际 `shape` 为准。
2. **test 泄漏**：先去重后划分；`subject` 不进特征；test 只读一次。
3. **title/body 错位**：所有 DataFrame 用 `id` 对齐；合并前 `assert (df.id == other.id).all()`。
4. **TF-IDF 在 val 上 fit**：只允许 `fit_transform(train)`；val/test 只能 `transform`。
5. **SVM 没有概率**：用 `CalibratedClassifierCV(LinearSVC(), cv=5)`；不要用已弃用的 `cv="prefit"`。
6. **α 在 test 上调**：α 只在 val 上选；test 只评估。
7. **DistilBERT 截断**：head+tail；同时跑“TF-IDF 只看前 256 token”的对照。
8. **Bi-LSTM 变长序列**：`pack_padded_sequence`；`lengths` 必须在 CPU。
9. **类别标签**：`True.csv` = 0，`Fake.csv` = 1；正类 = 1。
10. **空标题**：`fillna("")`；不要 drop，标题可能仍有信号（空标题本身是特征）。
11. **Colab 路径**：不要写 `C:\...`；用 `Path` + config 相对路径。
12. **重复评估**：`evaluate.py` 做幂等检查。

---

## 12. 新会话启动清单

按顺序执行：

```bash
git clone <repo> && cd fake-news-detection
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"

# 1) 数据
python -m fakenews.data.download --config configs/base.yaml
python -m fakenews.data.eda --config configs/base.yaml

# 2) 清洗 + 划分
python -m fakenews.data.clean --config configs/base.yaml
python -m fakenews.data.split --config configs/base.yaml
pytest -q tests/test_clean.py tests/test_split.py

# 3) 基线
python -m fakenews.train.classical --config configs/base.yaml --model lr
python -m fakenews.train.classical --config configs/base.yaml --model svm
python -m fakenews.train.shortcut --config configs/shortcut.yaml

# 4) 神经（先 1 seed）
python -m fakenews.train.bilstm --config configs/bilstm.yaml --seed 42 --condition fusion
python -m fakenews.train.distilbert --config configs/distilbert.yaml --seed 42 --condition fusion

# 5) 评估
python -m fakenews.eval.evaluate --config configs/base.yaml
```

第一轮只需要跑到第 3 步；确认 `metrics.csv` 有 LR/SVM 的 title/body/concat/fusion 四路结果，再继续。

**第一个 PR 建议只包含**：仓库骨架 + `utils/config` + `clean/split` + 测试。这样 review 成本最低，也最容易回滚。

---

## 13. 附录

### A. `pyproject.toml` 依赖

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "fakenews"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "pandas>=2.0",
  "numpy>=1.24",
  "scipy>=1.10",
  "scikit-learn>=1.3,<1.6",
  "pyarrow>=12",
  "pyyaml>=6",
  "matplotlib>=3.7",
  "tqdm>=4.66",
  "torch>=2.1",
  "transformers>=4.41",
  "joblib>=1.3",
]

[project.optional-dependencies]
dev = ["pytest>=7", "ruff>=0.4", "datasketch>=1.6"]

[tool.setuptools.packages.find]
include = ["fakenews*"]

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
markers = ["slow: tests that need torch or a GPU"]
```

> `scikit-learn<1.6` 是为了避免 `CalibratedClassifierCV(cv="prefit")` 的弃用问题；代码里用 `cv=5`，不依赖 `prefit`。

### B. `Makefile`

```make
install:
	pip install -e ".[dev]"

data:
	python -m fakenews.data.download --config configs/base.yaml
	python -m fakenews.data.clean --config configs/base.yaml
	python -m fakenews.data.split --config configs/base.yaml

classical:
	python -m fakenews.train.classical --config configs/base.yaml --model lr
	python -m fakenews.train.classical --config configs/base.yaml --model svm

shortcut:
	python -m fakenews.train.shortcut --config configs/shortcut.yaml

bilstm:
	python -m fakenews.train.bilstm --config configs/bilstm.yaml --seed 42 --condition fusion

distilbert:
	python -m fakenews.train.distilbert --config configs/distilbert.yaml --seed 42 --condition fusion

eval:
	python -m fakenews.eval.evaluate --config configs/base.yaml

figures:
	python -m fakenews.viz.figures --config configs/base.yaml

test:
	pytest -q

lint:
	ruff check .

smoke:
	python -m fakenews.train.classical --config configs/base.yaml --smoke

all: data classical shortcut
```

`requirements.txt` 由 `pip freeze > requirements.txt` 生成，作为锁定版本；日常安装仍用 `pip install -e ".[dev]"`。

### B2. `scripts/run_all.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

python -m fakenews.data.download --config configs/base.yaml
python -m fakenews.data.clean    --config configs/base.yaml
python -m fakenews.data.split    --config configs/base.yaml

python -m fakenews.train.classical --config configs/base.yaml --model lr
python -m fakenews.train.classical --config configs/base.yaml --model svm
python -m fakenews.train.shortcut  --config configs/shortcut.yaml

python -m fakenews.eval.evaluate --config configs/base.yaml
python -m fakenews.viz.figures   --config configs/base.yaml
```

`scripts/smoke.sh` 只跑 200 行子集 + 1 个 C + 1 个 α，用于 CI 或本地快速回归。

### C. README 必须包含

- 项目一句话 + 数据来源 + 许可证。
- 环境安装（Python 版本、`pip install -e`）。
- 复现命令（`make data && make classical && make eval`）。
- 预期输出（`metrics.csv` 的列、图的位置）。
- 硬件要求（CPU / Colab T4）与运行时长。
- 随机种子与划分 manifest 的位置。
- 已知限制（来源标签、单数据集、英文、截断）。

### D. Definition of Done（总）

- [ ] 新机器按 README 能跑出 LR/SVM 的 val/test 指标。
- [ ] `metrics.csv` 覆盖四个模型 × 四个条件（经典）/两个条件（神经）× 3 seeds。
- [ ] `predictions/` 有逐样本概率；`figures/` 有 6 张图。
- [ ] `source_cue.json` 有 ΔF1 + CI；`shortcut.json` 有 F1。
- [ ] `pytest` 与 `ruff` 通过；CI 绿。
- [ ] `test` 只被评估一次（`evaluate.py` 幂等检查）。
- [ ] README 写明复现命令、硬件、种子、限制。
