# ICAIC 2026 — Part 2: Data Preprocessing

> **Goal:** Convert unfamiliar raw training/validation/competition-test data into valid model inputs **without leakage**, and reuse the process across models. This guide generalizes beyond Titanic and Logistic Regression.

## 1. Core architecture

```text
Raw training dataframe X, labels y
                 |
       Split into train/validation   ← BEFORE FITTING TRANSFORMATIONS
                 |
    ┌────────────┴────────────┐
    |                         |
 Numerical columns      Categorical columns
    |                         |
 SimpleImputer          SimpleImputer
    |                         |
 StandardScaler*        OneHotEncoder*
    |                         |
    └────────────┬────────────┘
                 |
          ColumnTransformer
                 |
       make_pipeline(preprocessor, model)
                 |
       .fit(X_train, y_train)
       .predict(X_val)
```

`*` Depending on model and data, these steps may change or be omitted. In a full pipeline, **do not manually fit/transform validation data**; `.predict(X_val)` applies the fitted preprocessing automatically.

## 2. Split before learning preprocessing statistics

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["target"])
y = df["target"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Use `stratify=y` for suitable classification settings; usually omit for continuous regression. Use time-based or group-based validation when random splits would leak future/group information.

**Defining** `SimpleImputer()` before a split does not leak anything. **Fitting** it on held-out data does.

## 3. Fit vs transform vs predict — the key confusion

| Method | Applies to | Meaning |
|---|---|---|
| `transformer.fit(X_train)` | Imputer/scaler/encoder | Learn parameters (medians, means, categories) |
| `transformer.transform(X)` | Already-fitted transformer | Apply learned transformation to new rows |
| `transformer.fit_transform(X_train)` | Transformer | Fit and transform training rows |
| `estimator.fit(X_train, y_train)` | Supervised model/pipeline | Train model; for pipelines, fit preceding transformations too |
| `estimator.predict(X_val)` | Fitted model/pipeline | Transform validation internally and predict |
| `estimator.predict_proba(X_val)` | Supported classifiers | Class probabilities (not always available) |

**Separate preprocessing:**

```python
X_train_ready = preprocessor.fit_transform(X_train)
X_val_ready = preprocessor.transform(X_val)
model.fit(X_train_ready, y_train)
y_pred = model.predict(X_val_ready)
```

**End-to-end (preferred):**

```python
model = make_pipeline(preprocessor, LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
```

**Never:** `preprocessor.fit_transform(X_val)` after fitting on train. That relearns statistics/categories from validation and can change the representation.

## 4. Numerical preprocessing

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

numeric_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)
```

### Imputation strategies

| Strategy | Fits what? | Typical use |
|---|---|---|
| `"mean"` | Column average | Numerical columns without severe skew |
| `"median"` | Column median | Robust numerical baseline |
| `"most_frequent"` | Column mode | Categorical or discrete values |
| `"constant"` | User-defined fill value | Explicit sentinel/missing category |

```python
SimpleImputer(strategy="constant", fill_value=0)
SimpleImputer(strategy="median", add_indicator=True)  # flags training-observed missingness
```

A fully missing training column may be dropped by default; consider `keep_empty_features=True` if appropriate. If missingness itself predicts the outcome, consider indicator features. Avoid computing a median manually on train+validation together.

### Scaling

`StandardScaler` learns mean and standard deviation on training data:

\[z=(x-\mu_{train})/\sigma_{train}\]

Other choices: `MinMaxScaler` (fixed range), `RobustScaler` (robust to outliers), `PowerTransformer` or `QuantileTransformer` (distribution transformations), `FunctionTransformer` (custom logic). A log transform can help with strongly skewed nonnegative inputs; handle zeros/negative values and validate benefits.

**Scaling is recommended for:** Logistic Regression, SVM, KNN, regularized linear models, often PCA/clustering/neural nets. **Not generally necessary for:** decision trees, random forests, histogram/tree boosting such as XGBoost/LightGBM/CatBoost.

## 5. Categorical preprocessing

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

categorical_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)
```

- **One-hot:** `City=Toronto/Vancouver` becomes indicator columns; no artificial numeric order.
- **Ordinal:** useful for ordered categories (`low < medium < high`) but imposes a numeric representation; consider whether distance/ordering matches the model.
- **Unseen categories:** `handle_unknown="ignore"` maps unseen values to all-zero indicators for that original feature instead of crashing.
- **High cardinality:** many unique categories can create huge sparse matrices. Consider `min_frequency`, `max_categories`, hashing, careful target encoding, or native-categorical estimators.

```python
OneHotEncoder(handle_unknown="ignore", min_frequency=10)
OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # dense; beware memory
```

Most scikit-learn linear/SVM/KNN estimators cannot consume raw strings. Native categorical support exists in CatBoost, LightGBM, and suitably configured XGBoost, but each has input requirements. Missing string categories may need explicit sentinels for native categorical models.

## 6. `ColumnTransformer`: different transformations for different columns

This addresses a central confusion: **you can feed the entire original `X_train` DataFrame into `ColumnTransformer` even if you only want some features.**

```python
from sklearn.compose import ColumnTransformer

numeric_features = ["Age", "Fare", "SibSp", "Parch"]
categorical_features = ["Sex", "Embarked", "Pclass"]

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])
```

Each specification is `(step_name, transformer, column_names)`. The branches execute **in parallel on their selected columns**, and their results are concatenated into one feature matrix.

### What happens to omitted features?

**Default: `remainder="drop"`**, so `PassengerId`, `Name`, `Ticket`, `Cabin` and other unspecified columns are **excluded** from model inputs. This is valid feature selection, not an error.

Alternative:

```python
preprocessor = ColumnTransformer(
    [("num", numeric_pipeline, numeric_features),
     ("cat", categorical_pipeline, categorical_features)],
    remainder="passthrough"
)
```

This includes all other columns unchanged and is risky: raw strings, NaNs, identifiers or a mistakenly included target may reach the model. **Prefer explicit selected feature lists** for a competition baseline.

**Semantic types matter:** `Pclass` is stored as an integer but is an ordinal category; one-hot encoding is a reasonable baseline. `PassengerId` is an identifier despite numeric dtype. The target must *never* be an input feature.

**Dimensionality changes:** Four original columns can become seven or more processed columns after one-hot expansion. The final matrix may be a NumPy array or a sparse matrix, not the original DataFrame.

## 7. Combine preprocessor and estimator into one pipeline

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(preprocessor, LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
```

`model.fit(...)`: each preprocessing step learns its training statistics and transforms its input; the final estimator fits to processed training features. `model.predict(...)`: only transforms the validation features using learned statistics and predicts. **No manual `fit_transform` or `transform` required.**

### `Pipeline` vs `make_pipeline`

```python
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
```

`Pipeline` lets you name the steps explicitly; `make_pipeline` gives automatic names. Both support nested hyperparameter tuning:

```python
params = {"classifier__C": [0.1, 1.0, 10.0]}  # with named Pipeline above
# For make_pipeline(... LogisticRegression()): "logisticregression__C"
model.named_steps       # inspect step names
model.get_params().keys()
```

When comparing models, construct a **fresh** pipeline for each estimator or clone a preprocessing definition instead of relying on mutable objects shared between independently fitted pipelines.

## 8. Change preprocessing to fit the estimator

| Estimator family | Numeric missingness | Scaling | Categorical handling |
|---|---|---|---|
| Logistic/Ridge/Lasso | Impute | Recommended | Encode |
| SVM / KNN | Impute | Strongly recommended | Encode |
| Decision Tree | Often supports numeric NaNs in modern versions; otherwise impute | Unnecessary | Encode |
| Random Forest | Often supports numeric NaNs in modern versions; otherwise impute | Unnecessary | Encode |
| XGBoost | Native numeric missing support | Unnecessary | Encode or configure native categorical |
| LightGBM | Native missing support | Unnecessary | Encode or configure native categorical |
| CatBoost | Native numeric missing support | Unnecessary | Specify categorical features; represent categorical missingness appropriately |
| Neural networks | Often impute / mask | Usually normalize | Embed/encode as appropriate |

For tree ensembles, a useful starting numeric branch may be `SimpleImputer(strategy="median")` **without** a scaler. For native-categorical boosting, don't blindly reuse the one-hot pipeline; learn that library's data format.

## 9. Beyond basic tabular columns

| Inputs | Often-useful preprocessing |
|---|---|
| Date/time | Extract hour/day/month, elapsed time; avoid future leakage |
| Text | `TfidfVectorizer`, tokenizers, embeddings; not one-hot per entire sentence |
| Images | Resize, tensor conversion, model-specific normalization, train-only augmentation |
| Time series | Chronological split, lag/rolling features computed strictly from past values |
| High-cardinality categories | Hashing, grouping, native categorical, leak-safe target encoding |
| Highly skewed numbers | `log1p` where domain permits, robust transformations |
| Grouped / repeated observations | Group-aware validation; per-group aggregates must avoid leakage |

**Pipelines are a pattern, not a mandate:** sklearn's `ColumnTransformer` is great for ordinary tabular models; PyTorch datasets/transforms, deep-learning tokenizers, or native boosting interfaces may be more suitable elsewhere.

## 10. Data leakage rules

1. Split into appropriate training/validation folds **before** fitting imputers, scalers, encoders, feature selectors, PCA, etc.
2. During validation, `.transform(X_val)` but **never refit** on `X_val`.
3. Put preprocessing inside the pipeline passed to cross-validation so each fold fits its own statistics.
4. Avoid features created using the target or unavailable future outcomes.
5. If rows share a person/entity or depend on time, use the appropriate group/time split.
6. Handle target encoding with leakage-safe procedures, including proper within-fold training encodings.
7. Repeated experimentation on one validation set can overfit model *selection*; use CV/holdouts as needed.
8. After selecting your final approach, you can refit on all **permitted labeled training data** before predicting the competition's unlabeled test set.

## 11. Reusable tabular classification template

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train = pd.read_csv("data/train.csv")
target = "TARGET"  # change
X = train.drop(columns=[target])
y = train[target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_features = ["NUMERIC_COLUMN_1", "NUMERIC_COLUMN_2"]
categorical_features = ["CATEGORICAL_COLUMN_1"]

numeric_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)
categorical_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])
model = make_pipeline(preprocessor, LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

pred = model.predict(X_val)
print("Train accuracy:", model.score(X_train, y_train))
print("Validation accuracy:", accuracy_score(y_val, pred))
```

For regression, use a regressor, regression metrics, and generally omit `stratify=y`. Adapt columns and validation to the actual competition. Keep the end-to-end pattern.

## 12. Common errors and fixes

| Error / symptom | Likely reason | Action |
|---|---|---|
| `InvalidParameterError: Got 'Median'` | Parameter string is case-sensitive | Use `strategy="median"` |
| `could not convert string to float` | Raw categorical strings reached numeric model | Encode selected categorical features; check `remainder` |
| `Input X contains NaN` | An estimator received unhandled missing values | Add appropriate imputer/native handling |
| `NotFittedError` | Called transform/predict before fitting | Fit training pipeline first |
| `KeyError` or missing column | Selected name absent from DataFrame | Compare `X_train.columns` and `X_val.columns` |
| `Found unknown categories` | Encoder saw unseen validation category | `OneHotEncoder(handle_unknown="ignore")` |
| Different feature counts | Independently fitted train/validation encoders | Fit once on training; transform validation |
| Unexpected `None` after `df.info()` | `print(df.info())` | Call `df.info()` directly |
| Pipeline changes didn't take effect | Earlier notebook cells not rerun | Re-execute definitions in order; restart/run-all if needed |
| Giant memory use | Dense one-hot output/high cardinality | Prefer sparse representations or alternative encoders |

## 13. Five-second mental model

> **Split first → choose features → construct per-type preprocessors → `ColumnTransformer` selects and combines → pipeline with model → `.fit(X_train, y_train)` → `.predict(X_val)` → metric.**

If you remember just one distinction: **training learns preprocessing statistics; inference reuses them.**

**Previous:** [Part 1 — EDA](part1_eda.md).<br>
**Next:** [Part 3 — Model Selection and Hyperparameter Tuning](part3_model_selection.md).
