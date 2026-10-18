# ICAIC 2026 — Part 3: Model Selection and Hyperparameter Tuning

> **Goal:** Choose sensible model families, compare them fairly, tune the parameters that matter, and avoid accidentally selecting a model that only got lucky on one validation split.

## 1. The practical workflow

```text
Choose metric and validation strategy
                 |
Build a simple baseline pipeline
                 |
Compare a small set of model families
                 |
Inspect train vs validation/CV scores
                 |
Tune the most promising 1–3 families
                 |
Select one complete pipeline
                 |
Evaluate once on untouched test data (if available)
                 |
Refit on all permitted labeled data
```

**Model selection includes preprocessing.** An imputer + encoder + classifier is one candidate system. Any learned preprocessing must be fitted separately inside each validation fold, which is why the entire pipeline—not only the classifier—goes into cross-validation or `GridSearchCV`.

## 2. Choose a useful first set of models

`*ModelName*` marks a **priority tabular/Kaggle starting model**: either an essential baseline or a family that commonly performs strongly. It does **not** mean the model is guaranteed to win; dataset size, feature types, metric, leakage-safe validation, and tuning matter more than reputation.

### Recommended starting shortlist

For an ordinary tabular classification problem, a strong first pass is:

1. `DummyClassifier` to establish the trivial floor.
2. *`LogisticRegression`* as the fast, interpretable linear baseline.
3. *`RandomForestClassifier`* or *`ExtraTreesClassifier`* as a robust nonlinear baseline.
4. *`HistGradientBoostingClassifier`* if staying entirely within sklearn.
5. *`CatBoostClassifier`*, *`XGBClassifier`*, or *`LGBMClassifier`* when external packages are allowed.

You do not need to tune all of these deeply. Run a modest, fair comparison first, then tune the most promising two or three.

### Linear and statistical models

| Model | Good first choice when... | Main strengths | Main cautions |
|---|---|---|---|
| *`LogisticRegression`* | You need the default serious baseline | Fast, regularized, probabilities, works well on sparse one-hot data | Learns a linear boundary; scale numeric features |
| `RidgeClassifier` | Many correlated or one-hot features; probabilities are unnecessary | Very fast, stable linear classifier | No `predict_proba`; still a linear boundary |
| `SGDClassifier` | Data is too large for ordinary batch training | Scales to huge sparse datasets; supports several linear losses | Sensitive to scaling and optimization settings |
| `LinearSVC` | High-dimensional sparse features such as text | Fast linear margin classifier | No native probabilities; tune `C` and scale numeric features |
| `GaussianNB` | You want an extremely fast probabilistic baseline | Tiny training cost; can work surprisingly well | Strong feature-distribution and independence assumptions |
| `MultinomialNB` / `ComplementNB` | Inputs are nonnegative counts such as bag-of-words text | Excellent text baselines | Not a generic model for standardized continuous features |
| `LinearDiscriminantAnalysis` | Class distributions are roughly Gaussian with shared covariance | Fast; dimensionality reduction and classification | Assumptions can be restrictive; covariance can be unstable |
| `QuadraticDiscriminantAnalysis` | Classes need different quadratic boundaries and data is modest | More flexible than LDA | Needs enough data to estimate class covariances |

### Trees and sklearn ensembles

| Model | Good first choice when... | Main strengths | Main cautions |
|---|---|---|---|
| `DecisionTreeClassifier` | You want an interpretable nonlinear benchmark | Captures interactions; no scaling needed | Easily overfits; unstable without constraints |
| *`RandomForestClassifier`* | You want a robust nonlinear tabular baseline | Handles interactions, less fragile than one tree | Larger/slower; may trail tuned boosting |
| *`ExtraTreesClassifier`* | You want a fast, highly randomized forest baseline | Often strong with little tuning; reduces variance through randomization | Can need many trees; feature importance can mislead |
| `GradientBoostingClassifier` | Dataset is not huge and classic boosting is sufficient | Stronger than one tree; mature implementation | Usually slower/less scalable than histogram boosting |
| *`HistGradientBoostingClassifier`* | Medium/large dense tabular data has nonlinear patterns | Strong sklearn-only boosting baseline; native numerical NaNs and categorical support in current sklearn | Input/API requirements differ from a sparse one-hot pipeline |
| `AdaBoostClassifier` | Weak shallow trees can be combined into a useful boundary | Simple boosting concept; decent educational baseline | Often outperformed by modern gradient boosting; sensitive to noisy labels |
| `BaggingClassifier` | A high-variance base estimator needs stabilization | General bagging wrapper | Random forest/extra trees are usually more direct for decision trees |

### External gradient-boosted trees: frequent Kaggle contenders

| Model | Good first choice when... | Main strengths | Main cautions |
|---|---|---|---|
| *`CatBoostClassifier`* | Data has many categorical columns or categories with moderate/high cardinality | Native categorical handling, strong defaults, numerical missing values, ordered boosting | Categorical missing values/types need preparation; training can be slower |
| *`XGBClassifier`* | You want a widely used, highly tunable boosted-tree system | Strong tabular performance, sparse input, regularization, CPU/GPU support | Many interacting parameters; native categoricals require correct dtypes/settings |
| *`LGBMClassifier`* | Data is larger or fast iteration matters | Very fast histogram training, native categorical support, efficient on large data | Leaf-wise growth can overfit small data; constrain leaves/depth |

These are separate packages (`catboost`, `xgboost`, and `lightgbm`), not part of sklearn. They expose sklearn-style estimators, but their native categorical handling and early-stopping APIs deserve model-specific code.

### Kernel, neighbor, and neural models

| Model | Good first choice when... | Main strengths | Main cautions |
|---|---|---|---|
| `SVC` | Dataset is small/medium and boundaries may be complex | Flexible kernels; often strong after scaling | Can be slow; `probability=True` adds cost |
| `KNeighborsClassifier` | Local similarity may be meaningful and data is not huge | Simple, nonlinear, little model training | Very scale-sensitive; slow prediction; suffers in high dimensions |
| `MLPClassifier` | You have enough data and reason to expect a neural network to help | Flexible nonlinear function approximator | Scaling/tuning essential; usually not the first choice for ordinary tabular data |

### Combination models—not first-line baselines

| Model | Use when... | Caution |
|---|---|---|
| `VotingClassifier` | Several diverse, individually strong models make complementary errors | Weak or highly correlated members rarely help |
| `StackingClassifier` | You have enough data/compute to learn how to combine strong base models | Must generate meta-features out of fold to avoid leakage |

Start with diversity, not ten slight variants of one model. A linear model, a tree ensemble, and one other plausible family reveal more than a large unfocused search.

## 3. The hyperparameters worth knowing first

Hyperparameters are choices set **before fitting**; learned coefficients and tree splits are model parameters.

### Logistic regression

| Parameter | Effect | Sensible values to try |
|---|---|---|
| `C` | Inverse regularization strength; smaller = stronger regularization | `0.01, 0.1, 1, 10, 100` on a log scale |
| `penalty` | Type of regularization | Usually `"l2"`; try `"l1"` with a compatible solver |
| `class_weight` | Gives minority classes more influence | `None`, `"balanced"` when metric/imbalance warrants it |
| `max_iter` | Optimization limit, not a complexity control | Raise when convergence warnings occur |

### Decision tree / random forest

| Parameter | Effect | Practical intuition |
|---|---|---|
| `max_depth` | Maximum tree depth | Smaller values reduce overfitting |
| `min_samples_split` | Minimum samples required to split a node | Larger values make splitting more conservative |
| `min_samples_leaf` | Minimum samples in each leaf | Larger leaves smooth predictions |
| `max_features` | Features considered at each split | Smaller values diversify forest trees |
| `n_estimators` | Number of forest trees | More usually stabilizes results but costs time |
| `class_weight` | Reweights classes | Useful when aligned with the metric/problem |

The same core controls apply to `ExtraTreesClassifier`. Compared with a random forest, extra trees inject more randomness into split selection; compare both before spending time on fine tuning.

### Gradient-boosted trees

Parameter names vary slightly across libraries, but the main ideas transfer:

| Concept | CatBoost | XGBoost | LightGBM | Practical intuition |
|---|---|---|---|---|
| Number of boosting rounds | `iterations` | `n_estimators` | `n_estimators` | More trees can fit more structure; pair with a smaller learning rate |
| Step size | `learning_rate` | `learning_rate` | `learning_rate` | Smaller is usually safer but needs more rounds |
| Tree complexity | `depth` | `max_depth` | `num_leaves`, `max_depth` | More complexity captures interactions but overfits more easily |
| Row sampling | `subsample` | `subsample` | `subsample` | Values below 1 add randomness and may reduce overfitting |
| Feature sampling | `rsm` | `colsample_bytree` | `colsample_bytree` | Train each tree on only some features |
| L2 regularization | `l2_leaf_reg` | `reg_lambda` | `reg_lambda` | Larger values make leaf estimates more conservative |
| Positive-class weighting | `scale_pos_weight` | `scale_pos_weight` | `scale_pos_weight` | Use only when imbalance and the target metric justify it |

A good tuning order is: `learning_rate` + number of rounds, tree complexity, row/feature sampling, then regularization. Use early stopping when the library's validation interface fits your workflow; do not treat the final holdout as the early-stopping set during repeated experiments.

### SVC and KNN

| Model | Parameter | Practical intuition |
|---|---|---|
| SVC | `C` | Larger = weaker regularization and a tighter fit |
| SVC | `kernel` | `"linear"` for linear boundaries; `"rbf"` for nonlinear ones |
| SVC | `gamma` | With RBF, larger values make each point's influence more local |
| KNN | `n_neighbors` | Small = flexible/noisy; large = smoother |
| KNN | `weights` | `"distance"` gives closer neighbors more influence |
| KNN | `p` | Minkowski distance: `1` Manhattan, `2` Euclidean |

Inspect the available parameters instead of guessing names:

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree.get_params()                 # Direct parameters
pipeline.get_params().keys()     # Includes nested names such as model__max_depth
```

## 4. Preprocessing must match the model family

| Family | Numeric scaling | Missing values | Categorical features |
|---|---|---|---|
| Linear, SVC, KNN, MLP | Usually scale | Impute | One-hot/ordinal encode |
| Random forest / extra trees | Unnecessary | Native numeric support varies by estimator/version; a conservative pipeline may impute | Usually encode |
| sklearn histogram boosting | Unnecessary | Supports numerical NaNs | Can use ordinal/native categorical features; avoid accidental sparse output |
| XGBoost / LightGBM | Unnecessary | Native numerical missing-value handling | One-hot pipeline or carefully configured native categoricals |
| CatBoost | Unnecessary | Native numerical NaNs | Pass categorical column names/indices; convert categorical missing values consistently |

This means the earlier one-hot + scaling preprocessor is fair for models that require it, but it is not automatically the best representation for every boosted-tree library.

### XGBoost and LightGBM with the existing encoded pipeline

The simplest bridge from Part 2 is to reuse the imputed/one-hot representation:

```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

boosting_models = {
    "xgboost": XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    ),
    "lightgbm": LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    ),
}
```

Add these estimators to the comparison loop only if the packages are installed. For native categorical handling, XGBoost expects categorical dataframe dtypes with `enable_categorical=True`; LightGBM can use Pandas categorical columns. Keep category definitions consistent between training and inference.

### CatBoost with native categorical features

CatBoost normally should receive the selected raw columns instead of the one-hot `ColumnTransformer`:

```python
from catboost import CatBoostClassifier

feature_columns = numeric_features + categorical_features
X_catboost = X[feature_columns].copy()

# CatBoost handles numerical NaNs, but categorical values should be consistent strings.
for column in categorical_features:
    X_catboost[column] = X_catboost[column].fillna("__MISSING__").astype(str)

X_train_cb, X_val_cb, y_train_cb, y_val_cb = train_test_split(
    X_catboost,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

catboost_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="Accuracy",
    cat_features=categorical_features,
    random_seed=42,
    verbose=False,
)
catboost_model.fit(
    X_train_cb,
    y_train_cb,
    eval_set=(X_val_cb, y_val_cb),
    early_stopping_rounds=100,
)
```

For a final fair comparison, evaluate CatBoost on the same validation rows/folds and primary metric as every other candidate. This example uses a direct validation set to demonstrate native categorical input and early stopping.

## 5. Fair model-comparison loop

Use the same split, features, and metric for every model. Give stochastic estimators a fixed `random_state` for reproducibility.

```python
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    ),
    "extra_trees": ExtraTreesClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    ),
    "svc": SVC(),
    "knn": KNeighborsClassifier(),
}

results = []
fitted_models = {}

for name, estimator in models.items():
    candidate = Pipeline([
        ("preprocessing", clone(preprocessor)),
        ("model", estimator),
    ])
    candidate.fit(X_train, y_train)

    train_pred = candidate.predict(X_train)
    val_pred = candidate.predict(X_val)
    results.append({
        "model": name,
        "train_accuracy": accuracy_score(y_train, train_pred),
        "val_accuracy": accuracy_score(y_val, val_pred),
    })
    fitted_models[name] = candidate

results = pd.DataFrame(results).sort_values("val_accuracy", ascending=False)
print(results)
```

`HistGradientBoostingClassifier` is intentionally not dropped into this exact dictionary: `OneHotEncoder` may make the shared preprocessor sparse, while histogram boosting requires dense input. Give it a compatible dense/ordinal preprocessing pipeline and compare the resulting **whole pipeline** on the same rows. This is the reusable lesson: fairness means identical data and folds—not necessarily identical preprocessing.

`clone(preprocessor)` gives each candidate a fresh, unfitted copy. Save the **pipeline** in `fitted_models`, not just the estimator, because the pipeline owns the fitted imputer, scaler, and encoder needed at prediction time.

### Read train vs validation scores

| Pattern | Likely diagnosis | Next move |
|---|---|---|
| Low train, low validation | Underfitting or weak features | More expressive model/features; weaker regularization |
| High train, much lower validation | Overfitting | Stronger regularization; shallower trees; more data; better validation |
| Similar and strong | Promising fit | Confirm with cross-validation |
| Validation slightly above train | Can happen from sampling/noise | Check multiple folds; not automatically a bug |

Do not pick a winner from training score. Training performance diagnoses fit; held-out performance estimates generalization.

## 6. Prefer cross-validation for a reliable comparison

A single split can make one candidate look better by luck. Cross-validation rotates which rows are held out and reports the distribution of scores.

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []

for name, estimator in models.items():
    candidate = Pipeline([
        ("preprocessing", clone(preprocessor)),
        ("model", estimator),
    ])
    scores = cross_validate(
        candidate,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        return_train_score=True,
        n_jobs=-1,
    )
    rows.append({
        "model": name,
        "mean_train_accuracy": scores["train_score"].mean(),
        "mean_cv_accuracy": scores["test_score"].mean(),
        "cv_std": scores["test_score"].std(),
    })

cv_results = pd.DataFrame(rows).sort_values("mean_cv_accuracy", ascending=False)
print(cv_results)
```

`test_score` here means the held-out fold—not the competition's `test.csv`. Report mean and standard deviation. A tiny mean advantage smaller than the fold-to-fold variation is weak evidence that one model is truly better.

Choose the splitter that matches the data: `StratifiedKFold` for ordinary classification, `KFold` for independent regression, `GroupKFold` for repeated entities, and `TimeSeriesSplit` for chronological data.

## 7. Manual parameter combinations with `ParameterGrid`

This is useful when learning or when you want custom logging:

```python
from sklearn.model_selection import ParameterGrid, cross_val_score

search_space = {
    "n_estimators": [200, 500],
    "max_depth": [None, 5, 10],
    "min_samples_leaf": [1, 3, 10],
}

rows = []
for params in ParameterGrid(search_space):
    candidate = Pipeline([
        ("preprocessing", clone(preprocessor)),
        ("model", RandomForestClassifier(
            random_state=42, n_jobs=-1, **params
        )),
    ])
    scores = cross_val_score(candidate, X, y, cv=cv, scoring="accuracy")
    rows.append({**params, "mean_cv": scores.mean(), "std_cv": scores.std()})

manual_results = pd.DataFrame(rows).sort_values("mean_cv", ascending=False)
print(manual_results.head())
```

This is a **grid search written by hand**. Prefer sklearn's search classes for normal work because they handle refitting, parallelism, nested parameters, and result bookkeeping.

## 8. `GridSearchCV`: tune inside the pipeline

Name the final pipeline step so nested parameters have predictable names:

```python
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(max_iter=2000)),
])

param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__class_weight": [None, "balanced"],
}

search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="accuracy",       # Match the competition/application metric
    cv=cv,
    n_jobs=-1,
    return_train_score=True,
    refit=True,
)
search.fit(X_train, y_train)

print(search.best_params_)
print(search.best_score_)       # Mean CV score within X_train
best_pipeline = search.best_estimator_
print(best_pipeline.score(X_val, y_val))  # Final untouched validation check
```

With `refit=True`, `best_estimator_` is refitted on all data passed to `search.fit`. The untouched validation score is not the same as `best_score_`: the latter comes from CV folds used during tuning.

### Search different estimator families in one grid

`param_grid` may be a list of dictionaries. Each dictionary contains only parameters valid for its estimator:

```python
param_grid = [
    {
        "model": [LogisticRegression(max_iter=2000)],
        "model__C": [0.1, 1, 10],
    },
    {
        "model": [DecisionTreeClassifier(random_state=42)],
        "model__max_depth": [3, 5, 10, None],
        "model__min_samples_leaf": [1, 5, 10],
    },
    {
        "model": [RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=1
        )],
        "model__max_depth": [5, 10, None],
        "model__min_samples_leaf": [1, 3, 10],
        "model__max_features": ["sqrt", 0.5],
    },
]

search = GridSearchCV(
    pipe, param_grid, scoring="accuracy", cv=cv, n_jobs=-1
)
search.fit(X_train, y_train)
```

Avoid nested parallel overload: when `GridSearchCV(n_jobs=-1)` parallelizes candidates, use `n_jobs=1` inside a random forest.

## 9. Grid search vs randomized search

If a grid has \(5 \times 4 \times 3 = 60\) combinations and 5 folds, it performs 300 fits. Add model families and cost grows quickly.

| Search | Prefer when... |
|---|---|
| `GridSearchCV` | Search space is small and carefully chosen |
| `RandomizedSearchCV` | Many parameters/ranges; only some strongly matter |
| Successive halving | Many candidates can be cheaply eliminated early |
| Manual loop | You need custom behavior or are learning mechanics |

Use coarse-to-fine search. Search powers of ten for parameters like `C` and `gamma`, then narrow around a promising region. More combinations do not repair a bad validation strategy.

## 10. Common mistakes

| Mistake | Why it fails | Better move |
|---|---|---|
| Selecting the highest training score | Rewards memorization | Select using validation/CV metric |
| Reusing one validation set for dozens of decisions | Slowly overfits model selection to that split | Use CV; preserve a final holdout when possible |
| Preprocessing once before CV | Fold validation information can enter transformations | Cross-validate the entire pipeline |
| Changing metric between candidates | Comparison is no longer fair | Decide the primary metric first |
| Huge arbitrary grid | Wastes compute and increases selection noise | Search informed ranges, coarse to fine |
| Tuning every model equally | Spends effort on weak families | Screen broadly, then tune finalists |
| Calling `SVC.predict_proba` with defaults | Default `probability=False` | Set `probability=True` before fitting if probabilities are truly needed |
| Treating `random_state` as a performance knob | Chooses a lucky random run | Fix it for reproducibility or average across valid resamples |
| One-hot encoding before CatBoost by habit | Throws away a key native advantage | Pass prepared categorical columns via `cat_features` |
| Giving every family identical preprocessing | Can handicap models or cause sparse/dense errors | Keep folds/metric fixed; tailor leak-safe preprocessing by family |
| Ensembling before strong single models | Adds complexity without reliable diversity | Establish and diagnose strong individual models first |

## 11. Competition-ready checklist

- [ ] Metric and prediction format are known.
- [ ] Splitter matches independent, grouped, or temporal structure.
- [ ] Every candidate includes leak-safe preprocessing.
- [ ] Baseline model works end to end.
- [ ] Diverse families were compared on identical folds.
- [ ] Model-specific categorical and sparse/dense requirements were respected.
- [ ] Train/CV gap and fold variability were inspected.
- [ ] Only promising families received targeted tuning.
- [ ] Final choice was checked on data not used to choose hyperparameters.
- [ ] Full chosen pipeline—not only the classifier—was saved/refitted.

## 12. Official references

- [CatBoost categorical features and classifier parameters](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier)
- [XGBoost categorical-data guide](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html)
- [XGBoost parameter guide](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [LightGBM categorical-feature guide](https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html#categorical-feature-support)
- [LightGBM `LGBMClassifier` API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html)
- [sklearn ensemble guide](https://scikit-learn.org/stable/modules/ensemble.html)

## 13. Five-second mental model

> **Baseline first → compare diverse families fairly → use CV to reduce split luck → tune only promising models → preserve an untouched final check.**

**Previous:** [Part 2 — Data Preprocessing](part2_preprocessing.md).<br>
**Next:** [Part 4 — Classification Evaluation](part4_evaluation.md).
