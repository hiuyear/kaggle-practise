# ICAIC 2026 — Part 4: Classification Evaluation

> **Goal:** Evaluate a classifier with the metric and validation design that match the real objective, diagnose its mistakes, and report performance without leaking information or overstating certainty.

## 1. Evaluation starts before training

An evaluation is trustworthy only when these four pieces align:

1. **Question:** What kind of error matters?
2. **Metric:** What number represents that cost?
3. **Validation data:** What future/unseen population should the model generalize to?
4. **Prediction type:** Does the metric need labels, probabilities, or decision scores?

For a competition, the published metric is the primary optimization target. Other metrics still help diagnose behavior. Titanic commonly uses accuracy, but accuracy is not universally appropriate.

## 2. Labels, scores, and probabilities

```python
y_pred = model.predict(X_val)                 # Hard class labels
y_prob = model.predict_proba(X_val)[:, 1]    # Positive-class probabilities
y_score = model.decision_function(X_val)      # Ranking score, if supported
```

| Output | Used by |
|---|---|
| Hard labels | Accuracy, precision, recall, F1, confusion matrix |
| Probabilities | Log loss, calibration, threshold selection; ROC-AUC/PR-AUC |
| Decision scores | ROC-AUC and PR-AUC when probabilities are unavailable |

For binary `predict_proba`, inspect `model.classes_` when label ordering is unclear. Column `[:, 1]` is only the intended positive-class probability when the positive class is indeed the second entry.

## 3. Confusion matrix: the source of label metrics

```text
                     Predicted negative   Predicted positive
Actual negative             TN                   FP
Actual positive             FN                   TP
```

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_val, y_pred)
tn, fp, fn, tp = cm.ravel()  # Binary classification only
ConfusionMatrixDisplay(cm).plot()
```

- **False positive (FP):** predicted positive, actually negative.
- **False negative (FN):** predicted negative, actually positive.

Which is worse depends on the application. Missing a dangerous disease and sending an extra legitimate email to spam have different error costs.

## 4. Core classification metrics

| Metric | Formula | Answers |
|---|---|---|
| Accuracy | \((TP+TN)/N\) | What fraction of all labels were correct? |
| Precision | \(TP/(TP+FP)\) | When the model predicts positive, how often is it right? |
| Recall / sensitivity | \(TP/(TP+FN)\) | Of all actual positives, how many did it find? |
| Specificity | \(TN/(TN+FP)\) | Of all actual negatives, how many did it reject? |
| F1 | harmonic mean of precision and recall | Are precision and recall jointly strong? |
| Balanced accuracy | mean recall across classes | How well does it recognize classes despite imbalance? |

```python
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

print("accuracy:", accuracy_score(y_val, y_pred))
print("balanced accuracy:", balanced_accuracy_score(y_val, y_pred))
print("precision:", precision_score(y_val, y_pred))
print("recall:", recall_score(y_val, y_pred))
print("f1:", f1_score(y_val, y_pred))
print(classification_report(y_val, y_pred))
```

**Argument order is `y_true, y_pred`.** Swapping them leaves accuracy unchanged but can change precision and recall, making this a dangerous silent bug.

### Choosing among them

- Use **accuracy** when classes/error costs are reasonably balanced and it is the required metric.
- Prioritize **precision** when false positives are especially costly.
- Prioritize **recall** when false negatives are especially costly.
- Use **F1** when both matter and one positive class is the focus.
- Use **balanced accuracy** or macro-averaged metrics when class frequencies differ and every class matters.

Always compare against a trivial baseline:

```python
from sklearn.dummy import DummyClassifier

dummy = make_pipeline(
    preprocessor,
    DummyClassifier(strategy="most_frequent"),
)
dummy.fit(X_train, y_train)
print(dummy.score(X_val, y_val))
```

A model with 90% accuracy is useless if the majority class already gives 95%.

## 5. Probability and ranking metrics

```python
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

print("ROC-AUC:", roc_auc_score(y_val, y_prob))
print("PR-AUC / average precision:", average_precision_score(y_val, y_prob))
print("log loss:", log_loss(y_val, y_prob))
```

| Metric | Best interpretation | Key caution |
|---|---|---|
| ROC-AUC | Probability a random positive ranks above a random negative | Can look optimistic with very rare positives |
| PR-AUC / average precision | Precision-recall tradeoff for the positive class | Baseline depends on positive prevalence |
| Log loss | Quality/confidence of predicted probabilities | Confident wrong predictions are punished heavily |

ROC-AUC and PR-AUC measure ranking across thresholds, not performance at one operating threshold. A model can have strong AUC but poor default-threshold predictions.

## 6. Thresholds change precision and recall

Binary classifiers often turn probabilities into labels using a threshold near 0.5:

```python
threshold = 0.35
y_pred_035 = (y_prob >= threshold).astype(int)
```

Lowering the threshold usually predicts more positives: recall rises and precision may fall. Raising it usually does the reverse.

```python
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

rows = []
for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    pred = (y_prob >= threshold).astype(int)
    rows.append({
        "threshold": threshold,
        "precision": precision_score(y_val, pred, zero_division=0),
        "recall": recall_score(y_val, pred, zero_division=0),
        "f1": f1_score(y_val, pred, zero_division=0),
    })

print(pd.DataFrame(rows))
```

The threshold is itself a hyperparameter. Choose it on training/CV data according to an explicit objective, then evaluate it on untouched data. Do not choose the threshold that looks best on the final test set.

## 7. Multiclass averaging

For more than two classes, precision/recall/F1 need an averaging rule:

| Average | Meaning | Good when... |
|---|---|---|
| `"macro"` | Metric per class, then unweighted mean | Every class matters equally |
| `"weighted"` | Per-class metric weighted by class support | You want class-wise behavior but retain frequency influence |
| `"micro"` | Pool all decisions before metric | Every individual example matters equally |
| `None` | Return one value per class | You want detailed diagnosis |

```python
f1_score(y_val, y_pred, average="macro")
f1_score(y_val, y_pred, average=None)
```

Weighted F1 can hide weak performance on rare classes. Always inspect per-class support and scores.

## 8. Evaluate multiple metrics with cross-validation

Use one primary metric for selection and secondary metrics for diagnosis:

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

scores = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1,
)

summary = {
    metric.removeprefix("test_"): (values.mean(), values.std())
    for metric, values in scores.items()
    if metric.startswith("test_")
}
print(summary)
```

This returns metrics for each fold. Report, for example, `accuracy = 0.82 ± 0.03`, not only the best fold. The variation is not a formal confidence interval, but it reveals sensitivity to the split.

When tuning, `GridSearchCV` can calculate several metrics but must know which one selects/refits the winner:

```python
search = GridSearchCV(
    pipe,
    param_grid,
    scoring={"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"},
    refit="accuracy",
    cv=cv,
    n_jobs=-1,
)
```

## 9. Separate selection from final evaluation

```text
Training/CV data                     Untouched test data
----------------                     -------------------
fit preprocessing                    evaluate once
compare model families        --->   report final estimate
tune hyperparameters
choose threshold
```

- **Validation/CV** guides model, feature, hyperparameter, and threshold choices.
- **Final holdout/test** estimates performance after those choices are frozen.
- **Competition test.csv** usually has no labels and produces a submission; it cannot locally measure performance.

If data is limited, nested cross-validation gives a less biased performance estimate: the inner CV tunes candidates and the outer CV evaluates the whole selection procedure. It is more expensive and often unnecessary for an early competition baseline, but it matters when you need a defensible estimate.

## 10. Diagnose, not only score

A single metric cannot show *where* a model fails. Useful slices include:

```python
evaluation = X_val.copy()
evaluation["actual"] = y_val.to_numpy()
evaluation["predicted"] = y_pred
evaluation["correct"] = evaluation["actual"] == evaluation["predicted"]

print(evaluation.groupby("Sex")["correct"].mean())
print(evaluation.groupby("Pclass")["correct"].agg(["mean", "count"]))
```

Check performance by meaningful group, time period, missingness pattern, target class, and confidence range. Small slices are noisy, and group differences require context; use them to locate failure modes, not to invent certainty.

For probability-using applications, also inspect calibration: among predictions near 0.8, roughly 80% should be positive. Ranking quality and calibration are different properties.

## 11. Refit and predict for a competition

After model choice is frozen and local evaluation is complete:

```python
# best_pipeline contains preprocessing + classifier + chosen hyperparameters
best_pipeline.fit(X, y)                    # All permitted labeled training rows
test_predictions = best_pipeline.predict(test)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_predictions,
})
submission.to_csv("submission.csv", index=False)
```

Match the required column names, row count, row order, label/probability type, and CSV index rules exactly. Refitting on all labeled data is appropriate only **after** evaluation decisions are complete.

## 12. Common mistakes

| Mistake | Consequence | Fix |
|---|---|---|
| `accuracy_score(y_pred, y_val)` | Hides wrong argument order because accuracy is symmetric | Always use `metric(y_true, y_pred)` |
| Accuracy on a highly imbalanced target | Majority behavior looks deceptively strong | Add confusion matrix, recall/precision, balanced metrics |
| Passing hard labels to ROC-AUC | Throws away ranking information | Pass probabilities or decision scores |
| Tuning on the final holdout | Final score becomes optimistically biased | Freeze choices before final evaluation |
| Reporting the best CV fold | Cherry-picks sampling luck | Report mean and variability |
| Comparing models on different splits | Model and data effects are confounded | Reuse identical folds |
| Assuming 0.5 is optimal | Ignores asymmetric costs/metric | Tune threshold on validation/CV data |
| Using CV while preprocessing globally | Leaks held-out-fold information | Put preprocessing inside the evaluated pipeline |

## 13. Evaluation checklist

- [ ] Primary metric matches the real goal or competition metric.
- [ ] Validation strategy matches the data-generating process.
- [ ] A trivial baseline is recorded.
- [ ] `y_true` and predictions are passed in the correct order.
- [ ] Confusion matrix and important secondary metrics are inspected.
- [ ] Probability metrics receive probabilities/scores, not hard labels.
- [ ] Mean and fold variability are reported for CV.
- [ ] Threshold choices are made without touching final test labels.
- [ ] Important groups and failure cases are inspected.
- [ ] Final holdout is used only after all choices are frozen.

## 14. Five-second mental model

> **Choose the error that matters → use matching predictions and metrics → validate like deployment → diagnose mistakes → touch the final test only once.**

**Previous:** [Part 3 — Model Selection and Hyperparameter Tuning](part3_model_selection.md).
