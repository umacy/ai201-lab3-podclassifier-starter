# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:
- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does
Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`. |

### Output

| Return value | Type | Description |
|---|---|---|
| accuracy | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
accuracy = (number of positions where predicted label exactly equals the
           ground-truth label) / (total number of episodes)

"Correct" = an exact string match between predictions[i] and ground_truth[i]
at the same index. No partial credit, no case-folding (both come from the same
controlled VALID_LABELS vocabulary, plus "unknown"). We divide by the total
number of episodes (len of the lists), NOT by the number of correct ones.
```

---

**Step-by-step logic:**

```
1. If ground_truth is empty, return 0.0 (avoid divide-by-zero).
2. Pair predictions and ground_truth index-by-index (they are the same
   length and same order, as guaranteed by run_evaluation()).
3. Count the pairs where predicted == truth.
4. Return correct_count / len(ground_truth) as a float.
```

---

**Edge case — what if both lists are empty?**

```
Return 0.0. There are zero episodes, so the count of correct predictions is 0
and dividing 0 by 0 is undefined — we guard against it and return 0.0. (An
empty test set means "nothing was evaluated," and the report should show 0%
rather than crash with a ZeroDivisionError.)
```

---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

idx 0: interview == interview   ✓
idx 1: solo      == solo        ✓
idx 2: panel     != solo        ✗
idx 3: interview != narrative   ✗

correct = 2, total = 4
accuracy = 2 / 4 = 0.5
```

---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does
Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order. |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec fields — fill these in before writing code

**What does "correct" mean for a given class?**

```
For class C, an episode counts as correct when its GROUND-TRUTH label is C
AND the prediction for that same episode is also C. In other words, correct
is measured per ground-truth class: of the episodes that truly are "interview",
how many did we predict "interview"? This is recall for class C — it does NOT
count episodes that we predicted "interview" but were actually something else
(those false positives belong to the OTHER class's "total").
```

---

**What does "total" mean for a given class?**

```
For class C, "total" is the number of episodes whose GROUND-TRUTH label is C —
not the number of predictions of C, and not the overall episode count. It is
the denominator for that class's recall. Summing "total" across all classes
equals the number of episodes whose ground truth is a valid label.
```

---

**Step-by-step logic:**

```
1. Initialize a dict with one entry per label in VALID_LABELS, each
   {"correct": 0, "total": 0, "accuracy": 0.0}.
2. Loop over the (predicted, truth) pairs index-by-index.
3. For each pair:
     - If truth is a key in the dict, increment that class's "total".
     - If additionally predicted == truth, increment that class's "correct".
   (A truth label of "unknown" or anything outside VALID_LABELS is ignored,
    since we only report on the four real classes.)
4. After the loop, for each class compute accuracy = correct / total,
   guarding total == 0 by setting accuracy = 0.0.
5. Return the dict keyed by label.
```

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
Set accuracy = 0.0 (and leave correct = 0, total = 0). Why: with no episodes
of that class there is nothing to be right or wrong about, and 0/0 is undefined
— the docstring in evaluate.py specifies "0.0 if total is 0". This keeps every
class present in the report with a well-defined number instead of a NaN or a
KeyError, and the report can still render its bar at 0%.
```

---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

Group by GROUND TRUTH:
  interview: idx 0 (gt=interview, pred=interview ✓)          -> 1/1
  solo:      idx 1 (gt=solo, pred=interview ✗),
             idx 2 (gt=solo, pred=solo ✓)                    -> 1/2
  panel:     idx 3 (gt=panel, pred=panel ✓)                  -> 1/1
  narrative: idx 4 (gt=narrative, pred=panel ✗)              -> 0/1

label       correct  total  accuracy
----------  -------  -----  --------
interview   1        1      1.0
solo        1        2      0.5
panel       1        1      1.0
narrative   0        1      0.0
```

---

## Reflection questions (discuss at the checkpoint)

1. Your overall accuracy might be decent even if one class has very low accuracy.
   Why is per-class accuracy a more informative metric than overall accuracy alone?

2. If `panel` episodes consistently get misclassified as `interview`, what does
   that tell you about your training labels or your prompt?

3. You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
   How might the evaluation results change if you had labeled 100 training episodes?
   What if you had 200 test episodes?
