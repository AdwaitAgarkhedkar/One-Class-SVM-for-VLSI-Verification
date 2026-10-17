from google.colab import files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score

uploaded = files.upload()
CSV_PATH = list(uploaded.keys())[0]

N_TRAIN = 50
NU = 0.05
N_RANDOM_RUNS = 20

# Load the CSV (one row per clock cycle) and group it into tests
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]
if "test_id" not in df.columns:
    df = df.rename(columns={df.columns[0]: "test_id"})
df = df.sort_values(["test_id", "cycle"]).reset_index(drop=True)

tests = {t: g for t, g in df.groupby("test_id")}
test_ids = sorted(tests)
print(f"Loaded {len(df)} rows = {len(test_ids)} tests "
      f"x {df.groupby('test_id').size().iloc[0]} cycles each")


# Coverage bins hit by one test
def bins_of(g):
    sb = g.state_before.values
    sa = g.state_after.values
    d = g.din.values
    det = g.detected.values
    bins = set()

    for s in set(sa):
        bins.add(f"visit_S{s}")

    for x, y, z in zip(sb, d, sa):
        bins.add(f"trans_S{x}_d{y}_S{z}")

    n_det = det.sum()
    if n_det >= 1:
        bins.add("det_ge1")
        bins.add(f"first_det_cycle_{int(np.argmax(det == 1))}")
    if n_det >= 2:
        bins.add("det_ge2")

    run = best = 0
    for v in d:
        run = run + 1 if v == 1 else 0
        best = max(best, run)
    bins.add(f"max_run_ones_{best}")

    for i in range(len(sa) - 1):
        bins.add(f"path3_S{sb[i]}_S{sa[i]}_S{sa[i + 1]}")

    for c, s in enumerate(sa):
        bins.add(f"state_S{s}_phase{c // 4}")

    return bins


bins_per_test = [bins_of(tests[t]) for t in test_ids]
vocab = sorted(set().union(*bins_per_test))
print(f"Distinct coverage bins hit across all tests: {len(vocab)}")

# A test is novel if it hits a bin that no earlier test hit
seen, y = set(), []
for s in bins_per_test:
    y.append(int(len(s - seen) > 0))
    seen |= s
y = np.array(y)
print(f"Overall: {y.sum()} novel / {len(y)} tests ({100 * y.mean():.1f}%)")

# Features: binary bin vector plus scaled state-change count
X_bins = np.array([[int(b in s) for b in vocab] for s in bins_per_test], dtype=float)
changes = np.array([(tests[t].state_before.values != tests[t].state_after.values).sum()
                    for t in test_ids], dtype=float) / 12.0
X = np.hstack([X_bins, changes[:, None]])

train_idx = np.arange(N_TRAIN)
train_redundant_idx = train_idx[y[:N_TRAIN] == 0]
print(f"First batch: {N_TRAIN} tests, {y[:N_TRAIN].sum()} novel; "
      f"training One-Class SVM on {len(train_redundant_idx)} REDUNDANT tests")

# No feature scaling: standardizing binary bin vectors made the ranking worse
oc_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=NU)
oc_svm.fit(X[train_redundant_idx])

eval_idx = np.arange(N_TRAIN, len(test_ids))
score = -oc_svm.decision_function(X[eval_idx])  # higher = more unusual

seen_after_train = set().union(*bins_per_test[:N_TRAIN])
running = set(seen_after_train)
y_eval = []
for i in eval_idx:
    y_eval.append(int(len(bins_per_test[i] - running) > 0))
    running |= bins_per_test[i]
y_eval = np.array(y_eval)

print(f"Evaluation set: {len(eval_idx)} tests, {y_eval.sum()} truly novel "
      f"relative to the first batch's coverage")
if 0 < y_eval.sum() < len(y_eval):
    print(f"ROC-AUC (outlier score vs. ground-truth novelty): "
          f"{roc_auc_score(y_eval, score):.3f}")


# SVM ranking vs random order
def coverage_curve(order):
    s = set(seen_after_train)
    out = []
    for i in order:
        s |= bins_per_test[eval_idx[i]]
        out.append(len(s))
    return out


def tests_to_full(curve):
    for k, v in enumerate(curve):
        if v >= total_bins:
            return k + 1
    return None


total_bins = len(set().union(*bins_per_test))

curve_svm = coverage_curve(np.argsort(-score))

rng = np.random.RandomState(0)
random_curves = [coverage_curve(rng.permutation(len(eval_idx)))
                 for _ in range(N_RANDOM_RUNS)]
curve_random = np.mean(random_curves, axis=0)

random_full = [tests_to_full(c) for c in random_curves]
random_full_valid = [v for v in random_full if v is not None]

print(f"\nTotal bins reachable: {total_bins}")
print(f"Bins already covered by the first batch: {len(seen_after_train)}")
print("\nTests needed to reach full coverage:")
print(f"  SVM-ranked order                       : {tests_to_full(curve_svm)}")
print(f"  Random order (average of {N_RANDOM_RUNS} shuffles) : "
      f"{np.mean(random_full_valid):.0f}")

for k in (10, 25, 50, 100):
    if k <= len(curve_svm):
        print(f"  Bins covered after top-{k}: SVM={curve_svm[k - 1]}  "
              f"random(avg)={curve_random[k - 1]:.1f}")

# Plot 1: coverage vs tests kept
n_plot = min(150, len(curve_svm))
plt.figure(figsize=(8, 5))
plt.plot(range(1, n_plot + 1), curve_svm[:n_plot], label="One-Class SVM ranking", linewidth=2)
plt.plot(range(1, n_plot + 1), curve_random[:n_plot], label="Random order (average)", linewidth=2)
plt.axhline(total_bins, color="gray", linestyle=":", label="Max bins reachable")
plt.xlabel("Tests kept (of the evaluation set)")
plt.ylabel("Cumulative coverage bins")
plt.title(f"Sequence detector: coverage vs tests kept (first batch = {N_TRAIN} tests)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("seq_coverage_curve.png", dpi=150)
plt.show()

# Plot 2: outlier score of every evaluation test, novel tests marked
order_asc = np.argsort(score)
sorted_scores = score[order_asc]
novel_sorted = y_eval[order_asc].astype(bool)
ranks = np.arange(1, len(score) + 1)

plt.figure(figsize=(8, 5))
plt.plot(ranks, sorted_scores, color="tab:blue", linewidth=1.5, zorder=1)
plt.scatter(ranks[~novel_sorted], sorted_scores[~novel_sorted],
            s=8, color="tab:blue", alpha=0.5, label="Ordinary (redundant) tests", zorder=2)
plt.scatter(ranks[novel_sorted], sorted_scores[novel_sorted],
            s=70, color="red", edgecolor="black", label="Novel tests", zorder=3)
plt.axhline(0, color="gray", linestyle="--", linewidth=1,
            label="SVM boundary (above = flagged as outlier)")
plt.xlabel("Tests sorted from most ordinary to most unusual")
plt.ylabel("Outlier score (higher = more unusual)")
plt.title("Outlier score of every evaluation test: ordinary slope vs novel tests")
plt.legend(loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("seq_score_plot.png", dpi=150)
plt.show()

flagged = score > 0
print(f"\nTests above the SVM boundary (flagged as outliers): {flagged.sum()} of {len(score)}")
print(f"Novel tests above the boundary: {(flagged & (y_eval == 1)).sum()} of {y_eval.sum()}")
