import re
import numpy as np

# Path to your log file
log_file = "raw.txt"

# Read log
with open(log_file, "r", encoding="utf-8") as f:
    log_text = f.read()

# Regex: di-id followed by float
pattern = re.compile(r"(di-[A-Za-z0-9-]+)\s*\n\s*([0-9]*\.[0-9]+)")
matches = pattern.findall(log_text)

# Convert to list of (id, float)
all_preds = [(di_id, float(val)) for di_id, val in matches]

dicom_list = {
    "di-0CA9-8FC9-25BB", "di-1F8E-37E5-57ED", "di-1FD4-CB18-6EFC", "di-5A75-9C41-46D7", "di-5BB1-44DE-60BA",
    "di-62CF-F093-A156", "di-63D5-7095-7FB9", "di-A540-DBDD-7C1F", "di-AE68-A41B-5185", "di-C910-B188-A16F",
    "di-EB49-9AE0-F0D0", "di-6CF6-853C-CB97", "di-16B9-402D-037F", "di-46D0-7328-762F", "di-47EB-1516-2456",
    "di-7041-D238-665F", "di-8254-6AD3-C7FC", "di-A984-8D28-57F4", "di-B55F-84D9-6833", "di-C9E0-1668-D365",
    "di-E42E-19EA-16E7"
}

# Extract all score values
all_scores = [val for _, val in all_preds]

# Define thresholds to test (from min to max in steps)
thresholds = np.linspace(min(all_scores), max(all_scores), 50)

best_threshold = None
best_f1 = -1
results = []

for t in thresholds:
    filtered = {di_id for di_id, val in all_preds if val <= t}
    overlap = filtered & dicom_list
    tp = len(overlap)  # true positives
    fp = len(filtered - dicom_list)  # false positives
    fn = len(dicom_list - filtered)  # false negatives

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    results.append((t, tp, fp, fn, precision, recall, f1))

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"Best threshold: {best_threshold:.4f} (F1={best_f1:.3f})")

# Show a few top candidates
for t, tp, fp, fn, precision, recall, f1 in sorted(results, key=lambda x: -x[-1])[:5]:
    print(f"thr={t:.4f}  TP={tp} FP={fp} FN={fn}  prec={precision:.2f} rec={recall:.2f} f1={f1:.2f}")
