#!/usr/bin/env python3
"""
RoadWatch Python ML Pothole Classifier Trainer (Pure Python 3.11 stdlib).

Trains a Cost-Sensitive L2-Regularized Logistic Regression model using Adam
optimization on 10D candidate region features mined from labeled road images
and hard-negative road artifacts (lane lines, shadows, wet glare, tar joints).

Evaluates generalization via Stratified Grouped K-Fold Cross-Validation
(grouped by sampleId to prevent same-image leakage across folds) and tunes
the decision threshold tau on out-of-fold probabilities to suppress false
positives while maintaining high pothole recall.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from typing import Any


def sigmoid(z: float) -> float:
    if z > 30.0:
        return 1.0
    if z < -30.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def fit_standardized_logistic_regression(
    rows: list[dict[str, Any]],
    dims: int,
    false_positive_penalty: float,
    epochs: int = 550,
) -> dict[str, Any]:
    n = len(rows)
    means = [0.0] * dims
    stds = [1.0] * dims

    if n == 0:
        return {
            "means": means,
            "stds": stds,
            "weights": [0.0] * dims,
            "bias": 0.0,
        }

    for row in rows:
        feats = row["features"]
        for d in range(dims):
            means[d] += float(feats[d])
    for d in range(dims):
        means[d] /= n

    var_acc = [0.0] * dims
    for row in rows:
        feats = row["features"]
        for d in range(dims):
            diff = float(feats[d]) - means[d]
            var_acc[d] += diff * diff
    for d in range(dims):
        stds[d] = max(1e-3, math.sqrt(var_acc[d] / n))

    x_norm = [
        [(float(row["features"][d]) - means[d]) / stds[d] for d in range(dims)]
        for row in rows
    ]
    labels = [1 if int(row["y"]) == 1 else 0 for row in rows]

    weights = [0.0] * dims
    bias = 0.0
    m_w = [0.0] * dims
    v_w = [0.0] * dims
    m_b = 0.0
    v_b = 0.0

    lr = 0.045
    l2 = 0.035
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    for epoch in range(1, epochs + 1):
        grad_w = [0.0] * dims
        grad_b = 0.0

        for i in range(n):
            x = x_norm[i]
            y = labels[i]
            z = bias
            for d in range(dims):
                z += weights[d] * x[d]
            p = sigmoid(z)

            sample_weight = false_positive_penalty if y == 0 else 1.0
            err = (p - y) * sample_weight
            grad_b += err
            for d in range(dims):
                grad_w[d] += err * x[d]

        bc1 = 1.0 - (beta1**epoch)
        bc2 = 1.0 - (beta2**epoch)

        grad_b /= n
        m_b = beta1 * m_b + (1.0 - beta1) * grad_b
        v_b = beta2 * v_b + (1.0 - beta2) * (grad_b * grad_b)
        bias -= (lr * (m_b / bc1)) / (math.sqrt(v_b / bc2) + eps)

        for d in range(dims):
            g = (grad_w[d] / n) + l2 * weights[d]
            m_w[d] = beta1 * m_w[d] + (1.0 - beta1) * g
            v_w[d] = beta2 * v_w[d] + (1.0 - beta2) * (g * g)
            weights[d] -= (lr * (m_w[d] / bc1)) / (math.sqrt(v_w[d] / bc2) + eps)

    return {
        "means": means,
        "stds": stds,
        "weights": weights,
        "bias": bias,
    }


def predict_probability(features: list[float], model: dict[str, Any]) -> float:
    z = float(model["bias"])
    means = model["means"]
    stds = model["stds"]
    weights = model["weights"]
    for d in range(len(weights)):
        xn = (float(features[d]) - means[d]) / max(1e-5, stds[d])
        z += weights[d] * xn
    return sigmoid(z)


def compute_roc_auc(probs: list[float], labels: list[int]) -> float:
    pos = [p for p, y in zip(probs, labels) if y == 1]
    neg = [p for p, y in zip(probs, labels) if y == 0]
    if not pos or not neg:
        return 1.0
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def train_and_evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = payload.get("rows", [])
    feature_names: list[str] = payload.get("featureNames", [])
    dims = len(feature_names) if feature_names else (len(rows[0]["features"]) if rows else 10)
    false_positive_penalty = max(1.0, min(5.0, float(payload.get("falsePositivePenalty", 2.5))))
    previous_version = int(payload.get("previousVersion", 0))
    dataset_stats = payload.get("datasetStats", {})
    epochs = 550

    # Stratified group assignment by sampleId so positive and negative samples
    # are evenly distributed across CV folds with zero same-image leakage.
    pos_sample_ids: list[str] = []
    neg_sample_ids: list[str] = []
    sample_has_pos: dict[str, bool] = {}
    for r in rows:
        sid = str(r["sampleId"])
        sample_has_pos[sid] = sample_has_pos.get(sid, False) or (int(r["y"]) == 1)

    for sid, is_pos in sample_has_pos.items():
        if is_pos:
            pos_sample_ids.append(sid)
        else:
            neg_sample_ids.append(sid)

    total_groups = len(pos_sample_ids) + len(neg_sample_ids)
    k_folds = min(5, max(2, total_groups))
    fold_of_sample: dict[str, int] = {}
    for idx, sid in enumerate(pos_sample_ids):
        fold_of_sample[sid] = idx % k_folds
    for idx, sid in enumerate(neg_sample_ids):
        fold_of_sample[sid] = idx % k_folds

    oof_probs = [0.5] * len(rows)
    oof_labels = [1 if int(r["y"]) == 1 else 0 for r in rows]

    for fold in range(k_folds):
        train_rows = [r for r in rows if fold_of_sample.get(str(r["sampleId"]), 0) != fold]
        if len(train_rows) < 4:
            train_rows = rows
        fold_model = fit_standardized_logistic_regression(
            train_rows, dims, false_positive_penalty, epochs
        )
        for i, r in enumerate(rows):
            if fold_of_sample.get(str(r["sampleId"]), 0) == fold:
                oof_probs[i] = predict_probability(r["features"], fold_model)

    # Sweep threshold tau on out-of-fold predictions to minimize false positives
    # while maintaining high pothole recall.
    best_threshold = 0.52
    best_objective = -1e9

    for step in range(36, 86):
        t = step / 100.0
        tp = fp = tn = fn = 0
        for p, actual in zip(oof_probs, oof_labels):
            pred = 1 if p >= t else 0
            if pred == 1 and actual == 1:
                tp += 1
            elif pred == 1 and actual == 0:
                fp += 1
            elif pred == 0 and actual == 0:
                tn += 1
            else:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        acc = (tp + tn) / max(1, tp + fp + tn + fn)
        f05 = (
            (1.25 * precision * recall) / (0.25 * precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        recall_penalty = (0.88 - recall) * 1.6 if recall < 0.88 else 0.0
        objective = f05 * 0.65 + acc * 0.35 - fpr * 0.45 - recall_penalty

        if objective > best_objective:
            best_objective = objective
            best_threshold = round(t, 2)

    tp = fp = tn = fn = 0
    for p, actual in zip(oof_probs, oof_labels):
        pred = 1 if p >= best_threshold else 0
        if pred == 1 and actual == 1:
            tp += 1
        elif pred == 1 and actual == 0:
            fp += 1
        elif pred == 0 and actual == 0:
            tn += 1
        else:
            fn += 1

    total = max(1, tp + fp + tn + fn)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    f05_score = (
        (1.25 * precision * recall) / (0.25 * precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    roc_auc = compute_roc_auc(oof_probs, oof_labels)

    full_model = fit_standardized_logistic_regression(
        rows, dims, false_positive_penalty, epochs
    )

    pos_regions = sum(1 for y in oof_labels if y == 1)
    neg_regions = sum(1 for y in oof_labels if y == 0)

    return {
        "version": previous_version + 1,
        "trainedAt": datetime.now(timezone.utc).isoformat(),
        "algorithm": "Python 3.11 Cost-Sensitive L2 Logistic Regression (Adam) + Hard-Negative Mining + Stratified Grouped 5-Fold CV",
        "featureNames": feature_names,
        "featureMeans": [round(v, 5) for v in full_model["means"]],
        "featureStds": [round(v, 5) for v in full_model["stds"]],
        "weights": [round(v, 5) for v in full_model["weights"]],
        "bias": round(full_model["bias"], 5),
        "decisionThreshold": best_threshold,
        "falsePositivePenalty": false_positive_penalty,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1Score": round(f1_score, 4),
            "f05Score": round(f05_score, 4),
            "falsePositiveRate": round(false_positive_rate, 4),
            "rocAuc": round(roc_auc, 4),
            "truePositives": tp,
            "falsePositives": fp,
            "trueNegatives": tn,
            "falseNegatives": fn,
            "datasetSamples": int(dataset_stats.get("datasetSamples", len(sample_has_pos))),
            "positiveSamples": int(dataset_stats.get("positiveSamples", len(pos_sample_ids))),
            "negativeSamples": int(dataset_stats.get("negativeSamples", len(neg_sample_ids))),
            "candidateRegionsEvaluated": len(rows),
            "positiveRegions": pos_regions,
            "hardNegativeRegions": neg_regions,
            "trainingEpochs": epochs,
        },
    }


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        sys.stderr.write("Empty training payload on stdin\n")
        return 1
    payload = json.loads(raw)
    result = train_and_evaluate(payload)
    sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
