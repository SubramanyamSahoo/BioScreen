"""Evaluation benchmarks for screening models."""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)


class ScreeningBenchmark:
    """Comprehensive evaluation of screening models."""

    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = torch.device(device)
        self.model.eval()

    @torch.no_grad()
    def evaluate_binary(self, sequences: List[str], labels: np.ndarray, batch_size: int = 16) -> Dict[str, float]:
        all_probs = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            outputs = self.model(batch)
            probs = F.softmax(outputs["binary_logits"], dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
        scores = np.array(all_probs)
        auroc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
        benign_scores = scores[labels == 0]
        results = {"auroc": auroc, "avg_precision": ap}
        for fpr_target in [0.01, 0.05, 0.10]:
            threshold = np.percentile(benign_scores, 100 * (1 - fpr_target))
            preds = (scores >= threshold).astype(int)
            results[f"recall_at_{int(fpr_target*100)}pct_fpr"] = recall_score(labels, preds)
            results[f"f1_at_{int(fpr_target*100)}pct_fpr"] = f1_score(labels, preds)
            results[f"precision_at_{int(fpr_target*100)}pct_fpr"] = precision_score(labels, preds)
        return results

    @torch.no_grad()
    def evaluate_mechanism(self, sequences: List[str], labels: np.ndarray,
                           mechanism_names: List[str], batch_size: int = 16) -> Dict:
        all_preds = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            outputs = self.model(batch)
            preds = outputs["mechanism_logits"].argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
        all_preds = np.array(all_preds)
        report = classification_report(labels, all_preds, target_names=mechanism_names,
                                       output_dict=True, zero_division=0)
        return {
            "classification_report": report,
            "confusion_matrix": confusion_matrix(labels, all_preds).tolist(),
            "accuracy": float((all_preds == labels).mean()),
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
        }

    @torch.no_grad()
    def evaluate_adversarial(self, adversarial_sequences: List[str],
                             sequence_identities: np.ndarray,
                             identity_bins: List[float] = None,
                             batch_size: int = 16) -> Dict:
        if identity_bins is None:
            identity_bins = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        all_probs = []
        for i in range(0, len(adversarial_sequences), batch_size):
            batch = adversarial_sequences[i:i + batch_size]
            outputs = self.model(batch)
            probs = F.softmax(outputs["binary_logits"], dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
        adv_scores = np.array(all_probs)
        bin_edges = [0.0] + list(identity_bins)
        results_by_bin = {}
        for i in range(len(bin_edges) - 1):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (sequence_identities > lo) & (sequence_identities <= hi)
            if mask.sum() < 3:
                continue
            bin_scores = adv_scores[mask]
            results_by_bin[f"{lo:.0%}-{hi:.0%}"] = {
                "count": int(mask.sum()),
                "mean_threat_prob": float(bin_scores.mean()),
                "detection_rate_at_50pct": float((bin_scores >= 0.5).mean()),
                "detection_rate_at_90pct": float((bin_scores >= 0.9).mean()),
            }
        return {
            "overall_detection_rate": float((adv_scores >= 0.5).mean()),
            "overall_mean_threat_prob": float(adv_scores.mean()),
            "by_identity_bin": results_by_bin,
        }

    def full_evaluation(self, test_sequences, test_binary_labels, test_mechanism_labels,
                        mechanism_names, adversarial_sequences=None,
                        adversarial_identities=None, batch_size=16) -> Dict:
        results = {}
        results["binary"] = self.evaluate_binary(test_sequences, test_binary_labels, batch_size)
        results["mechanism"] = self.evaluate_mechanism(
            test_sequences, test_mechanism_labels, mechanism_names, batch_size)
        if adversarial_sequences is not None and adversarial_identities is not None:
            results["adversarial"] = self.evaluate_adversarial(
                adversarial_sequences, adversarial_identities, batch_size=batch_size)
        return results
