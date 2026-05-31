"""
Unit tests for core cascade routing functions.
These tests do not require datasets, models, or external APIs.
"""
import numpy as np
import pytest


SAFE = 0
THREAT = 1


# --- Helpers replicated from experiment scripts (pure functions) ---

def majority_vote(labels):
    from collections import Counter
    return Counter(labels).most_common(1)[0][0]


def to_binary_label_hatexplain(label_id):
    if label_id == 1:
        return SAFE
    if label_id in {0, 2}:
        return THREAT
    raise ValueError(f"Unexpected label id: {label_id!r}")


def false_negative_rate(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    total_threats = np.sum(y_true == THREAT)
    if total_threats == 0:
        return 0.0
    missed = np.sum((y_true == THREAT) & (y_pred == SAFE))
    return float(missed / total_threats)


def run_class_specific_cascade(tfidf_preds, tfidf_conf, roberta_preds, tau_threat, tau_safe):
    tfidf_preds = np.asarray(tfidf_preds)
    tfidf_conf = np.asarray(tfidf_conf, dtype=float)
    roberta_preds = np.asarray(roberta_preds)

    keep_threat = (tfidf_preds == THREAT) & (tfidf_conf >= tau_threat)
    keep_safe = (tfidf_preds == SAFE) & (tfidf_conf >= tau_safe)
    keep_tfidf = keep_threat | keep_safe
    escalate = ~keep_tfidf

    final_preds = tfidf_preds.copy()
    final_preds[escalate] = roberta_preds[escalate]

    tier1_fraction = float(np.mean(keep_tfidf))
    tier2_fraction = float(np.mean(escalate))
    return final_preds, tier1_fraction, tier2_fraction


# --- Tests ---

class TestMajorityVote:
    def test_unanimous(self):
        assert majority_vote([1, 1, 1]) == 1

    def test_majority(self):
        assert majority_vote([0, 1, 1]) == 1

    def test_single_element(self):
        assert majority_vote([2]) == 2

    def test_tie_returns_one(self):
        result = majority_vote([0, 1])
        assert result in {0, 1}


class TestToBinaryLabel:
    def test_normal_maps_to_safe(self):
        assert to_binary_label_hatexplain(1) == SAFE

    def test_hatespeech_maps_to_threat(self):
        assert to_binary_label_hatexplain(0) == THREAT

    def test_offensive_maps_to_threat(self):
        assert to_binary_label_hatexplain(2) == THREAT

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError):
            to_binary_label_hatexplain(99)


class TestFalseNegativeRate:
    def test_no_threats_returns_zero(self):
        y_true = [SAFE, SAFE, SAFE]
        y_pred = [SAFE, SAFE, SAFE]
        assert false_negative_rate(y_true, y_pred) == 0.0

    def test_all_threats_caught(self):
        y_true = [THREAT, THREAT, SAFE]
        y_pred = [THREAT, THREAT, SAFE]
        assert false_negative_rate(y_true, y_pred) == 0.0

    def test_all_threats_missed(self):
        y_true = [THREAT, THREAT]
        y_pred = [SAFE, SAFE]
        assert false_negative_rate(y_true, y_pred) == 1.0

    def test_half_threats_missed(self):
        y_true = [THREAT, THREAT, SAFE]
        y_pred = [SAFE, THREAT, SAFE]
        assert false_negative_rate(y_true, y_pred) == pytest.approx(0.5)

    def test_numpy_arrays(self):
        y_true = np.array([THREAT, THREAT, THREAT])
        y_pred = np.array([THREAT, SAFE, SAFE])
        assert false_negative_rate(y_true, y_pred) == pytest.approx(2 / 3)


class TestClassSpecificCascade:
    def test_all_high_confidence_stays_at_tier1(self):
        tfidf_preds = np.array([SAFE, THREAT, SAFE])
        tfidf_conf = np.array([0.95, 0.90, 0.92])
        roberta_preds = np.array([THREAT, SAFE, THREAT])
        final, tier1, tier2 = run_class_specific_cascade(
            tfidf_preds, tfidf_conf, roberta_preds,
            tau_threat=0.80, tau_safe=0.80,
        )
        np.testing.assert_array_equal(final, tfidf_preds)
        assert tier1 == pytest.approx(1.0)
        assert tier2 == pytest.approx(0.0)

    def test_all_low_confidence_escalates_to_tier2(self):
        tfidf_preds = np.array([SAFE, THREAT, SAFE])
        tfidf_conf = np.array([0.51, 0.52, 0.53])
        roberta_preds = np.array([THREAT, SAFE, THREAT])
        final, tier1, tier2 = run_class_specific_cascade(
            tfidf_preds, tfidf_conf, roberta_preds,
            tau_threat=0.90, tau_safe=0.90,
        )
        np.testing.assert_array_equal(final, roberta_preds)
        assert tier1 == pytest.approx(0.0)
        assert tier2 == pytest.approx(1.0)

    def test_mixed_routing(self):
        tfidf_preds = np.array([SAFE, THREAT, SAFE, THREAT])
        tfidf_conf  = np.array([0.95, 0.55,  0.55, 0.90])
        roberta_preds = np.array([THREAT, SAFE, THREAT, SAFE])
        final, tier1, tier2 = run_class_specific_cascade(
            tfidf_preds, tfidf_conf, roberta_preds,
            tau_threat=0.80, tau_safe=0.80,
        )
        assert final[0] == SAFE     # high-conf SAFE stays
        assert final[1] == SAFE     # low-conf THREAT → RoBERTa says SAFE
        assert final[2] == THREAT   # low-conf SAFE → RoBERTa says THREAT
        assert final[3] == THREAT   # high-conf THREAT stays
        assert tier1 == pytest.approx(0.5)
        assert tier2 == pytest.approx(0.5)

    def test_tier_fractions_sum_to_one(self):
        rng = np.random.default_rng(42)
        n = 100
        tfidf_preds = rng.integers(0, 2, n)
        tfidf_conf = rng.uniform(0.5, 1.0, n)
        roberta_preds = rng.integers(0, 2, n)
        _, tier1, tier2 = run_class_specific_cascade(
            tfidf_preds, tfidf_conf, roberta_preds,
            tau_threat=0.75, tau_safe=0.75,
        )
        assert tier1 + tier2 == pytest.approx(1.0)

    def test_asymmetric_thresholds(self):
        tfidf_preds = np.array([SAFE, SAFE, THREAT, THREAT])
        tfidf_conf  = np.array([0.85, 0.65, 0.85,  0.65])
        roberta_preds = np.array([THREAT, THREAT, SAFE, SAFE])
        final, _, _ = run_class_specific_cascade(
            tfidf_preds, tfidf_conf, roberta_preds,
            tau_threat=0.80, tau_safe=0.70,
        )
        assert final[0] == SAFE    # conf 0.85 >= tau_safe 0.70 → keep TF-IDF
        assert final[1] == THREAT  # conf 0.65 < tau_safe 0.70 → use RoBERTa
        assert final[2] == THREAT  # conf 0.85 >= tau_threat 0.80 → keep TF-IDF
        assert final[3] == SAFE    # conf 0.65 < tau_threat 0.80 → use RoBERTa

    def test_fnr_improves_when_escalating_low_confidence_threats(self):
        y_true = np.array([THREAT] * 10 + [SAFE] * 10)
        tfidf_preds = np.array([SAFE] * 10 + [SAFE] * 10)
        tfidf_conf = np.array([0.55] * 10 + [0.95] * 10)
        roberta_preds = np.array([THREAT] * 10 + [SAFE] * 10)

        tfidf_fnr = false_negative_rate(y_true, tfidf_preds)

        final, _, _ = run_class_specific_cascade(
            tfidf_preds, tfidf_conf, roberta_preds,
            tau_threat=0.80, tau_safe=0.80,
        )
        cascade_fnr = false_negative_rate(y_true, final)
        assert cascade_fnr < tfidf_fnr
