import pytest
from app.ml.naive_bayes import TicketClassifier

def test_classifier_predicts_label():
    clf = TicketClassifier.load_or_train()
    label, probs = clf.predict("I was billed incorrectly this month")
    assert label in {"billing", "technical", "account"}
    assert abs(sum(probs.values()) - 1.0) < 1e-6
