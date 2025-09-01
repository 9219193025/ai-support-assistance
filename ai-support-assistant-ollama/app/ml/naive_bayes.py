import json, os
from typing import Dict, Tuple, List
from collections import defaultdict
from dataclasses import dataclass

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ticket_nb_model.json")

def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = []
    cur = []
    for ch in text:
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                tokens.append("".join(cur))
                cur = []
    if cur:
        tokens.append("".join(cur))
    return [t for t in tokens if len(t) > 1]

@dataclass
class TicketClassifier:
    label_counts: Dict[str, int]
    token_counts: Dict[str, Dict[str, int]]
    vocab: set
    total_docs: int

    @staticmethod
    def load_or_train():
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return TicketClassifier(
                label_counts=data["label_counts"],
                token_counts=data["token_counts"],
                vocab=set(data["vocab"]),
                total_docs=data["total_docs"],
            )
        samples = _seed_samples()
        clf = TicketClassifier.train(samples)
        clf.save()
        return clf

    @staticmethod
    def train(samples: List[Tuple[str, str]]):
        label_counts = defaultdict(int)
        token_counts = defaultdict(lambda: defaultdict(int))
        vocab = set()
        total_docs = 0

        for label, text in samples:
            total_docs += 1
            label_counts[label] += 1
            for tok in tokenize(text):
                vocab.add(tok)
                token_counts[label][tok] += 1

        return TicketClassifier(
            label_counts=dict(label_counts),
            token_counts={lbl: dict(tokmap) for lbl, tokmap in token_counts.items()},
            vocab=vocab,
            total_docs=total_docs,
        )

    def save(self):
        with open(MODEL_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "label_counts": self.label_counts,
                    "token_counts": self.token_counts,
                    "vocab": list(self.vocab),
                    "total_docs": self.total_docs,
                },
                f,
            )

    def predict(self, text: str) -> Tuple[str, Dict[str, float]]:
        tokens = tokenize(text)
        probs = {}
        vocab_size = max(1, len(self.vocab))
        for label in self.label_counts:
            log_prior = _safe_log(self.label_counts[label] / self.total_docs)
            total_label_tokens = sum(self.token_counts[label].values()) + vocab_size
            log_likelihood = 0.0
            for tok in tokens:
                count = self.token_counts[label].get(tok, 0) + 1
                log_likelihood += _safe_log(count / total_label_tokens)
            probs[label] = log_prior + log_likelihood
        m = max(probs.values())
        expd = {k: pow(2.718281828, v - m) for k, v in probs.items()}
        s = sum(expd.values())
        norm = {k: (v / s if s > 0 else 0.0) for k, v in expd.items()}
        label = max(norm, key=norm.get)
        return label, norm

def _safe_log(x: float) -> float:
    if x <= 0:
        return -1e9
    import math
    return math.log(x)

def _seed_samples():
    return [
        ("billing", "I was charged twice this month. Please refund the extra amount."),
        ("billing", "Invoice shows wrong amount and late fee."),
        ("billing", "Need to change my payment method and update credit card."),
        ("technical", "App crashes when I click the upload button on Android."),
        ("technical", "Website returns 500 error after login."),
        ("technical", "Cannot reset password; the link times out with an error."),
        ("account", "Please delete my account permanently."),
        ("account", "How can I change my username and update my email address?"),
        ("account", "I want to deactivate my profile for a few weeks."),
    ]
