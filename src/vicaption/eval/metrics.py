from __future__ import annotations

import math
from collections import Counter
from typing import Iterable


def _tokens(text: str) -> list[str]:
    return text.strip().split()


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(len(tokens) - n + 1, 0)))


def _closest_ref_len(pred_len: int, ref_lens: Iterable[int]) -> int:
    return min(ref_lens, key=lambda ref_len: (abs(ref_len - pred_len), ref_len))


def compute_bleu(predictions: dict[str, str], references: dict[str, list[str]]) -> dict[str, float]:
    """Compute BLEU-1 through BLEU-4 with clipped corpus precision."""
    pred_len_total = 0
    ref_len_total = 0
    precisions: list[float] = []

    for n in range(1, 5):
        clipped_total = 0
        pred_total = 0
        pred_len_total = 0
        ref_len_total = 0

        for image_id, prediction in predictions.items():
            if image_id not in references:
                continue
            pred_tokens = _tokens(prediction)
            ref_tokens_list = [_tokens(ref) for ref in references[image_id]]
            pred_len_total += len(pred_tokens)
            ref_len_total += _closest_ref_len(len(pred_tokens), [len(ref) for ref in ref_tokens_list])

            pred_counts = _ngrams(pred_tokens, n)
            max_ref_counts: Counter[tuple[str, ...]] = Counter()
            for ref_tokens in ref_tokens_list:
                ref_counts = _ngrams(ref_tokens, n)
                for gram, count in ref_counts.items():
                    max_ref_counts[gram] = max(max_ref_counts[gram], count)

            clipped_total += sum(min(count, max_ref_counts[gram]) for gram, count in pred_counts.items())
            pred_total += sum(pred_counts.values())

        precisions.append(clipped_total / pred_total if pred_total else 0.0)

    if pred_len_total == 0:
        brevity_penalty = 0.0
    elif pred_len_total > ref_len_total:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1 - ref_len_total / pred_len_total)

    scores: dict[str, float] = {}
    for n in range(1, 5):
        active = precisions[:n]
        if any(precision == 0.0 for precision in active):
            score = 0.0
        else:
            score = brevity_penalty * math.exp(sum(math.log(p) for p in active) / n)
        scores[f"BLEU-{n}"] = score
    return scores


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for token_a in a:
        prev = 0
        for index, token_b in enumerate(b, 1):
            saved = dp[index]
            if token_a == token_b:
                dp[index] = prev + 1
            else:
                dp[index] = max(dp[index], dp[index - 1])
            prev = saved
    return dp[-1]


def compute_rouge_l(predictions: dict[str, str], references: dict[str, list[str]]) -> float:
    """Compute average max ROUGE-L F1 over references."""
    scores: list[float] = []
    for image_id, prediction in predictions.items():
        if image_id not in references:
            continue
        pred_tokens = _tokens(prediction)
        best = 0.0
        for reference in references[image_id]:
            ref_tokens = _tokens(reference)
            lcs = _lcs_len(pred_tokens, ref_tokens)
            precision = lcs / len(pred_tokens) if pred_tokens else 0.0
            recall = lcs / len(ref_tokens) if ref_tokens else 0.0
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            best = max(best, f1)
        scores.append(best)
    return sum(scores) / len(scores) if scores else 0.0


def _fallback_meteor(prediction: str, refs: list[str]) -> float:
    pred = set(_tokens(prediction))
    if not pred:
        return 0.0
    best = 0.0
    for ref in refs:
        ref_tokens = set(_tokens(ref))
        overlap = len(pred & ref_tokens)
        precision = overlap / len(pred) if pred else 0.0
        recall = overlap / len(ref_tokens) if ref_tokens else 0.0
        score = 0.0 if precision + recall == 0 else 10 * precision * recall / (recall + 9 * precision)
        best = max(best, score)
    return best


def compute_meteor(predictions: dict[str, str], references: dict[str, list[str]]) -> float:
    """Compute METEOR with NLTK when possible, otherwise a unigram fallback."""
    scores: list[float] = []
    try:
        from nltk.translate.meteor_score import meteor_score

        for image_id, prediction in predictions.items():
            if image_id in references:
                scores.append(meteor_score([_tokens(ref) for ref in references[image_id]], _tokens(prediction)))
    except Exception:
        for image_id, prediction in predictions.items():
            if image_id in references:
                scores.append(_fallback_meteor(prediction, references[image_id]))
    return sum(scores) / len(scores) if scores else 0.0


def compute_cider(predictions: dict[str, str], references: dict[str, list[str]]):
    """CIDEr is optional; return None when pycocoevalcap is unavailable."""
    try:
        from pycocoevalcap.cider.cider import Cider
    except Exception:
        return None

    image_ids = [image_id for image_id in predictions if image_id in references]
    if not image_ids:
        return 0.0

    scorer = Cider()
    gts = {image_id: references[image_id] for image_id in image_ids}
    res = {image_id: [predictions[image_id]] for image_id in image_ids}
    score, _ = scorer.compute_score(gts, res)
    return float(score)


def compute_all_metrics(predictions: dict[str, str], references: dict[str, list[str]]) -> dict:
    metrics = compute_bleu(predictions, references)
    metrics["ROUGE-L"] = compute_rouge_l(predictions, references)
    metrics["METEOR"] = compute_meteor(predictions, references)
    metrics["CIDEr"] = compute_cider(predictions, references)
    return metrics
