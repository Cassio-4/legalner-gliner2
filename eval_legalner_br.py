"""Evaluate a GLiNER2 extractor on a Brazilian legal NER dataset (LeNER-Br or UlyssesNER-Br).

Follows the entity-extraction workflow from ``tutorial/2-ner.md``: entity
types are declared with Portuguese descriptions via ``create_schema()``, and
``batch_extract_entities(..., include_spans=True)`` is used to get
character-level predictions that can be scored against the CoNLL gold
annotations.

The data location, split files, entity descriptions and query names come from
each dataset's description module (see ``DATASETS``), e.g.
``data/lener/lener_dataset.py``.

``evaluate()`` can also be imported and called directly (``train_lener_br.py``
does so for its ``--eval-test`` step); it returns the metrics dict.

Example:
    python scripts/eval_lener_br.py --dataset lener --split test --model fastino/gliner2.5-multi-v1
    python scripts/eval_lener_br.py --dataset ulysses --split test --model ./ulysses_model/best --device cuda
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Sequence, Tuple

from gliner2 import AutoExtractor
from gliner2.training.metrics import f1_from_counts

REPO_ROOT = Path(__file__).resolve().parent.parent

# Dataset name -> description module defining DATA_DIR, SPLIT_FILES,
# ENTITY_DESCRIPTIONS and ENTITY_NAMES.
DATASETS = {
    "lener": "data.lener.lener_dataset",
    "ulysses": "data.ulyssesbr.pl_corpus_v2_categories.ulysses_ner_br",
}


def load_dataset_module(name: str) -> ModuleType:
    """Import the description module for dataset ``name`` (a key of ``DATASETS``)."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module(DATASETS[name])


def parse_conll(path: Path) -> List[Tuple[List[str], List[str]]]:
    """Parse a whitespace-separated CoNLL file into (tokens, tags) sentences."""
    sentences: List[Tuple[List[str], List[str]]] = []
    tokens: List[str] = []
    tags: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if tokens:
                    sentences.append((tokens, tags))
                    tokens, tags = [], []
                continue
            parts = line.split()
            tokens.append(parts[0])
            tags.append(parts[-1])
    if tokens:
        sentences.append((tokens, tags))
    return sentences


def bio_to_token_spans(tags: Sequence[str]) -> List[Tuple[int, int, str]]:
    """Convert BIO tags to half-open ``(start, end, label)`` token spans.

    A stray ``I-`` tag with no matching ``B-`` (or a type mismatch) starts a
    new span rather than being dropped, since the LeNER-Br release has a
    handful of such inconsistencies (and UlyssesNER-Br has a few as well).
    """
    spans: List[Tuple[int, int, str]] = []
    start: int | None = None
    label: str | None = None
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag.startswith("B-"):
            if start is not None:
                spans.append((start, i, label))
            start, label = i, tag[2:]
        elif tag.startswith("I-"):
            current = tag[2:]
            if start is None or current != label:
                if start is not None:
                    spans.append((start, i, label))
                start, label = i, current
        else:
            if start is not None:
                spans.append((start, i, label))
                start, label = None, None
    return spans


def sentence_to_text_and_gold(
    tokens: Sequence[str], tags: Sequence[str]
) -> Tuple[str, List[Tuple[int, int, str]]]:
    """Reconstruct a space-joined text and the matching gold character spans."""
    offsets = []
    cursor = 0
    for tok in tokens:
        offsets.append((cursor, cursor + len(tok)))
        cursor += len(tok) + 1  # +1 for the single joining space
    text = " ".join(tokens)
    gold = [
        (offsets[s][0], offsets[e - 1][1], label)
        for s, e, label in bio_to_token_spans(tags)
    ]
    return text, gold


def load_split(ds: ModuleType, split: str, limit: int | None = None):
    path = Path(ds.DATA_DIR) / ds.SPLIT_FILES[split]
    sentences = parse_conll(path)
    if limit is not None:
        sentences = sentences[:limit]
    texts, golds = [], []
    for tokens, tags in sentences:
        text, gold = sentence_to_text_and_gold(tokens, tags)
        unknown = {label for _, _, label in gold} - set(ds.ENTITY_DESCRIPTIONS)
        if unknown:
            raise ValueError(f"{path}: unexpected entity types {sorted(unknown)}")
        texts.append(text)
        golds.append(gold)
    return texts, golds


def evaluate(
    model: str,
    dataset: str = "lener",
    split: str = "test",
    threshold: float = 0.5,
    batch_size: int = 16,
    device: str = "cpu",
    max_samples: int | None = None,
    output: str | Path | None = None,
    dump: bool = False,
) -> Dict:
    """Score ``model`` on ``split`` of ``dataset``, print a report and return the metrics.

    Per-entity metrics are keyed by CoNLL tag (e.g. ``ORGANIZACAO``); the model
    is queried with the dataset's ``ENTITY_NAMES`` and ``ENTITY_DESCRIPTIONS``.
    """
    ds = load_dataset_module(dataset)
    labels = list(ds.ENTITY_DESCRIPTIONS)
    entity_types = {ds.ENTITY_NAMES[label]: desc for label, desc in ds.ENTITY_DESCRIPTIONS.items()}

    texts, golds = load_split(ds, split, limit=max_samples)
    print(f"Loaded {len(texts)} sentences from the '{split}' split of {dataset}.")

    print(f"Loading {model} (device={device})...")
    extractor = AutoExtractor.from_pretrained(model, map_location=device)

    t0 = time.time()
    results = extractor.batch_extract_entities(
        texts,
        entity_types,
        batch_size=batch_size,
        threshold=threshold,
        include_spans=True,
    )
    elapsed = time.time() - t0
    rate = len(texts) / elapsed if elapsed else 0.0
    print(f"Inference on {len(texts)} sentences took {elapsed:.1f}s ({rate:.1f} sents/s)")

    tp = {label: 0 for label in labels}
    fp = {label: 0 for label in labels}
    fn = {label: 0 for label in labels}
    predictions_dump = []

    for text, gold, result in zip(texts, golds, results):
        gold_by_label: Dict[str, set] = {label: set() for label in labels}
        for s, e, label in gold:
            gold_by_label[label].add((s, e))

        entities = result.get("entities", {})
        pred_by_label: Dict[str, set] = {label: set() for label in labels}
        for label in labels:
            for item in entities.get(ds.ENTITY_NAMES[label], []):
                if isinstance(item, dict):
                    pred_by_label[label].add((item["start"], item["end"]))

        for label in labels:
            g, p = gold_by_label[label], pred_by_label[label]
            tp[label] += len(g & p)
            fp[label] += len(p - g)
            fn[label] += len(g - p)

        if dump:
            predictions_dump.append(
                {
                    "text": text,
                    "gold": sorted(gold),
                    "predicted": sorted(
                        (s, e, label) for label in labels for s, e in pred_by_label[label]
                    ),
                }
            )

    header = f"{'Entity':<16}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}"
    print()
    print(header)
    print("-" * len(header))

    per_entity = {}
    total_tp = total_fp = total_fn = 0
    for label in labels:
        p, r, f1 = f1_from_counts(tp[label], fp[label], fn[label])
        support = tp[label] + fn[label]
        per_entity[label] = {"precision": p, "recall": r, "f1": f1, "support": support}
        print(f"{label:<16}{p:>10.3f}{r:>10.3f}{f1:>10.3f}{support:>10d}")
        total_tp += tp[label]
        total_fp += fp[label]
        total_fn += fn[label]

    print("-" * len(header))
    micro_p, micro_r, micro_f1 = f1_from_counts(total_tp, total_fp, total_fn)
    total_support = total_tp + total_fn
    print(f"{'MICRO AVG':<16}{micro_p:>10.3f}{micro_r:>10.3f}{micro_f1:>10.3f}{total_support:>10d}")

    macro_p = sum(per_entity[l]["precision"] for l in labels) / len(labels)
    macro_r = sum(per_entity[l]["recall"] for l in labels) / len(labels)
    macro_f1 = sum(per_entity[l]["f1"] for l in labels) / len(labels)
    print(f"{'MACRO AVG':<16}{macro_p:>10.3f}{macro_r:>10.3f}{macro_f1:>10.3f}{total_support:>10d}")

    out = {
        "model": str(model),
        "dataset": dataset,
        "split": split,
        "threshold": threshold,
        "n_sentences": len(texts),
        "elapsed_seconds": elapsed,
        "per_entity": per_entity,
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
    }
    if dump:
        out["predictions"] = predictions_dump
    if output:
        Path(output).write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"\nSaved results to {output}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        default="fastino/gliner2.5-multi-v1",
        help="GLiNER2 checkpoint to evaluate (multilingual recommended for pt-BR)",
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="lener")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        default="cpu",
        help="map_location passed to AutoExtractor.from_pretrained, e.g. 'cpu' or 'cuda'",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N sentences (quick smoke test)",
    )
    parser.add_argument("--output", default=None, help="Optional path to dump metrics as JSON")
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Include per-sentence gold/predicted spans in --output JSON",
    )
    args = parser.parse_args()
    evaluate(**vars(args))


if __name__ == "__main__":
    main()
