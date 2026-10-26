"""Fine-tune a GLiNER2 extractor on LeNER-Br or UlyssesNER-Br.

Follows ``tutorial/9-training.md``: loads the GLiNER2 JSONL produced by the
dataset's ``conll_to_gliner2.py`` (generated on the fly if missing), attaches
the Portuguese entity descriptions from the dataset's description module (the
same ones ``eval_lener_br.py`` uses), and trains with ``ExtractorTrainer``
using the dev split for evaluation and best-checkpoint selection.

Entity type names are normalised to the lowercase, accent-free form that
``eval_lener_br.py`` queries (e.g. ``organização`` -> ``organizacao``), so the
resulting checkpoint can be scored directly with:

    python scripts/eval_lener_br.py --dataset <dataset> --model <output-dir>/best --device cuda

Example:
    python scripts/train_lener_br.py --output-dir ./lener_model --epochs 10 --eval-test
    python scripts/train_lener_br.py --dataset ulysses --output-dir ./ulysses_model --eval-test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import unicodedata
from dataclasses import asdict
from pathlib import Path

import torch
from gliner2 import AutoExtractor
from gliner2.training.data import InputExample, TrainingDataset
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

import eval_legalner_br


def normalize_label(label: str) -> str:
    """``organização`` -> ``organizacao`` (matches eval_lener_br's query names)."""
    decomposed = unicodedata.normalize("NFKD", label)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def ensure_jsonl(data_dir: Path) -> None:
    missing = [s for s in ("train", "dev", "test") if not (data_dir / f"{s}.jsonl").exists()]
    if missing:
        print(f"Missing {missing} JSONL in {data_dir}; running conll_to_gliner2.py...")
        subprocess.run(
            [sys.executable, str(data_dir / "conll_to_gliner2.py"), str(data_dir), str(data_dir)],
            check=True,
        )


def load_split(path: Path, descriptions: dict[str, str], use_descriptions: bool) -> TrainingDataset:
    """Load a JSONL split; ``descriptions`` maps entity name (as eval queries it) -> description."""
    examples = []
    for ex in TrainingDataset.load(path):
        entities = {normalize_label(k): v for k, v in ex.entities.items()}
        unknown = set(entities) - set(descriptions)
        if unknown:
            raise ValueError(f"{path}: unexpected entity types {sorted(unknown)}")
        examples.append(
            InputExample(
                text=ex.text,
                entities=entities,
                entity_descriptions=dict(descriptions) if use_descriptions else None,
            )
        )
    return TrainingDataset(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        default="fastino/gliner2.5-multi-v1",
        help="Base checkpoint (multilingual recommended for pt-BR)",
    )
    parser.add_argument("--dataset", choices=sorted(eval_legalner_br.DATASETS), default="lener")
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="JSONL/CoNLL folder; default: the dataset module's DATA_DIR"
    )
    parser.add_argument("--output-dir", default=None, help="Default: ./<dataset>_model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--task-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01, help="0 makes AdamW plain Adam")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=0, help="Overrides --warmup-ratio when > 0")
    parser.add_argument("--scheduler", default="cosine", choices=["linear", "cosine", "cosine_restarts", "constant"])
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="Word-token cap per sentence (train has a few sentences up to 755 tokens); default: model config",
    )
    parser.add_argument(
        "--allow-invalid-samples",
        action="store_true",
        help="Drop entities cut off by --max-len (and skip unprocessable samples) instead of raising",
    )
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--early-stopping-patience", type=int, default=3, help="0 disables early stopping")
    parser.add_argument(
        "--early-stopping-threshold", type=float, default=0.0, help="Min dev-loss decrease that resets patience"
    )
    parser.add_argument("--no-descriptions", action="store_true", help="Train without entity descriptions")
    parser.add_argument("--max-train-samples", type=int, default=-1, help="Subset for smoke tests")
    parser.add_argument("--max-eval-samples", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    # LoRA
    parser.add_argument("--lora", action="store_true", help="Parameter-efficient fine-tuning")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    # Logging
    parser.add_argument("--wandb-project", default=None, help="Enable W&B logging to this project")
    parser.add_argument("--run-name", default=None)
    # Final evaluation
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="After training, score <output-dir>/best on the dataset's test split with eval_lener_br",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for --eval-test")
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=4,
        help="Inference batch for --eval-test (16 OOMs on 12 GB with the 654-token test sentence)",
    )
    args = parser.parse_args()

    ds = eval_legalner_br.load_dataset_module(args.dataset)
    descriptions = {ds.ENTITY_NAMES[tag]: desc for tag, desc in ds.ENTITY_DESCRIPTIONS.items()}
    if args.data_dir is None:
        args.data_dir = Path(ds.DATA_DIR)
    if args.output_dir is None:
        args.output_dir = f"./{args.dataset}_model"

    ensure_jsonl(args.data_dir)
    use_descriptions = not args.no_descriptions
    train_data = load_split(args.data_dir / "train.jsonl", descriptions, use_descriptions)
    dev_data = load_split(args.data_dir / "dev.jsonl", descriptions, use_descriptions)
    print(f"Loaded {len(train_data)} train / {len(dev_data)} dev examples from {args.data_dir}")
    train_data.print_stats()

    model = AutoExtractor.from_pretrained(args.model)

    config = TrainingConfig(
        output_dir=args.output_dir,
        experiment_name=args.run_name or f"{args.dataset}_br",
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        encoder_lr=args.encoder_lr,
        task_lr=args.task_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        scheduler_type=args.scheduler,
        fp16=args.precision == "fp16",
        bf16=args.precision == "bf16",
        max_len=args.max_len,
        allow_invalid_samples=args.allow_invalid_samples,
        eval_strategy="epoch",  # evaluate + checkpoint at the end of every epoch
        save_best=True,
        save_total_limit=2,
        early_stopping=args.early_stopping_patience > 0,
        early_stopping_patience=max(args.early_stopping_patience, 1),
        early_stopping_threshold=args.early_stopping_threshold,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        num_workers=args.num_workers,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to_wandb=args.wandb_project is not None,
        wandb_project=args.wandb_project,
        wandb_run_name=args.run_name,
        wandb_tags=[f"{args.dataset}-br", "ner"],
        use_lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        # Save full (merged) weights so best/ loads with AutoExtractor.from_pretrained.
        save_adapter_only=False,
    )

    trainer = ExtractorTrainer(model, config)
    results = trainer.train(train_data=train_data, eval_data=dev_data)

    print("\nTraining completed!")
    print(f"Best dev loss: {results['best_metric']:.4f}")
    print(f"Total steps: {results['total_steps']}  epochs: {results['total_epochs']}")
    print(f"Training time: {results['total_time_seconds'] / 60:.1f} min")

    final_results = {k: v for k, v in results.items() if not k.endswith("_history")}
    summary = final_results | {"args": {k: str(v) for k, v in vars(args).items()}}
    Path(args.output_dir, "train_summary.json").write_text(json.dumps(summary, indent=2))

    best_dir = Path(args.output_dir) / "best"
    if not best_dir.exists():
        print(f"No {best_dir} checkpoint found; skipping experiment config and test evaluation.")
        return

    # Keep the full experiment setup next to the weights so best/ is self-describing
    # even if copied away from the output dir.
    experiment = {
        "base_model": args.model,
        "dataset": args.dataset,
        "args": vars(args),
        "training_config": asdict(config),
        "entity_descriptions": descriptions if use_descriptions else None,
        "results": final_results,
    }
    experiment_path = best_dir / "experiment_config.json"
    experiment_path.write_text(json.dumps(experiment, indent=2, ensure_ascii=False, default=str))
    print(f"Saved experiment config to {experiment_path}")

    if args.eval_test:
        # Free the training copy (and optimizer state) before loading best/.
        del trainer, model
        torch.cuda.empty_cache()
        eval_legalner_br.evaluate(
            model=str(best_dir),
            dataset=args.dataset,
            split="test",
            threshold=args.threshold,
            batch_size=args.test_batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            output=Path(args.output_dir) / "test_metrics.json",
        )


if __name__ == "__main__":
    main()
