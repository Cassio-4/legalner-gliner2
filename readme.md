# legalner-gliner2

Fine-tunes GLiNER2 extractors on Brazilian legal NER datasets (LeNER-Br, UlyssesNER-Br)
and scores the best dev checkpoint on the test split.

| Path | Purpose |
|---|---|
| `train_legalner_br.py` | Training script (ExtractorTrainer; dev loss picks `best/`) |
| `eval_legalner_br.py` | Test-split scorer (writes `test_metrics.json`) |
| `run_legalner_deberta_base.sh` | Launcher for LeNER-Br runs (`SEED=...`, `GPU=...`) |
| `output/<EXP-id>/<run>/` | `training_config.json`, `train_summary.json`, `test_metrics.json`, `train.log`, `best/` |

## Experiments

### How to log a run here

- Group runs whose configs are identical except for the seed. Report each seed and then
  mean ± std over the seeds.
- Std is the sample standard deviation (n-1).
- F1 is entity-level micro F1 on the test split, with threshold 0.5.
- Put each experiment's runs under `output/<EXP-id>/`, one folder per seed.
- The source of truth is `output/<EXP-id>/<run>/test_metrics.json` (`"micro"` → `"f1"`).

### Shared settings

These apply to every experiment below unless the experiment says otherwise.

| Setting | Value |
|---|---|
| Training | Full fine-tuning (no LoRA) |
| Precision | bf16 (AMP; TF32 enabled for fp32 matmuls) |
| Optimizer | AdamW, `encoder_lr` = `task_lr` = 2e-5, weight decay 0, max grad norm 1.0 |
| Scheduler | Cosine, 500 warmup steps |
| Epochs | Max 30; early stopping on dev `eval_loss` (patience 5, min delta 0.0005) |
| Checkpoint | `best/` = lowest dev `eval_loss` (evaluated each epoch) |
| Max length | 512 word tokens, with `--allow-invalid-samples` |
| Descriptions | Portuguese entity descriptions ON |
| Eval threshold | 0.5 |

### EXP-01: UlyssesNER-Br, gliner2.5-base-v1 (DeBERTa-v3-base), effective batch 16

| Setting | Value |
|---|---|
| Model | `fastino/gliner2.5-base-v1` |
| Dataset | `ulysses` (test: 592 sentences) |
| Batch | 8 × grad accum 2 = effective 16, eval batch 8. Seed 0 used 4 × 4 with eval batch 4 (see notes). |
| Runs | `output/EXP-01/deberta_base_ulysses_s{0,42,135,295,515,580,584,736,780,914}` |

| Seed | Micro F1 | Epochs | Best dev loss | Train time (min) |
|---:|---:|---:|---:|---:|
| 0 | 0.8583 | 12 | 33.27 | 25.6 |
| 42 | 0.8602 | 13 | 66.16 | 15.2 |
| 135 | 0.8468 | 13 | 67.46 | 15.2 |
| 295 | 0.8635 | 15 | 65.82 | 17.6 |
| 515 | 0.8435 | 10 | 69.51 | 11.6 |
| 580 | 0.8441 | 10 | 68.98 | 11.6 |
| 584 | 0.8564 | 14 | 67.70 | 15.1 |
| 736 | 0.8399 | 10 | 67.48 | 12.1 |
| 780 | 0.8501 | 11 | 66.98 | 13.4 |
| 914 | 0.8457 | 12 | 67.77 | 14.0 |

**Micro F1 (n=10): 0.8508 ± 0.0081** (min 0.8399, max 0.8635)

Notes:

- Seed 0 used a per-device batch of 4 × 4 instead of 8 × 2. The effective batch (16) is
  the same, so it is grouped here. Without seed 0 (n=9), micro F1 is 0.8500 ± 0.0082.
- Seed 0's dev loss (33.27) is not comparable to the other seeds' (~67). `eval_loss` is
  the mean of per-batch total losses, and these appear to scale with eval batch size
  (4 for seed 0, 8 for the rest). That would explain the roughly 2× gap.

### EXP-02: LeNER-Br, gliner2.5-multi-v1 (mDeBERTa-v3-base), effective batch 16

| Setting | Value |
|---|---|
| Model | `fastino/gliner2.5-multi-v1` |
| Dataset | `lener` (test: 1389 sentences) |
| Batch | 8 × grad accum 2 = effective 16, eval batch 8 |
| Launcher | `run_legalner_deberta_base.sh` |
| Runs | `output/EXP-02/gliner2_mDeBERTa_base_lener_s{0,42}` |

| Seed | Micro F1 | Epochs | Best dev loss | Train time (min) |
|---:|---:|---:|---:|---:|
| 0 | 0.8944 | 7 | 102.23 | 50.9 |
| 42 | 0.9006 | 30 | 73.72 | 234.5 |

**Micro F1 (n=2): 0.8975 ± 0.0044**

Notes:

- Seed 0 early-stopped at epoch 7. Its best dev loss came at epoch 2, and it never
  improved after that.
- Seed 42 ran all 30 epochs and was still improving at epoch 30 (dev loss 75.9 → 73.7).
- Despite the very different dev losses, test micro F1 differs by only 0.6 points.
- With n=2 the std is not reliable. More seeds are needed.
