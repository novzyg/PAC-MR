# PAC-MR: Patient-Aware Conflict Adjustment for Safe Medication Recommendation

This repository contains the research code for **PAC-MR**, a two-stage medication recommendation framework that models patient medication demands and refines initial predictions through patient-aware conflict adjustment.

## ✨ Overview

Medication recommendation must account for both a patient's medication demands and potential drug–drug interactions (DDIs). A known DDI relation alone does not determine how strongly a drug pair should be adjusted for a particular patient, or how the adjustment should be distributed between its medications.

PAC-MR addresses these two questions through **patient-conditioned conflict budgeting** and **cost-guided asymmetric adjustment**. It first estimates medication probabilities from diagnoses, procedures, visit history, and drug graphs. It then determines an adjustment budget for each known DDI pair, allocates that budget between its endpoints, and applies bounded updates to medication representations before producing final predictions.

Medication labels provide supervision only; they are not used as inference inputs. Conflict budgets and adjustment costs are learned recommendation quantities, not clinical adverse-event probabilities or direct measures of clinical necessity.

## 🏆 Key Contributions

- We propose **PAC-MR**, a two-stage medication recommendation framework that first estimates patient medication demands and then refines initial predictions through patient-conditioned conflict adjustment.
- We introduce **Patient-Conditioned Conflict Budgeting** to adapt the adjustment strength of each known DDI relation using joint recommendation tendencies and patient and drug-pair context.
- We develop **cost-guided asymmetric allocation** with a closed-form solution and bounded representation updates to account for unequal recommendation support and control cumulative adjustment across multiple conflicts.
- Experiments on **MIMIC-III and MIMIC-IV**, as reported in the manuscript, achieve the highest mean Jaccard, F1, and PRAUC among the evaluated baselines, with competitive DDI rates. 
---

## 📊 Results Reported in the Manuscript

The supplied manuscript evaluates PAC-MR on MIMIC-III and MIMIC-IV. Table 1 reports the following results, in percent, as mean ± standard deviation over five random seeds:

| Dataset | Jaccard ↑ | F1 ↑ | PRAUC ↑ | DDI ↓ |
|:---|---:|---:|---:|---:|
| MIMIC-III | 33.46 ± 0.39 | 48.81 ± 0.41 | 56.50 ± 0.52 | 6.73 ± 0.30 |
| MIMIC-IV | 38.11 ± 0.03 | 53.90 ± 0.03 | 60.51 ± 0.003 | 4.75 ± 0.01 |

These are manuscript-reported results, not results independently reproduced from this checkout. The repository currently does not include the corresponding experiment logs, pretrained checkpoints, or a complete five-seed reproduction configuration.

## 📖 Usage

### Installation

Clone the repository and install dependencies:
```
# Clone the repository
cd PAC-MR
# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

Prepare preprocessed patient records and medication graphs. Each dataset directory must contain:

| File | Contents |
|:---|:---|
| `records_final.pkl` | Patient visit sequences containing diagnosis, procedure, and medication indices |
| `voc_final.pkl` | Diagnosis, procedure, and medication vocabularies |
| `ddi_A_final.pkl` | Medication DDI adjacency matrix |

The manuscript uses the following preprocessed datasets:

| Dataset | Patients | Visits | Diagnoses | Procedures | Medications |
|:---|---:|---:|---:|---:|---:|
| MIMIC-III | 6,360 | 16,976 | 4,672 | 1,420 | 718 |
| MIMIC-IV | 8,949 | 24,106 | 11,030 | 4,810 | 877 |

These counts describe the manuscript's preprocessing version, not every distribution of MIMIC. Raw datasets and preprocessing scripts are not included. The code builds the medication co-occurrence graph from training patients only.

Set the path to your prepared dataset:

```bash
export DATA_DIR=/absolute/path/to/dataset
```

The Shell scripts otherwise default to `data/mimic-iv_all_all/`. To inspect the split and data summary:

```bash
python src/main.py prepare --data "$DATA_DIR" --output saved/data_summary
```

`prepare` operates on preprocessed files; it does not transform raw EHR data.

### Training and Evaluation

1. **Set the output directory and device** in the same terminal:

   ```bash
   export OUTPUT_DIR="$PWD/saved/run_001"
   export DEVICE=cuda:0
   ```

2. **Run the two training stages:**

   ```bash
   bash scripts/train.sh
   ```

   The script trains the `base` model, then initializes `full` from `base/best.pt` and trains its adjustment parameters with the backbone frozen.

3. **Evaluate on the test set:**

   ```bash
   bash scripts/evaluate.sh
   ```

   Evaluation loads `full/best.pt` and the threshold selected during validation. Results are saved in `full/test_metrics.json` and `full/test_predictions.npz`.

The commands above use the current script defaults, which differ from the manuscript's experiment settings. Additional training arguments apply to both stages:

```bash
DIM=256 SEED=42 DDI_WEIGHT=1.0 bash scripts/train.sh \
  --dropout 0.2 --batch-size 4 --eval-batch-size 8
```

Other supported environment variables include `GRAPH_LAYERS`, `ATTENTION_HEADS`, `BASE_EPOCHS`, and `ADJUST_EPOCHS`. The embedding dimension must be divisible by the number of attention heads.

```bash
python src/main.py train --help
python src/main.py evaluate --help
```

### Ablation Variants

The model provides the following variants through `python src/main.py train --variant ...`:

| Manuscript ablation | Code setting | Behavior |
|:---|:---|:---|
| Backbone only | `base` | Predict without conflict adjustment |
| w/o Relation Adaptation | `allocation_only` | Fix relation coefficients to 1; learn allocation costs |
| w/o Asymmetric Allocation | `edge_only` | Learn relation coefficients; allocate budgets equally |
| w/o Both | `uniform` | Fix relation coefficients to 1 and use equal allocation |
| w/o DDI Loss | `full --ddi-weight 0` | Keep full adjustment architecture without its DDI loss term |
| Full model | `full` | Learn relation coefficients and allocation costs |

All adjustment variants require `--init` pointing to a compatible base checkpoint. `scripts/train.sh` is the convenience entry for the base/full sequence; use the Python CLI directly for other variants. Architecture, dataset, and split must match the base run. Automated ablation, grid-search, and sensitivity-analysis scripts are not included in the current checkout.

### Model Checks

Run the synthetic checks after installing dependencies:

```bash
python tests/test_model.py
```

The checks cover score descent, frozen-backbone gradients, label-input isolation, padding, empty codes, training-only graph construction, parameter bounds, and checkpoint reloading.

For a small end-to-end CPU check with your prepared dataset:

```bash
DATA_DIR="$DATA_DIR" DEVICE=cpu OUTPUT_DIR="$PWD/saved/smoke" \
  BASE_EPOCHS=1 ADJUST_EPOCHS=1 bash scripts/train.sh --smoke
DEVICE=cpu OUTPUT_DIR="$PWD/saved/smoke" bash scripts/evaluate.sh
```

`--smoke` uses the first six patients and limits training steps. It checks execution, not recommendation performance.

---

## 🏗️ Project Structure

```text
PAC-MR/
├── src/
│   ├── main.py          # CLI: prepare / train / evaluate
│   ├── trainer.py       # Optimization, validation, testing, and checkpoint I/O
│   ├── models.py        # Medication effect modeling and conflict adjustment
│   ├── data.py          # Data loading, patient splits, batching, and graph construction
│   └── metrics.py       # Recommendation metrics and validation selection
├── scripts/
│   ├── env.sh           # Shared paths, interpreter, device, and thread settings
│   ├── train.sh         # Base training followed by full-model adjustment
│   └── evaluate.sh      # Independent test evaluation
├── tests/
│   └── test_model.py    # Synthetic implementation checks
├── requirements.txt
├── .gitignore
└── README.md
```

Data and generated run artifacts are excluded from Git. A run stores its configuration, environment, source/data fingerprints, checkpoints, predictions, and diagnostic files under the output directory.

## 📝 Evaluation and Implementation Notes

**Evaluation.** The default ordered patient split uses the first two-thirds for training, half of the remainder for testing, and the rest for validation. Checkpoint selection maximizes validation Jaccard, with lower DDI as the tie-breaker. Predictions use threshold 0.5 by default. Accuracy metrics are averaged by patient; DDI is computed globally over predicted medication pairs. The field `prauc` uses `average_precision_score` (AP).

**Training settings.** The manuscript and the current `train.sh` defaults differ:

| Setting | Manuscript experimental setup (§3.1) | Current script defaults |
|:---|:---|:---|
| Base / adjustment epochs | 50 / 30 | 30 / 20 |
| Base / adjustment learning rate | 0.0003 / 0.0003 | 0.0003 / 0.001 |
| Displacement cap | 0.2 | 0.1 |
| Random seeds | Five seeds; values not listed | One run, seed 42 |

Both specify dimension 256, two graph layers, four attention heads, dropout 0.2, batch size 4, DDI loss weight 1.0, and cost ratio 4.0. The manuscript's method section states a displacement cap of 0.1, while its experimental setup states 0.2; the table above follows the experimental setup.

**Architecture correspondence.** The code follows the three-module design, with details that differ from the manuscript equations. Its patient vector fuses the current visit, final GRU state, and attention-pooled visit states, rather than using the final GRU state alone. Its relation network explicitly concatenates the patient vector with the symmetric drug-pair features (4d inputs), while Equation (1) describes only the pair features (3d inputs). The cost network uses hidden dimension 32. These differences should be resolved before treating this checkout as an exact reproduction of the manuscript.

**Checkpoint compatibility.** Changes to source code, data, or configuration require a new output directory. Old non-default cap/cost checkpoints without a parameter-wiring marker are rejected. Pre-reorganization runs have different source fingerprints, and older checkpoints may contain absolute server paths. Single-run test evaluation refuses to overwrite existing results.

## 📄 Paper

**PAC-MR: Patient-Aware Conflict Adjustment for Safe Medication Recommendation**

The method description, dataset statistics, and reported results above are based on the author-provided manuscript. Formal publication and citation metadata will be added when confirmed.
