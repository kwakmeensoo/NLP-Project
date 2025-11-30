# Experiment Module

Research evaluation and optimization scripts for the audio-to-slide matching algorithm.

## Overview

This module contains scripts for evaluating the slide matching algorithm on the MaViLS benchmark dataset, performing hyperparameter optimization, and conducting ablation studies to understand feature contributions.

## Files

### Core Scripts

- **`evaluate.py`**: Main evaluation script for benchmarking on MaViLS dataset
  - Computes accuracy, precision, recall metrics
  - Generates detailed per-lecture reports
  - Supports both quick (4 lectures) and full (20 lectures) evaluation modes

- **`grid_search.py`**: Hyperparameter optimization via grid search
  - Systematically explores parameter combinations
  - Identifies optimal hyperparameter settings
  - Outputs ranked configurations by accuracy

- **`ablation_study.py`**: Feature contribution analysis
  - Tests impact of disabling individual algorithm features
  - Quantifies contribution of each component
  - Helps understand which features matter most

- **`match.py`**: Core slide matching processor (research version)
  - Implements dynamic programming-based matching algorithm
  - Includes caching optimizations for faster batch evaluation
  - Same core algorithm as `inference/match.py` but with research-focused optimizations

### Dataset

- **`dataset/`**: MaViLS benchmark data
  - 20 academic lectures with ground truth annotations
  - Audio/video files, PDF slides, and Excel ground truth files
  - See [MaViLS repository](https://github.com/andererka/MaViLS) for details

## Quick Start

### Basic Evaluation

Run evaluation on first 4 lectures with default parameters:

```bash
python evaluate.py
```

### Full Benchmark Evaluation

Evaluate on all 20 lectures:

```bash
python evaluate.py --full
```

### Model Selection

Use 1B model instead of 3B (faster, lower VRAM):

```bash
python evaluate.py --model-size 1b
```

### Hyperparameter Tuning

Find optimal parameters via grid search:

```bash
python grid_search.py
```

### Ablation Study

Analyze feature contributions:

```bash
python ablation_study.py
```

## Command Reference

### evaluate.py

```bash
python evaluate.py [OPTIONS]
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--full` | Use all 20 lectures instead of first 4 | False |
| `--model-size {1b,3b}` | Model size for embeddings | `3b` |
| `--batch-size N` | Batch size for embedding computation | `4` |
| `--device {cuda,cpu}` | Device to run on | `cuda` |
| `--jump-penalty FLOAT` | Penalty for non-sequential slides | `1.5` |
| `--backward-weight FLOAT` | Multiplier for backward jumps | `1.85` |
| `--exponential-scale FLOAT` | Exponential similarity scaling | `2.785` |
| `--confidence-threshold FLOAT` | Threshold for confidence boost | `0.913` |
| `--confidence-weight FLOAT` | Weight for confidence boost | `2.18` |
| `--context-weight FLOAT` | Weight for context similarity | `0.04` |
| `--context-update-rate FLOAT` | EMA update rate for context | `0.24` |
| `--min-sentence-length INT` | Min words to use similarity | `2` |
| `--no-exponential-scaling` | Disable exponential scaling | False |
| `--no-confidence-boosting` | Disable confidence boost | False |
| `--no-context-similarity` | Disable context similarity | False |
| `--use-cached-similarity` | Use cached similarity matrices | True |

**Examples:**

```bash
# Quick evaluation with custom parameters
python evaluate.py --jump-penalty 2.0 --backward-weight 2.0

# Full evaluation with 1B model
python evaluate.py --full --model-size 1b

# Evaluation with all features disabled
python evaluate.py --no-exponential-scaling --no-confidence-boosting --no-context-similarity
```

### grid_search.py

```bash
python grid_search.py [OPTIONS]
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--model-size {1b,3b}` | Model size for embeddings | `3b` |
| `--batch-size N` | Batch size for embedding computation | `4` |
| `--device {cuda,cpu}` | Device to run on | `cuda` |

**Output:**
- Saves results to `grid_search_results_{timestamp}.json`
- Prints ranked configurations by accuracy
- Explores combinations of: jump penalty, backward weight, exponential scale, confidence threshold, confidence weight, context weight, context update rate

**Example:**

```bash
# Run grid search with 1B model
python grid_search.py --model-size 1b
```

### ablation_study.py

```bash
python ablation_study.py [OPTIONS]
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--model-size {1b,3b}` | Model size for embeddings | `3b` |
| `--batch-size N` | Batch size for embedding computation | `4` |
| `--device {cuda,cpu}` | Device to run on | `cuda` |
| `--full` | Use all 20 lectures instead of first 4 | False |

**Output:**
- Saves results to `ablation_results/ablation_study_{timestamp}.json`
- Tests baseline + all feature combinations
- Shows accuracy impact of each feature

**Example:**

```bash
# Run ablation study on full dataset
python ablation_study.py --full
```

## Output Files

Evaluation results are saved to `evaluation_results/` in run-specific subdirectories:

```
evaluation_results/
└── run_YYYYMMDD_HHMMSS/
    ├── evaluation_summary.json          # Summary statistics (JSON)
    ├── cryptocurrency.xlsx              # Per-lecture detailed results
    ├── psychology.xlsx
    ├── reinforcement_learning.xlsx
    └── ...
```

Each Excel file contains:
- **Predictions sheet**: Sentence-by-sentence results with ground truth comparison
- **Summary sheet**: Aggregate metrics (accuracy, MAE, RMSE)
- **Errors sheet**: Only incorrect predictions with detailed analysis

Grid search results are saved to `grid_search_results/`:
```
grid_search_results/
└── grid_search_YYYYMMDD_HHMMSS.json
```

Ablation study results are saved to `ablation_results/`:
```
ablation_results/
└── ablation_study_YYYYMMDD_HHMMSS.json
```

## Metrics

The evaluation computes:

- **Accuracy**: Percentage of correctly matched slides
- **Precision/Recall/F1**: Standard classification metrics
- **Per-lecture breakdown**: Performance on each lecture
- **Error analysis**: Common failure patterns

## Default Hyperparameters

The default parameters (optimized via grid search):

```python
jump_penalty = 1.5              # Penalty for skipping slides
backward_weight = 1.85          # Multiplier for backward jumps
exponential_scale = 2.785       # Exponential similarity scaling
confidence_threshold = 0.913    # Threshold for confidence boost
confidence_weight = 2.18        # Weight for confidence boost
context_weight = 0.04           # Weight for context similarity (EMA)
context_update_rate = 0.24      # EMA update rate
min_sentence_length = 2         # Min words to use similarity score
```

## Performance Tips

1. **Memory Management**: The script automatically manages GPU memory, clearing cache between lectures
2. **Caching**: Similarity matrices are cached by default for faster repeated evaluations. Disable with `--use-cached-similarity False`
3. **Model Selection**: Use 1B model for faster iteration during development, 3B for final benchmarks
4. **Batch Size**: Increase batch size for faster embedding computation if VRAM allows

## Dataset Format

Ground truth files (Excel format):
- Column `Value`: Sentence text
- Column `Slidenumber`: Ground truth page number (1-indexed, -1 means unevaluated/excluded)

Dataset structure:
```
dataset/
├── lectures/
│   ├── cryptocurrency.pdf
│   ├── psychology.pdf
│   ├── reinforcement_learning.pdf
│   └── ...
├── ground_truth_files/
│   ├── ground_truth_cryptocurrency_MIT.xlsx
│   ├── ground_truth_psychology.xlsx
│   ├── ground_truth_reinforcement_learning.xlsx
│   └── ...
└── audioscripts/
    ├── cryptocurrency.srt
    ├── psychology_MIT.srt
    ├── reinforcement_learning_silver.srt
    └── ...
```

## Troubleshooting

**Out of Memory Errors**:
- Reduce batch size: `--batch-size 2`
- Use 1B model: `--model-size 1b`
- Disable caching: `--use-cached-similarity False`

**Slow Evaluation**:
- Enable caching: `--use-cached-similarity True` (default)
- Increase batch size if VRAM allows
- Use 1B model for faster iteration

**Missing Ground Truth**:
- Ensure Excel files are in `dataset/ground_truth_files/`
- Check that `Slidenumber` column exists
- Sentences with `Slidenumber == -1` are automatically excluded

## Research Workflow

Typical workflow for algorithm development:

1. **Quick iteration**: `python evaluate.py` (4 lectures, 3B model)
2. **Hyperparameter tuning**: `python grid_search.py`
3. **Feature analysis**: `python ablation_study.py`
4. **Final benchmark**: `python evaluate.py --full` (20 lectures)
5. **Sync changes**: Copy algorithm changes to `inference/match.py`

## Citation

If using the MaViLS dataset, please cite:

```bibtex
@inproceedings{anderer24_interspeech,
  title = {MaViLS, a Benchmark Dataset for Video-to-Slide Alignment, Assessing Baseline Accuracy with a Multimodal Alignment Algorithm Leveraging Speech, OCR, and Visual Features},
  author = {Katharina Anderer and Andreas Reich and Matthias Wölfel},
  year = {2024},
  booktitle = {Interspeech 2024},
  pages = {1375--1379},
  doi = {10.21437/Interspeech.2024-978},
  issn = {2958-1796}
}
```

**Dataset Links:**
- Paper: [Interspeech 2024](https://www.isca-archive.org/interspeech_2024/anderer24_interspeech.html) | [arXiv](https://arxiv.org/abs/2409.16765)
- Code: [GitHub](https://github.com/andererka/MaViLS)
- License: Dataset is available under permissive terms for research use
