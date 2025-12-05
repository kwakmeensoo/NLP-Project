# Audio-to-Slide Matching via Multimodal Embeddings and Dynamic Programming

A research project for automatic alignment of lecture audio transcripts to presentation slides using multimodal retrieval models and dynamic programming.

## Motivation

### Problem: Lecture Video Review is Cumbersome

Traditional lecture video recording systems face significant challenges:

- **Technical failures**: Recording errors require re-shooting entire lectures
- **Privacy concerns**: Instructors' image rights and consent issues
- **Infrastructure requirements**: Need professional recording facilities
- **Storage & bandwidth**: Large video files are expensive to store and stream

### Our Proposal

Use **audio-only recordings** to align with lecture slide PDFs:

- **Minimal recording setup**: Just a microphone
- **No privacy/consent issues**: No instructor images required
- **Much smaller file sizes**: Easier distribution and storage
- **Enables automated lecture review materials**: Students can navigate lectures via slides

This approach makes lecture recording more accessible while maintaining educational value through automatic slide-audio alignment.

## Overview

This project addresses the challenge of matching spoken lecture content to corresponding presentation slides using only audio input. Building upon the MaViLS (Matching Videos to Lecture Slides) dataset and methodology, I enhance the dynamic programming pipeline with state-of-the-art multimodal retrieval embeddings and introduce several novel features to improve matching accuracy.

The system consists of two main components:

- **Research Pipeline** (`experiment/`): Evaluation framework for benchmarking algorithms on the MaViLS dataset with 20 annotated lectures
- **Inference Pipeline** (`inference/`): End-to-end practical application system for processing arbitrary lecture audio and slides (ASR → Matching → TTS)

## Novel Contributions

This project extends the MaViLS baseline approach with several key innovations:

1. **State-of-the-art Multimodal Embeddings**: Integration of NVIDIA NeMo Retriever ColEmbed (1B/3B parameters) for superior text-image representation learning
2. **Confidence Boosting Mechanism**: Novel scoring adjustment based on the margin between top-k matches to improve decision quality
3. **Context-Aware Matching**: Exponential moving average (EMA) of recent matches for temporal consistency
4. **Comprehensive Hyperparameter Optimization**: Systematic grid search across 8 key parameters to identify optimal configuration


## Key Features

### Enhanced Dynamic Programming Algorithm

Our approach extends traditional DP-based slide matching with:

- **Multimodal Embeddings**: NVIDIA NeMo Retriever ColEmbed models (1B/3B parameters) for joint text-image representation
- **Adaptive Penalties**: Jump and backward transition costs to enforce temporal coherence
- **Confidence Boosting**: Dynamic score adjustment based on margin between top matches
- **Context Awareness**: Exponential moving average of historical matches for local consistency
- **Exponential Scaling**: Amplification of similarity score differences for sharper decision boundaries

### Practical Application Pipeline

- Automatic speech recognition using NVIDIA Parakeet TDT (0.6B parameters)
- Sentence-level timestamp extraction and segmentation
- Text-to-speech synthesis with Kokoro-82M for regenerating aligned audio
- Comprehensive output including matched slides, confidence scores, and timing information

## Project Structure

```
.
├── experiment/              # Research and evaluation
│   ├── evaluate.py          # Benchmark evaluation on MaViLS dataset
│   ├── grid_search.py       # Hyperparameter optimization
│   ├── ablation_study.py    # Feature contribution analysis
│   ├── match.py             # Core matching algorithm
│   ├── dataset/             # MaViLS benchmark data
│   │   ├── lectures/        # Audio files and PDF slides
│   │   └── ground_truth_files/  # Manually annotated alignments
│   └── results/             # Pre-computed evaluation results
│       ├── run_evaluate/    # 3B model full dataset results (81.66%)
│       ├── run_evaluate_1b/ # 1B model full dataset results (78.49%)
│       └── run_ablation/    # Feature ablation study results
│
├── inference/               # Practical application pipeline
│   ├── run.py              # Command-line interface
│   ├── pipeline.py         # Integrated ASR → Matching → TTS pipeline
│   ├── match.py            # Core matching algorithm
│   ├── asr_module.py       # Audio transcription module
│   └── tts_module.py       # Text-to-speech synthesis module
│
└── setup.sh                # Environment setup script
```

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support (≥8GB VRAM for inference, ≥12GB recommended for experiments)
- Conda or Miniconda

**Tested Environments:**
- Development/Inference: NVIDIA RTX 3070, Ubuntu 24.04, CUDA 12.6 (1B model)
- Experiments/Benchmarks: NVIDIA L40S (3B model)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd NLP-Project
   ```

2. **Run setup script**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   Or specify a custom environment name:
   ```bash
   ./setup.sh -n nlp-env
   ```

3. **Post-installation model configuration**

   After installation, you must manually patch the NeMo Retriever configuration files because of dependency issues:

   ```bash
   conda activate nlp-project

   # Download models to populate cache
   python -c 'from transformers import AutoModel; AutoModel.from_pretrained("nvidia/llama-nemoretriever-colembed-1b-v1")'
   python -c 'from transformers import AutoModel; AutoModel.from_pretrained("nvidia/llama-nemoretriever-colembed-3b-v1")'
   ```

   Then edit the configuration files at:
   - `~/.cache/huggingface/modules/transformers_modules/nvidia/llama_hyphen_nemoretriever_hyphen_colembed_hyphen_1b_hyphen_v1/*/configuration_eagle_chat.py`
   - `~/.cache/huggingface/modules/transformers_modules/nvidia/llama_hyphen_nemoretriever_hyphen_colembed_hyphen_3b_hyphen_v1/*/configuration_eagle_chat.py`

   In the `to_dict()` method (around lines 78-87), wrap `vision_config` and `llm_config` with `hasattr()` checks to prevent serialization errors.

## Usage

### Research Evaluation

Evaluate the matching algorithm on the MaViLS benchmark dataset:

```bash
# Quick evaluation (first 4 lectures, 3B model)
python experiment/evaluate.py

# Full evaluation (all 20 lectures)
python experiment/evaluate.py --full

# Use 1B model for faster evaluation
python experiment/evaluate.py --model-size 1b

# Custom hyperparameters
python experiment/evaluate.py --jump-penalty 2.0 --backward-weight 2.0 --confidence-threshold 0.95
```

**Hyperparameter Optimization:**
```bash
# Grid search over parameter space
python experiment/grid_search.py

# Ablation study to measure feature contributions
python experiment/ablation_study.py
```

### Inference Pipeline

Process custom lecture audio and slides:

```bash
# Basic usage
python inference/run.py --audio lecture.wav --pdf slides.pdf

# Advanced configuration
python inference/run.py \
  --audio lecture.wav \
  --pdf slides.pdf \
  --model-size 3b \
  --output-dir results/lecture1 \
  --tts-voice af_bella \
  --tts-speed 1.1
```

**Output Files:**
- `tts_output.wav`: Regenerated speech aligned with slides
- `matching_results.json`: Structured matching results with timestamps
- `report.txt`: Human-readable summary

**Available Options:**
- `--model-size {1b,3b}`: Embedding model size (default: 1b for inference)
- `--jump-penalty FLOAT`: Penalty for non-sequential slide transitions (default: 1.5)
- `--backward-weight FLOAT`: Multiplier for backward jumps (default: 1.85)
- `--confidence-threshold FLOAT`: Threshold for confidence boosting (default: 0.913)
- `--no-exponential-scaling`: Disable exponential similarity scaling
- `--tts-voice {af_bella,af_sarah,...}`: TTS voice selection
- `--tts-speed FLOAT`: Speech rate multiplier (default: 1.0)

## Algorithm Details

### Multimodal Slide Matching

The core matching algorithm employs a dynamic programming approach optimized for temporal slide sequences:

1. **Embedding Extraction**
   - Convert PDF slides to images (150 DPI)
   - Encode transcript sentences and slide images using NeMo Retriever ColEmbed
   - Compute normalized similarity matrix between all text-image pairs

2. **Similarity Preprocessing**

   Apply enhancement features before dynamic programming:

   - **Exponential Scaling**: Transform normalized similarities to amplify differences
     ```
     S[i,j] ← exp(α · (S[i,j] - 1))
     ```
     where α = 2.785 is the exponential scale factor

   - **Confidence Boosting**: Multiply scores when second-best match is weak
     ```
     S[i,j] ← S[i,j] · w_c   if S[i,second_best] < τ
     ```
     where w_c = 2.18 (confidence weight) and τ = 0.913 (confidence threshold)

3. **Dynamic Programming with Context Awareness**

   For each sentence i and slide j, compute the optimal cumulative score:

   ```
   dp[i,j] = max_k { dp[i-1,k] + score[i,j] - penalty(k,j) }

   where:
     score[i,j] = S[i,j] + w_ctx · C[j]
     penalty(k,j) = { (j-k-1) · γ           if k < j  (forward)
                    { (k-j) · γ · w_b       if k ≥ j  (backward)
   ```

   Key components:
   - `S[i,j]`: Preprocessed similarity score (with exponential scaling and confidence boost)
   - `C[j]`: Context score (exponential moving average) for slide j
   - `w_ctx = 0.04`: Weight for context similarity contribution
   - `γ = 1.5`: Base jump penalty
   - `w_b = 1.85`: Backward jump weight multiplier

   Context update after each sentence:
   ```
   C ← C + β · (S[i] - C)
   ```
   where β = 0.24 is the EMA update rate

4. **Backtracking**
   - Viterbi-style backtracking to recover optimal alignment path
   - Per-sentence confidence scores extracted from original normalized similarity matrix

### Optimized Hyperparameters

Default values obtained through grid search on MaViLS dataset:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `jump_penalty` | 1.5 | Base penalty for skipping slides |
| `backward_weight` | 1.85 | Multiplier for backward transitions |
| `exponential_scale` | 2.785 | Exponential scaling factor for similarities |
| `confidence_threshold` | 0.913 | Minimum margin for confidence boost |
| `confidence_weight` | 2.18 | Weight applied when confidence boost triggers |
| `context_weight` | 0.04 | Weight for EMA context similarity |
| `context_update_rate` | 0.24 | EMA decay rate for context updates |
| `min_sentence_length` | 2 | Minimum words to use similarity scores |

## Results

### Benchmark Performance

Pre-computed evaluation results on the full MaViLS dataset (20 lectures, 12,830 sentences) are available in [experiment/results/](experiment/results/).

**Full Dataset Evaluation - Model Comparison**:

| Model | Overall Accuracy | Average MAE | Average RMSE | Total Evaluated | Results Directory |
|-------|------------------|-------------|--------------|-----------------|-------------------|
| **3B** (nvidia/llama-nemoretriever-colembed-3b-v1) | **81.66%** | **0.262** | **0.768** | 8,746 | [run_evaluate/](experiment/results/run_evaluate/) |
| **1B** (nvidia/llama-nemoretriever-colembed-1b-v1) | **78.49%** | **0.326** | **0.851** | 8,746 | [run_evaluate_1b/](experiment/results/run_evaluate_1b/) |

**Key Observations**:
- The 3B model achieves **3.17%p higher accuracy** than the 1B model (81.66% vs 78.49%)
- The 3B model shows **better precision** with lower MAE (0.262 vs 0.326) and RMSE (0.768 vs 0.851)
- Both models evaluated on the same 8,746 sentences across all 20 lectures
- The accuracy gap is consistent with the ~3x parameter difference (3B vs 1B)

**Per-Lecture Performance Comparison (Top 5 Lectures by 3B Accuracy)**:

| Lecture | 3B Accuracy | 1B Accuracy | Δ (3B - 1B) |
|---------|-------------|-------------|-------------|
| creating_breakthrough_products | 93.1% | 89.1% | +4.0%p |
| deeplearning | 92.6% | 91.8% | +0.8%p |
| computer_vision_2_2 | 92.5% | 92.5% | 0.0%p |
| climate_science_policy | 91.2% | 92.7% | -1.5%p |
| team_dynamics_game_design | 90.9% | 90.9% | 0.0%p |

Detailed per-lecture results for both models available in respective results directories.

**Ablation Study (3B Model)**:

Feature contribution analysis showing impact of each algorithm component (± values indicate standard deviation across 20 lectures):

| Configuration | Accuracy | Δ from Baseline |
|--------------|----------|-----------------|
| All Features (baseline) | 81.66% ± 9.85% | - |
| No Sentence Length Filter | 81.62% ± 9.84% | -0.04%p |
| No Context Similarity | 81.18% ± 10.21% | -0.48%p |
| No Exponential Scaling | 79.02% ± 12.56% | -2.64%p |
| No Confidence Boost | 78.40% ± 13.64% | -3.26%p |
| No Features | 78.37% ± 13.60% | -3.29%p |

**Key Findings**:
- Confidence boosting and exponential scaling are the most impactful features
- Context similarity provides modest but consistent improvement (~0.5%p)
- Sentence length filtering has minimal impact on accuracy
- All features combined achieve best performance with lowest variance

Complete ablation results available in [experiment/results/run_ablation/](experiment/results/run_ablation/).

### Reproducibility

**For reproducing experimental results**:
- Use the **3B model** (`--model-size 3b`) for maximum accuracy in research evaluation
- The reported 81.66% accuracy was obtained with the 3B model on all 20 lectures
- All experiment scripts default to 3B model: `experiment/evaluate.py`, `experiment/grid_search.py`, `experiment/ablation_study.py`

**For practical inference applications**:
- Use the **1B model** (`--model-size 1b`) for inference with lower VRAM requirements
- The 1B model achieves 78.49% accuracy on the benchmark (3.17%p lower than 3B)
- The 1B model requires ≥8GB VRAM (vs ≥12GB for 3B)
- The inference pipeline defaults to 1B model: `inference/run.py`

**Model Comparison**:

| Model | VRAM Requirement | Benchmark Accuracy | Default Usage |
|-------|------------------|-------------------|---------------|
| 3B | ≥12GB | 81.66% | Research evaluation |
| 1B | ≥8GB | 78.49% | Practical inference |

**Performance vs Resource Trade-off**:
- The 3B model provides **3.17%p absolute accuracy improvement** but requires more VRAM
- For applications where accuracy is critical, use the 3B model
- For deployments with limited VRAM (8-12GB), the 1B model offers a strong balance of performance and efficiency

## Dataset

The research component uses the **MaViLS (Matching Videos to Lecture Slides)** benchmark dataset:

- **Size**: 20 full academic lectures across diverse domains
- **Duration**: Over 22 hours of video content
- **Annotations**: 12,830 manually labeled sentence-to-slide alignments
- **Domains**: Cryptocurrency, psychology, reinforcement learning, robotics, climate science, computer vision, deep learning, physics, and more
- **Source**: [MaViLS GitHub Repository](https://github.com/andererka/MaViLS)
- **License**: Apache-2.0

Ground truth files are provided in Excel format with columns:
- `Value`: Transcript sentence text
- `Slidenumber`: Ground truth page number (1-indexed, -1 for unevaluated segments)

## Model Architecture

### Components

1. **ASR Model**: NVIDIA Parakeet TDT 0.6B v2
   - FastConformer XL architecture with TDT decoder
   - Trained on 120,000 hours of English speech (Granary dataset)
   - Sentence-level segmentation with precise timestamps
   - Chunked processing for long-form audio (300s chunks)
   - License: CC-BY-4.0 (commercial use permitted)

2. **Embedding Model**: NVIDIA NeMo Retriever ColEmbed
   - **Architecture**: SigLIP-2 (vision) + Llama 3.2 (language) with ColBERT-style late interaction
   - **1B variant**: ~4GB VRAM, faster inference (based on Llama-3.2-1B)
   - **3B variant**: ~8GB VRAM, 3.17%p higher accuracy (based on Llama-3.2-3B)
   - Multimodal architecture supporting both text and image inputs
   - Shared embedding space for cross-modal retrieval
   - License: Non-commercial/research use only

3. **TTS Model**: Kokoro-82M
   - Lightweight 82M-parameter synthesis model
   - Multiple voice options with configurable speech rate
   - Timestamp tracking for alignment verification
   - License: Apache 2.0 (commercial use permitted)

### Memory Management

- Staged model loading/unloading to minimize GPU memory footprint
- Automatic cache clearing between pipeline stages
- Optional similarity matrix caching for batch evaluations
- GPU memory typically peaks at 8-10GB with 3B model during matching phase

## Citation

This project builds upon the MaViLS dataset and methodology.

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

## Acknowledgments

- NVIDIA for NeMo Retriever and Parakeet models
- MaViLS dataset contributors
- Kokoro TTS model developers
