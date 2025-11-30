# Audio-to-Slide Matching via Multimodal Embeddings and Dynamic Programming

A research project for automatic alignment of lecture audio transcripts to presentation slides using multimodal retrieval models and dynamic programming.

## Overview

This project addresses the challenge of matching spoken lecture content to corresponding presentation slides using only audio input. Building upon the MaViLS (Matching Videos to Lecture Slides) dataset and methodology, we enhance the dynamic programming pipeline with state-of-the-art multimodal retrieval embeddings and introduce several novel features to improve matching accuracy.

The system consists of two main components:

- **Research Pipeline** (`experiment/`): Evaluation framework for benchmarking algorithms on the MaViLS dataset with 20 annotated lectures
- **Inference Pipeline** (`inference/`): End-to-end production system for processing arbitrary lecture audio and slides (ASR → Matching → TTS)

## Key Features

### Enhanced Dynamic Programming Algorithm

Our approach extends traditional DP-based slide matching with:

- **Multimodal Embeddings**: NVIDIA NeMo Retriever ColEmbed models (1B/3B parameters) for joint text-image representation
- **Adaptive Penalties**: Jump and backward transition costs to enforce temporal coherence
- **Confidence Boosting**: Dynamic score adjustment based on margin between top matches
- **Context Awareness**: Exponential moving average of historical matches for local consistency
- **Exponential Scaling**: Amplification of similarity score differences for sharper decision boundaries

### Production-Ready Pipeline

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
│   ├── match.py             # Core matching algorithm (research version)
│   └── dataset/             # MaViLS benchmark data
│       ├── lectures/        # Audio files and PDF slides
│       └── ground_truth_files/  # Manually annotated alignments
│
├── inference/               # Production pipeline
│   ├── run.py              # Command-line interface
│   ├── pipeline.py         # Integrated ASR → Matching → TTS pipeline
│   ├── match.py            # Slide matching processor (production version)
│   ├── asr_module.py       # Audio transcription module
│   └── tts_module.py       # Text-to-speech synthesis module
│
└── setup.sh                # Environment setup script
```

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support (≥8GB VRAM recommended for 3B model)
- Conda or Miniconda

**Tested Local Environment:**
- GPU: NVIDIA RTX 3070
- OS: Ubuntu 24.04
- CUDA: 12.6
- Note: Full benchmark evaluations were conducted with the 3B model on a higher-spec GPU environment

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

   After installation, you must manually patch the NeMo Retriever configuration files:

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
   - Compute cosine similarity matrix between all text-image pairs

2. **Dynamic Programming with Enhanced Features**
   ```
   score[i][j] = similarity[i][j] * scale_factor
                 + context_similarity[i][j]
                 + confidence_boost[i][j]
                 - transition_penalty[i][j]
   ```

   Where:
   - `similarity[i][j]`: Base cosine similarity between sentence i and slide j
   - `scale_factor`: Exponential scaling to amplify differences
   - `context_similarity`: EMA of recent matches for local coherence
   - `confidence_boost`: Bonus when second-best match is significantly weaker
   - `transition_penalty`: Jump penalties with special handling for backward transitions

3. **Backtracking**
   - Viterbi-style backtracking to recover optimal alignment path
   - Per-sentence confidence scores based on similarity margins

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

1. **ASR Model**: NVIDIA Parakeet TDT (0.6B parameters)
   - Transducer-based architecture for streaming speech recognition
   - Sentence-level segmentation with precise timestamps
   - Chunked processing for long-form audio (300s chunks)

2. **Embedding Model**: NVIDIA NeMo Retriever ColEmbed
   - **1B variant**: ~4GB VRAM, faster inference
   - **3B variant**: ~8GB VRAM, 2-3% higher accuracy on benchmark
   - Multimodal architecture supporting both text and image inputs
   - Shared embedding space for cross-modal retrieval

3. **TTS Model**: Kokoro-82M
   - Lightweight synthesis model for audio regeneration
   - Multiple voice options with configurable speech rate
   - Timestamp tracking for alignment verification

### Memory Management

- Staged model loading/unloading to minimize GPU memory footprint
- Automatic cache clearing between pipeline stages
- Optional similarity matrix caching for batch evaluations
- GPU memory typically peaks at 8-10GB with 3B model during matching phase

## Performance Considerations

- **Model Selection**:
  - Use 1B model for rapid prototyping and resource-constrained environments
  - Use 3B model for maximum accuracy in research evaluations

- **Batch Processing**:
  - Similarity matrices computed in vectorized batches on GPU
  - PDF rendering parallelized across slides

- **Caching**:
  - Embeddings cached within single evaluation runs
  - Disable with `use_cached_similarity=False` for memory-constrained scenarios

## Development Notes

### Code Organization

The `match.py` module exists in two locations:
- `experiment/match.py`: Research version with caching optimizations for batch evaluation
- `inference/match.py`: Production version optimized for single-run inference

When modifying the core algorithm, ensure changes are synchronized across both versions.

### Extension Points

- **Custom Features**: Add new scoring components in the DP score calculation
- **Alternative Models**: Swap embedding models by modifying model loading in `match.py`
- **Post-processing**: Add filtering or smoothing in the backtracking phase
- **Evaluation Metrics**: Extend `evaluate.py` with additional performance measures

## Citation

This project builds upon the MaViLS dataset and methodology. If you use this work, please cite:

```bibtex
@inproceedings{anderer2023mavils,
  title={MaViLS: A Benchmark for Matching Videos to Lecture Slides},
  author={Anderer, Katharina and others},
  booktitle={Proceedings of the Conference},
  year={2023}
}
```

## Acknowledgments

- NVIDIA for NeMo Retriever and Parakeet models
- MaViLS dataset contributors
- Kokoro TTS model developers
