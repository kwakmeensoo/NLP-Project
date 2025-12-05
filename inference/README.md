# Inference Module

Practical application pipeline for audio-to-slide matching with integrated ASR and TTS.

## Overview

This module provides a complete end-to-end pipeline that:
1. Transcribes audio lectures with timestamps (ASR)
2. Matches transcript sentences to PDF slide pages
3. Synthesizes aligned speech output (TTS)

**⚠️ Language Support:** This pipeline currently supports **English audio only**. The ASR model (NVIDIA Parakeet TDT 0.6B v2) is trained exclusively on English speech data.

## Files

- **`run.py`**: Command-line interface and entry point
- **`pipeline.py`**: Integrated pipeline orchestrating ASR → Matching → TTS
- **`asr_module.py`**: Audio transcription with NVIDIA Parakeet TDT
- **`match.py`**: Slide matching processor
- **`tts_module.py`**: Text-to-speech synthesis with Kokoro-82M

## Quick Start

### Basic Usage

```bash
python run.py --audio lecture.wav --pdf slides.pdf
```

**Note:** Input audio must be in English.

This will:
- Transcribe the audio
- Match sentences to slides
- Generate synthesized speech
- Save results to `output/` directory

### Custom Output Directory

```bash
# Specify custom output directory
python run.py --audio lecture.wav --pdf slides.pdf --output-dir results/run1

# Without --output-dir, uses run_TIMESTAMP format (e.g., run_20251130_143025)
python run.py --audio lecture.wav --pdf slides.pdf
```

### Use 3B Model (Better Accuracy)

```bash
python run.py --audio lecture.wav --pdf slides.pdf --model-size 3b
```

Default is 1B for faster processing and VRAM constraint.

## Command Reference

```bash
python run.py [OPTIONS]
```

### Required Arguments

| Flag | Description |
|------|-------------|
| `--audio PATH` | Path to input audio file (WAV, MP3, etc.) |
| `--pdf PATH` | Path to input PDF slide file |

### Model Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--model-size {1b,3b}` | `1b` | Matching model size |
| `--asr-model NAME` | `nvidia/parakeet-tdt-0.6b-v2` | ASR model identifier |
| `--tts-model NAME` | `hexgrad/Kokoro-82M` | TTS model identifier |
| `--device {cuda,cpu}` | `cuda` | Device for computation |

### Matching Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--jump-penalty FLOAT` | `1.5` | Penalty for non-sequential slide transitions |
| `--backward-weight FLOAT` | `1.85` | Multiplier for backward jumps |
| `--exponential-scale FLOAT` | `2.785` | Exponential similarity scaling |
| `--confidence-threshold FLOAT` | `0.913` | Threshold for confidence boost |
| `--confidence-weight FLOAT` | `2.18` | Weight for confidence boost |
| `--context-weight FLOAT` | `0.04` | Weight for context similarity (EMA) |
| `--context-update-rate FLOAT` | `0.24` | EMA update rate |
| `--min-sentence-length INT` | `2` | Min words to use similarity score |
| `--no-exponential-scaling` | False | Disable exponential scaling |
| `--no-confidence-boosting` | False | Disable confidence boost |
| `--no-context-similarity` | False | Disable context similarity |

### ASR Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--asr-batch-size INT` | `4` | Batch size for ASR processing |
| `--asr-chunk-duration FLOAT` | `300.0` | Audio chunk duration in seconds |

### TTS Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--tts-voice NAME` | `af_bella` | Voice ID for TTS |
| `--tts-speed FLOAT` | `1.0` | Speech speed multiplier |
| `--skip-tts` | False | Skip TTS synthesis step |

### Output Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir PATH` | `run_TIMESTAMP` | Directory to save results (default: run_YYYYMMDD_HHMMSS) |

## Pipeline Stages

### 1. ASR (Audio Transcription)

**Module**: `asr_module.py`

Transcribes audio using NVIDIA Parakeet TDT 0.6B v2 model:
- **Language**: English only (trained on 120,000 hours of English speech)
- Processes audio in chunks (default 300s)
- Extracts sentence-level segments with timestamps
- Supports punctuation, capitalization, and timestamp prediction
- Returns full transcript + timestamped segments

**Example Output**:
```json
{
  "full_transcript": "Welcome to today's lecture...",
  "segments": [
    {
      "text": "Welcome to today's lecture",
      "start_time": 0.0,
      "end_time": 2.5
    },
    ...
  ]
}
```

### 2. Slide Matching

**Module**: `match.py`

Matches transcript sentences to PDF slides:
- Extracts PDF pages as images (150 DPI)
- Computes multimodal embeddings (text + images)
- Runs dynamic programming algorithm with:
  - Similarity scores (cosine)
  - Jump penalties
  - Backward penalties
  - Exponential scaling
  - Confidence boosting
  - Context similarity (EMA)

**Example Output**:
```json
[
  {
    "text": "Welcome to today's lecture",
    "matched_page": 1,
    "confidence_score": 0.95
  },
  ...
]
```

### 3. TTS (Speech Synthesis)

**Module**: `tts_module.py`

Synthesizes speech using Kokoro pipeline:
- Generates natural-sounding speech
- Tracks timestamps for each segment
- Configurable voice and speed

**Example Output**:
- Audio array (numpy)
- Segment timing information

## Output Files

Results are saved to the specified output directory (default: `run_YYYYMMDD_HHMMSS/`):

```
run_20251130_143025/
├── tts_output.wav              # Synthesized speech audio
├── matching_results.json       # Structured matching results
└── report.txt                  # Human-readable report
```

### matching_results.json

```json
{
  "full_transcript": "Welcome to today's lecture...",
  "language": "en",
  "segments": [
    {
      "text": "Welcome to today's lecture",
      "matched_page": 1,
      "confidence_score": 0.95,
      "asr_start_time": 0.0,
      "asr_end_time": 2.5,
      "tts_start_time": 0.0,
      "tts_end_time": 2.3
    },
    ...
  ]
}
```

### report.txt

Human-readable summary including:
- Overall statistics
- Per-slide sentence groupings
- Confidence scores
- Timing information

## Examples

### Standard Usage

```bash
# Basic pipeline with default settings
python run.py --audio lecture.wav --pdf slides.pdf

# With custom output directory
python run.py --audio lecture.wav --pdf slides.pdf --output-dir results/lecture1
```

### Advanced Configuration

```bash
# Use 3B model with custom matching parameters
python run.py \
  --audio lecture.wav \
  --pdf slides.pdf \
  --model-size 3b \
  --jump-penalty 2.0 \
  --backward-weight 2.5 \
  --output-dir results/high_accuracy

# Custom TTS settings
python run.py \
  --audio lecture.wav \
  --pdf slides.pdf \
  --tts-voice af_sky \
  --tts-speed 1.2 \
  --output-dir results/fast_speech

# Skip TTS synthesis (matching only)
python run.py \
  --audio lecture.wav \
  --pdf slides.pdf \
  --skip-tts
```

### Disable Specific Features

```bash
# Disable all advanced matching features
python run.py \
  --audio lecture.wav \
  --pdf slides.pdf \
  --no-exponential-scaling \
  --no-confidence-boosting \
  --no-context-similarity
```

## Model Selection

### 1B vs 3B Comparison

| Model | VRAM Requirement | Benchmark Accuracy | Use Case |
|-------|------------------|-------------------|----------|
| 1B | ≥8GB | 78.49% | Resource-constrained deployments |
| 3B | ≥12GB | 81.66% | High-accuracy requirements |

**Performance Notes**:
- The 3B model achieves **3.17%p higher accuracy** than 1B on the MaViLS benchmark
- Both models evaluated on 8,746 sentences across 20 lectures
- Default is 1B for broader GPU compatibility

## Memory Management

The pipeline automatically manages GPU memory:
- Models are loaded on-demand
- Each stage unloads its model after completion
- GPU cache is cleared between stages

## Performance Tips

1. **Faster Processing**:
   - Use 1B model: `--model-size 1b`
   - Skip TTS if not needed: `--skip-tts`
   - Use CPU if GPU unavailable: `--device cpu`

2. **Higher Accuracy**:
   - Use 3B model: `--model-size 3b`
   - Tune matching parameters based on content type
   - Increase confidence threshold for conservative matching

3. **Memory Optimization**:
   - Reduce ASR batch size: `--asr-batch-size 2`
   - Use 1B model
   - Process shorter audio chunks: `--asr-chunk-duration 180`

## Supported Audio Formats

ASR module supports common formats:
- WAV (recommended)
- MP3
- ...

Audio is automatically resampled to 16kHz mono if needed.

## TTS Voice Options

Available voices (Kokoro-82M model):
- `af_heart` - Default female voice (warm, expressive)
- `af_bella` - Female voice (clear, professional)
- `af_sarah` - Female voice (soft, gentle)
- `am_adam` - Male voice (deep, authoritative)
- `am_michael` - Male voice (neutral, friendly)

Specify with: `--tts-voice VOICE_NAME`

Example:
```bash
python run.py --audio lecture.wav --pdf slides.pdf --tts-voice am_adam
```

## Hyperparameters

Default values are optimized via grid search on MaViLS benchmark:

```python
jump_penalty = 1.5              # Penalty for skipping slides
backward_weight = 1.85          # Multiplier for backward jumps
exponential_scale = 2.785       # Exponential similarity scaling
confidence_threshold = 0.913    # Threshold for confidence boost
confidence_weight = 2.18        # Weight for confidence boost
context_weight = 0.04           # Weight for context similarity
context_update_rate = 0.24      # EMA update rate
min_sentence_length = 2         # Min words to use similarity
```

**When to adjust**:
- **jump_penalty**: Increase for lectures with sequential slides, decrease for non-linear presentations
- **backward_weight**: Increase to strongly discourage going back to earlier slides
- **exponential_scale**: Increase to amplify differences between good/bad matches
- **confidence_threshold**: Increase for more conservative confidence boosting

## Troubleshooting

### Out of Memory

```bash
# Reduce ASR batch size
python run.py --audio lecture.wav --pdf slides.pdf --asr-batch-size 2

# Use 1B model
python run.py --audio lecture.wav --pdf slides.pdf --model-size 1b

# Use CPU (slower but no VRAM limit)
python run.py --audio lecture.wav --pdf slides.pdf --device cpu
```

### Audio Processing Issues

- **Non-English audio**: This pipeline only supports English audio. The ASR model cannot transcribe other languages.
- Ensure audio is clear and not heavily compressed
- Try reducing chunk duration: `--asr-chunk-duration 180`
- Check audio format is supported

### PDF Rendering Issues

- Ensure PDF is not password-protected
- Check PDF has actual content (not just scanned images)
- Try re-saving PDF with standard settings

## License and Model Information

This module is part of the NLP audio-to-slide matching project.

### Models and Licenses

**ASR Model:**
- **Model**: NVIDIA Parakeet TDT 0.6B v2
- **Language Support**: English only
- **Training Data**: 120,000 hours of English speech
- **License**: CC-BY-4.0 (Creative Commons Attribution 4.0)
- **Commercial Use**: ✓ Permitted with attribution
- **HuggingFace**: [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)

**Matching Model:**
- **Model**: NVIDIA NeMo Retriever ColEmbed (1B/3B variants)
- **License**: NVIDIA Non-Commercial License (for base model)
  - Uses Apache 2.0 components (SigLIP)
  - Uses LLAMA 3.2 Community License (language model)
- **Commercial Use**: Requires NVIDIA AI Product Agreement or NVIDIA AI Enterprise license
- **HuggingFace**: [nvidia/llama-nemoretriever-colembed-1b-v1](https://huggingface.co/nvidia/llama-nemoretriever-colembed-1b-v1)

**TTS Model:**
- **Model**: Kokoro-82M
- **License**: Apache 2.0
- **Commercial Use**: ✓ Fully permitted
- **Training Data**: Exclusively permissive/non-copyrighted audio
- **HuggingFace**: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)

### Important Notes

- **NVIDIA Parakeet TDT** and **Kokoro-82M** can be used commercially with proper attribution
- **NVIDIA NeMo Retriever** requires additional licensing for commercial deployment
  - Academic/research use is generally permitted
  - Commercial users should obtain NVIDIA AI Enterprise license
- For real-world deployment, verify license compliance for your specific use case

### Additional Resources

- NVIDIA Parakeet TDT: [CC-BY-4.0 License](https://creativecommons.org/licenses/by/4.0/)
- NVIDIA NeMo Retriever: [Developer Portal](https://developer.nvidia.com/nemo-retriever)
- Kokoro-82M: [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)
