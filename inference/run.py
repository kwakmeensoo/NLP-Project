"""
Run Audio-to-Slide Matching Pipeline

This script provides a command-line interface to run the audio-to-slide matching pipeline.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

from pipeline import AudioSlideMatchingPipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run audio-to-slide matching pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to input audio file"
    )

    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to input PDF slide file"
    )

    # Model configuration
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["1b", "3b"],
        default="1b",
        help="Matching model size (1b or 3b)"
    )

    parser.add_argument(
        "--asr-model",
        type=str,
        default="nvidia/parakeet-tdt-0.6b-v2",
        help="ASR model identifier"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu)"
    )

    # TTS configuration
    parser.add_argument(
        "--tts-voice",
        type=str,
        default="af_heart",
        choices=["af_heart", "af_bella", "af_sarah", "am_adam", "am_michael"],
        help="TTS voice preset"
    )

    parser.add_argument(
        "--tts-speed",
        type=float,
        default=1.0,
        help="TTS speech speed multiplier"
    )

    parser.add_argument(
        "--tts-lang",
        type=str,
        default="a",
        help="TTS language code"
    )

    # Matching parameters
    parser.add_argument(
        "--jump-penalty",
        type=float,
        default=1.5,
        help="Penalty for jumping between non-sequential slides"
    )

    parser.add_argument(
        "--backward-weight",
        type=float,
        default=1.85,
        help="Weight for backward slide jumps"
    )

    parser.add_argument(
        "--exponential-scale",
        type=float,
        default=2.785,
        help="Exponential scaling factor for jump penalties"
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.913,
        help="Confidence threshold for boosting"
    )

    parser.add_argument(
        "--confidence-weight",
        type=float,
        default=2.18,
        help="Weight for confidence boosting"
    )

    parser.add_argument(
        "--context-weight",
        type=float,
        default=0.04,
        help="Weight for context similarity"
    )

    parser.add_argument(
        "--context-update-rate",
        type=float,
        default=0.24,
        help="Rate at which context is updated"
    )

    parser.add_argument(
        "--min-sentence-length",
        type=int,
        default=2,
        help="Minimum sentence length to process"
    )

    parser.add_argument(
        "--no-exponential-scaling",
        action="store_true",
        help="Disable exponential scaling for jump penalties"
    )

    parser.add_argument(
        "--no-confidence-boost",
        action="store_true",
        help="Disable confidence boosting"
    )

    parser.add_argument(
        "--no-context-similarity",
        action="store_true",
        help="Disable context similarity"
    )

    # Batch size configuration
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for retriever embedding computation"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory (if not specified, uses run_TIMESTAMP format)"
    )

    return parser.parse_args()


def get_matching_model_name(model_size: str) -> str:
    """Get the full model name based on size."""
    model_map = {
        "1b": "nvidia/llama-nemoretriever-colembed-1b-v1",
        "3b": "nvidia/llama-nemoretriever-colembed-3b-v1"
    }
    return model_map[model_size]


def main():
    """Main execution function."""
    args = parse_args()

    # Validate input files
    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    if not os.path.exists(args.pdf):
        raise FileNotFoundError(f"PDF file not found: {args.pdf}")

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"run_{timestamp}"

    # Get matching model name
    matching_model = get_matching_model_name(args.model_size)

    print("\n" + "=" * 80)
    print("Audio-to-Slide Matching Pipeline - Run Configuration")
    print("=" * 80)
    print(f"Input Audio: {args.audio}")
    print(f"Input PDF: {args.pdf}")
    print(f"Output Directory: {output_dir}")
    print(f"\nModel Configuration:")
    print(f"  ASR Model: {args.asr_model}")
    print(f"  Matching Model: {matching_model} ({args.model_size})")
    print(f"  Device: {args.device}")
    print(f"\nTTS Configuration:")
    print(f"  Voice: {args.tts_voice}")
    print(f"  Speed: {args.tts_speed}x")
    print(f"  Language: {args.tts_lang}")
    print(f"\nMatching Parameters:")
    print(f"  Jump Penalty: {args.jump_penalty}")
    print(f"  Backward Weight: {args.backward_weight}")
    print(f"  Exponential Scaling: {not args.no_exponential_scaling} (scale={args.exponential_scale})")
    print(f"  Confidence Boost: {not args.no_confidence_boost} (threshold={args.confidence_threshold}, weight={args.confidence_weight})")
    print(f"  Context Similarity: {not args.no_context_similarity} (weight={args.context_weight}, update_rate={args.context_update_rate})")
    print(f"  Min Sentence Length: {args.min_sentence_length}")
    print(f"  Batch Size: {args.batch_size}")
    print("=" * 80 + "\n")

    # Initialize pipeline
    pipeline = AudioSlideMatchingPipeline(
        asr_model=args.asr_model,
        matching_model=matching_model,
        device=args.device,
        tts_voice=args.tts_voice,
        tts_speed=args.tts_speed,
        tts_lang_code=args.tts_lang,
        retriever_batch_size=args.batch_size
    )

    # Prepare matching parameters
    matching_params = {
        'jump_penalty': args.jump_penalty,
        'backward_weight': args.backward_weight,
        'use_exponential_scaling': not args.no_exponential_scaling,
        'exponential_scale': args.exponential_scale,
        'use_confidence_boost': not args.no_confidence_boost,
        'confidence_threshold': args.confidence_threshold,
        'confidence_weight': args.confidence_weight,
        'use_context_similarity': not args.no_context_similarity,
        'context_weight': args.context_weight,
        'context_update_rate': args.context_update_rate,
        'min_sentence_length': args.min_sentence_length
    }

    # Run pipeline
    results = pipeline.process(
        audio_path=args.audio,
        pdf_path=args.pdf,
        output_dir=output_dir,
        matching_params=matching_params
    )

    # Print summary
    print("\n" + "=" * 80)
    print("Execution Summary")
    print("=" * 80)
    print(f"ASR Segments: {len(results['asr_result']['segments'])}")
    print(f"Matched Segments: {len(results['matching_result'])}")
    print(f"Unique Slides: {len(set(r['matched_page'] for r in results['matching_result']))}")
    print(f"TTS Duration: {len(results['tts_audio']) / 22050:.2f}s")
    print(f"\nOutput Files:")
    for key, path in results['output_files'].items():
        print(f"  {key}: {path}")
    print("=" * 80 + "\n")

    print("✓ Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
