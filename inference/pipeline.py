"""
Audio-to-Slide Matching Pipeline with TTS Generation

This module integrates ASR, slide matching, and TTS generation into a unified pipeline.
"""

import os
from typing import Dict, List, Optional
import numpy as np
import json

from match import SlideMatchingProcessor
from asr_module import ASRProcessor
from tts_module import TTSProcessor


class AudioSlideMatchingPipeline:
    """
    End-to-end pipeline for audio-to-slide matching with TTS generation.

    Pipeline steps:
    1. ASR: Transcribe audio and extract sentence segments with timestamps
    2. Matching: Match sentences to PDF slides using multimodal embeddings
    3. TTS: Generate synthesized speech for matched segments
    4. Results: Return matching information, timestamps, and TTS audio
    """

    def __init__(
        self,
        asr_model: str = "nvidia/parakeet-tdt-0.6b-v2",
        matching_model: str = "nvidia/llama-nemoretriever-colembed-1b-v1",
        device: str = "cuda",
        tts_voice: str = "af_heart",
        tts_speed: float = 1.0,
        tts_lang_code: str = "a",
        retriever_batch_size: int = 4
    ):
        """
        Initialize the pipeline with all required models.

        Args:
            asr_model: ASR model identifier
            matching_model: Multimodal embedding model for slide matching
            device: Device to run on ('cuda' or 'cpu')
            tts_voice: Voice preset for TTS (af_heart, af_bella, af_sarah, am_adam, am_michael)
            tts_speed: Speech speed multiplier for TTS (1.0 = normal)
            tts_lang_code: Language code for TTS ('a' = American English)
            batch_size: Batch size for embedding computation
        """
        self.device = device

        # Initialize processors
        print("=" * 60)
        print("Initializing Audio-Slide Matching Pipeline")
        print("=" * 60)

        self.asr = ASRProcessor(
            model_name=asr_model,
            device=device
        )

        self.matcher = SlideMatchingProcessor(
            model_name=matching_model,
            device=device,
            batch_size=retriever_batch_size
        )

        self.tts = TTSProcessor(
            voice=tts_voice,
            speed=tts_speed,
            lang_code=tts_lang_code
        )

        print("=" * 60)
        print("Pipeline initialized successfully!")
        print("=" * 60)

    def process(
        self,
        audio_path: str,
        pdf_path: str,
        output_dir: Optional[str] = None,
        matching_params: Optional[Dict] = None
    ) -> Dict:
        """
        Run the complete pipeline on audio and PDF files.

        Args:
            audio_path: Path to input audio file
            pdf_path: Path to input PDF file
            output_dir: Directory to save outputs (if None, only returns results)
            matching_params: Optional parameters for slide matching

        Returns:
            Dictionary containing:
                - asr_result: Full ASR transcription result
                - matching_result: Slide matching results with page numbers
                - tts_audio: Generated TTS audio array
                - tts_segments: TTS segments with timestamps
                - output_files: Paths to saved output files (if output_dir provided)
        """
        print("\n" + "=" * 60)
        print("Starting Pipeline Processing")
        print("=" * 60)
        print(f"Audio: {audio_path}")
        print(f"PDF: {pdf_path}")
        print()

        # Step 1: ASR - Transcribe audio
        print("\n[Step 1/4] Running ASR...")
        print("-" * 60)
        asr_result = self.asr.transcribe(audio_path, return_timestamps=True)

        print(f"\nASR Results:")
        print(f"  Full transcript length: {len(asr_result['full_transcript'])} chars")
        print(f"  Number of segments: {len(asr_result['segments'])}")
        print(f"\nFirst few segments:")
        for i, seg in enumerate(asr_result['segments'][:3]):
            print(f"  [{seg['start_time']:.2f}s - {seg['end_time']:.2f}s] {seg['text'][:60]}...")

        # Unload ASR model to free VRAM
        print("\nUnloading ASR model...")
        self.asr.unload_model()

        # Step 2: Slide Matching
        print("\n[Step 2/4] Matching sentences to slides...")
        print("-" * 60)

        # Prepare sentences from ASR segments
        sentences = [seg['text'] for seg in asr_result['segments']]

        # Set matching parameters
        match_params = {
            'jump_penalty': 1.5,
            'backward_weight': 1.85,
            'use_exponential_scaling': True,
            'exponential_scale': 2.785,
            'use_confidence_boost': True,
            'confidence_threshold': 0.913,
            'confidence_weight': 2.18,
            'use_context_similarity': True,
            'context_weight': 0.04,
            'context_update_rate': 0.24,
            'min_sentence_length': 2
        }

        if matching_params:
            match_params.update(matching_params)

        # Run matching
        matching_result = self.matcher.match_transcript_to_slides(
            transcript="",  # Not used when sentences provided
            pdf_path=pdf_path,
            sentences=sentences,
            **match_params
        )

        print(f"\nMatching Results:")
        print(f"  Total segments matched: {len(matching_result)}")
        print(f"  Unique slides used: {len(set(r['matched_page'] for r in matching_result))}")
        print(f"\nSample matches:")
        for i, match in enumerate(matching_result[:3]):
            print(f"  Slide {match['matched_page']}: {match['text'][:50]}... "
                  f"(confidence: {match['confidence_score']:.3f})")

        # Merge matching results with ASR timestamps
        for i, (asr_seg, match) in enumerate(zip(asr_result['segments'], matching_result)):
            match['start_time'] = asr_seg['start_time']
            match['end_time'] = asr_seg['end_time']

        # Unload matcher model to free VRAM
        print("\nUnloading matcher model...")
        self.matcher.unload_model()
        self.matcher.clear_cache()

        # Step 3: TTS Generation
        print("\n[Step 3/4] Generating TTS audio...")
        print("-" * 60)

        tts_audio, tts_segments = self.tts.synthesize_segments(
            matching_result
        )

        print(f"\nTTS Results:")
        print(f"  Generated audio duration: {len(tts_audio) / self.tts.sample_rate:.2f}s")
        print(f"  Number of TTS segments: {len(tts_segments)}")

        # Unload TTS model to free VRAM
        print("\nUnloading TTS model...")
        self.tts.unload_model()

        # Step 4: Save outputs (if output_dir provided)
        output_files = {}
        if output_dir:
            print("\n[Step 4/4] Saving outputs...")
            print("-" * 60)

            os.makedirs(output_dir, exist_ok=True)

            # Save TTS audio
            tts_path = os.path.join(output_dir, "tts_output.wav")
            self.tts.save_audio(tts_audio, tts_path)
            output_files['tts_audio'] = tts_path

            # Save JSON results
            results_path = os.path.join(output_dir, "matching_results.json")
            self._save_results_json(
                asr_result,
                matching_result,
                tts_segments,
                results_path
            )
            output_files['results_json'] = results_path

            # Save readable text report
            report_path = os.path.join(output_dir, "report.txt")
            self._save_text_report(
                asr_result,
                matching_result,
                tts_segments,
                report_path
            )
            output_files['report_text'] = report_path

            print(f"\nOutputs saved to: {output_dir}")
            for key, path in output_files.items():
                print(f"  {key}: {path}")

        print("\n" + "=" * 60)
        print("Pipeline Processing Complete!")
        print("=" * 60)

        return {
            'asr_result': asr_result,
            'matching_result': matching_result,
            'tts_audio': tts_audio,
            'tts_segments': tts_segments,
            'output_files': output_files
        }

    def _save_results_json(
        self,
        asr_result: Dict,
        matching_result: List[Dict],
        tts_segments: List[Dict],
        output_path: str
    ):
        """Save results as JSON file with merged segment information."""
        # Merge all information into unified segments
        segments = []
        for i, match in enumerate(matching_result):
            segment = {
                'text': match['text'],
                'matched_page': match['matched_page'],
                'confidence_score': match['confidence_score'],
                'asr_start_time': match['start_time'],
                'asr_end_time': match['end_time']
            }

            # Add TTS timing if available
            if i < len(tts_segments):
                segment['tts_start_time'] = tts_segments[i]['start_time']
                segment['tts_end_time'] = tts_segments[i]['end_time']

            segments.append(segment)

        data = {
            'full_transcript': asr_result['full_transcript'],
            'language': asr_result.get('language'),
            'segments': segments
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  Saved JSON results: {output_path}")

    def _save_text_report(
        self,
        asr_result: Dict,
        matching_result: List[Dict],
        tts_segments: List[Dict],
        output_path: str
    ):
        """Save human-readable text report."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Audio-to-Slide Matching Report\n")
            f.write("=" * 80 + "\n\n")

            # ASR Summary
            f.write("ASR TRANSCRIPTION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Full Transcript:\n{asr_result['full_transcript']}\n\n")
            f.write(f"Number of segments: {len(asr_result['segments'])}\n\n")

            # Matching Results
            f.write("=" * 80 + "\n")
            f.write("SLIDE MATCHING RESULTS\n")
            f.write("=" * 80 + "\n\n")

            for i, match in enumerate(matching_result, 1):
                f.write(f"Segment {i}:\n")
                f.write(f"  Text: {match['text']}\n")
                f.write(f"  Matched Slide: Page {match['matched_page']}\n")
                f.write(f"  Confidence: {match['confidence_score']:.3f}\n")
                f.write(f"  Original Audio Time: [{match['start_time']:.2f}s - {match['end_time']:.2f}s]\n")

                # Find corresponding TTS segment
                tts_seg = tts_segments[i - 1] if i - 1 < len(tts_segments) else None
                if tts_seg:
                    f.write(f"  TTS Time: [{tts_seg['start_time']:.2f}s - {tts_seg['end_time']:.2f}s]\n")

                f.write("\n")

            # Summary statistics
            f.write("=" * 80 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total segments: {len(matching_result)}\n")
            f.write(f"Unique slides: {len(set(r['matched_page'] for r in matching_result))}\n")
            f.write(f"Average confidence: {np.mean([r['confidence_score'] for r in matching_result]):.3f}\n")
            f.write(f"Original audio duration: {asr_result['segments'][-1]['end_time']:.2f}s\n")
            if tts_segments:
                f.write(f"TTS audio duration: {tts_segments[-1]['end_time']:.2f}s\n")

        print(f"  Saved text report: {output_path}")
