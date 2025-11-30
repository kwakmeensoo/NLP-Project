"""
TTS Module using Kokoro-82M Model

This module provides text-to-speech synthesis with timestamp tracking
for generated audio segments.
"""

from typing import List, Dict, Tuple
import numpy as np
import torch
import gc


class TTSProcessor:
    """
    Text-to-Speech processor using Kokoro-82M model.
    Generates speech audio with segment-level timestamps.
    """

    def __init__(
        self,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang_code: str = "a"
    ):
        """
        Initialize TTS processor.

        Args:
            voice: Voice preset to use (af_heart, af_bella, af_sarah, am_adam, am_michael)
            speed: Speech speed multiplier (1.0 = normal)
            lang_code: Language code ('a' = American English)
        """
        self.voice = voice
        self.speed = speed
        self.lang_code = lang_code
        self.pipeline = None
        self.sample_rate = 24000  # Kokoro default sampling rate

        print(f"Initialized TTS Processor")
        print(f"  Voice: {voice}")
        print(f"  Speed: {speed}")
        print(f"  Lang: {lang_code}")

    def load_model(self):
        """Load the Kokoro TTS pipeline into memory."""
        if self.pipeline is not None:
            print("TTS pipeline already loaded")
            return

        print("Loading Kokoro TTS pipeline...")

        try:
            from kokoro import KPipeline
            self.pipeline = KPipeline(lang_code=self.lang_code)
            print("TTS pipeline loaded successfully!")

        except ImportError:
            raise ImportError(
                "Kokoro package not installed. Please install with: pip install kokoro-onnx"
            )

    def unload_model(self):
        """Unload the pipeline to free memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            gc.collect()
            print("TTS pipeline unloaded")

    def synthesize_segments(
        self,
        segments: List[Dict]
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Synthesize speech for multiple text segments.

        Args:
            segments: List of segments with 'text' field

        Returns:
            Tuple of:
                - audio_array: Combined audio as numpy array
                - tts_segments: List of segments with TTS timestamps
        """
        if self.pipeline is None:
            self.load_model()

        print(f"Synthesizing {len(segments)} segments...")

        all_audio = []
        tts_segments = []
        current_time = 0.0

        for i, segment in enumerate(segments):
            text = segment.get("text", "")
            if not text.strip():
                continue

            # Generate audio for this segment using Kokoro
            generator = self.pipeline(text, voice=self.voice, speed=self.speed)

            # Collect audio segments from generator
            audio_parts = []
            for _, _, audio in generator:
                audio_parts.append(audio)

            # Merge audio parts
            if audio_parts:
                audio_segment = np.concatenate(audio_parts)
            else:
                # Empty audio if nothing generated
                audio_segment = np.array([], dtype=np.float32)

            # Calculate duration
            duration = len(audio_segment) / self.sample_rate
            end_time = current_time + duration

            # Store segment info
            tts_segments.append({
                "text": text,
                "start_time": round(current_time, 2),
                "end_time": round(end_time, 2),
                "original_start_time": segment.get("start_time"),
                "original_end_time": segment.get("end_time")
            })

            all_audio.append(audio_segment)
            current_time = end_time

            if (i + 1) % 10 == 0:
                print(f"  Synthesized {i + 1}/{len(segments)} segments")

        # Concatenate all audio segments
        combined_audio = np.concatenate(all_audio) if all_audio else np.array([], dtype=np.float32)

        print(f"TTS synthesis completed: {len(combined_audio) / self.sample_rate:.2f}s total")

        return combined_audio, tts_segments

    def save_audio(
        self,
        audio_array: np.ndarray,
        output_path: str
    ):
        """
        Save audio array to WAV file.

        Args:
            audio_array: Audio data as numpy array
            output_path: Output file path (e.g., 'output.wav')
        """
        import soundfile as sf

        # Ensure audio is in correct format
        audio = audio_array.astype(np.float32)

        # Normalize to [-1, 1] range if needed
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        # Save as WAV file
        sf.write(output_path, audio, self.sample_rate)

        print(f"Audio saved to: {output_path}")
        print(f"  Duration: {len(audio) / self.sample_rate:.2f}s")
        print(f"  Sampling rate: {self.sample_rate} Hz")
