"""
ASR Module using NVIDIA Parakeet TDT Model

This module provides audio transcription functionality with sentence segmentation
and timestamp extraction.
"""

from typing import Dict
import torch
import gc
import os
import librosa
import soundfile as sf


class ASRProcessor:
    """
    Automatic Speech Recognition processor using NVIDIA Parakeet TDT.
    Provides transcript with sentence-level segmentation and timestamps.
    """

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v2",
        device: str = "cuda"
    ):
        """
        Initialize ASR processor.

        Args:
            model_name: NeMo model identifier
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device
        self.model = None

        print(f"Initialized ASR Processor")
        print(f"  Model: {model_name}")
        print(f"  Device: {device}")

    def load_model(self):
        """Load the ASR model into memory."""
        if self.model is not None:
            print("ASR model already loaded")
            return

        print("Loading ASR model...")

        try:
            import nemo.collections.asr as nemo_asr
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.model_name
            )
            print("ASR model loaded successfully!")

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"  GPU memory allocated: {allocated:.2f} GB")

        except ImportError:
            raise ImportError(
                "NeMo toolkit not installed. Please install with: pip install nemo_toolkit[asr]"
            )

    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            gc.collect()
            print("ASR model unloaded")

    def _prepare_audio_chunks(
        self,
        audio_path: str,
        chunk_seconds: int = 300
    ) -> tuple[list[str], str, float]:
        """
        Prepare audio chunks for transcription.
        Converts to mono 16kHz format required by Parakeet.

        Args:
            audio_path: Input audio file path
            chunk_seconds: Chunk duration in seconds (default: 300s = 5min)

        Returns:
            Tuple of (chunk_files, temp_dir, total_duration)
        """
        import uuid

        print(f"Loading audio file: {audio_path}")

        # Load audio as mono 16kHz (Parakeet requirement)
        audio_data, sample_rate = librosa.load(audio_path, sr=16000, mono=True)

        # Calculate duration
        total_duration = len(audio_data) / sample_rate
        print(f"Audio loaded - Duration: {total_duration:.1f}s ({total_duration/60:.1f}min), SR: {sample_rate}Hz")

        # Create temp directory for chunks
        temp_dir = f"temp_asr_chunks_{uuid.uuid4().hex[:8]}"
        os.makedirs(temp_dir, exist_ok=True)

        # Split into chunks
        chunk_samples = int(chunk_seconds * sample_rate)
        chunk_files = []
        chunk_num = 0

        if total_duration > chunk_seconds:
            print(f"Splitting into {chunk_seconds}s chunks...")
        else:
            print(f"Processing as single chunk")

        for i in range(0, len(audio_data), chunk_samples):
            chunk = audio_data[i:i + chunk_samples]

            # Skip chunks shorter than 1 second
            if len(chunk) < sample_rate:
                continue

            chunk_num += 1
            chunk_file = os.path.join(temp_dir, f"chunk_{chunk_num:03d}.wav")
            sf.write(chunk_file, chunk, sample_rate)
            chunk_files.append(chunk_file)

            chunk_duration = len(chunk) / sample_rate
            print(f"Chunk {chunk_num}: {chunk_duration:.1f}s")

        print(f"Total {len(chunk_files)} chunk(s) created")

        return chunk_files, temp_dir, total_duration

    def transcribe(
        self,
        audio_path: str,
        return_timestamps: bool = True,
        chunk_seconds: int = 300,
        batch_size: int = 3
    ) -> Dict:
        """
        Transcribe audio file to text with sentence segmentation.
        Automatically chunks long audio files and processes in batches.

        Args:
            audio_path: Path to audio file
            return_timestamps: Whether to return segment-level timestamps
            chunk_seconds: Chunk duration in seconds (default: 300s = 5min)
            batch_size: Number of chunks to process at once (default: 3)

        Returns:
            Dictionary containing:
                - transcript: Complete transcript text
                - segments: List of sentence segments with timestamps and text
        """
        if self.model is None:
            self.load_model()

        print(f"Transcribing audio: {audio_path}")
        print("="*60)

        temp_dir = None
        chunk_files = []

        try:
            # Prepare audio chunks (mono 16kHz)
            chunk_files, temp_dir, total_duration = self._prepare_audio_chunks(
                audio_path,
                chunk_seconds=chunk_seconds
            )

            # Clear GPU memory before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

            # Process chunks in batches
            all_transcripts = []
            all_segments = []
            chunk_offset = 0.0

            num_chunks = len(chunk_files)
            for batch_start in range(0, num_chunks, batch_size):
                batch_end = min(batch_start + batch_size, num_chunks)
                batch_chunks = chunk_files[batch_start:batch_end]

                print(f"\nProcessing batch {batch_start//batch_size + 1} (chunks {batch_start+1}-{batch_end})...")

                # Process batch (NeMo processes sequentially internally)
                for idx, chunk_file in enumerate(batch_chunks):
                    chunk_idx = batch_start + idx + 1
                    print(f"  Processing chunk {chunk_idx}/{num_chunks}...")

                    # Transcribe chunk
                    output = self.model.transcribe([chunk_file], timestamps=return_timestamps)

                    if not output or len(output) == 0:
                        print(f"  Warning: Chunk {chunk_idx} returned empty output, skipping")
                        continue

                    transcript_obj = output[0]
                    transcript = transcript_obj.text
                    all_transcripts.append(transcript)

                    print(f"  Chunk {chunk_idx}/{num_chunks}: {len(transcript)} characters")

                    # Extract segments with timestamps
                    if return_timestamps and hasattr(transcript_obj, 'timestamp') and transcript_obj.timestamp:
                        segment_data = transcript_obj.timestamp.get('segment', [])
                        for seg in segment_data:
                            all_segments.append({
                                "text": seg['segment'].strip(),
                                "start_time": seg['start'] + chunk_offset,
                                "end_time": seg['end'] + chunk_offset
                            })

                    # Update chunk offset for next chunk
                    if chunk_idx < num_chunks:
                        chunk_offset += chunk_seconds
                    else:
                        # For last chunk, calculate remaining duration
                        chunk_offset += (total_duration - (chunk_idx - 1) * chunk_seconds)

                    # Clear GPU memory after each chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()

            # Merge results
            full_transcript = ' '.join(filter(None, all_transcripts))

            result = {
                "full_transcript": full_transcript,
                "segments": all_segments
            }

            print("="*60)
            print(f"Transcription completed: {len(full_transcript)} characters, {len(all_segments)} segments")

            if torch.cuda.is_available():
                max_memory = torch.cuda.max_memory_allocated() / 1024**3
                print(f"Max GPU memory usage: {max_memory:.2f} GB")

            return result

        finally:
            # Clean up chunk files
            for chunk_file in chunk_files:
                try:
                    if os.path.exists(chunk_file):
                        os.remove(chunk_file)
                except Exception as e:
                    print(f"Warning: Failed to delete chunk {chunk_file}: {e}")

            # Clean up temp directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    os.rmdir(temp_dir)
                    print(f"Removed temporary directory: {temp_dir}")
                except Exception as e:
                    print(f"Warning: Failed to delete temp directory {temp_dir}: {e}")
