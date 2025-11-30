"""
Transcript-to-Slide Matching Module

This module provides functionality to match transcript sentences to PDF slides
using multimodal embeddings and dynamic programming optimization.
"""

import gc
import io
from typing import List, Dict, Optional

import torch
import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel


class SlideMatchingProcessor:
    """
    Matches transcript text to PDF slides using multimodal embeddings
    and torch-optimized dynamic programming.
    """

    def __init__(
        self,
        model_name: str = 'nvidia/llama-nemoretriever-colembed-1b-v1',
        device: str = 'cuda',
        batch_size: int = 4
    ):
        """
        Initialize the slide matching processor.

        Args:
            model_name: Name of the pretrained multimodal embedding model
            device: Device to run on ('cuda' or 'cpu')
            batch_size: Batch size for embedding computation
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None

        # Cache for embeddings and similarity matrix
        self._similarity_matrix_cache = None
        self._query_embeddings_cache = None
        self._image_embeddings_cache = None

        print(f"Initialized Slide Matching Processor")
        print(f"  Model: {model_name}")
        print(f"  Device: {device}")
        print(f"  Batch size: {batch_size}")

    def load_model(self):
        """Load the multimodal embedding model into memory."""
        if self.model is not None:
            print("Model already loaded")
            return

        print('Loading embedding model...')

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        self.model = AutoModel.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        ).eval()

        print("Model loaded successfully!")
        if torch.cuda.is_available():
            print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"  GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    def unload_model(self):
        """Unload the model to free GPU/CPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            print("Model unloaded")

    def clear_cache(self):
        """Clear cached embeddings and similarity matrices."""
        self._similarity_matrix_cache = None
        self._query_embeddings_cache = None
        self._image_embeddings_cache = None
        gc.collect()
        print("Cache cleared")

    def extract_slide_images(
        self,
        pdf_path: str,
        dpi: int = 150
    ) -> List[Image.Image]:
        """
        Extract all pages from a PDF file as images.

        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for rendering PDF pages

        Returns:
            List of PIL Image objects, one per page
        """
        print(f'Extracting slides from: {pdf_path}')

        pdf_document = fitz.open(pdf_path)
        slide_images = []

        for page_num in tqdm(range(pdf_document.page_count), desc='Extracting slides'):
            page = pdf_document[page_num]

            # Render page at specified DPI
            scale_factor = dpi / 72
            transform_matrix = fitz.Matrix(scale_factor, scale_factor)
            pixmap = page.get_pixmap(matrix=transform_matrix)

            # Convert to PIL Image
            image_bytes = pixmap.tobytes("png")
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            slide_images.append(image)

        pdf_document.close()
        print(f'Extracted {len(slide_images)} slides')
        return slide_images

    def compute_similarity_matrix(
        self,
        text_queries: List[str],
        slide_images: List[Image.Image],
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Compute normalized similarity matrix between text queries and slide images.

        Args:
            text_queries: List of text sentences/queries
            slide_images: List of slide images
            use_cache: Whether to cache the similarity matrix for reuse

        Returns:
            Normalized similarity matrix as torch.Tensor (num_queries x num_slides)
        """
        if use_cache and self._similarity_matrix_cache is not None:
            print("Using cached similarity matrix")
            return self._similarity_matrix_cache

        if self.model is None:
            self.load_model()

        print('Computing embeddings...')

        # Compute text embeddings
        print('  Encoding text queries...')
        with torch.no_grad():
            text_embeddings = self.model.forward_queries(
                text_queries,
                batch_size=self.batch_size
            )

        # Compute image embeddings (process one by one to avoid shared memory issues)
        print('  Encoding slide images...')
        image_embeddings = []
        for image in tqdm(slide_images, desc='Processing slides'):
            with torch.no_grad():
                embedding = self.model.forward_passages([image], batch_size=1)
                image_embeddings.append(embedding)
        image_embeddings = torch.cat(image_embeddings, dim=0)

        print(f'  Text embeddings shape: {text_embeddings.shape}')
        print(f'  Image embeddings shape: {image_embeddings.shape}')

        # Compute similarity scores
        print('Computing similarity scores...')
        with torch.no_grad():
            similarity_scores = self.model.get_scores(text_embeddings, image_embeddings)

        # Normalize scores per query (divide by max score for each query)
        max_scores = torch.max(similarity_scores, dim=1, keepdim=True)[0]
        normalized_similarity = similarity_scores / max_scores

        # Cache results if requested
        if use_cache:
            self._similarity_matrix_cache = normalized_similarity
            self._query_embeddings_cache = text_embeddings
            self._image_embeddings_cache = image_embeddings
            print("Similarity matrix cached")

        return normalized_similarity

    def _compute_jump_penalty_matrix(
        self,
        num_slides: int,
        jump_penalty: float,
        backward_weight: float,
        device: str
    ) -> torch.Tensor:
        """
        Precompute jump penalty matrix for all possible slide transitions.

        Args:
            num_slides: Number of slides
            jump_penalty: Base penalty for jumping between slides
            backward_weight: Multiplier for backward jumps
            device: Device to create the tensor on

        Returns:
            Penalty matrix of shape (num_slides x num_slides)
        """
        slide_indices = torch.arange(num_slides, device=device)
        prev_slides = slide_indices.unsqueeze(1)  # [num_slides, 1]
        curr_slides = slide_indices.unsqueeze(0)  # [1, num_slides]

        # Forward jumps: prev < curr, penalty = (curr - prev - 1) * jump_penalty
        forward_penalty = (curr_slides - prev_slides - 1) * jump_penalty

        # Backward jumps: curr < prev, penalty = (prev - curr) * jump_penalty * backward_weight
        backward_penalty = (prev_slides - curr_slides) * jump_penalty * backward_weight

        # Combine penalties based on jump direction
        penalty_matrix = torch.where(
            prev_slides < curr_slides,
            forward_penalty,
            backward_penalty
        )

        return penalty_matrix

    def match_with_dynamic_programming(
        self,
        similarity_matrix: torch.Tensor,
        text_queries: List[str],
        jump_penalty: float = 1.5,
        backward_weight: float = 1.85,
        use_exponential_scaling: bool = True,
        exponential_scale: float = 2.785,
        use_confidence_boost: bool = True,
        confidence_threshold: float = 0.913,
        confidence_weight: float = 2.18,
        use_context_similarity: bool = True,
        context_weight: float = 0.04,
        context_update_rate: float = 0.24,
        min_sentence_length: int = 2
    ) -> List[Dict]:
        """
        Find optimal transcript-to-slide matching using dynamic programming.

        Args:
            similarity_matrix: Pre-computed similarity matrix (num_queries x num_slides)
            text_queries: List of original text queries
            jump_penalty: Penalty for non-sequential slide transitions
            backward_weight: Multiplier for backward jump penalties
            use_exponential_scaling: Apply exponential scaling to similarity scores
            exponential_scale: Scale factor for exponential transformation
            use_confidence_boost: Boost scores when second-best match is weak
            confidence_threshold: Threshold for applying confidence boost
            confidence_weight: Weight multiplier for confidence boost
            use_context_similarity: Enable context-aware scoring using EMA
            context_weight: Weight for context similarity contribution
            context_update_rate: EMA update rate for context scores
            min_sentence_length: Minimum word count to use similarity score

        Returns:
            List of matching results with matched page numbers and confidence scores
        """
        # Move to appropriate device and ensure float32 precision
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        scores = similarity_matrix.to(device, dtype=torch.float32)
        num_queries, num_slides = scores.shape

        # Apply exponential scaling to amplify score differences
        if use_exponential_scaling:
            scores = torch.exp(exponential_scale * (scores - 1))

        # Apply confidence boost for queries with weak second-best matches
        if use_confidence_boost:
            top2_scores, _ = torch.topk(scores, k=2, dim=1)
            second_best_scores = top2_scores[:, 1]

            boost_mask = (second_best_scores < confidence_threshold).unsqueeze(1)
            scores = torch.where(boost_mask, scores * confidence_weight, scores)

        # Initialize context scores (exponential moving average per slide)
        context_scores = torch.zeros(num_slides, device=device, dtype=torch.float32)

        # Initialize DP table and backtracking pointers
        dp = torch.full((num_queries, num_slides), float('-inf'), device=device, dtype=torch.float32)
        backtrack = torch.zeros((num_queries, num_slides), device=device, dtype=torch.long)

        # Initialize first query
        first_word_count = len(text_queries[0].split())
        if first_word_count >= min_sentence_length:
            dp[0, :] = scores[0, :]
            if use_context_similarity:
                context_scores = context_scores + context_update_rate * (scores[0, :] - context_scores)
        else:
            dp[0, :] = 0.0

        # Precompute jump penalty matrix
        penalty_matrix = self._compute_jump_penalty_matrix(
            num_slides, jump_penalty, backward_weight, device
        )

        # Fill DP table using vectorized operations
        for i in range(1, num_queries):
            word_count = len(text_queries[i].split())
            sentence_valid = word_count >= min_sentence_length

            # Get current scores (with optional context boost)
            if sentence_valid:
                current_scores = scores[i, :].unsqueeze(0)  # [1, num_slides]
                if use_context_similarity:
                    current_scores = current_scores + context_weight * context_scores.unsqueeze(0)
            else:
                current_scores = torch.zeros(1, num_slides, device=device, dtype=torch.float32)

            # Vectorized DP transition
            prev_dp = dp[i - 1, :].unsqueeze(1)  # [num_slides, 1]
            current_scores_grid = current_scores.expand(num_slides, num_slides)

            # Compute transition scores: dp[i-1, k] + score[i, j] - penalty[k, j]
            transition_scores = prev_dp + current_scores_grid - penalty_matrix

            # Find best previous slide for each current slide
            dp[i, :], backtrack[i, :] = torch.max(transition_scores, dim=0)

            # Update context scores
            if use_context_similarity and sentence_valid:
                context_scores = context_scores + context_update_rate * (scores[i, :] - context_scores)

        # Backtrack to find optimal path
        matched_slides = torch.zeros(num_queries, device=device, dtype=torch.long)
        matched_slides[-1] = torch.argmax(dp[-1, :])

        for i in range(num_queries - 2, -1, -1):
            matched_slides[i] = backtrack[i + 1, matched_slides[i + 1]]

        # Extract confidence scores for matched slides
        query_indices = torch.arange(num_queries, device=device).unsqueeze(1)
        confidence_scores = similarity_matrix.to(device)[query_indices, matched_slides.unsqueeze(1)].squeeze()

        # Convert to CPU and build results
        matched_slides_cpu = matched_slides.cpu().numpy()
        confidence_scores_cpu = confidence_scores.cpu().numpy()

        results = []
        for i, query in enumerate(text_queries):
            result = {
                "text": query,
                "matched_page": int(matched_slides_cpu[i]) + 1,  # Convert to 1-based index
                "confidence_score": float(confidence_scores_cpu[i])
            }
            results.append(result)

        return results

    def match_transcript_to_slides(
        self,
        transcript: str,
        pdf_path: str,
        sentences: Optional[List[str]] = None,
        jump_penalty: float = 1.5,
        backward_weight: float = 1.85,
        use_exponential_scaling: bool = True,
        exponential_scale: float = 2.785,
        use_confidence_boost: bool = True,
        confidence_threshold: float = 0.913,
        confidence_weight: float = 2.18,
        use_context_similarity: bool = True,
        context_weight: float = 0.04,
        context_update_rate: float = 0.24,
        min_sentence_length: int = 2,
        use_cached_similarity: bool = True
    ) -> List[Dict]:
        """
        Match transcript sentences to PDF slides.

        Args:
            transcript: Full transcript text (used if sentences not provided)
            pdf_path: Path to the PDF file
            sentences: Optional pre-split sentences (if None, uses full transcript)
            jump_penalty: Penalty for non-sequential slide transitions
            backward_weight: Multiplier for backward jump penalties
            use_exponential_scaling: Apply exponential scaling to scores
            exponential_scale: Scale factor for exponential transformation
            use_confidence_boost: Boost scores when second-best match is weak
            confidence_threshold: Threshold for confidence boost
            confidence_weight: Weight multiplier for confidence boost
            use_context_similarity: Enable context-aware scoring
            context_weight: Weight for context similarity
            context_update_rate: EMA update rate for context
            min_sentence_length: Minimum word count to use similarity score
            use_cached_similarity: Whether to reuse cached similarity matrix

        Returns:
            List of matching results with page numbers and confidence scores
        """
        if self.model is None:
            self.load_model()

        # Prepare text queries
        text_queries = sentences if sentences is not None else [transcript]

        # Compute or retrieve similarity matrix
        if use_cached_similarity and self._similarity_matrix_cache is not None:
            similarity_matrix = self._similarity_matrix_cache
        else:
            slide_images = self.extract_slide_images(pdf_path)
            similarity_matrix = self.compute_similarity_matrix(
                text_queries,
                slide_images,
                use_cache=True
            )

        # Perform dynamic programming matching
        results = self.match_with_dynamic_programming(
            similarity_matrix,
            text_queries,
            jump_penalty=jump_penalty,
            backward_weight=backward_weight,
            use_exponential_scaling=use_exponential_scaling,
            exponential_scale=exponential_scale,
            use_confidence_boost=use_confidence_boost,
            confidence_threshold=confidence_threshold,
            confidence_weight=confidence_weight,
            use_context_similarity=use_context_similarity,
            context_weight=context_weight,
            context_update_rate=context_update_rate,
            min_sentence_length=min_sentence_length
        )

        return results
