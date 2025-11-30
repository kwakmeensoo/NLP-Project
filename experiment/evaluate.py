#!/usr/bin/env python3
"""
Evaluation Module for Slide Matching Algorithm

This module evaluates the slide matching algorithm on benchmark datasets.
Provides comprehensive metrics and detailed analysis reports.
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from match import SlideMatchingProcessor


class MatchingEvaluator:
    """
    Evaluates slide matching algorithm performance on lecture datasets.
    """

    def __init__(
        self,
        data_dir: str = "./dataset",
        output_dir: str = "./evaluation_results",
        device: str = "cuda",
        batch_size: int = 4,
        use_full_dataset: bool = False,
        model_size: str = "3b"
    ):
        """
        Initialize the evaluator.

        Args:
            data_dir: Directory containing lecture data (default: ./dataset)
            output_dir: Directory to save evaluation results
            device: Device to run on ('cuda' or 'cpu')
            batch_size: Batch size for embedding computation
            use_full_dataset: If True, use all 20 lectures; otherwise use first 4
            model_size: Model size to use ('1b' or '3b', default: '3b')
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = device
        self.batch_size = batch_size
        self.use_full_dataset = use_full_dataset
        self.model_size = model_size

        self.processor = None

    def get_lecture_metadata(self) -> List[Dict[str, str]]:
        """
        Get metadata for all available lectures.

        Returns:
            List of lecture metadata dictionaries
        """
        all_lectures = [
            {
                'name': 'cryptocurrency',
                'pdf': 'cryptocurrency.pdf',
                'ground_truth': 'ground_truth_cryptocurrency_MIT.xlsx',
                'srt': 'cryptocurrency.srt'
            },
            {
                'name': 'psychology',
                'pdf': 'psychology.pdf',
                'ground_truth': 'ground_truth_psychology.xlsx',
                'srt': 'psychology_MIT.srt'
            },
            {
                'name': 'reinforcement_learning',
                'pdf': 'reinforcement_learning.pdf',
                'ground_truth': 'ground_truth_reinforcement_learning.xlsx',
                'srt': 'reinforcement_learning_silver.srt'
            },
            {
                'name': 'cognitive_robotics_MIT',
                'pdf': 'cognitive_robotics_MIT.pdf',
                'ground_truth': 'ground_truth_cognitive_robotics_MIT.xlsx',
                'srt': 'cognitive_robotics_MIT.srt'
            },
            {
                'name': 'ML_for_health',
                'pdf': 'ML_for_health_care_MIT.pdf',
                'ground_truth': 'ground_truth_ML_for_health_MIT.xlsx',
                'srt': 'ML_for_health_MIT.srt'
            },
            {
                'name': 'climate_science_policy',
                'pdf': 'climate_science_policy_MIT.pdf',
                'ground_truth': 'ground_truth_climate_science_policy_MIT2.xlsx',
                'srt': 'climate_science_policy_MIT.srt'
            },
            {
                'name': 'cities_and_climate',
                'pdf': 'cities&climate.pdf',
                'ground_truth': 'ground_truth_climate_and_cities.xlsx',
                'srt': 'cities&climate.srt'
            },
            {
                'name': 'cities_and_decarbonization',
                'pdf': 'cities&decarbonization.pdf',
                'ground_truth': 'ground_truth_cities_and_decarbonization.xlsx',
                'srt': 'cities&decarbonization.srt'
            },
            {
                'name': 'computer_vision_2_2',
                'pdf': 'computer_vision_2_2_geiger.pdf',
                'ground_truth': 'ground_truth_computer_vision_2_2.xlsx',
                'srt': 'computer_vision_2_2_geiger.srt'
            },
            {
                'name': 'creating_breakthrough_products',
                'pdf': 'creating_breakthrough_products.pdf',
                'ground_truth': 'ground_truth_creating_breakthrough_products_MIT.xlsx',
                'srt': 'creating_breakthrough_products.srt'
            },
            {
                'name': 'deeplearning',
                'pdf': 'deeplearning_goodfellow.pdf',
                'ground_truth': 'ground_truth_deeplearning.xlsx',
                'srt': 'deeplearning_goodfellow.srt'
            },
            {
                'name': 'phonetics',
                'pdf': 'phonetics.pdf',
                'ground_truth': 'ground_truth_phonetics.xlsx',
                'srt': 'phonetics.srt'
            },
            {
                'name': 'numerics',
                'pdf': 'numerics.pdf',
                'ground_truth': 'ground_truth_numerics.xlsx',
                'srt': 'numerics.srt'
            },
            {
                'name': 'image_processing',
                'pdf': 'image_processing.pdf',
                'ground_truth': 'ground_truth_image_processing.xlsx',
                'srt': 'image_processing.srt'
            },
            {
                'name': 'physics',
                'pdf': 'physics_intro_02.pdf',
                'ground_truth': 'ground_truth_physics.xlsx',
                'srt': 'physics_intro_02.srt'
            },
            {
                'name': 'sensory_systems',
                'pdf': 'sensory_systems.pdf',
                'ground_truth': 'ground_truth_sensory_systems.xlsx',
                'srt': 'sensory_systems.srt'
            },
            {
                'name': 'short_range',
                'pdf': 'short_range_mit.pdf',
                'ground_truth': 'ground_truth_short_range.xlsx',
                'srt': 'short_range_mit.srt'
            },
            {
                'name': 'solar_resource',
                'pdf': 'solar_resource.pdf',
                'ground_truth': 'ground_truth_solar_resource.xlsx',
                'srt': 'solar_resource.srt'
            },
            {
                'name': 'team_dynamics_game_design',
                'pdf': 'team_dynamics_game_design_mit.pdf',
                'ground_truth': 'ground_truth_team_dynamics_game_design_MIT.xlsx',
                'srt': 'team_dynamics_game_design_mit.srt'
            },
            {
                'name': 'theory_of_computation',
                'pdf': 'theory_of_computation.pdf',
                'ground_truth': 'ground_truth_theory_of_computation.xlsx',
                'srt': 'theory_of_computation.srt'
            }
        ]

        return all_lectures if self.use_full_dataset else all_lectures[:4]

    def find_lecture_files(self) -> List[Dict[str, str]]:
        """
        Find and verify lecture files exist.

        Returns:
            List of lecture information with validated file paths
        """
        lectures = []
        metadata = self.get_lecture_metadata()

        lectures_dir = self.data_dir / "lectures"
        ground_truth_dir = self.data_dir / "ground_truth_files"

        for lecture in metadata:
            pdf_path = lectures_dir / lecture['pdf']
            gt_path = ground_truth_dir / lecture['ground_truth']

            if pdf_path.exists() and gt_path.exists():
                lectures.append({
                    'name': lecture['name'],
                    'pdf_path': str(pdf_path),
                    'ground_truth_path': str(gt_path)
                })
            else:
                print(f"⚠️  Missing files for {lecture['name']}")

        return lectures

    def load_ground_truth_sentences(self, gt_path: str) -> List[Dict[str, Any]]:
        """
        Load sentences with ground truth labels from Excel file.

        Args:
            gt_path: Path to ground truth Excel file

        Returns:
            List of dictionaries containing sentence text and ground truth page
        """
        df = pd.read_excel(gt_path)
        df_clean = df.rename(columns={'Value': 'text', 'Slidenumber': 'gt_page'})

        # Filter out empty or whitespace-only sentences
        df_filtered = df_clean[
            df_clean['text'].notna() &
            (df_clean['text'].astype(str).str.strip() != '')
        ]

        return df_filtered.to_dict(orient='records')

    def evaluate_lecture(
        self,
        lecture_info: Dict[str, str],
        matching_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate matching algorithm on a single lecture.

        Args:
            lecture_info: Lecture information (name, paths)
            matching_params: Parameters for the matching algorithm

        Returns:
            Dictionary containing evaluation results
        """
        print(f"\n{'='*60}")
        print(f"📚 Evaluating: {lecture_info['name']}")
        print(f"{'='*60}")

        # Load ground truth sentences
        ground_truth_data = self.load_ground_truth_sentences(
            lecture_info['ground_truth_path']
        )
        sentences = [str(item['text']).strip() for item in ground_truth_data]

        start_time = time.time()

        try:
            # Run matching algorithm
            predictions = self.processor.match_transcript_to_slides(
                transcript="",
                pdf_path=lecture_info['pdf_path'],
                sentences=sentences,
                use_cached_similarity=False,
                **matching_params
            )

            # Get similarity matrix for detailed analysis
            similarity_matrix = self.processor._similarity_matrix_cache

            # Compute metrics
            evaluation_results = self._compute_metrics(
                predictions,
                ground_truth_data,
                similarity_matrix
            )

            evaluation_results['lecture_name'] = lecture_info['name']
            evaluation_results['time_elapsed'] = time.time() - start_time
            evaluation_results['status'] = 'success'

            print(f"   ✅ Success")
            print(f"   Accuracy: {evaluation_results['accuracy']:.4f} "
                  f"({evaluation_results['correct']}/{evaluation_results['evaluated']})")
            print(f"   MAE: {evaluation_results['mae']:.2f} pages")
            print(f"   RMSE: {evaluation_results['rmse']:.2f} pages")
            print(f"   Time: {evaluation_results['time_elapsed']:.2f}s")

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            evaluation_results = {
                'lecture_name': lecture_info['name'],
                'status': 'failed',
                'error': str(e),
                'detailed_predictions': []
            }

        return evaluation_results

    def _compute_metrics(
        self,
        predictions: List[Dict],
        ground_truth_data: List[Dict],
        similarity_matrix: Optional[Any]
    ) -> Dict[str, Any]:
        """
        Compute evaluation metrics from predictions and ground truth.

        Args:
            predictions: List of prediction results
            ground_truth_data: List of ground truth annotations
            similarity_matrix: Cached similarity matrix (optional)

        Returns:
            Dictionary containing computed metrics and detailed results
        """
        import torch

        detailed_predictions = []
        correct_count = 0
        evaluated_count = 0

        for i, (pred, gt) in enumerate(zip(predictions, ground_truth_data)):
            predicted_page = pred['matched_page']
            ground_truth_page = gt['gt_page']
            text = gt['text']

            # Extract top-2 similarity scores
            if similarity_matrix is not None:
                sentence_scores = similarity_matrix[i].cpu().numpy()
                top2_indices = sentence_scores.argsort()[-2:][::-1]
                top1_score = float(sentence_scores[top2_indices[0]])
                top2_score = float(sentence_scores[top2_indices[1]]) if len(top2_indices) > 1 else 0.0
                top1_page = int(top2_indices[0]) + 1
                top2_page = int(top2_indices[1]) + 1 if len(top2_indices) > 1 else -1
            else:
                top1_score = pred['confidence_score']
                top2_score = 0.0
                top1_page = predicted_page
                top2_page = -1

            # Compute correctness (only for evaluated sentences)
            is_correct = None
            error = None
            is_evaluated = ground_truth_page != -1

            if is_evaluated:
                is_correct = (predicted_page == ground_truth_page)
                error = predicted_page - ground_truth_page
                evaluated_count += 1
                if is_correct:
                    correct_count += 1

            detailed_predictions.append({
                'sentence_id': i + 1,
                'text': text,
                'predicted_page': predicted_page,
                'ground_truth_page': ground_truth_page,
                'is_correct': is_correct,
                'error': error,
                'top1_similarity': top1_score,
                'top2_similarity': top2_score,
                'similarity_gap': top1_score - top2_score,
                'top1_page': top1_page,
                'top2_page': top2_page,
                'evaluated': is_evaluated
            })

        # Calculate aggregate metrics
        accuracy = correct_count / evaluated_count if evaluated_count > 0 else 0.0
        errors = [r['error'] for r in detailed_predictions if r['evaluated'] and r['error'] is not None]
        mae = np.mean(np.abs(errors)) if errors else 0.0
        rmse = np.sqrt(np.mean(np.square(errors))) if errors else 0.0

        return {
            'total_sentences': len(predictions),
            'evaluated': evaluated_count,
            'skipped': len(predictions) - evaluated_count,
            'correct': correct_count,
            'incorrect': evaluated_count - correct_count,
            'accuracy': accuracy,
            'mae': mae,
            'rmse': rmse,
            'detailed_predictions': detailed_predictions
        }

    def run_evaluation(
        self,
        matching_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run evaluation on all lectures.

        Args:
            matching_params: Parameters for matching algorithm (uses defaults if None)

        Returns:
            Dictionary containing all evaluation results and summary
        """
        # Default parameters
        default_params = {
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

        # Merge with provided parameters
        params = {**default_params, **(matching_params or {})}

        print("="*80)
        print("🔍 Slide Matching Algorithm Evaluation")
        print("="*80)

        print(f"\n📊 Matching parameters:")
        for key, value in params.items():
            print(f"   {key}: {value}")

        # Find lectures
        lectures = self.find_lecture_files()

        if not lectures:
            print("❌ No lecture files found!")
            return {'results': [], 'summary': {}}

        print(f"\n📚 Evaluating on {len(lectures)} lectures")

        # Initialize processor
        if self.processor is None:
            model_name = f'nvidia/llama-nemoretriever-colembed-{self.model_size}-v1'
            self.processor = SlideMatchingProcessor(
                model_name=model_name,
                device=self.device,
                batch_size=self.batch_size
            )

        # Run evaluation
        all_results = []

        for lecture_info in tqdm(lectures, desc="Evaluating lectures"):
            result = self.evaluate_lecture(lecture_info, params)
            all_results.append(result)

            # Clear cache after each lecture
            self.processor.clear_cache()

        # Cleanup
        if self.processor is not None:
            self.processor.unload_model()

        # Generate summary
        summary = self._generate_summary(all_results)

        # Save results
        self._save_results(all_results, summary, params)

        return {
            'results': all_results,
            'summary': summary,
            'parameters': params
        }

    def _generate_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Generate summary statistics from evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            Summary statistics dictionary
        """
        successful = [r for r in results if r['status'] == 'success']

        if not successful:
            return {
                'total_lectures': len(results),
                'successful': 0,
                'failed': len(results),
                'overall_accuracy': 0.0
            }

        # Aggregate metrics (lecture-wise average)
        total_evaluated = sum(r['evaluated'] for r in successful)
        total_correct = sum(r['correct'] for r in successful)
        overall_accuracy = np.mean([r['accuracy'] for r in successful]) if successful else 0.0

        # Per-lecture statistics
        lecture_stats = []
        for r in successful:
            lecture_stats.append({
                'lecture_name': r['lecture_name'],
                'accuracy': r['accuracy'],
                'evaluated': r['evaluated'],
                'mae': r['mae'],
                'rmse': r['rmse'],
                'time': r['time_elapsed']
            })

        # Sort by accuracy descending
        lecture_stats.sort(key=lambda x: x['accuracy'], reverse=True)

        return {
            'total_lectures': len(results),
            'successful': len(successful),
            'failed': len(results) - len(successful),
            'overall_accuracy': overall_accuracy,
            'total_evaluated': total_evaluated,
            'total_correct': total_correct,
            'average_mae': np.mean([r['mae'] for r in successful]),
            'average_rmse': np.mean([r['rmse'] for r in successful]),
            'total_time': sum(r['time_elapsed'] for r in successful),
            'lecture_stats': lecture_stats
        }

    def _save_results(
        self,
        results: List[Dict],
        summary: Dict,
        params: Dict
    ):
        """
        Save evaluation results to files.

        Args:
            results: List of evaluation results
            summary: Summary statistics
            params: Parameters used for matching
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Create run-specific subdirectory
        run_dir = self.output_dir / f"run_{timestamp}"
        run_dir.mkdir(exist_ok=True, parents=True)

        # Save JSON summary
        summary_file = run_dir / "evaluation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            results_without_details = [
                {k: v for k, v in r.items() if k != 'detailed_predictions'}
                for r in results
            ]
            json.dump({
                'parameters': params,
                'results': results_without_details,
                'summary': summary
            }, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to: {run_dir}")
        print(f"   📄 Summary: {summary_file}")

        # Save per-lecture Excel files
        successful = [r for r in results if r['status'] == 'success']

        for result in successful:
            lecture_name = result['lecture_name']
            predictions = result['detailed_predictions']

            if not predictions:
                continue

            # Create detailed results DataFrame
            df = pd.DataFrame(predictions)

            # Add status symbols
            df['status'] = df['is_correct'].apply(
                lambda x: '✓' if x is True else ('✗' if x is False else '-')
            )

            # Reorder columns
            column_order = [
                'sentence_id', 'text', 'ground_truth_page', 'predicted_page',
                'error', 'status', 'top1_similarity', 'top2_similarity',
                'similarity_gap', 'top1_page', 'top2_page', 'evaluated'
            ]
            df = df[column_order]

            # Save to Excel in run-specific directory
            excel_file = run_dir / f"{lecture_name}.xlsx"

            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # Main results sheet
                df.to_excel(writer, sheet_name='Predictions', index=False)

                # Summary sheet
                summary_data = {
                    'Metric': [
                        'Total Sentences', 'Evaluated Sentences',
                        'Skipped Sentences', 'Correct', 'Incorrect',
                        'Accuracy', 'MAE', 'RMSE'
                    ],
                    'Value': [
                        result['total_sentences'], result['evaluated'],
                        result['skipped'], result['correct'], result['incorrect'],
                        f"{result['accuracy']:.4f}",
                        f"{result['mae']:.2f}",
                        f"{result['rmse']:.2f}"
                    ]
                }
                pd.DataFrame(summary_data).to_excel(
                    writer, sheet_name='Summary', index=False
                )

                # Errors sheet
                errors = [p for p in predictions if p['evaluated'] and not p['is_correct']]
                if errors:
                    df_errors = pd.DataFrame(errors)
                    df_errors = df_errors[[
                        'sentence_id', 'text', 'ground_truth_page',
                        'predicted_page', 'error', 'top1_similarity',
                        'top2_similarity', 'similarity_gap'
                    ]]
                    df_errors.to_excel(writer, sheet_name='Errors', index=False)

            print(f"   📊 {lecture_name}: {excel_file.name}")

    def print_report(self, evaluation_data: Dict):
        """
        Print evaluation report to console.

        Args:
            evaluation_data: Evaluation results from run_evaluation()
        """
        summary = evaluation_data['summary']

        print(f"\n{'='*80}")
        print("📊 EVALUATION REPORT")
        print(f"{'='*80}")

        print(f"\n📈 Overall Statistics:")
        print(f"   Total lectures: {summary['total_lectures']}")
        print(f"   Successful: {summary['successful']}")
        print(f"   Failed: {summary['failed']}")
        print(f"   Overall accuracy (lecture-wise avg): {summary['overall_accuracy']:.4f}")
        print(f"   Total evaluated sentences: {summary['total_evaluated']}")
        print(f"   Total correct: {summary['total_correct']}")
        print(f"   Average MAE: {summary['average_mae']:.2f} pages")
        print(f"   Average RMSE: {summary['average_rmse']:.2f} pages")
        print(f"   Total time: {summary['total_time']:.2f}s")

        print(f"\n📚 Per-Lecture Results (sorted by accuracy):")
        for i, stats in enumerate(summary['lecture_stats'], 1):
            print(f"   {i:2d}. {stats['lecture_name']:<35} | "
                  f"Acc: {stats['accuracy']:.4f} | "
                  f"N: {stats['evaluated']:>4} | "
                  f"MAE: {stats['mae']:>5.2f} | "
                  f"RMSE: {stats['rmse']:>5.2f}")

        print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate slide matching algorithm on lecture datasets'
    )

    parser.add_argument('--data-dir', default='./dataset',
                        help='Path to dataset directory')
    parser.add_argument('--output-dir', default='./evaluation_results',
                        help='Path to output directory')
    parser.add_argument('--device', default='cuda',
                        help='Device to run on (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for embeddings')
    parser.add_argument('--full', action='store_true',
                        help='Use all 20 lectures (default: first 4)')
    parser.add_argument('--model-size', default='3b', choices=['1b', '3b'],
                        help='Model size to use (1b or 3b, default: 3b)')

    # Matching algorithm parameters
    parser.add_argument('--jump-penalty', type=float, default=1.5)
    parser.add_argument('--backward-weight', type=float, default=1.85)
    parser.add_argument('--exponential-scaling', action='store_true', default=True)
    parser.add_argument('--exponential-scale', type=float, default=2.785)
    parser.add_argument('--confidence-boost', action='store_true', default=True)
    parser.add_argument('--confidence-threshold', type=float, default=0.913)
    parser.add_argument('--confidence-weight', type=float, default=2.18)
    parser.add_argument('--context-similarity', action='store_true', default=True)
    parser.add_argument('--context-weight', type=float, default=0.04)
    parser.add_argument('--context-update-rate', type=float, default=0.24)
    parser.add_argument('--min-sentence-length', type=int, default=2)

    args = parser.parse_args()

    # Prepare matching parameters
    matching_params = {
        'jump_penalty': args.jump_penalty,
        'backward_weight': args.backward_weight,
        'use_exponential_scaling': args.exponential_scaling,
        'exponential_scale': args.exponential_scale,
        'use_confidence_boost': args.confidence_boost,
        'confidence_threshold': args.confidence_threshold,
        'confidence_weight': args.confidence_weight,
        'use_context_similarity': args.context_similarity,
        'context_weight': args.context_weight,
        'context_update_rate': args.context_update_rate,
        'min_sentence_length': args.min_sentence_length
    }

    # Create evaluator and run
    evaluator = MatchingEvaluator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        use_full_dataset=args.full,
        model_size=args.model_size
    )

    results = evaluator.run_evaluation(matching_params)
    evaluator.print_report(results)


if __name__ == '__main__':
    main()
