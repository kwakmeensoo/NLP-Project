#!/usr/bin/env python3
"""
Grid search for optimal SlideMatchingProcessor parameters.

Evaluates hyperparameter combinations to find best configuration for transcript-to-slide matching.
Parameters not specified in the grid will use their default values from SlideMatchingProcessor.

Default parameter values:
    - jump_penalty: 1.5
    - backward_weight: 1.85
    - use_exponential_scaling: True
    - exponential_scale: 2.785
    - use_confidence_boost: True
    - confidence_threshold: 0.913
    - confidence_weight: 2.18
    - use_context_similarity: True
    - context_weight: 0.04
    - context_update_rate: 0.24
    - min_sentence_length: 2
"""

import json
import time
import argparse
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

from match import SlideMatchingProcessor


class GridSearch:
    """Grid search for SlideMatchingProcessor hyperparameters."""

    def __init__(
        self,
        data_dir: str = "./dataset",
        device: str = "cuda",
        batch_size: int = 4,
        output_dir: str = "./grid_search_results",
        use_full_lectures: bool = False,
        model_size: str = "3b"
    ):
        """
        Initialize grid search.

        Args:
            data_dir: Directory containing dataset (lectures, ground_truth_files)
            device: Device to run on ('cuda' or 'cpu')
            batch_size: Batch size for embedding computation
            output_dir: Directory to save results
            use_full_lectures: If True, use all 20 lectures; else use first 4
            model_size: Model size to use ('1b' or '3b', default: '3b')
        """
        self.data_dir = Path(data_dir)
        self.device = device
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.use_full_lectures = use_full_lectures
        self.model_size = model_size

        # Single shared processor instance for reusing similarity matrices
        self.processor = None

    def get_lecture_mappings(self) -> List[Dict[str, str]]:
        """Get lecture file mappings for evaluation."""
        all_mappings = [
            {
                'name': 'cryptocurrency',
                'pdf': 'cryptocurrency.pdf',
                'gt': 'ground_truth_cryptocurrency_MIT.xlsx',
                'srt': 'cryptocurrency.srt'
            },
            {
                'name': 'psychology',
                'pdf': 'psychology.pdf',
                'gt': 'ground_truth_psychology.xlsx',
                'srt': 'psychology_MIT.srt'
            },
            {
                'name': 'reinforcement_learning',
                'pdf': 'reinforcement_learning.pdf',
                'gt': 'ground_truth_reinforcement_learning.xlsx',
                'srt': 'reinforcement_learning_silver.srt'
            },
            {
                'name': 'cognitive_robotics_MIT',
                'pdf': 'cognitive_robotics_MIT.pdf',
                'gt': 'ground_truth_cognitive_robotics_MIT.xlsx',
                'srt': 'cognitive_robotics_MIT.srt'
            },
            {
                'name': 'ML_for_health',
                'pdf': 'ML_for_health_care_MIT.pdf',
                'gt': 'ground_truth_ML_for_health_MIT.xlsx',
                'srt': 'ML_for_health_MIT.srt'
            },
            {
                'name': 'climate_science_policy',
                'pdf': 'climate_science_policy_MIT.pdf',
                'gt': 'ground_truth_climate_science_policy_MIT2.xlsx',
                'srt': 'climate_science_policy_MIT.srt'
            },
            {
                'name': 'cities_and_climate',
                'pdf': 'cities&climate.pdf',
                'gt': 'ground_truth_climate_and_cities.xlsx',
                'srt': 'cities&climate.srt'
            },
            {
                'name': 'cities_and_decarbonization',
                'pdf': 'cities&decarbonization.pdf',
                'gt': 'ground_truth_cities_and_decarbonization.xlsx',
                'srt': 'cities&decarbonization.srt'
            },
            {
                'name': 'computer_vision_2_2',
                'pdf': 'computer_vision_2_2_geiger.pdf',
                'gt': 'ground_truth_computer_vision_2_2.xlsx',
                'srt': 'computer_vision_2_2_geiger.srt'
            },
            {
                'name': 'creating_breakthrough_products',
                'pdf': 'creating_breakthrough_products.pdf',
                'gt': 'ground_truth_creating_breakthrough_products_MIT.xlsx',
                'srt': 'creating_breakthrough_products.srt'
            },
            {
                'name': 'deeplearning',
                'pdf': 'deeplearning_goodfellow.pdf',
                'gt': 'ground_truth_deeplearning.xlsx',
                'srt': 'deeplearning_goodfellow.srt'
            },
            {
                'name': 'phonetics',
                'pdf': 'phonetics.pdf',
                'gt': 'ground_truth_phonetics.xlsx',
                'srt': 'phonetics.srt'
            },
            {
                'name': 'numerics',
                'pdf': 'numerics.pdf',
                'gt': 'ground_truth_numerics.xlsx',
                'srt': 'numerics.srt'
            },
            {
                'name': 'image_processing',
                'pdf': 'image_processing.pdf',
                'gt': 'ground_truth_image_processing.xlsx',
                'srt': 'image_processing.srt'
            },
            {
                'name': 'physics',
                'pdf': 'physics_intro_02.pdf',
                'gt': 'ground_truth_physics.xlsx',
                'srt': 'physics_intro_02.srt'
            },
            {
                'name': 'sensory_systems',
                'pdf': 'sensory_systems.pdf',
                'gt': 'ground_truth_sensory_systems.xlsx',
                'srt': 'sensory_systems.srt'
            },
            {
                'name': 'short_range',
                'pdf': 'short_range_mit.pdf',
                'gt': 'ground_truth_short_range.xlsx',
                'srt': 'short_range_mit.srt'
            },
            {
                'name': 'solar_resource',
                'pdf': 'solar_resource.pdf',
                'gt': 'ground_truth_solar_resource.xlsx',
                'srt': 'solar_resource.srt'
            },
            {
                'name': 'team_dynamics_game_design',
                'pdf': 'team_dynamics_game_design_mit.pdf',
                'gt': 'ground_truth_team_dynamics_game_design_MIT.xlsx',
                'srt': 'team_dynamics_game_design_mit.srt'
            },
            {
                'name': 'theory_of_computation',
                'pdf': 'theory_of_computation.pdf',
                'gt': 'ground_truth_theory_of_computation.xlsx',
                'srt': 'theory_of_computation.srt'
            }
        ]

        if self.use_full_lectures:
            return all_mappings
        else:
            return all_mappings[:4]

    def find_lecture_pairs(self) -> List[Dict[str, str]]:
        """Find and validate lecture file pairs."""
        lecture_pairs = []
        mappings = self.get_lecture_mappings()

        lectures_dir = self.data_dir / "lectures"
        gt_dir = self.data_dir / "ground_truth_files"

        for mapping in mappings:
            pdf_path = lectures_dir / mapping['pdf']
            gt_path = gt_dir / mapping['gt']

            if pdf_path.exists() and gt_path.exists():
                lecture_pairs.append({
                    'name': mapping['name'],
                    'pdf_path': str(pdf_path),
                    'gt_path': str(gt_path)
                })
            else:
                print(f"Warning: Missing files for {mapping['name']}")

        return lecture_pairs

    def load_ground_truth(self, gt_path: str) -> pd.DataFrame:
        """Load ground truth Excel file."""
        return pd.read_excel(gt_path)

    def get_sentences_from_gt(self, gt_path: str) -> List[Dict]:
        """
        Extract sentences from ground truth file.
        Includes all sentences (even with gt_page == -1) for context.
        Filters out empty or whitespace-only text.
        """
        gt_df = self.load_ground_truth(gt_path)
        df_renamed = gt_df.rename(columns={'Value': 'text', 'Slidenumber': 'gt_page'})

        # Filter out invalid text
        df_filtered = df_renamed[
            df_renamed['text'].notna() &
            (df_renamed['text'].astype(str).str.strip() != '')
        ]

        return df_filtered.to_dict(orient='records')

    def calculate_accuracy(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> float:
        """
        Calculate accuracy, excluding sentences with gt_page == -1.
        Such sentences provide context but are not evaluated.
        """
        valid_pairs = [
            (pred, gt) for pred, gt in zip(predictions, ground_truth)
            if gt['gt_page'] != -1
        ]

        if not valid_pairs:
            return 0.0

        correct = sum(1 for pred, gt in valid_pairs
                     if pred['matched_page'] == gt['gt_page'])
        return correct / len(valid_pairs)

    def evaluate_configuration(
        self,
        lecture_info: Dict[str, str],
        params: Dict[str, Any],
        is_first_config: bool = False
    ) -> Dict:
        """
        Evaluate a single parameter configuration on one lecture.

        Args:
            lecture_info: Dictionary with lecture name, pdf_path, gt_path
            params: Hyperparameters to evaluate
            is_first_config: If True, computes similarity matrix; else reuses cache

        Returns:
            Dictionary with evaluation results
        """
        sentences_with_gt = self.get_sentences_from_gt(lecture_info['gt_path'])
        sentences = [str(s['text']).strip() for s in sentences_with_gt]

        start_time = time.time()

        try:
            # Use cached similarity matrix for all configs except the first
            results = self.processor.match_transcript_to_slides(
                transcript="",
                pdf_path=lecture_info['pdf_path'],
                sentences=sentences,
                use_cached_similarity=not is_first_config,
                **params
            )

            accuracy = self.calculate_accuracy(results, sentences_with_gt)

            result = {
                'lecture_name': lecture_info['name'],
                **params,
                'accuracy': accuracy,
                'total_sentences': len(sentences),
                'time': time.time() - start_time,
                'status': 'success'
            }

        except Exception as e:
            result = {
                'lecture_name': lecture_info['name'],
                **params,
                'accuracy': 0.0,
                'total_sentences': len(sentences) if sentences else 0,
                'time': time.time() - start_time,
                'status': 'failed',
                'error': str(e)
            }

        return result

    def generate_parameter_combinations(
        self,
        param_grid: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate all valid parameter combinations.
        Filters out invalid combinations where feature weights are varied
        when their corresponding boolean flag is disabled.
        """
        keys = param_grid.keys()
        values = param_grid.values()

        all_combinations = []

        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))

            # Validate combination: skip if weight params vary when feature is disabled
            skip = False

            # Exponential scaling validation
            if 'use_exponential_scaling' in param_dict and not param_dict['use_exponential_scaling']:
                if 'exponential_scale' in param_dict and 'exponential_scale' in param_grid:
                    if len(param_grid['exponential_scale']) > 1:
                        if param_dict['exponential_scale'] != param_grid['exponential_scale'][0]:
                            skip = True

            # Confidence boost validation
            if 'use_confidence_boost' in param_dict and not param_dict['use_confidence_boost']:
                if 'confidence_threshold' in param_dict and 'confidence_threshold' in param_grid:
                    if len(param_grid['confidence_threshold']) > 1:
                        if param_dict['confidence_threshold'] != param_grid['confidence_threshold'][0]:
                            skip = True
                if 'confidence_weight' in param_dict and 'confidence_weight' in param_grid:
                    if len(param_grid['confidence_weight']) > 1:
                        if param_dict['confidence_weight'] != param_grid['confidence_weight'][0]:
                            skip = True

            # Context similarity validation
            if 'use_context_similarity' in param_dict and not param_dict['use_context_similarity']:
                if 'context_weight' in param_dict and 'context_weight' in param_grid:
                    if len(param_grid['context_weight']) > 1:
                        if param_dict['context_weight'] != param_grid['context_weight'][0]:
                            skip = True
                if 'context_update_rate' in param_dict and 'context_update_rate' in param_grid:
                    if len(param_grid['context_update_rate']) > 1:
                        if param_dict['context_update_rate'] != param_grid['context_update_rate'][0]:
                            skip = True

            if not skip:
                all_combinations.append(param_dict)

        return all_combinations

    def run_grid_search(
        self,
        param_grid: Dict[str, List[Any]]
    ) -> Dict:
        """
        Execute grid search across all parameter combinations.

        Optimizes by computing similarity matrix once per lecture
        and reusing it across all parameter configurations.

        Args:
            param_grid: Dictionary mapping parameter names to value lists
                       Example: {
                           'jump_penalty': [0.08, 0.10, 0.12],
                           'backward_weight': [1.5, 2.0, 2.5],
                           'use_context_similarity': [True, False]
                       }

                       Note: Parameters not specified in param_grid will use
                       their default values from SlideMatchingProcessor:
                       - jump_penalty: 1.5
                       - backward_weight: 1.85
                       - use_exponential_scaling: True
                       - exponential_scale: 2.785
                       - use_confidence_boost: True
                       - confidence_threshold: 0.913
                       - confidence_weight: 2.18
                       - use_context_similarity: True
                       - context_weight: 0.04
                       - context_update_rate: 0.24
                       - min_sentence_length: 2

        Returns:
            Dictionary containing all results and summary statistics
        """
        print("=" * 80)
        print("Grid Search for SlideMatchingProcessor")
        print("=" * 80)

        # Generate all parameter combinations
        param_combinations = self.generate_parameter_combinations(param_grid)

        print(f"\nParameter Grid:")
        for param_name, values in param_grid.items():
            print(f"   {param_name}: {values} ({len(values)} values)")
        print(f"\n   Valid combinations: {len(param_combinations)}")
        print(f"\nNote: Parameters not specified will use default values from SlideMatchingProcessor")

        # Find lectures
        lecture_pairs = self.find_lecture_pairs()

        if not lecture_pairs:
            print("Error: No lecture pairs found!")
            return {'results': [], 'summary': {}}

        print(f"\nEvaluating on {len(lecture_pairs)} lectures:")
        for i, lecture in enumerate(lecture_pairs, 1):
            print(f"   {i}. {lecture['name']}")

        # Initialize processor
        if self.processor is None:
            model_name = f'nvidia/llama-nemoretriever-colembed-{self.model_size}-v1'
            self.processor = SlideMatchingProcessor(
                model_name=model_name,
                device=self.device,
                batch_size=self.batch_size
            )

        # Run evaluations
        all_results = []
        total_evaluations = len(param_combinations) * len(lecture_pairs)

        print(f"\nRunning {total_evaluations} total evaluations...")
        print("Similarity matrices computed once per lecture and reused")

        pbar = tqdm(total=total_evaluations, desc="Grid Search Progress")

        # Iterate lecture-first to maximize similarity matrix cache reuse
        for lecture_idx, lecture_info in enumerate(lecture_pairs):
            print(f"\n{'=' * 60}")
            print(f"Processing lecture: {lecture_info['name']}")
            print(f"{'=' * 60}")

            # Clear cache when switching to new lecture
            if lecture_idx > 0:
                self.processor.clear_cache()

            for param_idx, params in enumerate(param_combinations):
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                pbar.set_description(f"{lecture_info['name'][:20]}: {param_str[:40]}")

                # First config computes similarity matrix; others reuse it
                is_first = (param_idx == 0)

                result = self.evaluate_configuration(
                    lecture_info,
                    params,
                    is_first_config=is_first
                )
                all_results.append(result)
                pbar.update(1)

            # Show intermediate results for this lecture
            lecture_results = [r for r in all_results if r['lecture_name'] == lecture_info['name']]
            successful = [r for r in lecture_results if r['status'] == 'success']

            if successful:
                best_result = max(successful, key=lambda x: x['accuracy'])
                avg_time = np.mean([r['time'] for r in successful])

                print(f"\n   Completed {len(successful)}/{len(lecture_results)} configs")
                print(f"   Best accuracy: {best_result['accuracy']:.4f}")
                print(f"   Avg time per config: {avg_time:.2f}s")

        pbar.close()

        # Cleanup
        if self.processor is not None:
            self.processor.unload_model()

        # Generate summary
        summary = self.generate_summary(all_results, param_combinations)

        # Save results
        self.save_results(all_results, summary, param_grid)

        return {
            'results': all_results,
            'summary': summary
        }

    def generate_summary(
        self,
        results: List[Dict],
        param_combinations: List[Dict[str, Any]]
    ) -> Dict:
        """Generate summary statistics from evaluation results."""
        successful_results = [r for r in results if r['status'] == 'success']

        if not successful_results:
            return {
                'total_evaluations': len(results),
                'successful_evaluations': 0,
                'best_params': None,
                'best_accuracy': 0.0
            }

        # Calculate average accuracy for each parameter combination
        param_accuracy = {}

        for params in param_combinations:
            matching_results = [
                r for r in successful_results
                if all(r.get(k) == v for k, v in params.items())
            ]

            if matching_results:
                avg_acc = np.mean([r['accuracy'] for r in matching_results])
                param_key = str(params)
                param_accuracy[param_key] = {
                    'params': params,
                    'avg_accuracy': avg_acc,
                    'num_lectures': len(matching_results)
                }

        # Find best parameters
        best_entry = max(param_accuracy.values(), key=lambda x: x['avg_accuracy'])

        summary = {
            'total_evaluations': len(results),
            'successful_evaluations': len(successful_results),
            'failed_evaluations': len(results) - len(successful_results),
            'best_params': best_entry['params'],
            'best_accuracy': best_entry['avg_accuracy'],
            'all_configs': param_accuracy
        }

        return summary

    def save_results(
        self,
        results: List[Dict],
        summary: Dict,
        param_grid: Dict[str, List[Any]]
    ):
        """Save optimization results to JSON and CSV files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Create run-specific subdirectory
        run_dir = self.output_dir / f"run_{timestamp}"
        run_dir.mkdir(exist_ok=True, parents=True)

        # Save detailed JSON
        results_file = run_dir / "grid_search_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'param_grid': {k: v for k, v in param_grid.items()},
                'results': results,
                'summary': {
                    'total_evaluations': summary['total_evaluations'],
                    'successful_evaluations': summary['successful_evaluations'],
                    'failed_evaluations': summary['failed_evaluations'],
                    'best_params': summary['best_params'],
                    'best_accuracy': summary['best_accuracy']
                }
            }, f, indent=2, ensure_ascii=False)

        # Save CSV
        csv_file = run_dir / "grid_search_results.csv"
        df_results = pd.DataFrame(results)
        df_results.to_csv(csv_file, index=False)

        # Save top configurations
        self.save_top_configurations(results, run_dir)

        print(f"\nResults saved to: {run_dir}")
        print(f"   JSON: {results_file.name}")
        print(f"   CSV: {csv_file.name}")

    def save_top_configurations(
        self,
        results: List[Dict],
        run_dir: Path,
        top_k: int = 30
    ):
        """Save top K configurations ranked by average accuracy."""
        successful_results = [r for r in results if r['status'] == 'success']

        if not successful_results:
            return

        # Identify parameter keys
        param_keys = [k for k in successful_results[0].keys()
                     if k not in ['lecture_name', 'accuracy', 'total_sentences',
                                 'time', 'status', 'error']]

        # Aggregate statistics per configuration
        config_stats = {}

        for result in successful_results:
            config_tuple = tuple((k, result[k]) for k in sorted(param_keys))

            if config_tuple not in config_stats:
                config_stats[config_tuple] = {
                    'accuracies': [],
                    'lectures': [],
                    'times': []
                }

            config_stats[config_tuple]['accuracies'].append(result['accuracy'])
            config_stats[config_tuple]['lectures'].append(result['lecture_name'])
            config_stats[config_tuple]['times'].append(result['time'])

        # Create summary data
        config_summary = []
        for config_tuple, stats in config_stats.items():
            config_dict = dict(config_tuple)
            config_dict['avg_accuracy'] = np.mean(stats['accuracies'])
            config_dict['std_accuracy'] = np.std(stats['accuracies'])
            config_dict['min_accuracy'] = np.min(stats['accuracies'])
            config_dict['max_accuracy'] = np.max(stats['accuracies'])
            config_dict['num_lectures'] = len(stats['accuracies'])
            config_dict['avg_time'] = np.mean(stats['times'])
            config_dict['total_time'] = np.sum(stats['times'])

            # Per-lecture breakdown
            lecture_breakdown = {
                lecture: acc
                for lecture, acc in zip(stats['lectures'], stats['accuracies'])
            }
            config_dict['lecture_accuracies'] = lecture_breakdown

            config_summary.append(config_dict)

        # Sort by average accuracy
        config_summary.sort(key=lambda x: x['avg_accuracy'], reverse=True)
        top_configs = config_summary[:top_k]

        # Save top K configurations as JSON
        top_k_file = run_dir / f"top_{top_k}_configs.json"
        with open(top_k_file, 'w', encoding='utf-8') as f:
            json.dump(top_configs, f, indent=2, ensure_ascii=False)

        # Save top K configurations as CSV (summary)
        top_k_csv_data = []
        for rank, config in enumerate(top_configs, 1):
            row = {'rank': rank}
            for key in param_keys:
                row[key] = config[key]
            row['avg_accuracy'] = config['avg_accuracy']
            row['std_accuracy'] = config['std_accuracy']
            row['min_accuracy'] = config['min_accuracy']
            row['max_accuracy'] = config['max_accuracy']
            row['num_lectures'] = config['num_lectures']
            row['avg_time'] = config['avg_time']
            row['total_time'] = config['total_time']
            top_k_csv_data.append(row)

        top_k_csv_file = run_dir / f"top_{top_k}_configs.csv"
        df_top_k = pd.DataFrame(top_k_csv_data)
        df_top_k.to_csv(top_k_csv_file, index=False)

        # Save per-lecture breakdown
        per_lecture_data = []
        for rank, config in enumerate(top_configs, 1):
            for lecture, accuracy in config['lecture_accuracies'].items():
                row = {'rank': rank}
                for key in param_keys:
                    row[key] = config[key]
                row['lecture'] = lecture
                row['accuracy'] = accuracy
                row['avg_accuracy'] = config['avg_accuracy']
                per_lecture_data.append(row)

        per_lecture_file = run_dir / f"top_{top_k}_per_lecture.csv"
        df_per_lecture = pd.DataFrame(per_lecture_data)
        df_per_lecture.to_csv(per_lecture_file, index=False)

        print(f"   Top {top_k} configs (JSON): {top_k_file.name}")
        print(f"   Top {top_k} summary (CSV): {top_k_csv_file.name}")
        print(f"   Per-lecture breakdown: {per_lecture_file.name}")

    def print_final_report(self, optimization_results: Dict):
        """Print final optimization report."""
        summary = optimization_results['summary']
        results = optimization_results['results']

        print(f"\n{'=' * 80}")
        print("GRID SEARCH REPORT")
        print(f"{'=' * 80}")

        print(f"\nStatistics:")
        print(f"   Total evaluations: {summary['total_evaluations']}")
        print(f"   Successful: {summary['successful_evaluations']}")
        print(f"   Failed: {summary['failed_evaluations']}")

        if summary['best_params']:
            print(f"\nBEST CONFIGURATION:")
            for key, value in summary['best_params'].items():
                print(f"   {key}: {value}")
            print(f"   Average Accuracy: {summary['best_accuracy']:.4f}")

            # Per-lecture breakdown
            print(f"\nPER-LECTURE ACCURACY (Best Configuration):")
            best_results = [
                r for r in results
                if all(r.get(k) == v for k, v in summary['best_params'].items())
                and r['status'] == 'success'
            ]

            for r in sorted(best_results, key=lambda x: x['accuracy'], reverse=True):
                print(f"   {r['lecture_name']:<30} | Accuracy: {r['accuracy']:.4f}")

        print(f"\n{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description='Grid search for SlideMatchingProcessor parameters'
    )

    parser.add_argument('--data-dir', default='./dataset',
                       help='Directory containing dataset')
    parser.add_argument('--output-dir', default='./grid_search_results',
                       help='Directory to save results')
    parser.add_argument('--device', default='cuda',
                       help='Device to run on (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for embedding computation')
    parser.add_argument('--full', action='store_true',
                       help='Use all 20 lectures instead of 4')
    parser.add_argument('--model-size', default='3b', choices=['1b', '3b'],
                       help='Model size to use (1b or 3b, default: 3b)')

    # Parameter grid (JSON format)
    parser.add_argument('--params', type=str,
                       help='JSON string defining parameter grid')

    # Individual parameters
    parser.add_argument('--jump-penalty', type=float, nargs='+',
                       help='Jump penalty values to test')
    parser.add_argument('--backward-weight', type=float, nargs='+',
                       help='Backward weight values to test')
    parser.add_argument('--exponential-scale', type=float, nargs='+',
                       help='Exponential scale values to test')
    parser.add_argument('--confidence-threshold', type=float, nargs='+',
                       help='Confidence threshold values to test')
    parser.add_argument('--confidence-weight', type=float, nargs='+',
                       help='Confidence weight values to test')
    parser.add_argument('--context-weight', type=float, nargs='+',
                       help='Context weight values to test')
    parser.add_argument('--context-update-rate', type=float, nargs='+',
                       help='Context update rate values to test')
    parser.add_argument('--min-sentence-length', type=int, nargs='+',
                       help='Minimum sentence length values to test')

    # Boolean flags
    parser.add_argument('--use-exponential-scaling', type=str, nargs='+',
                       choices=['True', 'False'],
                       help='Whether to use exponential scaling')
    parser.add_argument('--use-confidence-boost', type=str, nargs='+',
                       choices=['True', 'False'],
                       help='Whether to use confidence boost')
    parser.add_argument('--use-context-similarity', type=str, nargs='+',
                       choices=['True', 'False'],
                       help='Whether to use context similarity')

    args = parser.parse_args()

    # Build parameter grid
    param_grid = {}

    if args.params:
        # Load from JSON string
        param_grid = json.loads(args.params)
    else:
        # Build from individual arguments
        if args.jump_penalty:
            param_grid['jump_penalty'] = args.jump_penalty
        if args.backward_weight:
            param_grid['backward_weight'] = args.backward_weight
        if args.exponential_scale:
            param_grid['exponential_scale'] = args.exponential_scale
        if args.confidence_threshold:
            param_grid['confidence_threshold'] = args.confidence_threshold
        if args.confidence_weight:
            param_grid['confidence_weight'] = args.confidence_weight
        if args.context_weight:
            param_grid['context_weight'] = args.context_weight
        if args.context_update_rate:
            param_grid['context_update_rate'] = args.context_update_rate
        if args.min_sentence_length:
            param_grid['min_sentence_length'] = args.min_sentence_length

        if args.use_exponential_scaling:
            param_grid['use_exponential_scaling'] = [v == 'True' for v in args.use_exponential_scaling]
        if args.use_confidence_boost:
            param_grid['use_confidence_boost'] = [v == 'True' for v in args.use_confidence_boost]
        if args.use_context_similarity:
            param_grid['use_context_similarity'] = [v == 'True' for v in args.use_context_similarity]

    if not param_grid:
        print("Error: No parameters specified!")
        print("\nExample usage:")
        print("  python grid_search.py --jump-penalty 0.08 0.10 0.12 --backward-weight 1.5 2.0 2.5")
        print('  python grid_search.py --params \'{"jump_penalty": [0.1, 0.2], "use_context_similarity": [true, false]}\'')
        print("\nNote: Parameters not specified will use default values from SlideMatchingProcessor")
        print("Use --full to evaluate on all 20 lectures")
        return

    # Run grid search
    grid_search = GridSearch(
        data_dir=args.data_dir,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        use_full_lectures=args.full,
        model_size=args.model_size
    )

    results = grid_search.run_grid_search(param_grid)
    grid_search.print_final_report(results)


if __name__ == '__main__':
    main()
