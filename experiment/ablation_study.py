#!/usr/bin/env python3
"""
Ablation Study for Slide Matching Algorithm

Tests the contribution of individual features by systematically disabling them.
Evaluates 4 key features:
  1. exponential_scaling
  2. confidence_boost
  3. context_similarity
  4. sentence_length (disabled when min_sentence_length=0)
"""

import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from match import SlideMatchingProcessor


class AblationStudy:
    """Ablation study for SlideMatchingProcessor features."""

    def __init__(
        self,
        data_dir: str = "./dataset",
        output_dir: str = "./ablation_results",
        device: str = "cuda",
        batch_size: int = 4,
        use_full_dataset: bool = False,
        model_size: str = "3b"
    ):
        """
        Initialize ablation study.

        Args:
            data_dir: Directory containing lecture data
            output_dir: Directory to save results
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
        """Get lecture metadata (same as evaluate.py)."""
        all_lectures = [
            {
                'name': 'cryptocurrency',
                'pdf': 'cryptocurrency.pdf',
                'ground_truth': 'ground_truth_cryptocurrency_MIT.xlsx',
            },
            {
                'name': 'psychology',
                'pdf': 'psychology.pdf',
                'ground_truth': 'ground_truth_psychology.xlsx',
            },
            {
                'name': 'reinforcement_learning',
                'pdf': 'reinforcement_learning.pdf',
                'ground_truth': 'ground_truth_reinforcement_learning.xlsx',
            },
            {
                'name': 'cognitive_robotics_MIT',
                'pdf': 'cognitive_robotics_MIT.pdf',
                'ground_truth': 'ground_truth_cognitive_robotics_MIT.xlsx',
            },
            {
                'name': 'ML_for_health',
                'pdf': 'ML_for_health_care_MIT.pdf',
                'ground_truth': 'ground_truth_ML_for_health_MIT.xlsx',
            },
            {
                'name': 'climate_science_policy',
                'pdf': 'climate_science_policy_MIT.pdf',
                'ground_truth': 'ground_truth_climate_science_policy_MIT2.xlsx',
            },
            {
                'name': 'cities_and_climate',
                'pdf': 'cities&climate.pdf',
                'ground_truth': 'ground_truth_climate_and_cities.xlsx',
            },
            {
                'name': 'cities_and_decarbonization',
                'pdf': 'cities&decarbonization.pdf',
                'ground_truth': 'ground_truth_cities_and_decarbonization.xlsx',
            },
            {
                'name': 'computer_vision_2_2',
                'pdf': 'computer_vision_2_2_geiger.pdf',
                'ground_truth': 'ground_truth_computer_vision_2_2.xlsx',
            },
            {
                'name': 'creating_breakthrough_products',
                'pdf': 'creating_breakthrough_products.pdf',
                'ground_truth': 'ground_truth_creating_breakthrough_products_MIT.xlsx',
            },
            {
                'name': 'deeplearning',
                'pdf': 'deeplearning_goodfellow.pdf',
                'ground_truth': 'ground_truth_deeplearning.xlsx',
            },
            {
                'name': 'phonetics',
                'pdf': 'phonetics.pdf',
                'ground_truth': 'ground_truth_phonetics.xlsx',
            },
            {
                'name': 'numerics',
                'pdf': 'numerics.pdf',
                'ground_truth': 'ground_truth_numerics.xlsx',
            },
            {
                'name': 'image_processing',
                'pdf': 'image_processing.pdf',
                'ground_truth': 'ground_truth_image_processing.xlsx',
            },
            {
                'name': 'physics',
                'pdf': 'physics_intro_02.pdf',
                'ground_truth': 'ground_truth_physics.xlsx',
            },
            {
                'name': 'sensory_systems',
                'pdf': 'sensory_systems.pdf',
                'ground_truth': 'ground_truth_sensory_systems.xlsx',
            },
            {
                'name': 'short_range',
                'pdf': 'short_range_mit.pdf',
                'ground_truth': 'ground_truth_short_range.xlsx',
            },
            {
                'name': 'solar_resource',
                'pdf': 'solar_resource.pdf',
                'ground_truth': 'ground_truth_solar_resource.xlsx',
            },
            {
                'name': 'team_dynamics_game_design',
                'pdf': 'team_dynamics_game_design_mit.pdf',
                'ground_truth': 'ground_truth_team_dynamics_game_design_MIT.xlsx',
            },
            {
                'name': 'theory_of_computation',
                'pdf': 'theory_of_computation.pdf',
                'ground_truth': 'ground_truth_theory_of_computation.xlsx',
            }
        ]

        return all_lectures if self.use_full_dataset else all_lectures[:4]

    def find_lecture_files(self) -> List[Dict[str, str]]:
        """Find and verify lecture files."""
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
                print(f"Warning: Missing files for {lecture['name']}")

        return lectures

    def load_ground_truth_sentences(self, gt_path: str) -> List[Dict[str, Any]]:
        """Load sentences with ground truth labels."""
        df = pd.read_excel(gt_path)
        df_clean = df.rename(columns={'Value': 'text', 'Slidenumber': 'gt_page'})

        df_filtered = df_clean[
            df_clean['text'].notna() &
            (df_clean['text'].astype(str).str.strip() != '')
        ]

        return df_filtered.to_dict(orient='records')

    def compute_metrics(
        self,
        predictions: List[Dict],
        ground_truth_data: List[Dict]
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        correct_count = 0
        evaluated_count = 0
        errors = []

        for pred, gt in zip(predictions, ground_truth_data):
            if gt['gt_page'] != -1:
                evaluated_count += 1
                if pred['matched_page'] == gt['gt_page']:
                    correct_count += 1
                errors.append(pred['matched_page'] - gt['gt_page'])

        accuracy = correct_count / evaluated_count if evaluated_count > 0 else 0.0
        mae = np.mean(np.abs(errors)) if errors else 0.0
        rmse = np.sqrt(np.mean(np.square(errors))) if errors else 0.0

        return {
            'accuracy': accuracy,
            'correct': correct_count,
            'evaluated': evaluated_count,
            'mae': mae,
            'rmse': rmse
        }

    def generate_ablation_configs(
        self,
        baseline_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate ablation configurations.

        Tests each feature individually and in combination:
        - All features ON (baseline)
        - Each feature OFF individually
        - All features OFF

        Args:
            baseline_params: Default parameters with all features enabled

        Returns:
            List of parameter configurations to test
        """
        configs = []

        # Baseline: all features ON
        configs.append({
            'name': 'All Features',
            'params': baseline_params.copy(),
            'features': {
                'exponential_scaling': True,
                'confidence_boost': True,
                'context_similarity': True,
                'sentence_length': True
            }
        })

        # Ablate exponential_scaling
        config = baseline_params.copy()
        config['use_exponential_scaling'] = False
        configs.append({
            'name': 'No Exponential Scaling',
            'params': config,
            'features': {
                'exponential_scaling': False,
                'confidence_boost': True,
                'context_similarity': True,
                'sentence_length': True
            }
        })

        # Ablate confidence_boost
        config = baseline_params.copy()
        config['use_confidence_boost'] = False
        configs.append({
            'name': 'No Confidence Boost',
            'params': config,
            'features': {
                'exponential_scaling': True,
                'confidence_boost': False,
                'context_similarity': True,
                'sentence_length': True
            }
        })

        # Ablate context_similarity
        config = baseline_params.copy()
        config['use_context_similarity'] = False
        configs.append({
            'name': 'No Context Similarity',
            'params': config,
            'features': {
                'exponential_scaling': True,
                'confidence_boost': True,
                'context_similarity': False,
                'sentence_length': True
            }
        })

        # Ablate sentence_length
        config = baseline_params.copy()
        config['min_sentence_length'] = 0
        configs.append({
            'name': 'No Sentence Length Filter',
            'params': config,
            'features': {
                'exponential_scaling': True,
                'confidence_boost': True,
                'context_similarity': True,
                'sentence_length': False
            }
        })

        # All features OFF
        config = baseline_params.copy()
        config['use_exponential_scaling'] = False
        config['use_confidence_boost'] = False
        config['use_context_similarity'] = False
        config['min_sentence_length'] = 0
        configs.append({
            'name': 'No Features',
            'params': config,
            'features': {
                'exponential_scaling': False,
                'confidence_boost': False,
                'context_similarity': False,
                'sentence_length': False
            }
        })

        return configs

    def evaluate_configuration(
        self,
        lecture_info: Dict[str, str],
        config: Dict[str, Any],
        is_first_config: bool = False
    ) -> Dict[str, Any]:
        """Evaluate a single configuration on one lecture."""
        ground_truth_data = self.load_ground_truth_sentences(
            lecture_info['ground_truth_path']
        )
        sentences = [str(item['text']).strip() for item in ground_truth_data]

        start_time = time.time()

        try:
            predictions = self.processor.match_transcript_to_slides(
                transcript="",
                pdf_path=lecture_info['pdf_path'],
                sentences=sentences,
                use_cached_similarity=not is_first_config,
                **config['params']
            )

            metrics = self.compute_metrics(predictions, ground_truth_data)

            result = {
                'lecture_name': lecture_info['name'],
                'config_name': config['name'],
                **config['features'],
                **metrics,
                'time': time.time() - start_time,
                'status': 'success'
            }

        except Exception as e:
            result = {
                'lecture_name': lecture_info['name'],
                'config_name': config['name'],
                **config['features'],
                'accuracy': 0.0,
                'correct': 0,
                'evaluated': 0,
                'mae': 0.0,
                'rmse': 0.0,
                'time': time.time() - start_time,
                'status': 'failed',
                'error': str(e)
            }

        return result

    def run_ablation_study(
        self,
        baseline_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run ablation study.

        Args:
            baseline_params: Baseline parameters (uses defaults if None)

        Returns:
            Dictionary containing all results and summary
        """
        # Default baseline parameters
        default_baseline = {
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
        baseline = {**default_baseline, **(baseline_params or {})}

        print("=" * 80)
        print("Ablation Study for Slide Matching Algorithm")
        print("=" * 80)

        print(f"\nBaseline parameters:")
        for key, value in baseline.items():
            print(f"   {key}: {value}")

        # Generate ablation configurations
        configs = self.generate_ablation_configs(baseline)

        print(f"\nTesting {len(configs)} configurations:")
        for i, config in enumerate(configs, 1):
            features_str = ", ".join([k for k, v in config['features'].items() if v])
            print(f"   {i}. {config['name']:<30} [{features_str}]")

        # Find lectures
        lectures = self.find_lecture_files()

        if not lectures:
            print("Error: No lecture files found!")
            return {'results': [], 'summary': {}}

        print(f"\nEvaluating on {len(lectures)} lectures")

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
        total_evals = len(configs) * len(lectures)

        print(f"\nRunning {total_evals} total evaluations...")
        pbar = tqdm(total=total_evals, desc="Ablation Study")

        # Iterate lecture-first to maximize cache reuse
        for lecture_idx, lecture_info in enumerate(lectures):
            # Clear cache when switching lectures
            if lecture_idx > 0:
                self.processor.clear_cache()

            for config_idx, config in enumerate(configs):
                pbar.set_description(f"{lecture_info['name'][:20]}: {config['name'][:30]}")

                # First config computes similarity matrix; others reuse it
                is_first = (config_idx == 0)

                result = self.evaluate_configuration(
                    lecture_info,
                    config,
                    is_first_config=is_first
                )
                all_results.append(result)
                pbar.update(1)

        pbar.close()

        # Cleanup
        if self.processor is not None:
            self.processor.unload_model()

        # Generate summary
        summary = self.generate_summary(all_results, configs)

        # Save results
        self.save_results(all_results, summary, baseline)

        return {
            'results': all_results,
            'summary': summary,
            'baseline_params': baseline
        }

    def generate_summary(
        self,
        results: List[Dict],
        configs: List[Dict]
    ) -> Dict[str, Any]:
        """Generate summary statistics."""
        successful = [r for r in results if r['status'] == 'success']

        if not successful:
            return {
                'total_evaluations': len(results),
                'successful': 0,
                'config_stats': []
            }

        # Aggregate by configuration
        config_stats = []

        for config in configs:
            config_results = [
                r for r in successful
                if r['config_name'] == config['name']
            ]

            if config_results:
                accuracies = [r['accuracy'] for r in config_results]
                maes = [r['mae'] for r in config_results]
                rmses = [r['rmse'] for r in config_results]

                config_stats.append({
                    'config_name': config['name'],
                    'features': config['features'],
                    'avg_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'min_accuracy': np.min(accuracies),
                    'max_accuracy': np.max(accuracies),
                    'avg_mae': np.mean(maes),
                    'avg_rmse': np.mean(rmses),
                    'num_lectures': len(config_results)
                })

        # Sort by average accuracy
        config_stats.sort(key=lambda x: x['avg_accuracy'], reverse=True)

        return {
            'total_evaluations': len(results),
            'successful': len(successful),
            'failed': len(results) - len(successful),
            'config_stats': config_stats
        }

    def save_results(
        self,
        results: List[Dict],
        summary: Dict,
        baseline_params: Dict
    ):
        """Save ablation study results."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Create run-specific subdirectory
        run_dir = self.output_dir / f"run_{timestamp}"
        run_dir.mkdir(exist_ok=True, parents=True)

        # Save detailed results as CSV
        csv_file = run_dir / "ablation_results.csv"
        df_results = pd.DataFrame(results)
        df_results.to_csv(csv_file, index=False)

        # Save summary as JSON
        summary_file = run_dir / "ablation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'baseline_params': baseline_params,
                'summary': summary
            }, f, indent=2, ensure_ascii=False)

        # Generate visualization report
        self.generate_visualization(results, summary, run_dir)

        print(f"\nResults saved to: {run_dir}")
        print(f"   CSV: {csv_file.name}")
        print(f"   JSON: {summary_file.name}")

    def generate_visualization(
        self,
        results: List[Dict],
        summary: Dict,
        run_dir: Path
    ):
        """Generate text-based visualization report."""
        report_file = run_dir / "ablation_report.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ABLATION STUDY REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Summary statistics
            f.write(f"Total Evaluations: {summary['total_evaluations']}\n")
            f.write(f"Successful: {summary['successful']}\n")
            f.write(f"Failed: {summary['failed']}\n\n")

            # Configuration comparison
            f.write("-" * 80 + "\n")
            f.write("CONFIGURATION COMPARISON (sorted by accuracy)\n")
            f.write("-" * 80 + "\n\n")

            for i, config in enumerate(summary['config_stats'], 1):
                f.write(f"{i}. {config['config_name']}\n")

                # Feature status
                features = config['features']
                f.write(f"   Features: ")
                feature_strs = []
                feature_strs.append(f"EXP={'ON' if features['exponential_scaling'] else 'OFF'}")
                feature_strs.append(f"CONF={'ON' if features['confidence_boost'] else 'OFF'}")
                feature_strs.append(f"CTX={'ON' if features['context_similarity'] else 'OFF'}")
                feature_strs.append(f"LEN={'ON' if features['sentence_length'] else 'OFF'}")
                f.write(" | ".join(feature_strs) + "\n")

                # Metrics
                f.write(f"   Avg Accuracy: {config['avg_accuracy']:.4f} (+/- {config['std_accuracy']:.4f})\n")
                f.write(f"   Range: [{config['min_accuracy']:.4f}, {config['max_accuracy']:.4f}]\n")
                f.write(f"   Avg MAE: {config['avg_mae']:.2f}\n")
                f.write(f"   Avg RMSE: {config['avg_rmse']:.2f}\n")
                f.write(f"   Lectures: {config['num_lectures']}\n\n")

            # Feature impact analysis
            f.write("-" * 80 + "\n")
            f.write("FEATURE IMPACT ANALYSIS\n")
            f.write("-" * 80 + "\n\n")

            baseline_config = next((c for c in summary['config_stats'] if c['config_name'] == 'All Features'), None)

            if baseline_config:
                baseline_acc = baseline_config['avg_accuracy']
                f.write(f"Baseline (All Features): {baseline_acc:.4f}\n\n")

                # Individual feature contributions
                feature_impacts = []

                ablation_map = {
                    'No Exponential Scaling': 'exponential_scaling',
                    'No Confidence Boost': 'confidence_boost',
                    'No Context Similarity': 'context_similarity',
                    'No Sentence Length Filter': 'sentence_length'
                }

                for config_name, feature_name in ablation_map.items():
                    ablated_config = next((c for c in summary['config_stats'] if c['config_name'] == config_name), None)
                    if ablated_config:
                        impact = baseline_acc - ablated_config['avg_accuracy']
                        feature_impacts.append({
                            'feature': feature_name,
                            'impact': impact,
                            'ablated_acc': ablated_config['avg_accuracy']
                        })

                # Sort by impact (descending)
                feature_impacts.sort(key=lambda x: x['impact'], reverse=True)

                for impact_data in feature_impacts:
                    f.write(f"{impact_data['feature']:<25} | "
                           f"Impact: {impact_data['impact']:+.4f} | "
                           f"Accuracy w/o: {impact_data['ablated_acc']:.4f}\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"   Report: {report_file.name}")

    def print_report(self, ablation_data: Dict):
        """Print ablation study report to console."""
        summary = ablation_data['summary']

        print(f"\n{'=' * 80}")
        print("ABLATION STUDY REPORT")
        print(f"{'=' * 80}")

        print(f"\nOverall Statistics:")
        print(f"   Total evaluations: {summary['total_evaluations']}")
        print(f"   Successful: {summary['successful']}")
        print(f"   Failed: {summary['failed']}")

        print(f"\nConfiguration Results (sorted by accuracy):")
        for i, config in enumerate(summary['config_stats'], 1):
            features = config['features']
            feature_str = " ".join([
                f"EXP={'Y' if features['exponential_scaling'] else 'N'}",
                f"CONF={'Y' if features['confidence_boost'] else 'N'}",
                f"CTX={'Y' if features['context_similarity'] else 'N'}",
                f"LEN={'Y' if features['sentence_length'] else 'N'}"
            ])

            print(f"   {i}. {config['config_name']:<30} | "
                  f"[{feature_str}] | "
                  f"Acc: {config['avg_accuracy']:.4f} | "
                  f"MAE: {config['avg_mae']:.2f}")

        # Feature impact
        baseline = next((c for c in summary['config_stats'] if c['config_name'] == 'All Features'), None)
        if baseline:
            print(f"\nFeature Impact (decrease in accuracy when removed):")

            impacts = []
            ablation_map = {
                'No Exponential Scaling': 'Exponential Scaling',
                'No Confidence Boost': 'Confidence Boost',
                'No Context Similarity': 'Context Similarity',
                'No Sentence Length Filter': 'Sentence Length Filter'
            }

            for config_name, feature_name in ablation_map.items():
                ablated = next((c for c in summary['config_stats'] if c['config_name'] == config_name), None)
                if ablated:
                    impact = baseline['avg_accuracy'] - ablated['avg_accuracy']
                    impacts.append((feature_name, impact))

            impacts.sort(key=lambda x: x[1], reverse=True)

            for feature_name, impact in impacts:
                print(f"   {feature_name:<30} {impact:+.4f}")

        print(f"\n{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description='Ablation study for slide matching algorithm features'
    )

    parser.add_argument('--data-dir', default='./dataset',
                       help='Path to dataset directory')
    parser.add_argument('--output-dir', default='./ablation_results',
                       help='Path to output directory')
    parser.add_argument('--device', default='cuda',
                       help='Device to run on (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for embeddings')
    parser.add_argument('--full', action='store_true',
                       help='Use all 20 lectures (default: first 4)')
    parser.add_argument('--model-size', default='3b', choices=['1b', '3b'],
                       help='Model size to use (1b or 3b, default: 3b)')

    # Baseline parameters (optional overrides)
    parser.add_argument('--jump-penalty', type=float, default=1.5)
    parser.add_argument('--backward-weight', type=float, default=1.85)
    parser.add_argument('--exponential-scale', type=float, default=2.785)
    parser.add_argument('--confidence-threshold', type=float, default=0.913)
    parser.add_argument('--confidence-weight', type=float, default=2.18)
    parser.add_argument('--context-weight', type=float, default=0.04)
    parser.add_argument('--context-update-rate', type=float, default=0.24)
    parser.add_argument('--min-sentence-length', type=int, default=2)

    args = parser.parse_args()

    # Prepare baseline parameters
    baseline_params = {
        'jump_penalty': args.jump_penalty,
        'backward_weight': args.backward_weight,
        'use_exponential_scaling': True,
        'exponential_scale': args.exponential_scale,
        'use_confidence_boost': True,
        'confidence_threshold': args.confidence_threshold,
        'confidence_weight': args.confidence_weight,
        'use_context_similarity': True,
        'context_weight': args.context_weight,
        'context_update_rate': args.context_update_rate,
        'min_sentence_length': args.min_sentence_length
    }

    # Create and run ablation study
    ablation = AblationStudy(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        use_full_dataset=args.full,
        model_size=args.model_size
    )

    results = ablation.run_ablation_study(baseline_params)
    ablation.print_report(results)


if __name__ == '__main__':
    main()
